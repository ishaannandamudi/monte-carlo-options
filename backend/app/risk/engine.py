"""Historical, parametric, and Monte Carlo portfolio VaR/ES engines."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from typing import cast

import numpy as np
from numpy.random import SeedSequence
from numpy.typing import NDArray
from scipy.stats import norm

from backend.app.core.exceptions import PortfolioRiskError
from backend.app.models.heston import HestonParameters
from backend.app.models.portfolio import (
    BasketPosition,
    MarketScenario,
    Portfolio,
    PortfolioGreeksResult,
    PortfolioPosition,
    PortfolioRequest,
    PortfolioScenarioRequest,
    PortfolioStressRequest,
    PositionGreeksResult,
    VanillaPosition,
)
from backend.app.models.risk import (
    BasePortfolioRiskRequest,
    CovarianceInput,
    CovarianceSource,
    FactorRiskContribution,
    GroupedRiskContribution,
    HistoricalRiskDataset,
    HistoricalRiskRequest,
    LossDistributionSummary,
    MissingFactorPolicy,
    MonteCarloRiskRequest,
    ParametricRiskRequest,
    PortfolioRiskReport,
    PortfolioRiskReportRequest,
    PortfolioRiskRequest,
    PositionRiskContribution,
    PositionTailContribution,
    RiskConcentration,
    RiskContributionSummary,
    RiskDistribution,
    RiskFactorMapping,
    RiskFactorType,
    RiskMethod,
    RiskValuationMode,
    StressLossPoint,
    WorstRiskScenario,
    model_name_for_position,
)
from backend.app.portfolio.engine import (
    portfolio_greeks,
    run_scenario,
    run_stress_scenarios,
)
from backend.app.risk.statistics import (
    PreparedCovariance,
    distribution_summary,
    empirical_tail,
    estimate_sample_covariance,
    pnl_to_losses,
    validate_covariance,
)
from backend.app.simulation.random import create_rng

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class _ExposureState:
    greeks: PortfolioGreeksResult
    position_exposures: FloatArray
    position_gamma_exposures: FloatArray
    theta_pnl: FloatArray
    factor_ids: tuple[str, ...]

    @property
    def exposures(self) -> FloatArray:
        return np.asarray(np.sum(self.position_exposures, axis=0), dtype=np.float64)

    @property
    def gamma_exposures(self) -> FloatArray:
        return np.asarray(np.sum(self.position_gamma_exposures, axis=0), dtype=np.float64)


def _underlyings(position: PortfolioPosition) -> list[str]:
    if isinstance(position, BasketPosition):
        assert position.market.asset_ids is not None
        return list(position.market.asset_ids)
    return [position.underlying_id]


def _spot_for_underlying(position: PortfolioPosition, underlying_id: str) -> float | None:
    if isinstance(position, BasketPosition):
        assert position.market.asset_ids is not None
        try:
            index = position.market.asset_ids.index(underlying_id)
        except ValueError:
            return None
        return position.market.spots[index]
    return position.instrument.spot if position.underlying_id == underlying_id else None


def _mapping_exposure(
    mapping: RiskFactorMapping,
    position: PortfolioPosition,
    result: PositionGreeksResult,
    *,
    require_available: bool,
) -> tuple[float, float]:
    explicit = mapping.explicit_position_exposures.get(position.position_id)
    explicit_gamma = mapping.explicit_position_gamma_exposures.get(position.position_id)
    if explicit is not None:
        return explicit, explicit_gamma or 0.0
    if mapping.factor_type is RiskFactorType.SPOT_RETURN:
        assert mapping.underlying_id is not None
        spot = _spot_for_underlying(position, mapping.underlying_id)
        if spot is None:
            return 0.0, 0.0
        if isinstance(position, BasketPosition):
            delta = result.per_asset_delta.get(mapping.underlying_id)
            if delta is None:
                raise PortfolioRiskError(
                    f"position {position.position_id} lacks per-asset Delta for {mapping.factor_id}"
                )
            return delta * spot, explicit_gamma or 0.0
        delta = result.position_greeks.delta
        gamma = result.position_greeks.gamma
        if delta is None:
            raise PortfolioRiskError(
                f"position {position.position_id} lacks Delta for {mapping.factor_id}"
            )
        return delta * spot, 0.0 if gamma is None else gamma * spot * spot
    if mapping.factor_type is RiskFactorType.VOLATILITY_CHANGE:
        assert mapping.underlying_id is not None
        if mapping.underlying_id not in _underlyings(position):
            return 0.0, 0.0
        vega = result.position_greeks.vega
        if vega is None:
            raise PortfolioRiskError(
                f"position {position.position_id} lacks Vega for {mapping.factor_id}"
            )
        return vega, explicit_gamma or 0.0
    if mapping.factor_type is RiskFactorType.RATE_CHANGE:
        rho = result.position_greeks.rho
        if rho is None:
            raise PortfolioRiskError(
                f"position {position.position_id} lacks Rho for {mapping.factor_id}"
            )
        return rho, explicit_gamma or 0.0
    if mapping.factor_type is RiskFactorType.DIVIDEND_YIELD_CHANGE:
        assert mapping.underlying_id is not None
        if mapping.underlying_id in _underlyings(position) and require_available:
            raise PortfolioRiskError(
                f"position {position.position_id} requires an explicit dividend-yield exposure"
            )
        return 0.0, explicit_gamma or 0.0
    if mapping.factor_type in {
        RiskFactorType.HESTON_V0_CHANGE,
        RiskFactorType.HESTON_THETA_CHANGE,
        RiskFactorType.HESTON_KAPPA_CHANGE,
        RiskFactorType.HESTON_SIGMA_V_CHANGE,
        RiskFactorType.HESTON_RHO_CHANGE,
        RiskFactorType.CORRELATION_CHANGE,
    }:
        if mapping.position_id == position.position_id and require_available:
            raise PortfolioRiskError(
                f"position {position.position_id} requires an explicit exposure for "
                f"{mapping.factor_type.value}"
            )
        return 0.0, explicit_gamma or 0.0
    if mapping.factor_type is RiskFactorType.CUSTOM_GAUSSIAN:
        return 0.0, explicit_gamma or 0.0
    raise PortfolioRiskError(f"unsupported risk factor type: {mapping.factor_type}")


def _build_exposures(request: BasePortfolioRiskRequest) -> _ExposureState:
    portfolio_request = PortfolioRequest(
        portfolio=request.portfolio, config=request.portfolio_config
    )
    greeks = portfolio_greeks(portfolio_request)
    by_id = {item.position_id: item for item in greeks.positions}
    rows: list[list[float]] = []
    gamma_rows: list[list[float]] = []
    theta: list[float] = []
    for position in request.portfolio.positions:
        result = by_id[position.position_id]
        if result.status.value != "SUCCESS":
            raise PortfolioRiskError(f"Greek evaluation failed for {position.position_id}")
        position_row: list[float] = []
        gamma_row: list[float] = []
        for mapping in request.factor_mappings:
            exposure, gamma_exposure = _mapping_exposure(
                mapping,
                position,
                result,
                require_available=(
                    request.config.valuation_mode is not RiskValuationMode.FULL_REVALUATION
                ),
            )
            position_row.append(exposure)
            gamma_row.append(gamma_exposure)
        rows.append(position_row)
        gamma_rows.append(gamma_row)
        theta.append(
            (result.position_greeks.theta or 0.0) * request.config.horizon.year_fraction
            if request.config.include_time_passage
            else 0.0
        )
    return _ExposureState(
        greeks=greeks,
        position_exposures=np.asarray(rows, dtype=np.float64),
        position_gamma_exposures=np.asarray(gamma_rows, dtype=np.float64),
        theta_pnl=np.asarray(theta, dtype=np.float64),
        factor_ids=tuple(mapping.factor_id for mapping in request.factor_mappings),
    )


def _historical_matrix(
    dataset: HistoricalRiskDataset,
    mappings: list[RiskFactorMapping],
    policy: MissingFactorPolicy,
) -> tuple[FloatArray, list[int]]:
    expected = [mapping.factor_id for mapping in mappings]
    expected_set = set(expected)
    rows: list[list[float]] = []
    kept: list[int] = []
    for index, scenario in enumerate(dataset.scenarios):
        values = {item.factor_id: item.value for item in scenario.observations}
        unknown = set(values) - expected_set
        if unknown:
            raise PortfolioRiskError(
                f"historical scenario contains unmapped factors: {sorted(unknown)}"
            )
        missing = expected_set - set(values)
        if missing:
            if policy is MissingFactorPolicy.ERROR:
                raise PortfolioRiskError(
                    f"historical scenario {scenario.timestamp.isoformat()} is missing factors: "
                    f"{sorted(missing)}"
                )
            if policy is MissingFactorPolicy.DROP_SCENARIO:
                continue
        rows.append([values.get(identifier, 0.0) for identifier in expected])
        kept.append(index)
    if len(rows) < 2:
        raise PortfolioRiskError(
            "at least two historical scenarios must remain after missing-factor handling"
        )
    return np.asarray(rows, dtype=np.float64), kept


def _base_heston_parameters(position: VanillaPosition) -> HestonParameters:
    if position.heston_parameters is not None:
        return position.heston_parameters
    if position.heston_calibration_result is not None:
        return position.heston_calibration_result.calibrated_parameters
    raise PortfolioRiskError(f"position {position.position_id} has no Heston parameters")


def _scenario_from_shocks(
    portfolio: Portfolio,
    mappings: list[RiskFactorMapping],
    shocks: FloatArray,
    *,
    time_shift_years: float,
) -> MarketScenario:
    spots: dict[str, float] = {}
    vols: dict[str, float] = {}
    dividends: dict[str, float] = {}
    correlation: dict[str, float] = {}
    heston_changes: dict[str, dict[str, float]] = {}
    rate = 0.0
    positions = {item.position_id: item for item in portfolio.positions}
    heston_field = {
        RiskFactorType.HESTON_V0_CHANGE: "v0",
        RiskFactorType.HESTON_THETA_CHANGE: "theta",
        RiskFactorType.HESTON_KAPPA_CHANGE: "kappa",
        RiskFactorType.HESTON_SIGMA_V_CHANGE: "sigma_v",
        RiskFactorType.HESTON_RHO_CHANGE: "rho",
    }
    for mapping, value in zip(mappings, shocks, strict=True):
        change = float(value)
        if mapping.factor_type is RiskFactorType.SPOT_RETURN:
            assert mapping.underlying_id is not None
            spots[mapping.underlying_id] = spots.get(mapping.underlying_id, 0.0) + change
        elif mapping.factor_type is RiskFactorType.VOLATILITY_CHANGE:
            assert mapping.underlying_id is not None
            vols[mapping.underlying_id] = vols.get(mapping.underlying_id, 0.0) + change
        elif mapping.factor_type is RiskFactorType.RATE_CHANGE:
            rate += change
        elif mapping.factor_type is RiskFactorType.DIVIDEND_YIELD_CHANGE:
            assert mapping.underlying_id is not None
            dividends[mapping.underlying_id] = dividends.get(mapping.underlying_id, 0.0) + change
        elif mapping.factor_type in heston_field:
            assert mapping.position_id is not None
            heston_changes.setdefault(mapping.position_id, {})[
                heston_field[mapping.factor_type]
            ] = (
                heston_changes.setdefault(mapping.position_id, {}).get(
                    heston_field[mapping.factor_type], 0.0
                )
                + change
            )
        elif mapping.factor_type is RiskFactorType.CORRELATION_CHANGE:
            assert mapping.position_id is not None
            correlation[mapping.position_id] = correlation.get(mapping.position_id, 0.0) + change
        elif mapping.factor_type is RiskFactorType.CUSTOM_GAUSSIAN:
            raise PortfolioRiskError("CUSTOM_GAUSSIAN factors cannot be used for full revaluation")
    heston_parameters: dict[str, HestonParameters] = {}
    for position_id, updates in heston_changes.items():
        position = positions[position_id]
        if not isinstance(position, VanillaPosition):
            raise PortfolioRiskError("Heston factor mapping requires a vanilla position")
        parameters = _base_heston_parameters(position)
        values = parameters.model_dump(exclude={"feller_satisfied", "feller_ratio"})
        for field, change in updates.items():
            values[field] = cast(float, values[field]) + change
        heston_parameters[position_id] = HestonParameters.model_validate(values)
    return MarketScenario(
        per_underlying_spot_relative_shifts=spots,
        per_underlying_volatility_absolute_shifts=vols,
        rate_absolute_shift=rate,
        per_underlying_dividend_yield_shifts=dividends,
        time_shift_years=time_shift_years,
        heston_parameters_by_position=heston_parameters,
        equicorrelation_shift_by_position=correlation,
    )


def _approximate_position_pnl(
    exposures: _ExposureState, shocks: FloatArray, mode: RiskValuationMode
) -> FloatArray:
    result = shocks @ exposures.position_exposures.T + exposures.theta_pnl
    if mode is RiskValuationMode.DELTA_GAMMA_DIAGONAL:
        result = result + 0.5 * (shocks * shocks) @ exposures.position_gamma_exposures.T
    return np.asarray(result, dtype=np.float64)


def _prepare_covariance(
    covariance: CovarianceInput | None,
    dataset: HistoricalRiskDataset | None,
    mappings: list[RiskFactorMapping],
    policy: MissingFactorPolicy,
) -> PreparedCovariance:
    expected = [item.factor_id for item in mappings]
    if covariance is not None:
        if set(covariance.factor_ids) != set(expected):
            raise PortfolioRiskError("covariance factor_ids must exactly match factor mappings")
        index = [covariance.factor_ids.index(identifier) for identifier in expected]
        matrix = np.asarray(covariance.covariance_matrix, dtype=np.float64)[np.ix_(index, index)]
        means = (
            np.zeros(len(expected), dtype=np.float64)
            if covariance.mean_changes is None
            else np.asarray(covariance.mean_changes, dtype=np.float64)[index]
        )
        return validate_covariance(expected, matrix, means, source=CovarianceSource.USER_SUPPLIED)
    assert dataset is not None
    values, _ = _historical_matrix(dataset, mappings, policy)
    return estimate_sample_covariance(expected, values)


def _horizon_covariance(
    prepared: PreparedCovariance, trading_days: int, scale: bool
) -> tuple[FloatArray, FloatArray]:
    if not scale or trading_days == 1:
        return prepared.means, prepared.covariance
    return prepared.means * trading_days, prepared.covariance * trading_days


def _factorization(covariance: FloatArray) -> FloatArray:
    values, vectors = np.linalg.eigh(covariance)
    if float(values[0]) < -1e-10 * max(1.0, float(values[-1])):
        raise PortfolioRiskError("cannot simulate a materially indefinite covariance")
    return np.asarray(vectors @ np.diag(np.sqrt(np.maximum(values, 0.0))), dtype=np.float64)


def _standard_draws(
    seed: int,
    count: int,
    dimension: int,
    batch_size: int,
    distribution: RiskDistribution,
    degrees_of_freedom: float,
    antithetic: bool,
) -> Iterator[tuple[int, FloatArray]]:
    rng = create_rng(SeedSequence(seed))
    offset = 0
    while offset < count:
        size = min(batch_size, count - offset)
        rows: FloatArray
        if antithetic:
            pairs = size // 2
            positive = rng.standard_normal((pairs, dimension))
            if distribution is RiskDistribution.MULTIVARIATE_STUDENT_T:
                scales = np.sqrt(
                    (degrees_of_freedom - 2.0) / rng.chisquare(degrees_of_freedom, size=(pairs, 1))
                )
                positive = positive * scales
            rows = np.empty((size, dimension), dtype=np.float64)
            rows[0 : 2 * pairs : 2] = positive
            rows[1 : 2 * pairs : 2] = -positive
            if size % 2:
                single = rng.standard_normal((1, dimension))
                if distribution is RiskDistribution.MULTIVARIATE_STUDENT_T:
                    single *= math.sqrt(
                        (degrees_of_freedom - 2.0) / float(rng.chisquare(degrees_of_freedom))
                    )
                rows[-1] = single[0]
        else:
            rows = rng.standard_normal((size, dimension))
            if distribution is RiskDistribution.MULTIVARIATE_STUDENT_T:
                rows *= np.sqrt(
                    (degrees_of_freedom - 2.0) / rng.chisquare(degrees_of_freedom, size=(size, 1))
                )
        yield offset, rows
        offset += size


def _position_tail_losses(
    position_pnl: FloatArray, tail_indices: NDArray[np.int64], weights: FloatArray
) -> FloatArray:
    selected = -position_pnl[tail_indices]
    return np.asarray(np.average(selected, axis=0, weights=weights), dtype=np.float64)


def _groups(
    request: BasePortfolioRiskRequest,
    positions: list[PositionRiskContribution],
    dimension: str,
) -> list[GroupedRiskContribution]:
    by_id = {item.position_id: item for item in request.portfolio.positions}
    totals: dict[str, list[float]] = {}
    es_totals: dict[str, list[float]] = {}
    ids: dict[str, set[str]] = {}
    for contribution in positions:
        position = by_id[contribution.position_id]
        if dimension == "underlying":
            keys = _underlyings(position)
            divisor = len(keys)
        elif dimension == "instrument":
            keys = [position.instrument_class.value]
            divisor = 1
        else:
            keys = [model_name_for_position(position).value]
            divisor = 1
        for key in keys:
            if contribution.component_var is not None:
                totals.setdefault(key, []).append(contribution.component_var / divisor)
            if contribution.expected_shortfall_contribution is not None:
                es_totals.setdefault(key, []).append(
                    contribution.expected_shortfall_contribution / divisor
                )
            ids.setdefault(key, set()).add(position.position_id)
    return [
        GroupedRiskContribution(
            key=key,
            component_var=math.fsum(totals.get(key, [])) if key in totals else None,
            expected_shortfall_contribution=(
                math.fsum(es_totals.get(key, [])) if key in es_totals else None
            ),
            position_count=len(ids[key]),
        )
        for key in sorted(ids)
    ]


def _hhi(values: list[float | None]) -> float | None:
    absolute = [abs(value) for value in values if value is not None]
    total = math.fsum(absolute)
    return math.fsum((value / total) ** 2 for value in absolute) if total > 0.0 else None


def _contribution_summary(
    request: BasePortfolioRiskRequest,
    exposures: _ExposureState,
    *,
    component_var: FloatArray | None = None,
    incremental_var: FloatArray | None = None,
    es_contributions: FloatArray | None = None,
    factor_marginal: FloatArray | None = None,
    factor_component: FloatArray | None = None,
    factor_es: FloatArray | None = None,
) -> RiskContributionSummary:
    valuations = {item.position_id: item for item in exposures.greeks.valuation.positions}
    positions: list[PositionRiskContribution] = []
    for index, position in enumerate(request.portfolio.positions):
        positions.append(
            PositionRiskContribution(
                position_id=position.position_id,
                underlying_keys=_underlyings(position),
                instrument_class=position.instrument_class,
                model=model_name_for_position(position),
                market_value=valuations[position.position_id].position_value,
                market_value_fraction=(
                    None
                    if exposures.greeks.valuation.total_value in {None, 0.0}
                    or valuations[position.position_id].position_value is None
                    else cast(float, valuations[position.position_id].position_value)
                    / cast(float, exposures.greeks.valuation.total_value)
                ),
                selected_greek_exposure=float(np.sum(exposures.position_exposures[index])),
                component_var=None if component_var is None else float(component_var[index]),
                incremental_var=(
                    None if incremental_var is None else float(incremental_var[index])
                ),
                expected_shortfall_contribution=(
                    None if es_contributions is None else float(es_contributions[index])
                ),
            )
        )
    factors = [
        FactorRiskContribution(
            factor_id=identifier,
            exposure=float(exposures.exposures[index]),
            gamma_exposure=float(exposures.gamma_exposures[index]),
            marginal_var=None if factor_marginal is None else float(factor_marginal[index]),
            component_var=None if factor_component is None else float(factor_component[index]),
            component_es=None if factor_es is None else float(factor_es[index]),
        )
        for index, identifier in enumerate(exposures.factor_ids)
    ]
    component_values = [item.component_var for item in positions]
    es_values = [item.expected_shortfall_contribution for item in positions]
    sorted_component = sorted(
        (item for item in positions if item.component_var is not None),
        key=lambda item: abs(cast(float, item.component_var)),
        reverse=True,
    )
    sorted_es = sorted(
        (item for item in positions if item.expected_shortfall_contribution is not None),
        key=lambda item: abs(cast(float, item.expected_shortfall_contribution)),
        reverse=True,
    )
    return RiskContributionSummary(
        factors=factors,
        positions=positions,
        by_underlying=_groups(request, positions, "underlying"),
        by_instrument_class=_groups(request, positions, "instrument"),
        by_model=_groups(request, positions, "model"),
        component_var_sum=(
            math.fsum(cast(float, value) for value in component_values)
            if all(value is not None for value in component_values)
            else None
        ),
        expected_shortfall_contribution_sum=(
            math.fsum(cast(float, value) for value in es_values)
            if all(value is not None for value in es_values)
            else None
        ),
        concentration=RiskConcentration(
            largest_component_var_position=(
                sorted_component[0].position_id if sorted_component else None
            ),
            largest_expected_shortfall_contributor=(
                sorted_es[0].position_id if sorted_es else None
            ),
            component_var_absolute_hhi=_hhi(component_values),
            expected_shortfall_absolute_hhi=_hhi(es_values),
            top_component_var_positions=[item.position_id for item in sorted_component[:5]],
            top_expected_shortfall_positions=[item.position_id for item in sorted_es[:5]],
        ),
    )


def _worst_scenarios(
    losses: FloatArray,
    pnl: FloatArray,
    shocks: FloatArray,
    factor_ids: tuple[str, ...],
    maximum: int,
    *,
    timestamps: list[datetime] | None = None,
    position_pnl: FloatArray | None = None,
    position_ids: list[str] | None = None,
) -> list[WorstRiskScenario]:
    count = min(maximum, losses.size)
    indices = np.argsort(-losses, kind="stable")[:count]
    results: list[WorstRiskScenario] = []
    for index_value in indices:
        index = int(index_value)
        contributions = (
            []
            if position_pnl is None
            else [
                PositionTailContribution(
                    position_id=(str(column) if position_ids is None else position_ids[column]),
                    loss=float(-value),
                )
                for column, value in enumerate(position_pnl[index])
            ]
        )
        timestamp = None if timestamps is None else timestamps[index]
        results.append(
            WorstRiskScenario(
                scenario_index=index,
                timestamp=timestamp,
                pnl=float(pnl[index]),
                loss=float(losses[index]),
                factor_shocks={
                    identifier: float(value)
                    for identifier, value in zip(factor_ids, shocks[index], strict=True)
                },
                position_contributions=contributions,
            )
        )
    return results


def _empirical_report(
    request: BasePortfolioRiskRequest,
    exposures: _ExposureState,
    pnl: FloatArray,
    shocks: FloatArray,
    *,
    started: float,
    position_pnl: FloatArray | None,
    timestamps: list[datetime] | None,
    covariance: PreparedCovariance | None,
    warnings: list[str],
) -> PortfolioRiskReport:
    losses = pnl_to_losses(pnl)
    tail = empirical_tail(
        losses,
        request.config.confidence_level,
        quantile_method=request.config.quantile_method,
    )
    contributions = None
    if request.config.include_contributions and position_pnl is not None:
        es_values = _position_tail_losses(position_pnl, tail.ordered_indices, tail.weights)
        incremental = None
        if request.config.include_incremental_var:
            incremental_values = []
            for column in range(position_pnl.shape[1]):
                reduced_losses = losses + position_pnl[:, column]
                reduced = empirical_tail(
                    reduced_losses,
                    request.config.confidence_level,
                    quantile_method=request.config.quantile_method,
                )
                incremental_values.append(tail.value_at_risk - reduced.value_at_risk)
            incremental = np.asarray(incremental_values, dtype=np.float64)
        contributions = _contribution_summary(
            request,
            exposures,
            incremental_var=incremental,
            es_contributions=es_values,
        )
    base_value = exposures.greeks.valuation.total_value
    worst = (
        _worst_scenarios(
            losses,
            pnl,
            shocks,
            exposures.factor_ids,
            request.config.max_tail_scenarios,
            timestamps=timestamps,
            position_pnl=position_pnl,
            position_ids=[item.position_id for item in request.portfolio.positions],
        )
        if request.config.include_tail_scenarios
        else []
    )
    dataset = getattr(request, "dataset", None) or getattr(request, "historical_dataset", None)
    return PortfolioRiskReport(
        portfolio_id=request.portfolio.portfolio_id,
        base_value=base_value,
        value_at_risk=tail.value_at_risk,
        expected_shortfall=tail.expected_shortfall,
        confidence_level=request.config.confidence_level,
        horizon=request.config.horizon,
        method=request.config.method,
        valuation_mode=request.config.valuation_mode,
        scenario_count=pnl.size,
        simulations=(pnl.size if request.config.method is RiskMethod.MONTE_CARLO else None),
        distribution=(
            request.config.distribution if request.config.method is RiskMethod.MONTE_CARLO else None
        ),
        distribution_summary=distribution_summary(
            pnl,
            confidence_level=request.config.confidence_level,
            tail_count=len(tail.ordered_indices),
            quantile_method=request.config.quantile_method,
        ),
        worst_scenarios=worst,
        contributions=contributions,
        covariance_diagnostics=None if covariance is None else covariance.diagnostics,
        historical_start=None if dataset is None else dataset.start_timestamp,
        historical_end=None if dataset is None else dataset.end_timestamp,
        sampling_frequency=None if dataset is None else dataset.sampling_frequency,
        warnings=warnings,
        runtime_ms=(perf_counter() - started) * 1_000.0,
    )


def calculate_historical_risk(request: HistoricalRiskRequest) -> PortfolioRiskReport:
    """Replay same-date historical shocks with full repricing or local Greeks."""

    started = perf_counter()
    shocks, kept = _historical_matrix(
        request.dataset, request.factor_mappings, request.config.missing_factor_policy
    )
    exposures = _build_exposures(request)
    warnings: list[str] = []
    if request.config.horizon.trading_days != 1:
        warnings.append(
            "historical factor shocks are replayed without square-root scaling; horizon affects "
            "only time passage"
        )
    if request.config.valuation_mode is RiskValuationMode.FULL_REVALUATION:
        base = exposures.greeks.valuation
        position_rows: list[list[float]] = []
        totals: list[float] = []
        relative_errors: list[float] = []
        for row in shocks:
            scenario = _scenario_from_shocks(
                request.portfolio,
                request.factor_mappings,
                row,
                time_shift_years=(
                    request.config.horizon.year_fraction
                    if request.config.include_time_passage
                    else 0.0
                ),
            )
            result = run_scenario(
                PortfolioScenarioRequest(
                    portfolio=request.portfolio,
                    config=request.portfolio_config,
                    scenario=scenario,
                ),
                _base=base,
                _greeks=exposures.greeks,
            )
            if result.full_repricing_pnl is None:
                raise PortfolioRiskError("historical full revaluation produced no portfolio P&L")
            row_values = [contribution.full_repricing_pnl for contribution in result.contributions]
            if any(value is None for value in row_values):
                raise PortfolioRiskError("historical full revaluation contains failed positions")
            totals.append(result.full_repricing_pnl)
            if result.relative_approximation_error is not None:
                relative_errors.append(result.relative_approximation_error)
            if request.config.include_contributions:
                position_rows.append([cast(float, value) for value in row_values])
        pnl = np.asarray(totals, dtype=np.float64)
        position_pnl = (
            np.asarray(position_rows, dtype=np.float64)
            if request.config.include_contributions
            else None
        )
        if relative_errors:
            warnings.append(
                "full repricing is authoritative; maximum local Greek approximation relative "
                f"error was {max(relative_errors):.6g}"
            )
    else:
        if request.config.include_contributions:
            position_pnl = _approximate_position_pnl(
                exposures, shocks, request.config.valuation_mode
            )
            pnl = np.sum(position_pnl, axis=1)
        else:
            pnl = shocks @ exposures.exposures + float(np.sum(exposures.theta_pnl))
            if request.config.valuation_mode is RiskValuationMode.DELTA_GAMMA_DIAGONAL:
                pnl = pnl + 0.5 * (shocks * shocks) @ exposures.gamma_exposures
            position_pnl = None
        warnings.append("Greek approximation omits cross-Gamma and all unmapped cross terms")
    timestamps = [request.dataset.scenarios[index].timestamp for index in kept]
    return _empirical_report(
        request,
        exposures,
        pnl,
        shocks,
        started=started,
        position_pnl=position_pnl,
        timestamps=timestamps,
        covariance=None,
        warnings=warnings,
    )


def _normal_summary(
    mean_pnl: float, standard_deviation: float, confidence: float
) -> LossDistributionSummary:
    levels = sorted({0.01, 0.05, 0.5, 0.95, 0.975, 0.99, 0.995, confidence})
    mean_loss = -mean_pnl
    return LossDistributionSummary(
        pnl_mean=mean_pnl,
        pnl_standard_deviation=standard_deviation,
        minimum_pnl=None,
        maximum_pnl=None,
        loss_mean=mean_loss,
        loss_standard_deviation=standard_deviation,
        median_loss=mean_loss,
        worst_loss=None,
        best_pnl=None,
        skewness=0.0,
        excess_kurtosis=0.0,
        loss_quantiles={
            f"{level:.6g}": mean_loss + standard_deviation * float(norm.ppf(level))
            for level in levels
        },
        tail_count=0,
    )


def calculate_parametric_risk(request: ParametricRiskRequest) -> PortfolioRiskReport:
    """Calculate analytical delta-normal VaR, ES, and Euler contributions."""

    started = perf_counter()
    exposures = _build_exposures(request)
    prepared = _prepare_covariance(
        request.covariance,
        request.historical_dataset,
        request.factor_mappings,
        request.config.missing_factor_policy,
    )
    means, covariance = _horizon_covariance(
        prepared, request.config.horizon.trading_days, request.config.iid_horizon_scaling
    )
    weights = exposures.exposures
    theta_total = float(np.sum(exposures.theta_pnl))
    mean_pnl = float(weights @ means) + theta_total
    with np.errstate(all="ignore"):
        variance_value = float(weights @ covariance @ weights)
    if not math.isfinite(variance_value):
        raise PortfolioRiskError("parametric portfolio variance is non-finite")
    variance = max(0.0, variance_value)
    sigma = math.sqrt(variance)
    z = float(norm.ppf(request.config.confidence_level))
    density = float(norm.pdf(z))
    mean_loss = -mean_pnl
    value_at_risk = mean_loss + z * sigma
    expected_shortfall = mean_loss + sigma * density / (1.0 - request.config.confidence_level)
    marginal = -means.copy()
    es_marginal = -means.copy()
    if sigma > 0.0:
        with np.errstate(all="ignore"):
            covariance_gradient = covariance @ weights / sigma
        if not np.all(np.isfinite(covariance_gradient)):
            raise PortfolioRiskError("parametric marginal VaR is non-finite")
        marginal += z * covariance_gradient
        es_marginal += density / (1.0 - request.config.confidence_level) * covariance_gradient
    factor_component = weights * marginal
    factor_es = weights * es_marginal
    position_mean = exposures.position_exposures @ means + exposures.theta_pnl
    with np.errstate(all="ignore"):
        covariance_projection = exposures.position_exposures @ covariance @ weights
    if not np.all(np.isfinite(covariance_projection)):
        raise PortfolioRiskError("parametric position contribution is non-finite")
    component = -position_mean
    position_es = -position_mean
    if sigma > 0.0:
        component += z * covariance_projection / sigma
        position_es += (
            density / (1.0 - request.config.confidence_level) * (covariance_projection / sigma)
        )
    incremental = None
    if request.config.include_incremental_var:
        values: list[float] = []
        for index in range(len(request.portfolio.positions)):
            reduced = weights - exposures.position_exposures[index]
            reduced_mean_pnl = mean_pnl - position_mean[index]
            with np.errstate(all="ignore"):
                reduced_variance = float(reduced @ covariance @ reduced)
            if not math.isfinite(reduced_variance):
                raise PortfolioRiskError("incremental VaR variance is non-finite")
            reduced_sigma = math.sqrt(max(0.0, reduced_variance))
            reduced_var = -reduced_mean_pnl + z * reduced_sigma
            values.append(value_at_risk - reduced_var)
        incremental = np.asarray(values, dtype=np.float64)
    contributions = (
        _contribution_summary(
            request,
            exposures,
            component_var=component,
            incremental_var=incremental,
            es_contributions=position_es,
            factor_marginal=marginal,
            factor_component=factor_component,
            factor_es=factor_es,
        )
        if request.config.include_contributions
        else None
    )
    warnings = [
        "normal VaR/ES use a linear Delta exposure model",
        "iid horizon scaling multiplies daily mean by h and covariance by h",
    ]
    if not request.config.iid_horizon_scaling:
        warnings[1] = "daily covariance and means were used without horizon scaling"
    return PortfolioRiskReport(
        portfolio_id=request.portfolio.portfolio_id,
        base_value=exposures.greeks.valuation.total_value,
        value_at_risk=value_at_risk,
        expected_shortfall=expected_shortfall,
        confidence_level=request.config.confidence_level,
        horizon=request.config.horizon,
        method=RiskMethod.PARAMETRIC_NORMAL,
        valuation_mode=RiskValuationMode.DELTA,
        scenario_count=0,
        distribution=RiskDistribution.MULTIVARIATE_NORMAL,
        distribution_summary=_normal_summary(mean_pnl, sigma, request.config.confidence_level),
        contributions=contributions,
        covariance_diagnostics=prepared.diagnostics,
        historical_start=(
            None
            if request.historical_dataset is None
            else request.historical_dataset.start_timestamp
        ),
        historical_end=(
            None if request.historical_dataset is None else request.historical_dataset.end_timestamp
        ),
        sampling_frequency=(
            None
            if request.historical_dataset is None
            else request.historical_dataset.sampling_frequency
        ),
        warnings=warnings,
        runtime_ms=(perf_counter() - started) * 1_000.0,
    )


def _mc_shock_batches(
    request: MonteCarloRiskRequest, mean: FloatArray, covariance: FloatArray
) -> Iterator[tuple[int, FloatArray]]:
    factor = _factorization(covariance)
    for offset, standard in _standard_draws(
        request.config.root_seed,
        request.config.simulations,
        len(request.factor_mappings),
        request.config.batch_size,
        request.config.distribution,
        request.config.student_t_degrees_of_freedom,
        request.config.antithetic,
    ):
        with np.errstate(all="ignore"):
            shocks = np.asarray(mean + standard @ factor.T, dtype=np.float64)
        if not np.all(np.isfinite(shocks)):
            raise PortfolioRiskError("factor simulation produced non-finite shocks")
        yield offset, shocks


def calculate_monte_carlo_risk(request: MonteCarloRiskRequest) -> PortfolioRiskReport:
    """Simulate correlated factors in batches and evaluate empirical VaR and ES."""

    started = perf_counter()
    exposures = _build_exposures(request)
    prepared = _prepare_covariance(
        request.covariance,
        request.historical_dataset,
        request.factor_mappings,
        request.config.missing_factor_policy,
    )
    means, covariance = _horizon_covariance(
        prepared, request.config.horizon.trading_days, request.config.iid_horizon_scaling
    )
    count = request.config.simulations
    pnl = np.empty(count, dtype=np.float64)
    position_pnl_full = (
        np.empty((count, len(request.portfolio.positions)), dtype=np.float64)
        if request.config.include_contributions
        and request.config.valuation_mode is RiskValuationMode.FULL_REVALUATION
        else None
    )
    base = exposures.greeks.valuation
    relative_errors: list[float] = []
    for offset, shocks in _mc_shock_batches(request, means, covariance):
        end = offset + len(shocks)
        if request.config.valuation_mode is RiskValuationMode.FULL_REVALUATION:
            for local, shock in enumerate(shocks):
                scenario = _scenario_from_shocks(
                    request.portfolio,
                    request.factor_mappings,
                    shock,
                    time_shift_years=(
                        request.config.horizon.year_fraction
                        if request.config.include_time_passage
                        else 0.0
                    ),
                )
                result = run_scenario(
                    PortfolioScenarioRequest(
                        portfolio=request.portfolio,
                        config=request.portfolio_config,
                        scenario=scenario,
                    ),
                    _base=base,
                    _greeks=exposures.greeks,
                )
                if result.full_repricing_pnl is None:
                    raise PortfolioRiskError("Monte Carlo full revaluation produced no P&L")
                if result.relative_approximation_error is not None:
                    relative_errors.append(result.relative_approximation_error)
                pnl[offset + local] = result.full_repricing_pnl
                if position_pnl_full is not None:
                    contribution_values = [item.full_repricing_pnl for item in result.contributions]
                    if any(value is None for value in contribution_values):
                        raise PortfolioRiskError("Monte Carlo scenario contains failed positions")
                    position_pnl_full[offset + local] = [
                        cast(float, value) for value in contribution_values
                    ]
        else:
            position_values = _approximate_position_pnl(
                exposures, shocks, request.config.valuation_mode
            )
            pnl[offset:end] = np.sum(position_values, axis=1)
    losses = pnl_to_losses(pnl)
    tail = empirical_tail(
        losses,
        request.config.confidence_level,
        quantile_method=request.config.quantile_method,
    )
    worst_indices = (
        np.argsort(-losses, kind="stable")[: request.config.max_tail_scenarios]
        if request.config.include_tail_scenarios
        else np.asarray([], dtype=np.int64)
    )
    selected = {int(value) for value in worst_indices}
    if (
        request.config.include_contributions
        and request.config.valuation_mode is not RiskValuationMode.FULL_REVALUATION
    ):
        selected.update(int(value) for value in tail.ordered_indices)
    selected_rows: dict[int, FloatArray] = {}
    selected_position_rows: dict[int, FloatArray] = {}
    for offset, shocks in _mc_shock_batches(request, means, covariance) if selected else []:
        relevant = [index for index in range(len(shocks)) if offset + index in selected]
        if not relevant:
            continue
        for local in relevant:
            selected_rows[offset + local] = shocks[local]
        if request.config.valuation_mode is not RiskValuationMode.FULL_REVALUATION:
            selected_values = _approximate_position_pnl(
                exposures, shocks[relevant], request.config.valuation_mode
            )
            for row_index, local in enumerate(relevant):
                selected_position_rows[offset + local] = selected_values[row_index]
    # A bounded reconstruction supplies exact empirical ES contributions without retaining N x P.
    position_pnl = position_pnl_full
    if request.config.include_contributions and position_pnl is None and selected_position_rows:
        es_values = np.average(
            np.asarray(
                [-selected_position_rows[int(index)] for index in tail.ordered_indices],
                dtype=np.float64,
            ),
            axis=0,
            weights=tail.weights,
        )
        incremental = None
        if request.config.include_incremental_var:
            incremental_values: list[float] = []
            for column in range(len(request.portfolio.positions)):
                reduced_pnl = pnl.copy()
                for offset, shocks in _mc_shock_batches(request, means, covariance):
                    position_values = _approximate_position_pnl(
                        exposures, shocks, request.config.valuation_mode
                    )
                    reduced_pnl[offset : offset + len(shocks)] -= position_values[:, column]
                reduced_tail = empirical_tail(
                    pnl_to_losses(reduced_pnl),
                    request.config.confidence_level,
                    quantile_method=request.config.quantile_method,
                )
                incremental_values.append(tail.value_at_risk - reduced_tail.value_at_risk)
            incremental = np.asarray(incremental_values, dtype=np.float64)
        contributions = _contribution_summary(
            request,
            exposures,
            incremental_var=incremental,
            es_contributions=np.asarray(es_values, dtype=np.float64),
        )
    else:
        contributions = None
    if position_pnl_full is not None:
        es_values = _position_tail_losses(position_pnl_full, tail.ordered_indices, tail.weights)
        incremental = None
        if request.config.include_incremental_var:
            incremental = np.asarray(
                [
                    tail.value_at_risk
                    - empirical_tail(
                        losses + position_pnl_full[:, column],
                        request.config.confidence_level,
                        quantile_method=request.config.quantile_method,
                    ).value_at_risk
                    for column in range(position_pnl_full.shape[1])
                ],
                dtype=np.float64,
            )
        contributions = _contribution_summary(
            request,
            exposures,
            incremental_var=incremental,
            es_contributions=es_values,
        )
    worst: list[WorstRiskScenario] = []
    for index_value in worst_indices:
        index = int(index_value)
        shock = selected_rows.get(index)
        if shock is None:
            # max_tail_scenarios can exceed the exact ES tail for tiny samples
            for offset, batch in _mc_shock_batches(request, means, covariance):
                if offset <= index < offset + len(batch):
                    shock = batch[index - offset]
                    break
        assert shock is not None
        position_contributions = []
        if position_pnl_full is not None:
            position_contributions = [
                PositionTailContribution(position_id=position.position_id, loss=float(-value))
                for position, value in zip(
                    request.portfolio.positions, position_pnl_full[index], strict=True
                )
            ]
        elif index in selected_position_rows:
            position_contributions = [
                PositionTailContribution(position_id=position.position_id, loss=float(-value))
                for position, value in zip(
                    request.portfolio.positions, selected_position_rows[index], strict=True
                )
            ]
        worst.append(
            WorstRiskScenario(
                scenario_index=index,
                pnl=float(pnl[index]),
                loss=float(losses[index]),
                factor_shocks={
                    mapping.factor_id: float(value)
                    for mapping, value in zip(request.factor_mappings, shock, strict=True)
                },
                position_contributions=position_contributions,
            )
        )
    warnings = ["Monte Carlo factors are generated in bounded deterministic batches"]
    if relative_errors:
        warnings.append(
            "full repricing is authoritative; maximum local Greek approximation relative error "
            f"was {max(relative_errors):.6g}"
        )
    if request.config.antithetic:
        warnings.append(
            "antithetic factor draws are enabled; no universal tail-quantile improvement is assumed"
        )
    if request.config.valuation_mode is RiskValuationMode.DELTA_GAMMA_DIAGONAL:
        warnings.append("Delta-Gamma mode includes diagonal Gamma only; cross-Gamma is omitted")
    return PortfolioRiskReport(
        portfolio_id=request.portfolio.portfolio_id,
        base_value=base.total_value,
        value_at_risk=tail.value_at_risk,
        expected_shortfall=tail.expected_shortfall,
        confidence_level=request.config.confidence_level,
        horizon=request.config.horizon,
        method=RiskMethod.MONTE_CARLO,
        valuation_mode=request.config.valuation_mode,
        scenario_count=count,
        simulations=count,
        distribution=request.config.distribution,
        distribution_summary=distribution_summary(
            pnl,
            confidence_level=request.config.confidence_level,
            tail_count=len(tail.ordered_indices),
            quantile_method=request.config.quantile_method,
        ),
        worst_scenarios=worst,
        contributions=contributions,
        covariance_diagnostics=prepared.diagnostics,
        historical_start=(
            None
            if request.historical_dataset is None
            else request.historical_dataset.start_timestamp
        ),
        historical_end=(
            None if request.historical_dataset is None else request.historical_dataset.end_timestamp
        ),
        sampling_frequency=(
            None
            if request.historical_dataset is None
            else request.historical_dataset.sampling_frequency
        ),
        warnings=warnings,
        runtime_ms=(perf_counter() - started) * 1_000.0,
    )


def calculate_portfolio_risk(request: PortfolioRiskRequest) -> PortfolioRiskReport:
    """Dispatch a unified request without changing method-specific semantics."""

    common = {
        "portfolio": request.portfolio,
        "factor_mappings": request.factor_mappings,
        "config": request.config,
        "portfolio_config": request.portfolio_config,
    }
    if request.config.method is RiskMethod.HISTORICAL:
        assert request.historical_dataset is not None
        return calculate_historical_risk(
            HistoricalRiskRequest(**common, dataset=request.historical_dataset)
        )
    if request.config.method is RiskMethod.PARAMETRIC_NORMAL:
        return calculate_parametric_risk(
            ParametricRiskRequest(
                **common,
                covariance=request.covariance,
                historical_dataset=request.historical_dataset,
            )
        )
    return calculate_monte_carlo_risk(
        MonteCarloRiskRequest(
            **common,
            covariance=request.covariance,
            historical_dataset=request.historical_dataset,
        )
    )


def calculate_risk_report(request: PortfolioRiskReportRequest) -> PortfolioRiskReport:
    """Attach named stress losses while keeping them outside the VaR distribution."""

    risk = calculate_portfolio_risk(
        PortfolioRiskRequest(
            portfolio=request.portfolio,
            factor_mappings=request.factor_mappings,
            config=request.config,
            portfolio_config=request.portfolio_config,
            historical_dataset=request.historical_dataset,
            covariance=request.covariance,
        )
    )
    if not request.stress_scenarios:
        return risk
    stress = run_stress_scenarios(
        PortfolioStressRequest(
            portfolio=request.portfolio,
            config=request.portfolio_config,
            scenarios=request.stress_scenarios,
        )
    )
    return risk.model_copy(
        update={
            "stress_losses": [
                StressLossPoint(
                    name=item.name,
                    pnl=item.result.full_repricing_pnl,
                    loss=(
                        None
                        if item.result.full_repricing_pnl is None
                        else -item.result.full_repricing_pnl
                    ),
                )
                for item in stress.scenarios
            ],
            "warnings": [
                *risk.warnings,
                "named stress losses are reported separately and are not VaR distribution samples",
            ],
        }
    )
