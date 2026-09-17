"""Cross-engine pricing, fit, convergence, stability, and model-risk analytics."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from datetime import UTC, datetime
from statistics import median
from time import perf_counter
from typing import Any

import numpy as np

from backend.app.analytics.heston_calibration import split_heston_calibration_holdout
from backend.app.calibration.datasets import (
    calibration_dataset_from_surface,
    prepare_calibration_data,
)
from backend.app.core.exceptions import QuantitativeError
from backend.app.model_comparison.registry import (
    compatibility_exclusion,
    model_capabilities,
)
from backend.app.models.greeks import Greek, GreekMethod
from backend.app.models.heston import HestonParameters
from backend.app.models.heston_calibration import (
    CalibrationMoneynessBucket,
    CalibrationSelectionPolicy,
    HestonCalibrationConfig,
    HestonCalibrationHoldoutConfig,
    HestonCalibrationTarget,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.model_comparison import (
    CalibrationEnsembleMember,
    CalibrationEnsembleResult,
    CalibrationStabilityValuation,
    ComputeBudgetType,
    ComputeFrontierPoint,
    ComputeFrontierRequest,
    ComputeFrontierResult,
    ConsistencyCheck,
    ConsistencyRequest,
    ConsistencyResult,
    CrossModelDiagnostic,
    ExperimentRecord,
    FitMetrics,
    GreekDispersion,
    GroupFitMetrics,
    MarketFitPoint,
    MarketFitRequest,
    MarketFitResult,
    ModelCalibrationMetadata,
    ModelComparisonRequest,
    ModelComparisonResult,
    ModelComplexityFit,
    ModelGreekValues,
    ModelMarketFit,
    ModelOutput,
    ModelRiskFlag,
    ModelRiskReport,
    ModelRiskRequest,
    ModelRunStatus,
    ModelStabilityResult,
    OutOfSampleComparisonResult,
    OutOfSampleModelFit,
    PairwiseMatrix,
    ParameterUncertaintyResult,
    PriceDispersion,
    PriceRangeNormalization,
    PricingModel,
    ReferenceDeviation,
    ScalarDispersion,
    SmileFitMetrics,
    StabilityInput,
    StabilityPerturbation,
    StabilityPoint,
    SurfaceComparisonRequest,
    UncertaintyDecomposition,
)
from backend.app.models.option import ExerciseStyle, OptionContract
from backend.app.models.volatility import ImpliedVolatilityMarket, ImpliedVolatilityObservation
from backend.app.pricing.binomial import american_binomial_price, binomial_price
from backend.app.pricing.black_scholes import black_scholes_price
from backend.app.pricing.black_scholes_greeks import black_scholes_greeks
from backend.app.pricing.heston import heston_monte_carlo_price
from backend.app.pricing.heston_analytical import (
    HestonAnalyticalBatchPricer,
    heston_analytical_price,
)
from backend.app.pricing.implied_volatility import solve_implied_volatility
from backend.app.pricing.longstaff_schwartz import longstaff_schwartz_price
from backend.app.pricing.monte_carlo import MonteCarloPricer
from backend.app.pricing.monte_carlo_greeks import monte_carlo_greeks
from backend.app.simulation.random import spawn_child_seeds


def deterministic_signature(value: Any) -> str:
    """Hash a canonical JSON representation without runtime or credential state."""

    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")

    def normalize(item: Any) -> Any:
        if isinstance(item, dict):
            excluded = {
                "runtime_ms",
                "simulation_runtime_ms",
                "regression_runtime_ms",
                "valuation_runtime_ms",
                "prices_per_second",
            }
            return {
                str(key): normalize(child) for key, child in item.items() if key not in excluded
            }
        if isinstance(item, list):
            return [normalize(child) for child in item]
        return item

    encoded = json.dumps(normalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def scalar_dispersion(values: Sequence[float]) -> ScalarDispersion:
    """Population dispersion for a descriptive finite collection."""

    if not values:
        return ScalarDispersion(count=0)
    array = np.asarray(values, dtype=np.float64)
    return ScalarDispersion(
        count=len(values),
        minimum=float(np.min(array)),
        maximum=float(np.max(array)),
        range=float(np.ptp(array)),
        mean=float(np.mean(array)),
        median=float(np.median(array)),
        standard_deviation=float(np.std(array, ddof=0)),
    )


def price_dispersion(
    prices: Sequence[float],
    *,
    normalization: PriceRangeNormalization,
    spot: float,
    reference_price: float | None,
    floor: float,
) -> PriceDispersion:
    base = scalar_dispersion(prices)
    if not prices:
        return PriceDispersion(**base.model_dump(), normalization=normalization)
    denominator: float | None
    if normalization is PriceRangeNormalization.SPOT:
        denominator = spot
    elif normalization is PriceRangeNormalization.REFERENCE_PRICE:
        denominator = reference_price
    else:
        denominator = base.median
    if denominator is None:
        denominator = base.median
    assert denominator is not None
    safe_denominator = max(abs(float(denominator)), floor)
    assert base.range is not None
    return PriceDispersion(
        **base.model_dump(),
        normalization=normalization,
        normalization_value=safe_denominator,
        relative_range=base.range / safe_denominator,
    )


def pairwise_matrix(outputs: Sequence[ModelOutput], floor: float) -> PairwiseMatrix | None:
    successful = [
        item for item in outputs if item.status is ModelRunStatus.SUCCESS and item.price is not None
    ]
    if not successful:
        return None
    prices = []
    for item in successful:
        assert item.price is not None
        prices.append(item.price)
    signed = [[left - right for right in prices] for left in prices]
    relative = [[(left - right) / max(abs(right), floor) for right in prices] for left in prices]
    return PairwiseMatrix(
        models=[item.model for item in successful],
        signed_differences=signed,
        relative_differences=relative,
        relative_denominator="price_j with absolute floor",
    )


def greek_dispersion(outputs: Sequence[ModelOutput]) -> list[GreekDispersion]:
    results: list[GreekDispersion] = []
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        available = [
            (item.model, float(value))
            for item in outputs
            if item.status is ModelRunStatus.SUCCESS
            and (value := getattr(item.greeks, name)) is not None
        ]
        values = [value for _, value in available]
        results.append(
            GreekDispersion(
                greek=name.upper(),
                dispersion=scalar_dispersion(values),
                models=[model for model, _ in available],
                pairwise_differences=[[left - right for right in values] for left in values],
            )
        )
    return results


def _heston_parameters(request: ModelComparisonRequest) -> HestonParameters | None:
    if request.heston_parameters is not None:
        return request.heston_parameters
    if request.heston_calibration_result is not None:
        return request.heston_calibration_result.calibrated_parameters
    return None


def _calibration_metadata(request: ModelComparisonRequest) -> ModelCalibrationMetadata:
    result = request.heston_calibration_result
    if result is None:
        return ModelCalibrationMetadata(
            calibration_required=True,
            source="explicit_parameters" if request.heston_parameters is not None else None,
        )
    return ModelCalibrationMetadata(
        calibration_required=True,
        calibration_runtime_ms=result.runtime_ms,
        objective_evaluations=result.evaluations,
        observations=result.observations_used,
        prices_per_second=result.prices_per_second,
        fitted_rmse=result.rmse,
        source="phase14_calibration_result",
    )


def _equivalent_iv(
    contract: OptionContract, market: MarketEnvironment, price: float
) -> float | None:
    if contract.exercise_style is not ExerciseStyle.EUROPEAN or contract.maturity_years == 0.0:
        return None
    try:
        result = solve_implied_volatility(
            ImpliedVolatilityObservation(
                contract=contract,
                market=ImpliedVolatilityMarket(
                    risk_free_rate=market.risk_free_rate,
                    dividend_yield=market.dividend_yield,
                ),
                observed_price=price,
            )
        )
    except QuantitativeError:
        return None
    return result.implied_volatility


def _heston_greeks(
    request: ModelComparisonRequest, parameters: HestonParameters
) -> ModelGreekValues:
    contract = request.contract
    market = request.market
    bumps = request.config.heston_greek_bumps
    spot_bump = contract.spot * bumps.relative_spot_bump

    def price(c: OptionContract, m: MarketEnvironment) -> float:
        return heston_analytical_price(c, m, parameters, request.heston_integration).price

    base = price(contract, market)
    up_contract = contract.model_copy(update={"spot": contract.spot + spot_bump})
    down_contract = contract.model_copy(update={"spot": contract.spot - spot_bump})
    up = price(up_contract, market)
    down = price(down_contract, market)
    delta = (up - down) / (2.0 * spot_bump)
    gamma = (up - 2.0 * base + down) / (spot_bump * spot_bump)
    rate_bump = bumps.rate_bump
    rate_up = price(
        contract,
        market.model_copy(update={"risk_free_rate": market.risk_free_rate + rate_bump}),
    )
    rate_down = price(
        contract,
        market.model_copy(update={"risk_free_rate": market.risk_free_rate - rate_bump}),
    )
    rho = (rate_up - rate_down) / (2.0 * rate_bump)
    maturity_bump = min(bumps.maturity_bump, contract.maturity_years / 2.0)
    if maturity_bump > 0.0:
        later = price(
            contract.model_copy(update={"maturity_years": contract.maturity_years + maturity_bump}),
            market,
        )
        earlier = price(
            contract.model_copy(update={"maturity_years": contract.maturity_years - maturity_bump}),
            market,
        )
        theta = -(later - earlier) / (2.0 * maturity_bump)
    else:
        theta = 0.0
    return ModelGreekValues(delta=delta, gamma=gamma, theta=theta, rho=rho)


def _run_model_once(
    model: PricingModel,
    request: ModelComparisonRequest,
    seed: int,
) -> ModelOutput:
    contract, market = request.contract, request.market
    parameters = _heston_parameters(request)
    warnings: list[str] = []
    if model is PricingModel.BLACK_SCHOLES:
        started = perf_counter()
        greeks = black_scholes_greeks(contract, market) if request.config.include_greeks else None
        price = greeks.option_price if greeks is not None else black_scholes_price(contract, market)
        elapsed = (perf_counter() - started) * 1_000.0
        values = (
            ModelGreekValues(
                delta=greeks.delta,
                gamma=greeks.gamma,
                vega=greeks.vega,
                theta=greeks.theta,
                rho=greeks.rho,
            )
            if greeks is not None
            else ModelGreekValues()
        )
        return ModelOutput(
            model=model,
            status=ModelRunStatus.SUCCESS,
            price=price,
            runtime_ms=elapsed,
            deterministic=True,
            greeks=values,
            calibration=ModelCalibrationMetadata(
                calibration_required=False, source="calibration_not_required"
            ),
        )
    if model is PricingModel.MONTE_CARLO_GBM:
        config = request.gbm_monte_carlo.model_copy(update={"seed": seed})
        mc_result = MonteCarloPricer().price(contract, market, config)
        values = ModelGreekValues()
        if request.config.include_greeks:
            try:
                estimates = monte_carlo_greeks(
                    contract,
                    market,
                    config,
                    [Greek.DELTA, Greek.GAMMA, Greek.VEGA, Greek.THETA, Greek.RHO],
                    GreekMethod.FINITE_DIFFERENCE,
                )
                by_name = {item.greek.value.lower(): item.estimate for item in estimates.results}
                values = ModelGreekValues(**by_name)
            except (QuantitativeError, ValueError) as exc:
                warnings.append(f"Monte Carlo Greek comparison unavailable: {exc}")
        return ModelOutput(
            model=model,
            status=ModelRunStatus.SUCCESS,
            price=mc_result.price,
            runtime_ms=mc_result.runtime_ms,
            deterministic=False,
            standard_error=mc_result.standard_error,
            confidence_interval_low=mc_result.confidence_interval_low,
            confidence_interval_high=mc_result.confidence_interval_high,
            greeks=values,
            paths=mc_result.num_simulations,
            warnings=warnings,
            calibration=ModelCalibrationMetadata(
                calibration_required=False, source="calibration_not_required"
            ),
        )
    if model in {PricingModel.CRR_BINOMIAL, PricingModel.CRR_AMERICAN}:
        tree_result = (
            american_binomial_price(contract, market, request.tree)
            if model is PricingModel.CRR_AMERICAN
            else binomial_price(contract, market, request.tree)
        )
        values = ModelGreekValues(
            delta=tree_result.delta, gamma=tree_result.gamma, theta=tree_result.theta
        )
        return ModelOutput(
            model=model,
            status=ModelRunStatus.SUCCESS,
            price=tree_result.price,
            runtime_ms=tree_result.runtime_ms,
            deterministic=True,
            greeks=values if request.config.include_greeks else ModelGreekValues(),
            tree_steps=tree_result.steps,
            calibration=ModelCalibrationMetadata(
                calibration_required=False, source="calibration_not_required"
            ),
        )
    if model is PricingModel.HESTON_ANALYTICAL:
        assert parameters is not None
        analytical_result = heston_analytical_price(
            contract, market, parameters, request.heston_integration
        )
        values = ModelGreekValues()
        if request.config.include_greeks:
            values = _heston_greeks(request, parameters)
            warnings.append(
                "Generic Heston Vega is omitted because v0 and sqrt(v0) sensitivities "
                "are not interchangeable with BSM Vega."
            )
        return ModelOutput(
            model=model,
            status=ModelRunStatus.SUCCESS,
            price=analytical_result.price,
            runtime_ms=analytical_result.runtime_ms,
            deterministic=True,
            greeks=values,
            calibration=_calibration_metadata(request),
            warnings=warnings,
        )
    if model is PricingModel.HESTON_MONTE_CARLO:
        assert parameters is not None
        heston_config = request.heston_monte_carlo.model_copy(update={"seed": seed})
        heston_result = heston_monte_carlo_price(contract, market, parameters, heston_config)
        return ModelOutput(
            model=model,
            status=ModelRunStatus.SUCCESS,
            price=heston_result.price,
            runtime_ms=heston_result.runtime_ms,
            deterministic=False,
            standard_error=heston_result.standard_error,
            confidence_interval_low=heston_result.confidence_interval_low,
            confidence_interval_high=heston_result.confidence_interval_high,
            calibration=_calibration_metadata(request),
            paths=heston_result.num_simulations,
            time_steps=heston_result.num_time_steps,
            warnings=["Heston Monte Carlo Greeks are unavailable in Phase 15."]
            if request.config.include_greeks
            else [],
        )
    if model is PricingModel.LONGSTAFF_SCHWARTZ:
        lsm_config = request.longstaff_schwartz.model_copy(update={"seed": seed})
        lsm_result = longstaff_schwartz_price(contract, market, lsm_config)
        return ModelOutput(
            model=model,
            status=ModelRunStatus.SUCCESS,
            price=lsm_result.price,
            runtime_ms=lsm_result.runtime_ms,
            deterministic=False,
            standard_error=lsm_result.standard_error,
            confidence_interval_low=lsm_result.confidence_interval_low,
            confidence_interval_high=lsm_result.confidence_interval_high,
            paths=lsm_result.num_paths,
            time_steps=lsm_result.num_time_steps,
            early_exercise_fraction=lsm_result.early_exercise_fraction,
            warnings=["Longstaff-Schwartz Greeks are unavailable."]
            if request.config.include_greeks
            else [],
            calibration=ModelCalibrationMetadata(
                calibration_required=False, source="calibration_not_required"
            ),
        )
    raise ValueError(f"unregistered pricing model {model}")


def _run_model(model: PricingModel, request: ModelComparisonRequest, seed: int) -> ModelOutput:
    try:
        runs = [
            _run_model_once(model, request, seed) for _ in range(request.config.runtime_repetitions)
        ]
        first = runs[0]
        runtimes = [item.runtime_ms for item in runs if item.runtime_ms is not None]
        runtime = float(median(runtimes)) if runtimes else first.runtime_ms
        price = first.price
        iv = (
            _equivalent_iv(request.contract, request.market, price)
            if price is not None and request.config.include_iv
            else None
        )
        return first.model_copy(
            update={
                "runtime_ms": runtime,
                "implied_volatility": iv,
                "implied_volatility_residual": (
                    iv - request.market.volatility if iv is not None else None
                ),
            }
        )
    except Exception as exc:  # per-model isolation is deliberate at this boundary
        return ModelOutput(
            model=model,
            status=ModelRunStatus.FAILED,
            deterministic=model_capabilities(model).deterministic,
            error_type=type(exc).__name__,
            message=str(exc),
        )


def compare_models(request: ModelComparisonRequest) -> ModelComparisonResult:
    """Run compatible engines independently and retain partial failures."""

    started = perf_counter()
    parameters = _heston_parameters(request)
    exclusions = []
    compatible: list[PricingModel] = []
    for model in request.config.selected_models:
        exclusion = compatibility_exclusion(
            model, request.contract, heston_parameters_available=parameters is not None
        )
        if exclusion is None:
            compatible.append(model)
        else:
            exclusions.append(exclusion)
    seeds = spawn_child_seeds(request.config.deterministic_seed, len(compatible))
    outputs = [
        _run_model(model, request, seed) for model, seed in zip(compatible, seeds, strict=True)
    ]
    successful = [
        item for item in outputs if item.status is ModelRunStatus.SUCCESS and item.price is not None
    ]
    reference = next(
        (item for item in successful if item.model is request.config.reference_model), None
    )
    prices = [float(item.price) for item in successful if item.price is not None]
    dispersion = price_dispersion(
        prices,
        normalization=request.config.normalization,
        spot=request.contract.spot,
        reference_price=float(reference.price)
        if reference is not None and reference.price is not None
        else None,
        floor=request.config.relative_price_floor,
    )
    reference_deviations: list[ReferenceDeviation] = []
    if reference is not None and reference.price is not None:
        reference_price = float(reference.price)
        for item in successful:
            assert item.price is not None
            difference = item.price - reference_price
            reference_deviations.append(
                ReferenceDeviation(
                    model=item.model,
                    reference_model=reference.model,
                    signed_difference=difference,
                    absolute_difference=abs(difference),
                    relative_difference=difference
                    / max(abs(reference_price), request.config.relative_price_floor),
                )
            )
    greek_stats = greek_dispersion(outputs) if request.config.include_greeks else []
    output_by_model = {item.model: item for item in successful}
    diagnostic_pairs = (
        ("BSM_VS_GBM_MONTE_CARLO", PricingModel.MONTE_CARLO_GBM, PricingModel.BLACK_SCHOLES),
        ("BSM_VS_CRR", PricingModel.CRR_BINOMIAL, PricingModel.BLACK_SCHOLES),
        (
            "HESTON_ANALYTICAL_VS_MONTE_CARLO",
            PricingModel.HESTON_MONTE_CARLO,
            PricingModel.HESTON_ANALYTICAL,
        ),
        (
            "AMERICAN_CRR_VS_LONGSTAFF_SCHWARTZ",
            PricingModel.LONGSTAFF_SCHWARTZ,
            PricingModel.CRR_AMERICAN,
        ),
    )
    diagnostics: list[CrossModelDiagnostic] = []
    for name, left_model, right_model in diagnostic_pairs:
        left = output_by_model.get(left_model)
        right = output_by_model.get(right_model)
        if left is None or right is None or left.price is None or right.price is None:
            continue
        difference = left.price - right.price
        diagnostics.append(
            CrossModelDiagnostic(
                name=name,
                left_model=left_model,
                right_model=right_model,
                signed_difference=difference,
                absolute_difference=abs(difference),
                z_score=(
                    difference / left.standard_error
                    if left.standard_error is not None and left.standard_error > 0.0
                    else None
                ),
                runtime_ratio=(
                    left.runtime_ms / right.runtime_ms
                    if left.runtime_ms is not None
                    and right.runtime_ms is not None
                    and right.runtime_ms > 0.0
                    else None
                ),
            )
        )
    flags: list[ModelRiskFlag] = []
    if (
        dispersion.relative_range is not None
        and dispersion.relative_range > request.config.thresholds.max_relative_price_dispersion
    ):
        flags.append(ModelRiskFlag.LARGE_PRICE_DISPERSION)
    if any(
        item.greek == "DELTA"
        and item.dispersion.range is not None
        and item.dispersion.range > request.config.thresholds.max_delta_dispersion
        for item in greek_stats
    ):
        flags.append(ModelRiskFlag.LARGE_GREEK_DISPERSION)
    ivs = [item.implied_volatility for item in successful if item.implied_volatility is not None]
    iv_range = scalar_dispersion([float(value) for value in ivs]).range
    if iv_range is not None and iv_range > request.config.thresholds.max_iv_dispersion:
        flags.append(ModelRiskFlag.LARGE_IV_DISPERSION)
    if any(
        item.standard_error is not None
        and item.price is not None
        and item.standard_error / max(item.price, request.config.relative_price_floor)
        > request.config.thresholds.max_relative_monte_carlo_se
        for item in successful
    ):
        flags.append(ModelRiskFlag.HIGH_MONTE_CARLO_UNCERTAINTY)
    if any(
        diagnostic.absolute_difference > request.config.thresholds.numerical_consistency_tolerance
        and (diagnostic.z_score is None or abs(diagnostic.z_score) > 4.0)
        for diagnostic in diagnostics
    ):
        flags.append(ModelRiskFlag.NUMERICAL_CONSISTENCY_WARNING)
    signature = deterministic_signature(request)
    elapsed = (perf_counter() - started) * 1_000.0
    return ModelComparisonResult(
        contract=request.contract,
        market=request.market,
        outputs=outputs,
        exclusions=exclusions,
        price_dispersion=dispersion,
        pairwise=pairwise_matrix(outputs, request.config.relative_price_floor)
        if request.config.include_pairwise
        else None,
        implied_volatility_pairwise=(
            pairwise_matrix(
                [
                    item.model_copy(update={"price": item.implied_volatility})
                    for item in outputs
                    if item.implied_volatility is not None
                ],
                1e-8,
            )
            if request.config.include_pairwise and request.config.include_iv
            else None
        ),
        reference_deviations=reference_deviations,
        greek_dispersion=greek_stats,
        consistency_diagnostics=diagnostics,
        flags=flags,
        successful_models=[item.model for item in successful],
        failed_models=[item.model for item in outputs if item.status is ModelRunStatus.FAILED],
        excluded_models=[item.model for item in exclusions],
        experiment=ExperimentRecord(
            experiment_identifier=f"model-comparison-{signature[:16]}",
            timestamp=datetime.now(UTC),
            input_signature=signature,
            model_configs={
                "platform_version": "0.15.0",
                "comparison": request.config.model_dump(mode="json"),
                "gbm_monte_carlo": request.gbm_monte_carlo.model_dump(mode="json"),
                "tree": request.tree.model_dump(mode="json"),
                "heston_monte_carlo": request.heston_monte_carlo.model_dump(mode="json"),
                "heston_integration": request.heston_integration.model_dump(mode="json"),
                "longstaff_schwartz": request.longstaff_schwartz.model_dump(mode="json"),
                "heston_parameters": (
                    parameters.model_dump(mode="json") if parameters is not None else None
                ),
            },
            total_runtime_ms=elapsed,
        ),
    )


def _fit_metrics(points: Sequence[MarketFitPoint]) -> FitMetrics:
    valid = [
        point
        for point in points
        if point.status is ModelRunStatus.SUCCESS and point.price_residual is not None
    ]
    if not valid:
        return FitMetrics(observation_count=0, bid_ask_observation_count=0, iv_observation_count=0)
    residual_values = []
    for point in valid:
        assert point.price_residual is not None
        residual_values.append(point.price_residual)
    residuals = np.asarray(residual_values)
    normalized_residuals = np.asarray(
        [
            float(point.normalized_price_residual)
            for point in valid
            if point.normalized_price_residual is not None
        ]
    )
    weights = np.asarray([point.weight for point in valid])
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        weights = np.ones_like(residuals)
        weight_sum = float(len(valid))
    spread = [point.spread_distance for point in valid if point.spread_distance is not None]
    iv_residuals = [
        float(point.implied_volatility_residual)
        for point in valid
        if point.implied_volatility_residual is not None
    ]
    return FitMetrics(
        observation_count=len(valid),
        rmse=float(np.sqrt(np.mean(residuals * residuals))),
        mae=float(np.mean(np.abs(residuals))),
        weighted_rmse=float(np.sqrt(np.sum(weights * residuals * residuals) / weight_sum)),
        normalized_rmse=(
            float(np.sqrt(np.mean(normalized_residuals * normalized_residuals)))
            if len(normalized_residuals)
            else None
        ),
        normalized_mae=(
            float(np.mean(np.abs(normalized_residuals))) if len(normalized_residuals) else None
        ),
        max_absolute_error=float(np.max(np.abs(residuals))),
        median_absolute_error=float(np.median(np.abs(residuals))),
        bias=float(np.mean(residuals)),
        bid_ask_observation_count=len(spread),
        fraction_inside_bid_ask=(
            sum(math.isclose(float(value), 0.0, abs_tol=1e-15) for value in spread) / len(spread)
            if spread
            else None
        ),
        average_distance_outside_spread=float(np.mean(spread)) if spread else None,
        maximum_distance_outside_spread=float(np.max(spread)) if spread else None,
        iv_observation_count=len(iv_residuals),
        iv_rmse=float(np.sqrt(np.mean(np.square(iv_residuals)))) if iv_residuals else None,
        iv_mae=float(np.mean(np.abs(iv_residuals))) if iv_residuals else None,
        max_iv_error=float(np.max(np.abs(iv_residuals))) if iv_residuals else None,
        iv_bias=float(np.mean(iv_residuals)) if iv_residuals else None,
    )


def _moneyness_bucket(point: MarketFitPoint) -> CalibrationMoneynessBucket:
    signed = point.log_moneyness if point.option_type == "CALL" else -point.log_moneyness
    if signed > 0.20:
        return CalibrationMoneynessBucket.DEEP_OTM
    if signed > 0.05:
        return CalibrationMoneynessBucket.OTM
    if signed >= -0.05:
        return CalibrationMoneynessBucket.ATM
    return CalibrationMoneynessBucket.ITM


def _group_fit(
    points: Sequence[MarketFitPoint],
) -> tuple[list[GroupFitMetrics], list[GroupFitMetrics]]:
    per_expiry = [
        GroupFitMetrics(
            label=f"{maturity:.12g}",
            maturity_years=maturity,
            metrics=_fit_metrics([point for point in points if point.maturity_years == maturity]),
        )
        for maturity in sorted({point.maturity_years for point in points})
    ]
    per_moneyness = []
    for bucket in CalibrationMoneynessBucket:
        selected = [point for point in points if _moneyness_bucket(point) is bucket]
        if selected:
            per_moneyness.append(
                GroupFitMetrics(label=bucket.value, metrics=_fit_metrics(selected))
            )
    return per_expiry, per_moneyness


def _smile_values(
    points: Sequence[MarketFitPoint], *, model: bool
) -> tuple[float | None, float | None, float | None]:
    from scipy.interpolate import PchipInterpolator

    coordinates: dict[float, float] = {}
    for point in points:
        value = point.model_implied_volatility if model else point.market_implied_volatility
        if value is not None:
            coordinates[point.log_moneyness] = value
    if len(coordinates) < 2:
        return None, None, None
    ordered = sorted(coordinates.items())
    x = np.asarray([item[0] for item in ordered])
    maturity = points[0].maturity_years
    y = np.asarray([item[1] * item[1] * maturity for item in ordered])
    if x[0] > 0.0 or x[-1] < 0.0:
        return None, None, None
    interpolator = PchipInterpolator(x, y, extrapolate=False)
    atm = math.sqrt(max(float(interpolator(0.0)), 0.0) / maturity)
    bump = 0.01
    if x[0] > -bump or x[-1] < bump:
        return atm, None, None
    low = math.sqrt(max(float(interpolator(-bump)), 0.0) / maturity)
    high = math.sqrt(max(float(interpolator(bump)), 0.0) / maturity)
    return atm, (high - low) / (2.0 * bump), (high - 2.0 * atm + low) / (bump * bump)


def _smile_fit(points: Sequence[MarketFitPoint]) -> list[SmileFitMetrics]:
    results = []
    for maturity in sorted({point.maturity_years for point in points}):
        selected = [point for point in points if point.maturity_years == maturity]
        market_atm, market_skew, market_curvature = _smile_values(selected, model=False)
        model_atm, model_skew, model_curvature = _smile_values(selected, model=True)
        results.append(
            SmileFitMetrics(
                maturity_years=maturity,
                market_atm_iv=market_atm,
                model_atm_iv=model_atm,
                atm_iv_residual=(
                    model_atm - market_atm
                    if model_atm is not None and market_atm is not None
                    else None
                ),
                market_skew=market_skew,
                model_skew=model_skew,
                skew_residual=(
                    model_skew - market_skew
                    if model_skew is not None and market_skew is not None
                    else None
                ),
                market_curvature=market_curvature,
                model_curvature=model_curvature,
                curvature_residual=(
                    model_curvature - market_curvature
                    if model_curvature is not None and market_curvature is not None
                    else None
                ),
            )
        )
    return results


def _market_model_price(
    model: PricingModel,
    request: MarketFitRequest,
    contract: OptionContract,
    market_iv: float,
    parameters: HestonParameters | None,
    seed: int,
    heston_batches: dict[float, HestonAnalyticalBatchPricer],
) -> tuple[float, float]:
    market_volatility = (
        market_iv
        if model is PricingModel.BLACK_SCHOLES and request.use_observed_iv_for_black_scholes
        else request.black_scholes_volatility
    )
    market = MarketEnvironment(
        risk_free_rate=request.dataset.risk_free_rate,
        dividend_yield=request.dataset.dividend_yield,
        volatility=market_volatility,
    )
    started = perf_counter()
    if model is PricingModel.BLACK_SCHOLES:
        price = black_scholes_price(contract, market)
    elif model is PricingModel.CRR_BINOMIAL:
        price = binomial_price(contract, market, request.tree).price
    elif model is PricingModel.MONTE_CARLO_GBM:
        gbm_config = request.gbm_monte_carlo.model_copy(update={"seed": seed})
        price = MonteCarloPricer().price(contract, market, gbm_config).price
    elif model is PricingModel.HESTON_ANALYTICAL:
        if parameters is None:
            raise ValueError("Heston parameters are required")
        batch = heston_batches.get(contract.maturity_years)
        if batch is None:
            batch = HestonAnalyticalBatchPricer(
                spot=contract.spot,
                maturity_years=contract.maturity_years,
                market=market,
                parameters=parameters,
                config=request.integration,
            )
            heston_batches[contract.maturity_years] = batch
        price = batch.price(contract).price
    elif model is PricingModel.HESTON_MONTE_CARLO:
        if parameters is None:
            raise ValueError("Heston parameters are required")
        heston_config = request.heston_monte_carlo.model_copy(update={"seed": seed})
        price = heston_monte_carlo_price(contract, market, parameters, heston_config).price
    else:
        raise ValueError(
            "market-fit datasets contain European observations; American engines are incompatible"
        )
    return price, (perf_counter() - started) * 1_000.0


def evaluate_market_fit(request: MarketFitRequest) -> MarketFitResult:
    """Evaluate selected models against one filtered Phase 14 observation dataset."""

    prepared = prepare_calibration_data(request.dataset, request.preprocessing)
    parameters = request.heston_parameters
    if parameters is None and request.heston_calibration_result is not None:
        parameters = request.heston_calibration_result.calibrated_parameters
    all_models: list[ModelMarketFit] = []
    warnings: list[str] = []
    seed_count = len(request.selected_models) * len(prepared.observations)
    seeds = iter(spawn_child_seeds(request.preprocessing.random_seed, seed_count))
    for model in request.selected_models:
        capability = model_capabilities(model)
        if not capability.supports_european or (
            capability.requires_calibration and parameters is None
        ):
            message = (
                "model does not support European market-fit observations"
                if not capability.supports_european
                else "Heston parameters or a calibration result are required"
            )
            all_models.append(
                ModelMarketFit(
                    model=model,
                    status=ModelRunStatus.EXCLUDED,
                    metrics=FitMetrics(
                        observation_count=0, bid_ask_observation_count=0, iv_observation_count=0
                    ),
                    per_expiry=[],
                    per_moneyness=[],
                    per_metadata_group=[],
                    smile_fit=[],
                    residuals=[],
                    runtime_ms=0.0,
                    message=message,
                )
            )
            for _ in prepared.observations:
                next(seeds)
            continue
        points: list[MarketFitPoint] = []
        total_runtime = 0.0
        heston_batches: dict[float, HestonAnalyticalBatchPricer] = {}
        for item in prepared.observations:
            seed = next(seeds)
            observation = item.observation
            try:
                price, runtime = _market_model_price(
                    model,
                    request,
                    item.contract,
                    item.market_implied_volatility,
                    parameters,
                    seed,
                    heston_batches,
                )
                total_runtime += runtime
                model_iv = _equivalent_iv(
                    item.contract,
                    MarketEnvironment(
                        risk_free_rate=request.dataset.risk_free_rate,
                        dividend_yield=request.dataset.dividend_yield,
                        volatility=request.black_scholes_volatility,
                    ),
                    price,
                )
                if observation.bid is not None and observation.ask is not None:
                    if observation.bid <= price <= observation.ask:
                        spread_distance = 0.0
                    else:
                        spread_distance = min(
                            abs(price - observation.bid), abs(price - observation.ask)
                        )
                else:
                    spread_distance = None
                points.append(
                    MarketFitPoint(
                        model=model,
                        observation_index=item.original_index,
                        strike=observation.strike,
                        maturity_years=observation.maturity_years,
                        moneyness=item.moneyness,
                        log_moneyness=item.log_moneyness,
                        option_type=observation.option_type.value,
                        market_price=item.market_price,
                        model_price=price,
                        price_residual=price - item.market_price,
                        normalized_price_residual=(price - item.market_price)
                        / max(abs(item.market_price), request.relative_price_floor),
                        market_implied_volatility=item.market_implied_volatility,
                        model_implied_volatility=model_iv,
                        implied_volatility_residual=(
                            model_iv - item.market_implied_volatility
                            if model_iv is not None
                            else None
                        ),
                        bid=observation.bid,
                        ask=observation.ask,
                        spread_distance=spread_distance,
                        weight=item.weight,
                        status=ModelRunStatus.SUCCESS,
                    )
                )
            except Exception as exc:
                points.append(
                    MarketFitPoint(
                        model=model,
                        observation_index=item.original_index,
                        strike=observation.strike,
                        maturity_years=observation.maturity_years,
                        moneyness=item.moneyness,
                        log_moneyness=item.log_moneyness,
                        option_type=observation.option_type.value,
                        market_price=item.market_price,
                        model_price=None,
                        price_residual=None,
                        normalized_price_residual=None,
                        market_implied_volatility=item.market_implied_volatility,
                        model_implied_volatility=None,
                        implied_volatility_residual=None,
                        bid=observation.bid,
                        ask=observation.ask,
                        spread_distance=None,
                        weight=item.weight,
                        status=ModelRunStatus.FAILED,
                        message=f"{type(exc).__name__}: {exc}",
                    )
                )
        valid_count = sum(point.status is ModelRunStatus.SUCCESS for point in points)
        status = ModelRunStatus.SUCCESS if valid_count else ModelRunStatus.FAILED
        if valid_count != len(points):
            warnings.append(
                f"{model.value}: {len(points) - valid_count} observation evaluations failed"
            )
        per_expiry, per_moneyness = _group_fit(points)
        per_metadata_group = [
            GroupFitMetrics(
                label=label,
                metrics=_fit_metrics(
                    [
                        point
                        for point in points
                        if request.observation_labels.get(point.observation_index) == label
                    ]
                ),
            )
            for label in sorted(set(request.observation_labels.values()))
        ]
        calibration = None
        if capability.requires_calibration:
            if request.heston_calibration_result is not None:
                result = request.heston_calibration_result
                calibration = ModelCalibrationMetadata(
                    calibration_required=True,
                    calibration_runtime_ms=result.runtime_ms,
                    objective_evaluations=result.evaluations,
                    observations=result.observations_used,
                    prices_per_second=result.prices_per_second,
                    fitted_rmse=result.rmse,
                    source="phase14_calibration_result",
                )
            else:
                calibration = ModelCalibrationMetadata(
                    calibration_required=True, source="explicit_parameters"
                )
        all_models.append(
            ModelMarketFit(
                model=model,
                status=status,
                metrics=_fit_metrics(points),
                per_expiry=per_expiry,
                per_moneyness=per_moneyness,
                per_metadata_group=per_metadata_group,
                smile_fit=_smile_fit(points),
                residuals=points[: request.max_returned_residuals],
                calibration=calibration,
                runtime_ms=total_runtime,
                message=None
                if status is ModelRunStatus.SUCCESS
                else "all model evaluations failed",
            )
        )
    return MarketFitResult(
        models=all_models,
        observations_used=len(prepared.observations),
        observations_excluded=len(prepared.exclusions),
        exclusion_reasons=prepared.exclusion_reasons,
        input_signature=deterministic_signature(request),
        warnings=warnings,
    )


def compare_surface(request: SurfaceComparisonRequest) -> MarketFitResult:
    """Flatten a Phase 6 surface and compare model-equivalent prices and IVs."""

    dataset = calibration_dataset_from_surface(request.surface)
    return evaluate_market_fit(
        MarketFitRequest(
            dataset=dataset,
            selected_models=request.selected_models,
            heston_parameters=request.heston_parameters,
            heston_calibration_result=request.heston_calibration_result,
            black_scholes_volatility=0.20,
            use_observed_iv_for_black_scholes=True,
            preprocessing=HestonCalibrationConfig(
                target=HestonCalibrationTarget.PRICE,
                selection_policy=CalibrationSelectionPolicy.ALL,
            ),
            tree=request.tree,
            integration=request.integration,
            gbm_monte_carlo=request.gbm_monte_carlo,
            heston_monte_carlo=request.heston_monte_carlo,
            max_returned_residuals=request.max_returned_residuals,
        )
    )


def compare_models_out_of_sample(
    request: MarketFitRequest,
    holdout: HestonCalibrationHoldoutConfig,
) -> OutOfSampleComparisonResult:
    """Keep structured training and validation metrics separate."""

    training, validation = split_heston_calibration_holdout(
        request.dataset, request.preprocessing, holdout
    )
    training_result = evaluate_market_fit(request.model_copy(update={"dataset": training}))
    validation_result = evaluate_market_fit(request.model_copy(update={"dataset": validation}))
    validation_by_model = {item.model: item for item in validation_result.models}
    models = [
        OutOfSampleModelFit(
            model=item.model,
            training_metrics=item.metrics,
            validation_metrics=validation_by_model[item.model].metrics,
        )
        for item in training_result.models
        if item.model in validation_by_model
    ]
    return OutOfSampleComparisonResult(
        holdout=holdout,
        training_observations=training_result.observations_used,
        validation_observations=validation_result.observations_used,
        models=models,
    )


def compute_frontier(request: ComputeFrontierRequest) -> ComputeFrontierResult:
    """Vary the natural compute budget while holding the model inputs fixed."""

    base = request.comparison
    reference_request = base.model_copy(
        update={
            "config": base.config.model_copy(
                update={
                    "selected_models": [request.reference_model],
                    "reference_model": request.reference_model,
                    "runtime_repetitions": 1,
                    "include_greeks": False,
                }
            )
        }
    )
    reference_result = compare_models(reference_request)
    reference_output = next(
        (item for item in reference_result.outputs if item.status is ModelRunStatus.SUCCESS), None
    )
    if reference_output is None or reference_output.price is None:
        raise ValueError("compute-frontier reference model did not produce a price")
    reference_price = reference_output.price
    child_seeds = spawn_child_seeds(base.config.deterministic_seed, len(request.budgets))
    points: list[ComputeFrontierPoint] = []
    budget_type = request.budget_type
    if budget_type is None:
        budget_type = (
            ComputeBudgetType.TREE_STEPS
            if request.model in {PricingModel.CRR_BINOMIAL, PricingModel.CRR_AMERICAN}
            else ComputeBudgetType.PATHS
        )
    for budget, seed in zip(request.budgets, child_seeds, strict=True):
        updates: dict[str, Any] = {}
        approximate_memory: int | None = None
        if request.model is PricingModel.MONTE_CARLO_GBM:
            updates["gbm_monte_carlo"] = base.gbm_monte_carlo.model_copy(
                update={"num_simulations": budget, "seed": seed}
            )
            budget_type = ComputeBudgetType.PATHS
            approximate_memory = budget * 8 * 3
        elif request.model in {PricingModel.CRR_BINOMIAL, PricingModel.CRR_AMERICAN}:
            updates["tree"] = base.tree.model_copy(update={"steps": budget})
            budget_type = ComputeBudgetType.TREE_STEPS
            approximate_memory = (budget + 1) * 8 * 2
        elif request.model is PricingModel.HESTON_MONTE_CARLO:
            heston_updates = (
                {"num_time_steps": budget, "seed": seed}
                if budget_type is ComputeBudgetType.TIME_STEPS
                else {"num_simulations": budget, "seed": seed}
            )
            heston_config = base.heston_monte_carlo.model_copy(update=heston_updates)
            updates["heston_monte_carlo"] = heston_config
            approximate_memory = heston_config.num_simulations * 8 * 6
        elif request.model is PricingModel.LONGSTAFF_SCHWARTZ:
            lsm_updates = (
                {"num_time_steps": budget, "seed": seed}
                if budget_type is ComputeBudgetType.TIME_STEPS
                else {"num_paths": budget, "seed": seed}
            )
            lsm_config = base.longstaff_schwartz.model_copy(update=lsm_updates)
            updates["longstaff_schwartz"] = lsm_config
            approximate_memory = (
                lsm_config.total_simulated_paths * (lsm_config.num_time_steps + 1) * 8
            )
        else:
            raise ValueError("selected model has no natural discretization or simulation budget")
        updates["config"] = base.config.model_copy(
            update={
                "selected_models": [request.model],
                "reference_model": None,
                "runtime_repetitions": 1,
                "deterministic_seed": seed,
                "include_greeks": False,
            }
        )
        result = compare_models(base.model_copy(update=updates))
        output = next(
            (item for item in result.outputs if item.status is ModelRunStatus.SUCCESS), None
        )
        if output is None or output.price is None or output.runtime_ms is None:
            message = result.outputs[0].message if result.outputs else "model was excluded"
            raise ValueError(f"compute-frontier model failed: {message}")
        absolute_error = abs(output.price - reference_price)
        points.append(
            ComputeFrontierPoint(
                budget=budget,
                budget_type=budget_type,
                price=output.price,
                reference_price=reference_price,
                absolute_error=absolute_error,
                relative_error=absolute_error
                / max(abs(reference_price), base.config.relative_price_floor),
                standard_error=output.standard_error,
                confidence_interval_width=(
                    output.confidence_interval_high - output.confidence_interval_low
                    if output.confidence_interval_high is not None
                    and output.confidence_interval_low is not None
                    else None
                ),
                runtime_ms=output.runtime_ms,
                approximate_memory_bytes=approximate_memory,
                seed=seed if not output.deterministic else None,
            )
        )
    return ComputeFrontierResult(
        model=request.model,
        reference_model=request.reference_model,
        natural_convergence_variable=budget_type,
        points=points,
        input_signature=deterministic_signature(request),
    )


def _single_model_price(request: ModelComparisonRequest, model: PricingModel) -> ModelOutput:
    comparison = compare_models(
        request.model_copy(
            update={
                "config": request.config.model_copy(
                    update={
                        "selected_models": [model],
                        "reference_model": None,
                        "runtime_repetitions": 1,
                        "include_greeks": False,
                        "include_iv": False,
                    }
                )
            }
        )
    )
    if not comparison.outputs:
        return ModelOutput(
            model=model,
            status=ModelRunStatus.EXCLUDED,
            deterministic=model_capabilities(model).deterministic,
            message=comparison.exclusions[0].reason if comparison.exclusions else "model excluded",
        )
    return comparison.outputs[0]


def analyze_model_stability(
    request: ModelComparisonRequest,
    perturbations: Sequence[StabilityPerturbation] | None = None,
) -> ModelStabilityResult:
    """Bump valuation inputs and preserve each model's own numerical method."""

    policies = list(
        perturbations
        or [
            StabilityPerturbation(input=StabilityInput.SPOT),
            StabilityPerturbation(input=StabilityInput.VOLATILITY_OR_V0),
            StabilityPerturbation(input=StabilityInput.RISK_FREE_RATE, absolute_bump=1e-4),
            StabilityPerturbation(input=StabilityInput.DIVIDEND_YIELD, absolute_bump=1e-4),
            StabilityPerturbation(input=StabilityInput.MATURITY),
        ]
    )
    points: list[StabilityPoint] = []
    for model in request.config.selected_models:
        base = _single_model_price(request, model)
        if base.price is None or base.status is not ModelRunStatus.SUCCESS:
            continue
        for policy in policies:
            contract_up = request.contract
            contract_down: OptionContract | None = request.contract
            market_up = request.market
            market_down = request.market
            parameters_up = _heston_parameters(request)
            parameters_down = parameters_up
            denominator = 0.0
            if policy.input is StabilityInput.SPOT:
                bump = policy.absolute_bump or request.contract.spot * policy.relative_bump
                contract_up = request.contract.model_copy(
                    update={"spot": request.contract.spot + bump}
                )
                contract_down = request.contract.model_copy(
                    update={"spot": request.contract.spot - bump}
                )
                denominator = 2.0 * bump
            elif policy.input is StabilityInput.VOLATILITY_OR_V0 and model in {
                PricingModel.HESTON_ANALYTICAL,
                PricingModel.HESTON_MONTE_CARLO,
            }:
                if parameters_up is None:
                    continue
                bump = policy.absolute_bump or max(parameters_up.v0 * policy.relative_bump, 1e-8)
                parameters_up = parameters_up.model_copy(update={"v0": parameters_up.v0 + bump})
                parameters_down = parameters_up.model_copy(
                    update={"v0": max(parameters_up.v0 - 2.0 * bump, 0.0)}
                )
                denominator = parameters_up.v0 - parameters_down.v0
            elif policy.input is StabilityInput.VOLATILITY_OR_V0:
                bump = policy.absolute_bump or max(
                    request.market.volatility * policy.relative_bump, 1e-8
                )
                market_up = request.market.model_copy(
                    update={"volatility": request.market.volatility + bump}
                )
                market_down = request.market.model_copy(
                    update={"volatility": max(request.market.volatility - bump, 0.0)}
                )
                denominator = market_up.volatility - market_down.volatility
            elif policy.input is StabilityInput.RISK_FREE_RATE:
                bump = policy.absolute_bump or max(
                    abs(request.market.risk_free_rate) * policy.relative_bump, 1e-4
                )
                market_up = request.market.model_copy(
                    update={"risk_free_rate": request.market.risk_free_rate + bump}
                )
                market_down = request.market.model_copy(
                    update={"risk_free_rate": request.market.risk_free_rate - bump}
                )
                denominator = 2.0 * bump
            elif policy.input is StabilityInput.DIVIDEND_YIELD:
                bump = policy.absolute_bump or max(
                    abs(request.market.dividend_yield) * policy.relative_bump, 1e-4
                )
                market_up = request.market.model_copy(
                    update={"dividend_yield": request.market.dividend_yield + bump}
                )
                market_down = request.market.model_copy(
                    update={"dividend_yield": request.market.dividend_yield - bump}
                )
                denominator = 2.0 * bump
            else:
                bump = policy.absolute_bump or max(
                    request.contract.maturity_years * policy.relative_bump, 1e-6
                )
                contract_up = request.contract.model_copy(
                    update={"maturity_years": request.contract.maturity_years + bump}
                )
                if request.contract.maturity_years > bump:
                    contract_down = request.contract.model_copy(
                        update={"maturity_years": request.contract.maturity_years - bump}
                    )
                    denominator = 2.0 * bump
                else:
                    contract_down = None
                    denominator = bump
            up_request = request.model_copy(
                update={
                    "contract": contract_up,
                    "market": market_up,
                    "heston_parameters": parameters_up,
                }
            )
            up = _single_model_price(up_request, model)
            down = None
            if contract_down is not None:
                down_request = request.model_copy(
                    update={
                        "contract": contract_down,
                        "market": market_down,
                        "heston_parameters": parameters_down,
                    }
                )
                down = _single_model_price(down_request, model)
            if (
                up.price is None
                or up.status is not ModelRunStatus.SUCCESS
                or (down is not None and down.price is None)
            ):
                points.append(
                    StabilityPoint(
                        model=model,
                        input=policy.input,
                        base_price=base.price,
                        downward_price=None,
                        upward_price=base.price,
                        downward_change=None,
                        upward_change=0.0,
                        central_sensitivity=None,
                        status=ModelRunStatus.FAILED,
                        message=(
                            up.message
                            if up.status is ModelRunStatus.FAILED
                            else down.message
                            if down is not None
                            else "perturbation failed"
                        ),
                    )
                )
                continue
            assert up.price is not None
            down_price = down.price if down is not None else None
            sensitivity = (
                (up.price - down_price) / denominator
                if down_price is not None and denominator > 0.0
                else (up.price - base.price) / denominator
            )
            points.append(
                StabilityPoint(
                    model=model,
                    input=policy.input,
                    base_price=base.price,
                    downward_price=down_price,
                    upward_price=up.price,
                    downward_change=(down_price - base.price if down_price is not None else None),
                    upward_change=up.price - base.price,
                    central_sensitivity=sensitivity,
                    status=ModelRunStatus.SUCCESS,
                )
            )
    return ModelStabilityResult(
        points=points,
        input_signature=deterministic_signature(
            {
                "request": request.model_dump(mode="json"),
                "perturbations": [item.model_dump(mode="json") for item in policies],
            }
        ),
    )


def heston_calibration_ensemble(
    contract: OptionContract,
    market: MarketEnvironment,
    parameter_sets: Sequence[HestonParameters],
    *,
    provenance: Sequence[str] | None = None,
) -> CalibrationEnsembleResult:
    """Propagate distinct valid calibration solutions into prices and equivalent IVs."""

    labels = list(
        provenance or [f"ensemble_member_{index}" for index in range(len(parameter_sets))]
    )
    if len(labels) != len(parameter_sets):
        raise ValueError("ensemble provenance must match parameter-set count")
    members = []
    for index, (parameters, label) in enumerate(zip(parameter_sets, labels, strict=True)):
        price = heston_analytical_price(contract, market, parameters).price
        members.append(
            CalibrationEnsembleMember(
                index=index,
                parameters=parameters,
                price=price,
                implied_volatility=_equivalent_iv(contract, market, price),
                provenance=label,
            )
        )
    return CalibrationEnsembleResult(
        members=members,
        price_dispersion=scalar_dispersion([member.price for member in members]),
        implied_volatility_dispersion=scalar_dispersion(
            [
                member.implied_volatility
                for member in members
                if member.implied_volatility is not None
            ]
        ),
    )


def propagate_heston_parameter_uncertainty(
    contract: OptionContract,
    market: MarketEnvironment,
    parameters: HestonParameters,
    covariance: Sequence[Sequence[float]],
) -> ParameterUncertaintyResult:
    """Apply the local first-order delta method to Phase 14 covariance."""

    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.shape != (5, 5) or not np.all(np.isfinite(matrix)):
        raise ValueError("local Heston covariance must be a finite 5x5 matrix")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise ValueError("local Heston covariance must be symmetric")
    if float(np.min(np.linalg.eigvalsh(matrix))) < -1e-10:
        raise ValueError("local Heston covariance must be positive semidefinite")
    names = ["v0", "kappa", "theta", "sigma_v", "rho"]
    vector = np.asarray([getattr(parameters, name) for name in names], dtype=np.float64)
    gradient = []
    for index, name in enumerate(names):
        bump = max(abs(vector[index]) * 1e-4, 1e-6)
        lower = vector.copy()
        upper = vector.copy()
        upper[index] += bump
        lower[index] -= bump
        if name == "rho":
            upper[index] = min(upper[index], 0.999999)
            lower[index] = max(lower[index], -0.999999)
        else:
            lower[index] = max(lower[index], 0.0)
        denominator = upper[index] - lower[index]
        upper_params = HestonParameters(**dict(zip(names, upper.tolist(), strict=True)))
        lower_params = HestonParameters(**dict(zip(names, lower.tolist(), strict=True)))
        upper_price = heston_analytical_price(contract, market, upper_params).price
        lower_price = heston_analytical_price(contract, market, lower_params).price
        gradient.append((upper_price - lower_price) / denominator)
    gradient_array = np.asarray(gradient)
    variance = float(gradient_array @ matrix @ gradient_array)
    variance = max(variance, 0.0)
    return ParameterUncertaintyResult(
        parameter_order=names,
        price_gradient=gradient,
        approximate_price_variance=variance,
        approximate_price_standard_deviation=math.sqrt(variance),
    )


def build_model_risk_report(request: ModelRiskRequest) -> ModelRiskReport:
    """Assemble valuation, fit, calibration, market, and numerical evidence."""

    comparison = compare_models(request.comparison)
    market_fit = evaluate_market_fit(request.market_fit) if request.market_fit is not None else None
    out_of_sample = (
        compare_models_out_of_sample(request.market_fit, request.holdout)
        if request.market_fit is not None and request.holdout is not None
        else None
    )
    parameters = _heston_parameters(request.comparison)
    parameter_sets = list(request.calibration_ensemble)
    provenance = [f"provided_solution_{index}" for index in range(len(parameter_sets))]
    if request.calibration_stability is not None:
        parameter_sets.extend(run.parameters for run in request.calibration_stability.runs)
        provenance.extend(
            f"phase14_stability_run_{run.run_index}" for run in request.calibration_stability.runs
        )
    calibration_stability = None
    if parameter_sets:
        ensemble = heston_calibration_ensemble(
            request.comparison.contract,
            request.comparison.market,
            parameter_sets,
            provenance=provenance,
        )
        calibration_stability = CalibrationStabilityValuation(
            parameter_standard_deviations=(
                request.calibration_stability.parameter_standard_deviations
                if request.calibration_stability is not None
                else {}
            ),
            price_dispersion=ensemble.price_dispersion,
            implied_volatility_dispersion=ensemble.implied_volatility_dispersion,
            ensemble=ensemble,
        )
    parameter_uncertainty = None
    if request.local_parameter_covariance is not None and parameters is not None:
        parameter_uncertainty = propagate_heston_parameter_uncertainty(
            request.comparison.contract,
            request.comparison.market,
            parameters,
            request.local_parameter_covariance,
        )
    fit_by_model = (
        {item.model: item for item in market_fit.models} if market_fit is not None else {}
    )
    out_of_sample_by_model = (
        {item.model: item for item in out_of_sample.models} if out_of_sample is not None else {}
    )
    output_by_model = {item.model: item for item in comparison.outputs}
    complexity = []
    for model in request.comparison.config.selected_models:
        capability = model_capabilities(model)
        output = output_by_model.get(model)
        fit = fit_by_model.get(model)
        validation_fit = out_of_sample_by_model.get(model)
        complexity.append(
            ModelComplexityFit(
                model=model,
                state_variables=capability.state_variables,
                stochastic_factors=capability.stochastic_factors,
                calibrated_parameter_count=capability.calibrated_parameter_count,
                calibration_required=capability.requires_calibration,
                calibration_runtime_ms=(
                    output.calibration.calibration_runtime_ms
                    if output is not None and output.calibration is not None
                    else None
                ),
                pricing_runtime_ms=output.runtime_ms if output is not None else None,
                fit_rmse=fit.metrics.rmse if fit is not None else None,
                out_of_sample_rmse=(
                    validation_fit.validation_metrics.rmse if validation_fit is not None else None
                ),
            )
        )
    flags = list(comparison.flags)
    thresholds = request.comparison.config.thresholds
    if request.calibration_stability is not None:
        means = request.calibration_stability.parameter_means
        stds = request.calibration_stability.parameter_standard_deviations
        if any(
            stds.get(name, 0.0) / max(abs(value), 1e-12) > thresholds.max_calibration_parameter_cv
            for name, value in means.items()
        ):
            flags.append(ModelRiskFlag.CALIBRATION_UNSTABLE)
    spread_width_by_observation: dict[int, float] = {}
    outside = False
    if market_fit is not None:
        for model_fit in market_fit.models:
            for point in model_fit.residuals:
                if point.bid is not None and point.ask is not None:
                    spread_width_by_observation[point.observation_index] = point.ask - point.bid
                    outside = outside or bool(point.spread_distance and point.spread_distance > 0.0)
        if outside:
            flags.append(ModelRiskFlag.MODEL_OUTSIDE_BID_ASK)
    if out_of_sample is not None and any(
        item.validation_metrics.rmse is not None
        and item.validation_metrics.rmse > thresholds.max_out_of_sample_rmse
        for item in out_of_sample.models
    ):
        flags.append(ModelRiskFlag.POOR_OUT_OF_SAMPLE_FIT)
    sampling = {
        item.model.value: item.standard_error
        for item in comparison.outputs
        if item.standard_error is not None
    }
    numerical: dict[str, float] = {}
    spread_widths = list(spread_width_by_observation.values())
    for diagnostic in comparison.consistency_diagnostics:
        if diagnostic.name in {"BSM_VS_CRR", "HESTON_ANALYTICAL_VS_MONTE_CARLO"}:
            numerical[diagnostic.name] = diagnostic.absolute_difference
    uncertainty = UncertaintyDecomposition(
        monte_carlo_sampling=sampling,
        model_dispersion=comparison.price_dispersion,
        calibration_dispersion=(
            calibration_stability.price_dispersion if calibration_stability is not None else None
        ),
        market_spread={
            "observation_count": float(len(spread_widths)),
            "mean_width": float(np.mean(spread_widths)) if spread_widths else 0.0,
            "maximum_width": float(np.max(spread_widths)) if spread_widths else 0.0,
        },
        numerical_discretization=numerical,
    )
    return ModelRiskReport(
        comparison=comparison,
        market_fit=market_fit,
        out_of_sample=out_of_sample,
        calibration_stability=calibration_stability,
        parameter_uncertainty=parameter_uncertainty,
        uncertainty=uncertainty,
        complexity_vs_fit=complexity,
        flags=list(dict.fromkeys(flags)),
        warnings=[
            "The reference model is a comparison convention, not an assertion of truth.",
            "Model-price range is descriptive and has no probability coverage interpretation.",
        ],
    )


def _consistency_check(
    name: str,
    left: ModelOutput,
    right: ModelOutput,
    tolerance: float,
) -> ConsistencyCheck:
    if left.price is None or right.price is None:
        return ConsistencyCheck(
            name=name,
            left_model=left.model,
            right_model=right.model,
            left_price=left.price,
            right_price=right.price,
            absolute_difference=None,
            tolerance=tolerance,
            passed=False,
            message="one or both models did not produce a price",
        )
    difference = abs(left.price - right.price)
    statistical_tolerance = (
        4.0 * left.standard_error
        if left.standard_error is not None
        else 4.0 * right.standard_error
        if right.standard_error is not None
        else 0.0
    )
    effective_tolerance = max(tolerance, statistical_tolerance)
    stochastic_se = left.standard_error or right.standard_error
    return ConsistencyCheck(
        name=name,
        left_model=left.model,
        right_model=right.model,
        left_price=left.price,
        right_price=right.price,
        absolute_difference=difference,
        tolerance=effective_tolerance,
        passed=difference <= effective_tolerance,
        z_score=(
            (left.price - right.price) / stochastic_se
            if stochastic_se is not None and stochastic_se > 0.0
            else None
        ),
        message="within configured/statistical tolerance"
        if difference <= effective_tolerance
        else "theoretically comparable engines differ beyond tolerance",
    )


def run_consistency_suite(request: ConsistencyRequest) -> ConsistencyResult:
    """Cross-check engines in regimes with known equivalence or convergence."""

    if request.contract.exercise_style is ExerciseStyle.AMERICAN:
        comparison_request = ModelComparisonRequest(
            contract=request.contract,
            market=request.market,
            config={
                "selected_models": [PricingModel.CRR_AMERICAN, PricingModel.LONGSTAFF_SCHWARTZ],
                "reference_model": PricingModel.CRR_AMERICAN,
                "deterministic_seed": request.seed,
            },
            tree=request.tree,
            longstaff_schwartz={
                "num_paths": request.simulation.num_simulations,
                "num_time_steps": min(request.tree.steps, 100),
                "seed": request.seed,
            },
        )
        result = compare_models(comparison_request)
        by_model = {item.model: item for item in result.outputs}
        checks = [
            _consistency_check(
                "AMERICAN_CRR_VS_LONGSTAFF_SCHWARTZ",
                by_model.get(
                    PricingModel.LONGSTAFF_SCHWARTZ,
                    ModelOutput(
                        model=PricingModel.LONGSTAFF_SCHWARTZ,
                        status=ModelRunStatus.FAILED,
                        deterministic=False,
                    ),
                ),
                by_model.get(
                    PricingModel.CRR_AMERICAN,
                    ModelOutput(
                        model=PricingModel.CRR_AMERICAN,
                        status=ModelRunStatus.FAILED,
                        deterministic=True,
                    ),
                ),
                request.tolerance,
            )
        ]
    else:
        variance = request.market.volatility * request.market.volatility
        deterministic_heston = HestonParameters(
            v0=variance, kappa=1.0, theta=variance, sigma_v=0.0, rho=0.0
        )
        comparison_request = ModelComparisonRequest(
            contract=request.contract,
            market=request.market,
            config={
                "selected_models": [
                    PricingModel.BLACK_SCHOLES,
                    PricingModel.MONTE_CARLO_GBM,
                    PricingModel.CRR_BINOMIAL,
                    PricingModel.HESTON_ANALYTICAL,
                ],
                "reference_model": PricingModel.BLACK_SCHOLES,
                "deterministic_seed": request.seed,
            },
            heston_parameters=deterministic_heston,
            gbm_monte_carlo=request.simulation.model_copy(update={"seed": request.seed}),
            tree=request.tree,
            heston_integration=request.heston_integration,
        )
        result = compare_models(comparison_request)
        by_model = {item.model: item for item in result.outputs}
        bsm = by_model[PricingModel.BLACK_SCHOLES]
        checks = [
            _consistency_check(
                "BSM_VS_GBM_MONTE_CARLO",
                by_model[PricingModel.MONTE_CARLO_GBM],
                bsm,
                request.tolerance,
            ),
            _consistency_check(
                "BSM_VS_CRR", by_model[PricingModel.CRR_BINOMIAL], bsm, request.tolerance
            ),
            _consistency_check(
                "BSM_VS_HESTON_DETERMINISTIC_VARIANCE",
                by_model[PricingModel.HESTON_ANALYTICAL],
                bsm,
                request.tolerance,
            ),
        ]
        from backend.app.models.heston import HestonSimulationScheme

        analytical = by_model[PricingModel.HESTON_ANALYTICAL]
        heston_seeds = spawn_child_seeds(request.seed, 2)
        for scheme, seed in zip(HestonSimulationScheme, heston_seeds, strict=True):
            config = request.heston_monte_carlo.model_copy(update={"scheme": scheme, "seed": seed})
            try:
                mc = heston_monte_carlo_price(
                    request.contract, request.market, deterministic_heston, config
                )
                output = ModelOutput(
                    model=PricingModel.HESTON_MONTE_CARLO,
                    status=ModelRunStatus.SUCCESS,
                    price=mc.price,
                    runtime_ms=mc.runtime_ms,
                    deterministic=False,
                    standard_error=mc.standard_error,
                )
            except Exception as exc:
                output = ModelOutput(
                    model=PricingModel.HESTON_MONTE_CARLO,
                    status=ModelRunStatus.FAILED,
                    deterministic=False,
                    message=str(exc),
                )
            checks.append(
                _consistency_check(
                    f"HESTON_ANALYTICAL_VS_{scheme.value}", output, analytical, request.tolerance
                )
            )
    flags = (
        []
        if all(check.passed for check in checks)
        else [ModelRiskFlag.NUMERICAL_CONSISTENCY_WARNING]
    )
    return ConsistencyResult(
        checks=checks,
        flags=flags,
        input_signature=deterministic_signature(request),
    )
