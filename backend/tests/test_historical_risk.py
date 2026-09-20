"""Historical replay, Greek approximation, policies, and ES contribution tests."""

from datetime import UTC, datetime, timedelta

import pytest

from backend.app.core.exceptions import PortfolioRiskError
from backend.app.models.portfolio import MarketScenario, PortfolioScenarioRequest
from backend.app.models.risk import (
    HistoricalRiskDataset,
    HistoricalRiskFactorObservation,
    HistoricalRiskRequest,
    HistoricalRiskScenario,
    MissingFactorPolicy,
    PortfolioRiskConfig,
    RiskFactorMapping,
    RiskFactorType,
    RiskMethod,
    RiskValuationMode,
)
from backend.app.portfolio.engine import run_scenario
from backend.app.risk.engine import calculate_historical_risk
from backend.tests.risk_test_helpers import history, single_portfolio, spot_mapping


def _request(
    values: list[float],
    mode: RiskValuationMode = RiskValuationMode.DELTA_GAMMA_DIAGONAL,
    **config: object,
) -> HistoricalRiskRequest:
    return HistoricalRiskRequest(
        portfolio=single_portfolio(),
        factor_mappings=spot_mapping(),
        dataset=history(values),
        config=PortfolioRiskConfig(
            method=RiskMethod.HISTORICAL,
            confidence_level=0.8,
            valuation_mode=mode,
            **config,
        ),
    )


def test_historical_delta_gamma_distribution_matches_manual_factor_formula() -> None:
    request = _request([-0.10, -0.05, 0.0, 0.03, 0.08])
    result = calculate_historical_risk(request)
    factor = result.contributions.factors[0]  # type: ignore[union-attr]
    exposure = factor.exposure
    gamma = factor.gamma_exposure
    expected_pnl = [exposure * x + 0.5 * gamma * x * x for x in [-0.10, -0.05, 0, 0.03, 0.08]]
    assert result.distribution_summary.pnl_mean == pytest.approx(sum(expected_pnl) / 5)
    assert result.scenario_count == 5


def test_historical_full_revaluation_matches_phase16_scenario() -> None:
    request = _request([-0.02, 0.0, 0.02], RiskValuationMode.FULL_REVALUATION)
    result = calculate_historical_risk(request)
    direct = run_scenario(
        PortfolioScenarioRequest(
            portfolio=request.portfolio,
            scenario=MarketScenario(per_underlying_spot_relative_shifts={"SPY": -0.02}),
        )
    )
    first = next(point for point in result.worst_scenarios if point.scenario_index == 0)
    assert first.pnl == pytest.approx(direct.full_repricing_pnl)


def test_small_shock_full_revaluation_and_delta_gamma_are_close() -> None:
    full = calculate_historical_risk(
        _request([-0.001, 0.0, 0.001], RiskValuationMode.FULL_REVALUATION)
    )
    approximate = calculate_historical_risk(_request([-0.001, 0.0, 0.001]))
    assert full.value_at_risk == pytest.approx(approximate.value_at_risk, abs=1e-5)


def test_es_contributions_and_groups_sum_to_es() -> None:
    result = calculate_historical_risk(_request([-0.1, -0.05, 0.0, 0.05, 0.1]))
    contributions = result.contributions
    assert contributions is not None
    assert contributions.expected_shortfall_contribution_sum == pytest.approx(
        result.expected_shortfall
    )
    assert sum(
        item.expected_shortfall_contribution or 0.0 for item in contributions.by_underlying
    ) == pytest.approx(result.expected_shortfall)
    assert sum(
        item.expected_shortfall_contribution or 0.0 for item in contributions.by_instrument_class
    ) == pytest.approx(result.expected_shortfall)


def test_incremental_historical_var_is_reported() -> None:
    result = calculate_historical_risk(
        _request([-0.1, -0.05, 0.0, 0.05], include_incremental_var=True)
    )
    position = result.contributions.positions[0]  # type: ignore[union-attr]
    assert position.incremental_var == pytest.approx(result.value_at_risk)


def _missing_dataset() -> HistoricalRiskDataset:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    return HistoricalRiskDataset(
        scenarios=[
            HistoricalRiskScenario(
                timestamp=start,
                observations=[HistoricalRiskFactorObservation(factor_id="SPY", value=-0.01)],
            ),
            HistoricalRiskScenario(
                timestamp=start + timedelta(days=1),
                observations=[HistoricalRiskFactorObservation(factor_id="VOL", value=0.01)],
            ),
            HistoricalRiskScenario(
                timestamp=start + timedelta(days=2),
                observations=[HistoricalRiskFactorObservation(factor_id="SPY", value=0.01)],
            ),
        ]
    )


def _missing_request(policy: MissingFactorPolicy) -> HistoricalRiskRequest:
    return HistoricalRiskRequest(
        portfolio=single_portfolio(),
        factor_mappings=[
            RiskFactorMapping(
                factor_id="SPY",
                factor_type=RiskFactorType.SPOT_RETURN,
                underlying_id="SPY",
            ),
            RiskFactorMapping(
                factor_id="VOL",
                factor_type=RiskFactorType.VOLATILITY_CHANGE,
                underlying_id="SPY",
            ),
        ],
        dataset=_missing_dataset(),
        config=PortfolioRiskConfig(
            method=RiskMethod.HISTORICAL,
            valuation_mode=RiskValuationMode.DELTA,
            missing_factor_policy=policy,
        ),
    )


def test_missing_factor_error_policy() -> None:
    with pytest.raises(PortfolioRiskError):
        calculate_historical_risk(_missing_request(MissingFactorPolicy.ERROR))


def test_missing_factor_zero_policy_keeps_scenarios() -> None:
    result = calculate_historical_risk(_missing_request(MissingFactorPolicy.ZERO_SHOCK))
    assert result.scenario_count == 3


def test_missing_factor_drop_policy_drops_incomplete_scenarios() -> None:
    # Every row has one missing factor, so the engine rejects an empty post-filter dataset.
    with pytest.raises(PortfolioRiskError, match="at least two historical scenarios"):
        calculate_historical_risk(_missing_request(MissingFactorPolicy.DROP_SCENARIO))


def test_zero_shocks_have_zero_var_when_time_passage_is_disabled() -> None:
    result = calculate_historical_risk(_request([0.0, 0.0, 0.0]))
    assert result.value_at_risk == 0.0
    assert result.expected_shortfall == 0.0


def test_historical_horizon_does_not_scale_observed_shocks() -> None:
    one = calculate_historical_risk(_request([-0.02, 0.02]))
    ten = calculate_historical_risk(
        _request(
            [-0.02, 0.02],
            horizon={"trading_days": 10, "trading_days_per_year": 252},
        )
    )
    assert ten.value_at_risk == one.value_at_risk
    assert any("without square-root scaling" in warning for warning in ten.warnings)
