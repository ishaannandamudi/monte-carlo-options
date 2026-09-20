"""Backtesting, stress separation, API routes, and Phase 17 resource controls."""

import math
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings, get_settings
from backend.app.main import app
from backend.app.models.portfolio import MarketScenario
from backend.app.models.risk import (
    HistoricalRiskRequest,
    MonteCarloRiskRequest,
    ParametricRiskRequest,
    PortfolioRiskConfig,
    PortfolioRiskReportRequest,
    PortfolioRiskRequest,
    RiskMethod,
    RiskValuationMode,
    VaRBacktestObservation,
    VaRBacktestRequest,
)
from backend.app.risk.backtesting import backtest_var
from backend.app.risk.engine import calculate_risk_report
from backend.tests.risk_test_helpers import history, single_portfolio, spot_covariance, spot_mapping

client = TestClient(app)


def _payload(model: object) -> dict[str, object]:
    return model.model_dump(mode="json")  # type: ignore[attr-defined,no-any-return]


def _backtest(losses: list[float], value_at_risk: float = 1.0) -> VaRBacktestRequest:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    return VaRBacktestRequest(
        confidence_level=0.99,
        observations=[
            VaRBacktestObservation(
                date=start + timedelta(days=index),
                predicted_var=value_at_risk,
                realized_pnl=-loss,
            )
            for index, loss in enumerate(losses)
        ],
    )


def test_exception_is_strict_and_equality_does_not_count() -> None:
    result = backtest_var(_backtest([0.9, 1.0, 1.1]))
    assert result.exceptions == [False, False, True]
    assert result.exception_count == 1


@pytest.mark.parametrize("losses", [[0.0] * 20, [2.0] * 20])
def test_kupiec_is_finite_for_zero_and_all_exceptions(losses: list[float]) -> None:
    result = backtest_var(_backtest(losses))
    assert math.isfinite(result.kupiec_statistic)
    assert 0.0 <= result.kupiec_p_value <= 1.0


def test_christoffersen_transition_counts_and_conditional_coverage() -> None:
    result = backtest_var(_backtest([0.0, 2.0, 2.0, 0.0, 2.0]))
    assert (
        result.transitions_00,
        result.transitions_01,
        result.transitions_10,
        result.transitions_11,
    ) == (0, 2, 1, 1)
    assert math.isfinite(result.christoffersen_statistic)
    assert result.conditional_coverage_statistic == pytest.approx(
        result.kupiec_statistic + result.christoffersen_statistic
    )


def test_kupiec_and_christoffersen_match_fixed_reference_values() -> None:
    exceptions = {10, 40, 80, 130, 200}
    result = backtest_var(_backtest([2.0 if index in exceptions else 0.0 for index in range(250)]))
    assert result.kupiec_statistic == pytest.approx(1.956809788230622)
    assert result.kupiec_p_value == pytest.approx(0.1618549171960387)
    assert result.christoffersen_statistic == pytest.approx(0.20493237652149787)
    assert result.christoffersen_p_value == pytest.approx(0.6507686878924301)


def _requests() -> tuple[HistoricalRiskRequest, ParametricRiskRequest, MonteCarloRiskRequest]:
    portfolio = single_portfolio()
    mappings = spot_mapping()
    historical = HistoricalRiskRequest(
        portfolio=portfolio,
        factor_mappings=mappings,
        dataset=history([-0.02, -0.01, 0.0, 0.01, 0.02]),
        config=PortfolioRiskConfig(
            method=RiskMethod.HISTORICAL,
            valuation_mode=RiskValuationMode.DELTA_GAMMA_DIAGONAL,
        ),
    )
    parametric = ParametricRiskRequest(
        portfolio=portfolio,
        factor_mappings=mappings,
        covariance=spot_covariance(),
        config=PortfolioRiskConfig(
            method=RiskMethod.PARAMETRIC_NORMAL,
            valuation_mode=RiskValuationMode.DELTA,
        ),
    )
    monte_carlo = MonteCarloRiskRequest(
        portfolio=portfolio,
        factor_mappings=mappings,
        covariance=spot_covariance(),
        config=PortfolioRiskConfig(
            method=RiskMethod.MONTE_CARLO,
            valuation_mode=RiskValuationMode.DELTA,
            simulations=1_000,
        ),
    )
    return historical, parametric, monte_carlo


def test_dedicated_unified_contribution_and_backtest_endpoints() -> None:
    historical, parametric, monte_carlo = _requests()
    requests = {
        "/api/risk/historical": historical,
        "/api/risk/parametric": parametric,
        "/api/risk/monte-carlo": monte_carlo,
        "/api/risk/portfolio": PortfolioRiskRequest(
            portfolio=parametric.portfolio,
            factor_mappings=parametric.factor_mappings,
            config=parametric.config,
            covariance=parametric.covariance,
        ),
        "/api/risk/contributions": PortfolioRiskRequest(
            portfolio=historical.portfolio,
            factor_mappings=historical.factor_mappings,
            config=historical.config,
            historical_dataset=historical.dataset,
        ),
        "/api/risk/backtest": _backtest([0.0, 0.0, 2.0]),
    }
    for path, request in requests.items():
        response = client.post(path, json=_payload(request))
        assert response.status_code == 200, (path, response.text)


def test_stress_report_keeps_stress_outside_distribution() -> None:
    historical, _, _ = _requests()
    request = PortfolioRiskReportRequest(
        portfolio=historical.portfolio,
        factor_mappings=historical.factor_mappings,
        config=historical.config,
        historical_dataset=historical.dataset,
        stress_scenarios=[MarketScenario(name="crash", spot_relative_shift=-0.25)],
    )
    result = calculate_risk_report(request)
    assert result.scenario_count == len(historical.dataset.scenarios)
    assert len(result.stress_losses) == 1
    assert result.stress_losses[0].name == "crash"
    response = client.post("/api/risk/report", json=_payload(request))
    assert response.status_code == 200


def test_api_rejects_historical_mc_tail_and_scenario_position_limits() -> None:
    historical, _, monte_carlo = _requests()
    settings = Settings(
        max_risk_historical_scenarios=2,
        max_risk_monte_carlo_simulations=500,
        max_risk_tail_output=2,
        max_risk_scenario_position_evaluations=2,
    )
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        assert client.post("/api/risk/historical", json=_payload(historical)).status_code == 422
        assert client.post("/api/risk/monte-carlo", json=_payload(monte_carlo)).status_code == 422
        limited_tail = monte_carlo.model_copy(
            update={"config": monte_carlo.config.model_copy(update={"max_tail_scenarios": 3})}
        )
        assert client.post("/api/risk/monte-carlo", json=_payload(limited_tail)).status_code == 422
    finally:
        app.dependency_overrides.clear()

    full = historical.model_copy(
        update={
            "config": historical.config.model_copy(
                update={"valuation_mode": RiskValuationMode.FULL_REVALUATION}
            )
        }
    )
    app.dependency_overrides[get_settings] = lambda: Settings(
        max_risk_historical_scenarios=100,
        max_risk_scenario_position_evaluations=2,
    )
    try:
        response = client.post("/api/risk/historical", json=_payload(full))
        assert response.status_code == 422
        assert "scenario-position" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_api_rejects_backtest_and_contribution_limits() -> None:
    historical, _, _ = _requests()
    settings = Settings(max_risk_backtest_observations=2, max_risk_contribution_positions=0 + 1)
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        assert (
            client.post("/api/risk/backtest", json=_payload(_backtest([0, 0, 0]))).status_code
            == 422
        )
        # Make two distinct positions to trip the contribution limit.
        portfolio = historical.portfolio.model_copy(
            update={
                "positions": [
                    historical.portfolio.positions[0],
                    historical.portfolio.positions[0].model_copy(update={"position_id": "second"}),
                ]
            }
        )
        request = PortfolioRiskRequest(
            portfolio=portfolio,
            factor_mappings=historical.factor_mappings,
            config=historical.config,
            historical_dataset=historical.dataset,
        )
        assert client.post("/api/risk/contributions", json=_payload(request)).status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_api_validation_rejects_bad_confidence_insufficient_history_and_covariance() -> None:
    historical, parametric, _ = _requests()
    payload = _payload(historical)
    payload["config"]["confidence_level"] = 1.0  # type: ignore[index]
    assert client.post("/api/risk/historical", json=payload).status_code == 422
    payload = _payload(historical)
    payload["dataset"]["scenarios"] = payload["dataset"]["scenarios"][:1]  # type: ignore[index]
    assert client.post("/api/risk/historical", json=payload).status_code == 422
    payload = _payload(parametric)
    payload["covariance"]["covariance_matrix"] = [[1.0, 2.0], [2.0, 1.0]]  # type: ignore[index]
    assert client.post("/api/risk/parametric", json=payload).status_code == 422


def test_api_monte_carlo_reproducibility_excluding_runtime() -> None:
    _, _, request = _requests()
    first = client.post("/api/risk/monte-carlo", json=_payload(request)).json()
    second = client.post("/api/risk/monte-carlo", json=_payload(request)).json()
    for result in (first, second):
        result.pop("runtime_ms")
    assert first == second
