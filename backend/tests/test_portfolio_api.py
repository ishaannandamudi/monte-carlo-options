"""FastAPI coverage and resource bounds for Phase 16 portfolio routes."""

from fastapi.testclient import TestClient

from backend.app.core.config import Settings, get_settings
from backend.app.main import app
from backend.app.models.portfolio import (
    MarketScenario,
    Portfolio,
    PortfolioModelRiskRequest,
    PortfolioPnLExplainRequest,
    PortfolioPricingModel,
    PortfolioRequest,
    PortfolioScenarioRequest,
    PortfolioStressRequest,
    ScenarioAxis,
    ScenarioGridAxis,
    ScenarioGridRequest,
)
from backend.app.models.simulation import SimulationConfig
from backend.tests.portfolio_test_helpers import vanilla

client = TestClient(app)


def _payload(model: object) -> dict[str, object]:
    return model.model_dump(mode="json")  # type: ignore[attr-defined,no-any-return]


def test_all_portfolio_endpoints() -> None:
    portfolio = Portfolio(positions=[vanilla()])
    requests = {
        "/api/portfolio/value": PortfolioRequest(portfolio=portfolio),
        "/api/portfolio/greeks": PortfolioRequest(portfolio=portfolio),
        "/api/portfolio/scenario": PortfolioScenarioRequest(
            portfolio=portfolio, scenario=MarketScenario(spot_relative_shift=-0.05)
        ),
        "/api/portfolio/scenario-grid": ScenarioGridRequest(
            portfolio=portfolio,
            axes=[ScenarioGridAxis(axis=ScenarioAxis.SPOT_RELATIVE, values=[-0.05, 0.05])],
        ),
        "/api/portfolio/stress": PortfolioStressRequest(
            portfolio=portfolio,
            scenarios=[MarketScenario(name="stress", volatility_absolute_shift=0.05)],
        ),
        "/api/portfolio/pnl-explain": PortfolioPnLExplainRequest(
            portfolio=portfolio, scenario=MarketScenario(rate_absolute_shift=0.001)
        ),
        "/api/portfolio/model-risk": PortfolioModelRiskRequest(
            portfolio=portfolio,
            reference_model=PortfolioPricingModel.BLACK_SCHOLES,
            alternate_models=[PortfolioPricingModel.CRR_BINOMIAL],
        ),
    }
    for path, request in requests.items():
        response = client.post(path, json=_payload(request))
        assert response.status_code == 200, (path, response.text)


def test_position_and_grid_limits_reject_before_work() -> None:
    settings = Settings(
        max_portfolio_positions=1,
        max_portfolio_scenario_points=2,
        max_portfolio_grid_points=2,
    )
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        too_many_positions = PortfolioRequest(
            portfolio=Portfolio(positions=[vanilla("one"), vanilla("two")])
        )
        response = client.post("/api/portfolio/value", json=_payload(too_many_positions))
        assert response.status_code == 422
        grid = ScenarioGridRequest(
            portfolio=Portfolio(positions=[vanilla()]),
            axes=[ScenarioGridAxis(axis=ScenarioAxis.SPOT_RELATIVE, values=[-0.1, 0.0, 0.1])],
        )
        response = client.post("/api/portfolio/scenario-grid", json=_payload(grid))
        assert response.status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_stress_and_model_risk_limits_reject_before_work() -> None:
    settings = Settings(
        max_portfolio_stress_scenarios=1,
        max_portfolio_model_risk_models=1,
    )
    app.dependency_overrides[get_settings] = lambda: settings
    portfolio = Portfolio(positions=[vanilla()])
    try:
        stress = PortfolioStressRequest(
            portfolio=portfolio,
            scenarios=[
                MarketScenario(name="one", spot_relative_shift=-0.1),
                MarketScenario(name="two", volatility_absolute_shift=0.1),
            ],
        )
        assert client.post("/api/portfolio/stress", json=_payload(stress)).status_code == 422
        model_risk = PortfolioModelRiskRequest(
            portfolio=portfolio,
            reference_model=PortfolioPricingModel.BLACK_SCHOLES,
            alternate_models=[
                PortfolioPricingModel.CRR_BINOMIAL,
                PortfolioPricingModel.HESTON_ANALYTICAL,
            ],
        )
        assert (
            client.post("/api/portfolio/model-risk", json=_payload(model_risk)).status_code == 422
        )
    finally:
        app.dependency_overrides.clear()


def test_position_level_api_failure_returns_partial_result() -> None:
    invalid = vanilla(model=PortfolioPricingModel.CRR_AMERICAN)
    request = PortfolioRequest(portfolio=Portfolio(positions=[vanilla("good"), invalid]))
    response = client.post("/api/portfolio/value", json=_payload(request))
    assert response.status_code == 200
    assert response.json()["completeness"] == "PARTIAL"
    assert response.json()["failed_position_count"] == 1


def test_aggregate_monte_carlo_work_is_bounded_across_positions() -> None:
    settings = Settings(max_api_simulations=1_000)
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        positions = [
            vanilla(f"mc-{index}", model=PortfolioPricingModel.MONTE_CARLO_GBM).model_copy(
                update={"simulation": SimulationConfig(num_simulations=600)}
            )
            for index in range(2)
        ]
        request = PortfolioRequest(portfolio=Portfolio(positions=positions))
        response = client.post("/api/portfolio/value", json=_payload(request))
        assert response.status_code == 422
        assert "aggregate" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()
