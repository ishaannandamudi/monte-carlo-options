from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from backend.app import __version__
from backend.app.core.config import Settings
from backend.app.main import app


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


def test_version_endpoint_reports_service_package_version(client: TestClient) -> None:
    response = client.get("/version")

    assert response.status_code == 200
    assert response.json() == {
        "service": "monte-carlo-options",
        "version": __version__,
    }


def test_local_frontend_origin_passes_cors_preflight(client: TestClient) -> None:
    response = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_unconfigured_origin_is_not_allowed(client: TestClient) -> None:
    response = client.options(
        "/health",
        headers={
            "Origin": "https://untrusted.example",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert "access-control-allow-origin" not in response.headers


def test_cors_origins_load_from_comma_separated_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "CORS_ALLOWED_ORIGINS",
        "https://research.example, http://localhost:4173 ",
    )

    settings = Settings.from_environment()

    assert settings.cors_allowed_origins == (
        "https://research.example",
        "http://localhost:4173",
    )


def test_phase22_frontend_routes_match_openapi_contract() -> None:
    required_methods = {
        "/api/pricing/black-scholes": "post",
        "/api/pricing/monte-carlo": "post",
        "/api/pricing/binomial": "post",
        "/api/pricing/heston/analytical": "post",
        "/api/pricing/heston/monte-carlo": "post",
        "/api/pricing/merton/analytical": "post",
        "/api/pricing/merton/monte-carlo": "post",
        "/api/greeks/black-scholes": "post",
        "/api/volatility/implied": "post",
        "/api/analytics/models/compare": "post",
        "/api/analytics/models/market-fit": "post",
        "/api/analytics/models/risk": "post",
        "/api/analytics/volatility/smile": "post",
        "/api/analytics/volatility/surface/build": "post",
        "/api/analytics/heston/smile": "post",
        "/api/analytics/heston/sensitivity": "post",
        "/api/calibration/heston": "post",
        "/api/analytics/heston/calibration/diagnostics": "post",
        "/api/analytics/heston/calibration/fitted-surface": "post",
        "/api/analytics/merton/compare": "post",
        "/api/analytics/merton/greeks": "post",
        "/api/analytics/merton/smile": "post",
        "/api/analytics/merton/jump-diagnostics": "post",
        "/api/portfolio/value": "post",
        "/api/portfolio/greeks": "post",
        "/api/portfolio/scenario": "post",
        "/api/portfolio/scenario-grid": "post",
        "/api/portfolio/stress": "post",
        "/api/portfolio/pnl-explain": "post",
        "/api/risk/historical": "post",
        "/api/risk/parametric": "post",
        "/api/risk/monte-carlo": "post",
        "/api/risk/contributions": "post",
        "/api/risk/report": "post",
        "/api/risk/backtest": "post",
        "/api/regime/hmm/fit": "post",
        "/api/regime/hmm/decode": "post",
        "/api/regime/hmm/forecast": "post",
        "/api/regime/hmm/model-order": "post",
        "/api/regime/markov/diagnostics": "post",
        "/api/regime/risk": "post",
        "/api/pricing/qmc/european": "post",
        "/api/pricing/qmc/asian": "post",
        "/api/pricing/qmc/basket": "post",
        "/api/analytics/qmc/sampling": "post",
        "/api/analytics/qmc/compare": "post",
        "/api/analytics/qmc/convergence": "post",
        "/api/experiments": "get",
        "/api/experiments/run": "post",
        "/api/experiments/compare": "post",
        "/api/experiments/sweep": "post",
        "/api/experiments/benchmark": "post",
        "/api/experiments/{run_id}": "get",
        "/api/experiments/{run_id}/reproduce": "post",
    }
    paths = app.openapi()["paths"]

    for path, method in required_methods.items():
        assert path in paths
        assert method in paths[path]
