from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings, get_settings
from backend.app.main import app
from backend.app.volatility_surface.synthetic import (
    generate_synthetic_smile,
    generate_synthetic_surface,
)
from backend.tests.test_volatility_surface_market import market_snapshot


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def smile_payload() -> dict[str, object]:
    smile = generate_synthetic_smile(
        maturity_years=1.0,
        log_moneyness=[-0.1, 0.0, 0.1],
        base_volatility=0.2,
        skew=-0.1,
    )
    return smile.model_dump(mode="json")


def surface_payload() -> dict[str, object]:
    surface = generate_synthetic_surface(
        maturities=[0.5, 1.0],
        log_moneyness=[-0.1, 0.0, 0.1],
        base_volatility=0.2,
    )
    return surface.model_dump(mode="json")


def test_smile_build_and_query_endpoints(client: TestClient) -> None:
    smile = generate_synthetic_smile(
        maturity_years=1.0,
        log_moneyness=[-0.1, 0.0, 0.1],
        base_volatility=0.2,
    )
    build = client.post(
        "/api/analytics/volatility/smile/build",
        json={
            "points": [point.model_dump(mode="json") for point in smile.points],
            "underlying_symbol": "SYNTHETIC",
            "valuation_timestamp": smile.valuation_timestamp.isoformat(),
            "expiration": smile.expiration.isoformat(),
            "maturity_years": 1.0,
            "spot": 100.0,
            "risk_free_rate": 0.02,
            "dividend_yield": 0.0,
        },
    )
    assert build.status_code == 200
    assert build.json()["diagnostics"]["iv_points_accepted"] == 3
    query = client.post(
        "/api/analytics/volatility/smile/query",
        json={"smile": build.json()["smile"], "log_moneyness": 0.05},
    )
    assert query.status_code == 200
    assert query.json()["implied_volatility"] == pytest.approx(0.2)
    assert query.json()["interpolated"] is True


def test_surface_build_query_diagnostics_and_atm_endpoints(client: TestClient) -> None:
    source = generate_synthetic_surface(
        maturities=[0.5, 1.0],
        log_moneyness=[-0.1, 0.0, 0.1],
        base_volatility=0.2,
    )
    build = client.post(
        "/api/analytics/volatility/surface/build",
        json={"smiles": [smile.model_dump(mode="json") for smile in source.smiles]},
    )
    assert build.status_code == 200
    body = build.json()
    assert body["quality_summary"]["smile_count"] == 2
    surface = body["surface"]
    query = client.post(
        "/api/analytics/volatility/surface/query",
        json={"surface": surface, "maturity_years": 0.75, "log_moneyness": 0.0},
    )
    diagnostics = client.post(
        "/api/analytics/volatility/surface/diagnostics",
        json={"surface": surface},
    )
    term = client.post(
        "/api/analytics/volatility/surface/atm-term-structure",
        json={"surface": surface},
    )
    assert query.status_code == 200
    assert query.json()["implied_volatility"] == pytest.approx(0.2)
    assert diagnostics.status_code == 200
    assert diagnostics.json()["calendar_checks"] > 0
    assert term.status_code == 200
    assert len(term.json()["points"]) == 2


def test_surface_build_endpoint_accepts_normalized_phase5_chain(client: TestClient) -> None:
    response = client.post(
        "/api/analytics/volatility/surface/build",
        json={
            "option_chain": market_snapshot().model_dump(mode="json"),
            "risk_free_rate": 0.0,
            "dividend_yield": 0.0,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["diagnostics"]["total_quotes_considered"] == 12
    assert body["quality_summary"]["smile_count"] == 2


def test_smile_query_rejects_ambiguous_and_out_of_domain_requests(
    client: TestClient,
) -> None:
    ambiguous = client.post(
        "/api/analytics/volatility/smile/query",
        json={"smile": smile_payload(), "strike": 100.0, "log_moneyness": 0.0},
    )
    outside = client.post(
        "/api/analytics/volatility/smile/query",
        json={"smile": smile_payload(), "log_moneyness": 0.5},
    )
    assert ambiguous.status_code == 422
    assert "exactly one" in ambiguous.json()["detail"]
    assert outside.status_code == 422
    assert "outside smile domain" in outside.json()["detail"]


def test_surface_api_resource_limits(client: TestClient) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(
        max_smile_points=2,
        max_surface_smiles=1,
        max_diagnostic_grid_points=3,
    )
    smile_query = client.post(
        "/api/analytics/volatility/smile/query",
        json={"smile": smile_payload(), "log_moneyness": 0.0},
    )
    surface_query = client.post(
        "/api/analytics/volatility/surface/query",
        json={"surface": surface_payload(), "maturity_years": 0.75, "log_moneyness": 0.0},
    )
    assert smile_query.status_code == 422
    assert "API limit" in smile_query.json()["detail"]
    assert surface_query.status_code == 422
    assert "API limit" in surface_query.json()["detail"]


def test_diagnostic_grid_limit_is_enforced(client: TestClient) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(max_diagnostic_grid_points=5)
    response = client.post(
        "/api/analytics/volatility/surface/diagnostics",
        json={
            "surface": surface_payload(),
            "diagnostic_config": {"default_grid_size": 3},
        },
    )
    assert response.status_code == 422
    assert "diagnostic grid" in response.json()["detail"]
