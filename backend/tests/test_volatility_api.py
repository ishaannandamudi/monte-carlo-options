from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings, get_settings
from backend.app.main import app


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def contract(strike: float = 100.0) -> dict[str, object]:
    return {
        "option_type": "CALL",
        "exercise_style": "EUROPEAN",
        "spot": 100.0,
        "strike": strike,
        "maturity_years": 1.0,
    }


def observation(
    strike: float = 100.0,
    observed_price: float = 9.826297782739111,
) -> dict[str, object]:
    return {
        "contract": contract(strike),
        "risk_free_rate": 0.05,
        "dividend_yield": 0.01,
        "observed_price": observed_price,
    }


def test_single_implied_volatility_endpoint(client: TestClient) -> None:
    response = client.post(
        "/api/volatility/implied",
        json={**observation(), "solver_config": {"method": "BRENT"}},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["implied_volatility"] == pytest.approx(0.2, abs=1e-8)
    assert body["method_used"] == "BRENT"
    assert body["absolute_price_error"] <= 1e-10
    assert body["final_vega"] > 0.0


def test_single_endpoint_maps_quantitative_failures_to_422(client: TestClient) -> None:
    response = client.post(
        "/api/volatility/implied",
        json=observation(observed_price=101.0),
    )
    assert response.status_code == 422
    assert "no-arbitrage bounds" in response.json()["detail"]


def test_batch_endpoint_returns_partial_failures(client: TestClient) -> None:
    response = client.post(
        "/api/volatility/implied/batch",
        json={
            "observations": [observation(), observation(observed_price=101.0), observation()],
            "solver_config": {"method": "AUTO"},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success_count"] == 2
    assert body["failure_count"] == 1
    assert [item["success"] for item in body["items"]] == [True, False, True]


def test_smile_and_solver_comparison_endpoints(client: TestClient) -> None:
    smile = client.post(
        "/api/analytics/volatility/smile",
        json={
            "spot": 100.0,
            "maturity_years": 1.0,
            "risk_free_rate": 0.05,
            "dividend_yield": 0.01,
            "option_type": "CALL",
            "strikes": [100.0, 100.0],
            "option_prices": [9.826297782739111, 9.826297782739111],
        },
    )
    comparison = client.post(
        "/api/analytics/volatility/solver-comparison",
        json=observation(),
    )
    assert smile.status_code == 200
    assert [point["implied_volatility"] for point in smile.json()["points"]] == pytest.approx(
        [0.2, 0.2]
    )
    assert comparison.status_code == 200
    assert [point["method"] for point in comparison.json()["points"]] == [
        "BRENT",
        "BISECTION",
        "NEWTON",
    ]


def test_smile_length_validation(client: TestClient) -> None:
    response = client.post(
        "/api/analytics/volatility/smile",
        json={
            "spot": 100.0,
            "maturity_years": 1.0,
            "risk_free_rate": 0.05,
            "option_type": "PUT",
            "strikes": [90.0, 100.0],
            "option_prices": [1.0],
        },
    )
    assert response.status_code == 422


def test_batch_and_smile_resource_limits(client: TestClient) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(
        max_iv_batch_size=2,
        max_smile_points=2,
    )
    batch = client.post(
        "/api/volatility/implied/batch",
        json={"observations": [observation(), observation(), observation()]},
    )
    smile = client.post(
        "/api/analytics/volatility/smile",
        json={
            "spot": 100.0,
            "maturity_years": 1.0,
            "risk_free_rate": 0.05,
            "option_type": "CALL",
            "strikes": [90.0, 100.0, 110.0],
            "option_prices": [15.0, 9.826297782739111, 5.0],
        },
    )
    assert batch.status_code == 422
    assert smile.status_code == 422


def test_empty_batch_and_invalid_solver_are_rejected(client: TestClient) -> None:
    empty = client.post("/api/volatility/implied/batch", json={"observations": []})
    bad_solver = client.post(
        "/api/volatility/implied",
        json={**observation(), "solver_config": {"method": "SECANT"}},
    )
    assert empty.status_code == 422
    assert bad_solver.status_code == 422
