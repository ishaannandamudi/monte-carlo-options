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


def request_body(
    *, option_type: str = "PUT", steps: int = 200, style: str = "AMERICAN"
) -> dict[str, object]:
    return {
        "contract": {
            "option_type": option_type,
            "exercise_style": style,
            "spot": 100.0,
            "strike": 100.0,
            "maturity_years": 1.0,
        },
        "market": {
            "risk_free_rate": 0.05,
            "volatility": 0.2,
            "dividend_yield": 0.0,
        },
        "tree": {"steps": steps, "model": "CRR", "exercise_tolerance": 1e-12},
    }


@pytest.mark.parametrize("option_type", ["CALL", "PUT"])
def test_american_binomial_pricing_serialization(client: TestClient, option_type: str) -> None:
    response = client.post("/api/pricing/binomial", json=request_body(option_type=option_type))
    assert response.status_code == 200
    body = response.json()
    assert body["exercise_style"] == "AMERICAN"
    assert body["price"] >= body["european_tree_price"]
    assert "early_exercise_premium" in body
    assert "boundary_points" not in body


def test_american_comparison_endpoint(client: TestClient) -> None:
    response = client.post("/api/analytics/binomial/compare", json=request_body(steps=300))
    assert response.status_code == 200
    body = response.json()
    assert body["american"]["exercise_style"] == "AMERICAN"
    assert body["european"]["price"] == pytest.approx(body["american"]["european_tree_price"])
    assert body["early_exercise_premium"] > 0.0
    assert body["delta"]["difference"] is not None


def test_exercise_boundary_endpoint_and_european_rejection(client: TestClient) -> None:
    response = client.post(
        "/api/analytics/binomial/exercise-boundary", json=request_body(steps=200)
    )
    assert response.status_code == 200
    body = response.json()
    assert body["points"]
    assert body["points"][0]["exercise_region_direction"] == "LOW_STOCK"
    assert all(point["step"] < 200 for point in body["points"])

    rejected = client.post(
        "/api/analytics/binomial/exercise-boundary",
        json=request_body(style="EUROPEAN"),
    )
    assert rejected.status_code == 422
    assert "AMERICAN" in rejected.json()["detail"]


def test_american_convergence_endpoint(client: TestClient) -> None:
    request = request_body()
    request.pop("tree")
    request["step_counts"] = [25, 50, 100]
    response = client.post("/api/analytics/binomial/convergence", json=request)
    assert response.status_code == 200
    body = response.json()
    assert [point["steps"] for point in body["points"]] == [25, 50, 100]
    assert all("american_price" in point for point in body["points"])
    assert "odd_even_diagnostics" not in body


def test_boundary_and_american_convergence_obey_resource_limits(
    client: TestClient,
) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(
        max_binomial_steps=100,
        max_binomial_convergence_points=2,
    )
    boundary = client.post(
        "/api/analytics/binomial/exercise-boundary", json=request_body(steps=101)
    )
    convergence = request_body()
    convergence.pop("tree")
    convergence["step_counts"] = [10, 20, 30]
    convergence_response = client.post("/api/analytics/binomial/convergence", json=convergence)
    assert boundary.status_code == 422
    assert convergence_response.status_code == 422


def test_non_binomial_pricing_endpoints_reject_american(client: TestClient) -> None:
    request = request_body()
    contract = request["contract"]
    market = request["market"]
    assert isinstance(contract, dict)
    assert isinstance(market, dict)
    black_scholes = client.post(
        "/api/pricing/black-scholes",
        json={"contract": contract, "market": market},
    )
    monte_carlo = client.post(
        "/api/pricing/monte-carlo",
        json={
            "contract": contract,
            "market": market,
            "simulation": {
                "num_simulations": 100,
                "seed": 42,
                "variance_reduction": "NONE",
                "confidence_level": 0.95,
            },
        },
    )
    assert black_scholes.status_code == 422
    assert monte_carlo.status_code == 422
    assert "EUROPEAN" in black_scholes.json()["detail"]
    assert "EUROPEAN" in monte_carlo.json()["detail"]


def test_unknown_exercise_style_is_rejected(client: TestClient) -> None:
    request = request_body()
    contract = request["contract"]
    assert isinstance(contract, dict)
    contract["exercise_style"] = "BERMUDAN"
    response = client.post("/api/pricing/binomial", json=request)
    assert response.status_code == 422
