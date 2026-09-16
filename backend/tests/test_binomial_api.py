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


def request_body(*, steps: int = 200, option_type: str = "CALL") -> dict[str, object]:
    return {
        "contract": {
            "option_type": option_type,
            "exercise_style": "EUROPEAN",
            "spot": 100.0,
            "strike": 100.0,
            "maturity_years": 1.0,
        },
        "market": {
            "risk_free_rate": 0.05,
            "volatility": 0.2,
            "dividend_yield": 0.01,
        },
        "tree": {"steps": steps, "model": "CRR"},
    }


def test_binomial_price_endpoint(client: TestClient) -> None:
    response = client.post("/api/pricing/binomial", json=request_body())
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "CRR"
    assert body["steps"] == 200
    assert abs(body["signed_error"]) < 0.02
    assert body["delta"] is not None
    assert body["gamma"] is not None
    assert body["theta"] is not None


def test_binomial_comparison_endpoint(client: TestClient) -> None:
    response = client.post("/api/analytics/binomial/compare", json=request_body(steps=500))
    assert response.status_code == 200
    body = response.json()
    assert body["absolute_error"] < 0.01
    assert body["delta"]["absolute_error"] < 0.001
    assert body["gamma"]["absolute_error"] < 0.0001


def test_binomial_convergence_endpoint(client: TestClient) -> None:
    request = request_body()
    request.pop("tree")
    request["step_counts"] = [9, 10, 50, 100]
    request["model"] = "CRR"
    response = client.post("/api/analytics/binomial/convergence", json=request)
    assert response.status_code == 200
    body = response.json()
    assert [point["steps"] for point in body["points"]] == [9, 10, 50, 100]
    assert len(body["odd_even_diagnostics"]) == 1


def test_binomial_api_step_and_point_limits(client: TestClient) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(
        max_binomial_steps=100,
        max_binomial_convergence_points=2,
    )
    too_many_steps = client.post("/api/pricing/binomial", json=request_body(steps=101))
    request = request_body()
    request.pop("tree")
    request["step_counts"] = [10, 20, 30]
    too_many_points = client.post("/api/analytics/binomial/convergence", json=request)
    assert too_many_steps.status_code == 422
    assert "API limit" in too_many_steps.json()["detail"]
    assert too_many_points.status_code == 422
    assert "API limit" in too_many_points.json()["detail"]


def test_invalid_model_is_rejected_and_american_exercise_is_accepted(
    client: TestClient,
) -> None:
    invalid_model = request_body()
    invalid_model["tree"] = {"steps": 10, "model": "TIAN"}
    model_response = client.post("/api/pricing/binomial", json=invalid_model)
    american = request_body()
    contract = american["contract"]
    assert isinstance(contract, dict)
    contract["exercise_style"] = "AMERICAN"
    exercise_response = client.post("/api/pricing/binomial", json=american)
    assert model_response.status_code == 422
    assert exercise_response.status_code == 200
    assert exercise_response.json()["exercise_style"] == "AMERICAN"


def test_invalid_crr_probability_returns_clear_422(client: TestClient) -> None:
    request = request_body(steps=1)
    market = request["market"]
    assert isinstance(market, dict)
    market["risk_free_rate"] = 0.5
    market["volatility"] = 0.01
    response = client.post("/api/pricing/binomial", json=request)
    assert response.status_code == 422
    assert "risk-neutral probability" in response.json()["detail"]
