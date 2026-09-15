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


def contract() -> dict[str, object]:
    return {
        "option_type": "CALL",
        "exercise_style": "EUROPEAN",
        "spot": 100.0,
        "strike": 100.0,
        "maturity_years": 1.0,
    }


def market() -> dict[str, float]:
    return {"risk_free_rate": 0.05, "volatility": 0.2, "dividend_yield": 0.01}


def simulation(variance_reduction: str = "ANTITHETIC") -> dict[str, object]:
    return {
        "num_simulations": 10_001,
        "seed": 42,
        "variance_reduction": variance_reduction,
        "random_engine": "PCG64",
        "confidence_level": 0.95,
    }


def test_analytical_greeks_endpoint(client: TestClient) -> None:
    response = client.post(
        "/api/greeks/black-scholes",
        json={"contract": contract(), "market": market()},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["option_price"] == pytest.approx(9.826297782739111)
    assert body["delta"] == pytest.approx(0.6117631008098845)
    assert body["theta"] < 0.0
    assert body["dividend_rho"] < 0.0


def test_finite_difference_greeks_endpoint(client: TestClient) -> None:
    response = client.post(
        "/api/greeks/monte-carlo",
        json={
            "contract": contract(),
            "market": market(),
            "simulation": simulation("ANTITHETIC_CONTROL_VARIATE"),
            "greeks": ["DELTA", "GAMMA", "VEGA", "THETA", "RHO", "DIVIDEND_RHO"],
            "estimator_method": "FINITE_DIFFERENCE",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["method"] == "FINITE_DIFFERENCE"
    assert [item["greek"] for item in body["results"]] == [
        "DELTA",
        "GAMMA",
        "VEGA",
        "THETA",
        "RHO",
        "DIVIDEND_RHO",
    ]
    assert all(item["standard_error"] >= 0.0 for item in body["results"])
    assert all(item["num_simulations"] == 10_001 for item in body["results"])


def test_pathwise_greeks_endpoint(client: TestClient) -> None:
    response = client.post(
        "/api/greeks/monte-carlo",
        json={
            "contract": contract(),
            "market": market(),
            "simulation": simulation(),
            "greeks": ["DELTA", "VEGA"],
            "estimator_method": "PATHWISE",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["method"] == "PATHWISE"
    assert [item["method"] for item in body["results"]] == ["PATHWISE", "PATHWISE"]


@pytest.mark.parametrize(
    "payload_update",
    [
        {"greeks": ["GAMMA"], "estimator_method": "PATHWISE"},
        {
            "greeks": ["DELTA"],
            "estimator_method": "PATHWISE",
            "variance_reduction": "CONTROL_VARIATE",
        },
        {"greeks": ["DELTA"], "estimator_method": "ANALYTICAL"},
    ],
)
def test_monte_carlo_greeks_reject_unsupported_combinations(
    client: TestClient,
    payload_update: dict[str, object],
) -> None:
    variance_reduction = str(payload_update.get("variance_reduction", "ANTITHETIC"))
    response = client.post(
        "/api/greeks/monte-carlo",
        json={
            "contract": contract(),
            "market": market(),
            "simulation": simulation(variance_reduction),
            "greeks": payload_update["greeks"],
            "estimator_method": payload_update["estimator_method"],
        },
    )
    assert response.status_code == 422


def test_monte_carlo_greeks_reject_invalid_greek_and_bump(client: TestClient) -> None:
    invalid_greek = client.post(
        "/api/greeks/monte-carlo",
        json={
            "contract": contract(),
            "market": market(),
            "simulation": simulation(),
            "greeks": ["VANNA"],
            "estimator_method": "FINITE_DIFFERENCE",
        },
    )
    invalid_bump = client.post(
        "/api/greeks/monte-carlo",
        json={
            "contract": contract(),
            "market": market(),
            "simulation": simulation(),
            "greeks": ["DELTA"],
            "estimator_method": "FINITE_DIFFERENCE",
            "bump_config": {"spot_relative_bump": 0.0},
        },
    )
    assert invalid_greek.status_code == 422
    assert invalid_bump.status_code == 422


def test_greek_comparison_endpoint(client: TestClient) -> None:
    response = client.post(
        "/api/analytics/greeks/compare",
        json={
            "contract": contract(),
            "market": market(),
            "simulation": simulation(),
            "greek": "DELTA",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["greek"] == "DELTA"
    assert [item["method"] for item in body["estimates"]] == [
        "FINITE_DIFFERENCE",
        "PATHWISE",
    ]
    assert all(item["z_score"] is not None for item in body["estimates"])


def test_sensitivity_endpoint_and_grid_limit(client: TestClient) -> None:
    response = client.post(
        "/api/analytics/sensitivity",
        json={
            "contract": contract(),
            "market": market(),
            "parameter": "SPOT",
            "values": [90.0, 100.0, 110.0],
        },
    )
    assert response.status_code == 200
    assert response.json()["varied_parameter"] == "SPOT"
    assert len(response.json()["points"]) == 3

    app.dependency_overrides[get_settings] = lambda: Settings(max_sensitivity_points=2)
    limited = client.post(
        "/api/analytics/sensitivity",
        json={
            "contract": contract(),
            "market": market(),
            "parameter": "SPOT",
            "values": [90.0, 100.0, 110.0],
        },
    )
    assert limited.status_code == 422
    assert "API limit" in limited.json()["detail"]


def test_sensitivity_endpoint_rejects_invalid_grid_value(client: TestClient) -> None:
    response = client.post(
        "/api/analytics/sensitivity",
        json={
            "contract": contract(),
            "market": market(),
            "parameter": "VOLATILITY",
            "values": [-0.1, 0.2],
        },
    )
    assert response.status_code == 422
    assert "non-negative" in response.json()["detail"]


@pytest.mark.parametrize(
    "path",
    ["/api/greeks/monte-carlo", "/api/analytics/greeks/compare"],
)
def test_greek_endpoints_preserve_simulation_limit(client: TestClient, path: str) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(max_api_simulations=100)
    payload: dict[str, object] = {
        "contract": contract(),
        "market": market(),
        "simulation": simulation(),
        "greek": "DELTA",
    }
    if path == "/api/greeks/monte-carlo":
        payload.pop("greek")
        payload["greeks"] = ["DELTA"]
        payload["estimator_method"] = "FINITE_DIFFERENCE"
    response = client.post(path, json=payload)
    assert response.status_code == 422
    assert "API limit" in response.json()["detail"]
