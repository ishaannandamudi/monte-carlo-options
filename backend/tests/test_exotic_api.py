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


def asian_body(averaging_type: str = "ARITHMETIC") -> dict[str, object]:
    return {
        "contract": {
            "option_type": "CALL",
            "spot": 100,
            "strike": 100,
            "maturity_years": 1,
            "averaging_type": averaging_type,
            "averaging_schedule": {"monitoring_count": 4},
            "include_initial_spot_in_average": False,
        },
        "market": {"risk_free_rate": 0.05, "volatility": 0.2, "dividend_yield": 0.01},
        "simulation": {
            "num_simulations": 2_000,
            "seed": 42,
            "variance_reduction": "ANTITHETIC",
        },
    }


def digital_body(method: str = "MONTE_CARLO") -> dict[str, object]:
    return {
        "contract": {
            "option_type": "CALL",
            "spot": 100,
            "strike": 100,
            "maturity_years": 1,
            "digital_type": "CASH_OR_NOTHING",
            "cash_payout": 10,
        },
        "market": {"risk_free_rate": 0.05, "volatility": 0.2, "dividend_yield": 0.01},
        "method": method,
        "simulation": {
            "num_simulations": 5_000,
            "seed": 42,
            "variance_reduction": "ANTITHETIC",
        },
    }


def test_asian_arithmetic_control_and_geometric_analytical_endpoints(
    client: TestClient,
) -> None:
    body = asian_body()
    body["simulation"] = {
        "num_simulations": 2_000,
        "seed": 42,
        "variance_reduction": "ANTITHETIC_GEOMETRIC_CONTROL",
    }
    arithmetic = client.post("/api/pricing/asian", json=body)
    assert arithmetic.status_code == 200
    data = arithmetic.json()
    assert data["averaging_type"] == "ARITHMETIC"
    assert data["analytical_geometric_control_price"] is not None
    assert data["control_variate_beta"] is not None
    assert "observations" not in data and "paths" not in data

    geometric_body = asian_body("GEOMETRIC")
    analytical = client.post(
        "/api/pricing/asian/geometric-analytical",
        json={
            "contract": geometric_body["contract"],
            "market": geometric_body["market"],
        },
    )
    assert analytical.status_code == 200
    assert analytical.json()["price"] > 0
    invalid = client.post(
        "/api/pricing/asian/geometric-analytical",
        json={"contract": body["contract"], "market": body["market"]},
    )
    assert invalid.status_code == 422


def test_asian_comparison_and_both_convergence_modes(client: TestClient) -> None:
    comparison = client.post("/api/analytics/asian/compare", json=asian_body())
    assert comparison.status_code == 200 and len(comparison.json()["points"]) == 4
    for mode, values in [("PATH_COUNT", [500, 1000]), ("MONITORING_COUNT", [2, 4])]:
        response = client.post(
            "/api/analytics/asian/convergence",
            json={**asian_body("GEOMETRIC"), "mode": mode, "values": values},
        )
        assert response.status_code == 200
        assert len(response.json()["points"]) == 2


def test_invalid_monitoring_schedule_and_asian_limits(client: TestClient) -> None:
    body = asian_body()
    contract = dict(body["contract"])  # type: ignore[arg-type]
    contract["averaging_schedule"] = {"monitoring_times": [0.5, 1.1]}
    assert client.post("/api/pricing/asian", json={**body, "contract": contract}).status_code == 422

    app.dependency_overrides[get_settings] = lambda: Settings(max_asian_monitoring_points=3)
    assert client.post("/api/pricing/asian", json=body).status_code == 422
    app.dependency_overrides[get_settings] = lambda: Settings(max_asian_total_path_nodes=5_000)
    response = client.post("/api/pricing/asian", json=body)
    assert response.status_code == 422 and "path nodes" in response.json()["detail"]
    app.dependency_overrides[get_settings] = lambda: Settings(max_api_simulations=1_000)
    assert client.post("/api/pricing/asian", json=body).status_code == 422


def test_huge_regular_schedule_is_rejected_before_allocation(client: TestClient) -> None:
    body = asian_body("GEOMETRIC")
    contract = dict(body["contract"])  # type: ignore[arg-type]
    contract["averaging_schedule"] = {"monitoring_count": 10**12}

    monte_carlo = client.post("/api/pricing/asian", json={**body, "contract": contract})
    analytical = client.post(
        "/api/pricing/asian/geometric-analytical",
        json={"contract": contract, "market": body["market"]},
    )

    assert monte_carlo.status_code == 422
    assert analytical.status_code == 422
    assert "monitoring count" in monte_carlo.json()["detail"]
    assert "monitoring count" in analytical.json()["detail"]


def test_exotic_convergence_point_limit(client: TestClient) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(max_exotic_convergence_points=1)
    asian_response = client.post(
        "/api/analytics/asian/convergence",
        json={**asian_body(), "mode": "PATH_COUNT", "values": [500, 1000]},
    )
    assert asian_response.status_code == 422
    digital_request = dict(digital_body())
    digital_request.pop("method")
    digital_response = client.post(
        "/api/analytics/digital/convergence",
        json={**digital_request, "simulation_counts": [500, 1000]},
    )
    assert digital_response.status_code == 422


@pytest.mark.parametrize("method", ["ANALYTICAL", "MONTE_CARLO"])
def test_digital_price_methods_and_serialization(client: TestClient, method: str) -> None:
    first = client.post("/api/pricing/digital", json=digital_body(method))
    again = client.post("/api/pricing/digital", json=digital_body(method))
    assert first.status_code == again.status_code == 200
    data = first.json()
    assert data["method"] == method and data["price"] == again.json()["price"]
    if method == "MONTE_CARLO":
        assert data["analytical_price"] > 0 and data["standard_error"] > 0


def test_digital_comparison_convergence_and_limits(client: TestClient) -> None:
    request = dict(digital_body())
    request.pop("method")
    comparison = client.post("/api/analytics/digital/compare", json=request)
    assert comparison.status_code == 200
    assert comparison.json()["analytical"]["price"] > 0
    convergence = client.post(
        "/api/analytics/digital/convergence",
        json={**request, "simulation_counts": [500, 1000]},
    )
    assert convergence.status_code == 200 and len(convergence.json()["points"]) == 2
    app.dependency_overrides[get_settings] = lambda: Settings(max_api_simulations=400)
    assert client.post("/api/analytics/digital/compare", json=request).status_code == 422


def test_exotic_settings_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_ASIAN_MONITORING_POINTS", "17")
    monkeypatch.setenv("MAX_ASIAN_TOTAL_PATH_NODES", "12345")
    monkeypatch.setenv("MAX_EXOTIC_CONVERGENCE_POINTS", "3")
    settings = Settings.from_environment()
    assert settings.max_asian_monitoring_points == 17
    assert settings.max_asian_total_path_nodes == 12345
    assert settings.max_exotic_convergence_points == 3
