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


def base_contract() -> dict[str, object]:
    return {
        "option_type": "CALL",
        "exercise_style": "EUROPEAN",
        "spot": 100.0,
        "strike": 100.0,
        "maturity_years": 1.0,
    }


def base_market() -> dict[str, float]:
    return {"risk_free_rate": 0.05, "volatility": 0.2, "dividend_yield": 0.0}


def base_simulation() -> dict[str, object]:
    return {
        "num_simulations": 10_000,
        "seed": 42,
        "variance_reduction": "ANTITHETIC",
        "random_engine": "PCG64",
        "confidence_level": 0.95,
    }


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "monte-carlo-options"}


def test_black_scholes_endpoint(client: TestClient) -> None:
    response = client.post(
        "/api/pricing/black-scholes",
        json={"contract": base_contract(), "market": base_market()},
    )
    assert response.status_code == 200
    assert response.json()["price"] == pytest.approx(10.450583572185565)


def test_monte_carlo_endpoint_has_diagnostics_but_no_arrays(client: TestClient) -> None:
    response = client.post(
        "/api/pricing/monte-carlo",
        json={
            "contract": base_contract(),
            "market": base_market(),
            "simulation": base_simulation(),
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["num_simulations"] == 10_000
    assert "standard_error" in body
    assert "terminal_prices" not in body
    assert "discounted_payoffs" not in body


def test_comparison_endpoint(client: TestClient) -> None:
    response = client.post(
        "/api/pricing/compare",
        json={
            "contract": base_contract(),
            "market": base_market(),
            "simulation": base_simulation(),
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "monte_carlo" in body
    assert body["black_scholes_inside_confidence_interval"] is True


def test_convergence_endpoint(client: TestClient) -> None:
    response = client.post(
        "/api/analytics/convergence",
        json={
            "contract": base_contract(),
            "market": base_market(),
            "simulation_counts": [1_000, 4_000, 16_000],
            "root_seed": 42,
            "variance_reduction": "ANTITHETIC",
            "confidence_level": 0.95,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["points"]) == 3
    assert body["slope_points_used"] == 3


def test_api_simulation_limit_is_enforced(client: TestClient) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(
        max_api_simulations=100,
        max_convergence_points=2,
    )
    simulation = base_simulation()
    simulation["num_simulations"] = 101
    response = client.post(
        "/api/pricing/monte-carlo",
        json={
            "contract": base_contract(),
            "market": base_market(),
            "simulation": simulation,
        },
    )
    assert response.status_code == 422
    assert "API limit" in response.json()["detail"]


def test_api_convergence_point_and_path_limits_are_enforced(client: TestClient) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(
        max_api_simulations=1_000,
        max_convergence_points=2,
    )
    too_many_points = client.post(
        "/api/analytics/convergence",
        json={
            "contract": base_contract(),
            "market": base_market(),
            "simulation_counts": [10, 20, 30],
        },
    )
    assert too_many_points.status_code == 422

    too_many_paths = client.post(
        "/api/analytics/convergence",
        json={
            "contract": base_contract(),
            "market": base_market(),
            "simulation_counts": [100, 1_001],
        },
    )
    assert too_many_paths.status_code == 422


def test_invalid_input_and_unknown_enum_return_422(client: TestClient) -> None:
    bad_contract = base_contract()
    bad_contract["spot"] = 0.0
    bad_contract["option_type"] = "DIGITAL"
    response = client.post(
        "/api/pricing/black-scholes",
        json={"contract": bad_contract, "market": base_market()},
    )
    assert response.status_code == 422


def test_quantitative_domain_failure_returns_clear_422(client: TestClient) -> None:
    contract = base_contract()
    contract["maturity_years"] = 2.0
    market = base_market()
    market["risk_free_rate"] = -1e308
    market["volatility"] = 0.0
    response = client.post(
        "/api/pricing/black-scholes",
        json={"contract": contract, "market": market},
    )
    assert response.status_code == 422
    assert "floating-point range" in response.json()["detail"]


@pytest.mark.parametrize(
    "method",
    [
        "NONE",
        "ANTITHETIC",
        "CONTROL_VARIATE",
        "ANTITHETIC_CONTROL_VARIATE",
    ],
)
def test_monte_carlo_api_accepts_all_variance_reduction_methods(
    client: TestClient, method: str
) -> None:
    simulation = base_simulation()
    simulation["variance_reduction"] = method
    response = client.post(
        "/api/pricing/monte-carlo",
        json={
            "contract": base_contract(),
            "market": base_market(),
            "simulation": simulation,
        },
    )
    assert response.status_code == 200
    assert response.json()["variance_reduction"] == method


def test_control_variate_diagnostics_serialize(client: TestClient) -> None:
    simulation = base_simulation()
    simulation["variance_reduction"] = "ANTITHETIC_CONTROL_VARIATE"
    response = client.post(
        "/api/pricing/compare",
        json={
            "contract": base_contract(),
            "market": base_market(),
            "simulation": simulation,
        },
    )
    assert response.status_code == 200
    monte_carlo = response.json()["monte_carlo"]
    assert monte_carlo["control_variate_beta"] is not None
    assert monte_carlo["raw_standard_error"] > monte_carlo["standard_error"]
    assert monte_carlo["adjusted_standard_error"] == monte_carlo["standard_error"]


def test_invalid_variance_reduction_method_returns_422(client: TestClient) -> None:
    simulation = base_simulation()
    simulation["variance_reduction"] = "STRATIFIED"
    response = client.post(
        "/api/pricing/monte-carlo",
        json={
            "contract": base_contract(),
            "market": base_market(),
            "simulation": simulation,
        },
    )
    assert response.status_code == 422


def test_variance_reduction_comparison_endpoint(client: TestClient) -> None:
    response = client.post(
        "/api/analytics/variance-reduction",
        json={
            "contract": base_contract(),
            "market": base_market(),
            "num_simulations": 20_000,
            "root_seed": 42,
            "confidence_level": 0.95,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["baseline_method"] == "NONE"
    assert [point["method"] for point in body["points"]] == [
        "NONE",
        "ANTITHETIC",
        "CONTROL_VARIATE",
        "ANTITHETIC_CONTROL_VARIATE",
    ]


def test_variance_reduction_endpoint_preserves_api_limit(client: TestClient) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(max_api_simulations=100)
    response = client.post(
        "/api/analytics/variance-reduction",
        json={
            "contract": base_contract(),
            "market": base_market(),
            "num_simulations": 101,
        },
    )
    assert response.status_code == 422
