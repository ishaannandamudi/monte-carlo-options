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


def body(
    basket_type: str = "ARITHMETIC",
    option_type: str = "CALL",
    variance_reduction: str = "ANTITHETIC",
) -> dict[str, object]:
    return {
        "contract": {
            "option_type": option_type,
            "strike": 95,
            "maturity_years": 1,
            "weights": [0.5, 0.5],
            "basket_type": basket_type,
        },
        "market": {
            "spots": [100, 90],
            "risk_free_rate": 0.02,
            "dividend_yields": [0.01, 0.0],
            "volatilities": [0.2, 0.3],
            "correlation_matrix": [[1, 0.3], [0.3, 1]],
            "asset_ids": ["A", "B"],
        },
        "simulation": {
            "num_simulations": 2_000,
            "seed": 42,
            "variance_reduction": variance_reduction,
        },
    }


@pytest.mark.parametrize("option_type", ["CALL", "PUT"])
@pytest.mark.parametrize(
    "method",
    ["NONE", "ANTITHETIC", "GEOMETRIC_CONTROL", "ANTITHETIC_GEOMETRIC_CONTROL"],
)
def test_arithmetic_basket_price_api(client: TestClient, option_type: str, method: str) -> None:
    response = client.post("/api/pricing/basket", json=body("ARITHMETIC", option_type, method))
    assert response.status_code == 200
    data = response.json()
    assert data["price"] >= 0 and data["num_assets"] == 2
    assert "terminal" not in data and "paths" not in data
    if "CONTROL" in method:
        assert data["analytical_geometric_price"] is not None


def test_geometric_mc_and_analytical_api(client: TestClient) -> None:
    request = body("GEOMETRIC")
    monte_carlo = client.post("/api/pricing/basket", json=request)
    analytical = client.post(
        "/api/pricing/basket/geometric-analytical",
        json={"contract": request["contract"], "market": request["market"]},
    )
    assert monte_carlo.status_code == analytical.status_code == 200
    assert abs(monte_carlo.json()["price"] - analytical.json()["price"]) < 1.0


def test_singular_and_invalid_correlation_api(client: TestClient) -> None:
    singular = body()
    singular_market = dict(singular["market"])  # type: ignore[arg-type]
    singular_market["correlation_matrix"] = [[1, 1], [1, 1]]
    response = client.post("/api/pricing/basket", json={**singular, "market": singular_market})
    assert response.status_code == 200
    assert response.json()["factorization"]["method"] == "EIGEN"

    invalid = body()
    invalid_market = dict(invalid["market"])  # type: ignore[arg-type]
    invalid_market["correlation_matrix"] = [[1, 0.5], [0.2, 1]]
    assert (
        client.post("/api/pricing/basket", json={**invalid, "market": invalid_market}).status_code
        == 422
    )


def test_comparison_convergence_and_sensitivity_apis(client: TestClient) -> None:
    comparison = client.post("/api/analytics/basket/compare", json=body())
    assert comparison.status_code == 200 and len(comparison.json()["points"]) == 4

    convergence = client.post(
        "/api/analytics/basket/convergence",
        json={**body("GEOMETRIC"), "simulation_counts": [500, 1_000]},
    )
    assert convergence.status_code == 200 and len(convergence.json()["points"]) == 2

    correlations = client.post(
        "/api/analytics/basket/correlation-sensitivity",
        json={**body(), "rho_values": [-0.5, 0.0, 0.5]},
    )
    assert correlations.status_code == 200 and len(correlations.json()["points"]) == 3

    weights = client.post(
        "/api/analytics/basket/weight-sensitivity",
        json={**body(), "first_asset_weights": [0.0, 0.5, 1.0]},
    )
    assert weights.status_code == 200 and len(weights.json()["points"]) == 3


def test_empirical_correlation_api(client: TestClient) -> None:
    request = body()
    response = client.post(
        "/api/analytics/multi-asset/correlation",
        json={
            "market": request["market"],
            "diagnostic": {"num_simulations": 10_000, "seed": 4},
        },
    )
    assert response.status_code == 200
    assert response.json()["maximum_absolute_error"] < 0.04
    assert "draws" not in response.json()


def test_api_resource_limits(client: TestClient) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(max_multi_asset_count=1)
    assert client.post("/api/pricing/basket", json=body()).status_code == 422

    app.dependency_overrides[get_settings] = lambda: Settings(max_multi_asset_path_nodes=100)
    response = client.post("/api/pricing/basket", json=body())
    assert response.status_code == 422 and "path nodes" in response.json()["detail"]

    app.dependency_overrides[get_settings] = lambda: Settings(max_api_simulations=1_000)
    assert client.post("/api/pricing/basket", json=body()).status_code == 422

    app.dependency_overrides[get_settings] = lambda: Settings(max_basket_convergence_points=1)
    assert (
        client.post(
            "/api/analytics/basket/convergence",
            json={**body(), "simulation_counts": [100, 200]},
        ).status_code
        == 422
    )

    app.dependency_overrides[get_settings] = lambda: Settings(max_basket_sensitivity_points=1)
    assert (
        client.post(
            "/api/analytics/basket/correlation-sensitivity",
            json={**body(), "rho_values": [0.0, 0.5]},
        ).status_code
        == 422
    )
