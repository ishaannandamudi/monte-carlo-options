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
    *,
    direction: str = "DOWN",
    style: str = "OUT",
    option_type: str = "CALL",
    mode: str = "DISCRETE",
) -> dict[str, object]:
    return {
        "contract": {
            "option_type": option_type,
            "barrier_direction": direction,
            "barrier_style": style,
            "spot": 100,
            "strike": 100,
            "barrier": 90 if direction == "DOWN" else 110,
            "maturity_years": 1,
            "monitoring_schedule": {"monitoring_count": 4},
            "monitoring_mode": mode,
            "rebate": 0,
            "rebate_timing": "AT_EXPIRY",
        },
        "market": {"risk_free_rate": 0.05, "volatility": 0.2, "dividend_yield": 0.01},
        "simulation": {
            "num_simulations": 2_000,
            "seed": 42,
            "variance_reduction": "ANTITHETIC",
        },
    }


@pytest.mark.parametrize(
    "direction,style,option_type",
    [
        ("DOWN", "OUT", "CALL"),
        ("DOWN", "IN", "CALL"),
        ("UP", "OUT", "PUT"),
        ("UP", "IN", "PUT"),
    ],
)
@pytest.mark.parametrize("mode", ["DISCRETE", "BROWNIAN_BRIDGE"])
def test_barrier_price_endpoint_combinations(
    client: TestClient,
    direction: str,
    style: str,
    option_type: str,
    mode: str,
) -> None:
    response = client.post(
        "/api/pricing/barrier",
        json=body(direction=direction, style=style, option_type=option_type, mode=mode),
    )
    assert response.status_code == 200
    data = response.json()
    assert data["price"] >= 0
    assert data["monitoring_mode"] == mode
    assert data["hit_probability"] + data["no_hit_probability"] == pytest.approx(1)
    assert "observations" not in data and "paths" not in data


def test_barrier_analytical_compare_parity_and_research_endpoints(client: TestClient) -> None:
    request = body()
    analytical = client.post(
        "/api/pricing/barrier/analytical",
        json={"contract": request["contract"], "market": request["market"]},
    )
    assert analytical.status_code == 200 and analytical.json()["price"] > 0
    comparison = client.post("/api/analytics/barrier/compare", json=request)
    assert comparison.status_code == 200
    assert comparison.json()["analytical_continuous"] is not None
    parity = client.post("/api/analytics/barrier/parity", json=request)
    assert parity.status_code == 200
    assert abs(parity.json()["parity_residual"]) < 1e-12
    convergence = client.post(
        "/api/analytics/barrier/convergence",
        json={**request, "mode": "PATH_COUNT", "values": [500, 1000]},
    )
    assert convergence.status_code == 200 and len(convergence.json()["points"]) == 2
    monitoring = client.post(
        "/api/analytics/barrier/convergence",
        json={**request, "mode": "MONITORING_COUNT", "values": [2, 4]},
    )
    assert monitoring.status_code == 200 and len(monitoring.json()["points"]) == 2
    sensitivity = client.post(
        "/api/analytics/barrier/sensitivity",
        json={**request, "barrier_levels": [70, 80, 90]},
    )
    assert sensitivity.status_code == 200 and len(sensitivity.json()["levels"]) == 3


def test_rebates_validation_and_serialization(client: TestClient) -> None:
    request = body()
    contract = dict(request["contract"])  # type: ignore[arg-type]
    contract.update({"rebate": 3, "rebate_timing": "AT_HIT"})
    valid = client.post("/api/pricing/barrier", json={**request, "contract": contract})
    assert valid.status_code == 200 and valid.json()["rebate"] == 3
    unsupported_comparison = client.post(
        "/api/analytics/barrier/compare",
        json={**request, "contract": contract},
    )
    assert unsupported_comparison.status_code == 422
    contract["monitoring_mode"] = "BROWNIAN_BRIDGE"
    assert (
        client.post("/api/pricing/barrier", json={**request, "contract": contract}).status_code
        == 422
    )
    contract["monitoring_mode"] = "DISCRETE"
    contract["barrier_style"] = "IN"
    assert (
        client.post("/api/pricing/barrier", json={**request, "contract": contract}).status_code
        == 422
    )


def test_barrier_resource_limits_and_huge_count_preallocation_rejection(
    client: TestClient,
) -> None:
    request = body()
    app.dependency_overrides[get_settings] = lambda: Settings(max_api_simulations=100)
    assert client.post("/api/pricing/barrier", json=request).status_code == 422

    app.dependency_overrides[get_settings] = lambda: Settings(max_barrier_monitoring_points=3)
    assert client.post("/api/pricing/barrier", json=request).status_code == 422

    app.dependency_overrides[get_settings] = lambda: Settings(max_barrier_total_path_nodes=5_000)
    assert client.post("/api/pricing/barrier", json=request).status_code == 422

    app.dependency_overrides[get_settings] = lambda: Settings(max_barrier_monitoring_points=5_000)
    contract = dict(request["contract"])  # type: ignore[arg-type]
    contract["monitoring_schedule"] = {"monitoring_count": 10**12}
    response = client.post("/api/pricing/barrier", json={**request, "contract": contract})
    assert response.status_code == 422
    assert "monitoring count" in response.json()["detail"]


def test_barrier_point_limits_and_environment_settings(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = body()
    app.dependency_overrides[get_settings] = lambda: Settings(max_barrier_convergence_points=1)
    convergence = client.post(
        "/api/analytics/barrier/convergence",
        json={**request, "mode": "PATH_COUNT", "values": [500, 1000]},
    )
    sensitivity = client.post(
        "/api/analytics/barrier/sensitivity",
        json={**request, "barrier_levels": [80, 90]},
    )
    assert convergence.status_code == sensitivity.status_code == 422

    monkeypatch.setenv("MAX_BARRIER_MONITORING_POINTS", "17")
    monkeypatch.setenv("MAX_BARRIER_TOTAL_PATH_NODES", "12345")
    monkeypatch.setenv("MAX_BARRIER_CONVERGENCE_POINTS", "3")
    settings = Settings.from_environment()
    assert settings.max_barrier_monitoring_points == 17
    assert settings.max_barrier_total_path_nodes == 12345
    assert settings.max_barrier_convergence_points == 3
