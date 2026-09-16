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


def body(option_type: str = "PUT") -> dict[str, object]:
    return {
        "contract": {
            "option_type": option_type,
            "exercise_style": "AMERICAN",
            "spot": 100,
            "strike": 100,
            "maturity_years": 1,
        },
        "market": {"risk_free_rate": 0.05, "volatility": 0.2, "dividend_yield": 0},
        "lsm": {
            "num_paths": 1000,
            "num_time_steps": 10,
            "seed": 42,
            "store_regression_diagnostics": True,
        },
    }


@pytest.mark.parametrize("option_type", ["PUT", "CALL"])
def test_lsm_price_serialization_and_reproducibility(client: TestClient, option_type: str) -> None:
    first = client.post("/api/pricing/longstaff-schwartz", json=body(option_type))
    again = client.post("/api/pricing/longstaff-schwartz", json=body(option_type))
    assert first.status_code == again.status_code == 200
    data = first.json()
    assert data["price"] == again.json()["price"]
    assert data["evaluation_mode"] == "OUT_OF_SAMPLE"
    assert data["training_seed"] != data["valuation_seed"]
    assert "paths" not in data and "cashflows" not in data
    assert data["num_paths"] == 1000


def test_lsm_comparison_and_convergence_endpoints(client: TestClient) -> None:
    request = {**body(), "tree": {"steps": 100}}
    response = client.post("/api/analytics/longstaff-schwartz/compare", json=request)
    assert response.status_code == 200
    assert response.json()["crr_american"]["steps"] == 100
    for mode, values in [("PATH_COUNT", [500, 1000]), ("TIME_STEPS", [5, 10])]:
        response = client.post(
            "/api/analytics/longstaff-schwartz/convergence",
            json={**request, "mode": mode, "values": values},
        )
        assert response.status_code == 200
        assert len(response.json()["points"]) == 2


def test_lsm_basis_comparison_endpoint(client: TestClient) -> None:
    response = client.post(
        "/api/analytics/longstaff-schwartz/basis-comparison",
        json={
            **body(),
            "families": ["POWER", "LAGUERRE"],
            "degrees": [1, 2],
            "tree": {"steps": 100},
        },
    )
    assert response.status_code == 200
    assert len(response.json()["points"]) == 4


def test_optional_boundary_comparison_endpoint(client: TestClient) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(max_lsm_diagnostic_dates=2)
    response = client.post(
        "/api/analytics/longstaff-schwartz/compare",
        json={**body(), "tree": {"steps": 100}, "include_boundary_comparison": True},
    )
    assert response.status_code == 200
    assert 0 < len(response.json()["boundary_comparison"]) <= 2


@pytest.mark.parametrize(
    "updates",
    [
        {"num_paths": 0},
        {"num_time_steps": 0},
        {"basis_degree": 6},
        {"confidence_level": 1},
        {"variance_reduction": "CONTROL_VARIATE"},
        {"exercise_steps": [1, 2]},
    ],
)
def test_invalid_lsm_configuration(client: TestClient, updates: dict[str, object]) -> None:
    request = body()
    request["lsm"] = {"num_paths": 100, "num_time_steps": 10, **updates}
    assert client.post("/api/pricing/longstaff-schwartz", json=request).status_code == 422


def test_lsm_rejects_european(client: TestClient) -> None:
    request = body()
    request["contract"] = {
        "option_type": "PUT",
        "exercise_style": "EUROPEAN",
        "spot": 100,
        "strike": 100,
        "maturity_years": 1,
    }
    response = client.post("/api/pricing/longstaff-schwartz", json=request)
    assert response.status_code == 422 and "AMERICAN" in response.json()["detail"]


@pytest.mark.parametrize(
    "settings,updates,detail",
    [
        ({"max_lsm_paths": 100}, {}, "path count"),
        ({"max_lsm_paths": 1000}, {"regression_paths": 1001}, "path count"),
        ({"max_lsm_time_steps": 5}, {}, "time-step"),
        ({"max_lsm_total_path_nodes": 15000}, {}, "path nodes"),
    ],
)
def test_resource_limits_include_both_training_and_valuation(
    client: TestClient, settings: dict[str, int], updates: dict[str, int], detail: str
) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings.model_validate(settings)
    request = body()
    request["lsm"] = {"num_paths": 1000, "num_time_steps": 10, **updates}
    response = client.post("/api/pricing/longstaff-schwartz", json=request)
    assert response.status_code == 422 and detail in response.json()["detail"]


def test_diagnostic_dates_are_capped_without_changing_prices(client: TestClient) -> None:
    full = client.post("/api/pricing/longstaff-schwartz", json=body()).json()
    app.dependency_overrides[get_settings] = lambda: Settings(max_lsm_diagnostic_dates=2)
    response = client.post("/api/pricing/longstaff-schwartz", json=body())
    assert response.status_code == 200
    capped = response.json()
    assert len(capped["regression_diagnostics"]) == 2
    assert len(capped["boundary_diagnostics"]) <= 2
    assert capped["regression_diagnostics_truncated"]
    assert capped["price"] == full["price"]


def test_experiment_aggregate_work_and_tree_limits(client: TestClient) -> None:
    request = {**body(), "tree": {"steps": 100}, "mode": "PATH_COUNT", "values": [500, 1000]}
    app.dependency_overrides[get_settings] = lambda: Settings(max_lsm_convergence_points=1)
    assert (
        client.post("/api/analytics/longstaff-schwartz/convergence", json=request).status_code
        == 422
    )
    app.dependency_overrides[get_settings] = lambda: Settings(max_lsm_total_path_nodes=25000)
    response = client.post("/api/analytics/longstaff-schwartz/convergence", json=request)
    assert response.status_code == 422 and "path nodes" in response.json()["detail"]
    app.dependency_overrides[get_settings] = lambda: Settings(max_binomial_steps=50)
    assert (
        client.post(
            "/api/analytics/longstaff-schwartz/compare", json={**body(), "tree": {"steps": 100}}
        ).status_code
        == 422
    )


def test_lsm_settings_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_LSM_PATHS", "123")
    monkeypatch.setenv("MAX_LSM_TIME_STEPS", "17")
    monkeypatch.setenv("MAX_LSM_TOTAL_PATH_NODES", "10000")
    monkeypatch.setenv("MAX_LSM_CONVERGENCE_POINTS", "3")
    monkeypatch.setenv("MAX_LSM_DIAGNOSTIC_DATES", "4")
    settings = Settings.from_environment()
    assert settings.max_lsm_paths == 123 and settings.max_lsm_time_steps == 17
    assert settings.max_lsm_total_path_nodes == 10000
    assert settings.max_lsm_convergence_points == 3 and settings.max_lsm_diagnostic_dates == 4
