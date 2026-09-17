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


def body(*, option_type: str = "CALL", scheme: str = "QUADRATIC_EXPONENTIAL") -> dict[str, object]:
    return {
        "contract": {
            "option_type": option_type,
            "exercise_style": "EUROPEAN",
            "spot": 100,
            "strike": 100,
            "maturity_years": 1,
        },
        "market": {"risk_free_rate": 0.02, "dividend_yield": 0.01, "volatility": 0.0},
        "parameters": {"v0": 0.04, "kappa": 2, "theta": 0.04, "sigma_v": 0.3, "rho": -0.7},
        "simulation": {
            "num_simulations": 2_000,
            "num_time_steps": 20,
            "seed": 42,
            "scheme": scheme,
            "antithetic": True,
        },
    }


def analytical_body(request: dict[str, object] | None = None) -> dict[str, object]:
    source = request or body()
    return {key: source[key] for key in ("contract", "market", "parameters")}


@pytest.mark.parametrize("option_type", ["CALL", "PUT"])
def test_analytical_endpoint_serializes_diagnostics(client: TestClient, option_type: str) -> None:
    request = body(option_type=option_type)
    response = client.post(
        "/api/pricing/heston/analytical",
        json=analytical_body(request),
    )
    assert response.status_code == 200
    data = response.json()
    assert data["price"] >= 0.0
    assert data["method"] == "LITTLE_TRAP_FOURIER"
    assert data["integration_evaluations"] > 0


@pytest.mark.parametrize("scheme", ["FULL_TRUNCATION_EULER", "QUADRATIC_EXPONENTIAL"])
def test_monte_carlo_endpoint_has_no_raw_paths(client: TestClient, scheme: str) -> None:
    response = client.post("/api/pricing/heston/monte-carlo", json=body(scheme=scheme))
    assert response.status_code == 200
    data = response.json()
    assert data["price"] >= 0.0 and data["scheme"] == scheme
    assert data["num_simulations"] == 2_000
    assert "terminal_spots" not in data and "paths" not in data


def test_heston_research_endpoints(client: TestClient) -> None:
    request = body()
    comparison = client.post("/api/analytics/heston/compare", json=request)
    assert comparison.status_code == 200 and len(comparison.json()["points"]) == 2

    path_convergence = client.post(
        "/api/analytics/heston/convergence",
        json={**request, "mode": "PATH_COUNT", "values": [500, 1_000]},
    )
    time_convergence = client.post(
        "/api/analytics/heston/convergence",
        json={**request, "mode": "TIME_STEP", "values": [5, 10]},
    )
    assert path_convergence.status_code == time_convergence.status_code == 200
    assert len(path_convergence.json()["points"]) == 2

    smile = client.post(
        "/api/analytics/heston/smile",
        json={**analytical_body(request), "strikes": [80, 100, 120]},
    )
    sensitivity = client.post(
        "/api/analytics/heston/sensitivity",
        json={
            **analytical_body(request),
            "parameter": "rho",
            "values": [-0.8, -0.4, 0],
        },
    )
    diagnostics = client.post("/api/analytics/heston/variance-diagnostics", json=request)
    assert smile.status_code == sensitivity.status_code == diagnostics.status_code == 200
    assert [point["strike"] for point in smile.json()["points"]] == [80, 100, 120]
    assert len(sensitivity.json()["points"]) == 3
    assert len(diagnostics.json()["points"]) == 21


def test_invalid_parameters_and_american_contract_are_rejected(client: TestClient) -> None:
    request = body()
    invalid = {**request, "parameters": {**request["parameters"], "rho": 1.1}}  # type: ignore[dict-item]
    assert client.post("/api/pricing/heston/monte-carlo", json=invalid).status_code == 422

    american_contract = {**request["contract"], "exercise_style": "AMERICAN"}  # type: ignore[dict-item]
    american = {**request, "contract": american_contract}
    assert client.post("/api/pricing/heston/monte-carlo", json=american).status_code == 422
    assert (
        client.post("/api/pricing/heston/analytical", json=analytical_body(american)).status_code
        == 422
    )


@pytest.mark.parametrize(
    "settings,endpoint,payload,detail",
    [
        (Settings(max_heston_paths=100), "/api/pricing/heston/monte-carlo", body(), "path count"),
        (
            Settings(max_heston_time_steps=10),
            "/api/pricing/heston/monte-carlo",
            body(),
            "time-step count",
        ),
        (
            Settings(max_heston_total_path_nodes=1_000),
            "/api/pricing/heston/monte-carlo",
            body(),
            "path nodes",
        ),
        (
            Settings(max_heston_convergence_points=1),
            "/api/analytics/heston/convergence",
            {**body(), "mode": "PATH_COUNT", "values": [500, 1_000]},
            "point count",
        ),
        (
            Settings(max_heston_smile_points=1),
            "/api/analytics/heston/smile",
            {**analytical_body(), "strikes": [90, 100]},
            "point count",
        ),
        (
            Settings(max_heston_sensitivity_points=1),
            "/api/analytics/heston/sensitivity",
            {**analytical_body(), "parameter": "rho", "values": [-0.5, 0]},
            "point count",
        ),
    ],
)
def test_heston_resource_limits(
    client: TestClient,
    settings: Settings,
    endpoint: str,
    payload: dict[str, object],
    detail: str,
) -> None:
    app.dependency_overrides[get_settings] = lambda: settings
    response = client.post(endpoint, json=payload)
    assert response.status_code == 422
    assert detail in response.json()["detail"]


def test_heston_environment_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_HESTON_PATHS", "123")
    monkeypatch.setenv("MAX_HESTON_TIME_STEPS", "17")
    monkeypatch.setenv("MAX_HESTON_TOTAL_PATH_NODES", "999")
    monkeypatch.setenv("MAX_HESTON_CONVERGENCE_POINTS", "4")
    monkeypatch.setenv("MAX_HESTON_SMILE_POINTS", "8")
    monkeypatch.setenv("MAX_HESTON_SENSITIVITY_POINTS", "6")
    settings = Settings.from_environment()
    assert (
        settings.max_heston_paths,
        settings.max_heston_time_steps,
        settings.max_heston_total_path_nodes,
        settings.max_heston_convergence_points,
        settings.max_heston_smile_points,
        settings.max_heston_sensitivity_points,
    ) == (123, 17, 999, 4, 8, 6)
