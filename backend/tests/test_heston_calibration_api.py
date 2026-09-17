from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from backend.app.calibration.synthetic import (
    FELLER_SATISFIED_PARAMETERS,
    generate_synthetic_heston_dataset,
)
from backend.app.core.config import Settings, get_settings
from backend.app.main import app
from backend.app.volatility_surface.synthetic import generate_synthetic_surface


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def dataset_payload() -> dict[str, object]:
    dataset = generate_synthetic_heston_dataset(
        maturities=(0.5, 1.0), log_moneyness=(-0.1, 0.0, 0.1)
    )
    return dataset.model_dump(mode="json")


def config_payload(**updates: object) -> dict[str, object]:
    parameters = {
        name: getattr(FELLER_SATISFIED_PARAMETERS, name)
        for name in ("v0", "kappa", "theta", "sigma_v", "rho")
    }
    values: dict[str, object] = {
        "target": "PRICE",
        "initial_parameters": parameters,
        "max_iterations": 3,
        "max_returned_residuals": 20,
        "identifiability_diagnostics": False,
    }
    values.update(updates)
    return values


def request_payload(**config_updates: object) -> dict[str, object]:
    return {"dataset": dataset_payload(), "config": config_payload(**config_updates)}


def test_calibration_and_analytics_endpoints_serialize_without_raw_arrays(
    client: TestClient,
) -> None:
    calibration = client.post("/api/calibration/heston", json=request_payload())
    assert calibration.status_code == 200, calibration.text
    calibrated = calibration.json()
    assert calibrated["rmse"] < 2e-9
    assert calibrated["observations_used"] == 6
    assert "paths" not in calibrated and "characteristic_values" not in calibrated

    analytics_payload = {
        **request_payload(),
        "parameters": {
            name: getattr(FELLER_SATISFIED_PARAMETERS, name)
            for name in ("v0", "kappa", "theta", "sigma_v", "rho")
        },
    }
    diagnostics = client.post(
        "/api/analytics/heston/calibration/diagnostics", json=analytics_payload
    )
    fitted = client.post("/api/analytics/heston/calibration/fitted-surface", json=analytics_payload)
    assert diagnostics.status_code == fitted.status_code == 200
    assert diagnostics.json()["metrics"]["rmse"] < 2e-9
    assert len(fitted.json()["points"]) == 6
    assert len(fitted.json()["atm_term_structure"]) == 2


def test_iv_calibration_and_same_request_reproducibility(client: TestClient) -> None:
    payload = request_payload(target="IMPLIED_VOLATILITY", max_iterations=5)
    first = client.post("/api/calibration/heston", json=payload)
    second = client.post("/api/calibration/heston", json=payload)
    assert first.status_code == second.status_code == 200
    assert first.json()["calibrated_parameters"] == second.json()["calibrated_parameters"]
    assert first.json()["objective_value"] == second.json()["objective_value"]
    assert first.json()["target"] == "IMPLIED_VOLATILITY"


def test_surface_calibration_endpoint(client: TestClient) -> None:
    surface = generate_synthetic_surface(
        maturities=(0.5, 1.0),
        log_moneyness=(-0.1, 0.0, 0.1),
        base_volatility=0.2,
        skew=-0.1,
    )
    response = client.post(
        "/api/calibration/heston/surface",
        json={
            "surface": surface.model_dump(mode="json"),
            "config": config_payload(selection_policy="ALL", max_iterations=1),
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["observations_used"] == 6


def test_stability_and_holdout_endpoints(client: TestClient) -> None:
    stability = client.post(
        "/api/analytics/heston/calibration/stability",
        json={**request_payload(max_iterations=1), "run_count": 2},
    )
    validation = client.post(
        "/api/analytics/heston/calibration/validate",
        json={
            **request_payload(),
            "holdout": {"policy": "EXPIRY", "maturities": [1.0]},
        },
    )
    assert stability.status_code == validation.status_code == 200
    assert len(stability.json()["runs"]) == 2
    assert validation.json()["training_observations"] == 3
    assert validation.json()["validation_observations"] == 3


@pytest.mark.parametrize(
    ("settings", "payload", "endpoint", "detail"),
    [
        (
            Settings(max_heston_calibration_observations=2),
            request_payload(),
            "/api/calibration/heston",
            "observation count",
        ),
        (
            Settings(max_heston_calibration_starts=1),
            request_payload(multi_start_count=2),
            "/api/calibration/heston",
            "start count",
        ),
        (
            Settings(max_heston_calibration_iterations=1),
            request_payload(max_iterations=2),
            "/api/calibration/heston",
            "iteration count",
        ),
        (
            Settings(max_heston_calibration_diagnostic_points=2),
            request_payload(max_returned_residuals=3),
            "/api/calibration/heston",
            "diagnostic point count",
        ),
    ],
)
def test_calibration_resource_limits(
    client: TestClient,
    settings: Settings,
    payload: dict[str, object],
    endpoint: str,
    detail: str,
) -> None:
    app.dependency_overrides[get_settings] = lambda: settings
    response = client.post(endpoint, json=payload)
    assert response.status_code == 422
    assert detail in response.json()["detail"]


def test_surface_and_stability_resource_limits(client: TestClient) -> None:
    surface = generate_synthetic_surface(maturities=(0.5,), log_moneyness=(-0.1, 0.0, 0.1))
    app.dependency_overrides[get_settings] = lambda: Settings(
        max_heston_calibration_surface_points=2
    )
    response = client.post(
        "/api/calibration/heston/surface",
        json={"surface": surface.model_dump(mode="json"), "config": config_payload()},
    )
    assert response.status_code == 422
    assert "surface point count" in response.json()["detail"]

    app.dependency_overrides[get_settings] = lambda: Settings(max_heston_calibration_starts=1)
    response = client.post(
        "/api/analytics/heston/calibration/stability",
        json={**request_payload(), "run_count": 2},
    )
    assert response.status_code == 422
    assert "stability run count" in response.json()["detail"]


def test_invalid_calibration_request_is_http_422(client: TestClient) -> None:
    payload = request_payload()
    payload["config"] = {**config_payload(), "target": "NOT_A_TARGET"}
    assert client.post("/api/calibration/heston", json=payload).status_code == 422

    payload = request_payload()
    payload["config"] = {
        **config_payload(),
        "bounds": {"rho": {"lower": 0.5, "upper": -0.5}},
    }
    assert client.post("/api/calibration/heston", json=payload).status_code == 422

    payload = request_payload()
    dataset = payload["dataset"]
    assert isinstance(dataset, dict)
    dataset["spot"] = 0.0
    assert client.post("/api/calibration/heston", json=payload).status_code == 422


def test_calibration_environment_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_HESTON_CALIBRATION_OBSERVATIONS", "41")
    monkeypatch.setenv("MAX_HESTON_CALIBRATION_STARTS", "4")
    monkeypatch.setenv("MAX_HESTON_CALIBRATION_ITERATIONS", "77")
    monkeypatch.setenv("MAX_HESTON_CALIBRATION_SURFACE_POINTS", "101")
    monkeypatch.setenv("MAX_HESTON_CALIBRATION_DIAGNOSTIC_POINTS", "88")
    settings = Settings.from_environment()
    assert (
        settings.max_heston_calibration_observations,
        settings.max_heston_calibration_starts,
        settings.max_heston_calibration_iterations,
        settings.max_heston_calibration_surface_points,
        settings.max_heston_calibration_diagnostic_points,
    ) == (41, 4, 77, 101, 88)
