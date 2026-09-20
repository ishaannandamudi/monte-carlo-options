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


def contract() -> dict[str, object]:
    return {
        "option_type": "CALL",
        "exercise_style": "EUROPEAN",
        "spot": 100.0,
        "strike": 100.0,
        "maturity_years": 1.0,
    }


def market() -> dict[str, object]:
    return {"risk_free_rate": 0.03, "volatility": 0.2, "dividend_yield": 0.01}


def parameters() -> dict[str, object]:
    return {
        name: getattr(FELLER_SATISFIED_PARAMETERS, name)
        for name in ("v0", "kappa", "theta", "sigma_v", "rho")
    }


def comparison_payload(models: list[str] | None = None) -> dict[str, object]:
    return {
        "contract": contract(),
        "market": market(),
        "config": {
            "selected_models": models or ["BLACK_SCHOLES", "MONTE_CARLO_GBM", "CRR_BINOMIAL"],
            "reference_model": "BLACK_SCHOLES",
            "deterministic_seed": 71,
        },
        "gbm_monte_carlo": {"num_simulations": 2_000, "seed": 4},
        "tree": {"steps": 100},
        "heston_parameters": parameters(),
        "heston_monte_carlo": {
            "num_simulations": 1_000,
            "num_time_steps": 5,
            "seed": 4,
        },
        "longstaff_schwartz": {
            "num_paths": 1_000,
            "regression_paths": 1_000,
            "num_time_steps": 5,
        },
    }


def market_fit_payload() -> dict[str, object]:
    dataset = generate_synthetic_heston_dataset(maturities=(0.5,), log_moneyness=(-0.1, 0.0, 0.1))
    return {
        "dataset": dataset.model_dump(mode="json"),
        "selected_models": ["BLACK_SCHOLES", "HESTON_ANALYTICAL"],
        "heston_parameters": parameters(),
        "use_observed_iv_for_black_scholes": True,
        "preprocessing": {"target": "PRICE", "selection_policy": "ALL"},
    }


def test_registry_compare_and_partial_failure_endpoints(client: TestClient) -> None:
    registry = client.get("/api/analytics/models/registry")
    assert registry.status_code == 200
    assert len(registry.json()) == 9

    first = client.post("/api/analytics/models/compare", json=comparison_payload())
    second = client.post("/api/analytics/models/compare", json=comparison_payload())
    assert first.status_code == second.status_code == 200, first.text
    assert len(first.json()["successful_models"]) == 3
    assert (
        first.json()["experiment"]["input_signature"]
        == second.json()["experiment"]["input_signature"]
    )
    assert [item["price"] for item in first.json()["outputs"]] == [
        item["price"] for item in second.json()["outputs"]
    ]

    failed_payload = comparison_payload(["BLACK_SCHOLES", "CRR_BINOMIAL"])
    failed_payload["market"] = {
        "risk_free_rate": 10.0,
        "volatility": 0.01,
        "dividend_yield": 0.0,
    }
    failed_payload["tree"] = {"steps": 1}
    partial = client.post("/api/analytics/models/compare", json=failed_payload)
    assert partial.status_code == 200
    assert partial.json()["successful_models"] == ["BLACK_SCHOLES"]
    assert partial.json()["failed_models"] == ["CRR_BINOMIAL"]


def test_market_fit_risk_frontier_consistency_and_surface_endpoints(
    client: TestClient,
) -> None:
    fit = client.post("/api/analytics/models/market-fit", json=market_fit_payload())
    assert fit.status_code == 200, fit.text
    assert fit.json()["observations_used"] == 3
    assert fit.json()["models"][1]["metrics"]["rmse"] == pytest.approx(0.0)

    risk = client.post(
        "/api/analytics/models/risk",
        json={
            "comparison": comparison_payload(["BLACK_SCHOLES", "HESTON_ANALYTICAL"]),
            "market_fit": market_fit_payload(),
            "calibration_ensemble": [
                parameters(),
                {**parameters(), "rho": -0.6},
            ],
        },
    )
    assert risk.status_code == 200, risk.text
    assert risk.json()["calibration_stability"] is not None
    assert "model_dispersion" in risk.json()["uncertainty"]

    frontier = client.post(
        "/api/analytics/models/compute-frontier",
        json={
            "comparison": comparison_payload(["MONTE_CARLO_GBM"]),
            "model": "MONTE_CARLO_GBM",
            "budgets": [200, 500],
            "reference_model": "BLACK_SCHOLES",
        },
    )
    assert frontier.status_code == 200, frontier.text
    assert [item["budget"] for item in frontier.json()["points"]] == [200, 500]

    consistency = client.post(
        "/api/analytics/models/consistency",
        json={
            "contract": contract(),
            "market": market(),
            "simulation": {"num_simulations": 1_000, "seed": 3},
            "tree": {"steps": 100},
            "heston_monte_carlo": {
                "num_simulations": 500,
                "num_time_steps": 5,
                "seed": 3,
            },
            "tolerance": 0.2,
        },
    )
    assert consistency.status_code == 200, consistency.text
    assert len(consistency.json()["checks"]) == 5

    surface = generate_synthetic_surface(
        maturities=(0.5,), log_moneyness=(-0.1, 0.0, 0.1), skew=-0.1
    )
    surface_result = client.post(
        "/api/analytics/models/surface-comparison",
        json={
            "surface": surface.model_dump(mode="json"),
            "selected_models": ["BLACK_SCHOLES"],
        },
    )
    assert surface_result.status_code == 200, surface_result.text
    assert surface_result.json()["models"][0]["metrics"]["iv_rmse"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("settings", "endpoint", "payload", "detail"),
    [
        (
            Settings(max_model_comparison_models=1),
            "/api/analytics/models/compare",
            comparison_payload(["BLACK_SCHOLES", "CRR_BINOMIAL"]),
            "model comparison count",
        ),
        (
            Settings(max_model_comparison_observations=2),
            "/api/analytics/models/market-fit",
            market_fit_payload(),
            "observation count",
        ),
        (
            Settings(max_model_comparison_experiment_points=1),
            "/api/analytics/models/compute-frontier",
            {
                "comparison": comparison_payload(["MONTE_CARLO_GBM"]),
                "model": "MONTE_CARLO_GBM",
                "budgets": [100, 200],
                "reference_model": "BLACK_SCHOLES",
            },
            "experiment point count",
        ),
        (
            Settings(max_model_comparison_runtime_repetitions=1),
            "/api/analytics/models/compare",
            {
                **comparison_payload(["BLACK_SCHOLES"]),
                "config": {
                    "selected_models": ["BLACK_SCHOLES"],
                    "runtime_repetitions": 2,
                },
            },
            "runtime repetitions",
        ),
    ],
)
def test_model_comparison_resource_limits(
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


def test_surface_comparison_resource_limit(client: TestClient) -> None:
    surface = generate_synthetic_surface(maturities=(0.5,), log_moneyness=(-0.1, 0.0, 0.1))
    app.dependency_overrides[get_settings] = lambda: Settings(max_model_comparison_surface_points=2)
    response = client.post(
        "/api/analytics/models/surface-comparison",
        json={
            "surface": surface.model_dump(mode="json"),
            "selected_models": ["BLACK_SCHOLES"],
        },
    )
    assert response.status_code == 422
    assert "surface point count" in response.json()["detail"]
