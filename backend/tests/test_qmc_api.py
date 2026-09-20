from fastapi.testclient import TestClient

from backend.app.core.config import get_settings
from backend.app.main import app

client = TestClient(app)


def vanilla() -> dict[str, object]:
    return {
        "contract": {
            "option_type": "CALL",
            "spot": 100.0,
            "strike": 100.0,
            "maturity_years": 1.0,
        },
        "market": {"risk_free_rate": 0.03, "volatility": 0.2, "dividend_yield": 0.01},
    }


def qmc(dimension: int = 1) -> dict[str, object]:
    return {
        "method": "SCRAMBLED_SOBOL",
        "dimension": dimension,
        "num_samples": 512,
        "scramble": True,
        "seed": 7,
        "num_randomizations": 4,
        "power_of_two_policy": "REQUIRE_POWER_OF_TWO",
        "path_construction": "STANDARD",
    }


def test_pricing_endpoints_and_reproducibility() -> None:
    european_body = vanilla() | {"qmc": qmc()}
    first = client.post("/api/pricing/qmc/european", json=european_body)
    second = client.post("/api/pricing/qmc/european", json=european_body)
    assert first.status_code == 200, first.text
    assert first.json()["price"] == second.json()["price"]
    assert first.json()["standard_error"] is not None

    asian = {
        "contract": {
            "option_type": "CALL",
            "spot": 100.0,
            "strike": 100.0,
            "maturity_years": 1.0,
            "averaging_type": "GEOMETRIC",
            "averaging_schedule": {"monitoring_count": 4},
        },
        "market": vanilla()["market"],
        "qmc": qmc(4),
    }
    assert client.post("/api/pricing/qmc/asian", json=asian).status_code == 200

    basket = {
        "contract": {
            "option_type": "CALL",
            "strike": 100.0,
            "maturity_years": 1.0,
            "weights": [0.5, 0.5],
            "basket_type": "GEOMETRIC",
        },
        "market": {
            "asset_ids": ["A", "B"],
            "spots": [100.0, 105.0],
            "volatilities": [0.2, 0.25],
            "risk_free_rate": 0.03,
            "dividend_yields": [0.01, 0.02],
            "correlation_matrix": [[1.0, 0.4], [0.4, 1.0]],
        },
        "qmc": qmc(2),
    }
    response = client.post("/api/pricing/qmc/basket", json=basket)
    assert response.status_code == 200, response.text


def test_analytics_endpoints() -> None:
    diagnostics = client.post(
        "/api/analytics/qmc/sampling",
        json={"config": qmc(2), "include_points": True, "max_output_points": 3},
    )
    assert diagnostics.status_code == 200, diagnostics.text
    assert len(diagnostics.json()["first_points"]) == 3

    convergence = vanilla() | {
        "minimum_power": 7,
        "maximum_power": 9,
        "randomized_replicates": 4,
        "root_seed": 5,
    }
    response = client.post("/api/analytics/qmc/convergence", json=convergence)
    assert response.status_code == 200, response.text
    assert len(response.json()["points"]) == 9

    comparison = vanilla() | {
        "num_samples": 512,
        "randomized_replicates": 4,
        "root_seed": 5,
    }
    response = client.post("/api/analytics/qmc/compare", json=comparison)
    assert response.status_code == 200, response.text
    assert len(response.json()["points"]) == 3


def test_power_policy_validation_and_resource_limits(monkeypatch: object) -> None:
    invalid = vanilla() | {
        "qmc": qmc()
        | {"method": "SOBOL", "scramble": False, "num_randomizations": 1, "num_samples": 500}
    }
    assert client.post("/api/pricing/qmc/european", json=invalid).status_code == 422

    rounded = vanilla() | {
        "qmc": qmc()
        | {
            "method": "SOBOL",
            "scramble": False,
            "num_randomizations": 1,
            "num_samples": 500,
            "power_of_two_policy": "ROUND_UP",
        }
    }
    response = client.post("/api/pricing/qmc/european", json=rounded)
    assert response.status_code == 200, response.text
    assert response.json()["effective_samples"] == 512

    monkeypatch.setenv("MAX_QMC_SAMPLES", "100")  # type: ignore[attr-defined]
    monkeypatch.setenv("MAX_QMC_DIMENSION", "1")  # type: ignore[attr-defined]
    monkeypatch.setenv("MAX_QMC_REPLICATES", "2")  # type: ignore[attr-defined]
    monkeypatch.setenv("MAX_QMC_TOTAL_COORDINATES", "100")  # type: ignore[attr-defined]
    get_settings.cache_clear()
    try:
        assert (
            client.post("/api/pricing/qmc/european", json=vanilla() | {"qmc": qmc()}).status_code
            == 422
        )
        assert (
            client.post("/api/analytics/qmc/sampling", json={"config": qmc(2)}).status_code == 422
        )
    finally:
        get_settings.cache_clear()


def test_api_rejects_product_dimension_and_rounded_effective_sample_limit(
    monkeypatch: object,
) -> None:
    invalid_dimension = vanilla() | {"qmc": qmc(2)}
    assert client.post("/api/pricing/qmc/european", json=invalid_dimension).status_code == 422

    monkeypatch.setenv("MAX_QMC_SAMPLES", "510")  # type: ignore[attr-defined]
    monkeypatch.setenv("MAX_QMC_DIMENSION", "10")  # type: ignore[attr-defined]
    monkeypatch.setenv("MAX_QMC_REPLICATES", "10")  # type: ignore[attr-defined]
    monkeypatch.setenv("MAX_QMC_TOTAL_COORDINATES", "100000")  # type: ignore[attr-defined]
    get_settings.cache_clear()
    try:
        comparison = vanilla() | {
            "num_samples": 500,
            "randomized_replicates": 4,
            "power_of_two_policy": "ROUND_UP",
        }
        assert client.post("/api/analytics/qmc/compare", json=comparison).status_code == 422

        monkeypatch.setenv("MAX_QMC_SAMPLES", "1000")  # type: ignore[attr-defined]
        monkeypatch.setenv("MAX_QMC_TOTAL_COORDINATES", "2500")  # type: ignore[attr-defined]
        get_settings.cache_clear()
        comparison = vanilla() | {"num_samples": 512, "randomized_replicates": 4}
        assert client.post("/api/analytics/qmc/compare", json=comparison).status_code == 422
    finally:
        get_settings.cache_clear()
