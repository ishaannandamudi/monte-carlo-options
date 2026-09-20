from fastapi.testclient import TestClient

from backend.app.core.config import get_settings
from backend.app.main import app

client = TestClient(app)


def payload(option_type: str = "CALL") -> dict[str, object]:
    return {
        "contract": {
            "option_type": option_type,
            "exercise_style": "EUROPEAN",
            "spot": 100.0,
            "strike": 100.0,
            "maturity_years": 0.5,
        },
        "market": {"risk_free_rate": 0.02, "dividend_yield": 0.01, "volatility": 0.18},
        "parameters": {
            "jump_intensity": 0.8,
            "jump_mean": -0.12,
            "jump_volatility": 0.25,
        },
    }


def test_analytical_call_and_put() -> None:
    for option_type in ("CALL", "PUT"):
        response = client.post("/api/pricing/merton/analytical", json=payload(option_type))
        assert response.status_code == 200
        assert response.json()["price"] >= 0.0
        assert response.json()["converged"] is True


def test_monte_carlo_reproducibility_call_and_put() -> None:
    for option_type in ("CALL", "PUT"):
        body = payload(option_type) | {"simulation": {"num_simulations": 20_000, "seed": 43}}
        first = client.post("/api/pricing/merton/monte-carlo", json=body)
        second = client.post("/api/pricing/merton/monte-carlo", json=body)
        assert first.status_code == 200
        assert first.json()["price"] == second.json()["price"]
        assert first.json()["mean_jump_count"] == second.json()["mean_jump_count"]


def test_analytics_endpoints() -> None:
    base = payload()
    requests = (
        ("compare", base | {"simulation": {"num_simulations": 10_000, "seed": 9}}),
        ("convergence", base | {"path_counts": [1_000, 4_000], "root_seed": 9}),
        ("series-convergence", base | {"tolerances": [1e-4, 1e-10]}),
        ("smile", base | {"strikes": [80.0, 100.0, 120.0]}),
        (
            "sensitivity",
            base | {"parameter": "jump_mean", "values": [-0.2, -0.1, 0.1]},
        ),
        (
            "jump-diagnostics",
            base | {"simulation": {"num_simulations": 10_000, "seed": 9}},
        ),
        ("greeks", base),
    )
    for endpoint, body in requests:
        response = client.post(f"/api/analytics/merton/{endpoint}", json=body)
        assert response.status_code == 200, response.text


def test_invalid_parameters_and_resource_limits(monkeypatch: object) -> None:
    invalid = payload()
    invalid["parameters"] = {
        "jump_intensity": -0.1,
        "jump_mean": 0.0,
        "jump_volatility": 0.2,
    }
    assert client.post("/api/pricing/merton/analytical", json=invalid).status_code == 422
    monkeypatch.setenv("MAX_MERTON_PATHS", "100")  # type: ignore[attr-defined]
    monkeypatch.setenv("MAX_MERTON_SERIES_TERMS", "5")  # type: ignore[attr-defined]
    get_settings.cache_clear()
    try:
        mc = payload() | {"simulation": {"num_simulations": 101, "seed": 1}}
        assert client.post("/api/pricing/merton/monte-carlo", json=mc).status_code == 422
        analytical = payload() | {"series": {"max_terms": 6}}
        assert client.post("/api/pricing/merton/analytical", json=analytical).status_code == 422
    finally:
        get_settings.cache_clear()
