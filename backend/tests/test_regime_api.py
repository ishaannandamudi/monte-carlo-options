from fastapi.testclient import TestClient

from backend.app.core.config import get_settings
from backend.app.main import app

client = TestClient(app)


def parameter_payload() -> dict[str, object]:
    return {
        "num_states": 2,
        "feature_dimension": 1,
        "covariance_type": "DIAGONAL",
        "initial_probabilities": [0.5, 0.5],
        "transition_matrix": [[0.95, 0.05], [0.1, 0.9]],
        "means": [[-1.0], [1.0]],
        "covariances": [[[0.25]], [[0.36]]],
    }


def dataset_payload() -> dict[str, object]:
    observations = [[-1.1], [-0.9], [-1.2], [-0.7], [0.8], [1.2], [0.9], [1.1]] * 5
    return {"observations": observations, "feature_names": ["return"]}


def test_regime_endpoints() -> None:
    fit_body = {
        "dataset": dataset_payload(),
        "config": {"num_states": 2, "num_starts": 1, "max_iterations": 30, "seed": 7},
    }
    first = client.post("/api/regime/hmm/fit", json=fit_body)
    second = client.post("/api/regime/hmm/fit", json=fit_body)
    assert first.status_code == 200, first.text
    assert first.json()["parameters"] == second.json()["parameters"]
    parameters = first.json()["parameters"]

    decode = client.post(
        "/api/regime/hmm/decode",
        json={
            "dataset": dataset_payload(),
            "parameters": parameters,
            "include_probabilities": True,
            "probability_stride": 4,
        },
    )
    assert decode.status_code == 200, decode.text
    assert len(decode.json()["state_sequence"]) == 40
    assert len(decode.json()["probability_series"]) <= 11

    forecast = client.post(
        "/api/regime/hmm/forecast",
        json={"parameters": parameters, "current_probabilities": [0.4, 0.6], "horizons": [1, 5]},
    )
    assert forecast.status_code == 200, forecast.text

    filtered = client.post(
        "/api/regime/hmm/filter",
        json={"parameters": parameters, "prior_probabilities": [0.4, 0.6], "observation": [0.2]},
    )
    assert filtered.status_code == 200, filtered.text

    model_order = client.post(
        "/api/regime/hmm/model-order",
        json={
            "dataset": dataset_payload(),
            "state_counts": [2, 3],
            "config": {"num_starts": 1, "max_iterations": 15, "seed": 4},
        },
    )
    assert model_order.status_code == 200, model_order.text

    markov = client.post(
        "/api/regime/markov/diagnostics",
        json={
            "chain": {"transition_matrix": [[0.9, 0.1], [0.2, 0.8]]},
            "horizons": [1, 3],
            "current_probabilities": [0.5, 0.5],
        },
    )
    assert markov.status_code == 200, markov.text

    risk = client.post(
        "/api/regime/risk",
        json={
            "parameters": parameter_payload(),
            "state_probabilities": [0.8, 0.2],
            "factor_ids": ["return"],
            "delta_exposures": [100.0],
            "num_simulations": 10_000,
            "seed": 9,
        },
    )
    assert risk.status_code == 200, risk.text
    assert risk.json()["mixture_expected_shortfall"] >= risk.json()["mixture_value_at_risk"]


def test_validation_and_resource_limits(monkeypatch: object) -> None:
    invalid = client.post(
        "/api/regime/markov/diagnostics",
        json={"chain": {"transition_matrix": [[0.8, 0.3], [0.2, 0.8]]}},
    )
    assert invalid.status_code == 422
    monkeypatch.setenv("MAX_HMM_OBSERVATIONS", "10")  # type: ignore[attr-defined]
    monkeypatch.setenv("MAX_REGIME_RISK_SIMULATIONS", "100")  # type: ignore[attr-defined]
    get_settings.cache_clear()
    try:
        fit = client.post(
            "/api/regime/hmm/fit",
            json={"dataset": dataset_payload(), "config": {"num_states": 2}},
        )
        assert fit.status_code == 422
        risk = client.post(
            "/api/regime/risk",
            json={
                "parameters": parameter_payload(),
                "state_probabilities": [0.5, 0.5],
                "factor_ids": ["return"],
                "delta_exposures": [1.0],
                "num_simulations": 101,
            },
        )
        assert risk.status_code == 422
    finally:
        get_settings.cache_clear()


def test_probability_series_is_bounded(monkeypatch: object) -> None:
    monkeypatch.setenv("MAX_HMM_PROBABILITY_OUTPUT_POINTS", "3")  # type: ignore[attr-defined]
    get_settings.cache_clear()
    try:
        response = client.post(
            "/api/regime/hmm/fit",
            json={
                "dataset": dataset_payload(),
                "config": {
                    "num_states": 2,
                    "num_starts": 1,
                    "max_iterations": 10,
                    "return_probability_series": True,
                },
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["probability_series_truncated"] is True
        assert len(response.json()["probability_series"]) == 3
    finally:
        get_settings.cache_clear()
