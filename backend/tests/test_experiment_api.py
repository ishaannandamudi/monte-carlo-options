"""FastAPI coverage for the synchronous Phase 21 research workflow."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.api.experiments import get_experiment_store
from backend.app.core.config import get_settings
from backend.app.main import app
from backend.app.research.store import ExperimentStore
from backend.tests.experiment_test_helpers import (
    bsm_definition,
    experiment_settings,
    noop_definition,
)


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    settings = experiment_settings(
        tmp_path,
        max_experiment_runs_per_sweep=4,
        max_experiment_page_size=3,
        max_benchmark_repetitions=5,
        max_experiment_result_bytes=500,
    )
    store = ExperimentStore(
        settings.experiment_db_path,
        settings.experiment_artifact_dir,
        max_artifacts_per_run=settings.max_experiment_artifacts,
        max_artifact_bytes=settings.max_experiment_artifact_bytes,
        max_notes_length=settings.max_experiment_notes_length,
        max_tags_per_run=settings.max_experiment_tags,
    )
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_experiment_store] = lambda: store
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def run_bsm(client: TestClient, *, strike: float = 100.0) -> dict[str, object]:
    response = client.post(
        "/api/experiments/run",
        json=bsm_definition(strike=strike).model_dump(mode="json"),
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_run_get_list_metrics_artifacts_export_and_history(client: TestClient) -> None:
    run = run_bsm(client)
    run_id = run["run_id"]

    fetched = client.get(f"/api/experiments/{run_id}")
    listing = client.get(
        "/api/experiments",
        params={
            "experiment_type": "PRICING",
            "status": "COMPLETED",
            "tag": "bsm",
            "page_size": 3,
        },
    )
    metrics = client.get(f"/api/experiments/{run_id}/metrics")
    artifacts = client.get(f"/api/experiments/{run_id}/artifacts")
    exported = client.get(f"/api/experiments/{run_id}/export")
    history = client.get(
        "/api/experiments/metrics/history", params={"name": "price", "page_size": 3}
    )

    assert fetched.status_code == listing.status_code == metrics.status_code == 200
    assert artifacts.status_code == exported.status_code == history.status_code == 200
    assert fetched.json()["run_id"] == run_id
    assert listing.json()["total"] == 1
    assert any(item["name"] == "price" for item in metrics.json())
    assert artifacts.json() == []
    assert exported.json()["schema_version"] == 1
    assert exported.json()["run"]["input_signature"] == run["input_signature"]
    assert history.json()["total"] == 1


def test_compare_and_reproduce_endpoints(client: TestClient) -> None:
    baseline = run_bsm(client, strike=90.0)
    changed = run_bsm(client, strike=110.0)

    comparison = client.post(
        "/api/experiments/compare",
        json={
            "run_ids": [baseline["run_id"], changed["run_id"]],
            "thresholds": {"max_metric_change": 0.1},
        },
    )
    reproduction = client.post(
        f"/api/experiments/{baseline['run_id']}/reproduce",
        json={"tolerance": {"absolute": 0.0, "relative": 0.0}},
    )

    assert comparison.status_code == reproduction.status_code == 200
    assert comparison.json()["comparisons"][0]["config_differences"]
    assert "RESULT_CHANGED" in comparison.json()["comparisons"][0]["flags"]
    assert reproduction.json()["signature_match"] is True
    assert reproduction.json()["result_match"] is True


def test_sweep_and_benchmark_endpoints(client: TestClient) -> None:
    sweep = client.post(
        "/api/experiments/sweep",
        json={
            "definition": bsm_definition().model_dump(mode="json"),
            "axes": [{"path": "contract.strike", "values": [90.0, 100.0]}],
        },
    )
    benchmark = client.post(
        "/api/experiments/benchmark",
        json={
            "target": noop_definition().model_dump(mode="json"),
            "operation": "no-op API benchmark",
            "workload_size": 1,
            "warmup_count": 1,
            "repetition_count": 3,
        },
    )

    assert sweep.status_code == benchmark.status_code == 200
    assert len(sweep.json()["children"]) == 2
    assert all(
        child["parent_run_id"] == sweep.json()["parent"]["run_id"]
        for child in sweep.json()["children"]
    )
    assert benchmark.json()["result_summary"]["repetition_count"] == 3
    assert len(benchmark.json()["result_summary"]["timing_samples_ms"]) == 3


def test_resource_validation_invalid_type_pagination_and_missing_run(client: TestClient) -> None:
    invalid = bsm_definition().model_dump(mode="json")
    invalid["experiment_type"] = "INVALID"
    assert client.post("/api/experiments/run", json=invalid).status_code == 422
    oversized = client.post(
        "/api/experiments/sweep",
        json={
            "definition": bsm_definition().model_dump(mode="json"),
            "axes": [{"path": "contract.strike", "values": [80, 90, 100, 110, 120]}],
        },
    )
    assert oversized.status_code == 422
    assert client.get("/api/experiments", params={"page_size": 4}).status_code == 422
    assert client.get("/api/experiments/missing").status_code == 404
    assert client.get("/api/experiments/missing/metrics").status_code == 404
    assert client.get("/api/experiments/missing/export").status_code == 404


def test_oversized_engine_result_is_reduced_before_api_return(client: TestClient) -> None:
    bsm = bsm_definition().input_payload
    definition = {
        "experiment_type": "QMC",
        "operation": "CONVERGENCE",
        "name": "bounded QMC convergence",
        "root_seed": 8,
        "input_payload": {
            "contract": bsm["contract"],
            "market": bsm["market"],
            "minimum_power": 6,
            "maximum_power": 7,
            "randomized_replicates": 2,
            "root_seed": 8,
        },
    }

    response = client.post("/api/experiments/run", json=definition)

    assert response.status_code == 200
    run = response.json()
    assert run["status"] == "COMPLETED"
    assert run["result_summary"]["summary_truncated"] is True
    assert "points" not in run["result_summary"]


def test_benchmark_and_metric_history_resource_limits(client: TestClient) -> None:
    too_many = client.post(
        "/api/experiments/benchmark",
        json={
            "target": noop_definition().model_dump(mode="json"),
            "operation": "too many repetitions",
            "workload_size": 1,
            "repetition_count": 6,
        },
    )
    history = client.get(
        "/api/experiments/metrics/history", params={"name": "price", "page_size": 4}
    )

    assert too_many.status_code == 422
    assert history.status_code == 422
