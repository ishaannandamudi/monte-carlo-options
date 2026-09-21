"""SQLite lifecycle, integrity, artifact, and query tests for Phase 21."""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from backend.app.core.exceptions import ExperimentError, ExperimentNotFoundError
from backend.app.models.experiments import (
    EnvironmentSnapshot,
    ExperimentFailure,
    ExperimentMetric,
    ExperimentRunStatus,
    ExperimentSort,
    ExperimentType,
)
from backend.app.research.signatures import experiment_signature
from backend.app.research.store import SCHEMA_VERSION, ExperimentStore
from backend.tests.experiment_test_helpers import bsm_definition, experiment_settings


def environment() -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        application_version="0.21.0",
        python_version="3.11",
        platform="test",
        packages={"numpy": "test"},
        git_commit="a" * 40,
        git_dirty=False,
    )


def store_at(tmp_path: Path, **limits: object) -> ExperimentStore:
    settings = experiment_settings(tmp_path, **limits)
    return ExperimentStore(
        settings.experiment_db_path,
        settings.experiment_artifact_dir,
        max_artifacts_per_run=settings.max_experiment_artifacts,
        max_artifact_bytes=settings.max_experiment_artifact_bytes,
        max_notes_length=settings.max_experiment_notes_length,
        max_tags_per_run=settings.max_experiment_tags,
    )


def completed_run(store: ExperimentStore, *, strike: float = 100.0) -> str:
    definition = bsm_definition(strike=strike)
    created = store.create_run(definition, experiment_signature(definition), environment())
    store.start_run(created.run_id)
    return store.complete_run(
        created.run_id,
        result_summary={"price": strike / 10.0},
        metrics=[ExperimentMetric(name="price", value=strike / 10.0, unit="USD")],
        duration_ms=strike,
        engine_runtime_ms=strike - 1.0,
        tracking_overhead_ms=1.0,
    ).run_id


def test_schema_version_wal_and_database_reopen(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    run_id = completed_run(store)
    with store.connection() as connection:
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()[0]

    reopened = store_at(tmp_path)

    assert reopened.schema_version() == SCHEMA_VERSION
    assert journal_mode.lower() == "wal"
    assert foreign_keys == 1
    assert reopened.get_run(run_id).status is ExperimentRunStatus.COMPLETED


def test_run_transitions_are_atomic_and_terminal_records_are_immutable(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    definition = bsm_definition()
    pending = store.create_run(definition, experiment_signature(definition), environment())
    running = store.start_run(pending.run_id)

    assert pending.status is ExperimentRunStatus.PENDING
    assert running.status is ExperimentRunStatus.RUNNING
    completed = store.complete_run(
        pending.run_id,
        result_summary={"price": 10.0},
        metrics=[ExperimentMetric(name="price", value=10.0)],
        duration_ms=2.0,
        engine_runtime_ms=1.5,
        tracking_overhead_ms=0.5,
    )
    assert completed.status is ExperimentRunStatus.COMPLETED
    with pytest.raises(ExperimentError, match="running"):
        store.complete_run(
            pending.run_id,
            result_summary={"price": 99.0},
            metrics=[],
            duration_ms=1.0,
            engine_runtime_ms=1.0,
            tracking_overhead_ms=0.0,
        )
    with pytest.raises(ExperimentError, match="immutable"):
        store.fail_run(
            pending.run_id,
            ExperimentFailure(failure_type="LateFailure", message="too late"),
            duration_ms=3.0,
            engine_runtime_ms=2.0,
            tracking_overhead_ms=1.0,
        )
    assert store.get_run(pending.run_id).result_summary == {"price": 10.0}


def test_failure_is_bounded_and_persisted(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    definition = bsm_definition()
    pending = store.create_run(definition, experiment_signature(definition), environment())
    store.start_run(pending.run_id)
    failed = store.fail_run(
        pending.run_id,
        ExperimentFailure(failure_type="Controlled", message="x" * 2_000),
        duration_ms=1.0,
        engine_runtime_ms=0.5,
        tracking_overhead_ms=0.5,
    )

    assert failed.status is ExperimentRunStatus.FAILED
    assert failed.failure is not None
    assert len(failed.failure.message) == 2_000
    assert failed.completed_at is not None


def test_parent_foreign_key_tags_notes_and_search(tmp_path: Path) -> None:
    store = store_at(tmp_path, max_experiment_notes_length=12, max_experiment_tags=2)
    parent_id = completed_run(store, strike=90.0)
    definition = bsm_definition(strike=110.0, tags=["child"])
    child = store.create_run(
        definition,
        experiment_signature(definition),
        environment(),
        parent_run_id=parent_id,
    )
    store.start_run(child.run_id)
    store.complete_run(
        child.run_id,
        result_summary={"price": 4.0},
        metrics=[ExperimentMetric(name="price", value=4.0)],
        duration_ms=4.0,
        engine_runtime_ms=3.0,
        tracking_overhead_ms=1.0,
    )
    updated = store.update_annotations(child.run_id, notes="reviewed", tags=["child", "ok"])

    assert updated.parent_run_id == parent_id
    assert updated.notes == "reviewed"
    assert updated.tags == ["child", "ok"]
    assert (
        store.list_runs(
            page=1,
            page_size=1,
            experiment_type=ExperimentType.PRICING,
            status=ExperimentRunStatus.COMPLETED,
            tag="child",
        ).total
        == 1
    )
    assert (
        store.list_runs(
            page=1,
            page_size=10,
            signature=updated.input_signature,
            created_from=datetime.now(UTC) - timedelta(minutes=1),
            created_to=datetime.now(UTC) + timedelta(minutes=1),
        ).total
        == 1
    )
    assert (
        store.list_runs(
            page=1,
            page_size=10,
            sort=ExperimentSort.METRIC_DESC,
            metric_name="price",
        )
        .items[0]
        .metrics[0]
        .value
        == 9.0
    )
    with pytest.raises(ExperimentError, match="notes"):
        store.update_annotations(child.run_id, notes="x" * 13)
    with pytest.raises(ExperimentError, match="tags"):
        store.update_annotations(child.run_id, tags=["duplicate", "duplicate"])
    with pytest.raises(ExperimentNotFoundError):
        store.create_run(
            definition,
            experiment_signature(definition),
            environment(),
            parent_run_id="missing",
        )


def test_artifact_metadata_hash_limits_and_path_safety(tmp_path: Path) -> None:
    store = store_at(tmp_path, max_experiment_artifacts=1, max_experiment_artifact_bytes=8)
    run_id = completed_run(store)
    artifact = store.write_artifact(
        run_id,
        "result.json",
        b"{}",
        artifact_type="JSON",
        description="bounded result",
    )

    assert artifact.path == f"{run_id}/result.json"
    assert len(artifact.content_hash or "") == 64
    assert (tmp_path / "artifacts" / artifact.path).read_bytes() == b"{}"
    assert store.get_run(run_id).artifacts == [artifact]
    with pytest.raises(ExperimentError, match="unsafe"):
        store.write_artifact(run_id, "../escape", b"x", artifact_type="TEXT")
    with pytest.raises(ExperimentError, match="size"):
        store.write_artifact(run_id, "large.txt", b"123456789", artifact_type="TEXT")
    with pytest.raises(ExperimentError, match="count"):
        store.write_artifact(run_id, "second.txt", b"x", artifact_type="TEXT")


def test_duplicate_artifact_filename_never_overwrites_content(tmp_path: Path) -> None:
    store = store_at(tmp_path, max_experiment_artifacts=2)
    run_id = completed_run(store)
    store.write_artifact(run_id, "fixed.txt", b"first", artifact_type="TEXT")

    with pytest.raises(ExperimentError, match="already exists"):
        store.write_artifact(run_id, "fixed.txt", b"second", artifact_type="TEXT")
    assert (tmp_path / "artifacts" / run_id / "fixed.txt").read_bytes() == b"first"


def test_metric_history_paginates(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    completed_run(store, strike=90.0)
    completed_run(store, strike=100.0)
    completed_run(store, strike=110.0)

    first_page = store.metric_history("price", page=1, page_size=2)

    assert first_page.total == 3
    assert len(first_page.points) == 2
    assert store.metric_history("missing", page=1, page_size=2).total == 0


def test_independent_connections_can_write_concurrently_without_corruption(tmp_path: Path) -> None:
    store = store_at(tmp_path)

    def persist(strike: float) -> str:
        return completed_run(store, strike=strike)

    with ThreadPoolExecutor(max_workers=4) as executor:
        run_ids = list(executor.map(persist, [90.0, 95.0, 100.0, 105.0, 110.0, 115.0]))

    assert len(set(run_ids)) == 6
    assert store.list_runs(page=1, page_size=10).total == 6
    assert all(store.get_run(run_id).status is ExperimentRunStatus.COMPLETED for run_id in run_ids)
