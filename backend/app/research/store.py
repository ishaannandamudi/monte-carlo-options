"""Transactional SQLite persistence for immutable completed research runs."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from backend.app.core.exceptions import ExperimentError, ExperimentNotFoundError
from backend.app.models.experiments import (
    TAG_PATTERN,
    EnvironmentSnapshot,
    ExperimentArtifact,
    ExperimentDefinition,
    ExperimentFailure,
    ExperimentListResult,
    ExperimentMetric,
    ExperimentRun,
    ExperimentRunStatus,
    ExperimentSort,
    ExperimentType,
    MetricHistoryPoint,
    MetricHistoryResult,
    ResearchFlag,
)
from backend.app.research.signatures import canonical_json, redact_secrets

SCHEMA_VERSION = 1
SAFE_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")


def utc_now() -> datetime:
    return datetime.now(UTC)


def _stamp(value: datetime | None) -> str | None:
    return None if value is None else value.astimezone(UTC).isoformat()


def _loads(value: str | None, default: Any) -> Any:
    return default if value is None else json.loads(value)


class ExperimentStore:
    """Open short-lived SQLite connections so local concurrent requests remain isolated."""

    def __init__(
        self,
        database_path: str | Path,
        artifact_directory: str | Path,
        *,
        max_artifacts_per_run: int = 20,
        max_artifact_bytes: int = 10_000_000,
        max_notes_length: int = 10_000,
        max_tags_per_run: int = 20,
    ) -> None:
        self.database_path = Path(database_path)
        self.artifact_directory = Path(artifact_directory).resolve()
        self.max_artifacts_per_run = max_artifacts_per_run
        self.max_artifact_bytes = max_artifact_bytes
        self.max_notes_length = max_notes_length
        self.max_tags_per_run = max_tags_per_run
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_directory.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_versions (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    parent_run_id TEXT REFERENCES runs(run_id) ON DELETE RESTRICT,
                    baseline_run_id TEXT REFERENCES runs(run_id) ON DELETE RESTRICT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    duration_ms REAL,
                    engine_runtime_ms REAL,
                    tracking_overhead_ms REAL,
                    experiment_type TEXT NOT NULL,
                    input_signature TEXT NOT NULL,
                    definition_json TEXT NOT NULL,
                    config_snapshot_json TEXT NOT NULL,
                    environment_json TEXT NOT NULL,
                    root_seed TEXT,
                    seed_lineage_json TEXT NOT NULL,
                    result_summary_json TEXT,
                    failure_type TEXT,
                    failure_message TEXT,
                    notes TEXT,
                    flags_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS metrics (
                    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE RESTRICT,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT,
                    category TEXT,
                    UNIQUE(run_id, name)
                );
                CREATE TABLE IF NOT EXISTS run_tags (
                    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE RESTRICT,
                    tag TEXT NOT NULL,
                    PRIMARY KEY(run_id, tag)
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE RESTRICT,
                    artifact_type TEXT NOT NULL,
                    path TEXT NOT NULL,
                    content_hash TEXT,
                    size_bytes INTEGER NOT NULL,
                    description TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at);
                CREATE INDEX IF NOT EXISTS idx_runs_type ON runs(experiment_type);
                CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
                CREATE INDEX IF NOT EXISTS idx_runs_signature ON runs(input_signature);
                CREATE INDEX IF NOT EXISTS idx_run_tags_tag ON run_tags(tag);
                CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(name);
                """
            )
            connection.execute(
                "INSERT OR IGNORE INTO schema_versions(version, applied_at) VALUES (?, ?)",
                (SCHEMA_VERSION, _stamp(utc_now())),
            )
            connection.commit()

    def schema_version(self) -> int:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT MAX(version) AS version FROM schema_versions"
            ).fetchone()
        return int(row["version"])

    def create_run(
        self,
        definition: ExperimentDefinition,
        input_signature: str,
        environment: EnvironmentSnapshot,
        *,
        parent_run_id: str | None = None,
        baseline_run_id: str | None = None,
        seed_lineage: list[int] | None = None,
    ) -> ExperimentRun:
        run_id = str(uuid4())
        created = utc_now()
        safe_definition = redact_secrets(definition.model_dump(mode="json"))
        config_snapshot = redact_secrets(
            {
                "input_payload": definition.input_payload,
                "config_payload": definition.config_payload,
                "metadata": definition.metadata,
            }
        )
        with self.connection() as connection, connection:
            if parent_run_id is not None:
                self._require_run(connection, parent_run_id)
            if baseline_run_id is not None:
                self._require_run(connection, baseline_run_id)
            connection.execute(
                """
                INSERT INTO runs(
                    run_id, experiment_id, parent_run_id, baseline_run_id, status, created_at,
                    experiment_type, input_signature, definition_json, config_snapshot_json,
                    environment_json, root_seed, seed_lineage_json, notes, flags_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    input_signature,
                    parent_run_id,
                    baseline_run_id,
                    ExperimentRunStatus.PENDING.value,
                    _stamp(created),
                    definition.experiment_type.value,
                    input_signature,
                    canonical_json(safe_definition),
                    canonical_json(config_snapshot),
                    canonical_json(environment),
                    None if definition.root_seed is None else f"seed:{definition.root_seed}",
                    canonical_json(seed_lineage or []),
                    definition.notes,
                    canonical_json([]),
                ),
            )
            connection.executemany(
                "INSERT INTO run_tags(run_id, tag) VALUES (?, ?)",
                [(run_id, tag) for tag in definition.tags],
            )
        return self.get_run(run_id)

    def start_run(self, run_id: str) -> ExperimentRun:
        with self.connection() as connection, connection:
            cursor = connection.execute(
                "UPDATE runs SET status = ?, started_at = ? WHERE run_id = ? AND status = ?",
                (
                    ExperimentRunStatus.RUNNING.value,
                    _stamp(utc_now()),
                    run_id,
                    ExperimentRunStatus.PENDING.value,
                ),
            )
            if cursor.rowcount != 1:
                raise ExperimentError("only a pending run can transition to running")
        return self.get_run(run_id)

    def complete_run(
        self,
        run_id: str,
        *,
        result_summary: dict[str, Any],
        metrics: list[ExperimentMetric],
        duration_ms: float,
        engine_runtime_ms: float,
        tracking_overhead_ms: float,
        flags: list[ResearchFlag] | None = None,
    ) -> ExperimentRun:
        if len({metric.name for metric in metrics}) != len(metrics):
            raise ExperimentError("metric names must be unique within a run")
        with self.connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = self._require_run(connection, run_id)
                if row["status"] != ExperimentRunStatus.RUNNING.value:
                    raise ExperimentError("only a running run can be completed")
                connection.executemany(
                    "INSERT INTO metrics(run_id, name, value, unit, category) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [
                        (run_id, metric.name, metric.value, metric.unit, metric.category)
                        for metric in metrics
                    ],
                )
                connection.execute(
                    """
                    UPDATE runs SET status = ?, completed_at = ?, duration_ms = ?,
                        engine_runtime_ms = ?, tracking_overhead_ms = ?,
                        result_summary_json = ?, flags_json = ?
                    WHERE run_id = ?
                    """,
                    (
                        ExperimentRunStatus.COMPLETED.value,
                        _stamp(utc_now()),
                        duration_ms,
                        engine_runtime_ms,
                        tracking_overhead_ms,
                        canonical_json(result_summary),
                        canonical_json(flags or []),
                        run_id,
                    ),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return self.get_run(run_id)

    def fail_run(
        self,
        run_id: str,
        failure: ExperimentFailure,
        *,
        duration_ms: float,
        engine_runtime_ms: float | None,
        tracking_overhead_ms: float,
    ) -> ExperimentRun:
        with self.connection() as connection, connection:
            row = self._require_run(connection, run_id)
            if row["status"] not in {
                ExperimentRunStatus.PENDING.value,
                ExperimentRunStatus.RUNNING.value,
            }:
                raise ExperimentError("completed or failed runs are immutable")
            connection.execute(
                """
                UPDATE runs SET status = ?, completed_at = ?, duration_ms = ?,
                    engine_runtime_ms = ?, tracking_overhead_ms = ?, failure_type = ?,
                    failure_message = ?, flags_json = ? WHERE run_id = ?
                """,
                (
                    ExperimentRunStatus.FAILED.value,
                    _stamp(utc_now()),
                    duration_ms,
                    engine_runtime_ms,
                    tracking_overhead_ms,
                    failure.failure_type,
                    failure.message[:2_000],
                    canonical_json([ResearchFlag.NUMERICAL_FAILURE]),
                    run_id,
                ),
            )
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> ExperimentRun:
        with self.connection() as connection:
            row = self._require_run(connection, run_id)
            metrics = self._metrics(connection, run_id)
            artifacts = self._artifacts(connection, run_id)
            tags = [
                item["tag"]
                for item in connection.execute(
                    "SELECT tag FROM run_tags WHERE run_id = ? ORDER BY tag", (run_id,)
                ).fetchall()
            ]
        return self._run_model(row, metrics, artifacts, tags)

    def list_runs(
        self,
        *,
        page: int,
        page_size: int,
        experiment_type: ExperimentType | None = None,
        status: ExperimentRunStatus | None = None,
        tag: str | None = None,
        signature: str | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
        sort: ExperimentSort = ExperimentSort.NEWEST,
        metric_name: str | None = None,
    ) -> ExperimentListResult:
        clauses: list[str] = []
        parameters: list[Any] = []
        joins = ""
        if tag is not None:
            joins += " JOIN run_tags t ON t.run_id = r.run_id AND t.tag = ?"
            parameters.append(tag)
        if sort in {ExperimentSort.METRIC_ASC, ExperimentSort.METRIC_DESC}:
            if metric_name is None:
                raise ExperimentError("metric_name is required for metric sorting")
            joins += " JOIN metrics sm ON sm.run_id = r.run_id AND sm.name = ?"
            parameters.append(metric_name)
        for column, value in (
            ("r.experiment_type", None if experiment_type is None else experiment_type.value),
            ("r.status", None if status is None else status.value),
            ("r.input_signature", signature),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters.append(value)
        if created_from is not None:
            clauses.append("r.created_at >= ?")
            parameters.append(_stamp(created_from))
        if created_to is not None:
            clauses.append("r.created_at <= ?")
            parameters.append(_stamp(created_to))
        where = "" if not clauses else " WHERE " + " AND ".join(clauses)
        ordering = {
            ExperimentSort.NEWEST: "r.created_at DESC, r.run_id ASC",
            ExperimentSort.OLDEST: "r.created_at ASC, r.run_id ASC",
            ExperimentSort.RUNTIME_ASC: "r.duration_ms ASC, r.run_id ASC",
            ExperimentSort.RUNTIME_DESC: "r.duration_ms DESC, r.run_id ASC",
            ExperimentSort.METRIC_ASC: "sm.value ASC, r.run_id ASC",
            ExperimentSort.METRIC_DESC: "sm.value DESC, r.run_id ASC",
        }[sort]
        offset = (page - 1) * page_size
        with self.connection() as connection:
            total_row = connection.execute(
                f"SELECT COUNT(DISTINCT r.run_id) AS count FROM runs r{joins}{where}", parameters
            ).fetchone()
            rows = connection.execute(
                f"SELECT DISTINCT r.* FROM runs r{joins}{where} "
                f"ORDER BY {ordering} LIMIT ? OFFSET ?",
                [*parameters, page_size, offset],
            ).fetchall()
            items = [
                self._run_model(
                    row,
                    self._metrics(connection, row["run_id"]),
                    self._artifacts(connection, row["run_id"]),
                    [
                        item["tag"]
                        for item in connection.execute(
                            "SELECT tag FROM run_tags WHERE run_id = ? ORDER BY tag",
                            (row["run_id"],),
                        ).fetchall()
                    ],
                )
                for row in rows
            ]
        return ExperimentListResult(
            items=items,
            page=page,
            page_size=page_size,
            total=int(total_row["count"]),
        )

    def metric_history(self, name: str, *, page: int, page_size: int) -> MetricHistoryResult:
        offset = (page - 1) * page_size
        with self.connection() as connection:
            total = connection.execute(
                "SELECT COUNT(*) AS count FROM metrics WHERE name = ?", (name,)
            ).fetchone()
            rows = connection.execute(
                """
                SELECT m.run_id, r.created_at, m.value, m.unit, m.category
                FROM metrics m JOIN runs r ON r.run_id = m.run_id
                WHERE m.name = ? ORDER BY r.created_at ASC, m.run_id ASC LIMIT ? OFFSET ?
                """,
                (name, page_size, offset),
            ).fetchall()
        return MetricHistoryResult(
            metric_name=name,
            points=[
                MetricHistoryPoint(
                    run_id=row["run_id"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    value=row["value"],
                    unit=row["unit"],
                    category=row["category"],
                )
                for row in rows
            ],
            page=page,
            page_size=page_size,
            total=int(total["count"]),
        )

    def update_annotations(
        self, run_id: str, *, notes: str | None = None, tags: list[str] | None = None
    ) -> ExperimentRun:
        if notes is not None and len(notes) > self.max_notes_length:
            raise ExperimentError("experiment notes exceed configured limit")
        if tags is not None:
            if len(tags) > self.max_tags_per_run:
                raise ExperimentError("experiment tag count exceeds configured limit")
            if len(tags) != len(set(tags)) or any(
                not tag or len(tag) > 64 or TAG_PATTERN.fullmatch(tag) is None for tag in tags
            ):
                raise ExperimentError("experiment tags must be unique safe labels")
        with self.connection() as connection, connection:
            self._require_run(connection, run_id)
            if notes is not None:
                connection.execute("UPDATE runs SET notes = ? WHERE run_id = ?", (notes, run_id))
            if tags is not None:
                connection.execute("DELETE FROM run_tags WHERE run_id = ?", (run_id,))
                connection.executemany(
                    "INSERT INTO run_tags(run_id, tag) VALUES (?, ?)",
                    [(run_id, tag) for tag in sorted(tags)],
                )
        return self.get_run(run_id)

    def write_artifact(
        self,
        run_id: str,
        filename: str,
        content: bytes,
        *,
        artifact_type: str,
        description: str | None = None,
    ) -> ExperimentArtifact:
        if SAFE_FILENAME.fullmatch(filename) is None or Path(filename).name != filename:
            raise ExperimentError("artifact filename contains unsafe path characters")
        if len(content) > self.max_artifact_bytes:
            raise ExperimentError("artifact exceeds configured size limit")
        run = self.get_run(run_id)
        if len(run.artifacts) >= self.max_artifacts_per_run:
            raise ExperimentError("artifact count exceeds configured run limit")
        run_directory = (self.artifact_directory / run_id).resolve()
        if self.artifact_directory not in run_directory.parents:
            raise ExperimentError("artifact directory escaped configured root")
        run_directory.mkdir(parents=True, exist_ok=True)
        path = (run_directory / filename).resolve()
        if run_directory not in path.parents:
            raise ExperimentError("artifact path escaped its run directory")
        if path.exists():
            raise ExperimentError("artifact filename already exists for this run")
        path.write_bytes(content)
        artifact = ExperimentArtifact(
            artifact_id=str(uuid4()),
            run_id=run_id,
            artifact_type=artifact_type,
            path=str(path.relative_to(self.artifact_directory)),
            content_hash=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
            description=description,
            created_at=utc_now(),
        )
        with self.connection() as connection, connection:
            connection.execute(
                """
                INSERT INTO artifacts(artifact_id, run_id, artifact_type, path, content_hash,
                    size_bytes, description, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact.artifact_id,
                    artifact.run_id,
                    artifact.artifact_type,
                    artifact.path,
                    artifact.content_hash,
                    artifact.size_bytes,
                    artifact.description,
                    _stamp(artifact.created_at),
                ),
            )
        return artifact

    @staticmethod
    def _require_run(connection: sqlite3.Connection, run_id: str) -> sqlite3.Row:
        row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise ExperimentNotFoundError(f"experiment run not found: {run_id}")
        return cast(sqlite3.Row, row)

    @staticmethod
    def _metrics(connection: sqlite3.Connection, run_id: str) -> list[ExperimentMetric]:
        rows = connection.execute(
            "SELECT name, value, unit, category FROM metrics WHERE run_id = ? ORDER BY name",
            (run_id,),
        ).fetchall()
        return [ExperimentMetric(**dict(row)) for row in rows]

    @staticmethod
    def _artifacts(connection: sqlite3.Connection, run_id: str) -> list[ExperimentArtifact]:
        rows = connection.execute(
            "SELECT * FROM artifacts WHERE run_id = ? ORDER BY created_at, artifact_id", (run_id,)
        ).fetchall()
        return [ExperimentArtifact(**dict(row)) for row in rows]

    @staticmethod
    def _run_model(
        row: sqlite3.Row,
        metrics: list[ExperimentMetric],
        artifacts: list[ExperimentArtifact],
        tags: list[str],
    ) -> ExperimentRun:
        failure = (
            None
            if row["failure_type"] is None
            else ExperimentFailure(failure_type=row["failure_type"], message=row["failure_message"])
        )
        return ExperimentRun(
            run_id=row["run_id"],
            experiment_id=row["experiment_id"],
            parent_run_id=row["parent_run_id"],
            baseline_run_id=row["baseline_run_id"],
            status=row["status"],
            created_at=datetime.fromisoformat(row["created_at"]),
            started_at=(
                None if row["started_at"] is None else datetime.fromisoformat(row["started_at"])
            ),
            completed_at=(
                None if row["completed_at"] is None else datetime.fromisoformat(row["completed_at"])
            ),
            duration_ms=row["duration_ms"],
            engine_runtime_ms=row["engine_runtime_ms"],
            tracking_overhead_ms=row["tracking_overhead_ms"],
            experiment_type=row["experiment_type"],
            input_signature=row["input_signature"],
            definition=ExperimentDefinition.model_validate(_loads(row["definition_json"], {})),
            config_snapshot=_loads(row["config_snapshot_json"], {}),
            environment_snapshot=EnvironmentSnapshot.model_validate(
                _loads(row["environment_json"], {})
            ),
            root_seed=(
                None
                if row["root_seed"] is None
                else int(str(row["root_seed"]).removeprefix("seed:"))
            ),
            seed_lineage=_loads(row["seed_lineage_json"], []),
            result_summary=_loads(row["result_summary_json"], None),
            metrics=metrics,
            artifacts=artifacts,
            failure=failure,
            notes=row["notes"],
            tags=tags,
            flags=_loads(row["flags_json"], []),
        )
