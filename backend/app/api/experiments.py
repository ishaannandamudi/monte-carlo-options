"""Synchronous local APIs for Phase 21 experiment tracking and research workflows."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.app.core.config import Settings, get_settings
from backend.app.core.exceptions import ExperimentNotFoundError
from backend.app.models.experiments import (
    ExperimentArtifact,
    ExperimentBenchmarkRequest,
    ExperimentComparisonRequest,
    ExperimentComparisonResult,
    ExperimentDefinition,
    ExperimentExport,
    ExperimentListResult,
    ExperimentMetric,
    ExperimentRun,
    ExperimentRunStatus,
    ExperimentSort,
    ExperimentSweepRequest,
    ExperimentSweepResult,
    ExperimentType,
    MetricHistoryResult,
    ReproduceRequest,
    ReproducibilityResult,
)
from backend.app.research.compare import compare_runs
from backend.app.research.runner import ExperimentRunner
from backend.app.research.store import SCHEMA_VERSION, ExperimentStore, utc_now

router = APIRouter(prefix="/api/experiments", tags=["research-experiments"])


@lru_cache
def get_experiment_store() -> ExperimentStore:
    settings = get_settings()
    return ExperimentStore(
        settings.experiment_db_path,
        settings.experiment_artifact_dir,
        max_artifacts_per_run=settings.max_experiment_artifacts,
        max_artifact_bytes=settings.max_experiment_artifact_bytes,
        max_notes_length=settings.max_experiment_notes_length,
        max_tags_per_run=settings.max_experiment_tags,
    )


def _runner(store: ExperimentStore, settings: Settings) -> ExperimentRunner:
    return ExperimentRunner(store, settings)


def _not_found(exc: ExperimentNotFoundError) -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, str(exc))


@router.post("/run", response_model=ExperimentRun)
def run_experiment(
    definition: ExperimentDefinition,
    settings: Annotated[Settings, Depends(get_settings)],
    store: Annotated[ExperimentStore, Depends(get_experiment_store)],
) -> ExperimentRun:
    return _runner(store, settings).run(definition)


@router.get("", response_model=ExperimentListResult)
def list_experiments(
    settings: Annotated[Settings, Depends(get_settings)],
    store: Annotated[ExperimentStore, Depends(get_experiment_store)],
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1)] = 25,
    experiment_type: ExperimentType | None = None,
    run_status: Annotated[ExperimentRunStatus | None, Query(alias="status")] = None,
    tag: str | None = None,
    signature: str | None = None,
    created_from: datetime | None = None,
    created_to: datetime | None = None,
    sort: ExperimentSort = ExperimentSort.NEWEST,
    metric_name: str | None = None,
) -> ExperimentListResult:
    if page_size > settings.max_experiment_page_size:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, "experiment page size exceeds configured limit"
        )
    return store.list_runs(
        page=page,
        page_size=page_size,
        experiment_type=experiment_type,
        status=run_status,
        tag=tag,
        signature=signature,
        created_from=created_from,
        created_to=created_to,
        sort=sort,
        metric_name=metric_name,
    )


@router.get("/metrics/history", response_model=MetricHistoryResult)
def metric_history(
    name: str,
    settings: Annotated[Settings, Depends(get_settings)],
    store: Annotated[ExperimentStore, Depends(get_experiment_store)],
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1)] = 25,
) -> MetricHistoryResult:
    if page_size > settings.max_experiment_page_size:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "metric-history page size exceeds configured limit",
        )
    return store.metric_history(name, page=page, page_size=page_size)


@router.post("/compare", response_model=ExperimentComparisonResult)
def compare_experiments(
    request: ExperimentComparisonRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    store: Annotated[ExperimentStore, Depends(get_experiment_store)],
) -> ExperimentComparisonResult:
    if request.max_differences > settings.max_experiment_metrics:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, "comparison difference count exceeds limit"
        )
    try:
        return compare_runs(store, request)
    except ExperimentNotFoundError as exc:
        raise _not_found(exc) from exc


@router.post("/sweep", response_model=ExperimentSweepResult)
def sweep_experiments(
    request: ExperimentSweepRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    store: Annotated[ExperimentStore, Depends(get_experiment_store)],
) -> ExperimentSweepResult:
    return _runner(store, settings).sweep(request)


@router.post("/benchmark", response_model=ExperimentRun)
def benchmark_experiment(
    request: ExperimentBenchmarkRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    store: Annotated[ExperimentStore, Depends(get_experiment_store)],
) -> ExperimentRun:
    try:
        return _runner(store, settings).benchmark(request)
    except ExperimentNotFoundError as exc:
        raise _not_found(exc) from exc


@router.get("/{run_id}", response_model=ExperimentRun)
def get_experiment(
    run_id: str,
    store: Annotated[ExperimentStore, Depends(get_experiment_store)],
) -> ExperimentRun:
    try:
        return store.get_run(run_id)
    except ExperimentNotFoundError as exc:
        raise _not_found(exc) from exc


@router.post("/{run_id}/reproduce", response_model=ReproducibilityResult)
def reproduce_experiment(
    run_id: str,
    request: ReproduceRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    store: Annotated[ExperimentStore, Depends(get_experiment_store)],
) -> ReproducibilityResult:
    try:
        return _runner(store, settings).reproduce(run_id, request)
    except ExperimentNotFoundError as exc:
        raise _not_found(exc) from exc


@router.get("/{run_id}/metrics", response_model=list[ExperimentMetric])
def get_experiment_metrics(
    run_id: str,
    store: Annotated[ExperimentStore, Depends(get_experiment_store)],
) -> list[ExperimentMetric]:
    try:
        return store.get_run(run_id).metrics
    except ExperimentNotFoundError as exc:
        raise _not_found(exc) from exc


@router.get("/{run_id}/artifacts", response_model=list[ExperimentArtifact])
def get_experiment_artifacts(
    run_id: str,
    store: Annotated[ExperimentStore, Depends(get_experiment_store)],
) -> list[ExperimentArtifact]:
    try:
        return store.get_run(run_id).artifacts
    except ExperimentNotFoundError as exc:
        raise _not_found(exc) from exc


@router.get("/{run_id}/export", response_model=ExperimentExport)
def export_experiment(
    run_id: str,
    store: Annotated[ExperimentStore, Depends(get_experiment_store)],
) -> ExperimentExport:
    try:
        return ExperimentExport(
            schema_version=SCHEMA_VERSION,
            exported_at=utc_now(),
            run=store.get_run(run_id),
        )
    except ExperimentNotFoundError as exc:
        raise _not_found(exc) from exc
