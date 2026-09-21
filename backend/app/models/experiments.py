"""Typed contracts for local research experiments, lineage, and comparisons."""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

TAG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
SWEEP_PATH_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


class ExperimentModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ExperimentType(StrEnum):
    PRICING = "PRICING"
    CONVERGENCE = "CONVERGENCE"
    MODEL_COMPARISON = "MODEL_COMPARISON"
    CALIBRATION = "CALIBRATION"
    PORTFOLIO_SCENARIO = "PORTFOLIO_SCENARIO"
    VAR_ES = "VAR_ES"
    REGIME = "REGIME"
    QMC = "QMC"
    BENCHMARK = "BENCHMARK"


class ExperimentRunStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ExperimentSort(StrEnum):
    NEWEST = "NEWEST"
    OLDEST = "OLDEST"
    RUNTIME_ASC = "RUNTIME_ASC"
    RUNTIME_DESC = "RUNTIME_DESC"
    METRIC_ASC = "METRIC_ASC"
    METRIC_DESC = "METRIC_DESC"


class ResearchFlag(StrEnum):
    RUNTIME_REGRESSION = "RUNTIME_REGRESSION"
    RESULT_CHANGED = "RESULT_CHANGED"
    ERROR_INCREASED = "ERROR_INCREASED"
    REPRODUCIBILITY_MISMATCH = "REPRODUCIBILITY_MISMATCH"
    NUMERICAL_FAILURE = "NUMERICAL_FAILURE"


class ExperimentDefinition(ExperimentModel):
    experiment_type: ExperimentType
    operation: str = Field(min_length=1, max_length=80)
    name: str = Field(min_length=1, max_length=160)
    description: str | None = Field(default=None, max_length=2_000)
    tags: list[str] = Field(default_factory=list)
    input_payload: dict[str, Any] = Field(default_factory=dict)
    config_payload: dict[str, Any] = Field(default_factory=dict)
    root_seed: int | None = Field(default=None, ge=0)
    repetitions: int = Field(default=1, gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None

    @model_validator(mode="after")
    def validate_labels(self) -> Self:
        normalized = [tag.strip() for tag in self.tags]
        if len(normalized) != len(set(normalized)):
            raise ValueError("experiment tags must be unique")
        if any(
            not tag or len(tag) > 64 or TAG_PATTERN.fullmatch(tag) is None for tag in normalized
        ):
            raise ValueError("tags must be 1-64 safe alphanumeric label characters")
        object.__setattr__(self, "tags", normalized)
        object.__setattr__(self, "operation", self.operation.strip().upper())
        if self.repetitions > 1 and self.root_seed is None:
            raise ValueError("repeated experiments require root_seed for deterministic lineage")
        return self


class EnvironmentSnapshot(ExperimentModel):
    application_version: str
    python_version: str
    platform: str
    packages: dict[str, str]
    git_commit: str | None = None
    git_dirty: bool | None = None
    variables: dict[str, str] = Field(default_factory=dict)


class ExperimentMetric(ExperimentModel):
    name: str = Field(min_length=1, max_length=160)
    value: float
    unit: str | None = Field(default=None, max_length=40)
    category: str | None = Field(default=None, max_length=80)


class ExperimentFailure(ExperimentModel):
    failure_type: str = Field(min_length=1, max_length=160)
    message: str = Field(min_length=1, max_length=2_000)


class ExperimentArtifact(ExperimentModel):
    artifact_id: str
    run_id: str
    artifact_type: str = Field(min_length=1, max_length=80)
    path: str = Field(min_length=1, max_length=1_000)
    content_hash: str | None = Field(default=None, min_length=64, max_length=64)
    size_bytes: int = Field(ge=0)
    description: str | None = Field(default=None, max_length=1_000)
    created_at: datetime


class ExperimentRun(ExperimentModel):
    run_id: str
    experiment_id: str
    parent_run_id: str | None = None
    baseline_run_id: str | None = None
    status: ExperimentRunStatus
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: float | None = Field(default=None, ge=0.0)
    engine_runtime_ms: float | None = Field(default=None, ge=0.0)
    tracking_overhead_ms: float | None = Field(default=None, ge=0.0)
    experiment_type: ExperimentType
    input_signature: str = Field(min_length=64, max_length=64)
    definition: ExperimentDefinition
    config_snapshot: dict[str, Any]
    environment_snapshot: EnvironmentSnapshot
    root_seed: int | None = Field(default=None, ge=0)
    seed_lineage: list[int] = Field(default_factory=list)
    result_summary: dict[str, Any] | None = None
    metrics: list[ExperimentMetric] = Field(default_factory=list)
    artifacts: list[ExperimentArtifact] = Field(default_factory=list)
    failure: ExperimentFailure | None = None
    notes: str | None = None
    tags: list[str] = Field(default_factory=list)
    flags: list[ResearchFlag] = Field(default_factory=list)


class MetricAggregate(ExperimentModel):
    name: str
    count: int = Field(gt=0)
    mean: float
    standard_deviation: float = Field(ge=0.0)
    minimum: float
    maximum: float
    median: float
    standard_error: float = Field(ge=0.0)


class SweepAxis(ExperimentModel):
    path: str = Field(min_length=1, max_length=240)
    values: list[Any] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        if SWEEP_PATH_PATTERN.fullmatch(self.path) is None:
            raise ValueError("sweep path must contain dotted identifier components")
        if self.path.startswith("root_seed."):
            raise ValueError("root_seed sweep path must address the scalar field directly")
        return self


class ExperimentSweepRequest(ExperimentModel):
    definition: ExperimentDefinition
    axes: list[SweepAxis] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_axes(self) -> Self:
        paths = [axis.path for axis in self.axes]
        if len(paths) != len(set(paths)):
            raise ValueError("sweep paths must be unique")
        return self


class ExperimentSweepResult(ExperimentModel):
    parent: ExperimentRun
    children: list[ExperimentRun]
    metric_aggregates: list[MetricAggregate]


class NumericalTolerance(ExperimentModel):
    absolute: float = Field(default=1e-12, ge=0.0)
    relative: float = Field(default=1e-12, ge=0.0)


class RegressionThresholds(ExperimentModel):
    max_runtime_regression_pct: float | None = Field(default=None, ge=0.0)
    max_metric_change: float | None = Field(default=None, ge=0.0)
    max_error_increase: float | None = Field(default=None, ge=0.0)


class ExperimentBenchmarkRequest(ExperimentModel):
    target: ExperimentDefinition
    operation: str = Field(min_length=1, max_length=160)
    workload_size: int | str
    warmup_count: int = Field(default=1, ge=0)
    repetition_count: int = Field(default=5, gt=0)
    baseline_run_id: str | None = None
    thresholds: RegressionThresholds = Field(default_factory=RegressionThresholds)


class ConfigDifference(ExperimentModel):
    path: str
    left: Any = None
    right: Any = None


class MetricDifference(ExperimentModel):
    name: str
    baseline_value: float | None
    comparison_value: float | None
    absolute_difference: float | None
    relative_difference: float | None


class PairwiseRunComparison(ExperimentModel):
    baseline_run_id: str
    comparison_run_id: str
    config_differences: list[ConfigDifference]
    metric_differences: list[MetricDifference]
    runtime_difference_ms: float | None
    runtime_difference_pct: float | None
    environment_differences: list[ConfigDifference]
    result_differences: list[ConfigDifference]
    flags: list[ResearchFlag]


class ExperimentComparisonRequest(ExperimentModel):
    run_ids: list[str] = Field(min_length=2, max_length=20)
    thresholds: RegressionThresholds = Field(default_factory=RegressionThresholds)
    max_differences: int = Field(default=200, gt=0)


class ExperimentComparisonResult(ExperimentModel):
    baseline_run_id: str
    comparisons: list[PairwiseRunComparison]


class ReproduceRequest(ExperimentModel):
    tolerance: NumericalTolerance = Field(default_factory=NumericalTolerance)
    override_definition: ExperimentDefinition | None = None


class ReproducibilityResult(ExperimentModel):
    original_run_id: str
    replay_run_id: str
    signature_match: bool
    result_match: bool
    metric_differences: list[MetricDifference]
    runtime_difference_ms: float | None
    diagnostics: list[str]
    flags: list[ResearchFlag]


class ExperimentListResult(ExperimentModel):
    items: list[ExperimentRun]
    page: int = Field(gt=0)
    page_size: int = Field(gt=0)
    total: int = Field(ge=0)


class MetricHistoryPoint(ExperimentModel):
    run_id: str
    created_at: datetime
    value: float
    unit: str | None = None
    category: str | None = None


class MetricHistoryResult(ExperimentModel):
    metric_name: str
    points: list[MetricHistoryPoint]
    page: int = Field(gt=0)
    page_size: int = Field(gt=0)
    total: int = Field(ge=0)


class ExperimentExport(ExperimentModel):
    schema_version: int = Field(ge=1)
    exported_at: datetime
    run: ExperimentRun
