"""Bounded recursive differences, run comparison, and reproducibility metrics."""

from __future__ import annotations

import math
from typing import Any

from backend.app.core.exceptions import ExperimentError
from backend.app.models.experiments import (
    ConfigDifference,
    ExperimentComparisonRequest,
    ExperimentComparisonResult,
    ExperimentMetric,
    ExperimentRun,
    MetricDifference,
    NumericalTolerance,
    PairwiseRunComparison,
    RegressionThresholds,
    ResearchFlag,
)
from backend.app.research.signatures import normalize_json
from backend.app.research.store import ExperimentStore


def recursive_json_diff(left: Any, right: Any, *, limit: int = 200) -> list[ConfigDifference]:
    """Return deterministic leaf differences without emitting unbounded raw structures."""

    if limit <= 0:
        raise ExperimentError("difference limit must be positive")
    differences: list[ConfigDifference] = []

    def walk(a: Any, b: Any, path: str) -> None:
        if len(differences) >= limit:
            return
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                child_path = f"{path}.{key}" if path else key
                if key not in a:
                    differences.append(ConfigDifference(path=child_path, left=None, right=b[key]))
                elif key not in b:
                    differences.append(ConfigDifference(path=child_path, left=a[key], right=None))
                else:
                    walk(a[key], b[key], child_path)
            return
        if isinstance(a, list) and isinstance(b, list):
            for index in range(max(len(a), len(b))):
                child_path = f"{path}[{index}]"
                if index >= len(a):
                    differences.append(ConfigDifference(path=child_path, left=None, right=b[index]))
                elif index >= len(b):
                    differences.append(ConfigDifference(path=child_path, left=a[index], right=None))
                else:
                    walk(a[index], b[index], child_path)
                if len(differences) >= limit:
                    break
            return
        if a != b:
            differences.append(ConfigDifference(path=path or "$", left=a, right=b))

    walk(normalize_json(left), normalize_json(right), "")
    return differences[:limit]


def metric_differences(
    baseline: list[ExperimentMetric], comparison: list[ExperimentMetric]
) -> list[MetricDifference]:
    left = {metric.name: metric.value for metric in baseline}
    right = {metric.name: metric.value for metric in comparison}
    result = []
    for name in sorted(set(left) | set(right)):
        left_value = left.get(name)
        right_value = right.get(name)
        if left_value is None or right_value is None:
            absolute = relative = None
        else:
            absolute = right_value - left_value
            relative = None if left_value == 0.0 else absolute / abs(left_value)
        result.append(
            MetricDifference(
                name=name,
                baseline_value=left_value,
                comparison_value=right_value,
                absolute_difference=absolute,
                relative_difference=relative,
            )
        )
    return result


def _flags(
    baseline: ExperimentRun,
    comparison: ExperimentRun,
    metrics: list[MetricDifference],
    thresholds: RegressionThresholds,
) -> list[ResearchFlag]:
    flags: list[ResearchFlag] = []
    performance_names = {
        metric.name
        for metric in [*baseline.metrics, *comparison.metrics]
        if metric.category == "performance"
    }
    if (
        thresholds.max_runtime_regression_pct is not None
        and baseline.engine_runtime_ms is not None
        and comparison.engine_runtime_ms is not None
        and baseline.engine_runtime_ms > 0.0
    ):
        percentage = (
            (comparison.engine_runtime_ms - baseline.engine_runtime_ms)
            / baseline.engine_runtime_ms
            * 100.0
        )
        if percentage > thresholds.max_runtime_regression_pct:
            flags.append(ResearchFlag.RUNTIME_REGRESSION)
    if thresholds.max_metric_change is not None and any(
        item.absolute_difference is not None
        and item.name not in performance_names
        and abs(item.absolute_difference) > thresholds.max_metric_change
        for item in metrics
    ):
        flags.append(ResearchFlag.RESULT_CHANGED)
    if thresholds.max_error_increase is not None and any(
        "error" in item.name.lower()
        and item.name not in performance_names
        and item.absolute_difference is not None
        and item.absolute_difference > thresholds.max_error_increase
        for item in metrics
    ):
        flags.append(ResearchFlag.ERROR_INCREASED)
    return flags


def compare_run_pair(
    baseline: ExperimentRun,
    comparison: ExperimentRun,
    *,
    thresholds: RegressionThresholds,
    max_differences: int,
) -> PairwiseRunComparison:
    metric_delta = metric_differences(baseline.metrics, comparison.metrics)
    runtime_difference = (
        None
        if baseline.engine_runtime_ms is None or comparison.engine_runtime_ms is None
        else comparison.engine_runtime_ms - baseline.engine_runtime_ms
    )
    runtime_percentage = (
        None
        if runtime_difference is None
        or baseline.engine_runtime_ms is None
        or baseline.engine_runtime_ms == 0.0
        else runtime_difference / baseline.engine_runtime_ms * 100.0
    )
    return PairwiseRunComparison(
        baseline_run_id=baseline.run_id,
        comparison_run_id=comparison.run_id,
        config_differences=recursive_json_diff(
            baseline.config_snapshot,
            comparison.config_snapshot,
            limit=max_differences,
        ),
        metric_differences=metric_delta,
        runtime_difference_ms=runtime_difference,
        runtime_difference_pct=runtime_percentage,
        environment_differences=recursive_json_diff(
            baseline.environment_snapshot,
            comparison.environment_snapshot,
            limit=max_differences,
        ),
        result_differences=recursive_json_diff(
            baseline.result_summary,
            comparison.result_summary,
            limit=max_differences,
        ),
        flags=_flags(baseline, comparison, metric_delta, thresholds),
    )


def compare_runs(
    store: ExperimentStore, request: ExperimentComparisonRequest
) -> ExperimentComparisonResult:
    runs = [store.get_run(run_id) for run_id in request.run_ids]
    baseline = runs[0]
    return ExperimentComparisonResult(
        baseline_run_id=baseline.run_id,
        comparisons=[
            compare_run_pair(
                baseline,
                run,
                thresholds=request.thresholds,
                max_differences=request.max_differences,
            )
            for run in runs[1:]
        ],
    )


def reproducible_metrics(
    baseline: ExperimentRun,
    replay: ExperimentRun,
    tolerance: NumericalTolerance,
) -> tuple[bool, list[MetricDifference], list[str]]:
    differences = metric_differences(baseline.metrics, replay.metrics)
    diagnostics: list[str] = []
    matches = True
    baseline_by_name = {metric.name: metric for metric in baseline.metrics}
    replay_by_name = {metric.name: metric for metric in replay.metrics}
    names = sorted(set(baseline_by_name) | set(replay_by_name))
    for name in names:
        left = baseline_by_name.get(name)
        right = replay_by_name.get(name)
        if left is None or right is None:
            matches = False
            diagnostics.append(f"metric presence differs: {name}")
            continue
        if left.category == "performance" or right.category == "performance":
            continue
        if not math.isclose(
            left.value,
            right.value,
            rel_tol=tolerance.relative,
            abs_tol=tolerance.absolute,
        ):
            matches = False
            diagnostics.append(f"metric differs beyond tolerance: {name}")
    return matches, differences, diagnostics
