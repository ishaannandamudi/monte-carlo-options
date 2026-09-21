"""Experiment execution, lineage, sweep, comparison, and benchmark tests."""

from pathlib import Path

import pytest

import backend.app.research.runner as runner_module
from backend.app.core.exceptions import ExperimentError
from backend.app.models.experiments import (
    ExperimentBenchmarkRequest,
    ExperimentComparisonRequest,
    ExperimentMetric,
    ExperimentRunStatus,
    ExperimentSweepRequest,
    NumericalTolerance,
    RegressionThresholds,
    ReproduceRequest,
    ResearchFlag,
    SweepAxis,
)
from backend.app.research.compare import compare_runs, recursive_json_diff
from backend.app.research.runner import DispatchOutput
from backend.tests.experiment_test_helpers import (
    bsm_definition,
    experiment_runner,
    mc_definition,
    noop_definition,
)


def metric(run: object, name: str) -> float:
    values = {item.name: item.value for item in run.metrics}  # type: ignore[attr-defined]
    return values[name]


def test_deterministic_bsm_run_persists_separated_runtime_and_metrics(tmp_path: Path) -> None:
    runner = experiment_runner(tmp_path)

    run = runner.run(bsm_definition())

    assert run.status is ExperimentRunStatus.COMPLETED
    assert metric(run, "price") == pytest.approx(8.827321225352122)
    assert metric(run, "engine_runtime_ms") >= 0.0
    assert run.result_summary == {
        "price": pytest.approx(8.827321225352122),
        "model": "BLACK_SCHOLES",
    }
    assert run.duration_ms is not None
    assert run.engine_runtime_ms is not None
    assert run.tracking_overhead_ms is not None
    assert run.duration_ms == pytest.approx(run.engine_runtime_ms + run.tracking_overhead_ms)
    assert runner.store.get_run(run.run_id) == run


def test_seeded_monte_carlo_reproduces_exact_nonperformance_metrics(tmp_path: Path) -> None:
    runner = experiment_runner(tmp_path)
    original = runner.run(mc_definition(simulations=4_096))

    replay = runner.reproduce(
        original.run_id,
        ReproduceRequest(tolerance=NumericalTolerance(absolute=0.0, relative=0.0)),
    )
    replay_run = runner.store.get_run(replay.replay_run_id)

    assert original.status is ExperimentRunStatus.COMPLETED
    assert replay.signature_match
    assert replay.result_match
    assert replay.diagnostics == []
    assert replay_run.parent_run_id == original.run_id
    assert metric(original, "price") == metric(replay_run, "price")
    assert metric(original, "standard_error") == metric(replay_run, "standard_error")


def test_controlled_execution_failure_is_recorded_without_traceback(tmp_path: Path) -> None:
    runner = experiment_runner(tmp_path)
    failed = runner.run(noop_definition().model_copy(update={"operation": "FAIL"}))

    assert failed.status is ExperimentRunStatus.FAILED
    assert failed.failure is not None
    assert failed.failure.failure_type == "ExperimentError"
    assert failed.failure.message == "controlled experiment failure"
    assert "Traceback" not in failed.failure.message
    assert failed.result_summary is None


def test_stochastic_repetitions_use_seedsequence_and_aggregate_metrics(tmp_path: Path) -> None:
    runner = experiment_runner(tmp_path)

    parent = runner.run(mc_definition(seed=19, repetitions=3, simulations=512))

    assert parent.status is ExperimentRunStatus.COMPLETED
    assert len(parent.seed_lineage) == 3
    assert len(set(parent.seed_lineage)) == 3
    assert parent.result_summary is not None
    child_ids = parent.result_summary["child_run_ids"]
    children = [runner.store.get_run(run_id) for run_id in child_ids]
    assert [child.parent_run_id for child in children] == [parent.run_id] * 3
    assert [child.root_seed for child in children] == parent.seed_lineage
    prices = [metric(child, "price") for child in children]
    assert metric(parent, "price.mean") == pytest.approx(sum(prices) / 3)
    assert metric(parent, "price.standard_error") >= 0.0
    assert all(item.category == "aggregation" for item in parent.metrics)


def test_cartesian_sweep_is_ordered_bounded_and_linked(tmp_path: Path) -> None:
    runner = experiment_runner(tmp_path, max_experiment_runs_per_sweep=4)
    request = ExperimentSweepRequest(
        definition=bsm_definition(),
        axes=[
            SweepAxis(path="contract.strike", values=[90.0, 110.0]),
            SweepAxis(path="market.volatility", values=[0.1, 0.3]),
        ],
    )

    result = runner.sweep(request)

    assert result.parent.status is ExperimentRunStatus.COMPLETED
    assert len(result.children) == 4
    assert [child.parent_run_id for child in result.children] == [result.parent.run_id] * 4
    inputs = [child.definition.input_payload for child in result.children]
    assert [(item["contract"]["strike"], item["market"]["volatility"]) for item in inputs] == [
        (90.0, 0.1),
        (90.0, 0.3),
        (110.0, 0.1),
        (110.0, 0.3),
    ]
    assert result.parent.result_summary is not None
    assert result.parent.result_summary["combinations"] == 4
    with pytest.raises(ExperimentError, match="run limit"):
        runner.sweep(
            ExperimentSweepRequest(
                definition=bsm_definition(),
                axes=[SweepAxis(path="contract.strike", values=[80, 90, 100, 110, 120])],
            )
        )
    failed = runner.sweep(
        ExperimentSweepRequest(
            definition=bsm_definition(),
            axes=[SweepAxis(path="contract.strike", values=["invalid"])],
        )
    )
    assert failed.parent.status is ExperimentRunStatus.FAILED
    assert failed.children[0].status is ExperimentRunStatus.FAILED


def test_sweep_root_seed_derives_deterministic_child_streams(tmp_path: Path) -> None:
    first = experiment_runner(tmp_path / "first")
    second = experiment_runner(tmp_path / "second")
    request = ExperimentSweepRequest(
        definition=mc_definition(seed=123, simulations=256),
        axes=[SweepAxis(path="contract.strike", values=[90.0, 100.0, 110.0])],
    )

    one = first.sweep(request)
    two = second.sweep(request)

    assert one.parent.seed_lineage == two.parent.seed_lineage
    assert [child.root_seed for child in one.children] == one.parent.seed_lineage
    assert [metric(child, "price") for child in one.children] == [
        metric(child, "price") for child in two.children
    ]


def test_root_seed_can_be_an_explicit_sweep_axis(tmp_path: Path) -> None:
    runner = experiment_runner(tmp_path)
    result = runner.sweep(
        ExperimentSweepRequest(
            definition=mc_definition(seed=1, simulations=128),
            axes=[SweepAxis(path="root_seed", values=[7, 11])],
        )
    )

    assert [child.root_seed for child in result.children] == [7, 11]
    assert result.parent.seed_lineage == [7, 11]


def test_sweep_parent_can_be_reproduced_with_same_aggregate_metrics(tmp_path: Path) -> None:
    runner = experiment_runner(tmp_path)
    original = runner.sweep(
        ExperimentSweepRequest(
            definition=bsm_definition(),
            axes=[SweepAxis(path="contract.strike", values=[90.0, 100.0])],
        )
    )

    reproduction = runner.reproduce(
        original.parent.run_id,
        ReproduceRequest(tolerance=NumericalTolerance(absolute=0.0, relative=0.0)),
    )

    assert reproduction.signature_match
    assert reproduction.result_match
    assert runner.store.get_run(reproduction.replay_run_id).parent_run_id == original.parent.run_id


def test_benchmark_warmups_are_excluded_and_baseline_flags_are_threshold_driven(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = experiment_runner(tmp_path)
    target = noop_definition(value=2.0)
    output = DispatchOutput(
        result={"value": 2.0},
        metrics=[ExperimentMetric(name="value", value=2.0)],
    )
    monkeypatch.setattr(runner.dispatcher, "dispatch", lambda _: output)
    clock = iter([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 11.0, 14.0, 15.0, 18.0, 20.0])
    monkeypatch.setattr(runner_module, "perf_counter", lambda: next(clock))
    baseline = runner.benchmark(
        ExperimentBenchmarkRequest(
            target=target,
            operation="noop baseline",
            workload_size=1,
            warmup_count=0,
            repetition_count=2,
        )
    )
    comparison = runner.benchmark(
        ExperimentBenchmarkRequest(
            target=target,
            operation="noop comparison",
            workload_size=1,
            warmup_count=0,
            repetition_count=2,
            baseline_run_id=baseline.run_id,
            thresholds=RegressionThresholds(max_runtime_regression_pct=50.0),
        )
    )

    assert baseline.result_summary is not None
    assert baseline.result_summary["timing_samples_ms"] == [1_000.0, 1_000.0]
    assert baseline.engine_runtime_ms == 2_000.0
    assert comparison.result_summary is not None
    assert comparison.result_summary["timing_samples_ms"] == [3_000.0, 3_000.0]
    assert comparison.result_summary["baseline_comparison"][
        "median_runtime_difference_ms"
    ] == pytest.approx(2_000.0)
    assert comparison.result_summary["baseline_comparison"][
        "median_runtime_difference_pct"
    ] == pytest.approx(200.0)
    assert comparison.baseline_run_id == baseline.run_id
    assert comparison.flags == [ResearchFlag.RUNTIME_REGRESSION]


def test_recursive_diff_and_run_comparison_are_bounded_and_deterministic(tmp_path: Path) -> None:
    differences = recursive_json_diff(
        {"simulation": {"num_paths": 100, "seed": 1}, "method": "MC"},
        {"simulation": {"num_paths": 200, "seed": 1}, "method": "QMC"},
        limit=1,
    )
    assert [item.path for item in differences] == ["method"]
    runner = experiment_runner(tmp_path)
    baseline = runner.run(bsm_definition(strike=90.0))
    changed = runner.run(bsm_definition(strike=110.0))

    comparison = compare_runs(
        runner.store,
        ExperimentComparisonRequest(
            run_ids=[baseline.run_id, changed.run_id],
            thresholds=RegressionThresholds(max_metric_change=0.1),
        ),
    ).comparisons[0]

    assert any(
        item.path == "input_payload.contract.strike" for item in comparison.config_differences
    )
    assert any(item.name == "price" for item in comparison.metric_differences)
    assert ResearchFlag.RESULT_CHANGED in comparison.flags


def test_reproduction_override_reports_signature_and_result_mismatch(tmp_path: Path) -> None:
    runner = experiment_runner(tmp_path)
    original = runner.run(bsm_definition(strike=100.0))

    replay = runner.reproduce(
        original.run_id,
        ReproduceRequest(override_definition=bsm_definition(strike=120.0)),
    )

    assert not replay.signature_match
    assert not replay.result_match
    assert ResearchFlag.REPRODUCIBILITY_MISMATCH in replay.flags
    assert "input signature differs" in replay.diagnostics


def test_benchmark_records_can_be_reproduced_with_lineage(tmp_path: Path) -> None:
    runner = experiment_runner(tmp_path)
    benchmark = runner.benchmark(
        ExperimentBenchmarkRequest(
            target=noop_definition(value=3.0),
            operation="replayable no-op",
            workload_size=1,
            warmup_count=1,
            repetition_count=2,
        )
    )

    replay = runner.reproduce(
        benchmark.run_id,
        ReproduceRequest(tolerance=NumericalTolerance(absolute=0.0, relative=0.0)),
    )

    assert replay.signature_match
    assert replay.result_match
    assert runner.store.get_run(replay.replay_run_id).parent_run_id == benchmark.run_id


def test_benchmark_error_threshold_is_explicit_and_ignores_timing_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = experiment_runner(tmp_path)
    outputs = iter(
        [
            DispatchOutput(
                result={"absolute_error": 1.0},
                metrics=[ExperimentMetric(name="absolute_error", value=1.0, category="error")],
            ),
            DispatchOutput(
                result={"absolute_error": 2.0},
                metrics=[ExperimentMetric(name="absolute_error", value=2.0, category="error")],
            ),
        ]
    )
    monkeypatch.setattr(runner.dispatcher, "dispatch", lambda _: next(outputs))
    baseline = runner.benchmark(
        ExperimentBenchmarkRequest(
            target=noop_definition(),
            operation="baseline error",
            workload_size=1,
            warmup_count=0,
            repetition_count=1,
        )
    )
    comparison = runner.benchmark(
        ExperimentBenchmarkRequest(
            target=noop_definition(),
            operation="increased error",
            workload_size=1,
            warmup_count=0,
            repetition_count=1,
            baseline_run_id=baseline.run_id,
            thresholds=RegressionThresholds(max_error_increase=0.5),
        )
    )

    assert comparison.flags == [ResearchFlag.ERROR_INCREASED]


def test_oversized_result_is_summarized_without_persisting_bulk_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = experiment_runner(tmp_path, max_experiment_result_bytes=220)
    monkeypatch.setattr(
        runner.dispatcher,
        "dispatch",
        lambda _: DispatchOutput(
            result={"price": 1.0, "raw_paths": list(range(1_000))},
            metrics=[ExperimentMetric(name="price", value=1.0)],
        ),
    )

    run = runner.run(noop_definition())

    assert run.status is ExperimentRunStatus.COMPLETED
    assert run.result_summary is not None
    assert run.result_summary["summary_truncated"] is True
    assert "raw_paths" not in run.result_summary
    assert run.result_summary["scalar_fields"] == {"price": 1.0}


def test_definition_resource_limits_are_enforced_before_creating_a_run(tmp_path: Path) -> None:
    runner = experiment_runner(
        tmp_path,
        max_experiment_repetitions=2,
        max_experiment_notes_length=3,
        max_experiment_tags=1,
    )

    with pytest.raises(ExperimentError, match="repetitions"):
        runner.run(mc_definition(repetitions=3))
    with pytest.raises(ExperimentError, match="notes"):
        runner.run(bsm_definition(tags=["one"]).model_copy(update={"notes": "long"}))
    with pytest.raises(ExperimentError, match="tag count"):
        runner.run(bsm_definition(tags=["one", "two"]))
    assert runner.store.list_runs(page=1, page_size=10).total == 0
