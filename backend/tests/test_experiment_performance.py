"""Opt-in latency safeguards for the local Phase 21 tracking layer."""

from pathlib import Path
from time import perf_counter

import pytest

from backend.app.models.experiments import (
    EnvironmentSnapshot,
    ExperimentBenchmarkRequest,
    ExperimentComparisonRequest,
    ExperimentMetric,
    ExperimentSweepRequest,
    ReproduceRequest,
    SweepAxis,
)
from backend.app.research.compare import compare_runs
from backend.app.research.signatures import experiment_signature
from backend.tests.experiment_test_helpers import (
    bsm_definition,
    experiment_runner,
    noop_definition,
)


@pytest.mark.performance
def test_tracking_operations_remain_lightweight(tmp_path: Path) -> None:
    runner = experiment_runner(tmp_path, max_experiment_runs_per_sweep=100)
    definition = bsm_definition()

    started = perf_counter()
    for _ in range(1_000):
        experiment_signature(definition)
    signature_ms = (perf_counter() - started) * 1_000.0

    environment = EnvironmentSnapshot(
        application_version="0.21.0",
        python_version="3.11",
        platform="benchmark",
        packages={},
    )
    started = perf_counter()
    run_ids: list[str] = []
    for index in range(100):
        item = noop_definition(value=float(index))
        pending = runner.store.create_run(item, experiment_signature(item), environment)
        runner.store.start_run(pending.run_id)
        completed = runner.store.complete_run(
            pending.run_id,
            result_summary={"value": float(index)},
            metrics=[ExperimentMetric(name="value", value=float(index))],
            duration_ms=1.0,
            engine_runtime_ms=0.1,
            tracking_overhead_ms=0.9,
        )
        run_ids.append(completed.run_id)
    persistence_ms = (perf_counter() - started) * 1_000.0

    started = perf_counter()
    for run_id in run_ids:
        runner.store.get_run(run_id)
    retrieval_ms = (perf_counter() - started) * 1_000.0
    started = perf_counter()
    listing = runner.store.list_runs(page=1, page_size=100)
    listing_ms = (perf_counter() - started) * 1_000.0
    started = perf_counter()
    compare_runs(
        runner.store,
        ExperimentComparisonRequest(run_ids=[run_ids[0], run_ids[-1]]),
    )
    comparison_ms = (perf_counter() - started) * 1_000.0

    tracked = runner.run(noop_definition())
    started = perf_counter()
    runner.reproduce(tracked.run_id, ReproduceRequest())
    reproduction_ms = (perf_counter() - started) * 1_000.0
    started = perf_counter()
    runner.sweep(
        ExperimentSweepRequest(
            definition=noop_definition(),
            axes=[SweepAxis(path="value", values=list(range(10)))],
        )
    )
    sweep_ms = (perf_counter() - started) * 1_000.0
    started = perf_counter()
    runner.benchmark(
        ExperimentBenchmarkRequest(
            target=noop_definition(),
            operation="tracking overhead",
            workload_size=1,
            warmup_count=2,
            repetition_count=10,
        )
    )
    benchmark_ms = (perf_counter() - started) * 1_000.0

    assert signature_ms < 1_000.0
    assert persistence_ms < 5_000.0
    assert retrieval_ms < 2_000.0
    assert listing.total >= 100 and listing_ms < 1_000.0
    assert comparison_ms < 500.0
    assert reproduction_ms < 1_000.0
    assert sweep_ms < 2_000.0
    assert benchmark_ms < 1_000.0
