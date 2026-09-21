# Phase 21 experiment-tracking benchmark

Measured on 2026-09-20 on the local Apple-arm64 development host under macOS 26.6.2, Python
3.12.14, and the repository virtual environment. The SQLite database and artifact directory were
new temporary local directories. Values are descriptive host measurements, not service-level
guarantees.

## Method

- Timings use `time.perf_counter`.
- Signature latency is the mean of 10,000 hashes of one BSM experiment definition.
- Persistence creates, starts, and completes 100 no-op records with one result scalar and metric.
- Metric-finalization time measures the atomic metric insert plus final run update and retrieval.
- Retrieval repeats one completed-run lookup 1,000 times.
- Listing returns a page of 100 completed records and is repeated 25 times.
- Comparison loads and compares two completed records 100 times.
- Reproduction replays a completed no-op through the full runner, including environment/Git capture.
- Sweep timing creates one parent and ten no-op children, including environment/Git capture and
  SQLite persistence for every child.
- Benchmark orchestration performs two unmeasured no-op warmups and ten measured no-op repetitions,
  then persists timing statistics.
- Financial-engine work is deliberately absent from the persistence tests. The replay, sweep, and
  benchmark tests use the `NOOP` dispatcher operation so their wall time represents orchestration.

## Results

| Operation | Local result |
|---|---:|
| Canonical SHA-256 signature | 35.86 µs/signature |
| Completed-run persistence, one metric | 4.73 ms mean, 5.20 ms p95 |
| Atomic finalization and metric insert | 1.54 ms mean, 1.68 ms p95 |
| Completed-run retrieval | 0.59 ms mean |
| 100-run listing | 5.96 ms mean |
| Two-run comparison | 1.19 ms mean |
| Seeded no-op reproduction | 52.22 ms |
| Ten-child no-op sweep | 577.85 ms total |
| Two-warmup/ten-repeat no-op benchmark | 54.18 ms total |

The recorded no-op benchmark separated `0.028 ms` of measured target-engine time from `52.19 ms`
of tracking overhead. Warmup time is retained in the result summary and excluded from both reported
values. Most full-run orchestration cost on this host comes from creating multiple transactional
records and collecting best-effort Git metadata, rather than canonical hashing or SQLite reads.

## Interpretation

Single-run persistence and reads remain in the low-millisecond range. Signature generation is
small relative to even a local database transaction. Listing and comparison remain bounded by API
page/difference limits. A ten-point sweep costs about 58 ms per child on this working tree because
each child receives an independent environment snapshot and durable lifecycle record.

The opt-in `backend/tests/test_experiment_performance.py` check uses deliberately generous ceilings
to detect large regressions without turning ordinary machine load into a flaky failure. It covers
signature generation, completed-run/metric persistence, retrieval, a 100-run listing, comparison,
replay, sweep orchestration, and benchmark orchestration. Ordinary pytest excludes that marker.

## Scope

These figures isolate Phase 21 tracking overhead. Heston calibration, portfolio risk, HMM fitting,
QMC, Monte Carlo, and other financial engines retain their own phase benchmark reports. SQLite WAL
supports normal local concurrent access; these results do not represent distributed storage,
network filesystems, or multi-process worker throughput. Phase 21 adds none of those systems.
