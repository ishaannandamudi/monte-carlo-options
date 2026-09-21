# Phase 21: Experiment tracking and reproducible research workflows

## 1. Purpose and reproducibility contract

Phase 21 turns the Phase 1-20 numerical services into repeatable research operations. It records
what was requested, which code and numerical environment ran it, which random streams were used,
what scalar metrics and bounded result were produced, and how the run relates to other runs. The
layer is local and synchronous. It does not change the mathematics of any wrapped engine.

A recorded run supports two related claims:

1. **Input identity:** its SHA-256 signature identifies the normalized execution inputs and seeds.
2. **Result replay:** a completed run can be executed again and its non-performance metrics checked
   with explicit absolute and relative tolerances.

These claims are scoped to the stored environment snapshot. Seeded engines promise deterministic
streams in the same supported numerical environment; floating-point results across platforms can
still require a nonzero replay tolerance.

## 2. Module boundaries

- `models/experiments.py` defines immutable public contracts, enums, filters, comparisons, sweeps,
  benchmarks, exports, and replay results.
- `research/signatures.py` owns canonical serialization, hashing, and recursive secret redaction.
- `research/environment.py` captures bounded application, Python, platform, package, Git, and safe
  environment metadata.
- `research/store.py` owns the SQLite schema, transactions, queries, artifacts, and lifecycle
  transitions.
- `research/runner.py` validates resources, dispatches existing engines, extracts metrics, creates
  repeats/sweeps, benchmarks operations, and performs replay.
- `research/compare.py` supplies bounded recursive JSON differences, metric changes, diagnostic
  thresholds, and replay comparison.
- `api/experiments.py` is a thin synchronous HTTP adapter. It contains no financial algorithms.

## 3. Definitions and supported operations

`ExperimentDefinition` contains an experiment type, operation, display name, optional description,
tags, input and configuration payloads, optional root seed, repetition count, metadata, and notes.
Pydantic rejects unknown top-level fields, non-finite model floats, invalid labels, negative seeds,
and repeated experiments without a root seed.

The dispatcher accepts these Phase 21 operations:

| Type | Operations |
|---|---|
| `PRICING` | `BLACK_SCHOLES`, `GBM_MONTE_CARLO`, `HESTON_ANALYTICAL` |
| `CONVERGENCE` | `GBM_MONTE_CARLO`, `CRR`, `HESTON`, `MERTON`, `QMC` |
| `MODEL_COMPARISON` | `COMPARE_MODELS` |
| `CALIBRATION` | `HESTON` |
| `PORTFOLIO_SCENARIO` | `SCENARIO` |
| `VAR_ES` | `PORTFOLIO_RISK` |
| `REGIME` | `HMM_FIT` |
| `QMC` | `EUROPEAN`, `ASIAN`, `BASKET`, `CONVERGENCE`, `COMPARE` |
| `BENCHMARK` | internal `NOOP` target plus benchmark records around any supported definition |

Every operation validates the existing typed Phase 1-20 request before calling its existing service.
There is no alternate implementation of pricing, calibration, portfolio risk, HMM fitting, or QMC.

## 4. Run lifecycle and failure persistence

`ExperimentRun` records identity, parent/baseline links, timestamps, status, timing partitions,
definition/config/environment snapshots, seed lineage, metrics, summary, artifacts, annotations,
flags, and an optional failure. The supported lifecycle is:

```text
PENDING -> RUNNING -> COMPLETED
                   -> FAILED
```

Cancellation is absent because synchronous execution cannot honor it safely. `complete_run` begins
an immediate transaction, inserts all metrics, writes the final summary and timing fields, changes
status, and commits. A run therefore cannot appear completed without its final records. On an
engine or bounded-summary failure, the runner persists the exception type and a message limited to
2,000 characters; it does not retain a traceback or partial numerical result.

Completed and failed numerical records cannot transition again. Library callers may update only
bounded notes and validated tags. No delete endpoint exists.

## 5. Canonical serialization and signatures

Canonical JSON recursively normalizes Pydantic models, enum values, dataclasses, paths, dates,
UTC timestamps, mappings, sequences, and sets. Mapping keys are sorted explicitly and JSON uses
compact separators with `allow_nan=false`. Sets are ordered by each member's canonical form.
Unsupported values and non-finite floats raise `ExperimentError`.

Python's shortest round-trip JSON float representation is retained. Financial inputs are not
rounded, so nearby valid values remain distinct. The signature is SHA-256 over:

```text
experiment_type
operation
input_payload
config_payload
root_seed
repetitions
```

It excludes names, descriptions, tags, notes, timestamps, and display metadata. Dict insertion
order does not change identity; a material contract, model, method, path-count, or seed change does.

## 6. Environment and secret policy

Each run records application version, Python version, platform, and cheap version lookups for
NumPy, SciPy, Pydantic, and FastAPI. Git commit and dirty state are captured with bounded subprocess
calls when a repository is available; their absence never fails a run. Environment variables use
an allowlist containing only numerical-runtime controls such as `PYTHONHASHSEED` and BLAS thread
counts.

Before hashing or persistence, any mapping key containing `KEY`, `TOKEN`, `SECRET`, `PASSWORD`, or
`CREDENTIAL` is replaced by `[REDACTED]`, recursively and case-insensitively. This covers input,
configuration, metadata, and extra environment metadata. General process environment enumeration
is deliberately absent. A credential value neither appears in the database nor distinguishes two
otherwise identical experiments.

## 7. Seed lineage and stochastic repeats

For `repetitions > 1`, `SeedSequence(root_seed).spawn(repetitions)` derives independent,
deterministic child streams. Each generated 64-bit seed is injected into recognized `seed`,
`root_seed`, and `deterministic_seed` payload fields and stored on its child. Seed values use a
lossless text encoding in SQLite because NumPy can produce unsigned 64-bit values beyond SQLite's
signed integer range. The parent stores the complete ordered lineage and child IDs.

For each non-performance metric available on completed children, repetition aggregation reports:

- count, arithmetic mean, sample standard deviation;
- minimum, maximum, and median;
- standard error `sample_std / sqrt(count)`.

If any child fails, the parent fails with a bounded child-failure record. Performance metrics are
excluded from statistical aggregation because host timing is not a stochastic estimator output.

## 8. Cartesian sweeps and hierarchy

`ExperimentSweepRequest` contains a base definition and one or more unique dotted paths with
explicit value lists. Paths default to `input_payload`; callers can explicitly address
`input_payload`, `config_payload`, or `root_seed`. Values are copied into the definition, never
executed as code. `itertools.product` gives deterministic axis-major order matching the request.

The runner calculates `Cartesian combinations * repetitions` before creating the parent and rejects
work above `MAX_EXPERIMENT_RUNS_PER_SWEEP`. With a root seed, one child seed is derived for each
combination. Every child has a foreign-key parent link. The parent stores child IDs, combination
count, and metric aggregates. Failed children fail the parent without hiding individual records.

## 9. Metrics and bounded summaries

Metrics are generic finite scalar records with name, value, optional unit, and optional category.
Names are unique within a run. Examples include price, standard error, objective, RMSE, VaR,
expected shortfall, log likelihood, BIC, convergence slope, and runtime. `engine_runtime_ms` is
always added for ordinary runs.

The complete typed engine result is first converted to JSON-compatible data. If its serialized byte
count fits `MAX_EXPERIMENT_RESULT_BYTES`, it becomes the stored summary. Otherwise the store keeps a
bounded record with original byte count, result type, and top-level scalar fields. If even that
summary cannot fit, execution fails cleanly. Raw simulation paths and state matrices are never
added by the dispatcher.

## 10. Artifact handling

Artifacts are files under `<EXPERIMENT_ARTIFACT_DIR>/<run_id>/`; SQLite holds only metadata:
artifact ID, run ID, type, relative path, SHA-256 content hash, size, description, and timestamp.
Filenames must match a conservative alphanumeric, dot, underscore, and hyphen policy, remain a
single relative component, and resolve inside the run directory. Count and byte limits are checked
before persistence. An existing filename is never overwritten.

The API currently lists artifact references. Library code can write artifacts through the store.
SQLite blob storage, arbitrary downloads, CSV generation, and automatic artifact pruning are not
part of Phase 21.

## 11. SQLite schema and integrity

The separate research database defaults to `.research/experiments.sqlite3`. An explicit
`schema_versions` table records schema version 1. The remaining tables are:

- `runs`: lifecycle, identities, links, snapshots, timings, result/failure, notes, and flags;
- `metrics`: unique `(run_id, name)` scalar values;
- `run_tags`: many-to-many-style validated labels per run;
- `artifacts`: file references and content metadata.

Parent and baseline IDs and every metric/tag/artifact use foreign keys with restrictive deletion.
Indexes cover creation time, experiment type, status, signature, tags, and metric name. Every
short-lived connection enables foreign keys and a 30-second busy timeout. Initialization enables
WAL and normal synchronous mode. This supports ordinary concurrent local FastAPI requests while
avoiding a shared mutable connection. It is not a distributed locking system.

## 12. Benchmark methodology and timing isolation

`ExperimentBenchmarkRequest` wraps any supported experiment definition. The runner executes the
requested warmups, discards their timings, then measures each requested repetition with
`time.perf_counter`. The final record stores every measured sample and mean, median, p95, minimum,
and maximum. At least one measured repetition is required, and the API cap is checked before work.

Timing fields have separate meanings:

- `engine_runtime_ms`: sum of measured target repetitions;
- `tracking_overhead_ms`: definition/signature/environment/storage overhead, excluding warmups;
- `duration_ms`: full orchestration wall time, including warmups.

Warmup duration is retained in the bounded benchmark summary. A prior benchmark can be linked as a
baseline. Runtime and target-metric flags are emitted only when the caller supplies the relevant
threshold. Threshold flags are descriptive research diagnostics.

## 13. Run comparison and regression flags

Comparison treats the first requested run as baseline. It returns configuration, environment, and
result-summary differences; all metric changes; and absolute/relative engine-runtime changes.
Recursive JSON diff sorts keys, reports list indexes, and stops at a caller-bounded difference
count. Paths resemble `input_payload.simulation.num_simulations` or `points[2].price`.

Optional thresholds produce `RUNTIME_REGRESSION`, `RESULT_CHANGED`, or `ERROR_INCREASED`. No
threshold means no automatic classification. Missing or zero baseline values are represented with
typed unavailable relative differences instead of NaN or infinity.

## 14. Reproducibility checks

Replay loads a completed definition, executes it as a child of the original, compares input
signatures, and compares all non-performance metrics with typed absolute/relative tolerances.
Performance metrics are reported but excluded from pass/fail because timings naturally vary.
Changed definitions, failed replays, missing metrics, or out-of-tolerance values add diagnostics
and `REPRODUCIBILITY_MISMATCH`. Deterministic and same-seed stochastic engines can be checked with
zero tolerance in the same environment.

## 15. Search, metric history, pagination, and export

Run listing filters by experiment type, status, exact tag, creation interval, and exact signature.
It supports deterministic newest/oldest and runtime ordering, plus named-metric ordering through a
bounded join. Every list is paginated. Metric history returns one named metric across runs in
creation order with its unit and category, also paginated.

JSON export contains schema version, export timestamp, and the complete persisted public run:
definition, normalized snapshots, seed lineage, metrics, bounded summary, failure, and artifact
references. Import is optional in the Phase 21 specification and is not implemented; no export is
treated as executable input.

## 16. HTTP API

The synchronous routes are:

```text
POST /api/experiments/run
GET  /api/experiments
GET  /api/experiments/metrics/history
POST /api/experiments/compare
POST /api/experiments/sweep
POST /api/experiments/benchmark
GET  /api/experiments/{run_id}
POST /api/experiments/{run_id}/reproduce
GET  /api/experiments/{run_id}/metrics
GET  /api/experiments/{run_id}/artifacts
GET  /api/experiments/{run_id}/export
```

Pydantic request validation and quantitative-domain failures return 4xx responses. Missing IDs
return 404. A model execution failure normally returns its persisted `FAILED` run, allowing the
caller to inspect a stable failure record without a separate background-job protocol.

## 17. Resource controls

Phase 21 adds:

| Setting | Default |
|---|---:|
| `MAX_EXPERIMENT_RUNS_PER_SWEEP` | 100 |
| `MAX_EXPERIMENT_REPETITIONS` | 32 |
| `MAX_EXPERIMENT_METRICS` | 200 |
| `MAX_EXPERIMENT_ARTIFACTS` | 20 |
| `MAX_EXPERIMENT_RESULT_BYTES` | 1,000,000 |
| `MAX_EXPERIMENT_ARTIFACT_BYTES` | 10,000,000 |
| `MAX_EXPERIMENT_NOTES_LENGTH` | 10,000 |
| `MAX_EXPERIMENT_TAGS` | 20 |
| `MAX_EXPERIMENT_PAGE_SIZE` | 100 |
| `MAX_BENCHMARK_REPETITIONS` | 100 |

`MAX_EXPERIMENT_SWEEP_RUNS` is accepted as a compatibility alias when the preferred sweep setting
is absent. Wrapped operations also enforce their established Phase 1-20 path, node, observation,
model, calibration, portfolio, risk, HMM, and QMC limits before expensive allocation.

## 18. Integration with Phases 1-20

The dispatcher calls Black-Scholes, GBM Monte Carlo, generic Monte Carlo convergence, CRR
convergence, Heston analytical and convergence services, Merton convergence, Phase 14 Heston
calibration, Phase 15 model comparison, Phase 16 portfolio scenario repricing, Phase 17 portfolio
risk, Phase 19 HMM fitting, and Phase 20 QMC pricing/comparison/convergence. Results retain the
source engine's typed semantics. Experiment signatures and lineage sit above those APIs and do not
replace their own reproducibility metadata.

## 19. Known limitations and Phase 22 boundary

- Execution is synchronous and local; there are no workers, queues, cancellation, multiprocessing,
  distributed execution, GPU dispatch, or cloud experiment service.
- SQLite and the artifact directory have no automatic retention, pruning, quota, or backup policy.
- Git metadata is best effort and package capture intentionally covers a small numerical set.
- Result summaries retain top-level scalars when truncated; detailed arrays require a separately
  bounded artifact created by explicit library code.
- JSON import and CSV export are optional and remain unimplemented.
- The dispatcher accepts a fixed operation allowlist, not arbitrary callables or code.
- Phase 21 adds no pricing model, model calibration method, or financial-risk estimator.
- No dashboard or frontend is present. Phase 22 may consume the read/search/history APIs under its
  own specification without changing Phase 21's identity and persistence contracts.
