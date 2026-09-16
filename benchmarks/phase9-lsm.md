# Phase 9 validation and benchmark observations

Recorded on 2026-09-16: macOS 26.6.2 arm64, Python 3.12.14, NumPy 2.2.6, SciPy 1.18.1.
Numerical reproducibility is scoped to the same supported environment; elapsed times vary.

## Completion audit

At the continuation audit, the working tree already contained full-path GBM, LSM configuration
and policies, regression bases and diagnostics, backward fitting and forward evaluation, deterministic
cases, CRR comparisons, convergence and basis research, four API routes, resource limits, tests,
and the design document. The prior recorded checks were 535 ordinary and eight performance tests
passing. That implementation was retained.

The final continuation added optional common-date LSM/CRR boundary comparisons, stricter boundary
sample and nonfinite regression-state checks, an expiry shortcut that avoids irrelevant extreme
market arithmetic, eight focused tests, this report, and corresponding documentation. It also
verified the live API and reviewed the complete diff.

Final checks:

- Ordinary tests: **543 passed**, eight performance tests deselected, 2.18 seconds.
- Opt-in performance tests: **8 passed**, 543 ordinary tests deselected, 1.37 seconds.
- The original 453 ordinary and four performance tests from Phases 1–8 remain unchanged and pass.
- Ruff check and formatting check: passed; 123 Python files already formatted.
- mypy: passed, 65 application source files.
- Live Uvicorn startup, `/health`, and American-put pricing: successful HTTP 200 responses.
- No Phase 10 functionality was added.

Reproduce the checks from the repository root:

```bash
python -m pytest
python -m pytest -m performance -s
ruff check .
ruff format --check .
mypy backend/app
```

## Pricing observations

All cases use S=K=100, T=1 year, r=0.05, sigma=0.20. LSM uses default configuration: 25,000
training paths plus 25,000 independent valuation paths, 50 exercise steps, seed 42, degree-two
POWER, antithetic sampling, and OUT_OF_SAMPLE evaluation. CRR uses 1,000 steps.

| Contract | q | LSM price | LSM SE | CRR price | Early-exercise fraction |
|---|---:|---:|---:|---:|---:|
| American put | 0 | 6.057951 | 0.028236 | 6.089595 | 0.36672 |
| American call | 0 | 10.430217 | 0.065199 | 10.448584 | 0 |
| American call | 0.15 | 4.673536 | 0.031570 | 4.715587 | 0.31448 |

The live API reproduced the first price and SE exactly. Its 95% interval was
[6.002610, 6.113293]. These intervals quantify conditional valuation sampling error for a frozen
policy, not uncertainty about the optimal continuously exercisable American value.

## Path-count convergence

Use the put above, 50 steps, root seed 42, training count equal to valuation count, and CRR
2,000-step reference 6.089990. Each point receives an independent deterministic child seed.

| Valuation paths | Price | SE | Absolute CRR error |
|---:|---:|---:|---:|
| 1,000 | 5.847999 | 0.127949 | 0.241991 |
| 5,000 | 6.054226 | 0.059400 | 0.035764 |
| 10,000 | 6.082910 | 0.044780 | 0.007080 |
| 25,000 | 6.099465 | 0.027911 | 0.009475 |
| 50,000 | 6.053424 | 0.019989 | 0.036566 |

SE decreases with path count in this experiment. Realized price error does not decrease
monotonically. More paths do not remove basis or exercise-grid approximation error.

## Exercise-grid convergence

Same put and CRR reference, with 25,000 paths in each sample. Independent point seeds mean the
50-step point here differs from the separate path-count experiment.

| Exercise steps | Price | SE | Absolute CRR error |
|---:|---:|---:|---:|
| 10 | 5.985240 | 0.029396 | 0.104750 |
| 25 | 6.037481 | 0.028669 | 0.052509 |
| 50 | 6.053528 | 0.027647 | 0.036462 |
| 100 | 6.099284 | 0.027292 | 0.009294 |

This particular sequence improves; no monotonicity assertion is made or tested.

## Basis comparison

Same put, default 25,000+25,000 paths and 50 steps, root seed 42 shared across every basis fit.
All fits in these cases had zero fallbacks.

| Degree | POWER price | LAGUERRE price | SE (both) | Max POWER condition number | Max LAGUERRE condition number |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.984097 | 5.984097 | 0.032336 | 117.62 | 60.15 |
| 2 | 6.057951 | 6.057951 | 0.028236 | 11,072.74 | 9,263.59 |
| 3 | 6.075344 | 6.075344 | 0.027739 | 903,895.31 | 2,349,911.70 |
| 4 | 6.074982 | 6.074982 | 0.027748 | 71,205,826.63 | 765,331,231.83 |

Standard unweighted POWER and LAGUERRE span the same polynomial space at equal degree; nearly
identical predictions are expected for well-resolved least squares. Conditioning differs.
Degree-four LAGUERRE triggers the configured 1e8 warning threshold; a warning alone does not
reject a full-rank fit. Higher degree is not assumed better.

To reproduce these research grids, use `run_lsm_convergence` with values
`[1000,5000,10000,25000,50000]` and mode `PATH_COUNT`, or values `[10,25,50,100]` and mode
`TIME_STEPS`. Use `compare_lsm_bases` with both families and degrees `[1,2,3,4]`.
Supply the default `LongstaffSchwartzConfig` and `BinomialTreeConfig(steps=2000)` to each.
Equivalent HTTP requests use the documented `/convergence` and `/basis-comparison` routes.

## Opt-in runtime observations

These are single-run local timings observed during the final performance validation pass.
They are not throughput guarantees or wall-clock assertions.
Each row uses the put above and an equal-size independent training set in addition to the listed
valuation paths. Simulation time includes both samples.

| Valuation paths × steps | Simulation ms | Backward regression ms | Forward evaluation ms | Total ms | One path matrix, decimal MB |
|---|---:|---:|---:|---:|---:|
| 10,000 × 50 | 6.69 | 10.99 | 2.82 | 21.53 | 4.08 |
| 50,000 × 50 | 36.20 | 40.86 | 10.61 | 91.65 | 20.40 |
| 100,000 × 50 | 71.46 | 86.30 | 26.82 | 191.78 | 40.80 |
| 50,000 × 100 | 64.87 | 79.81 | 21.66 | 173.22 | 40.40 |

Total time includes exercise-boundary summaries and estimator statistics. Path-matrix storage is
`8*max(training_paths,valuation_paths)*(steps+1)` bytes. This is not peak RSS: normal-driver,
log-increment, regression, and cash-flow buffers require additional memory. Full training and
valuation matrices are held sequentially. API limits account conservatively for both sets and
all experiment points before allocation.

## Interpretation and remaining assumptions

- Exact GBM transitions with constant rates, dividends, and volatility; American vanilla only.
- Finite exercise grids and regression policies introduce approximation error beyond sampling SE.
- Root exercise is selected from training continuation and frozen for independent evaluation.
  Held-out values may fall below intrinsic or European reference; diagnostics expose that event.
- Continue-all fallback can underexercise when a fit has insufficient support or rank.
- Boundary extrema are descriptive and require support; common-date CRR differences are not errors
  against an exact continuous-time frontier.
- No unresolved implementation or quality-gate failure remains. Phase 10 scope must be specified
  separately before work begins.
