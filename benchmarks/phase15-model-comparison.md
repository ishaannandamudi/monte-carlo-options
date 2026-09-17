# Phase 15 model-comparison validation and performance

## Measurement setup

Measurements were taken on the local arm64 development host with Python 3.12.14 on 2026-09-17.
They are local development measurements, not service-level latency guarantees. Except where stated,
the European contract was an at-the-money one-year call with `S=K=100`, `r=0.03`, `q=0.01`, and
BSM volatility `0.20`. Heston used
`v0=theta=0.04, kappa=2, sigma_v=0.3, rho=-0.7`. Timed calls rebuilt their result objects and
included normal validation and diagnostics.

## One-contract latency

The table reports median wall time over repeated complete calls. Min/max are included to show local
timing noise rather than imply a deterministic runtime.

| Engine and budget | Repetitions | Median (ms) | Min (ms) | Max (ms) |
|---|---:|---:|---:|---:|
| Black-Scholes | 1,000 | 0.0221 | 0.0213 | 0.2747 |
| European CRR, 500 steps | 20 | 1.2942 | 1.2743 | 1.7573 |
| GBM MC, 100k paths, antithetic + control | 10 | 1.7786 | 1.7380 | 3.5416 |
| Heston analytical | 20 | 0.3101 | 0.3039 | 0.6097 |
| Heston FTE, 10k paths x 50 steps | 5 | 5.2669 | 5.0683 | 6.0254 |
| Heston QE-M, 10k paths x 50 steps | 5 | 12.5517 | 12.2614 | 12.7248 |
| American CRR put, 500 steps | 10 | 9.6315 | 9.3444 | 10.2030 |
| LSM put, 10k training + 10k valuation x 50 steps | 5 | 19.4710 | 19.2693 | 26.3156 |

The Phase 15 public output uses the median of `runtime_repetitions` model-reported or measured
complete-call latency. Price and runtime stay separate; no latency-adjusted accuracy score is
created.

## 10- and 100-contract workloads

All contracts shared maturity and varied strike uniformly from 80 to 120. Values are median wall
time. BSM used 20 repetitions, CRR used three, and Heston used three scalar or five cached-batch
repetitions.

| Workload | 10 contracts (ms) | 100 contracts (ms) |
|---|---:|---:|
| Black-Scholes | 0.2226 | 2.2945 |
| CRR, 250 steps | 6.6036 | 68.2730 |
| Heston analytical, scalar calls | 2.9938 | 30.7391 |
| Heston analytical, maturity-scoped batch cache | 1.1862 | 9.5588 |

The cached Heston workload was roughly 2.5 times faster at 10 strikes and 3.2 times faster at 100
strikes on this host. Phase 15 market and surface fit construct one local batch pricer per maturity,
parameter vector, and market snapshot. The cache is not global and cannot mix maturities or
parameters.

## GBM accuracy/compute frontier

The frontier used independent deterministic child seeds, antithetic plus discounted-stock control
variates, and BSM as the named reference. Error is one stochastic realization at each budget, so it
need not fall monotonically.

| Paths | Runtime (ms) | Absolute error | Standard error |
|---:|---:|---:|---:|
| 1,000 | 0.1854 | 0.013406 | 0.087916 |
| 5,000 | 0.2278 | 0.059519 | 0.041825 |
| 20,000 | 0.4704 | 0.018394 | 0.020662 |
| 100,000 | 1.8923 | 0.001408 | 0.009176 |

The SE decreases with the expected sampling scale while realized absolute error fluctuates. Phase
15 returns the observations and never marks a budget optimal. CRR frontiers use steps; Heston MC
uses paths at fixed time steps; LSM reports its path-time workload and retains fixed regression and
exercise-grid policies.

## Fit and consistency validation

Ordinary tests use a six-observation, two-expiry synthetic Heston surface. Under the generating
parameters, Heston analytical price and IV RMSE are zero to numerical precision, every price lies
inside its synthetic bid/ask spread, and BSM reconstructed from each observed IV has numerical-zero
fit. Per-expiry, Phase 14 moneyness-bucket, ATM, Phase 6 PCHIP-total-variance skew, and curvature
diagnostics are checked directly.

The theoretical consistency suite checks:

- BSM against GBM MC with a four-SE statistical tolerance;
- BSM against high-step European CRR;
- BSM against Heston with `v0=theta=sigma_BSM^2` and `sigma_v=0`;
- analytical deterministic-variance Heston against FTE and QE-M;
- American CRR against LSM for an American put.

Deliberately coarse tests verify that material discrepancies create
`NUMERICAL_CONSISTENCY_WARNING` without altering any model result.

## Opt-in performance coverage

`test_model_comparison_performance.py` contains two performance-marked cases:

- a five-model comparison with three timing repetitions, 100k GBM paths, and 30k x 50 Heston paths;
- an 84-observation, four-expiry analytical Heston fit that exercises maturity-local batching.

The ordinary pytest configuration excludes these cases. Their ceilings are functional regression
guards rather than production latency promises. HTTP defaults cap models at 7, observations at 500,
surface/result points at 1,000, frontier points at 20, and timing repetitions at 10, in addition to
all earlier per-engine path, step, and node limits.
