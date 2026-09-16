# Phase 10 Exotic-Option Validation and Performance

## Environment and method

These local measurements use CPython 3.11+, NumPy float64 arrays, the project's PCG64 generator,
and one process. Wall-clock values are observations from one development-machine run and are not
performance guarantees. Ordinary tests exclude every `performance`-marked case.

The representative market is `S=K=100`, `T=1`, `r=5%`, `q=1%`, and `sigma=20%`. Asian schedules
are regular, exclude `S_0`, and use 12 observations unless stated otherwise. Monte Carlo uses
100,000 paths and seed 42 unless stated otherwise.

## Asian comparison

| Method | Price | SE | Variance reduction vs raw | Runtime (ms) |
|---|---:|---:|---:|---:|
| NONE | 5.835202 | 0.026264 | 1.00 | 11.71 |
| ANTITHETIC | 5.818008 | 0.018581 | 2.00 | 9.40 |
| GEOMETRIC_CONTROL | 5.833412 | 0.000729 | 1297.00 | 14.80 |
| ANTITHETIC_GEOMETRIC_CONTROL | 5.833138 | 0.000749 | 1230.10 | 12.52 |

This is the representative case in which the geometric control materially reduces variance. It is
an empirical property of this contract and sample, not a universal guarantee.

For a geometric Asian under the same inputs, the exact discrete analytical value was `5.628603`.
Plain MC returned `5.630340` with SE `0.025462`; antithetic MC returned `5.613859` with SE
`0.018092`. Both are statistically consistent with the analytical value.

## Convergence observations

Geometric Asian path-count points used deterministic, independent child seeds:

| Paths | MC price | SE | Absolute error |
|---:|---:|---:|---:|
| 1,000 | 5.535998 | 0.180506 | 0.092606 |
| 5,000 | 5.682571 | 0.080065 | 0.053968 |
| 10,000 | 5.695056 | 0.057817 | 0.066452 |
| 50,000 | 5.685972 | 0.026054 | 0.057368 |
| 100,000 | 5.605572 | 0.018134 | 0.023032 |

The SE contracts in aggregate while realized error is nonmonotonic, as expected.

With 50,000 antithetic paths, varying regular monitoring count produced:

| Observations | MC price | SE | Exact geometric price |
|---:|---:|---:|---:|
| 4 | 6.356591 | 0.029266 | 6.368289 |
| 12 | 5.636952 | 0.025602 | 5.628603 |
| 52 | 5.349516 | 0.024450 | 5.346077 |
| 252 | 5.266247 | 0.023994 | 5.278943 |

These are distinct discrete-monitoring contracts and are not labeled continuous-monitoring prices.

## Digital validation

For a cash-or-nothing call with cash payout 10, analytical value was `5.135001`; antithetic MC
returned `5.145961`, SE `0.005835`, and z-score `1.88`. The asset-or-nothing call analytical value
was `61.176310`; MC returned `61.301394`, SE `0.057074`, and z-score `2.19`.

Digital convergence was visibly noisy while its SE contracted:

| Paths | MC price | SE | Absolute error |
|---:|---:|---:|---:|
| 1,000 | 5.165176 | 0.059694 | 0.030175 |
| 5,000 | 5.136639 | 0.025811 | 0.001638 |
| 10,000 | 5.119517 | 0.017869 | 0.015484 |
| 50,000 | 5.136829 | 0.008163 | 0.001828 |
| 100,000 | 5.139302 | 0.005789 | 0.004301 |

## Opt-in timing observations

| Workload | Raw/total (ms) | Control (ms) | Simulation (ms) | Main matrix |
|---|---:|---:|---:|---:|
| Asian 10k x 12 | 1.35 | 1.54 | 1.02 | 0.96 MB |
| Asian 100k x 12 | 10.90 | 13.86 | 9.24 | 9.60 MB |
| Asian 100k x 52 | 46.69 | 52.76 | 42.24 | 41.60 MB |
| Asian 100k x 252 | 298.55 | 264.27 | 279.11 | 201.60 MB |
| Digital 100k | 1.41 | n/a | n/a | 0.80 MB terminal vector |
| Digital 1m | 13.55 | n/a | n/a | 8.00 MB terminal vector |

The 252-date control timing was lower than the raw measurement in this single run due to normal
local timing variability. The implementation makes no claim that the added control calculation is
intrinsically faster. The meaningful design observation is linear matrix storage in path count and
requested observation count, with no finer hidden simulation grid.
