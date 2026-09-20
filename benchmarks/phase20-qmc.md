# Phase 20 QMC benchmark report

## Method

Benchmarks use SciPy Sobol points, PCG64 pseudorandom baselines, exact existing GBM transition
functions, deterministic seeds, and analytical references where available. RQMC standard errors
come from independent scramble estimates. Deterministic Sobol has no reported SE.

Run the opt-in workloads with:

```bash
pytest -m performance backend/tests/test_qmc_performance.py
```

## Workloads

| Product | Sample shape | Construction | Reference |
|---|---:|---|---|
| European vanilla | 262,144 x 1 x 8 scrambles | terminal | Black-Scholes-Merton |
| geometric Asian | 32,768 x 52 x 4 scrambles | Brownian bridge | exact geometric Asian |
| geometric basket | 131,072 x 10 x 4 scrambles | correlated terminal | exact geometric basket |

The convergence benchmark compares pseudorandom, deterministic Sobol, and eight-scramble RQMC over
powers of two. It records sample count, dimension, price, analytical error, valid SE, runtime,
`error * sqrt(runtime_seconds)`, and empirical log-error/log-N slope. Path studies compare
chronological, Brownian bridge, and PCA construction at the same coordinate budget.

## Memory and interpretation

One replicate's coordinate matrix dominates memory: float64 storage is approximately
`8 * samples * dimension` bytes before product-specific state. Replicates run sequentially. The
Asian 32,768 x 52 coordinate matrix is about 13.6 MB. The basket 131,072 x 10 matrix is about
10.5 MB. API coordinate budgets bound total requested work, while sequential execution bounds peak
replicate memory.

## Local measurements

Measured on 2026-09-20 in the project virtual environment. The European analytical reference was
`8.8273212254`.

| Method | N | Price | Absolute error | SE | Runtime |
|---|---:|---:|---:|---:|---:|
| PCG64 MC | 256 | 8.337607 | 0.489714 | 0.827107 | 0.14 ms |
| deterministic Sobol | 256 | 8.655349 | 0.171972 | unavailable | 3.18 ms |
| 8-scramble RQMC | 256 | 8.807361 | 0.019960 | 0.005948 | 1.17 ms |
| PCG64 MC | 4,096 | 8.888425 | 0.061103 | 0.213424 | 0.14 ms |
| deterministic Sobol | 4,096 | 8.813060 | 0.014262 | unavailable | 0.12 ms |
| 8-scramble RQMC | 4,096 | 8.828438 | 0.001117 | 0.001369 | 1.35 ms |
| PCG64 MC | 8,192 | 9.098153 | 0.270832 | 0.154489 | 0.22 ms |
| deterministic Sobol | 8,192 | 8.819758 | 0.007563 | unavailable | 0.17 ms |
| 8-scramble RQMC | 8,192 | 8.827725 | 0.000403 | 0.000403 | 1.73 ms |

Across `N=2^8,...,2^13`, empirical absolute-error slopes were `-0.288` for this single-seed MC
path, `-0.902` for deterministic Sobol, and `-0.981` for RQMC. The MC path-level SE slope was
`-0.488`; the RQMC replicate-SE slope was `-0.675`. Error slopes are sample-path diagnostics and
are not theoretical guarantees.

At 4,096 points and eight scrambles, the exact-reference geometric Asian results were:

| Path construction | Price | Absolute error | RQMC SE | Runtime |
|---|---:|---:|---:|---:|
| chronological | 5.132352 | 0.006887 | 0.002555 | 9.07 ms |
| Brownian bridge | 5.140011 | 0.000772 | 0.001050 | 10.56 ms |
| PCA | 5.138584 | 0.000655 | 0.000224 | 9.55 ms |

For the three-asset geometric basket at 4,096 points, the analytical reference was `6.453019`:

| Method | Price | Absolute error | SE | Runtime |
|---|---:|---:|---:|---:|
| PCG64 MC | 6.203400 | 0.249619 | 0.153734 | 0.60 ms |
| 8-scramble RQMC | 6.452888 | 0.000131 | 0.001535 | 7.09 ms |

The marked performance workloads completed together in `0.85 s`: European 262,144 x 8 in `0.03
s`, Asian 32,768 x 52 x 4 in `0.16 s`, and basket 131,072 x 10 x 4 in `0.10 s`. Timings are local
observations rather than service guarantees. QMC improvement is descriptive and is never
guaranteed for discontinuous or tail-quantile objectives.
