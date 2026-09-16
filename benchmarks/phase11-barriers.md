# Phase 11 Barrier Validation and Performance

## Setup

Measurements use one local CPython process, NumPy float64 arrays, PCG64, and antithetic paths. The
representative contract is a down-and-out call with `S=K=100`, `H=90`, `T=1`, `r=5%`, `q=1%`, and
`sigma=20%`. Timings are observations, not guarantees.

## Continuous analytical validation

For `r=5%`, `q=0`, and `sigma=20%`, selected one-year zero-rebate values are:

| Contract | Continuous value |
|---|---:|
| Down-and-in call, H=90 | 1.785111914 |
| Down-and-out call, H=90 | 8.665471658 |
| Up-and-in put, H=110 | 1.375332211 |
| Up-and-out put, H=110 | 4.198193811 |

The corresponding IN plus OUT values equal vanilla Black-Scholes values to floating-point
precision. Bridge-weighted Monte Carlo tests for these four canonical contracts lie within five
reported standard errors of their analytical values.

## Performance results

| Paths x future observations | Discrete | Bridge | Retained state matrix |
|---|---:|---:|---:|
| 10,000 x 52 | 6.59 ms | 15.64 ms | 4.24 MB |
| 100,000 x 52 | 81.38 ms | 171.10 ms | 42.40 MB |
| 100,000 x 252 | 442.83 ms | 866.91 ms | 202.40 MB |

Bridge time includes an interval loop with O(paths) temporary vectors. It does not retain a full
crossing-probability matrix. Discrete detection remains vectorized over its fixed observation
matrix.

## The 100,000 x 1,000 case

The retained float64 state matrix alone would be approximately `800.8 MB`. Normal drivers and log-
GBM transition workspace temporarily require additional matrices of similar scale. It also
represents about 100.1 million path nodes, four times the default HTTP budget of 25 million.

The opt-in performance test contains this exact case behind
`RUN_LARGE_BARRIER_BENCHMARKS=1`; the normal performance run skips it to avoid machine-dependent
memory pressure. The default API rejects it before allocation. Future batching may support this
direct-library workload while preserving antithetic pairs and reproducible streams, but Phase 11
does not claim that optimization.

## Interpretation

Bridge monitoring produces a hit probability at least as large as sparse discrete observation in
the representative tests. Zero-rebate knock-out bridge values are correspondingly no larger. This
price ordering is not generalized to arbitrary rebates. Path-count errors remain stochastic and are
not required to improve monotonically.
