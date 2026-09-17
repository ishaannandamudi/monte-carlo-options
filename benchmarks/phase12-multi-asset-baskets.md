# Phase 12 Multi-Asset and Basket Benchmarks

These are local research measurements from the Phase 12 validation environment. They are useful for
allocation and API-limit decisions, not latency guarantees. Tests run with PCG64, antithetic
sampling, dense equicorrelation `rho=0.25`, equal spots/volatilities/weights, and vectorized NumPy.

## Terminal workloads

| Paths | Assets | Factorization | Normal generation | Terminal simulation | Full basket pricing | One `(paths, assets)` array |
|---:|---:|---:|---:|---:|---:|---:|
| 100,000 | 2 | 0.088 ms | 0.767 ms | 2.703 ms | 3.774 ms | 1.6 MB |
| 100,000 | 10 | 0.043 ms | 2.824 ms | 9.215 ms | 9.352 ms | 8.0 MB |
| 100,000 | 50 | 0.304 ms | 14.951 ms | 53.625 ms | 47.565 ms | 40.0 MB |
| 1,000,000 | 2 | 0.041 ms | 6.041 ms | 29.089 ms | 32.866 ms | 16.0 MB |

Each timing is a single warm local observation and naturally varies. Normal generation measures the
independent driver allocation alone. Terminal simulation includes generation, factor application,
GBM transformation, and terminal output. Full pricing separately repeats those operations and adds
basket reduction, payoff, uncertainty, and typed-result construction, so component columns are not
intended to sum to the pricing column.

The final column is exact `float64` storage for one terminal-shaped matrix. Vectorized evaluation
temporarily holds independent normals, correlated normals, transformed values, and terminal
prices; practical peak storage is several such matrix units. The API's 25-million path-asset-node
limit caps one matrix at 200 MB and limits aggregate analytics before allocation.

## Factorization scaling

| Assets | Dense factorization |
|---:|---:|
| 2 | 0.168 ms |
| 5 | 0.056 ms |
| 10 | 0.040 ms |
| 25 | 0.112 ms |
| 50 | 0.330 ms |
| 100 | 1.024 ms |

Small timings are dominated by fixed overhead and should not be read as monotone. The algorithmic
cost remains `O(N^3)` time and `O(N^2)` memory. Phase 12 deliberately caps HTTP inputs at 100 assets.

## Full-path workload

The optional `10,000 paths x 50 future times x 5 assets` experiment completed in **27.361 ms**.
The returned `(10,000, 51, 5)` state cube occupies **20.4 MB**. Driver, correlated-increment, and
transformed-state workspaces increase transient peak memory. Basket terminal pricing never uses
this path allocator.

## Automated performance gate

The six new opt-in cases completed in **0.74 seconds** together:

```text
6 passed in 0.74s
```

The checks cover the four terminal workloads above, factorization from 2 through 100 assets, and
the full-path case. They assert finite results, expected shapes and allocations, and broad runtime
ceilings designed to identify regressions without making CI depend on workstation noise.
