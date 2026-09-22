# American Options Analyzer benchmark

## Method

Measured on 2026-09-21 in the project Python 3.11+ environment using the deterministic in-memory
provider, both calls and puts, evenly spaced strikes from 75 to 125, a 50% strike window, and the
production default 400-step CRR tree. Each fair value used leave-one-out volatility and each Greek
set included native Delta/Gamma/Theta plus CRR bump-and-revalue Vega/Rho.

Provider network latency was excluded. `market data` below is local fake-provider lookup, bounded
filtering, and normalized snapshot construction. Results are one representative run and should be
treated as order-of-magnitude observations rather than stable machine guarantees.

| Contracts | Market data / normalization | IV solving | Smile construction | Base CRR pricing | Greek repricing | Total CPU |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 0.044 ms | 13.937 ms | 2.017 ms | 137.707 ms | 565.468 ms | 719.683 ms |
| 50 | 0.051 ms | 32.386 ms | 7.422 ms | 339.559 ms | 1,355.932 ms | 1,736.287 ms |
| 100 | 0.081 ms | 65.800 ms | 23.747 ms | 689.786 ms | 2,771.106 ms | 3,552.467 ms |

The opt-in `backend/tests/test_options_analyzer_performance.py` check covers 20, 50, and 100
contracts with a lighter 100-step tree for CI-friendly performance validation. Live network
latency is provider, subscription, geography, and market-session dependent and was not measured by
this offline CPU benchmark.
