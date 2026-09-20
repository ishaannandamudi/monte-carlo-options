# Phase 16 portfolio risk and scenario benchmarks

## Measurement setup

Measurements were taken on the local arm64 development host with Python 3.12.14 on 2026-09-18.
Each reported value is the median of three complete calls unless stated otherwise; min/max expose
local timing noise. The vanilla workload used identical one-year ATM BSM calls with `S=K=100`,
`r=0.03`, `q=0.01`, and `sigma=0.20`, but unique position identifiers. Timings include typed result
construction, validation, aggregation, and request-local caching. They are development observations,
not service guarantees.

## Vanilla portfolio scaling

| Positions | Value median (ms) | Value min/max (ms) | Greeks median (ms) | Greeks min/max (ms) |
|---:|---:|---:|---:|---:|
| 10 | 0.159 | 0.142 / 0.736 | 0.563 | 0.534 / 1.063 |
| 100 | 0.943 | 0.918 / 0.959 | 3.461 | 3.333 / 3.478 |
| 1,000 | 9.550 | 9.297 / 9.867 | 33.669 | 32.672 / 47.428 |

The repeated analytical inputs share one request-local unit price and Greek calculation; signed
quantities and typed position outputs remain per-position. Cache keys exclude position identifier,
quantity, metadata, and provenance, but include every pricing input. Stochastic positions are never
collapsed into this deterministic cache because each identifier owns a separate child stream.

## Mixed portfolio

The 100-position workload cycles through BSM European, 50-step American CRR, geometric Asian,
analytical digital, analytical continuous barrier, and geometric basket positions.

| Workload | Median (ms) | Min (ms) | Max (ms) |
|---|---:|---:|---:|
| Mixed 100 value | 2.083 | 2.004 | 5.267 |
| Mixed 100 Greeks/exposures | 14.395 | 14.210 | 14.720 |

Unavailable basket scalar Greeks remain null; this timing includes per-asset basket Delta finite
differences and all underlying/expiry/type/class/model group construction.

## Full repricing and approximation

A 100-position vanilla portfolio received a combined `-5%` spot and `+2` volatility-point shock.
Five full public scenario calls had a 6.448 ms median (5.951–18.033 ms). With base Greeks already
computed, the local Delta/Gamma/Vega/Theta/Rho arithmetic alone had a 0.0448 ms median over 20
repetitions (0.0440–0.0624 ms). Full repricing remains authoritative; the arithmetic timing does not
include Greek construction.

## Scenario grid scaling

Both grids used 100 BSM positions and spot-by-volatility Cartesian products. The grid engine reuses
one base valuation and Greek result across cells and reuses identical deterministic units within a
cell.

| Positions × cells | Median (ms) | Min (ms) | Max (ms) |
|---|---:|---:|---:|
| 100 × 25 | 42.660 | 42.381 | 42.939 |
| 100 × 100 | 162.510 | 155.462 | 169.557 |

The output retains cells and scalar portfolio diagnostics rather than raw paths. Memory therefore
scales with the position result and grid-cell counts. Monte Carlo workloads additionally allocate
only the existing engine's current position paths; the portfolio response never stores them.

## Opt-in checks

`backend/tests/test_portfolio_performance.py` covers 10/100/1,000-position analytical scaling and a
100-position × 25-cell grid under `pytest -m performance`. The 100-cell measurement above is kept in
this report rather than every routine test run. Performance thresholds are generous regression
guards and do not define API latency targets.
