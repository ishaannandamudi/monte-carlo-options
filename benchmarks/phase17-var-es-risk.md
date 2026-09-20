# Phase 17 VaR and Expected Shortfall benchmarks

## Measurement setup

Measurements were taken on the local arm64 development host with Python 3.12.14 on 2026-09-20.
The portfolios use one-year ATM BSM calls with `S=K=100`, `r=0.03`, `q=0.01`, and
`sigma=0.20`; position identifiers are unique. Historical and Monte Carlo spot returns have 1%
and 2% daily standard deviation respectively. Runs include Pydantic result construction, base
valuation/Greeks, distribution diagnostics, and contribution grouping when selected. Warm-run
medians are used where repetitions are stated. These are local observations, not latency promises.

## Parametric factor scaling

The workload has one position with equal explicit exposure across independent factors and excludes
contribution output. Requests were constructed before timing; each result still validates prepared
covariance and builds analytical distribution diagnostics.

| Factors | Median (ms) | Min / max (ms) |
|---:|---:|---:|
| 10 | 0.620 | 0.526 / 1.605 |
| 50 | 0.604 | 0.569 / 0.616 |
| 100 | 0.866 | 0.858 / 0.907 |
| 500 | 10.032 | 10.018 / 10.136 |

Covariance eigenvalidation is cubic in factor count and becomes the dominant cost for wide inputs.
The actual Delta-normal portfolio variance and normal VaR/ES arithmetic is quadratic or lower.

## Historical scaling: Greek approximation versus full repricing

The Greek workload uses diagonal Delta-Gamma and vectorized factor arithmetic. Full repricing sends
every dated row through the Phase 16 scenario engine while reusing base values and Greeks.

| Positions × scenarios | Delta-Gamma median (ms) | Full repricing median (ms) |
|---:|---:|---:|
| 100 × 250 | 4.410 | 574.732 |
| 100 × 1,000 | 4.728 | 2,291.921 |
| 1,000 × 250 | 36.850 | 6,597.235 (single measured run) |

For 100×250, the observed warm Delta-Gamma range was 4.226–15.890 ms and the full-repricing range
was 573.051–575.902 ms. For 100×1,000, ranges were 4.664–4.903 ms and
2,282.778–2,301.064 ms. The 1,000×250 full case demonstrates why HTTP limits use the
scenario-position product rather than either dimension alone.

## Monte Carlo factor simulation

The single-factor workloads use deterministic 25,000-row batches, omit tail payloads and
contributions, and retain only the exact O(N) portfolio P&L vector.

| Simulations | Delta (ms) | Delta-Gamma diagonal (ms) |
|---:|---:|---:|
| 100,000 | 16.602 | 16.358 |
| 500,000 | 79.814 | 78.240 |
| 1,000,000 | 170.670 | 166.885 |

The small Delta/Delta-Gamma differences are timing noise at one factor; both scale linearly in
simulation count. Exact quantile selection, distribution moments, and bounded typed results are
included.

## Monte Carlo full revaluation

A 10-position BSM portfolio under 1,000 factor scenarios completed in 261.107 ms using batches of
100. Full revaluation is intentionally benchmarked at a smaller workload because it performs
pricing work per scenario and position. No calibration or persistent cache is introduced.

## Contribution cost

A 100-position, 100,000-scenario Delta run took 31.456 ms without contributions and 35.494 ms with
empirical ES contributions. The contribution run deterministically regenerates the same factor
stream and retains only the selected tail-position rows; it does not retain a 100,000×100 matrix.
Full-revaluation contributions do require the position matrix and therefore face the stricter API
scenario-position cap.

## Backtesting

A 10,000-observation backtest, including strict exception construction, Kupiec POF,
Christoffersen independence, conditional coverage, exception dates, and traffic-light output, had
a 0.798 ms median across three calls (0.769–1.120 ms).

## Opt-in regression checks

`backend/tests/test_risk_performance.py` checks parametric scaling at 10, 50, 100, and 500 factors
and factor Monte Carlo at 100,000 and 500,000 simulations under `pytest -m performance`. The larger
historical/full-revaluation and one-million-scenario measurements stay in this report so ordinary
development runs remain fast. Resource ceilings are safety defaults based on state size and nested
pricing work, not performance guarantees.
