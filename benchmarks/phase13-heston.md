# Phase 13 Heston validation and performance

## Benchmark case

Measurements used Python 3.11 on the local development host with:

```text
S=100, K=100, T=1, r=0.02, q=0.01
v0=0.04, kappa=2, theta=0.04, sigma_v=0.3, rho=-0.7
seed=42, antithetic=true, paths=100,000
```

The semi-analytical call price was `8.0624614666`. Adaptive Little Heston Trap integration used
336 characteristic-function evaluations, stopped at frequency 120, and took 0.381 ms in this run.

## Terminal Monte Carlo timings

| Scheme | Steps | Price | SE | Price minus analytical | Simulation ms | Total ms |
|---|---:|---:|---:|---:|---:|---:|
| FTE | 50 | 8.062997 | 0.023527 | +0.000535 | 41.2 | 42.0 |
| FTE | 100 | 8.077138 | 0.023542 | +0.014676 | 80.6 | 81.3 |
| FTE | 250 | 8.091197 | 0.023485 | +0.028735 | 198.6 | 199.3 |
| QE-M | 50 | 8.068591 | 0.023301 | +0.006130 | 101.5 | 102.3 |
| QE-M | 100 | 8.066623 | 0.023379 | +0.004161 | 198.4 | 199.1 |
| QE-M | 250 | 8.121403 | 0.023389 | +0.058942 | 507.3 | 508.1 |
| FTE (500k paths) | 100 | 8.058198 | 0.010467 | -0.004263 | 409.6 | 412.9 |
| QE-M (500k paths) | 100 | 8.043113 | 0.010438 | -0.019348 | 1017.9 | 1021.3 |

The observed errors include Monte Carlo noise. Their nonmonotonic sequence is expected and is not
used to claim that a larger step count made a specific realized sample better. The analytical
price lies within about 2.6 reported SEs of every listed point. QE-M costs more per step because it
evaluates conditional moments, branch logic, and a conditional MGF; its principal benefit is better
variance transition and martingale behavior in difficult regimes rather than universal runtime or
realized-error dominance.

## Memory and scaling

Terminal simulation retains a handful of float64 path vectors and is `O(paths)`, independent of
the number of steps. It does not allocate a `(paths,steps)` driver cube. Full diagnostics allocate
both spot and variance matrices and therefore require at least
`16 * paths * (steps+1)` bytes for returned state alone. For 100,000 paths and 100 steps, those two
matrices are about 161.6 MB, excluding temporary vectors; the API's 25-million path-step budget
keeps this use bounded.

Opt-in performance tests cover both schemes at 100,000 paths with 50, 100, and 250 steps. The local
performance run completed all six cases plus analytical pricing in 1.51 seconds. Two 500,000-path
by 100-step cases are present behind `RUN_LARGE_PERFORMANCE_TESTS=1`; they are skipped by default to
keep routine validation predictable. With that flag enabled, all nine Phase 13 performance tests
passed in 2.92 seconds.

## Numerical validation

- The standard offline Heston benchmark with `r=5%`, `q=0` prices the call at `10.3942185523`.
- Put-call parity is checked to tight floating-point tolerance.
- `sigma_v=0` uses exact integrated deterministic variance and matches Black-Scholes.
- Low- and high-`psi` QE branches reproduce conditional CIR moments statistically.
- Full-path variance means agree with `theta+(v0-theta)exp(-kappa t)` at multiple dates.
- QE-M terminal spot means agree with `S0 exp((r-q)T)` under a sampling-scaled tolerance.
- FTE and QE-M calls and puts agree with the analytical value under SE-scaled tolerances.
- Feller-violating parameters, negative rates, dividends, and boundary values remain finite.

Timings are measurements of one local run, not service-level guarantees.
