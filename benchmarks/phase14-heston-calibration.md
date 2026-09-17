# Phase 14 Heston calibration validation and performance

## Measurement setup

Measurements were taken on the local arm64 development host with Python 3.12.14. They are
single-run development measurements rather than latency guarantees. All data was generated offline
with the Phase 13 semi-analytical engine from:

```text
S=100, r=0.02, q=0.01
v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7
target=PRICE, accuracy=STANDARD
```

The ordinary suite excludes performance-marked cases. The opt-in Phase 14 performance file contains
six cases and completed in 1.10 seconds on this host.

## Objective-evaluation scaling

The timed region contains one uncached objective evaluation at the true parameters. Dataset
generation is excluded.

| Surface | Observations | Time (ms) | Prices/second | Objective |
|---|---:|---:|---:|---:|
| 1 expiry x 10 strikes | 10 | 1.033 | 9,679 | `2.84e-24` |
| 5 expiries x 10 strikes | 50 | 5.379 | 9,296 | `1.10e-17` |
| 10 expiries x 20 strikes | 200 | 18.282 | 10,940 | `3.84e-18` |

Observed cost was close to linear in quote count. Pricing remains one deterministic analytical
contract result per observation, while a maturity-scoped batch object reuses characteristic-function
values at shared quadrature frequencies. Exact parameter-vector caching makes a repeated objective
vector a cache hit.

## Noise-free local recovery

A 50-observation surface used five expiries and ten log-forward-moneyness coordinates per expiry.
The warm initial vector was `(0.045, 1.8, 0.045, 0.32, -0.65)`, and bounded least-squares had a
30-evaluation budget.

| Metric | Result |
|---|---:|
| Runtime | 255.4 ms |
| Objective evaluations | 45 |
| Individual Heston prices | 2,100 |
| Price RMSE | `3.26e-9` |
| `v0` | `0.04000000003` |
| `kappa` | `2.0000000023` |
| `theta` | `0.03999999998` |
| `sigma_v` | `0.29999999971` |
| `rho` | `-0.70000000092` |

This overidentified noise-free grid recovered both prices and the generating parameters to high
precision. The tests require very low price error and only representative parameter closeness,
because exact parameter recovery is not a valid general requirement for sparse or noisy Heston
surfaces.

The smaller nine-observation recovery used by ordinary tests reached price RMSE below `2e-6` for
price, IV, and total-variance targets. The price-target case recovered `v0` within `1e-7` and `rho`
within `1e-6` from the documented close start.

## Noisy synthetic fit

A six-observation, two-expiry IV dataset used independent Gaussian IV noise with standard deviation
`0.002` and seed 11. Twelve bounded local evaluations from the documented close start produced IV
RMSE `1.4240e-4` and objective `2.0278e-8`; all parameters and diagnostics were finite and inside
bounds. The fitted vector was `(0.04182, 1.66563, 0.03575, 0.18895, -0.95867)`. Its movement away
from the generating vector on this small noisy surface is expected and illustrates why price/IV fit,
bounds, and identifiability diagnostics matter more than exact parameter recovery.

## Global and hybrid cost

A deliberately modest hybrid benchmark used one expiry, ten strikes, differential-evolution
population multiplier 2, and five global/local iterations. It completed in 145.5 ms, made 94
objective calls and 890 individual price calls, and ran both `GLOBAL` and `GLOBAL_REFINEMENT`
stages. The intentionally small budget produced price RMSE `0.0529`; it validates bounded,
deterministic orchestration rather than production fit quality. Larger global budgets improve search
at correspondingly higher cost.

The hybrid retains the global point as a candidate. The final objective therefore cannot be worse
than the global objective solely because local refinement failed or stopped early.

## Fitted surface and validation

Generating prices, BSM IVs, residuals, ATM term points, and skew fits for the 200-observation grid
took 204.9 ms. It returned 200 ordered points with price-target RMSE `1.96e-9` at the generating
parameters.

Holding out the complete two-year expiry from a three-expiry, nine-observation exact dataset left six
training and three validation observations. With the generating vector as the warm start, held-out
price RMSE was `3.10e-9`. This checks the validation accounting and complete-expiry split; it is not
evidence of expected market out-of-sample performance.

## Stability and identification

Three local runs from one exact warm start and two deterministic broad bounded starts completed in
about 81 ms on the nine-point surface. The exact start reached objective `4.29e-19`. Both broad
starts entered flat numerical-failure regions and returned the finite configured penalty `1e6`.
They are retained and marked nonconverged at the calibration-result layer. This illustrates why the
framework returns per-start failures and dispersion rather than treating a SciPy stop flag as proof
of a valid fit. In practice, a calibrated warm start, more local starts, or the hybrid strategy is
preferable to relying on one broad random start.

At the true vector, the nine-point test Jacobian has rank five and a finite condition number; the
local covariance approximation is available. A two-observation surface reports rank at most two and
withholds covariance. No fit is rejected merely because identification is weak.

## Performance-test coverage

The opt-in tests cover:

- one objective evaluation at 10, 50, and 200 observations;
- bounded local calibration at 50 observations;
- global plus local hybrid orchestration with a small budget;
- fitted-surface generation at 200 observations.

Each case has a generous regression ceiling to detect accidental runaway work without turning a
development-host measurement into a production service-level promise. API defaults cap observations
at 250, starts at 10, iterations at 500, source surface points at 500, and returned diagnostic points
at 500. Direct library callers may choose larger research workloads deliberately.
