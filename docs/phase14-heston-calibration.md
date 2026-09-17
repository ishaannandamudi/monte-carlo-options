# Phase 14 Heston calibration

## Scope and boundaries

Phase 14 fits the five risk-neutral Phase 13 Heston parameters to European option prices or
Black-Scholes-Merton implied volatilities. It adds calibration, fit diagnostics, structured
validation, and HTTP adapters without changing the Phase 13 analytical pricer, stochastic process,
or Monte Carlo engines.

The implementation is split by responsibility:

- `models/heston_calibration.py` defines observations, datasets, policies, bounds, results, and
  diagnostics.
- `calibration/datasets.py` converts Phase 5 chains and Phase 6 surfaces, filters observations,
  resolves duplicates, and assigns weights.
- `calibration/heston.py` owns objective evaluation, initialization, optimization, residuals,
  identifiability analysis, and fitted surfaces.
- `calibration/synthetic.py` creates deterministic offline datasets from known Phase 13 parameters.
- `analytics/heston_calibration.py` owns repeated-start stability and structured holdout studies.
- `api/heston_calibration.py` applies request limits and delegates to the library.

SABR, local volatility, stochastic local volatility, jumps, American Heston, portfolio risk,
real-time recalibration, QMC, parallel calibration, and trading integration remain outside this
phase.

## Calibration problem and parameters

For one shared spot `S`, continuously compounded rate `r`, and continuous dividend yield `q`, the
dataset may contain many strikes, maturities, and call/put observations. The optimizer fits the
native parameter vector in one canonical order:

```text
x = [v0, kappa, theta, sigma_v, rho].
```

`v0` and `theta` are variances, `kappa` is mean-reversion speed, `sigma_v` is volatility of
variance, and `rho` is spot/variance Brownian correlation. Conversion helpers are the only place
that maps between the vector and `HestonParameters`; all optimizers use this order.

Calibration operates directly in bounded native parameter space. SciPy's bounded least-squares and
differential-evolution algorithms make an unconstrained log/tanh transformation unnecessary. The
defaults are:

| Parameter | Lower | Upper |
|---|---:|---:|
| `v0` | `1e-6` | `2.0` |
| `kappa` | `1e-4` | `20.0` |
| `theta` | `1e-6` | `2.0` |
| `sigma_v` | `1e-4` | `5.0` |
| `rho` | `-0.999` | `0.999` |

Every interval is finite and user-overridable. Positive-parameter lower bounds must be
nonnegative, correlation bounds must remain inside `[-1,1]`, and every lower bound must be strictly
below its upper bound. Initial, warm-start, and diagnostic parameters are rejected if they lie
outside the selected bounds. Results list parameters within `1e-5` of a bound's width as active.

## Observations and targets

`HestonCalibrationObservation` stores strike, positive maturity, option type, optional market price
and IV, bid, ask, midpoint, custom weight, source, timestamp, and quality metadata. A dataset adds
spot, rate, dividend yield, optional valuation timestamp and symbol, plus free-form provenance
metadata. Inputs are finite floating-point research values rather than accounting decimals.

Preprocessing derives a missing price from supplied BSM IV or derives a missing IV from price with
the Phase 4 solver. An observation is retained only when both representations are available, since
pricing diagnostics return both. The three targets are:

```text
PRICE:              r_i = P_Heston,i - P_market,i
IMPLIED_VOLATILITY: r_i = sigma_Heston,i - sigma_market,i
TOTAL_VARIANCE:     r_i = T_i (sigma_Heston,i^2 - sigma_market,i^2).
```

Every Heston model price comes from the deterministic Phase 13 Little Heston Trap semi-analytical
engine. Monte Carlo is never called inside calibration. IV and total-variance targets invert each
model price with the Phase 4 solver under the dataset's `r` and `q`.

## Cleaning, OTM selection, and duplicates

Before optimization, each price is checked against the Phase 4 continuous-dividend European
bounds. The engine can exclude stale/future, crossed, missing-midpoint, arbitrage-invalid,
low-price, wide-spread, disallowed-maturity, and out-of-range moneyness observations. It never
repairs a quote. Every removal includes the original index and a stable reason; the result reports
both details and reason counts.

Moneyness uses the Phase 4/6 forward convention:

```text
F(T) = S exp((r-q)T),       k = log(K/F).
```

`ALL` retains both option sides. The default `OTM_ONLY` selects puts for `K < F`, calls for `K > F`,
and calls at forward equality under a scale-aware deterministic tolerance.

Duplicates share `(maturity, strike, option_type)`. The winner is selected by lowest relative
spread, then fewest quality issues, newest timestamp, and lexical source identifier. Removed copies
remain visible as `DUPLICATE_OBSERVATION` exclusions.

The Phase 5 converter consumes `EnrichedOptionChain`; it preserves selected market values, recovered
IV, timestamp, provider symbol, and existing quality decisions. The Phase 6 converter flattens
surface smiles, preserves European BSM IV, and reconstructs a BSM price when the point has no market
price. Neither converter rewrites its source subsystem.

## Weights and objective normalization

Each retained observation has a nonnegative weight. The supported raw definitions are:

```text
UNIFORM:       w_i = 1
VEGA:          w_i = max(Vega_BSM,i, vega_floor)^2
INVERSE_VEGA:  w_i = 1 / max(Vega_BSM,i, vega_floor)^2
BID_ASK:       w_i = 1 / max(ask_i-bid_i, spread_floor)^2
CUSTOM:        w_i = observation.weight.
```

`VEGA` emphasizes high-Vega observations. `INVERSE_VEGA` scales price residuals toward approximate
volatility errors and emphasizes low-Vega observations; it is deliberately distinct. `BID_ASK`
interprets spread as an error scale. Floors prevent infinite weights. Missing or invalid inputs for
the chosen policy produce an explicit exclusion.

Raw weights are normalized to mean one. With expiry balancing enabled, weights within each maturity
are first divided by that maturity's total, giving each expiry equal aggregate influence, then all
weights are normalized to mean one.

The data residual sent to least-squares is

```text
e_i = sqrt(w_i) r_i.
```

The reported raw RMSE, MAE, and maximum error use `r_i`; weighted RMSE uses `e_i`. Objective
contributions are sums of squared weighted residuals divided by the full observation count, so
per-expiry contributions add to the overall squared weighted RMSE for linear loss.

## Robust loss, Feller penalty, and regularization

Local least-squares supports SciPy's `linear`, `soft_l1`, and `huber` losses. `f_scale` controls the
transition from quadratic to robust behavior. The same robust formula is used when candidates from
multiple starts or global search are compared.

The Feller condition is diagnostic, not a hard constraint. An optional penalty is

```text
L_Feller = lambda_Feller max(sigma_v^2 - 2 kappa theta, 0)^2.
```

When enabled, a fixed extra residual `sqrt(n L_Feller)` is appended. A zero residual remains present
when the condition is satisfied, keeping the least-squares residual dimension constant across the
constraint boundary. Results expose Feller satisfaction, ratio, and the unmodified penalty value.

Optional prior regularization uses the canonical bound widths to avoid mixing parameter units:

```text
L_reg = lambda_reg sum_j ((x_j - x_prior,j)/(upper_j-lower_j))^2.
```

It is disabled by default and requires an explicit prior. Like the Feller term, its residual slot is
fixed whenever enabled. This feature supports stable warm-start research without claiming that the
prior is a Bayesian posterior.

## Failed evaluations and integration accuracy

An integration failure, invalid price, or IV-inversion failure creates a finite configured residual
for that observation. The optimizer continues; failures are counted and detailed residuals carry
the error message. A selected solution containing a failed model evaluation is marked
`converged=false` even if SciPy stopped successfully. No NaN or infinity is returned to SciPy for a
finite bounded vector.

Calibration-specific modes select exact numerical controls:

| Mode | Fourier upper bound | Segment | Abs tol | Rel/tail tol | Quad limit | IV price/vol tol |
|---|---:|---:|---:|---:|---:|---:|
| `FAST` | 200 | 25 | `1e-6` | `1e-6` | 100 | `1e-7` |
| `STANDARD` | 400 | 20 | `1e-8` | `1e-7` | 200 | `1e-9` |
| `HIGH_ACCURACY` | 600 | 20 | `1e-10` | `1e-9`/`1e-10` | 300 | `1e-11` |

An explicit `HestonIntegrationConfig` overrides these presets. Accuracy modes do not secretly alter
the optimizer budget.

## Initialization and optimizers

The heuristic uses the nearest-to-ATM short-maturity IV squared for `v0`, the corresponding
long-maturity value for `theta`, `kappa=1.5`, and `sigma_v=0.4`, clipped inside the active bounds.
It estimates a simple IV/log-moneyness slope and maps it to a bounded initial `rho`, defaulting to
`-0.5` only when a slope cannot be estimated. This is an initializer, not a market model.

`initial_parameters` is the user or previous calibration warm start. It is first when supplied.
The heuristic follows when distinct, and additional starts are drawn deterministically within the
bounds. Positive parameters use log-uniform draws, correlation uses uniform draws, and child seeds
come from the project's `SeedSequence` utility. `multi_start_count` is the total number of local
starts; the best configured objective wins.

The optimizer policies are:

- `LEAST_SQUARES`: bounded SciPy `least_squares` from every selected start.
- `DIFFERENTIAL_EVOLUTION`: deterministic single-process global search, optionally followed by
  SciPy's bounded polish when `local_refinement=true`.
- `HYBRID`: unpolished differential evolution followed by bounded least-squares from its global
  best. Extra requested starts may also receive local refinement. The global candidate remains in
  the candidate set, so refinement cannot force a worse reported objective.

Differential evolution uses one worker and immediate updates. Its seed, population multiplier,
iteration cap, tolerance, and polish policy are explicit. Least-squares reports status, message,
function/Jacobian evaluations, SciPy cost, first-order optimality, and runtime. Each retained start
summary includes its stage, endpoints, objective, convergence, work, cost, and optimality.

The objective owns an exact parameter-vector cache for its lifetime. A repeated vector reuses every
contract price and IV. Within a new vector evaluation, contracts are grouped by spot and maturity.
A scoped analytical batch pricer shares Phase 13 characteristic-function values across the adaptive
integrals in that group, while each contract keeps its own quadrature and failure boundary. The
batch object fixes market, parameters, integration settings, spot, and maturity at construction and
rejects mismatched contracts, so its cache cannot become stale. No cache crosses a parameter vector,
calibration, or request. The implementation is deterministic and single process.

## Residual and fit diagnostics

Each residual includes original index, contract coordinates, market/model price and IV, raw and
weighted residual, weight, `K/S`, `log(K/F)`, target, and any evaluation failure. Output counts cap
detailed residuals and start diagnostics without altering the fit.

Per-expiry diagnostics report count, RMSE, MAE, weighted RMSE, maximum error, average IV error, and
objective contribution. Per-moneyness diagnostics use signed OTM moneyness: call `k`, put `-k`.
The fixed buckets are deep OTM `(0.20, infinity)`, OTM `(0.05, 0.20]`, ATM `[-0.05,0.05]`, and ITM
below `-0.05`.

The fitted-surface service evaluates every retained market coordinate under fixed Heston
parameters. It returns ordered market/model prices, market/model IV, both residuals, nearest-forward
ATM points per maturity, and linear IV-versus-log-forward-moneyness slopes for market and model.
This is a model-fit grid; it does not replace the Phase 6 interpolation surface.

## Identifiability and local uncertainty

A central/one-sided finite-difference Jacobian of weighted data residuals is computed at the fitted
or supplied vector. Singular values, numerical rank, and the full-rank condition number expose
locally flat or correlated directions. Calibration remains valid when rank deficient.

When the Jacobian has full column rank and `n > 5`, the implementation forms the local linear
approximation

```text
s^2 = e^T e / (n-5)
Cov(x_hat) approximately s^2 (J^T J)^-1.
```

It also derives the corresponding correlation matrix. Singular, underdetermined, or non-finite
cases return `covariance_available=false`. This is explicitly a local least-squares approximation,
not posterior uncertainty and not a guarantee of global identification.

## Stability and structured holdouts

Stability analysis repeats local calibration with deterministic distinct bounded starts. It returns
every run's seed, parameters, objective, convergence, evaluations, and runtime plus best/worst
indices and population-standard-deviation dispersion for objectives and parameters. Failed or flat
starting regions remain visible rather than being hidden.

Holdout validation preprocesses once and splits the accepted coordinates structurally:

- `EXPIRY` holds out one or more complete maturities.
- `EVERY_NTH_STRIKE` holds out ordered strike positions inside every maturity.
- `WINGS` holds out observations with `abs(log(K/F))` at or beyond a configured boundary.

Both training and validation sets must be nonempty. The engine calibrates only the training set and
then applies the same target, weighting, and cleaning policy to the untouched validation set. It
returns training/validation counts, validation residuals, and full validation metrics. No default
random option-by-option split destroys the surface structure.

## Reproducibility, cost, and safeguards

Random starts and differential evolution are rooted in the configured seed. Identical dataset,
configuration, seed, SciPy/NumPy versions, and supported numerical environment reproduce the fitted
parameters and objective. Wall-clock fields naturally vary. Results distinguish objective calls,
individual option-pricing calls, failed evaluations, cache hits, runtime, and prices per second.

The HTTP layer defaults to at most 250 observations, 10 starts, 500 optimizer iterations, 500 input
surface points, and 500 returned diagnostic points. Stability runs reuse the start limit. These caps
do not constrain direct library use. Full path arrays, Fourier integrands, optimizer populations,
and caches never cross the API boundary.

Endpoints are:

- `POST /api/calibration/heston`
- `POST /api/calibration/heston/surface`
- `POST /api/analytics/heston/calibration/diagnostics`
- `POST /api/analytics/heston/calibration/fitted-surface`
- `POST /api/analytics/heston/calibration/stability`
- `POST /api/analytics/heston/calibration/validate`

## Limitations and Phase 15 boundary

Calibration assumes European prices, flat `r` and `q`, the Phase 13 constant-parameter risk-neutral
Heston model, and European BSM IV as the quote convention. Market midpoints are research inputs and
need not be executable. Parameter recovery can be weak even when prices fit closely; diagnostics
must be read with the bounds, objective, and data coverage. Local covariance omits market-data,
model, and numerical uncertainty. Robust losses, weights, penalties, and filters define materially
different estimation problems and must be recorded with a parameter set.

Phase 14 adds no SABR, local-volatility, jump, regime, American-Heston, risk, QMC, frontend, or
real-time service. Phase 15 may compare models using these calibration outputs; it must not require
replacement of the Phase 13 pricing engine or Phase 14 dataset, objective, and diagnostic
boundaries.
