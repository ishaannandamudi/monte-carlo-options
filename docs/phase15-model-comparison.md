# Phase 15 model comparison and model risk

## 1. Purpose and scope

Phase 15 compares the pricing engines delivered in Phases 1–14. It answers research questions about
valuation dispersion, fit, runtime, numerical agreement, and calibration sensitivity. It does not
select a universally best model and it does not turn model dispersion into a probabilistic
confidence interval.

The implementation has four boundaries:

- `models/model_comparison.py` contains immutable request, result, flag, fit, frontier, stability,
  and uncertainty contracts.
- `model_comparison/registry.py` is the ordered model registry and compatibility authority.
- `analytics/model_comparison.py` delegates to existing engines and computes cross-model evidence.
- `api/model_comparison.py` checks HTTP work limits and returns bounded serializable results.

No Phase 1–14 pricer is replaced. Merton, Bates, SABR, local volatility, regime models, portfolio
risk, QMC, parallelism, GPU work, trading integration, and a frontend remain outside this phase.

## 2. Registry and capability matrix

`PricingModel` has seven entries in stable enum order. Registry metadata is factual and public.

| Model | Exercise | Determinism | Greeks exposed | Stochastic vol | Calibration | Cost class |
|---|---|---|---|---|---|---|
| `BLACK_SCHOLES` | European | deterministic | Delta/Gamma/Vega/Theta/Rho | no | no | low |
| `MONTE_CARLO_GBM` | European | stochastic | Phase 3 finite difference | no | no | medium |
| `CRR_BINOMIAL` | European | deterministic | Delta/Gamma/Theta | no | no | medium |
| `HESTON_ANALYTICAL` | European | deterministic | Delta/Gamma/Theta/Rho FD | yes | yes | high |
| `HESTON_MONTE_CARLO` | European | stochastic | unavailable | yes | yes | very high |
| `CRR_AMERICAN` | American | deterministic | Delta/Gamma/Theta | no | no | medium |
| `LONGSTAFF_SCHWARTZ` | American | stochastic | unavailable | no | no | very high |

Every entry also records call, put and continuous-dividend support, state-variable count,
stochastic-factor count, calibrated-parameter count, integration/simulation requirements, and
early-exercise capability. A calibrated parameter count is objective complexity metadata; for BSM
and GBM the constant volatility input counts as one model parameter even though Phase 15 does not
calibrate it.

## 3. Compatibility checking

The registry checks exercise style before calling an engine. European-only engines are excluded
from American requests and American-only engines are excluded from European requests. Heston
engines also require either explicit `HestonParameters` or a `HestonCalibrationResult`; the latter's
calibrated vector is used directly. Phase 15 never silently substitutes BSM for Heston, European CRR
for American CRR, or CRR for LSM.

An incompatible pair produces `ModelExclusion(status=EXCLUDED, reason=...)`. A compatible engine
that throws during valuation produces `ModelOutput(status=FAILED, error_type, message)`. Successful
models remain in the response. This distinction lets research code tell an unsupported request from
a numerical failure.

## 4. Comparison request and reproducibility

`ModelComparisonRequest` contains the contract, market, selected-model policy, optional Heston
parameters/calibration, and the existing Phase 1, 7, 9, and 13 numerical configs. The comparison
policy records metrics, reference model, pairwise/Greek/IV switches, normalization, thresholds,
timing repetitions, deterministic root seed, and result cap.

Stochastic models receive distinct children of `SeedSequence(root_seed)` in selected-model order.
GBM and Heston never share RNG state because their drivers and dimensions differ. Repeated timing
runs recreate the same model-local seed, so numerical estimates are unchanged while runtime uses
the median. The existing Phase 3 common-random-number implementation remains responsible for
within-GBM bump comparisons. The policy never forces CRN across different processes.

`deterministic_signature` hashes normalized sorted JSON with SHA-256. It includes numerical inputs
and policies and excludes runtime, timestamps, environment settings, credentials, and provider
payloads. The experiment identifier is the first 16 hex characters prefixed by
`model-comparison-`. `ExperimentRecord.timestamp` is UTC wall time and is not part of the signature.

## 5. Per-model output and runtime

`ModelOutput` returns model, status, price, median runtime, deterministic/stochastic classification,
optional SE and CI, available Greeks, equivalent IV, optional calibration provenance, paths, time
steps, tree steps, early-exercise fraction, warnings, and failure details. No arrays, paths,
exercise matrices, or characteristic-function values cross the public boundary.

One-contract latency is the median of `runtime_repetitions` complete calls. Grid workload runtime is
the accumulated model pricing time in `ModelMarketFit`; benchmark reports also time 10- and
100-contract workloads. Runtime comparisons are host-dependent measurements. They are not mixed
with price, fit, or uncertainty measures.

## 6. Price comparison and reference semantics

Only successful prices enter dispersion. `PriceDispersion` contains population minimum, maximum,
range, mean, median, and population standard deviation (`ddof=0`). The model-risk range is

```text
max successful model price - min successful model price.
```

Relative range divides by one explicit positive floored denominator:

- `MEDIAN_PRICE`: `max(abs(median_price), floor)`; this is the default;
- `SPOT`: spot, which is already strictly positive;
- `REFERENCE_PRICE`: `max(abs(reference_price), floor)` if the reference succeeded, otherwise the
  floored median.

The default floor is `1e-8`. This avoids explosive ratios for nearly worthless options. The range
is descriptive variation over selected assumptions, not a confidence interval and not an estimate
of a probability distribution.

A reference model is a reporting convention. For each successful model, signed, absolute, and
floored-relative deviations are returned. A missing, excluded, or failed reference simply leaves
the reference-deviation collection empty. It does not invalidate independent model prices.

## 7. Pairwise differences

For successful models in request order, the `N x N` price matrix contains

```text
signed[i,j] = price_i - price_j
relative[i,j] = (price_i - price_j) / max(abs(price_j), floor).
```

The diagonal is zero and the signed matrix is antisymmetric. The relative matrix is directional
because its denominator is model `j`. Equivalent implied volatilities receive the same bounded
pairwise representation with a volatility floor of `1e-8`.

## 8. Greek comparison

Greek statistics use only available finite values. Missing Greeks remain `None` and are never
imputed. For each of Delta, Gamma, Vega, Theta, and Rho the response contains population
dispersion, contributing models, and an `N x N` signed matrix.

BSM reuses exact Phase 3 Greeks. European and American CRR reuse native lattice Delta, Gamma, and
calendar-time Theta. GBM Monte Carlo delegates to Phase 3 common-random-number finite differences
for all five requested Greeks and records a warning rather than failing the price when a Greek
calculation is unavailable.

Heston analytical Delta and Gamma use central spot differences with relative bump `1e-3`. Rho uses
a central absolute rate bump `1e-4`. Calendar-time Theta is minus the central maturity derivative,
using a `1e-4` year bump bounded away from negative maturity. Bumps are configurable.

Heston generic Vega is intentionally absent. `dV/dv0`, `dV/dsqrt(v0)`, and sensitivity to a BSM
market-volatility proxy have different units and interpretations. Calling any one ordinary Vega
would misstate comparability. Heston MC and LSM Greeks also remain unavailable because Phase 15 does
not add process-specific estimators solely to fill a table.

## 9. Equivalent implied volatility

For each successful European price with positive maturity, the Phase 4 solver returns the BSM
volatility that reconstructs that price under the contract's rate and dividend yield. Inversion
failure leaves IV unavailable while preserving price. Single-contract IV residual is
`model_equivalent_iv - request.market.volatility`.

This common scale does not assert that CRR or Heston has constant volatility. BSM-versus-Heston is
meaningful only with a named BSM input. In single-contract requests that input is exactly
`MarketEnvironment.volatility`. In a surface request, the BSM-equivalent baseline uses each Phase 6
observed IV, so its fit should be numerical zero by construction.

## 10. Market observations, filtering, and weights

Market comparison reuses `HestonCalibrationDataset` and `HestonCalibrationObservation`. It therefore
supports price, midpoint, bid, ask, observed IV, source, timestamps, arbitrary dataset metadata,
custom weight, freshness, arbitrage validity, and quality issue strings without duplicating Phase
14 conversion logic. An optional observation-index-to-label map adds user-supplied descriptive
groups such as `CALM` or `STRESSED`; Phase 15 computes the same fit metrics per label without
inferring or predicting a regime.

`prepare_calibration_data` performs the same filters, OTM/all selection, European price-bound check,
duplicate resolution, target derivation, weighting, and optional expiry balancing used by Heston
calibration. Retained weights are normalized to mean one. Results expose retained/excluded counts
and exclusion reasons. User-provided dataset metadata, including regime labels, remains attached to
the source dataset; Phase 15 does not infer regimes or run an HMM.

## 11. Price and bid/ask fit

For successful observation evaluations, price residual is `model_price - market_price`. Aggregate
metrics are:

- RMSE and MAE;
- weighted RMSE using Phase 14 normalized nonnegative weights;
- maximum and median absolute error;
- signed bias, the mean residual.

The same points also report a normalized residual
`(model_price-market_price)/max(abs(market_price), relative_price_floor)`. Aggregate normalized
RMSE and MAE therefore remain finite for near-zero option prices; they never divide blindly by the
observed price.

An observation with both bid and ask has zero spread violation when
`bid <= model_price <= ask`. Below bid, distance is `bid-model_price`; above ask it is
`model_price-ask`. The result reports eligible count, inside-spread fraction, average distance
outside including in-spread zeros, and maximum distance. Missing spreads do not enter these
statistics. No midpoint or spread is described as executable liquidity.

## 12. IV, expiry, and moneyness fit

Model prices are inverted to equivalent IV and compared with observed IV in absolute volatility
units. IV RMSE, MAE, maximum absolute error, and signed bias use only successful inversions. A
display layer may multiply these values by 10,000 for volatility basis points, but stored values
remain decimal volatility.

Per-expiry metrics group exact maturity values. Per-moneyness diagnostics reuse the Phase 14 signed
log-forward-moneyness convention: call `k=log(K/F)`, put `-k`; deep OTM is above `0.20`, OTM is
above `0.05`, ATM is `[-0.05,0.05]`, and the remainder is ITM. Each group repeats the full fit
metric schema.

## 13. ATM, skew, curvature, and surfaces

For every maturity with sufficient IV support, market and model values use the Phase 6 definition:

```text
ATM       = sigma(k=0)
skew      = [sigma(+h)-sigma(-h)] / (2h)
curvature = [sigma(+h)-2 sigma(0)+sigma(-h)] / h^2
h         = 0.01
```

Interpolation is PCHIP in total variance `T*sigma^2`, exactly as in Phase 6. If the smile does not
cover ATM or both bump points, the relevant values stay unavailable. Residuals are model minus
market.

`compare_surface` flattens a Phase 6 surface through its existing conversion function, preserving
observed prices, IVs, option types, quotes, source, timestamp, and quality. The bounded residual list
is the price/IV residual grid. Heston analytical pricing uses one
`HestonAnalyticalBatchPricer` per maturity so characteristic-function values are cached only inside
that parameter/maturity workload. No unsafe global cache exists.

## 14. Structured out-of-sample comparison

`compare_models_out_of_sample` calls the public Phase 14 split function for complete-expiry,
every-nth-strike, or wing holdouts. It evaluates the same supplied model parameters separately on
the retained and untouched datasets and returns distinct metric objects. Phase 15 does not silently
refit a model: callers that want trained Heston parameters supply the Phase 14 calibration result
from the training set. This keeps calibration policy explicit and prevents accidental leakage.

## 15. Complexity versus fit and calibration cost

`ModelComplexityFit` reports state variables, stochastic factors, calibrated-parameter count,
whether calibration is required, calibration runtime, one-contract pricing runtime, in-sample RMSE,
and optional out-of-sample RMSE. It has no rank or subjective score.

When a Phase 14 result supplies Heston parameters, `ModelCalibrationMetadata` records calibration
runtime, objective evaluations, observations, prices per second, fitted RMSE, and provenance. An
explicit Heston vector is labeled `explicit_parameters`. Non-calibrated models show
`calibration_required=false` through registry/complexity metadata rather than a fictitious zero-cost
fit.

## 16. Stochastic cost and accuracy/compute frontier

Monte Carlo output reports path count, Heston/LSM time steps, runtime, SE, and CI. Generalized pair
diagnostics return GBM MC versus BSM and Heston MC versus Heston analytical errors, z-scores when SE
is positive, and runtime ratios when both runtimes are positive. American LSM versus CRR similarly
retains LSM SE, exercise fraction, and runtime.

`compute_frontier` accepts sorted unique budgets and one explicit reference. Natural variables are:

- GBM MC: paths;
- Heston MC: paths at fixed time steps, or time steps at fixed paths;
- CRR: tree steps;
- LSM: valuation paths at a fixed exercise grid, or exercise-grid time steps at fixed paths.

Every point gets an independent deterministic child seed. It reports price, reference price,
absolute/floored-relative error, runtime, SE and CI width if stochastic, and a conservative array or
lattice byte estimate. The output never chooses an optimum. Running the function once per model
provides unified convergence evidence while respecting each numerical method's natural variable.

## 17. Input stability

`analyze_model_stability` bumps spot, volatility or Heston `v0`, rate, dividend yield, and maturity.
Positive-domain inputs use bounded central differences where possible; exact-zero maturity uses a
forward difference. Heston `v0` is bumped as variance and is labeled accordingly. Each row contains
base/down/up prices, price changes, a central or forward sensitivity, status, and failure message.
This is descriptive numerical/input stability, not calibration stability and not a replacement for
the canonical Greek definitions.

## 18. Calibration stability and ensemble dispersion

A Phase 14 `HestonCalibrationStabilityResult` supplies parameter dispersion and each run's fitted
vector. Phase 15 prices the target contract under every valid solution and returns parameter
provenance, individual prices/IVs, and population min/max/range/mean/median/standard deviation.
Callers may also supply an arbitrary collection of valid parameter vectors. This captures valuation
sensitivity to calibration solutions; it does not create posterior probabilities or assign weights
to the members.

## 19. Local parameter-uncertainty propagation

When a finite Phase 14 `5 x 5` local covariance is supplied, Phase 15 computes central finite
differences in canonical order `[v0,kappa,theta,sigma_v,rho]` and evaluates

```text
approximate Var(V) = gradient(V)^T Cov(theta) gradient(V).
```

Positive parameters are bounded at zero and rho inside `(-1,1)`. Negative roundoff variance is
floored at zero. The result is explicitly labeled a local first-order delta-method approximation.
It is not called a confidence interval and can be unreliable for weak identification, parameter
bounds, nonlinear regions, or non-Gaussian estimator behavior.

## 20. Uncertainty decomposition

`UncertaintyDecomposition` deliberately separates:

- Monte Carlo sampling: conditional estimator SE by stochastic model;
- numerical discretization: observed deterministic/numerical comparator differences;
- calibration: price dispersion over Heston fits or the optional local covariance result;
- model: price dispersion across successful selected models;
- market quote: observed bid/ask width summaries.

These quantities answer different questions and are not added, root-sum-squared, or converted into
a synthetic interval. Heston MC confidence intervals exclude time-discretization bias. LSM SE
excludes training-policy and exercise-grid uncertainty, consistent with Phase 9.

## 21. Model-risk report and flags

`ModelRiskReport` contains the single-contract comparison, optional market and holdout fit,
calibration ensemble, optional covariance propagation, decomposed uncertainty, complexity-versus-
fit rows, flags, and warnings. Flags are descriptive threshold crossings:

- `LARGE_PRICE_DISPERSION` for relative model range;
- `LARGE_GREEK_DISPERSION` for Delta range;
- `LARGE_IV_DISPERSION` for equivalent-IV range;
- `CALIBRATION_UNSTABLE` for a parameter coefficient of variation;
- `MODEL_OUTSIDE_BID_ASK` for any retained model quote outside an available spread;
- `HIGH_MONTE_CARLO_UNCERTAINTY` for relative SE;
- `POOR_OUT_OF_SAMPLE_FIT` for held-out RMSE;
- `NUMERICAL_CONSISTENCY_WARNING` for failed equivalence checks.

All thresholds live in `DispersionThresholds`. They only generate flags and never alter, discard,
blend, or rank prices.

## 22. Theoretical consistency checks

The European suite compares BSM with GBM MC, converged CRR, and Heston analytical under constant
deterministic variance (`v0=theta=sigma_BSM^2`, `sigma_v=0`). It also compares the same Heston
analytical value with FTE and QE-M Monte Carlo. Deterministic pairs use the configured absolute
tolerance; stochastic pairs use the larger of that tolerance and four estimator standard errors.

An American request compares LSM with American CRR. A failed check emits
`NUMERICAL_CONSISTENCY_WARNING`; it never changes either price. Single-comparison diagnostics also
expose BSM/GBM, BSM/CRR, Heston analytical/MC, and American CRR/LSM differences when both models are
present.

## 23. Failure isolation and serialization

Compatibility is evaluated before work. Each compatible model is then isolated in its own failure
boundary. Observation-level market fit also records failed points while aggregating successful
ones. Complete-model failure yields empty metrics with status `FAILED`; incompatibility yields
`EXCLUDED`. Result lists preserve requested order. All NumPy values are converted to Python scalars,
matrices to bounded nested lists, and errors to type/message strings.

## 24. API design and limits

The thin routes are:

| Route | Purpose |
|---|---|
| `GET /api/analytics/models/registry` | capability metadata |
| `POST /api/analytics/models/compare` | one-contract comparison |
| `POST /api/analytics/models/market-fit` | Phase 14 dataset fit |
| `POST /api/analytics/models/risk` | assembled model-risk report |
| `POST /api/analytics/models/compute-frontier` | bounded compute experiment |
| `POST /api/analytics/models/consistency` | theory-based engine checks |
| `POST /api/analytics/models/surface-comparison` | Phase 6 surface residuals |

Phase 15 settings default to 7 models, 500 observations, 1,000 surface/result points, 20 experiment
points, and 10 timing repetitions. Nested requests also obey every earlier engine limit for GBM
paths, Heston paths/steps/nodes, CRR steps, and LSM paths/steps/nodes. Direct library callers may
choose larger deliberate research workloads; settings never change mathematical results.

## 25. Limitations and Phase 16 boundary

Comparison quality depends on the selected models, BSM volatility convention, quote quality,
filtering, weights, numerical budgets, and calibration inputs. Population dispersion across a small
hand-selected engine set is not statistical model uncertainty. Equivalent IV may be unavailable at
expiry or when a numerical price violates European BSM bounds. Market fit assumes one dataset spot,
flat rate, and flat continuous dividend yield. Timing is host-specific. Local covariance is neither
Bayesian uncertainty nor robust global identification. No automated model ranking or approval is
performed.

Phase 16 may aggregate existing contract-level values and uncertainty categories into portfolio
risk. Phase 15 does not implement portfolio Greeks, scenarios, VaR, Expected Shortfall, optimization,
jump diffusion, regime detection, new volatility models, data streaming, or execution.
