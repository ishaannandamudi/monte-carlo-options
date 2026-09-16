# Phase 9: Longstaff-Schwartz American Monte Carlo

## Purpose and module boundaries

American exercise requires a decision based on the information available at each exercise date.
A terminal Monte Carlo draw cannot make that decision. Least-squares Monte Carlo (LSM) learns an
approximation to continuation value from full simulated paths, then compares continuation with
immediate exercise. Phase 9 implements single-asset American vanilla calls and puts under the same
constant-market risk-neutral GBM used by earlier phases.

The implementation extends the existing boundaries:

- `models/simulation.py`: `PathSimulationConfig`, restricted to NONE and ANTITHETIC.
- `simulation/base.py`: a separate `PathStochasticProcess` capability. Existing terminal-only
  processes need no new method. `GBMProcess` implements both capabilities.
- `models/longstaff_schwartz.py`: configuration, frozen date policies, regression and boundary
  diagnostics, pricing results, and experiment results.
- `pricing/lsm_regression.py`: dimensionless bases, least squares, and continuation predictions.
- `pricing/longstaff_schwartz.py`: fixed-path backward fitting, forward policy evaluation, stopping
  cash flows, deterministic cases, and the simulation-driven pricer.
- `analytics/longstaff_schwartz.py`: CRR comparisons and path, time-step, and basis experiments.
- `api/longstaff_schwartz.py`: typed request validation and aggregate resource limits.

The original terminal simulation and European-only pricing/Greek services are unchanged.

## Full-path GBM and exercise grid

The path matrix is float64 with shape `(num_simulations, num_time_steps + 1)`. Column zero equals
the supplied spot exactly. For `dt=T/M`, independent increments give

```text
log S(i,j+1) = log S(i,j) + (r-q-sigma²/2) dt + sigma sqrt(dt) Z(i,j).
```

The implementation forms cumulative log increments and exponentiates them. These are exact GBM
transitions; there is no Euler discretization. Each path's normal vector occupies a row. For
antithetic sampling, adjacent rows have driver vectors `Z` and `-Z` at every time step. An odd
requested count leaves one independently generated final primary row as a singleton. No extra
path is returned.

Exercise is permitted at root and every simulated date by default. Optional `exercise_steps`
contains a strictly increasing set of positive column indices and must end at `M`. Root is always
an available right. This is a discrete-grid approximation to continuously exercisable American
value, not an exact continuous-time American solution. The optional schedule does not introduce a
new Bermudan contract type.

All drivers use existing isolated PCG64 Generators and SeedSequence child seeds. A path simulation
does not promise the same terminal realization as the older one-draw terminal simulator: their
normal allocations differ. Their terminal distributions agree.

## Configuration and defaults

`LongstaffSchwartzConfig` defaults to 25,000 valuation paths, 50 steps, seed 42, degree-two POWER
basis, ANTITHETIC sampling, 95% confidence, and OUT_OF_SAMPLE evaluation. `regression_paths=None`
uses the valuation count for training. It may be set independently in out-of-sample mode; in-sample
mode rejects a separate training count.

Other explicit defaults are 20 minimum regression observations, `1e-12` absolute exercise
tolerance in payoff units, five minimum boundary observations, and `1e8` for a condition-number
warning. Basis degree is limited to 0 through 5. Detailed regression output is off by default,
boundary output is on, and each diagnostic list returns at most 100 dates by default. These output
switches never change the fitted policy or price.

## Terminal initialization and backward induction

For each path `i`, initialize the selected exercise index `tau_i=M` and cash flow

```text
F_i = h(S(i,M)),
h_call(S) = max(S-K,0),     h_put(S) = max(K-S,0).
```

`F_i` is in dollars at its selected exercise date, not already discounted. At each eligible date
`j=M-1,...,1`, select paths with strictly positive current intrinsic and a selected stop later than
`j`. Call candidates satisfy `S(i,j)>K`; put candidates satisfy `S(i,j)<K`. Equality is OTM.

During backward training a path selected to exercise at a *later* date must remain eligible at
earlier dates. That later cash flow is the continuation alternative. Removing it from training
would incorrectly lose the opportunity to replace a later stop with an earlier one. All selected
stops therefore stay later than the currently processed date until that date's decisions are made.

The response for each current candidate is its realized future cash flow discounted to date `j`:

```text
Y_i(j) = F_i exp[-r (tau_i-j) dt].
```

This uses the currently selected stopping index, including any previous backward replacement. It
does not continually regress a discounted terminal payoff after an earlier exercise was selected.

## Current-state bases and least squares

Only current stock enters the feature vector. Normalize it as `x=S(i,j)/K`. This dimensionless
state avoids raw large stock powers and gives the basis a consistent moneyness interpretation.
POWER uses `1,x,...,x^d`. LAGUERRE uses the standard *unweighted* Laguerre polynomials:

```text
L0(x)=1
L1(x)=1-x
L2(x)=1-2x+x²/2
L3(x)=1-3x+3x²/2-x³/6
(n+1)L(n+1)(x)=(2n+1-x)Ln(x)-n L(n-1)(x).
```

NumPy's polynomial Vandermonde functions generate these columns through degree five. There is no
undocumented exponential weight. POWER and unweighted LAGUERRE span the same polynomial space at
the same degree, so exact-arithmetic fits are equivalent; basis comparisons primarily reveal
numerical conditioning. Increasing degree need not improve a policy's valuation performance.

`numpy.linalg.lstsq(X,Y,rcond=None)` solves the least-squares problem without forming or inverting
`X.T @ X`. Diagnostics include ITM count, sample count, number of columns, numerical rank,
coefficients, residual sum of squares, and the singular-value ratio as condition number. An
unrepresentable/infinite condition ratio is returned as unavailable and flagged, not serialized as
infinity. A large but finite condition number alone does not reject a full-rank fit.

At least `max(min_regression_paths,d+1)` observations are required. No ITM observations, insufficient
support, rank deficiency, or a least-squares solver failure each produce a typed fallback reason.
The fallback is **continue all candidates at this date**. It never fabricates coefficients or an
exercise opportunity from an unreliable fit. This can forgo useful early exercise and lower the
expected value of the learned policy. Non-finite data or arithmetic instead raise domain errors.

Predicted continuation is floored at zero because future vanilla cash flows are nonnegative.
This is an explicit prediction rule, not a hidden refit.

## Exercise decisions and stopping-time replacement

For each fitted candidate, exercise when

```text
h(S(i,j)) > max(X_i beta_j,0) + exercise_tolerance.
```

Exercise sets `F_i=h(S(i,j))` and `tau_i=j`. Continuing leaves both values untouched. OTM paths
always continue. Here tolerance affects the actual learned stopping decision, unlike Phase 8's
diagnostic-only tolerance around the exact node maximum.

After backward fitting, discounted realized values at root are

```text
D_i = F_i exp(-r tau_i dt),
continuation_0 = mean(D_i).
```

There is no second discount on an already-discounted stored cash flow. Tests use a four-row fixed
matrix and a rate `log(2)` to verify the exact half-period discounting, coefficients, earlier
replacements, OTM continuation, and final values `[10,2.5,5,0]` with mean `4.375`.

## Root exercise and evaluation modes

Root needs no regression because all paths start at the same spot. In training, if intrinsic is
strictly greater than estimated root continuation, the root policy exercises immediately. Every
cash flow and stopping index is replaced by intrinsic and zero. Ties continue.

IN_SAMPLE reports the resulting backward sample. Its price is the maximum of intrinsic and the
estimated future continuation. Shared regressions and root selection can introduce look-ahead and
selection bias; ordinary path-level dispersion does not measure that bias.

OUT_OF_SAMPLE is the default. Child zero of the root SeedSequence produces training paths; child
one produces an independent valuation matrix. The training matrix is released before the valuation
matrix is allocated. Each frozen date stores basis family/degree, S/K normalization, coefficients,
fit count, rank, and fallback status. The policy also freezes the root decision. Applying it to
valuation paths proceeds *forward*: among alive ITM paths, stop at the first date whose rule says
exercise. Valuation never refits coefficients and never uses a path's future state as an input.

Freezing the root in training is necessary for independent policy valuation. Reapplying
`max(intrinsic, valuation_mean)` would introduce a new selection bias into that held-out sample.
Consequently, if training chose to continue, an out-of-sample realization can lie below intrinsic;
`below_intrinsic` reports this without clipping. If training selected root exercise, valuation is
exact intrinsic with zero sampling error. This distinction is explicit in the result's uncertainty
note. The same is true of a noisy estimate below the European reference.

For a no-dividend call with `r>=0`, the known no-early-exercise theorem supplies an exact
continue-to-maturity policy. The result flags this safeguard and skips regressions. It does not
apply to negative rates or positive dividend yield. Dividend-paying calls and puts use LSM.

## Deterministic cases

At `T=0`, public pricing returns intrinsic without path generation or regression, with zero SE,
zero early-exercise fraction, and no boundary. Positive intrinsic is classified as maturity
exercise at time zero; zero intrinsic is a zero cash flow.

At `sigma=0`, compute only the deterministic states `S(t)=S0 exp((r-q)t)` at the configured exercise
dates plus root. Select the largest `exp(-rt)h(S(t))`. Equal positive maxima choose the earliest
date; an everywhere-zero payoff expires at maturity. This agrees with Phase 8 on the same grid.
No simulated observations or empirical boundary are invented. Exercise-time fractions and the
selected time describe the deterministic decision. The result reports zero simulation memory and
runtime, no regressions, and zero SE.

## Estimator, uncertainty, and exercise summaries

Zero-cash-flow expiration paths remain in the mean and SE. For NONE, use the Bessel-corrected
standard deviation of all valuation `D_i` divided by `sqrt(N)`. For even ANTITHETIC samples, average
adjacent discounted cash flows within each pair, then compute sample standard deviation of those
pair means divided by `sqrt(number_of_pairs)`.

Odd counts reuse the Phase 1 independent-cluster estimate. With cluster totals `A_c`, sizes `n_c`
(two per pair, one for the singleton), path mean `mu`, and `C` clusters:

```text
Var(mu_hat) = C/(C-1) * sum_c (A_c-n_c mu_hat)^2 / N².
```

This retains the singleton's one-path weight. A stochastic estimator needs at least two independent
observations/clusters. Its normal interval is `price ± norm.ppf((1+confidence_level)/2)*SE`.

For independent valuation these are conditional sampling diagnostics for the *frozen policy*.
They omit variation due to training, finite basis, finite exercise grid, and model assumptions.
They are not confidence intervals for the optimal American value. In-sample SE is explicitly
descriptive because common fitted regressions also couple clusters. Antithetic sampling is not
assumed to reduce variance for every stopping rule.

All exercise fractions use the total valuation sample as denominator. `early_exercise_fraction`
includes immediate positive exercise; `maturity_exercise_fraction` includes positive maturity
cash flows; `zero_cashflow_fraction` includes all zero payoffs. These three sum to one.
`immediate_exercise_fraction` is also reported as a subset of early exercise. Mean/median exercise
times exclude zero-payoff paths; if none pays, both are unavailable.

## Empirical boundaries and regression output

Boundary extraction uses only the **final realized valuation stops**, not interim training
classifications. At each nonterminal date with enough positive exercise observations, the put
boundary is the highest exercised stock and the call boundary is the lowest. Report index, time,
time remaining, option side, direction, exercise count, and fraction of the full valuation sample.
Maturity is excluded; root can be included. Dates below `min_boundary_observations` are omitted.
No interpolation, unsupported frontier, or smooth-boundary claim is made. Sample extrema can be
noisy and regressions may not produce a single contiguous theoretical region.

Detailed regression dates and boundaries are sorted chronologically. Output caps keep the earliest
dates and include explicit truncation flags. Summary fit/fallback/conditioning counts use all dates
even when details are omitted or truncated.

## CRR comparison and experiments

Comparison uses the same contract and constant market inputs with a configurable CRR resolution
(default 1,000 steps). CRR is a discrete-tree benchmark, not exact continuous-time truth. The
comparison returns signed/absolute/relative error and an optional error divided by LSM SE. That
normalized error is descriptive: it also contains policy and grid error.

Black-Scholes supplies a separately labeled European reference. LSM minus this reference is a
noisy premium diagnostic and may be negative. The CRR premium retains Phase 8's definition against
its same-tree European comparator. A confidence-high diagnostic shows whether the LSM interval is
entirely below the European reference; pricing is not rejected on that basis.

The comparison endpoint optionally accepts `include_boundary_comparison=true`. It enables LSM
boundary collection and runs the existing public CRR boundary service in an additional tree pass.
Only exactly shared grid dates with supported boundaries in both results are matched. Matching
uses integer arithmetic: LSM index `j` matches CRR index `k` only when `j*N_tree = k*M_lsm`.
There is no nearest-date matching or interpolation. Each point reports both stocks, their signed
and absolute difference, and the LSM observation count and support fraction. `null` means the
diagnostic was not requested; an empty list means no supported common boundary dates exist.
LSM output caps and its truncation flag also delimit this comparison. Different exercise grids,
regression noise, and tree stock spacing can explain differences; equality is not required.

PATH_COUNT and TIME_STEPS convergence validate the complete grid before computing. Counts are
sorted and assigned deterministic independent child seeds. Each child then creates its own
training and valuation streams. In path-count experiments, training also scales with valuation
unless `regression_paths` is explicitly fixed. Time-step convergence requires the default all-date
schedule rather than ambiguously remapping a custom subset. The CRR benchmark resolution stays
fixed across a run.

Basis comparisons use the same seed and path sets across POWER/LAGUERRE and requested degrees.
Each point reports price, SE, CI, CRR error, exercise fraction, timing, successful fits, fallback
count, and conditioning summary. No monotone price/error claim or naive `N^-1/2` regression-bias
claim is made.

## Complexity, resources, API, and validation

For `N` paths, `M` dates, and `p=d+1` basis terms, simulation and matrix storage are O(NM).
Regressions cost approximately O(M N_ITM p²); basis degree is capped. All paths within a date are
processed with NumPy arrays. Python loops run over exercise dates or experiment configurations.
There are no objects or loops per simulated path. Current normal, log-increment, and path buffers
make peak allocation a multiple of the path-matrix size; reported `approximate_path_array_bytes`
is only `8*max(training_paths,valuation_paths)*(M+1)`, not peak process RSS.

Defaults for HTTP resources are:

```text
MAX_LSM_PATHS=500000                  # each training or valuation set
MAX_LSM_TIME_STEPS=500
MAX_LSM_TOTAL_PATH_NODES=25000000     # sum across training, valuation, and every experiment point
MAX_LSM_CONVERGENCE_POINTS=20
MAX_LSM_DIAGNOSTIC_DATES=100          # each optional output list
```

The node budget deliberately counts both sets even though their full matrices are allocated
sequentially. In-sample mode counts its single set once. The API preflights all experiments and
also enforces existing CRR limits. Diagnostic date caps affect output only. Direct library calls
use their explicit configuration and are not controlled by application environment settings.

Endpoints are `POST /api/pricing/longstaff-schwartz`, and `POST` under
`/api/analytics/longstaff-schwartz` for `/compare`, `/convergence`, and `/basis-comparison`.
Requests contain an American `contract`, `market`, and `lsm` configuration; benchmark requests also
accept `tree`. Detailed regressions are optional in the price result, avoiding a duplicate endpoint.
The API never returns path matrices or per-path cash flows. European requests fail clearly.

Tests verify exact GBM increments and moments, odd/even antithetic construction, analytic basis
columns, known least-squares coefficients, rank and support fallbacks, manual cash-flow replacement
and discounting, frozen policy behavior, training/valuation stream independence, correct held-out
SE, deterministic limits, supported boundaries, CRR comparisons, and resource limits. Performance
tests are opt-in for 10,000x50, 50,000x50, 100,000x50, and 50,000x100 experiments.

Recorded price, convergence, basis, and performance observations are in
[the Phase 9 validation report](../benchmarks/phase9-lsm.md). Timings are environment observations,
not service-level guarantees or test assertions.

## Deferred work

Phase 9 adds no American Greeks, American implied volatility, control variates for LSM, dual
upper-bound estimator, automatic market-chain/surface integration, exotics, running averages,
barriers, multiple assets, or stochastic-volatility state. Earlier European IV conventions remain
explicit. Later products can reuse path simulation, frozen policies, and cash-flow statistics, but
their payoff/state design and validation belong to future specifications. Phase 10 is not started.
