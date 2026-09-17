# Monte Carlo Options

Monte Carlo Options is the quantitative foundation for a long-term derivatives-research platform.
The current implementation prices European calls and puts under constant-market risk-neutral
geometric Brownian motion (GBM), with a Black-Scholes-Merton benchmark, statistically sound
uncertainty diagnostics, advanced variance reduction, first-class Greeks, and robust implied-
volatility inversion. It also normalizes real option-market snapshots and constructs queryable
volatility smiles and surfaces with static-arbitrage diagnostics. A memory-efficient CRR binomial
engine prices European and American contracts, extracts early-exercise boundaries, and supplies
native tree Greeks. Longstaff-Schwartz Monte Carlo adds full-path GBM, learned American exercise
policies, and independent policy valuation against the CRR benchmark. Phase 10 adds discretely
monitored arithmetic and geometric Asian options, an exact geometric-Asian benchmark and control
variate, plus analytical and Monte Carlo cash-or-nothing and asset-or-nothing digitals. Phase 11
adds single-barrier calls and puts with discrete monitoring, Brownian-bridge continuous correction,
rebates, hit diagnostics, continuous analytical benchmarks, and knock-in/out parity analytics.
Phase 12 adds correlated multi-asset GBM, robust correlation factorization, and weighted arithmetic
and geometric European basket options with an exact geometric benchmark and control variate.
Phase 13 adds Heston stochastic volatility with FTE and Andersen QE-M simulation, stable
characteristic-function pricing, model-implied smiles, and process/convergence diagnostics.

The inputs are continuous floating-point model values. They are not accounting ledger values or
currency-safe decimal amounts.

## Current capabilities

- Validated European/American call and put contracts and separate constant market assumptions
- Risk-neutral GBM terminal simulation with continuous dividend yield
- Plain Monte Carlo and antithetic variates
- Discounted-terminal-stock control variates and combined antithetic plus control variates
- Discounted-payoff sample variance, standard deviation, standard error, and configurable normal
  confidence intervals
- Exact pair-level uncertainty treatment for even antithetic samples and an explicit weighted
  cluster treatment for odd samples
- Black-Scholes-Merton analytical pricing with dividends
- Monte Carlo versus analytical error, z-score, and interval-containment diagnostics
- Independently seeded, reproducible convergence experiments and empirical standard-error slopes
- Fixed-budget comparisons across all four variance-reduction modes
- Analytical Delta, Gamma, Vega, calendar-time Theta, Rho, and dividend Rho
- Common-Random-Number finite-difference Greeks with direct estimator uncertainty
- Pathwise Delta and Vega with plain or antithetic sampling
- Analytical-versus-stochastic Greek comparisons and one-dimensional sensitivity grids
- Brent, bisection, Newton, and guarded AUTO implied-volatility solvers
- European no-arbitrage validation, exact-expiry policy, and reconstruction diagnostics
- Ordered batch inversion with per-observation failures and synthetic volatility-smile slices
- Provider-independent underlying, option-instrument, quote, and chain-snapshot models
- Massive (formerly Polygon.io) option-chain normalization with fully offline tests
- Structured quote-quality, staleness, spread, IV, Greek, smile, and parity diagnostics
- Canonical strike, spot-moneyness, forward-moneyness, and `log(K/F)` volatility points
- LINEAR and PCHIP smile interpolation in total implied variance
- Multi-expiration surface queries with total-variance maturity interpolation
- Explicit wing and maturity extrapolation policies, defaulting to no extrapolation
- OTM market quote selection, deterministic ATM/duplicate handling, and exclusion diagnostics
- Strike-monotonicity, irregular-grid butterfly, and calendar-total-variance diagnostics
- ATM-forward term structures and coordinate-based local skew and curvature metrics
- Cox-Ross-Rubinstein European call and put pricing with continuous dividends
- Cox-Ross-Rubinstein American call and put pricing with node-level early exercise
- Same-tree early-exercise premiums and compact exercise-boundary diagnostics
- O(N)-memory backward induction and explicit risk-neutral probability validation
- Tree-native Delta, Gamma, and calendar-time Theta from early lattice levels
- CRR convergence analytics with Black-Scholes and odd/even diagnostics
- Exact full-path GBM with reproducible interleaved antithetic paths
- Longstaff-Schwartz American pricing with POWER/LAGUERRE regression on normalized S/K
- Independent training and valuation streams, plus explicit in-sample mode
- Regression rank/fallback diagnostics, empirical stopping boundaries, and CRR comparisons
- Path-count, exercise-grid, and basis-comparison experiments
- Regular and explicit irregular Asian monitoring schedules with an explicit initial-spot policy
- Exact arbitrary-time GBM observations without a hidden fine grid
- Fixed-strike arithmetic and geometric Asian calls and puts
- Exact discrete geometric-Asian analytical pricing and geometric control variates
- Full-path antithetic pairing and pair-level control-variate uncertainty
- Cash-or-nothing and asset-or-nothing digital calls and puts
- Analytical and terminal-GBM Monte Carlo digital pricing with strict ITM equality
- Asian method comparisons, monitoring-frequency studies, and exotic convergence experiments
- Down/up and knock-in/knock-out European barrier calls and puts
- Inclusive time-zero and scheduled barrier detection with first observed hit times
- Discrete `AT_HIT` and `AT_EXPIRY` rebates plus bridge-compatible expiry rebates
- Exact log-GBM Brownian-bridge crossing probabilities on irregular intervals
- Continuous survival-probability weighting without extra Bernoulli noise
- All eight zero-rebate continuously monitored single-barrier analytical formulas
- Pathwise discrete/bridge knock-in plus knock-out parity diagnostics
- Barrier path-count, monitoring-frequency, and level-sensitivity analytics
- Validated Gaussian correlation matrices with tiny-noise PSD repair diagnostics
- Cholesky correlation factors with an eigen fallback for singular PSD matrices
- Correlated terminal and arbitrary-time full-path GBM for any supported asset count
- Weighted arithmetic and geometric European basket calls and puts
- Exact lognormal geometric-basket pricing and a geometric payoff control for arithmetic baskets
- Multivariate antithetic paths with pair-cluster and odd-singleton uncertainty
- Empirical correlation, basket convergence, correlation, and two-asset weight diagnostics
- Typed risk-neutral Heston parameters with non-rejecting Feller-condition diagnostics
- Correlated spot/variance Brownian drivers and reproducible PCG64 Heston streams
- Full-truncation Euler and martingale-corrected Andersen Quadratic-Exponential simulation
- O(paths) Heston terminal simulation and explicit full-path research simulation
- European Heston call/put Monte Carlo with pair-cluster antithetic uncertainty
- Little Heston Trap characteristic function and adaptive semi-analytical Fourier pricing
- FTE/QE comparison, path/time-step convergence, variance paths, smiles, and parameter sweeps
- Thin FastAPI endpoints with configurable resource limits
- Deterministic, numerical, statistical, and API tests

The current scope deliberately excludes surface fitting/repair, double/multi-asset barriers and
path-dependent payoffs beyond Asians and single barriers, non-CRR tree models, Heston calibration,
local volatility, jumps, portfolio risk, parallelism, a GPU backend,
quasi-Monte Carlo, and a frontend.

## Mathematical model

The terminal stock price under the risk-neutral measure is

```text
S_T = S_0 exp((r - q - 0.5 sigma^2)T + sigma sqrt(T) Z),  Z ~ N(0, 1)
```

The pricer evaluates the vectorized call payoff `max(S_T - K, 0)` or put payoff
`max(K - S_T, 0)`, then discounts every payoff by `exp(-rT)`. The price is the sample mean of those
discounted payoffs. All reported estimator uncertainty uses discounted values.

Black-Scholes-Merton supplies the exact analytical benchmark under the same assumptions. It is
used for model validation and convergence diagnostics, not to alter Monte Carlo estimates.

## Antithetic uncertainty

With antithetic sampling, each normal `Z` is followed by `-Z`. `num_simulations` always means total
terminal outcomes. An odd request returns exactly the requested count: complete pairs come first,
and one final independent positive-side draw is retained as a singleton.

For an even count, discounted payoffs are averaged within each pair. The reported standard error is
the sample standard deviation of those independent pair averages divided by the square root of the
number of pairs. Individual antithetic payoffs are correlated and are never treated as independent
for this calculation.

For an odd count, the final singleton remains in the path-weighted price. A finite-sample
independent-cluster variance estimate accounts for pair clusters of size two and the singleton
cluster of size one. The full formula and rationale are in
[docs/phase1-design.md](docs/phase1-design.md).

Antithetic sampling often lowers variance for vanilla options, but it is not guaranteed to improve
every payoff and parameter set.

## Control variates

The control variable is discounted terminal stock:

```text
X = exp(-rT) S_T
E[X] = S_0 exp(-qT)
```

For discounted payoff `Y`, the sample estimates

```text
beta_hat = Cov(Y, X) / Var(X)
Y_cv = Y - beta_hat (X - E[X])
```

The returned price and standard error come from `Y_cv`. Results retain the raw price and raw SE,
estimated beta, sample and known control means, raw and adjusted estimator variances, and the
variance-reduction ratio `raw estimator variance / adjusted estimator variance`. The two estimator
variances equal their corresponding standard errors squared; `payoff_variance` remains the variance
of individual discounted payoff values. A ratio above one indicates improvement for that sample;
no universal improvement is assumed. If control variance is zero, beta falls back to zero and the
adjusted estimate equals the raw estimate.

For `ANTITHETIC_CONTROL_VARIATE`, payoff and control values are both averaged within independent
antithetic clusters before beta is estimated or adjustment and SE are calculated. Odd counts retain
the singleton's one-path weight. See the explicit derivation in
[docs/phase2-variance-reduction.md](docs/phase2-variance-reduction.md).

## Greeks and unit conventions

For value `V(S, sigma, r, q, T)`, the library reports:

- Delta: `dV/dS`
- Gamma: `d²V/dS²`
- Vega: `dV/dsigma`
- Rho: `dV/dr`
- Dividend Rho: `dV/dq`
- Theta: `-dV/dT`, the change in value as calendar time passes

Values use raw model units. Vega is per absolute volatility unit `1.00`; rate sensitivities are per
absolute rate unit `1.00`; Theta is per year. The library does not silently scale Vega to one
percentage point or Rho to basis points.

Black-Scholes-Merton provides closed-form values for all six Greeks. Exact expiry uses an explicit
finite convention: call/put Delta at the strike is `0.5`/`-0.5`, while Gamma, Vega, Theta, Rho, and
dividend Rho are zero. Zero volatility differentiates the active deterministic discounted-payoff
branch; at the deterministic forward kink it uses half-weight branch derivatives and the right-hand
Vega limit. See [docs/phase3-greeks.md](docs/phase3-greeks.md) for every formula and boundary rule.

## Monte Carlo Greeks

Finite differences support Delta, Gamma, Vega, Theta, Rho, and dividend Rho under every existing
variance-reduction mode. Spot uses a relative bump; all other bumps use raw absolute model units.
Central differences are used when both states are valid. Vega near zero volatility and Theta near
expiry use a forward fallback so no invalid state is constructed.

The engine explicitly generates one PCG64 standard-normal driver vector and reuses it for the base,
upward, and downward valuations. These Common Random Numbers (CRN) make each per-driver Greek
observation a difference under the same shock. Standard errors come directly from those difference
observations, including antithetic pair clusters and an odd singleton. They are not formed by
treating bumped prices as independent.

Control-variate finite differences recompute the correct discounted-stock expectation for every
bumped state before combining its adjusted observations. The default bumps are:

```text
spot_relative_bump             = 1e-3
volatility_absolute_bump       = 1e-4
rate_absolute_bump             = 1e-4
dividend_yield_absolute_bump   = 1e-4
maturity_absolute_bump         = 1/3650 years
```

Large bumps have more truncation error; very small bumps can suffer cancellation and simulation
noise. Smaller is not universally better, so research callers should inspect bump stability.

Pathwise estimators are implemented for Delta and Vega. They differentiate terminal GBM and use
strict ITM indicators at the payoff kink. Pathwise sampling supports `NONE` and `ANTITHETIC`, with
SE calculated from independent paths or pair clusters. Pathwise Gamma is intentionally absent
because naively differentiating the payoff indicator does not produce a valid second-derivative
estimator.

## Implied volatility

Implied volatility is the non-negative volatility `sigma` that makes the existing analytical
Black-Scholes-Merton price equal an observed European option price. Phase 4 treats the risk-free
rate and dividend yield as known inputs in a dedicated model; it does not put a placeholder
volatility into `MarketEnvironment`.

Before solving, the library checks the continuous-dividend European bounds. With
`A = S exp(-qT)` and `B = K exp(-rT)`, call prices must lie in `[max(A-B, 0), A]` and put prices in
`[max(B-A, 0), B]`. A target outside its interval is rejected before any root evaluation. The
lower bound is recovered as zero volatility when it matches the configured zero-volatility
endpoint. An exact upper-bound price has no finite implied volatility and returns a typed failure.
At exact expiry, a price equal to intrinsic value is valid, but volatility is not identifiable, so
the result reports `implied_volatility=null` and `NON_IDENTIFIABLE`.

The solver methods are:

- `BRENT`: a SciPy bracketed solve with a price-aware absolute volatility tolerance.
- `BISECTION`: an explicit deterministic bracket-preserving solve that requires both price and
  volatility interval tolerances.
- `NEWTON`: Newton-Raphson using Phase 3 analytical raw-unit Vega, with guards for small Vega,
  non-finite values, stagnation, and steps outside the configured bracket.
- `AUTO`: tries Newton, falls back to Brent on an unsafe Newton step, then uses bisection if Brent
  cannot converge.

Every successful result is repriced and accepted only when its absolute price residual meets
`price_tolerance`. Diagnostics include the requested and used methods, iterations, counted pricing
evaluations, final price, residual, bracket, runtime, final Vega, and optional `1 / Vega` price-to-
volatility sensitivity. Small Vega signals an ill-conditioned inversion: a small price change can
produce a large volatility change. Full formulas and boundary policies are in
[docs/phase4-implied-volatility.md](docs/phase4-implied-volatility.md).

Batch inversion preserves input order and returns success or a typed failure for every observation.
Synthetic smile analysis computes `moneyness = K/S` and forward log-moneyness `log(K/F)`, where
`F = S exp((r-q)T)`. It returns a single maturity slice only; interpolation, surface fitting, and
arbitrage repair are outside Phase 4.

## Real-market option data

Phase 5 introduces a `MarketDataProvider` protocol. Application services consume normalized
`UnderlyingQuote`, `OptionInstrument`, `OptionQuote`, and `OptionChainSnapshot` models and never
depend on provider response keys. The concrete adapter uses the Massive REST API (formerly
Polygon.io). It uses Python's standard HTTP library, so no provider SDK was added.

Configure optional live access through environment variables:

```text
MARKET_DATA_PROVIDER=MASSIVE
MARKET_DATA_API_KEY=
MARKET_DATA_BASE_URL=https://api.massive.com
MARKET_DATA_TIMEOUT_SECONDS=10
```

An API key is required only when a live market-data endpoint is called. It is never logged or
returned. Normal tests use static response fixtures and `FakeMarketDataProvider`; an automatic
socket guard fails the suite if any test attempts an external connection.

### Normalization and time policy

Symbols are stripped and uppercased. Provider contract types become the existing `CALL` or `PUT`.
Provider exercise style is preserved separately as `AMERICAN`, `EUROPEAN`, `BERMUDAN`, or
`UNKNOWN`. Provider option symbols remain opaque identifiers; analytics do not parse or depend on
their encoding.

Every timestamp must be timezone-aware and is converted to UTC. Massive nanosecond timestamps are
normalized inside its adapter. A timestamp source identifies whether the option timestamp came
from a quote, trade, or snapshot. Massive supplies expiration as a date, so the adapter deliberately
maps it to 4:00 p.m. `America/New_York` on that date and then converts it to UTC. Maturity uses
ACT/365 elapsed seconds:

```text
T = max(expiration_utc - valuation_utc, 0) / (365 * 24 * 3600)
```

Expired contracts remain visible and receive an `EXPIRED` issue.

### Quote quality and price selection

An option midpoint exists only when bid and ask are present, non-negative, and `ask >= bid`:

```text
midpoint = (bid + ask) / 2
```

Missing, negative, crossed, stale, future-dated, expired, and arbitrage-invalid observations are
preserved with structured issues. Empty issue lists mean valid quotes. The default analytics price
is a fresh valid midpoint. A last trade is used only when `allow_last_price=true`; stale, future,
crossed, negative, and expired quotes remain ineligible. Midpoint is an analytical convention and
is not guaranteed executable.

Staleness defaults to `MAX_QUOTE_AGE_SECONDS=900`. Valid two-sided quotes report absolute spread
`ask - bid` and relative spread `(ask - bid) / midpoint` when the midpoint is positive. Volume,
open interest, bid size, and ask size remain optional and are never fabricated.

### Chain enrichment

The chain engine sorts by expiration, option type, strike, and provider symbol. Each quote reports
ACT/365 maturity, forward, `K/S` moneyness, `log(K/F)`, Phase 4 price bounds, selected market price,
our own implied volatility, Phase 3 analytical Greeks, spread diagnostics, and every quality issue.
A bad quote does not fail its chain. Provider IV and Greeks are retained only as reference fields.

The summary counts contracts, valid and invalid quotes, midpoints, recovered IVs, arbitrage
violations, stale quotes, calls, puts, and expirations. Smile extraction keeps calls and puts in
separate unsmoothed expiration slices. Reusable filters cover expiration, type, strike, moneyness,
midpoint availability, IV availability, and freshness.

Put-call parity indexes quotes by expiration and strike, avoiding a quadratic scan. For fresh valid
call and put midpoints it reports

```text
residual = (C_mid - P_mid) - (S exp(-qT) - K exp(-rT))
```

and flags residuals outside configurable absolute plus relative tolerance. It never repairs prices.
See [docs/phase5-market-data.md](docs/phase5-market-data.md) for the complete field mapping and
policy reference.

## Volatility smiles and surfaces

Phase 6 normalizes every accepted IV point in four coordinates: strike `K`, spot moneyness `K/S`,
forward moneyness `K/F`, and canonical log-forward-moneyness `k = log(K/F)`, with
`F(T) = S exp((r-q)T)`. The surface continues the current flat continuously compounded rate and
flat continuous-dividend assumptions; it does not introduce yield curves.

Smile and maturity interpolation use total implied variance:

```text
w(k,T) = sigma(k,T)^2 T
```

Across strike, callers choose piecewise `LINEAR` or shape-preserving `PCHIP` interpolation in
`(k,w)`. Across maturity, neighboring smile variances are linearly interpolated at fixed `k`, then
converted back with `sigma = sqrt(w/T)`. Exact observations return their stored IV directly. Wing
and maturity extrapolation are disabled by default; explicit `FLAT_VOL` is available when the
caller accepts that assumption.

Market construction uses fresh, arbitrage-valid IV observations within configured IV and relative-
spread bounds. The default `OTM_ONLY` policy uses puts below the forward and calls above it. At the
forward it chooses the tighter relative spread, then calls on a tie. Duplicate strikes use spread,
freshness, and provider symbol as deterministic tie-breakers. Insufficient expirations are reported
and omitted instead of receiving fabricated points.

Diagnostics reconstruct European option prices to check strike monotonicity and convexity. On
irregular strikes, convexity compares adjacent price slopes. Calendar checks require total variance
to be nondecreasing across maturity at fixed `k`. These checks return structured issues and never
repair a surface. ATM-forward term structure returns IV and total variance at `k=0`; local skew and
curvature are centered finite differences with respect to `k`, not delta-based market quotes.

See [docs/phase6-volatility-surfaces.md](docs/phase6-volatility-surfaces.md) for construction rules,
query algorithms, equations, diagnostic tolerances, testing strategy, and limitations.

## Cox-Ross-Rubinstein binomial trees and American exercise

The CRR engine prices European and American calls and puts on a recombining discrete-time tree. For
`N` steps,

```text
dt = T/N
u = exp(sigma sqrt(dt))
d = 1/u
p = [exp((r-q)dt) - d] / (u-d)
disc = exp(-r dt)
```

The risk-neutral probability must lie in `[0,1]`. Materially invalid values are rejected rather
than clamped; only endpoint overshoot within the configured floating-point tolerance is normalized.
This allows negative rates, positive or negative dividends, and `q > r` whenever the CRR no-
arbitrage condition remains valid.

Terminal stock prices are indexed by their number of up moves. Existing vectorized call/put payoffs
create the terminal vector. Each backward level first computes continuation
`disc[(1-p)V_down + pV_up]`. European nodes retain continuation; American nodes use
`max(intrinsic, continuation)`. Pricing uses the exact maximum, while a configurable tolerance is
used only to classify exercise nodes. The American result reports its premium over a European root
reduced on the same tree.

Only current vectors and the early levels needed for Greeks are retained. Peak working memory is
O(N), arithmetic is O(N²), and boundary extraction stores at most one constant-size record per
time level, using O(N) additional storage.
For puts the boundary is the highest exercising stock in the low-stock region; for dividend-paying
calls it is the lowest exercising stock in the high-stock region. Maturity payoff is never labeled
early exercise. A non-dividend American call under standard positive-rate assumptions has no early
exercise value; sufficiently high continuous dividends can make call exercise optimal. American
puts commonly have a positive early-exercise premium.

Zero maturity returns intrinsic value without constructing parameters. For zero-volatility
American contracts, the engine evaluates the deterministic stock path at every tree date and
maximizes the present intrinsic value across those exercise opportunities. Native Delta comes from
level one after exercise decisions. Gamma uses post-exercise level-two values, and Theta compares
the post-exercise middle level-two node with the root over `2dt`. American Greeks can be less smooth
near an exercise boundary.

European CRR values approach Black-Scholes as the number of steps grows, but strike placement commonly
causes alternating odd/even errors. Convergence analytics retain each canonical result and expose
neighboring-step averages only as diagnostics. They never silently replace a tree price.

See [docs/phase7-binomial-trees.md](docs/phase7-binomial-trees.md) for the European foundation and
[docs/phase8-american-options.md](docs/phase8-american-options.md) for the exercise policy,
premium, boundary definitions, deterministic case, American Greeks, convergence, and API design.

## Longstaff-Schwartz American Monte Carlo

LSM simulates a float64 matrix of shape `(paths, time_steps + 1)` using exact GBM log increments.
It learns continuation value backward over discrete exercise dates. At each date, only current
in-the-money paths enter a least-squares regression. The response is that path's currently selected
future cash flow discounted from its selected exercise date to the regression date. Exercising
replaces both the future cash flow and its stopping time; continuing retains them. The final value
discounts each selected cash flow once to time zero.

The default regression uses degree-two POWER polynomials on `x=S/K`. Standard unweighted LAGUERRE
polynomials are also available; both span the same polynomial space at a given degree.
`numpy.linalg.lstsq` supplies coefficients and rank diagnostics. Too few ITM paths or a rank-deficient
fit trigger an explicit continue-all fallback for that date. Large condition numbers are flagged.
Exercise occurs when intrinsic exceeds nonnegative predicted continuation plus the configured
tolerance. Zero-volatility pricing directly maximizes deterministic present intrinsic values over
the configured dates; zero maturity returns intrinsic without simulation.

OUT_OF_SAMPLE is the default: one deterministic child stream trains the policy and another
independent child evaluates it forward. The policy includes the root decision, so evaluation does
not refit or choose a new stopping rule from held-out results. IN_SAMPLE is also supported. All
dates are exercisable by default; an optional sorted `exercise_steps` subset must include maturity.
These are approximations to continuous American exercise. For no-dividend calls with nonnegative
rates, the engine uses the known continue-to-maturity policy to avoid regression-induced exercise.

The reported SE uses discounted realized valuation cash flows, including zero-payoff paths.
Antithetic SE uses pair averages, with the existing weighted-cluster treatment of an odd singleton.
In out-of-sample mode the normal interval measures sampling error conditional on the learned
policy; it excludes policy-training variation, basis error, and exercise-grid error. In-sample SE
is descriptive because shared fitting introduces dependence and bias. A held-out policy estimate
may fall below intrinsic or the European reference through sampling noise or an imperfect learned
policy; diagnostics report this without silently clipping the held-out mean.

Empirical boundaries use final positive exercise observations: the highest exercised stock for a
put and the lowest for a call, subject to a minimum sample count. Maturity is excluded. Boundaries
and regression details are bounded in API responses. No smooth or exact frontier is claimed.
Optional CRR boundary comparisons report stock differences at supported, exactly shared grid dates.

Defaults are 25,000 training plus 25,000 valuation paths, 50 time steps, antithetic sampling, and
seed 42. Full-path storage costs O(paths × steps); the training matrix is released before valuation
simulation. Research endpoints compare the LSM value with a configurable CRR American tree, vary
path count or time steps with independent child seeds, and compare bases with shared streams.
See [docs/phase9-longstaff-schwartz.md](docs/phase9-longstaff-schwartz.md) for formulas, policies,
uncertainty interpretation, resource accounting, and tests.
See [the Phase 9 validation report](benchmarks/phase9-lsm.md) for measured timings and research
examples.

## Asian and digital options

An Asian contract averages a discrete set of observed GBM prices and pays at maturity. A schedule
can generate `n` regular future times `iT/n` or hold unique explicit year fractions. Explicit times
are sorted deterministically and must lie in `[0,T]`. The contract's
`include_initial_spot_in_average` flag defaults to false; when true, `S_0` is included exactly once.
At zero maturity the average is defined as `S_0`.

The simulator applies exact GBM transitions over each requested time increment and stores only the
requested observations. Memory is therefore O(paths x observations), without a hidden daily grid.
Arithmetic and geometric averages are

```text
A_arith = (1/m) sum_i S_i
A_geo   = exp((1/m) sum_i log(S_i)).
```

Calls pay `max(A-K,0)` and puts pay `max(K-A,0)`, discounted from maturity. Geometric averaging is
calculated in log space. Antithetic rows use complete driver vectors `Z` and `-Z`; pair-level
uncertainty and the weighted odd singleton policy remain unchanged.

For the discrete geometric average, joint-normal GBM logs give

```text
mu_G = log(S_0) + (r-q-0.5 sigma^2) mean(t_i)
v_G  = (sigma^2/m^2) sum_i sum_j min(t_i,t_j).
```

The resulting exact lognormal payoff expectation supplies a benchmark for geometric Asians and a
control for arithmetic Asians. The arithmetic control uses the discounted geometric payoff and its
known discrete analytical expectation. In the combined antithetic mode, beta and both raw and
adjusted SEs are estimated after forming independent pair clusters. Result diagnostics expose beta,
raw and adjusted estimates and variances, control means, and the sampled variance-reduction ratio.

Digital options use strict ITM indicators: equality at strike pays zero. Cash-or-nothing analytical
prices are `cash exp(-rT) N(d2)` for calls and `cash exp(-rT) N(-d2)` for puts. Asset-or-nothing
prices are `S exp(-qT) N(d1)` and `S exp(-qT) N(-d1)`. Monte Carlo reuses terminal GBM with plain or
antithetic sampling. A digital's payoff discontinuity can produce high variance near strike and
noisy realized convergence, although its asymptotic Monte Carlo SE remains O(N^-1/2). Digital
pathwise Greeks and smoothing are outside Phase 10.

Research services compare all four arithmetic Asian variance-reduction modes, validate geometric
and digital Monte Carlo against their analytical prices, vary path count, and vary regular Asian
monitoring frequency. Convergence points receive independent deterministic `SeedSequence` child
streams. A discrete monitoring-frequency study does not claim a continuous-monitoring price.

See [docs/phase10-asian-digital-options.md](docs/phase10-asian-digital-options.md) for the complete
derivations, edge-case and cluster-variance policies, memory model, and API safeguards. Measured
research examples and opt-in timings are in
[benchmarks/phase10-exotics.md](benchmarks/phase10-exotics.md).

## Barrier options and monitoring correction

A barrier contract has one `UP` or `DOWN` level and `IN` or `OUT` semantics. Down barriers hit when
`S_t <= H`; up barriers hit when `S_t >= H`. Equality is a hit, and `S_0` is always checked even if
the schedule begins later. A knock-in activates the maturity vanilla payoff after a hit. A
knock-out retains that payoff only when no hit occurs.

Discrete mode examines time zero plus regular or explicit scheduled observations. It records the
first observed hit index and time, enabling a knock-out rebate paid either at that hit or at expiry.
A knock-in rebate is a no-hit payment at expiry. The terminal stock is simulated for payoff even
when an explicit barrier schedule omits maturity; it is treated as a discrete barrier observation
only when scheduled.

Discrete sampling can miss a crossing followed by a return between observations. For GBM, bridge
mode corrects this with the exact conditional log-price crossing probability. For a down barrier
with both interval endpoints above `h=log(H)`,

```text
p_cross = exp[-2 (X_i-h)(X_{i+1}-h)/(sigma^2 delta_i)].
```

The up formula reflects the distances below `h`. Endpoint breaches have probability one. Actual
irregular time increments are used. Interval survivals multiply to `w`; an OUT payoff is weighted
by `w`, and an IN payoff by `1-w`. This conditional expectation avoids extra Bernoulli draws and
makes zero-rebate IN plus OUT equal vanilla path by path. Bridge hit probability is
`1-mean(w)`. Bridge hit-time statistics are unavailable because no first-hit time is sampled.

Expiry rebates are conditionally weighted in bridge mode. Positive `AT_HIT` bridge rebates are
rejected: survival probability alone cannot supply the required expected hit-time discount. At
`sigma=0`, the path is monotone and any continuous crossing time is solved exactly. At `T=0`, only
the inclusive initial status and immediate payoff/rebate semantics apply.

All eight zero-rebate continuous single-barrier Black-Scholes-Merton combinations are implemented.
Reflection terms use log-normal CDF evaluation for far-barrier stability. Analytical and same-path
Monte Carlo checks enforce `knock-in + knock-out = vanilla`. Research endpoints compare discrete,
bridge, and analytical values; vary paths or monitoring count with independent child seeds; and
vary barrier level with Common Random Numbers.

Only requested states plus maturity are retained, so state memory is O(paths x observations).
Bridge survival uses O(paths) extra workspace. See
[docs/phase11-barrier-options.md](docs/phase11-barrier-options.md) for formulas, rebate restrictions,
parity, deterministic policies, and API safeguards. Measurements are in
[benchmarks/phase11-barriers.md](benchmarks/phase11-barriers.md).

## Correlated multi-asset GBM and basket options

`MultiAssetMarketEnvironment` keeps per-asset spots, dividend yields, volatilities, and optional
identifiers separate from the basket contract. Its correlation matrix must be finite, symmetric,
unit diagonal, bounded by one, and positive semidefinite. Eigenvalues below `-1e-10` are rejected.
Tiny negative eigenvalues through floating-point noise are clipped, the matrix is reconstructed and
renormalized, and the repair and Frobenius adjustment are exposed in diagnostics.

Positive-definite matrices use Cholesky; singular PSD matrices, including perfect correlation, use
a symmetric eigen factor. Independent PCG64 normal row vectors are transformed with `Z=epsilon
L^T`. Terminal simulation returns `(paths, assets)`. Full arbitrary-time paths return `(paths,
time_points, assets)`, including the initial state, and apply exact GBM transitions over each actual
time increment. Antithetic vectors are formed as `epsilon,-epsilon` before correlation.

Basket weights are nonnegative, sum to one within `1e-10`, and are normalized exactly after
validation. Arithmetic values are `sum(w_i S_i)`; geometric values use `exp(sum(w_i log S_i))`.
For a geometric basket,

```text
mu_G = sum_i w_i [log S_i(0) + (r-q_i-0.5 sigma_i^2)T]
v_G  = T sum_i sum_j w_i w_j sigma_i sigma_j rho_ij.
```

The analytical engine prices calls and puts directly from this lognormal distribution. It does not
treat the basket as a fictitious traded asset. Arithmetic Monte Carlo supports plain, antithetic,
geometric-control, and combined estimators. The control is the discounted geometric payoff whose
known expectation is the exact geometric price. Combined sampling estimates beta and SE after both
payoffs have been reduced to identical independent antithetic clusters.

Sensitivity services use Common Random Numbers for equicorrelation scenarios and two-asset weight
sweeps. Path-count convergence uses independent deterministic `SeedSequence` children and fits the
positive-SE points in log space. Terminal state memory is `O(paths * assets)`; dense factor
application costs `O(paths * assets^2)` and factorization costs `O(assets^3)`. See
[docs/phase12-multi-asset-baskets.md](docs/phase12-multi-asset-baskets.md) for formulas, tolerances,
shape conventions, and Phase 13 boundaries. Opt-in measurements are in
[benchmarks/phase12-multi-asset-baskets.md](benchmarks/phase12-multi-asset-baskets.md).

## Heston stochastic volatility

Heston replaces constant instantaneous variance with a correlated square-root process under the
risk-neutral measure:

```text
dS_t = (r-q)S_t dt + sqrt(v_t)S_t dW_t^S
dv_t = kappa(theta-v_t)dt + sigma_v sqrt(v_t)dW_t^v
corr(dW_t^S,dW_t^v) = rho.
```

`HestonParameters` stores `v0`, `kappa`, `theta`, `sigma_v`, and `rho`; supplied values are already
risk-neutral parameters. The Feller condition `2*kappa*theta >= sigma_v^2` and its ratio are
reported as diagnostics. Violating parameters remain valid because calibrated equity models often
use them.

The baseline Full-Truncation Euler scheme retains its raw variance update and uses `max(v,0)` in
every coefficient. It exposes negative-proposal frequency without ever taking an invalid square
root. The higher-quality Andersen QE implementation matches exact CIR conditional moments with a
quadratic-Gaussian low-variance branch and a point-mass/exponential high-variance branch. Its QE-M
log-spot update includes a conditional-MGF martingale correction. When vol-of-vol is zero, both
schemes use the exact deterministic variance transition and integrated variance.

Terminal-only pricing keeps O(paths) state. Full `(paths,steps+1)` spot and variance matrices are
allocated only by the diagnostic simulator. Antithetic FTE paths use `(Z1,Z2)` and
`(-Z1,-Z2)`; QE additionally uses `U` and `1-U`. SE is computed from independent pair clusters,
including the established weighted odd singleton rule. Simulation intervals quantify sampling
error and do not include finite-step bias.

The semi-analytical engine uses the branch-consistent Little Heston Trap characteristic function.
Segmented SciPy quadrature evaluates Heston's `P1/P2` integrals, checks the tail, and reports its
error estimate, evaluation count, used integration bound, and runtime. Puts follow from put-call
parity. Exact expiry returns intrinsic value, while deterministic and near-deterministic variance
use exact integrated variance and Black-Scholes. Negative rates, continuous dividends, `kappa=0`,
`theta=0`, `v0=0`, Feller violations, and correlations through exactly `+/-1` are covered.

Research services compare FTE and QE-M with one analytical reference, vary paths with independent
`SeedSequence` children, vary time steps with a common seed, and compare simulated variance means
with `theta+(v0-theta)exp(-kappa*t)`. Analytical calls across strikes feed the existing implied-vol
solver to produce a Heston smile. Negative `rho` commonly creates equity-style negative skew, and
larger `sigma_v` commonly changes smile curvature; neither behavior is hard-coded. Sweeps of all
five parameters are scenario analysis only. Calibration is deferred to Phase 14.

See [docs/phase13-heston.md](docs/phase13-heston.md) for the formulas, branch policies, memory model,
and API accounting. Measured timings and validation values are in
[benchmarks/phase13-heston.md](benchmarks/phase13-heston.md).

## Reproducibility and convergence

Every stochastic pricing call constructs an isolated `numpy.random.Generator` backed by `PCG64`.
The code never uses NumPy's global random state. Identical model inputs, simulation configuration,
seed, and supported numerical environment reproduce all quantitative outputs exactly; measured
wall-clock runtime naturally varies.

A convergence experiment creates `SeedSequence(root_seed)` and spawns one deterministic child per
simulation count. Points therefore use independent, reproducible streams rather than prefixes of
one shared stream. Valid positive standard errors are fitted by ordinary least squares:

```text
log(SE) = a + b log(N)
```

For all four methods, the expected asymptotic slope remains approximately `b = -0.5`. Variance
reduction lowers the error's constant factor; it does not change the IID Monte Carlo order. If fewer
than two positive-SE points exist, the slope is returned as unavailable rather than raising an
error.

Method comparisons use the same root seed for all modes. `NONE` and `CONTROL_VARIATE` therefore
share path normals, while `ANTITHETIC` and `ANTITHETIC_CONTROL_VARIATE` share primary normals and
pairing. Relative variance is method estimator variance divided by the `NONE` variance; relative
efficiency is its reciprocal. Runtime is reported separately and is not mixed into either metric.

## Repository structure

```text
backend/app/
├── analytics/     # convergence, comparisons, exotic experiments, grids, batches, and smiles
├── api/           # FastAPI request models, limits, and thin adapters
├── core/          # environment settings, logging, errors, numerical guards
├── market_data/   # provider protocol, Massive adapter, and deterministic fake
├── models/        # vanilla/exotic contracts, schedules, configs, enums, and typed results
├── payoffs/       # vectorized vanilla, path-dependent, digital, barrier, and basket payoffs
├── pricing/       # analytical, Monte Carlo, tree, LSM, exotic, basket, and Heston engines
├── simulation/    # GBM/Heston processes, correlation factors, CRN, and RNG ownership
└── volatility_surface/ # total-variance construction, queries, and synthetic fixtures
backend/tests/     # deterministic, numerical, statistical, and HTTP tests
benchmarks/        # quantitative benchmark and validation reports
docs/              # architecture and numerical methodology
experiments/       # reserved for explicit research experiments
```

## Installation

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The runtime dependencies are FastAPI, Pydantic, NumPy, SciPy, and Uvicorn. The requirements file
also installs pytest, Ruff, mypy, and HTTPX for development and API testing.

## Library example

```python
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.models.simulation import SimulationConfig, VarianceReduction
from backend.app.pricing.comparison import compare_prices

contract = OptionContract(
    option_type=OptionType.CALL,
    spot=100.0,
    strike=100.0,
    maturity_years=1.0,
)
market = MarketEnvironment(
    risk_free_rate=0.05,
    volatility=0.20,
    dividend_yield=0.01,
)
simulation = SimulationConfig(
    num_simulations=100_000,
    seed=42,
    variance_reduction=VarianceReduction.ANTITHETIC_CONTROL_VARIATE,
    confidence_level=0.95,
)

comparison = compare_prices(contract, market, simulation)
print(comparison.model_dump())
```

Arithmetic Asian with the discrete geometric control:

```python
from backend.app.models.asian import (
    AsianOptionContract,
    AsianSimulationConfig,
    AsianVarianceReduction,
    AveragingType,
)
from backend.app.models.monitoring import MonitoringSchedule
from backend.app.pricing.asian import asian_monte_carlo_price

asian = AsianOptionContract(
    option_type=OptionType.CALL,
    spot=100.0,
    strike=100.0,
    maturity_years=1.0,
    averaging_type=AveragingType.ARITHMETIC,
    averaging_schedule=MonitoringSchedule.regular(12),
)
asian_result = asian_monte_carlo_price(
    asian,
    market,
    AsianSimulationConfig(
        num_simulations=100_000,
        seed=42,
        variance_reduction=AsianVarianceReduction.ANTITHETIC_GEOMETRIC_CONTROL,
    ),
)
```

Analytical cash-or-nothing digital:

```python
from backend.app.models.digital import DigitalOptionContract, DigitalType
from backend.app.pricing.digital import digital_analytical_price

digital = DigitalOptionContract(
    option_type=OptionType.CALL,
    spot=100.0,
    strike=100.0,
    maturity_years=1.0,
    digital_type=DigitalType.CASH_OR_NOTHING,
    cash_payout=10.0,
)
digital_result = digital_analytical_price(digital, market)
```

Bridge-weighted down-and-out call:

```python
from backend.app.models.barrier import (
    BarrierDirection,
    BarrierMonitoringMode,
    BarrierOptionContract,
    BarrierSimulationConfig,
    BarrierStyle,
)
from backend.app.pricing.barrier import barrier_monte_carlo_price

barrier = BarrierOptionContract(
    option_type=OptionType.CALL,
    barrier_direction=BarrierDirection.DOWN,
    barrier_style=BarrierStyle.OUT,
    spot=100.0,
    strike=100.0,
    barrier=90.0,
    maturity_years=1.0,
    monitoring_schedule=MonitoringSchedule.regular(52),
    monitoring_mode=BarrierMonitoringMode.BROWNIAN_BRIDGE,
)
barrier_result = barrier_monte_carlo_price(
    barrier,
    market,
    BarrierSimulationConfig(num_simulations=100_000, seed=42),
)
```

Heston analytical and QE-M Monte Carlo pricing:

```python
from backend.app.models.heston import (
    HestonParameters,
    HestonSimulationConfig,
    HestonSimulationScheme,
)
from backend.app.pricing.heston import heston_monte_carlo_price
from backend.app.pricing.heston_analytical import heston_analytical_price

heston = HestonParameters(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7)
analytical_heston = heston_analytical_price(contract, market, heston)
monte_carlo_heston = heston_monte_carlo_price(
    contract,
    market,
    heston,
    HestonSimulationConfig(
        num_simulations=100_000,
        num_time_steps=100,
        seed=42,
        scheme=HestonSimulationScheme.QUADRATIC_EXPONENTIAL,
    ),
)
```

Analytical and pathwise Greeks:

```python
from backend.app.models.greeks import Greek
from backend.app.pricing.black_scholes_greeks import black_scholes_greeks
from backend.app.pricing.monte_carlo_greeks import pathwise_greeks

analytical = black_scholes_greeks(contract, market)
pathwise = pathwise_greeks(
    contract,
    market,
    simulation.model_copy(update={"variance_reduction": VarianceReduction.ANTITHETIC}),
    [Greek.DELTA, Greek.VEGA],
)
```

Implied-volatility round trip:

```python
from backend.app.models.volatility import (
    ImpliedVolatilityMarket,
    ImpliedVolatilityObservation,
)
from backend.app.pricing.implied_volatility import solve_implied_volatility

observation = ImpliedVolatilityObservation(
    contract=contract,
    market=ImpliedVolatilityMarket(risk_free_rate=0.05, dividend_yield=0.01),
    observed_price=9.826297782739111,
)
iv_result = solve_implied_volatility(observation)
print(iv_result.implied_volatility)  # approximately 0.20
```

`batch_size` is validated as an optional execution hint. The current engine evaluates a single
vector; changing `batch_size` does not change the random stream or quantitative output.

## API

Start the service from the repository root:

```bash
.venv/bin/uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Monte Carlo pricing:

```bash
curl -X POST http://127.0.0.1:8000/api/pricing/monte-carlo \
  -H 'Content-Type: application/json' \
  -d '{
    "contract": {
      "option_type": "CALL",
      "exercise_style": "EUROPEAN",
      "spot": 100.0,
      "strike": 100.0,
      "maturity_years": 1.0
    },
    "market": {
      "risk_free_rate": 0.05,
      "volatility": 0.2,
      "dividend_yield": 0.01
    },
    "simulation": {
      "num_simulations": 100000,
      "seed": 42,
      "variance_reduction": "ANTITHETIC_CONTROL_VARIATE",
      "random_engine": "PCG64",
      "confidence_level": 0.95
    }
  }'
```

Available endpoints:

- `GET /health`
- `POST /api/pricing/monte-carlo`
- `POST /api/pricing/black-scholes`
- `POST /api/pricing/compare`
- `POST /api/pricing/binomial`
- `POST /api/pricing/longstaff-schwartz`
- `POST /api/pricing/asian`
- `POST /api/pricing/asian/geometric-analytical`
- `POST /api/pricing/digital`
- `POST /api/pricing/barrier`
- `POST /api/pricing/barrier/analytical`
- `POST /api/pricing/basket`
- `POST /api/pricing/basket/geometric-analytical`
- `POST /api/pricing/heston/monte-carlo`
- `POST /api/pricing/heston/analytical`
- `POST /api/analytics/longstaff-schwartz/compare`
- `POST /api/analytics/longstaff-schwartz/convergence`
- `POST /api/analytics/longstaff-schwartz/basis-comparison`
- `POST /api/analytics/convergence`
- `POST /api/analytics/variance-reduction`
- `POST /api/greeks/black-scholes`
- `POST /api/greeks/monte-carlo`
- `POST /api/analytics/greeks/compare`
- `POST /api/analytics/sensitivity`
- `POST /api/volatility/implied`
- `POST /api/volatility/implied/batch`
- `POST /api/analytics/volatility/smile`
- `POST /api/analytics/volatility/solver-comparison`
- `GET /api/market-data/underlying/{symbol}`
- `GET /api/market-data/options/{symbol}`
- `GET /api/market-data/options/{symbol}/expirations`
- `POST /api/analytics/options/chain`
- `POST /api/analytics/options/smile`
- `POST /api/analytics/options/put-call-parity`
- `POST /api/analytics/volatility/smile/build`
- `POST /api/analytics/volatility/smile/query`
- `POST /api/analytics/volatility/surface/build`
- `POST /api/analytics/volatility/surface/query`
- `POST /api/analytics/volatility/surface/diagnostics`
- `POST /api/analytics/volatility/surface/atm-term-structure`
- `POST /api/analytics/binomial/compare`
- `POST /api/analytics/binomial/exercise-boundary`
- `POST /api/analytics/binomial/convergence`
- `POST /api/analytics/asian/compare`
- `POST /api/analytics/asian/convergence`
- `POST /api/analytics/digital/compare`
- `POST /api/analytics/digital/convergence`
- `POST /api/analytics/barrier/compare`
- `POST /api/analytics/barrier/convergence`
- `POST /api/analytics/barrier/parity`
- `POST /api/analytics/barrier/sensitivity`
- `POST /api/analytics/basket/compare`
- `POST /api/analytics/basket/convergence`
- `POST /api/analytics/basket/correlation-sensitivity`
- `POST /api/analytics/basket/weight-sensitivity`
- `POST /api/analytics/multi-asset/correlation`
- `POST /api/analytics/heston/compare`
- `POST /api/analytics/heston/convergence`
- `POST /api/analytics/heston/smile`
- `POST /api/analytics/heston/sensitivity`
- `POST /api/analytics/heston/variance-diagnostics`

Price an arithmetic two-asset basket with the analytical geometric payoff as a control:

```bash
curl -X POST http://127.0.0.1:8000/api/pricing/basket \
  -H 'Content-Type: application/json' \
  -d '{
    "contract": {
      "option_type": "CALL", "strike": 100, "maturity_years": 1,
      "weights": [0.5, 0.5], "basket_type": "ARITHMETIC"
    },
    "market": {
      "spots": [100, 95], "risk_free_rate": 0.03,
      "dividend_yields": [0.01, 0.02], "volatilities": [0.2, 0.25],
      "correlation_matrix": [[1, 0.4], [0.4, 1]], "asset_ids": ["A", "B"]
    },
    "simulation": {
      "num_simulations": 100000, "seed": 42,
      "variance_reduction": "ANTITHETIC_GEOMETRIC_CONTROL"
    }
  }'
```

Price the same European contract under Heston with QE-M:

```bash
curl -X POST http://127.0.0.1:8000/api/pricing/heston/monte-carlo \
  -H 'Content-Type: application/json' \
  -d '{
    "contract": {
      "option_type": "CALL", "exercise_style": "EUROPEAN",
      "spot": 100, "strike": 100, "maturity_years": 1
    },
    "market": {"risk_free_rate": 0.02, "dividend_yield": 0.01, "volatility": 0},
    "parameters": {"v0": 0.04, "kappa": 2, "theta": 0.04, "sigma_v": 0.3, "rho": -0.7},
    "simulation": {
      "num_simulations": 100000, "num_time_steps": 100, "seed": 42,
      "scheme": "QUADRATIC_EXPONENTIAL", "antithetic": true, "confidence_level": 0.95
    }
  }'
```

The Heston analytical endpoint takes the same contract, market, and parameters without a
simulation object. Convergence accepts `PATH_COUNT` or `TIME_STEP`; smile accepts a positive unique
`strikes` grid; sensitivity accepts one parameter name and a unique `values` grid. The market
`volatility` field is retained for compatibility with the shared Phase 1 market model and is not
used by either Heston engine.

Price an American put and return its same-tree European comparator and exercise summary:

```bash
curl -X POST http://127.0.0.1:8000/api/pricing/binomial \
  -H 'Content-Type: application/json' \
  -d '{
    "contract": {
      "option_type": "PUT", "exercise_style": "AMERICAN",
      "spot": 100.0, "strike": 100.0, "maturity_years": 1.0
    },
    "market": {
      "risk_free_rate": 0.05, "volatility": 0.2, "dividend_yield": 0.0
    },
    "tree": {"steps": 500, "model": "CRR", "exercise_tolerance": 1e-12}
  }'
```

Send the same body to `/api/analytics/binomial/exercise-boundary` to receive the ordered compact
frontier. A European contract is rejected by that boundary-only endpoint.

Price the same American put by independent LSM policy evaluation:

```bash
curl -X POST http://127.0.0.1:8000/api/pricing/longstaff-schwartz \
  -H 'Content-Type: application/json' \
  -d '{
    "contract": {
      "option_type": "PUT", "exercise_style": "AMERICAN",
      "spot": 100.0, "strike": 100.0, "maturity_years": 1.0
    },
    "market": {"risk_free_rate": 0.05, "volatility": 0.2, "dividend_yield": 0.0},
    "lsm": {
      "num_paths": 25000, "regression_paths": 25000, "num_time_steps": 50,
      "seed": 42, "evaluation_mode": "OUT_OF_SAMPLE",
      "variance_reduction": "ANTITHETIC", "basis_family": "POWER", "basis_degree": 2,
      "store_regression_diagnostics": true
    }
  }'
```

Use the LSM `/compare` endpoint with optional `"tree":{"steps":1000}` and
`"include_boundary_comparison":true` for supported common-date boundary differences.
LSM `/convergence` also
takes `"mode":"PATH_COUNT"` and `"values":[1000,5000,10000]`, or `"mode":"TIME_STEPS"` with
step counts. LSM `/basis-comparison` accepts `"families":["POWER","LAGUERRE"]` and
`"degrees":[1,2,3,4]`. Regression details are optional in the pricing response.

Price a regular monthly arithmetic Asian with antithetic geometric control:

```bash
curl -X POST http://127.0.0.1:8000/api/pricing/asian \
  -H 'Content-Type: application/json' \
  -d '{
    "contract": {
      "option_type": "CALL", "spot": 100.0, "strike": 100.0,
      "maturity_years": 1.0, "averaging_type": "ARITHMETIC",
      "averaging_schedule": {"monitoring_count": 12},
      "include_initial_spot_in_average": false
    },
    "market": {"risk_free_rate": 0.05, "volatility": 0.2, "dividend_yield": 0.01},
    "simulation": {
      "num_simulations": 100000, "seed": 42,
      "variance_reduction": "ANTITHETIC_GEOMETRIC_CONTROL",
      "confidence_level": 0.95
    }
  }'
```

Use `averaging_type=GEOMETRIC` with `/api/pricing/asian/geometric-analytical` for the exact
discrete benchmark. The Asian convergence endpoint accepts `mode=PATH_COUNT` or
`mode=MONITORING_COUNT` and a positive unique `values` list.

Price a cash digital analytically by posting this body to `/api/pricing/digital`. Set
`method=MONTE_CARLO` and include a digital simulation object for stochastic pricing:

```json
{
  "contract": {
    "option_type": "CALL", "spot": 100.0, "strike": 100.0,
    "maturity_years": 1.0, "digital_type": "CASH_OR_NOTHING", "cash_payout": 10.0
  },
  "market": {"risk_free_rate": 0.05, "volatility": 0.2, "dividend_yield": 0.01},
  "method": "ANALYTICAL"
}
```

Price a weekly monitored down-and-out call with continuous bridge correction:

```json
{
  "contract": {
    "option_type": "CALL", "barrier_direction": "DOWN", "barrier_style": "OUT",
    "spot": 100.0, "strike": 100.0, "barrier": 90.0, "maturity_years": 1.0,
    "monitoring_schedule": {"monitoring_count": 52},
    "monitoring_mode": "BROWNIAN_BRIDGE",
    "rebate": 0.0, "rebate_timing": "AT_EXPIRY"
  },
  "market": {"risk_free_rate": 0.05, "volatility": 0.2, "dividend_yield": 0.01},
  "simulation": {
    "num_simulations": 100000, "seed": 42,
    "variance_reduction": "ANTITHETIC", "confidence_level": 0.95
  }
}
```

Post this body to `/api/pricing/barrier`. The analytical endpoint supports zero-rebate continuous
monitoring. Barrier comparison returns discrete and bridge estimates; convergence accepts
`PATH_COUNT` or `MONITORING_COUNT`; parity requires zero rebate; sensitivity accepts a positive
unique `barrier_levels` list.

The original European `/api/analytics/convergence` body replaces `simulation` with
`simulation_counts`, `root_seed`,
`variance_reduction`, and `confidence_level`. Generated OpenAPI documentation is available at
`/docs` while the service is running.

Compare every variance-reduction method with one fixed budget:

```bash
curl -X POST http://127.0.0.1:8000/api/analytics/variance-reduction \
  -H 'Content-Type: application/json' \
  -d '{
    "contract": {
      "option_type": "CALL", "exercise_style": "EUROPEAN",
      "spot": 100.0, "strike": 100.0, "maturity_years": 1.0
    },
    "market": {
      "risk_free_rate": 0.05, "volatility": 0.2, "dividend_yield": 0.01
    },
    "num_simulations": 100000, "root_seed": 42, "confidence_level": 0.95
  }'
```

Request CRN finite-difference Greeks by sending the same contract, market, and simulation objects
to `/api/greeks/monte-carlo`, together with:

```json
{
  "greeks": ["DELTA", "GAMMA", "VEGA", "THETA", "RHO", "DIVIDEND_RHO"],
  "estimator_method": "FINITE_DIFFERENCE"
}
```

The optional `bump_config` object overrides individual bump defaults. `PATHWISE` accepts Delta and
Vega with `NONE` or `ANTITHETIC` variance reduction. Unsupported combinations return HTTP 422.

Single implied-volatility solve:

```bash
curl -X POST http://127.0.0.1:8000/api/volatility/implied \
  -H 'Content-Type: application/json' \
  -d '{
    "contract": {
      "option_type": "CALL", "exercise_style": "EUROPEAN",
      "spot": 100.0, "strike": 100.0, "maturity_years": 1.0
    },
    "risk_free_rate": 0.05,
    "dividend_yield": 0.01,
    "observed_price": 9.826297782739111,
    "solver_config": {"method": "AUTO"}
  }'
```

API allocation limits come from environment variables and do not constrain direct library use:

```text
MAX_API_SIMULATIONS=5000000
MAX_CONVERGENCE_POINTS=20
MAX_SENSITIVITY_POINTS=100
MAX_IV_BATCH_SIZE=500
MAX_SMILE_POINTS=500
MAX_SURFACE_SMILES=100
MAX_SURFACE_QUERY_POINTS=1000
MAX_DIAGNOSTIC_GRID_POINTS=10000
MAX_BINOMIAL_STEPS=10000
MAX_BINOMIAL_CONVERGENCE_POINTS=20
MAX_LSM_PATHS=500000
MAX_LSM_TIME_STEPS=500
MAX_LSM_TOTAL_PATH_NODES=25000000
MAX_LSM_CONVERGENCE_POINTS=20
MAX_LSM_DIAGNOSTIC_DATES=100
MAX_ASIAN_MONITORING_POINTS=1000
MAX_ASIAN_TOTAL_PATH_NODES=25000000
MAX_EXOTIC_CONVERGENCE_POINTS=20
MAX_BARRIER_MONITORING_POINTS=5000
MAX_BARRIER_TOTAL_PATH_NODES=25000000
MAX_BARRIER_CONVERGENCE_POINTS=20
MAX_MULTI_ASSET_COUNT=100
MAX_MULTI_ASSET_PATH_NODES=25000000
MAX_BASKET_CONVERGENCE_POINTS=20
MAX_BASKET_SENSITIVITY_POINTS=50
MAX_HESTON_PATHS=500000
MAX_HESTON_TIME_STEPS=2000
MAX_HESTON_TOTAL_PATH_NODES=25000000
MAX_HESTON_CONVERGENCE_POINTS=20
MAX_HESTON_SMILE_POINTS=200
MAX_HESTON_SENSITIVITY_POINTS=100
MAX_OPTION_CHAIN_CONTRACTS=5000
MAX_QUOTE_AGE_SECONDS=900
MARKET_DATA_PROVIDER=MASSIVE
MARKET_DATA_API_KEY=
MARKET_DATA_BASE_URL=https://api.massive.com
MARKET_DATA_TIMEOUT_SECONDS=10
LOG_LEVEL=INFO
```

After supplying a Massive API key, optional manual connectivity checks are:

```bash
curl http://127.0.0.1:8000/api/market-data/underlying/SPY
curl 'http://127.0.0.1:8000/api/market-data/options/SPY?expiration=2027-01-15'
```

These calls are manual and never run in CI. Provider entitlements determine quote availability and
whether data is real-time or delayed.

Invalid input and unsafe allocation requests return HTTP 422 with a clear detail. Raw terminal
prices and payoff vectors never cross the API boundary. A stochastic run also needs enough
independent replication to estimate uncertainty: at least two paths for plain Monte Carlo and at
least two antithetic clusters. Smaller stochastic requests fail clearly instead of reporting false
zero uncertainty; deterministic cases remain exact.

LSM's node budget sums training and valuation paths times `(steps+1)` across every point in a
request. Each sample set must also fit the path ceiling. Optional diagnostic lists are capped by
`MAX_LSM_DIAGNOSTIC_DATES`; output truncation is explicit and does not change prices.

Asian limits count effective observations, including `S_0` when selected. Total path nodes include
every method or convergence point represented by the request. Oversized regular counts are rejected
before the service expands the schedule or allocates a path matrix. Digital Monte Carlo reuses the
general path-count ceiling.

Barrier limits count mandatory time zero, scheduled observations, a payoff-only maturity state when
needed, and every pricing/parity run represented by an analytics request. The 25-million-node
default admits 100k×52 and is close to 100k×252, while rejecting 100k×1000 before allocation.

Multi-asset limits count every path-asset terminal node across all comparison, convergence, or
sensitivity runs. The asset limit also applies to analytical and correlation-diagnostic requests.
Basket terminal pricing does not allocate time steps; full-path multi-asset simulation remains a
library research primitive.

Heston limits count every requested path-step node, including both FTE and QE-M comparison runs and
all convergence points. Smile and sensitivity limits cap analytical quadrature calls. Normal
European pricing uses terminal-only state; variance diagnostics allocate full spot and variance
path matrices under the same node budget.

## Tests and quality checks

```bash
.venv/bin/pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy backend/app
```

The normal suite uses fixed seeds, moderate sample sizes, analytical identities, deterministic edge
cases, theoretical GBM moments, direct pair/control/CRN calculations, analytical Greek identities,
and statistically scaled Monte Carlo tolerances such as `abs(estimate - benchmark) <= k * SE`.
Binomial tests directly reproduce European and American one- and two-step lattices, check same-tree
dominance, boundary decisions, deterministic exercise timing, and canonical American put values,
then use Black-Scholes and analytical Greeks where a European comparison is valid. Convergence tests
do not assume every successive step count improves.
LSM tests include hand-constructed path matrices, exact stopping-time discounting and replacement,
frozen-policy evaluation, training/valuation independence, pair and odd-cluster uncertainty,
regression fallbacks, deterministic cases, and representative American comparisons with CRR.
Phase 10 tests use hand-built Asian observation matrices, exact irregular schedules and GBM
increments, discrete geometric moments, one-date vanilla reduction, analytical digital identities,
direct pair/control calculations, deterministic edges, and statistically scaled analytical
comparisons. Performance-marked cases cover 10k/100k Asian grids through 252 dates and 100k/1m
digital paths.
Phase 11 tests cover inclusive initial and scheduled hits, exact first-hit rebates, fixed-path
IN/OUT parity, manual bridge probabilities, irregular intervals, continuous analytical references,
deterministic crossing times, antithetic clusters, monitoring bias, resource limits, and every new
HTTP endpoint. Barrier performance tests cover 10k×52, 100k×52, and 100k×252; 100k×1000 is gated
behind an explicit environment opt-in because the state matrix alone is about 800.8 MB.
Phase 12 tests cover matrix shape/symmetry/diagonal/PSD rules, tiny repair, Cholesky and singular
eigen factors, empirical correlations, terminal moments, exact arbitrary-time paths, multivariate
antithetic pairing, fixed basket payoffs, single-asset and perfect-correlation reductions,
geometric analytical/Monte Carlo agreement, all four arithmetic estimators, pair-level controls,
sensitivities, convergence, serialization, and work limits. Performance cases cover terminal
100k×2, 100k×10, 100k×50, 1m×2, factorization through 100 assets, and a 10k×50×5 path cube.
Phase 13 tests cover parameter boundaries and Feller diagnostics, empirical Brownian correlation,
fixed FTE/QE updates, both QE moment branches, raw negative-variance handling, terminal/full-path
reproducibility, CIR and terminal-spot means, a hard-coded Heston benchmark, Little Heston Trap
stability, parity, the Black-Scholes limit, MC-versus-analytical agreement, antithetic cluster SE,
smiles, every parameter sweep, convergence seeds, APIs, and allocation limits. Performance cases
cover FTE and QE-M at 100k paths with 50, 100, and 250 steps; 500k×100 is separately opt-in.
Resource-intensive checks belong under the `performance` marker and are excluded from ordinary
pytest by project configuration. Market-data tests are fully offline: provider transport is mocked,
static Massive-shaped JSON fixtures test normalization, fake normalized chains test analytics and
APIs, and an autouse socket guard rejects accidental network access.

## Model assumptions and limitations

- European exercise for Black-Scholes, implied volatility, terminal Monte Carlo, and their Greeks
- European or American exercise for the CRR binomial engine
- Frictionless market and continuous trading as assumed by Black-Scholes
- Risk-neutral GBM with lognormal stock prices
- Constant volatility
- Constant continuously compounded risk-free rate, which may be negative
- Constant continuous dividend yield, which may be negative
- No jumps; stochastic volatility is available only through the Phase 13 Heston model
- No transaction costs, liquidity effects, discrete hedging, or model calibration
- Terminal and full-path GBM simulation; LSM regression uses current stock only
- Normal-approximation confidence intervals
- Greeks use raw model units and the calendar-time Theta convention
- Pathwise differentiation is limited to Delta and Vega
- Implied volatility assumes a European Black-Scholes-Merton inverse with finite configured bounds
- Surface smiles use total-variance interpolation without parametric fitting or repair
- Market bid/ask quotes may be missing, crossed, delayed, or stale; last trades may be older still
- Midpoint is a research input and does not represent a guaranteed executable price
- Chain analytics use an explicit flat interest rate and flat continuous dividend yield
- Surface log-moneyness uses the maturity-specific forward under those flat carry assumptions
- Default surface queries do not extrapolate beyond observed strike or maturity support
- Local skew and curvature are derivatives with respect to log-forward-moneyness
- Binomial pricing uses only CRR and constant per-tree market assumptions
- American exercise is available on discrete CRR or LSM dates; no discrete-dividend schedule is modeled
- LSM exercise uses a discrete simulation schedule and a polynomial continuation approximation
- LSM SE does not include policy-training variation or exercise-grid approximation error
- Asian options use discrete fixed-strike averaging and pay only at maturity
- The Asian initial spot is excluded by default and included only by explicit contract policy
- Arithmetic Asians have no analytical benchmark in this phase; geometric Asians use exact discrete moments
- Digital strike equality pays zero under the strict ITM convention
- Digital payoffs are discontinuous and can have relatively high Monte Carlo variance near strike
- Single barriers use inclusive hits and always monitor the initial spot
- Discrete barriers can miss between-date crossings; bridge mode corrects this only for constant-volatility GBM
- Bridge survival weighting supports expiry rebates but rejects positive hit-time rebates
- Continuous barrier formulas support zero rebate; bridge hit-time moments are unavailable
- Barrier Greeks, double barriers, and stochastic-volatility bridge corrections are not implemented
- Basket weights are nonnegative and normalized; signed spread portfolios are not modeled
- Cross-asset dependence is a constant Gaussian correlation matrix
- Basket Deltas, per-asset Vegas, cross-Gammas, and correlation Greeks are deferred
- Correlation sensitivity is an equicorrelation scenario diagnostic, not a claimed exact Greek
- Terminal basket pricing is European; basket barriers and basket Asians are not implemented
- Heston pricing supports European vanilla terminal payoffs; American and exotic Heston payoffs are absent
- Heston uses uniform time steps; MC confidence intervals exclude FTE/QE discretization bias
- FTE retains raw variance proposals, while QE-M uses a nonnegative moment-matched transition
- Heston parameters are risk-neutral inputs; surface calibration is deferred to Phase 14
- Heston assumes a CIR variance process with constant parameters, rates, and dividend yield
- CRR probability can be invalid for coarse steps or extreme carry/volatility combinations
- Primary tree storage is O(N), while backward-induction runtime is O(N²)
- Provider expirations are mapped to a documented 4:00 p.m. New York convention
- No settlement, exercise, or instrument taxonomy beyond reported provider metadata is inferred

U.S. listed equity and ETF options are generally American-style. Phase 5 preserves the provider's
actual exercise style but continues to use European Black-Scholes implied volatility as a stated
analytical convention. Phase 8 does not silently replace that IV convention or add American implied
volatility. The CRR engine can separately price an explicitly supplied American contract. Discrete
dividends, liquidity, transaction costs, and market microstructure are not modeled.

At `T = 0`, vanilla, Asian, and basket contracts return intrinsic value from their time-zero
underlying value; digitals apply their strict spot-versus-strike indicator; barriers apply initial
hit and IN/OUT rebate semantics. At
`sigma = 0`, European terminal engines use the discounted deterministic terminal payoff, Asians use
the deterministic monitored average, barriers evaluate scheduled or exact continuous crossings,
and the American CRR engine optimizes over its discrete deterministic exercise dates. Derived values
outside finite floating-point range raise a quantitative domain error rather than returning NaN or
infinity.

## Roadmap

Phase 14 can calibrate the Phase 13 Heston engine to observed volatility surfaces without replacing
its simulation or analytical pricing boundaries. Later phases can add parametric surface fitting,
quote-repair policy, additional variance-reduction techniques, richer stopping policies, jump diffusion, regime
switching, portfolio aggregation and risk, performance work, quasi-Monte Carlo, and eventually an
interactive dashboard. Those features are architectural consumers of the existing contract,
process, payoff, pricing, RNG, cluster, Greek, volatility, and normalized market-data boundaries;
none are implemented in Phase 13.
