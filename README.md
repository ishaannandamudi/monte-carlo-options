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
Phase 14 calibrates the same risk-neutral Heston model to option prices, BSM implied volatilities,
or total variance with bounded local/global optimization, market-data cleaning, fit diagnostics,
identifiability analysis, stability studies, and structured holdout validation.
Phase 15 adds a capability-aware registry and a research comparison layer across BSM, GBM Monte
Carlo, European/American CRR, Longstaff-Schwartz, and analytical/Monte Carlo Heston. It reports
price and Greek dispersion, equivalent IV, market fit, quote-spread diagnostics, compute frontiers,
calibration uncertainty, theoretical consistency checks, and explicit model-risk flags.
Phase 16 adds typed mixed-instrument portfolios, signed quantities, mark-to-model valuation,
canonical and scaled Greek aggregation, exposure groupings, deterministic scenario/stress grids,
full repricing, local Greek approximations, P&L explain, and portfolio model-spread analysis.
Phase 17 adds portfolio historical, parametric-normal, and correlated-factor Monte Carlo Value at
Risk and Expected Shortfall; Euler and empirical-tail risk contributions; deterministic backtests;
and a combined report that keeps named stresses separate from probabilistic loss samples.
Phase 18 adds risk-neutral Merton jump diffusion with lognormal compound-Poisson jumps, exact
aggregate-jump terminal and path simulation, a conditional-lognormal Poisson-mixture pricer,
implied-volatility smiles, jump diagnostics and sensitivities, plus comparison, portfolio, and
explicit risk-factor integration.
Phase 19 adds finite Markov-chain diagnostics and Gaussian hidden Markov models for real-world
latent-regime research, including log-space filtering/smoothing, Viterbi decoding, deterministic
multi-start Baum-Welch estimation, probability forecasts, and regime-conditioned VaR/ES.

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
- Typed multi-expiry Heston calibration datasets with Phase 5 chain and Phase 6 surface conversion
- Price, BSM implied-volatility, and total-variance calibration targets
- Uniform, Vega, inverse-Vega, bid/ask, and custom weights with optional expiry balancing
- OTM-side selection, quote-quality/moneyness filters, price bounds, and deterministic deduplication
- Native bounded least-squares, differential evolution, hybrid refinement, and deterministic starts
- Maturity-scoped characteristic-function reuse during each calibration vector evaluation
- Warm starts, robust loss, optional Feller/prior penalties, and finite failed-evaluation penalties
- Price/IV residual grids, expiry/moneyness loss, ATM/skew fit, and active-bound diagnostics
- Local Jacobian rank, singular values, condition number, and optional covariance/correlation
- Repeated-start stability and expiry/strike/wing holdout validation
- Thin FastAPI endpoints with configurable resource limits
- Deterministic, numerical, statistical, and API tests
- Typed model registry with contract compatibility, objective complexity metadata, and structured
  exclusions instead of silent engine substitution
- Cross-model price, pairwise, reference, Greek, equivalent-IV, and median-runtime comparisons
- Market price/IV fit with weighted and bid/ask-aware errors by expiry and moneyness
- ATM, skew, curvature, residual-surface, and structured holdout comparisons
- Accuracy-versus-compute frontiers in each engine's natural budget variable
- Heston calibration-ensemble valuation dispersion and local covariance propagation
- Input-perturbation stability, cross-engine consistency checks, and typed model-risk reports
- Mixed European, American, Asian, digital, barrier, and basket portfolios with explicit models
- Signed position quantities, gross long/short value, net value, and isolated position failures
- Position and portfolio Delta, Gamma, raw Vega, calendar Theta, and raw Rho where available
- Net/gross Greeks plus dollar Delta, one-percent Delta/Gamma, one-vol-point Vega, daily Theta,
  and one-percent Rho scaling
- Value and risk grouping by underlying, expiry, option type, instrument class, and pricing model
- Per-asset basket Delta from Common-Random-Number finite differences; no fabricated cross-Gamma
- Spot, volatility, rate, dividend, time, Heston-parameter, correlation, and per-asset scenarios
- Authoritative full repricing with local Greek approximations and explicit higher-order residuals
- Ordered one- and multi-dimensional scenario grids, named stresses, and contribution identities
- Position and portfolio P&L explain plus descriptive Phase 15 model-spread integration
- Explicit Phase 6 sticky-strike BSM surface valuation and Phase 14 calibrated-Heston references
- Dated same-period historical factor replay with ERROR, ZERO_SHOCK, and DROP_SCENARIO policies
- Explicit `Loss = -PnL`, configurable VaR quantiles, and fractional-mass empirical ES
- Delta-normal VaR/normal ES with sample or supplied PSD covariance and explicit IID horizon scaling
- Batched multivariate-normal and covariance-standardized Student-t factor Monte Carlo
- Full revaluation, Delta, and diagonal Delta-Gamma risk modes with explicit cross-term limits
- Marginal factor VaR, Euler component VaR, leave-one-out incremental VaR, and tail ES contributions
- Risk contribution groupings by underlying, instrument class, and selected pricing model
- Skew, excess-kurtosis, Jarque-Bera, quantile, worst-scenario, and concentration diagnostics
- Strict VaR exceptions plus stable Kupiec POF, Christoffersen independence, and coverage diagnostics
- Typed Merton log-jump parameters with the exact risk-neutral jump compensator
- O(paths) exact terminal compound-Poisson simulation and optional exact interval path transitions
- Stable conditional-lognormal Poisson-mixture call/put pricing with tail-mass diagnostics
- Merton Monte Carlo uncertainty, jump-count moments, zero-jump frequency, and path convergence
- Merton implied-volatility smiles, parameter sweeps, standard Greeks, and named jump sensitivities
- BSM/Heston descriptive comparisons and Phase 15 market-fit support by expiry and moneyness
- European vanilla portfolio pricing, diffusion-volatility Greeks, explicit jump-parameter shocks,
  deterministic realized-jump stresses, and opt-in Phase 17 jump-factor histories
- Validated finite-state Markov chains, stationary distributions, persistence, expected duration,
  entropy, and multi-horizon probability forecasts
- Univariate and multivariate Gaussian HMMs with diagonal or full covariance emissions
- Log-space forward/backward inference, filtered versus smoothed probabilities, and Viterbi paths
- Baum-Welch EM with covariance/transition floors, low-occupancy handling, and monotonic histories
- Quantile or seeded random initialization, deterministic multi-start fitting, and canonical labels
- State occupancy, moments, volatility, transition, duration, AIC/BIC, and model-order diagnostics
- Explicit standardization transforms, one-step online filtering, and bounded walk-forward research
- State-conditional normal risk and sampled Gaussian-mixture VaR/ES without averaging state VaRs
- Phase 15 residual grouping, unchanged Phase 16 stress context, and Phase 17 loss conventions
- Separate real-world Markov-switching GBM research simulation with no pricing-drift substitution

The current scope deliberately excludes general surface fitting/repair, double/multi-asset barriers
and path-dependent payoffs beyond Asians and single barriers, non-CRR tree models, local volatility,
non-Merton jump laws, stochastic-volatility-with-jumps, portfolio optimization, parallelism, a GPU backend,
quasi-Monte Carlo, and a frontend.

## Latent regimes and Gaussian HMMs

Phase 19 treats historical-return HMMs as real-world (**P**-measure) descriptive models. It never
uses fitted historical state means as risk-neutral option-pricing drifts. The default Gaussian HMM
uses diagonal covariance and orders fitted states by first-feature mean; full covariance and
volatility ordering are explicit alternatives. Filtered probabilities use data through the current
time, while smoothed probabilities use the full sample and may change with future observations.

Baum-Welch fitting runs in log space, derives deterministic PCG64 child streams for independent
starts, floors variances and transitions, and retains state parameters when posterior occupancy is
too small. The implementation accumulates transition sufficient statistics without a full
`T x K x K` allocation. AIC/BIC, state summaries, stationary probabilities, duration, model-order
comparison, one-step filtering, forecasts, and bounded walk-forward analysis support research and
diagnostics without semantic labels or trading recommendations.

The regime-risk endpoint samples the state and conditional Gaussian factor vector jointly, then
uses the Phase 17 `Loss = -PnL` convention. Mixture VaR/ES is calculated from mixture losses; it is
never a probability-weighted average of state VaRs.

Main endpoints:

- `POST /api/regime/hmm/fit`
- `POST /api/regime/hmm/decode`
- `POST /api/regime/hmm/forecast`
- `POST /api/regime/hmm/model-order`
- `POST /api/regime/hmm/filter`
- `POST /api/regime/risk`
- `POST /api/regime/markov/diagnostics`

See [docs/phase19-regime-hmm.md](docs/phase19-regime-hmm.md) for algorithms, edge policies,
look-ahead rules, and integration boundaries. Performance methodology is in
[benchmarks/phase19-regime-hmm.md](benchmarks/phase19-regime-hmm.md).

## Merton jump diffusion

Phase 18 models log jump sizes as `Y ~ N(mu_j, sigma_j²)` and the stock multiplier as
`J = exp(Y)`. Under the risk-neutral measure,

```text
dS_t / S_(t-) = (r - q - lambda*kappa_J) dt + sigma dW_t + (J - 1) dN_t
kappa_J = E[J - 1] = exp(mu_j + 0.5*sigma_j²) - 1
```

The compensator preserves `E[S_T] = S_0 exp((r-q)T)`. Terminal simulation draws one Poisson count,
one diffusion normal, and one aggregate jump normal per path; it never creates a path-by-jump
matrix. Conditional on `N_T=n`, terminal log spot has mean
`log(S_0)+(r-q-lambda*kappa_J-0.5*sigma²)T+n*mu_j` and variance
`sigma²T+n*sigma_j²`. The analytical engine prices that conditional lognormal distribution
directly and weights it by stable SciPy Poisson probabilities. Truncation stops at a configured
tail probability or maximum term count and reports omitted mass and convergence.

`lambda=0` dispatches exactly to Black-Scholes-Merton. Zero jump volatility produces deterministic
jump sizes, while zero diffusion volatility retains Poisson and jump-size uncertainty. Standard
Merton Vega means diffusion-volatility sensitivity. Sensitivities to intensity, jump mean, and jump
volatility are named separately. Parameter scenarios change risk-neutral model parameters; the
optional realized log-jump stress changes spot deterministically and is kept separate from model
probabilities. Phase 17 uses Merton factors only when the caller supplies explicit mappings and
histories; it performs no jump detection or calibration from returns.

The main endpoints are:

- `POST /api/pricing/merton/analytical`
- `POST /api/pricing/merton/monte-carlo`
- `POST /api/analytics/merton/compare`
- `POST /api/analytics/merton/convergence`
- `POST /api/analytics/merton/series-convergence`
- `POST /api/analytics/merton/smile`
- `POST /api/analytics/merton/sensitivity`
- `POST /api/analytics/merton/jump-diagnostics`
- `POST /api/analytics/merton/greeks`

See [docs/phase18-merton-jump-diffusion.md](docs/phase18-merton-jump-diffusion.md) for the numerical
and integration contracts. Local measurements are recorded in
[benchmarks/phase18-merton.md](benchmarks/phase18-merton.md).

## Portfolio VaR and Expected Shortfall

Phase 17 uses one sign convention throughout:

```text
PnL  = V_horizon - V_today
Loss = -PnL
VaR_alpha = alpha-quantile(Loss)
```

Historical and Monte Carlo ES use the worst exact empirical mass `N * (1-alpha)`. Whole tail
observations receive weight one and, when needed, the next observation receives the fractional
remaining weight. If the tail mass is below one observation, the worst observation defines ES.
VaR uses `numpy.quantile` with the explicitly configured method, defaulting to `linear`.

Historical risk replays simultaneous dated shocks so observed cross-factor dependence is retained.
Full revaluation delegates each scenario to Phase 16. Delta mode uses mapped local exposures;
diagonal Delta-Gamma adds `0.5 * Gamma_i * x_i^2` and explicitly omits cross-Gamma. Missing factors
error by default and can instead zero that named shock or drop the entire dated scenario. Historical
shocks are never silently square-root scaled. Optional time passage is separate and off by default.

Parametric risk treats daily factor changes as multivariate normal. With P&L exposure `w`, daily
mean `m`, and covariance `Sigma`, horizon scaling under the configured IID assumption uses
`m_h = h*m` and `Sigma_h = h*Sigma`. Normal loss VaR and ES are:

```text
mu_L = -(w' m_h + Theta passage)
sigma_L = sqrt(w' Sigma_h w)
VaR = mu_L + z_alpha sigma_L
ES  = mu_L + sigma_L phi(z_alpha)/(1-alpha)
```

Factor marginal VaR differentiates this expression. Position component VaR is the Euler derivative
with respect to each position scale and sums to total VaR, including mapped mean and optional Theta
passage. Incremental VaR removes one position and recomputes risk. For empirical methods, ES
contributions average position losses with exactly the same fractional tail weights as portfolio ES.

Monte Carlo builds correlated factor draws from the supplied or historical sample covariance using
isolated PCG64 state. Normal and covariance-standardized Student-t draws are supported. Batches
avoid retaining a simulation-by-position matrix in Greek modes; the portfolio loss vector remains
in memory for exact quantiles, and selected tail shocks are deterministically regenerated when
contributions are requested. Antithetic factors are optional and carry no promised tail benefit.

VaR backtesting counts an exception only when `Loss > predicted VaR`; equality is not an exception.
Kupiec proportion-of-failures, Christoffersen first-order independence, and their conditional-
coverage sum use stable likelihood calculations at zero and all exceptions. The traffic-light field
is a descriptive two-standard-deviation comparison with the expected exception count, not a
regulatory capital conclusion. Named Phase 16 stress losses remain separate fields and never enter
the VaR/ES distribution.

See [docs/phase17-var-es-risk.md](docs/phase17-var-es-risk.md) for exact mapping, estimator,
contribution, batching, failure, and resource conventions. Local measurements are in
[benchmarks/phase17-var-es-risk.md](benchmarks/phase17-var-es-risk.md).

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
five parameters are scenario analysis only. Phase 14 calibration consumes this analytical engine;
it does not change Phase 13 pricing semantics.

See [docs/phase13-heston.md](docs/phase13-heston.md) for the formulas, branch policies, memory model,
and API accounting. Measured timings and validation values are in
[benchmarks/phase13-heston.md](benchmarks/phase13-heston.md).

## Heston calibration

Phase 14 fits the canonical native parameter vector `[v0, kappa, theta, sigma_v, rho]` inside
explicit finite bounds. Every objective value uses the Phase 13 semi-analytical price. `PRICE`
minimizes price differences; `IMPLIED_VOLATILITY` prices under Heston and uses the Phase 4 solver to
compare European BSM IV; `TOTAL_VARIANCE` compares `T sigma^2`. Monte Carlo never runs inside the
optimizer.

The data layer accepts direct typed observations, converts enriched Phase 5 chains without
rewriting their quality decisions, and flattens Phase 6 surfaces while retaining their BSM IV
convention. It checks European price bounds, can remove stale, crossed, missing-mid, wide-spread,
arbitrage-invalid, maturity, price, or moneyness observations, and reports every exclusion. The
default OTM policy selects puts below forward, calls above forward, and calls at forward equality.
Duplicates are resolved by spread, issue count, freshness, and source.

Weights are normalized to mean one. `VEGA` uses floored BSM Vega squared and emphasizes liquid
high-Vega points; `INVERSE_VEGA` uses its reciprocal; `BID_ASK` uses inverse squared floored spread;
`CUSTOM` uses the nonnegative observation weight. Optional expiry balancing gives every retained
maturity equal total weight before final normalization.

Bounded SciPy least-squares supports linear, soft-L1, and Huber loss. Differential evolution is a
deterministic single-process global alternative. `HYBRID` sends the global best to least-squares and
retains the global candidate, while multi-start local calibration combines an optional user/warm
start, a surface heuristic, and deterministic bounded random starts. Feller is never a hard
constraint; an optional soft penalty is
`lambda * max(sigma_v^2 - 2*kappa*theta, 0)^2`. Optional bound-scaled prior regularization is
disabled by default.

Integration and IV failures become finite penalty residuals and remain visible in failure counts.
Results separately report raw RMSE/MAE/max error, weighted RMSE and objective, optimizer work,
runtime, prices per second, active bounds, start summaries, per-expiry and moneyness loss, and full
market/model residuals up to a configured payload cap. A finite-difference local Jacobian supplies
rank, singular values, condition number, and a local covariance/correlation approximation only when
full rank with enough observations. These diagnostics do not imply unique global identification.

Fit analytics return an ordered market-versus-Heston surface, nearest-forward ATM term structure,
and linear skew by expiry. Repeated-start stability exposes objective and parameter dispersion.
Structured validation can hold out entire expiries, every nth strike within each expiry, or wings;
it never defaults to a random option-by-option split.

See [docs/phase14-heston-calibration.md](docs/phase14-heston-calibration.md) for the exact target,
weight, penalty, optimizer, failure, and uncertainty definitions. Local timing and recovery results
are in [benchmarks/phase14-heston-calibration.md](benchmarks/phase14-heston-calibration.md).

## Model comparison and model risk

Phase 15 registers only engines that already exist. `BLACK_SCHOLES`, `MONTE_CARLO_GBM`,
`CRR_BINOMIAL`, `HESTON_ANALYTICAL`, and `HESTON_MONTE_CARLO` apply to European vanilla
contracts. `CRR_AMERICAN` and `LONGSTAFF_SCHWARTZ` apply to American vanilla contracts. Heston
engines require an explicit parameter set or a Phase 14 calibration result. Incompatible models are
returned as structured exclusions; a numerical failure in one compatible engine remains a
per-model failure while other results survive.

Price dispersion is the population minimum, maximum, range, mean, median, and standard deviation
of successful model prices. Relative range divides by a floored median by default, with spot or a
successful reference price available as explicit alternatives. The reference is a convention, not
a truth label. Pairwise matrices use `price_i - price_j`; relative entries divide by
`max(abs(price_j), floor)`. Model dispersion is descriptive and has no confidence coverage.

Available Greeks are compared without filling missing cells. BSM supplies analytical Delta, Gamma,
Vega, calendar-time Theta, and Rho; CRR supplies native Delta, Gamma, and Theta; GBM Monte Carlo can
use the existing common-random-number finite-difference estimators. Heston analytical comparison
uses central spot/rate/maturity differences for Delta, Gamma, Rho, and Theta. Generic Heston Vega
is omitted because sensitivity to `v0` or `sqrt(v0)` is not the same quantity as BSM Vega.

Every valid European model price is inverted through the Phase 4 solver to an equivalent BSM IV.
Single-contract IV residuals use the request's stated constant market volatility. Surface
comparison uses each observed Phase 6 IV as the BSM-equivalent baseline. Heston-versus-BSM results
therefore always expose which BSM volatility convention supplied the comparison.

Market fit reuses Phase 14 filtering, quote-quality decisions, OTM/all selection, weights, expiry
balancing, and deduplication. It reports price RMSE, MAE, weighted RMSE, maximum and median absolute
error, signed bias, IV RMSE/MAE/max/bias, fraction inside bid/ask, and distance to the nearest edge
when outside. Metrics are grouped by expiry and the Phase 14 signed-moneyness buckets. ATM, skew,
and curvature use the Phase 6 `k=log(K/F)` and total-variance PCHIP convention. Structured expiry,
strike, and wing holdouts keep training and validation metrics separate. Optional user labels group
observations descriptively without adding automatic regime detection.

Runtime is the median of configured repeated calls for one contract. Model output separately keeps
paths, time steps, tree steps, standard error, confidence interval, and early-exercise behavior.
Compute-frontier experiments vary paths for GBM/Heston Monte Carlo, steps for CRR, or path-time work
for LSM and report runtime, reference error, stochastic SE/CI width, and an allocation estimate.
No point is labeled optimal.

`ModelRiskReport` keeps Monte Carlo sampling error, numerical disagreement, Heston calibration
dispersion, cross-model dispersion, and market quote spread in distinct fields. A Heston ensemble
propagates repeated calibration solutions into price and IV dispersion. When a Phase 14 local
covariance matrix is supplied, `grad(V)^T Cov grad(V)` is returned as a local linear approximation,
not a confidence interval. Configurable thresholds only create descriptive flags.

See [docs/phase15-model-comparison.md](docs/phase15-model-comparison.md) for the complete capability,
compatibility, statistical, failure, and API conventions. Representative local measurements are in
[benchmarks/phase15-model-comparison.md](benchmarks/phase15-model-comparison.md).

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
├── calibration/   # Heston datasets, objectives, optimizers, diagnostics, and synthetic fixtures
├── core/          # environment settings, logging, errors, numerical guards
├── market_data/   # provider protocol, Massive adapter, and deterministic fake
├── models/        # vanilla/exotic contracts, schedules, configs, enums, and typed results
├── payoffs/       # vectorized vanilla, path-dependent, digital, barrier, and basket payoffs
├── pricing/       # analytical, Monte Carlo, tree, LSM, exotic, basket, and Heston engines
├── portfolio/     # mixed-position valuation, Greeks, scenarios, stress, and P&L explain
├── risk/          # VaR/ES estimators, factor simulation, contributions, and backtesting
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

Calibrate Heston to a deterministic price/IV dataset:

```python
from backend.app.calibration.heston import calibrate_heston
from backend.app.calibration.synthetic import generate_synthetic_heston_dataset
from backend.app.models.heston_calibration import (
    HestonCalibrationConfig,
    HestonCalibrationOptimizer,
    HestonCalibrationTarget,
)

calibration_data = generate_synthetic_heston_dataset(
    maturities=(0.25, 0.5, 1.0, 2.0),
    log_moneyness=(-0.2, -0.1, 0.0, 0.1, 0.2),
)
calibration_result = calibrate_heston(
    calibration_data,
    HestonCalibrationConfig(
        target=HestonCalibrationTarget.IMPLIED_VOLATILITY,
        optimizer=HestonCalibrationOptimizer.LEAST_SQUARES,
        multi_start_count=2,
        random_seed=42,
    ),
)
print(calibration_result.calibrated_parameters)
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

Phase 15 research routes are:

```text
GET  /api/analytics/models/registry
POST /api/analytics/models/compare
POST /api/analytics/models/market-fit
POST /api/analytics/models/risk
POST /api/analytics/models/compute-frontier
POST /api/analytics/models/consistency
POST /api/analytics/models/surface-comparison
```

Phase 16 portfolio routes are:

```text
POST /api/portfolio/value
POST /api/portfolio/greeks
POST /api/portfolio/scenario
POST /api/portfolio/scenario-grid
POST /api/portfolio/stress
POST /api/portfolio/pnl-explain
POST /api/portfolio/model-risk
```

Phase 17 market-risk routes are:

```text
POST /api/risk/historical
POST /api/risk/parametric
POST /api/risk/monte-carlo
POST /api/risk/portfolio
POST /api/risk/contributions
POST /api/risk/backtest
POST /api/risk/report
```

The three method endpoints use method-specific typed bodies; `/portfolio` dispatches a unified
request. `/contributions` requires contribution output. `/report` adds named Phase 16 stresses to a
separate response field. No endpoint returns raw loss samples or a full simulation-position matrix.

Each position carries a typed instrument, explicit market snapshot, signed quantity, optional
pricing model, and model-specific numerical configuration. Omitted models follow the documented
policy: BSM for European vanilla, American CRR for American vanilla, analytical pricing for
digitals and supported geometric payoffs, and Monte Carlo for arithmetic Asians, barriers, and
arithmetic baskets. Incompatible explicit models produce a structured position failure; they are
never silently replaced.

A minimal comparison request supplies `contract`, `market`, and
`config.selected_models`. Optional fields carry Heston parameters or a calibration result and the
existing GBM, tree, Heston, or LSM numerical configurations. Public responses contain bounded
lists and scalar diagnostics; simulation arrays and Heston characteristic-function values remain
internal.

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
- `POST /api/calibration/heston`
- `POST /api/calibration/heston/surface`
- `POST /api/analytics/heston/calibration/diagnostics`
- `POST /api/analytics/heston/calibration/fitted-surface`
- `POST /api/analytics/heston/calibration/stability`
- `POST /api/analytics/heston/calibration/validate`
- `POST /api/portfolio/value`
- `POST /api/portfolio/greeks`
- `POST /api/portfolio/scenario`
- `POST /api/portfolio/scenario-grid`
- `POST /api/portfolio/stress`
- `POST /api/portfolio/pnl-explain`
- `POST /api/portfolio/model-risk`

Calibrate a small IV dataset in bounded native Heston parameter space:

```bash
curl -X POST http://127.0.0.1:8000/api/calibration/heston \
  -H 'Content-Type: application/json' \
  -d '{
    "dataset": {
      "spot": 100.0, "risk_free_rate": 0.02, "dividend_yield": 0.01,
      "underlying_symbol": "SYNTHETIC",
      "observations": [
        {"strike": 90.0, "maturity_years": 1.0, "option_type": "PUT",
         "observed_implied_volatility": 0.24},
        {"strike": 101.0050167, "maturity_years": 1.0, "option_type": "CALL",
         "observed_implied_volatility": 0.20},
        {"strike": 110.0, "maturity_years": 1.0, "option_type": "CALL",
         "observed_implied_volatility": 0.19}
      ]
    },
    "config": {
      "target": "IMPLIED_VOLATILITY", "optimizer": "LEAST_SQUARES",
      "multi_start_count": 2, "random_seed": 42, "max_iterations": 100
    }
  }'
```

Supplying a previous five-field `HestonParameters` object as `config.initial_parameters` is a warm
start. Runtime, detailed residuals, and optimizer start summaries are outputs; no pricing paths or
Fourier integration arrays are returned.

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
MAX_HESTON_CALIBRATION_OBSERVATIONS=250
MAX_HESTON_CALIBRATION_STARTS=10
MAX_HESTON_CALIBRATION_ITERATIONS=500
MAX_HESTON_CALIBRATION_SURFACE_POINTS=500
MAX_HESTON_CALIBRATION_DIAGNOSTIC_POINTS=500
MAX_MODEL_COMPARISON_MODELS=7
MAX_MODEL_COMPARISON_OBSERVATIONS=500
MAX_MODEL_COMPARISON_SURFACE_POINTS=1000
MAX_MODEL_COMPARISON_EXPERIMENT_POINTS=20
MAX_MODEL_COMPARISON_RUNTIME_REPETITIONS=10
MAX_RISK_HISTORICAL_SCENARIOS=10000
MAX_RISK_MONTE_CARLO_SIMULATIONS=1000000
MAX_RISK_POSITIONS=1000
MAX_RISK_SCENARIO_POSITION_EVALUATIONS=250000
MAX_RISK_CONTRIBUTION_POSITIONS=250
MAX_RISK_BACKTEST_OBSERVATIONS=10000
MAX_RISK_TAIL_OUTPUT=100
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

Heston calibration limits separately cap source observations, deterministic starts, optimizer
iterations, Phase 6 surface points, and detailed residual output. Stability runs share the start
limit. These API caps do not alter direct-library mathematics or defaults.

Model-comparison limits cap requested engines, observation and surface grids, compute-frontier
points, returned residuals, and timing repetitions. Existing engine-specific path, node, tree-step,
and Heston limits are enforced on nested configurations before work starts.

Portfolio limits cap positions, scenario coordinates, Cartesian grid cells, stress definitions,
and alternate models. Existing path, monitoring, tree, LSM, Heston, and multi-asset node limits are
also checked for each nested position before valuation. Scenario products are checked before cells
are allocated.

Risk limits independently cap dated histories, generated factor scenarios, positions,
scenario-position full-revaluation work, contribution widths, backtest histories, and returned tail
scenarios. The API checks these products before factor generation, repricing, or matrix allocation,
then applies every existing nested pricing-engine limit. Library calls remain available for
deliberate larger research workloads.

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
Phase 14 tests cover typed datasets and bounds; all target and weighting modes; expiry balancing;
quality, price, spread, moneyness, OTM, and duplicate policies; Phase 5/6 conversion; exact and noisy
recovery; local, global, hybrid, multi-start, warm-start, robust-loss, Feller, regularization, and
finite-failure behavior; active bounds; residual, expiry, moneyness, Jacobian, covariance, fitted
surface, ATM, skew, stability, and structured holdout diagnostics; every endpoint and resource cap;
and deterministic request replay. Opt-in cases measure objective scaling at 10/50/200 observations,
50-observation local calibration, modest hybrid search, and a 200-point fitted surface.
Phase 15 tests cover registry ordering and capabilities; compatibility; exact price, pairwise,
reference, Greek, and IV dispersion; BSM/GBM, BSM/CRR, deterministic-Heston, Heston-MC, and
American CRR/LSM comparisons; market price/IV/spread metrics; expiry, moneyness, ATM, skew, and
curvature diagnostics; structured holdouts; compute frontiers; input and calibration stability;
local covariance propagation; consistency flags; partial failures; every API route; deterministic
signatures; serialization; and all new resource limits. Opt-in cases cover a repeated five-model
contract comparison and a maturity-batched Heston surface workload.
Phase 16 tests cover mixed typed portfolios, signed values, aggregate/scaled Greeks, deterministic
and stochastic dispatch, all scenario dimensions, full-reprice identities, local approximations,
grids, named stresses, P&L explain, model spreads, caching, APIs, and nested work limits.
Phase 17 tests cover the loss sign; explicit empirical quantiles; fractional-tail ES; zero-risk,
single-factor, covariance, correlation, and horizon identities; historical full and Greek replay;
missing factors; Heston and basket-correlation shocks; normal and Student-t Monte Carlo; exact odd
antithetic counts; component/incremental VaR and ES contribution identities; tail diagnostics;
strict exception equality; Kupiec and Christoffersen boundaries; stress separation; every route;
serialization; reproducibility; and pre-allocation rejection. Opt-in cases cover parametric models
through 500 factors and 100k/500k Monte Carlo factor draws.
Phase 18 tests cover parameter domains and the compensator; fixed aggregate-jump drivers; terminal
and full-path shapes; multiple jumps per interval; theoretical stock and Poisson moments; exact BSM
reduction; a frozen analytical benchmark; parity; large-intensity truncation; jump-only cases;
Monte Carlo error scaled by SE; deterministic child streams; smiles, sweeps, Greeks, objective
residuals, registry, market-fit, portfolio, scenario, Phase 17 mapping, APIs, and resource limits.
The opt-in benchmark prices one million terminal paths and a 100-strike smile.
Phase 19 tests compare forward/backward probabilities and Viterbi paths with brute-force state
enumeration; verify Markov validation, stationary distributions, duration, and forecasts; exercise
EM monotonicity, covariance floors, deterministic multi-start recovery, canonical labels, AIC/BIC,
multivariate covariance, look-ahead-safe filtering, walk-forward scoring, Phase 15/16/17
integrations, every API route, response bounds, and real-world switching-GBM reproducibility. Opt-in
cases cover 1,000 and 10,000 observations with two and four states.
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
- Discontinuous returns are available through Phase 18 Merton jump diffusion; Heston remains the
  separate stochastic-volatility model, and no combined stochastic-volatility/jump model exists
- No transaction costs, executable-price guarantee, or discrete hedging
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
- Standalone closed-form barrier Greeks remain absent; the portfolio layer can finite-difference
  its assigned barrier engine. Double barriers and stochastic-volatility bridges are absent
- Basket weights are nonnegative and normalized; signed spread portfolios are not modeled
- Cross-asset dependence is a constant Gaussian correlation matrix
- Phase 16 supplies basket per-asset Delta with CRN finite differences; per-asset Vegas,
  cross-Gammas, and correlation Greeks remain deferred
- Correlation sensitivity is an equicorrelation scenario diagnostic, not a claimed exact Greek
- Terminal basket pricing is European; basket barriers and basket Asians are not implemented
- Heston pricing supports European vanilla terminal payoffs; American and exotic Heston payoffs are absent
- Heston uses uniform time steps; MC confidence intervals exclude FTE/QE discretization bias
- FTE retains raw variance proposals, while QE-M uses a nonnegative moment-matched transition
- Heston calibration estimates risk-neutral parameters from European research prices or BSM IVs
- Calibration assumes one flat risk-free rate and dividend yield for the complete input dataset
- Heston parameters can be weakly identified even when model prices fit closely
- Calibration covariance is a local linear approximation, not posterior or model-risk uncertainty
- Weights, filters, robust loss, bounds, and penalties define different estimation problems
- Heston assumes a CIR variance process with constant parameters, rates, and dividend yield
- Cross-model dispersion depends on the explicitly selected engines and is not a probability law
- A comparison reference model is a reporting convention and is not presumed correct
- Equivalent implied volatility is a BSM representation of price, not a claim that the source model
  has constant volatility
- Market-fit results inherit the selected quote, filter, weighting, and carry assumptions
- Runtime measurements depend on the host and are descriptive medians rather than service guarantees
- Heston covariance propagation is local and linear; ensemble dispersion depends on supplied fits
- Model-risk flags are threshold diagnostics, not ratings or automated model approvals
- Portfolio Vega and Rho use raw absolute model units; scaled values are separately named
- Portfolio Theta is calendar-time Theta per year; `theta_1day` divides it by 365
- Basket value and scenario P&L grouping uses contractual weights as a reporting allocation;
  per-asset Delta is directly bumped, while cross-Gamma and per-asset Vega remain unavailable
- Generic volatility shocks do not alter Heston parameters; Heston scenarios name parameter shifts
- Full repricing is authoritative. Greek approximations omit vanna, volga, charm, cross-Gamma,
  rate/volatility interactions, correlation risk, and other higher-order terms
- Barrier scenarios assume the existing unhit/current-state contract; past barrier hits are absent
- Asian forward-time scenarios do not model historical fixings or partially observed averages
- Surface-based portfolio BSM valuation is opt-in and uses a sticky-strike Phase 6 query
- Portfolio market inputs are explicit snapshots; `market_data_reference` is provenance only and
  no live market-data fetch occurs during portfolio valuation
- Model-price spread is descriptive rather than probabilistic risk
- VaR and ES are loss-distribution estimates under the selected factor history/model and pricing
  mode; they are not maximum possible losses or predictions of a particular future outcome
- Historical observations are replayed jointly by timestamp; no independent factor resampling,
  filtered historical simulation, volatility scaling, or EVT tail fit is performed
- Historical VaR/ES sampling uncertainty is not assigned a confidence interval in Phase 17
- Delta-normal risk assumes a normal linear factor model; IID multi-day scaling is explicit
- Delta-Gamma risk contains diagonal Gamma only and omits cross-Gamma and other cross sensitivities
- Generic factor Monte Carlo is normal or Student-t and does not infer a Heston process, calibrated
  Heston changes, volatility-surface PCA, or changing correlations without explicit factors
- Empirical VaR and ES retain the portfolio loss vector for exact selection; full-revaluation
  contribution requests are subject to stricter API limits
- Backtest traffic-light output is descriptive and is not a Basel regulatory classification
- Merton parameters are constant and risk-neutral; jump sizes are lognormal and arrivals are
  independent homogeneous Poisson events independent of the diffusion Brownian motion
- Merton analytical and terminal Monte Carlo pricing supports European vanilla contracts only
- Merton full paths use exact independent increments on a regular grid and retain no latent jump list
- Merton calibration is limited to a finite price-residual objective helper; no automatic fit is run
- Generic volatility shifts change Merton diffusion volatility and never alter jump parameters
- Explicit Merton factor histories and stresses are user inputs, not inferred realized jumps
- Bates, Kou, infinite-activity Lévy models, and switching Heston/Merton remain outside the current
  scope
- Historical Gaussian HMMs describe the real-world measure; their fitted means are never silently
  used as risk-neutral derivative-pricing drifts
- HMM observations must be complete and finite; Student-t emissions and live streaming are deferred
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

Phase 20 may address Sobol or other quasi-Monte Carlo research under its own specification. Bates,
other jump laws, switching Heston/Merton, live HFT/broker integration, portfolio optimization, and
a frontend remain unimplemented.
