# Phase 10: Asian and Digital Options

## 1. Scope and architecture

Phase 10 adds two European exotic families under the existing constant-parameter,
risk-neutral GBM model:

- fixed-strike, discretely monitored arithmetic and geometric Asian calls and puts;
- cash-or-nothing and asset-or-nothing digital calls and puts.

`AsianOptionContract` and `DigitalOptionContract` are separate immutable Pydantic models. The
vanilla `OptionContract` is not filled with optional exotic fields. They reuse `OptionType`,
`ExerciseStyle.EUROPEAN`, and `MarketEnvironment`, while each family owns its relevant payout and
simulation configuration. Phase 10 does not model early exercise, continuous Asian averaging,
barriers, rebates, Greeks, or multi-asset payoffs.

The implementation remains divided by responsibility:

- `models/monitoring.py` defines deterministic observation schedules.
- `models/asian.py` and `models/digital.py` define contracts, configurations, enums, and results.
- `simulation/gbm.py` simulates the requested GBM states without payoff knowledge.
- `payoffs/asian.py` and `payoffs/digital.py` contain vectorized payoff transformations.
- `pricing/asian.py` and `pricing/digital.py` contain estimators and analytical formulas.
- `analytics/asian.py` and `analytics/digital.py` contain comparisons and convergence experiments.
- `api/asian.py` and `api/digital.py` enforce HTTP resource limits before calling library code.

## 2. Path-dependent payoff boundary

Asian payoff helpers accept an observation matrix with one path per row and one included
observation per column. They calculate an average and then apply a fixed-strike payoff. They do not
generate paths or discount payoffs. This boundary permits exact fixed-matrix tests and keeps process,
payoff, and pricing rules independent.

The abstraction is deliberately small. A barrier option needs first-passage and rebate semantics,
which are absent here and should not be inferred from an Asian average helper.

## 3. Monitoring schedule

`MonitoringSchedule` supports exactly one of:

- `monitoring_count=n`, generating the future times `iT/n`, for `i=1,...,n`; or
- `monitoring_times=(t_1,...,t_m)`, normalized into increasing order.

Times are year fractions from valuation. Counts must be positive. Explicit times must be finite,
unique, nonnegative, and no greater than maturity. A positive-maturity schedule must contain an
observation. The maturity point is allowed but is not required for an explicit schedule. A regular
schedule always includes maturity. A regular interval that underflows to zero is rejected rather
than producing duplicate times.

At zero maturity, the effective schedule is `(0.0,)`; the average is therefore well defined as
`S_0`. Validation of a regular count does not allocate its time tuple. This lets the API reject an
oversized count before allocating either the schedule or a path matrix.

## 4. Exact GBM at arbitrary times

For strictly increasing positive observation times, define `delta_i = t_i - t_{i-1}`, with
`t_0=0`. The simulator applies the exact transition

```text
log(S_i / S_{i-1})
  = (r - q - 0.5 sigma^2) delta_i + sigma sqrt(delta_i) Z_i.
```

The normal-driver matrix has shape `(paths, future_observations)`. Cumulative log returns produce
only requested states. No hidden daily or uniform grid is created, and irregular dates are not
rounded to indices. The implementation is vectorized across paths and observations.

For antithetic sampling, each complete row `Z=(Z_1,...,Z_m)` is immediately followed by `-Z`.
This preserves dependence across the whole path. An odd request retains one final independent row,
so the returned row count is exactly the requested path count. `sigma=0` and a schedule with no
future time produce deterministic observations without consuming random draws.

## 5. Arithmetic average

For `m` included observations,

```text
A_arith = (1/m) sum_i S_i.
```

NumPy computes row means directly. Arithmetic Asian options have no closed form in this phase.
Their comparison and convergence results therefore do not fabricate an analytical reference.

## 6. Geometric average

The geometric average is evaluated in log space:

```text
A_geo = exp((1/m) sum_i log(S_i)).
```

This avoids direct products and their unnecessary overflow or underflow. GBM states must be finite
and strictly positive before the logarithm. Invalid numerical states raise `NumericalError`.

## 7. Initial-spot convention

`include_initial_spot_in_average` defaults to `false`. When true, `S_0` is prepended exactly once.
For an explicit schedule, a supplied zero is accepted only with this flag; when the flag is true
and zero is omitted, zero is prepended. Thus the policy has one unambiguous effective schedule and
never double-counts `S_0`.

In the geometric distribution, the zero-time observation contributes `log(S_0)` to the mean and
zero Brownian covariance. In path simulation it is a deterministic column, not a random transition.

## 8. Asian payoff and payout date

For either average `A`, the fixed-strike payoffs are

```text
call = max(A - K, 0)
put  = max(K - A, 0).
```

Equality is out of the money and pays zero. Payment occurs at contract maturity, even when the last
explicit averaging observation is earlier. Every payoff is therefore discounted by `exp(-rT)`.

## 9. Discrete geometric distribution

At deterministic times `t_i`, risk-neutral GBM gives

```text
log(S_ti) = log(S_0) + (r-q-0.5 sigma^2)t_i + sigma W_ti.
```

For `G = exp(m^-1 sum_i log(S_ti))`, joint normality implies

```text
log(G) ~ Normal(mu_G, v_G)

mu_G = log(S_0) + (r-q-0.5 sigma^2) (1/m) sum_i t_i

v_G = (sigma^2/m^2) sum_i sum_j min(t_i,t_j).
```

For sorted times, the double covariance sum is evaluated in O(m):

```text
sum_i [2(m-i)-1] t_i,
```

using zero-based `i`. Zero-time observations naturally add no covariance. This is the exact
discrete formula for regular or irregular schedules; it is not a continuous-monitoring proxy.

## 10. Geometric Asian analytical price

Let `s_G=sqrt(v_G)`,

```text
d2 = (mu_G - log(K)) / s_G
d1 = d2 + s_G
E[G] = exp(mu_G + 0.5 v_G).
```

Then

```text
C_0 = exp(-rT) [E[G] N(d1) - K N(d2)]
P_0 = exp(-rT) [K N(-d2) - E[G] N(-d1)].
```

The derivation uses the known lognormal law of `G`; it does not assign `G` an ordinary tradable-
asset drift. With one maturity observation and no `S_0`, it reduces to Black-Scholes-Merton. With
one future observation plus `S_0`, `G=sqrt(S_0 S_T)` and the same general moments apply.

When `v_G=0`, the implementation discounts the deterministic payoff directly and never divides by
zero. At `T=0`, the average is `S_0`. Negative rates, continuous dividends, irregular schedules,
small maturity, and deep strikes follow the same finite formulas. Any intermediate outside finite
floating-point range raises a domain error instead of leaking `NaN` or infinity.

## 11. Arithmetic Asian Monte Carlo

The raw discounted observation for path `j` is

```text
Y_j = exp(-rT) payoff(A_arith,j).
```

The price is its path-weighted sample mean. Plain Monte Carlo treats paths as independent.
Antithetic Monte Carlo forms independent complete-path clusters as described below. Results include
price, estimator variance and SE, normal confidence interval, average diagnostics, ITM frequency,
seed, path and observation counts, method, and measured runtime.

At `sigma=0`, each monitored stock value is `S_0 exp((r-q)t_i)`. The implementation calculates the
deterministic arithmetic or geometric average once and returns exact zero uncertainty without a
stochastic simulation.

## 12. Geometric control variate

For an arithmetic target, use

```text
Y = exp(-rT) arithmetic payoff
X = exp(-rT) geometric payoff
E[X] = exact discrete geometric Asian price

beta_hat = Cov(Y,X) / Var(X)
Y_cv = Y - beta_hat (X-E[X]).
```

`GEOMETRIC_CONTROL` estimates this relationship from independent path observations.
`ANTITHETIC_GEOMETRIC_CONTROL` first clusters both variables using identical antithetic pairs.
Geometric targets reject geometric-control modes because the proposed control would be the target
itself. If sampled control estimator variance is zero, beta is zero and the raw estimator is
retained.

## 13. Antithetic paths

An antithetic cluster contains payoff observations from full normal vectors `Z` and `-Z`. For an
even path count, each pair mean is one independent estimator observation. For an odd path count,
the last independent path is a singleton cluster. Prices retain path weights, so a pair has weight
two and the singleton weight one.

This convention is shared by raw Asian, controlled Asian, digital, vanilla, and LSM estimators.
No method treats correlated members of a pair as independent.

## 14. Control and standard-error methodology

For cluster `c`, let its size be `n_c`, its mean be `U_c`, total paths be `N=sum_c n_c`, and number
of independent clusters be `C`. The path-weighted estimate is

```text
U_hat = (1/N) sum_c n_c U_c.
```

The finite-sample cluster estimator variance is

```text
Var_hat(U_hat)
  = [C/(C-1)] [1/N^2] sum_c [n_c(U_c-U_hat)]^2.
```

The cluster covariance uses the corresponding product of centered, size-weighted scores. These
variances and covariance determine beta for the combined method. The adjusted cluster values are
then evaluated with the same weights and variance formula. The result exposes beta, raw and
adjusted price, raw and adjusted SE, both estimator variances, control sample mean, analytical
control mean, and the sampled variance-reduction ratio. A representative at-the-money call test
requires material reduction, while the code makes no universal improvement claim.

## 15. Digital payoff

Strict ITM indicators define every digital payoff:

```text
cash call  = cash * 1[S_T > K]
cash put   = cash * 1[S_T < K]
asset call = S_T  * 1[S_T > K]
asset put  = S_T  * 1[S_T < K].
```

Equality pays zero for both option types. Cash payout is nonnegative and defaults to one for a cash
digital. Asset digitals reject a cash-payout field so their units remain unambiguous.

## 16. Cash-or-nothing analytical formula

For positive maturity and volatility, the existing Black-Scholes `d1,d2` terms give

```text
cash call = cash exp(-rT) N(d2)
cash put  = cash exp(-rT) N(-d2).
```

The reported ITM probability is the risk-neutral probability `N(d2)` or `N(-d2)`. For positive
cash,

```text
price / [cash exp(-rT)] = P_Q(ITM).
```

Call plus put equals discounted cash for positive volatility and maturity, where terminal equality
has probability zero.

## 17. Asset-or-nothing analytical formula

Asset digitals use

```text
asset call = S_0 exp(-qT) N(d1)
asset put  = S_0 exp(-qT) N(-d1).
```

Tests verify the decompositions

```text
vanilla call = asset call - K * unit-cash call
vanilla put  = K * unit-cash put - asset put.
```

The result's `probability_itm` still reports the risk-neutral event probability from `d2`, not the
share-measure factor from `d1`.

## 18. Digital Monte Carlo

Digital Monte Carlo reuses `GBMProcess.simulate_terminal` and the PCG64 generator factory. It
discounts vectorized terminal digital payoffs and computes price, estimator variance, SE, interval,
ITM frequency, analytical reference, and absolute error. `NONE` and `ANTITHETIC` are supported.
Antithetic uncertainty uses pair clusters and the weighted odd-singleton policy.

At zero maturity, the spot is compared with strike under the strict convention. At zero volatility,
`S_T=S_0 exp((r-q)T)` is compared with strike and discounted; analytical and Monte Carlo results are
identical with zero SE. Zero cash payout also produces exact zero value and uncertainty.

## 19. Discontinuity and estimator variance

A digital indicator is discontinuous at the strike. Near the strike, small terminal changes can
flip the full payout, so Monte Carlo variance can be relatively high and realized convergence can
look noisy. The estimator still has the ordinary asymptotic `N^-1/2` standard-error order under the
model. Phase 10 does not add pathwise digital Greeks or smoothing estimators.

## 20. Comparisons and convergence

Arithmetic Asian comparison runs `NONE`, `ANTITHETIC`, `GEOMETRIC_CONTROL`, and
`ANTITHETIC_GEOMETRIC_CONTROL` with the same root seed and path budget. It reports uncertainty,
runtime, and variance ratios without claiming a closed-form arithmetic value.

Geometric comparison runs plain and antithetic Monte Carlo against the exact discrete analytical
price. Asian convergence supports:

- `PATH_COUNT`: vary path counts with a fixed schedule;
- `MONITORING_COUNT`: vary regular discrete observation counts with a fixed path budget.

Digital convergence compares Monte Carlo with the analytical price. Every experiment uses
`SeedSequence(root_seed)` child seeds, giving independent, deterministic streams across points.
Errors are not required to improve monotonically.

## 21. Memory and performance

Asian path storage is O(paths times effective monitoring count), with float64 observations. No
unrequested intermediate states are retained. Payoffs, averages, and control values add O(paths)
vectors. Vectorized NumPy kernels avoid loops over paths; any scheduling work scales only with the
number of dates. Digital storage is O(paths) because it needs only `S_T`.

The API's node budget counts `paths times effective observations` for every pricing run represented
by a request. Comparison requests account for all two or four method runs. This is a conservative
work/allocation budget; direct library calls remain unrestricted by HTTP settings.

Measured Phase 10 timings and matrix sizes are recorded in
[`benchmarks/phase10-exotics.md`](../benchmarks/phase10-exotics.md). Performance tests are opt-in.

## 22. API safeguards

The endpoints are:

- `POST /api/pricing/asian`
- `POST /api/pricing/asian/geometric-analytical`
- `POST /api/analytics/asian/compare`
- `POST /api/analytics/asian/convergence`
- `POST /api/pricing/digital`
- `POST /api/analytics/digital/compare`
- `POST /api/analytics/digital/convergence`

`MAX_API_SIMULATIONS`, `MAX_ASIAN_MONITORING_POINTS`,
`MAX_ASIAN_TOTAL_PATH_NODES`, and `MAX_EXOTIC_CONVERGENCE_POINTS` bound HTTP requests. Limits are
checked before numerical arrays are allocated. Result schemas contain aggregates and bounded
diagnostics; observation, driver, terminal, and payoff arrays never cross the API boundary.

Input validation and unsupported method combinations return HTTP 422. Finite input that produces
unsafe exponentials, covariance calculations, or GBM variance also raises a quantitative domain
error translated to 422. Application settings do not change mathematical library results.

## 23. Why barriers remain Phase 11

The reusable monitoring schedule and arbitrary-time GBM engine are suitable foundations for
barrier monitoring, but a barrier contract also needs direction, level, knock-in/knock-out state,
rebate amount and timing, discrete-versus-continuous monitoring semantics, and possibly Brownian-
bridge corrections. Those are contract and estimator decisions, not Asian averaging variants.

Phase 10 therefore stops at reusable observations and vectorized path-payoff boundaries. Phase 11
can add barrier-specific contracts, stateful path classification, analytical comparisons where
available, and monitoring-bias policy without replacing the Phase 10 schedule, RNG, GBM, cluster,
or API resource-accounting foundations.
