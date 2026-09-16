# Phase 11: Single-Barrier Options and Monitoring Bias

## 1. Contract architecture

Phase 11 adds `BarrierOptionContract`, a separate immutable Pydantic model. It reuses
`OptionType`, `ExerciseStyle.EUROPEAN`, `MarketEnvironment`, Phase 10's `MonitoringSchedule`, and
the exact arbitrary-time GBM process. It does not add optional barrier fields to vanilla, Asian,
or digital contracts.

The contract specifies one `UP` or `DOWN` barrier, `IN` or `OUT` semantics, positive spot, strike,
barrier and nonnegative maturity, a discrete schedule, `DISCRETE` or `BROWNIAN_BRIDGE` monitoring,
a nonnegative rebate, and `AT_HIT` or `AT_EXPIRY` timing. Double barriers, American exercise, and
continuous-model variants outside constant-volatility GBM are absent.

## 2. Hit convention

Hits are inclusive and consistent in every deterministic and stochastic path:

```text
DOWN hit iff S_t <= H
UP   hit iff S_t >= H.
```

Equality is a hit. Fixed-path helpers, time-zero checks, discrete detection, bridge endpoint logic,
and continuous analytical handling all use this convention.

## 3. Initial barrier check

Time zero is always monitored even when the schedule contains only future times. A down contract
starts hit when `S_0 <= H`; an up contract starts hit when `S_0 >= H`. The initial point does not
consume the configured future monitoring count.

An initially hit knock-in is already active. An initially hit knock-out cannot earn its vanilla
payoff and receives only its configured rebate. An `AT_HIT` knock-out rebate at time zero is not
discounted; an `AT_EXPIRY` rebate is discounted from maturity.

## 4. Knock-in semantics

A knock-in pays the maturity vanilla payoff if the barrier has hit. If it never hits, it pays its
rebate at expiry. Zero rebate therefore produces

```text
exp(-rT) vanilla_payoff * 1[hit].
```

Phase 11 supports knock-in rebates only as no-hit payments at expiry. `AT_HIT` is rejected because
there is no hit-triggered knock-in rebate in the chosen contract semantics.

## 5. Knock-out semantics

A knock-out pays the maturity vanilla payoff only if the barrier is never hit. On hit it pays its
rebate according to the supported timing rule:

```text
no hit: exp(-rT) vanilla_payoff
hit, AT_EXPIRY: rebate exp(-rT)
hit, AT_HIT:    rebate exp(-r tau_hit).
```

There is exactly one discounting step. Negative rates are supported, so a hit-time discount factor
may exceed one.

## 6. Rebate semantics and restrictions

Discrete monitoring records an observed first-hit time, so knock-out rebates may be `AT_HIT` or
`AT_EXPIRY`. Knock-in no-hit rebates are `AT_EXPIRY` only.

Brownian-bridge survival weighting identifies a conditional probability of crossing within each
interval but does not identify the conditional first-hit time required for `exp(-r tau_hit)`.
Positive bridge `AT_HIT` rebates are therefore rejected. Bridge `AT_EXPIRY` rebates are exact under
the weighting estimator. Zero rebates may carry either timing label because the payment is zero.

The continuous analytical formulas support zero rebate only. Unsupported combinations fail with a
domain error instead of receiving an invented interpretation.

## 7. Discrete monitoring

Regular schedules generate `iT/n`, `i=1,...,n`. Explicit schedules accept unique, sorted year
fractions in `[0,T]`. Discrete detection examines time zero plus only the scheduled future points.
If an explicit schedule omits maturity, `S_T` is still simulated for the terminal payoff but is not
silently treated as a barrier observation.

For a fixed observation matrix, vectorized comparisons create a hit matrix. `any` gives path hit
status; the first true position gives the first monitored hit index and time. Internally, `-1` and
`NaN` represent no hit. Those sentinels never appear in public result fields.

## 8. Monitoring bias

Discrete monitoring cannot see an excursion through the barrier followed by a return between two
observations. It therefore underestimates hit probability relative to continuous monitoring. For a
zero-rebate knock-out this generally overstates value; for a knock-in it generally understates
value. Rebate structures can change price ordering, so only the hit-probability ordering is treated
as general.

Monitoring-count analytics report discrete and bridge prices side by side. A finer discrete grid
approaches continuous behavior in representative GBM cases, but sampled errors need not be
monotone at every count.

## 9. Log-price Brownian bridge

Let `X_t=log(S_t)` and `h=log(H)`. Conditional on two GBM log-price endpoints, the drift cancels
from the bridge crossing law. For a down barrier with both endpoints above `h`,

```text
p_i = exp[-2 (X_i-h)(X_{i+1}-h) / (sigma^2 delta_i)].
```

For an up barrier with both endpoints below `h`,

```text
p_i = exp[-2 (h-X_i)(h-X_{i+1}) / (sigma^2 delta_i)].
```

If either endpoint satisfies the inclusive hit condition, `p_i=1`. Actual irregular
`delta_i=t_{i+1}-t_i` values are used.

## 10. Crossing-probability derivation

After conditioning a Brownian motion on its two endpoints, subtracting the linear interpolation
produces a zero-endpoint Brownian bridge. Applying the reflection principle to the bridge gives the
exponential formulas above. GBM becomes arithmetic Brownian motion in log space, so the same result
applies to `log(S)` with conditional variance `sigma^2 delta_i`. Constant drift affects endpoint
distribution but not the conditional crossing probability.

This result is exact for constant-volatility, continuous-path GBM. It is not carried over to jumps,
Heston, or local volatility.

## 11. Survival-probability weighting

Given simulated endpoints, interval survival is `1-p_i`. Conditional independence across disjoint
bridge intervals gives path survival weight

```text
w = product_i (1-p_i).
```

The implementation accumulates log survival one interval at a time and retains only O(paths)
bridge workspace beyond the state matrix. Endpoint hits force `w=0`. Exponential underflow becomes
zero survival, while material probabilities outside `[0,1]` raise an error.

For a zero-rebate knock-out,

```text
Y_out = exp(-rT) vanilla_payoff * w.
```

This conditional-expectation estimator avoids an extra uniform draw and its Bernoulli variance.

## 12. Knock-in bridge weighting

The complementary conditional hit probability is `1-w`. A zero-rebate knock-in uses

```text
Y_in = exp(-rT) vanilla_payoff * (1-w).
```

For expiry rebates,

```text
Y_out = exp(-rT) [w vanilla_payoff + (1-w) rebate]
Y_in  = exp(-rT) [(1-w) vanilla_payoff + w rebate].
```

With zero rebate, `Y_in + Y_out` equals the discounted vanilla payoff path by path, up to floating-
point arithmetic.

## 13. Bridge rebate limitation

Survival weighting gives `P(hit | endpoints)`, not
`E[exp(-r tau_hit) 1[hit] | endpoints]`. Using an observation time, midpoint, or expiry as a proxy
would bias an `AT_HIT` rebate. Phase 11 rejects that case. A future extension may add a validated
first-passage-time integral or a separate stochastic hit-time estimator with an independent RNG
stream.

## 14. First-hit handling

Discrete paths expose hit probability, survival probability, mean hit time, and median hit time.
No-hit paths are excluded from time statistics. An initial hit contributes time zero.

Bridge weighting reports expected hit probability

```text
1 - mean(w)
```

and expected survival probability `mean(w)`. It returns hit-time statistics as unavailable because
survival weighting does not sample a first-hit time.

## 15. Zero-volatility path

For `sigma=0`,

```text
S_t = S_0 exp((r-q)t)
```

is deterministic and monotone. Discrete mode evaluates only its scheduled points. Bridge mode
checks continuous crossing exactly; when a new crossing occurs and `r-q != 0`,

```text
tau_hit = log(H/S_0)/(r-q).
```

No RNG draws or Brownian formula are used. Prices and hit diagnostics have zero estimator
uncertainty.

## 16. Irregular monitoring

The simulator uses the Phase 10 exact transition for every requested `delta_i`. The state-time set
is the union of scheduled future times and maturity. Bridge mode uses every adjacent state interval,
including a final interval to maturity when an explicit schedule omits `T`. This represents
continuous monitoring through expiry; discrete mode keeps that terminal state payoff-only unless it
was scheduled.

## 17. Continuous analytical formulas

Phase 11 implements all eight zero-rebate, continuously monitored single-barrier prices under
Black-Scholes-Merton. Define `b=r-q`,

```text
mu = (b - 0.5 sigma^2)/sigma^2
x1 = log(S/K)/(sigma sqrt(T)) + (1+mu)sigma sqrt(T)
x2 = log(S/H)/(sigma sqrt(T)) + (1+mu)sigma sqrt(T)
y1 = log(H^2/(SK))/(sigma sqrt(T)) + (1+mu)sigma sqrt(T)
y2 = log(H/S)/(sigma sqrt(T)) + (1+mu)sigma sqrt(T).
```

With `phi=+1` for calls and `-1` for puts, and `eta=+1` for down and `-1` for up, the standard
Reiner-Rubinstein building blocks are

```text
A = phi S e^-qT N(phi x1) - phi K e^-rT N(phi[x1-sigma sqrt(T)])
B = phi S e^-qT N(phi x2) - phi K e^-rT N(phi[x2-sigma sqrt(T)])
C = phi S e^-qT (H/S)^(2[mu+1]) N(eta y1)
  - phi K e^-rT (H/S)^(2mu) N(eta[y1-sigma sqrt(T)])
D = phi S e^-qT (H/S)^(2[mu+1]) N(eta y2)
  - phi K e^-rT (H/S)^(2mu) N(eta[y2-sigma sqrt(T)]).
```

The knock-in selection table is:

| Contract | `K >= H` | `K < H` |
|---|---:|---:|
| Down-in call | `C` | `A-B+D` |
| Up-in call | `A` | `B-C+D` |
| Down-in put | `B-C+D` | `A` |
| Up-in put | `A-B+D` | `C` |

Knock-out value is vanilla minus knock-in. Reflected normal terms are evaluated with `log_ndtr` so
far barriers do not form an overflowing power before multiplication by a vanishing normal CDF.
Initial hits, zero maturity, and zero volatility are handled before these formulas.

## 18. Knock-in/out parity

For equal zero-rebate contracts,

```text
V_in + V_out = V_vanilla.
```

Analytical knock-outs are constructed from this identity after validating the knock-in value.
Monte Carlo parity analytics generate the IN and OUT values from exactly reproduced PCG64 path
matrices. Discrete indicator complements and bridge weights make the sample residual machine zero,
apart from final summation rounding.

## 19. Hit-probability analytics

Every pricing result reports hit, no-hit, and activation probabilities. Activation means hit for an
IN option and survival for an OUT option. Discrete values are empirical frequencies. Bridge values
are conditional-probability estimates. Probabilities are checked and reported within `[0,1]` and
sum to one.

Level-sensitivity analytics use common seed-defined streams across barrier levels to reduce visual
noise. They report both prices, SEs, hit probabilities, and bridge-minus-discrete differences.

## 20. Convergence

`PATH_COUNT` varies simulations while holding the schedule fixed. `MONITORING_COUNT` varies a
regular future schedule while holding the path budget fixed. Each point receives an independent,
deterministic `SeedSequence(root_seed)` child seed. Every point reports discrete and bridge values,
SEs, hit probabilities, runtime, and the continuous analytical reference for zero-rebate contracts.

The analytical error applies to bridge continuous monitoring. A discrete price is not labeled an
error against the continuous contract because it represents a different monitoring rule.

## 21. Antithetic statistics

Every normal increment vector `Z` is paired with `-Z` across the full path. Requested path counts
remain exact; an odd final path is an independent singleton. Discounted realized or weighted
payoffs are clustered using the existing pair means and weighted singleton estimator. Correlated
pair members are never treated as independent.

Antithetic sampling can help but is not asserted to reduce variance for every discontinuous barrier
configuration.

## 22. Memory and performance

Only scheduled states plus a payoff-only maturity state when needed are simulated. Retained state
memory is approximately

```text
8 * paths * (future_states + 1) bytes
```

for float64 values including time zero. This is O(paths times observations). Bridge survival loops
over intervals and uses O(paths) extra workspace rather than retaining a second crossing-probability
matrix. It does not create a hidden fine grid.

Measured timings and the large-case policy are recorded in
[`benchmarks/phase11-barriers.md`](../benchmarks/phase11-barriers.md).

## 23. API safeguards

Endpoints are:

- `POST /api/pricing/barrier`
- `POST /api/pricing/barrier/analytical`
- `POST /api/analytics/barrier/compare`
- `POST /api/analytics/barrier/convergence`
- `POST /api/analytics/barrier/parity`
- `POST /api/analytics/barrier/sensitivity`

HTTP limits cap simulations, future monitoring points, total path nodes across every represented
run, and convergence/sensitivity points. Oversized regular counts are rejected before schedule or
path expansion. Comparison accounting includes its parity runs. The library itself is not governed
by HTTP settings. No path, crossing, or payoff arrays are serialized.

## 24. Deferred barrier Greeks

Barrier events are discontinuous in both spot and path. Naive pathwise differentiation is invalid,
and finite differences need careful Common Random Numbers, barrier shifts, and monitoring-bias
interpretation. Phase 11 adds no barrier Greeks. A later phase may introduce a separately tested
finite-difference estimator without changing barrier payoff semantics.

## 25. Phase 12 boundary

Phase 11 stops at one underlying and one barrier under constant-volatility GBM. Phase 12 may add
multi-asset simulation, correlation matrices, basket payoffs, and related controls. It must not
reinterpret the single-asset barrier contract or apply the GBM bridge formula unchanged to
correlated stochastic-volatility, local-volatility, or jump paths.
