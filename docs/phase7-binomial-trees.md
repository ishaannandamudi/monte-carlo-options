# Phase 7 Cox-Ross-Rubinstein binomial trees

## Purpose and scope

A binomial tree replaces continuous stock evolution with a recombining sequence of up and down
moves. It provides a transparent discrete-time expectation, native local price differences, and the
backward-induction foundation needed by later exercise-dependent products. Phase 7 implements only
European vanilla calls and puts under the Cox-Ross-Rubinstein (CRR) parameterization. There is no
early-exercise comparison anywhere in this phase.

The implementation has three boundaries:

- `models/binomial.py` defines configuration, parameter, price, comparison, and convergence models.
- `pricing/binomial.py` validates CRR inputs, creates the terminal vector, and performs backward
  induction in O(N) memory.
- `analytics/binomial.py` compares price/Greeks with Black-Scholes and evaluates convergence across
  step counts without assuming monotone error reduction.

The normal path never constructs or returns a full O(N²) lattice. No visualization endpoint or
full-tree mode is included.

## CRR time step and factors

For maturity `T` and `N` steps:

```text
dt = T / N
u  = exp(sigma sqrt(dt))
d  = 1 / u = exp(-sigma sqrt(dt))
a  = exp((r - q)dt)
p  = (a - d) / (u - d)
disc = exp(-r dt)
```

`r` is the continuously compounded risk-free rate and `q` is the continuous dividend yield.
Neither rate is constrained to be non-negative. A negative rate or `q > r` is valid when the
resulting CRR probability remains in `[0,1]`.

At time level `i`, node `j` denotes exactly `j` up moves and `i-j` down moves:

```text
S(i,j) = S0 u^j d^(i-j),   j = 0,...,i
```

Because `ud=1`, paths with the same up-move count recombine. At maturity the implementation forms
one NumPy vector from log prices:

```text
log S(N,j) = log S0 + (2j-N) log u
```

Log construction postpones overflow, while existing finite-number guards reject any terminal node
that is zero, infinite, or NaN in the supported floating-point environment.

## Risk-neutral probability validity

No-arbitrage for one CRR step requires `d <= a <= u`, which is equivalent to `0 <= p <= 1`. Coarse
trees, low volatility, or large carry can violate this condition. A materially invalid probability
raises `InvalidRiskNeutralProbabilityError` and suggests increasing steps or revising parameters.

`BinomialTreeConfig.probability_tolerance` defaults to `1e-12` and cannot exceed `1e-6`. A raw
probability in `[-tol,0)` or `(1,1+tol]` is treated as floating-point endpoint noise and normalized
to exactly zero or one. Larger overshoot is never clamped. The raw value must also be finite.

## Terminal payoff and backward induction

The engine calls the existing vectorized vanilla payoff functions:

```text
call payoff = max(S(N,j) - K, 0)
put payoff  = max(K - S(N,j), 0)
```

There is no discounting inside the payoff. For levels `i=N-1,...,0`, European continuation is:

```text
V(i,j) = disc [(1-p)V(i+1,j) + p V(i+1,j+1)]
```

Only the next-level vector is needed. Every reduction replaces a vector of length `i+2` with one of
length `i+1`; peak storage is therefore O(N), although the total arithmetic work is O(N²). The
continuation calculation is isolated in `_continuation_values`. Phase 8 can put an exercise-policy
comparison around this step without changing CRR parameters or terminal construction. Phase 7 does
not calculate exercise value or expose an American contract.

## Deterministic boundaries

### Zero maturity

At `T=0`, the engine returns intrinsic value immediately. It does not calculate `dt`, `u`, `d`, or
`p`. Parameter and native-Greek fields are unavailable, and the result reports
`ZERO_MATURITY`.

### Zero volatility

At `sigma=0`, standard CRR has `u=d=1` and the probability ratio is undefined. The engine instead
uses the exact risk-neutral terminal value:

```text
S_T   = S0 exp((r-q)T)
price = exp(-rT) payoff(S_T)
```

It does not divide by `u-d` or build a tree. This agrees with the existing Black-Scholes
deterministic limit up to floating-point evaluation order and reports `ZERO_VOLATILITY`. Tree-native
Delta, Gamma, and Theta are unavailable because the node spacings degenerate.

## Tree-native Greeks

Backward induction retains only levels one and two in addition to the root. This is constant extra
storage and does not change O(N) memory.

### Delta

At level one:

```text
Delta = [V(1,1) - V(1,0)] / [S0 u - S0 d]
```

Delta is available for `N>=1` when the node spacing is positive and finite.

### Gamma

At level two, with `Vdd=V(2,0)`, `Vud=V(2,1)`, and `Vuu=V(2,2)`:

```text
Delta_down = (Vud - Vdd) / (S0 ud - S0 d^2)
Delta_up   = (Vuu - Vud) / (S0 u^2 - S0 ud)
Gamma      = (Delta_up - Delta_down) / [(S0 u^2 - S0 d^2)/2]
```

The denominator uses the average distance between the outer level-two stock nodes. Gamma requires
`N>=2`; no value is fabricated for a one-step tree.

### Theta

The middle level-two node has stock `S0 ud = S0`, so it compares the same stock level after two time
steps with the root:

```text
Theta = [V(2,1) - V(0,0)] / (2dt)
```

This is change in option value as calendar time passes, matching the Phase 3 Theta convention
`-dV/dT`. Theta requires `N>=2`. Vega and rate sensitivities are not added; Phase 3 already provides
well-defined analytical and bump-based methods.

## Price bounds and analytical comparison

Every result reuses the Phase 4 European bounds:

```text
call: max(S e^-qT - K e^-rT, 0) <= C <= S e^-qT
put:  max(K e^-rT - S e^-qT, 0) <= P <= K e^-rT
```

A price materially outside the interval is a tree-domain failure. The default comparison tolerance
is `1e-10`. Results include the Black-Scholes benchmark, signed error `tree-BS`, absolute error, and
relative absolute error when the benchmark is nonzero. The comparison service additionally reports
Delta, Gamma, and Theta errors against Phase 3 analytical Greeks.

European call and put trees share the same stock lattice and probabilities. Consequently their
prices converge to Black-Scholes under the shared constant-volatility assumptions and satisfy
put-call parity within floating-point error:

```text
C - P = S e^-qT - K e^-rT
```

## Convergence and odd/even behavior

CRR convergence can oscillate because the strike falls differently relative to terminal nodes for
odd and even `N`. An error sequence may alternate around Black-Scholes even while its envelope
shrinks. The convergence service therefore reports every canonical price without smoothing and
does not assert that every larger step count is better.

Each point contains step parity, price errors, runtime, native Greeks, and Greek errors. When the
input includes neighboring `N` and `N+1`, an `OddEvenDiagnostic` reports their price difference and
adjacent average. The average is research context only; it never replaces either canonical price.
Richardson extrapolation is not implemented.

## Continuous dividends and negative rates

Continuous dividends enter only through risk-neutral growth `a=exp((r-q)dt)`. Per-step discounting
still uses `exp(-r dt)`. Negative `r`, negative `q`, and `q>r` are accepted if exponentials remain
finite and `p` is valid. The engine never assumes a discount factor below one.

## Numerical and resource safeguards

All exponentials use the existing guarded numerical helpers. Degenerate `dt`, indistinguishable
`u/d`, non-finite terminal nodes, non-finite backward values, non-finite Greeks, and invalid root
prices raise typed quantitative errors. The tree is deterministic and introduces no RNG.

Direct library calls are not capped by application settings. HTTP requests default to:

```text
MAX_BINOMIAL_STEPS=10000
MAX_BINOMIAL_CONVERGENCE_POINTS=20
```

The API checks every convergence count individually. Since runtime is O(N²), callers should treat
the 10,000-step ceiling as a safety limit rather than a routine resolution.

## API

- `POST /api/pricing/binomial` returns `BinomialPricingResult`.
- `POST /api/analytics/binomial/compare` returns price and Greek comparison diagnostics.
- `POST /api/analytics/binomial/convergence` returns ordered points and adjacent odd/even pairs.

Requests use the existing `OptionContract` and `MarketEnvironment` plus `BinomialTreeConfig`.
`OptionContract` currently admits only `EUROPEAN`; an `AMERICAN` request is rejected during API
validation and the pricer also checks exercise style defensively.

## Validation strategy

Tests verify CRR formulas directly, manually reproduce one- and two-step trees, check deterministic
boundaries, reject invalid probabilities, and cover valid endpoint probabilities. Calls and puts
are compared with Black-Scholes for dividends, negative rates, short/long maturities, low/high
volatility, and deep ITM/OTM states. Economic monotonicity, put-call parity, Phase 4 bounds, exact
repeatability, native Greek convergence, and explicit odd/even sign oscillation are tested. API
tests cover serialization, limits, unsupported models/exercise, and typed numerical errors.

## Deferred Phase 8 and future extensions

American early exercise, Bermudan schedules, exercise boundaries, and Longstaff-Schwartz remain
outside Phase 7. Phase 8 can compare continuation with intrinsic value after each continuation step
without replacing terminal payoff or CRR construction. Additional parameterizations such as
Jarrow-Rudd, Tian, or Leisen-Reimer may later implement the same typed tree interface, but Phase 7
contains only the fully tested CRR model.
