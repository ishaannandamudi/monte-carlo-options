# Phase 8: American Options and Early Exercise

## 1. American exercise rights

An American option may be exercised at valuation time or at any discrete CRR time level before
maturity. Phase 8 represents this right with the existing `ExerciseStyle` enum, now containing
`EUROPEAN` and `AMERICAN`. It does not introduce another contract type or exercise-style taxonomy.

## 2. Difference from European options

A European contract can exercise only at maturity. An American holder has every European exercise
opportunity plus earlier tree dates, so on the same lattice its value must satisfy

```text
V_American >= V_European.
```

The implementation enforces this same-tree invariant within the configured numerical tolerance.
It does not apply European put-call parity to American prices.

Every American value must also exceed immediate intrinsic. A call is checked against spot when the
continuous dividend yield is nonnegative; a put is checked against strike when the interest rate is
nonnegative. Those conditional upper checks are deliberately not extended to negative-carry cases
where the familiar bound may be invalid.

## 3. CRR backward induction

Phase 8 reuses Phase 7 terminal nodes, probabilities, discount factors, vectorized payoffs, and
O(N)-memory reduction. For `N` steps,

```text
dt   = T/N
u    = exp(sigma sqrt(dt))
d    = 1/u
p    = [exp((r-q)dt) - d] / (u-d)
disc = exp(-r dt).
```

Node `j` at level `i` has stock `S(i,j) = S0 u^j d^(i-j)`. Risk-neutral probability validation is
unchanged from Phase 7, including support for negative rates whenever `p` is valid.

## 4. Continuation value

At every nonterminal node the value of retaining the option is

```text
continuation(i,j) = disc [(1-p)V(i+1,j) + pV(i+1,j+1)].
```

The same isolated continuation function serves both exercise styles.

## 5. Exercise value

The immediate exercise value is the undiscounted intrinsic value at that node:

```text
call exercise value = max(S(i,j)-K, 0)
put exercise value  = max(K-S(i,j), 0).
```

Intrinsic value is evaluated with the existing vectorized payoff implementation.

## 6. Node decision

European nodes retain continuation. American nodes use

```text
V(i,j) = max(exercise(i,j), continuation(i,j)).
```

The maximum is exact: the exercise diagnostic tolerance never changes the price. Terminal values
equal intrinsic for both styles and maturity is not recorded as early exercise.

## 7. Diagnostic tolerance

`BinomialTreeConfig.exercise_tolerance` defaults to `1e-12`, must be nonnegative, and cannot exceed
`1e-4`. It is an absolute tolerance in option-value units. A node is *reported* as early exercise
only when

```text
exercise(i,j) > continuation(i,j) + exercise_tolerance.
```

This suppresses classifications caused by floating-point equality near a boundary. Changing the
tolerance can change node counts or boundary points, but cannot change the computed price, premium,
or tree Greeks.

## 8. Early-exercise premium

The reported premium is

```text
early_exercise_premium = American CRR price - European CRR price.
```

Both roots come from the same terminal vector, CRR parameters, market inputs, and step count. The
European vector is reduced alongside the American vector. Black-Scholes is reported separately and
is never used to define the premium.

## 9. American calls without dividends

Under the standard positive-rate, no-dividend assumptions, early exercise of a vanilla call is not
optimal. Tests require the American and European tree roots and native Greeks to agree and require
no pre-maturity exercise classifications. Negative rates can change familiar qualitative results,
so the theorem is not generalized beyond its assumptions.

## 10. Dividend-paying calls

Continuous dividend yield enters CRR growth through `exp((r-q)dt)`. A sufficiently high dividend
yield can make giving up future optionality worthwhile, so an American call can have a positive
premium and a high-stock exercise region. Phase 8 tests a strong-dividend case; it does not claim
that every positive dividend produces early exercise.

## 11. American puts

Positive-rate American puts commonly exercise at sufficiently low stock prices. Representative ATM
and deep-ITM tests require a positive premium and exercise boundary. Negative-rate cases are priced
from the discrete model without imposing positive-rate shortcuts or an unconditional strike upper
bound.

## 12. Root immediate exercise

Level zero is processed like every other nonterminal level. If intrinsic exceeds continuation, the
root price equals intrinsic and the boundary includes `step=0`, `time_from_valuation=0`. This is
early exercise because `0 < N`.

## 13. Maturity treatment

At level `N`, American and European values equal intrinsic payoff. Terminal nodes seed backward
induction but never enter `exercise_node_count` or boundary output. Every boundary point therefore
has `step < N` and strictly positive time to maturity.

## 14. Zero-volatility deterministic case

When `sigma=0`, `u=d=1` and the CRR probability is undefined. Phase 8 follows the deterministic path

```text
S(t_i) = S0 exp((r-q)t_i),  t_i = iT/N,
```

and performs scalar backward induction across all exercise dates. Equivalently, the root value is

```text
max_i exp(-r t_i) intrinsic(S(t_i)),  i=0,...,N.
```

The maturity payoff remains the European comparator. An exercising deterministic level contributes
one node and one boundary point; no multi-node region is fabricated. Degenerate tree Greeks remain
unavailable.

At `T=0`, both prices equal intrinsic, the premium is zero, and no early exercise is reported.

## 15. Boundary extraction

For every level with a diagnostic exercise mask, the induction pass stores one typed boundary
point: step, elapsed time, remaining time, boundary stock, option type, region direction, node count,
and contiguity flag. Points are reversed after backward induction so the public result runs from
valuation toward maturity. No interpolation between tree dates is invented.

## 16. Put boundary definition

Stock nodes are ordered from low to high. For a put, the boundary is the highest stock price still
classified for exercise. The expected exercise region is the low-stock prefix from node zero through
that boundary and is labeled `LOW_STOCK`.

## 17. Call boundary definition

For a call, the boundary is the lowest stock price classified for exercise. The expected region is
the high-stock suffix from that boundary through the highest node and is labeled `HIGH_STOCK`.

## 18. Memory behavior

The engine retains an American value vector, a same-tree European vector, current stock and
intrinsic vectors, and the small early levels needed for Greeks. Each is at most O(N). Boundary
metadata contains at most one record per nonterminal level, also O(N). It never retains an O(N²)
node lattice. Runtime remains O(N²), as every level must be reduced.

Contiguity is checked from each level's mask. Unexpected non-contiguous regions increment
`non_contiguous_level_count` and mark the point; this diagnostic does not alter a valid price.

## 19. American Delta

After exercise decisions have been applied at level one,

```text
Delta = [V(1,1)-V(1,0)] / [S0u-S0d].
```

It therefore measures the exercised American value surface, rather than a European proxy.

## 20. American Gamma

At level two, down and up Deltas are formed from the three post-exercise option values. Gamma is
their difference divided by the average outer-node stock spacing, using the unchanged Phase 7
formula.

## 21. American Theta

The level-two middle node has stock `S0ud=S0`. Calendar-time Theta uses

```text
Theta = [V(2,1)-V(0,0)] / (2dt),
```

with exercise decisions present in both values. American values can be nonsmooth near an exercise
boundary, so these discrete Greeks may converge less smoothly than European analytical Greeks.

## 22. Convergence and parity effects

American convergence points report American price, same-tree European price, premium, Delta,
Gamma, Theta, runtime, exercise-node count, and earliest exercise time. CRR strike placement and a
moving exercise boundary can both create odd/even oscillation. The service reports canonical
values at every requested step count and makes no monotonic-convergence claim. There is no claimed
closed-form American benchmark.

Tests include the commonly used one-year ATM put (`S=K=100`, `r=5%`, `sigma=20%`) near `6.09` and
the `S=K=50`, `r=10%`, `sigma=40%`, `T=5/12` put near `4.28`, with tolerances appropriate to a
2,000-step CRR tree. The classic `K=40`, `r=6%`, `sigma=20%`, one-year put set at spots `36`, `40`,
and `44` is also checked near `4.486`, `2.319`, and `1.113`.

## 23. API design

- `POST /api/pricing/binomial` accepts either exercise style. European responses retain the Phase 7
  `BinomialPricingResult`; American responses use `AmericanBinomialPricingResult`.
- `POST /api/analytics/binomial/compare` preserves the European response and returns an American,
  same-tree European, Black-Scholes-European, premium, and Greek-difference result for American
  input.
- `POST /api/analytics/binomial/exercise-boundary` requires an American contract and returns the
  ordered compact frontier.
- `POST /api/analytics/binomial/convergence` selects a style-specific response.

All endpoints reuse `MAX_BINOMIAL_STEPS` and `MAX_BINOMIAL_CONVERGENCE_POINTS`. Pricing responses do
not expose node arrays.

## 24. Why Black-Scholes remains European

The implemented Black-Scholes-Merton formula values a fixed maturity payoff and does not solve a
free-boundary problem. Treating an American request as European would silently discard exercise
rights. Black-Scholes pricing and analytical Greeks therefore reject `AMERICAN` contracts. The
American result includes a Black-Scholes value only after making an explicit European copy for
benchmark context.

Terminal Monte Carlo and Phase 3 Monte Carlo Greeks likewise reject American contracts because a
single terminal draw cannot represent an optimal stopping decision.

## 25. Why the Phase 5 IV convention is unchanged

Phase 5 market enrichment and Phase 6 surfaces continue using European Black-Scholes implied
volatility as their documented convention, including for provider metadata that identifies a U.S.
listed contract as American. Phase 8 adds an American price engine; it does not add American implied
volatility or change stored surface semantics.

## 26. Longstaff-Schwartz is deferred

Phase 8 contains no American Monte Carlo, regression basis, stopping-time path simulation, or
Longstaff-Schwartz implementation. A future Phase 9 can compare its stopping estimator with this
CRR benchmark while reusing existing contracts, markets, RNG ownership, payoff functions, and typed
analytics. Bermudan schedules, discrete dividends, exotics, and alternative lattice
parameterizations are also outside this phase.
