# Phase 4 implied-volatility design

## 1. Scope and definition

For a European call or put with contract inputs \((S,K,T)\), known continuously compounded
risk-free rate \(r\), continuous dividend yield \(q\), and observed price \(P^*\), implied
volatility is a non-negative solution of

\[
V_{BS}(S,K,T,r,q,\sigma)=P^*.
\]

Phase 4 inverts only the existing European Black-Scholes-Merton price. It does not use Monte Carlo,
infer rates, ingest market data, fit a surface, or calibrate another model. `ImpliedVolatilityMarket`
contains only \(r\) and \(q\); it deliberately avoids inserting an arbitrary volatility into the
Phase 1 `MarketEnvironment` while volatility is unknown. An `ImpliedVolatilityObservation` combines
that rates-only market, an existing `OptionContract`, and the observed price.

The implementation is split as follows:

- `models/volatility.py` owns solver enums, statuses, known-market inputs, observations, and config.
- `pricing/implied_volatility.py` owns bounds, forward price, objectives, solvers, and diagnostics.
- `analytics/volatility.py` owns ordered batches, smile slices, and solver comparisons.
- `api/volatility.py` owns HTTP request shapes and resource limits only.
- `models/results.py` owns all serializable Phase 4 result contracts.

The existing `black_scholes_price` and `black_scholes_greeks` implementations remain the only
pricing and analytical-Vega definitions.

## 2. Why the inversion is numerical

The Black-Scholes price contains normal cumulative distribution functions of

\[
d_1=\frac{\log(S/K)+(r-q+\tfrac12\sigma^2)T}{\sigma\sqrt T},
\qquad d_2=d_1-\sigma\sqrt T.
\]

Volatility appears in both the denominator and numerator and then inside normal CDFs. There is no
general elementary closed-form inverse for \(\sigma\). Phase 4 therefore solves the scalar root

\[
f(\sigma)=V_{BS}(\sigma)-P^*=0
\]

inside an explicit finite interval. Calls and puts use the same objective and solver machinery.

## 3. European no-arbitrage price bounds

Define discounted spot and strike

\[
A=Se^{-qT}, \qquad B=Ke^{-rT}.
\]

For a call,

\[
\max(A-B,0) \le C \le A.
\]

For a put,

\[
\max(B-A,0) \le P \le B.
\]

`call_price_bounds`, `put_price_bounds`, and `european_option_price_bounds` expose these formulas
for reuse by later option-chain validation. Negative rates and negative yields are allowed. Finite
exponential guards are reused from the pricing core. A target below the lower bound or above the
upper bound by more than `price_tolerance` raises `ArbitrageBoundsError` before a root solver runs.

## 4. Monotonicity and the root bracket

For positive maturity and volatility, Black-Scholes call and put Vega are identical:

\[
\mathcal V=\frac{\partial V}{\partial\sigma}
=Se^{-qT}\phi(d_1)\sqrt T \ge 0.
\]

The price is therefore nondecreasing in volatility. This property makes a sign-changing volatility
bracket valid for Brent and bisection. Phase 4 evaluates the configured `min_volatility` and
`max_volatility` once and requires their residuals to contain a root, allowing a residual within
`price_tolerance` at either endpoint. The default interval is `[0, 5]`. It is never expanded
silently, and failure to bracket raises `ImpliedVolatilityBracketError`.

## 5. Solver configuration

`ImpliedVolatilityConfig` defaults are:

| Field | Default | Meaning |
|---|---:|---|
| `method` | `AUTO` | solver/fallback policy |
| `initial_volatility` | `0.20` | Newton starting value |
| `min_volatility` | `0.0` | finite lower bracket |
| `max_volatility` | `5.0` | finite upper bracket |
| `price_tolerance` | `1e-10` | maximum accepted absolute price residual |
| `volatility_tolerance` | `1e-10` | bracket/root location tolerance |
| `vega_tolerance` | `1e-12` | unsafe Newton denominator threshold |
| `max_iterations` | `100` | per-solver iteration protection |

All tolerances and iteration counts must be positive. Bounds must be ordered, non-negative, and
finite; the Newton guess must lie inside them. These are direct library inputs rather than hidden
application settings.

## 6. Brent method

`BRENT` uses SciPy's `brentq` on the already validated bracket. Brent combines bisection's bracket
safety with interpolation steps. Relative tolerance is held at SciPy's floating-point minimum.
The absolute volatility tolerance is the smaller of the configured volatility tolerance and a
price-driven value obtained from the analytical upper bound

\[
\mathcal V \le \frac{Se^{-qT}\sqrt T}{\sqrt{2\pi}}.
\]

This keeps a root-location termination from producing a price residual looser than the requested
price tolerance in ordinary finite cases. The common post-solve reconstruction check remains the
authoritative acceptance rule.

## 7. Bisection

`BISECTION` is implemented explicitly. Every midpoint retains the sign-changing bracket. It returns
only when both conditions hold:

\[
|f(\sigma_m)|\le\epsilon_P,
\qquad \frac{\sigma_U-\sigma_L}{2}\le\epsilon_\sigma.
\]

It fails after `max_iterations` or if floating-point resolution prevents a new midpoint. This makes
bisection deterministic and provides the final robust AUTO fallback.

## 8. Newton-Raphson and Vega

`NEWTON` uses the Phase 3 raw-unit analytical Vega and updates

\[
\sigma_{n+1}=\sigma_n-\frac{V_{BS}(\sigma_n)-P^*}{\mathcal V(\sigma_n)}.
\]

It stops successfully only on the price residual. It fails cleanly when Vega is non-finite or at or
below `vega_tolerance`, a candidate is non-finite, a step leaves the configured volatility range,
the iterate stagnates, or the iteration limit is reached. It never clips an unsafe step because
clipping could conceal divergence or report a bracket endpoint as a Newton solution.

Vega is also a conditioning diagnostic. Where Vega is meaningfully positive,

\[
\frac{d\sigma}{dV}\approx\frac{1}{\mathcal V}.
\]

The result reports this value as `price_to_vol_sensitivity`. It is `None` when Vega is too small,
preventing infinity from entering a response. Deep moneyness and short maturity often have small
Vega, so their IVs can be numerically and economically unstable even when the price fit is tight.

## 9. AUTO fallback policy

After target and bracket validation, `AUTO` performs:

1. Newton from `initial_volatility`.
2. Brent on the unchanged validated bracket if Newton is unsafe or does not converge.
3. Explicit bisection if Brent does not converge.

`method_requested`, `method_used`, and `fallback_used` make this behavior visible. Function
evaluations count all Black-Scholes objective calls across failed attempts, endpoint validation,
and final reconstruction. An endpoint match bypasses iterative solvers and reports
`method_used=None`.

## 10. Reconstruction and convergence diagnostics

Every candidate is repriced by the existing analytical engine. A result is marked converged only
when

\[
|V_{BS}(\widehat\sigma)-P^*|\le\texttt{price_tolerance}.
\]

Successful results include IV, requested and used method, iterations, pricing evaluations, final
and target prices, absolute residual, price bounds, volatility bounds, initial guess, fallback flag,
runtime, final Vega, inverse-Vega conditioning, status, and message. Pydantic rejects non-finite
result values. Runtime naturally varies and is not part of quantitative reproducibility.

## 11. Exact expiry

At \(T=0\), the feasible option price collapses to intrinsic value, and the price carries no
information about volatility. If the observed price differs from intrinsic value
by more than `price_tolerance`, normal bound validation rejects it. If it matches, the library
returns a valid `NON_IDENTIFIABLE` result with `implied_volatility=None`, `converged=False`, and no
method used. It does not fabricate zero volatility.

## 12. Boundary prices

At positive maturity, the lower price bound is the deterministic \(\sigma=0\) price. If the target
matches the configured zero-volatility endpoint within price tolerance, Phase 4 returns zero with
an explicit endpoint diagnostic. If the configured minimum is above zero and the target lies below
that endpoint, the finite bracket fails.

An exact upper-bound call or put price is generally approached only as
\(\sigma\rightarrow\infty\). A target within `price_tolerance` of that exact mathematical upper
bound is treated as having no finite identifiable root and raises `ImpliedVolatilityBracketError`.
A price below the upper bound can still fail if its finite IV is above `max_volatility`; the library
does not invent a value or expand the caller's range.

## 13. Failure taxonomy

- Pydantic validation covers malformed, non-finite, negative-price, and invalid-config requests.
- `ArbitrageBoundsError` identifies a price outside mathematical European bounds.
- `ImpliedVolatilityBracketError` identifies no finite root in the configured range.
- `ImpliedVolatilityConvergenceError` identifies solver safety or convergence failure.
- `NumericalError` identifies unsafe floating-point representation.

The single-solve API maps quantitative failures to HTTP 422. Batch and smile services convert each
quantitative failure to `ARBITRAGE_VIOLATION`, `NO_FINITE_ROOT`, `NON_CONVERGENCE`, or
`NUMERICAL_FAILURE` without hiding other valid items.

## 14. Batch policy

`solve_implied_volatility_batch` accepts heterogeneous contracts and rates, loops over observations,
and preserves exact input order. Each item has its original zero-based index, a success flag, an
optional full result, status, and message. A mathematical failure in one observation does not fail
the batch. Structurally malformed HTTP input is rejected by request validation before analytics.
The library has no hidden item limit; the API defaults to `MAX_IV_BATCH_SIZE=500`.

Exact-expiry non-identifiability is retained as a full item result with `success=false`, since no IV
is available, rather than being mislabeled as an arbitrage or numerical error.

## 15. Synthetic smile architecture

The smile service accepts one spot, maturity, rate, yield, option type, and parallel strike/price
vectors. It constructs one ordinary observation per strike and uses the batch policy, so invalid
prices are visible alongside valid points. Caller order is preserved. The API validates equal,
non-empty vector lengths and defaults to `MAX_SMILE_POINTS=200`.

The output is a single synthetic slice. It contains no interpolation between strikes, relationship
between maturities, smoothing, static-arbitrage repair, SVI, splines, or local volatility.

## 16. Moneyness and forward definitions

The reusable forward helper uses continuous carry:

\[
F=S e^{(r-q)T}.
\]

Every smile point reports simple strike moneyness

\[
m=\frac KS
\]

and forward log-moneyness

\[
k=\log\left(\frac KF\right).
\]

Thus \(k=0\) is at-the-money-forward. These definitions are explicit so later surface work can use
the same coordinates without reinterpretation.

## 17. Solver comparison and validation

`compare_implied_volatility_solvers` runs `BRENT`, `BISECTION`, and `NEWTON` independently with all
other config fields fixed. It returns per-method success or failure and the full diagnostics where
available. It does not require identical iteration counts or bitwise roots.

Tests validate direct call/put bounds, negative rates, dividends, expiry, upper/lower boundaries,
configuration, high finite volatility, price monotonicity, broad call/put price-to-volatility round
trips, each solver, iteration protection, low-Vega Newton fallback, batch order and partial failure,
synthetic smile recovery, forward coordinates, serialization, HTTP validation, and resource limits.
Existing Phase 1–3 tests remain unchanged.

## 18. Deferred functionality

Phase 4 creates reusable inversion and coordinate primitives that the Phase 5 normalized option-
chain layer now consumes. It does not add quote cleaning, cross-strike or cross-maturity arbitrage
checks, interpolation, extrapolation, SVI, local volatility, American-option pricing, or stochastic-
volatility calibration. Later surface and model phases can call these typed bounds, forward,
solver, batch, and diagnostic interfaces without replacing the Phase 1–3 pricing foundation.
