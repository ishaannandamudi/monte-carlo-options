# Phase 3 Greeks and sensitivity design

## 1. Scope

Phase 3 makes first-order and selected second-order sensitivities explicit quantitative results. It
adds analytical Black-Scholes-Merton Greeks, Common-Random-Number finite differences, pathwise
Delta and Vega, analytical-versus-stochastic comparisons, and one-dimensional analytical
sensitivity grids. It preserves the Phase 1 contract, market, payoff, process, RNG, price, and
confidence-interval behavior and the Phase 2 estimator-cluster and control-variate rules.

The canonical Greek names are `DELTA`, `GAMMA`, `VEGA`, `THETA`, `RHO`, and `DIVIDEND_RHO`.
Estimator methods are `ANALYTICAL`, `FINITE_DIFFERENCE`, and `PATHWISE`. Support is deliberately
method-specific:

| Greek | Analytical | Finite difference | Pathwise |
|---|---:|---:|---:|
| Delta | yes | yes | yes |
| Gamma | yes | yes | no |
| Vega | yes | yes | yes |
| Theta | yes | yes | no |
| Rho | yes | yes | no |
| Dividend rho | yes | yes | no |

Finite differences support all four pricing variance modes. Pathwise estimators support `NONE` and
`ANTITHETIC`; unsupported combinations raise a typed quantitative error.

## 2. Definitions and raw-unit conventions

For option value \(V(S,\sigma,r,q,T)\), where \(T\) is remaining maturity:

\[
\Delta=\frac{\partial V}{\partial S}, \qquad
\Gamma=\frac{\partial^2 V}{\partial S^2}, \qquad
\mathcal V=\frac{\partial V}{\partial\sigma},
\]

\[
\rho=\frac{\partial V}{\partial r}, \qquad
\rho_q=\frac{\partial V}{\partial q}.
\]

Theta follows the calendar-time market convention. As one calendar year passes, remaining
maturity falls by one year, so

\[
\Theta=-\frac{\partial V}{\partial T}.
\]

Every value uses raw model units:

- Delta is price change per one spot unit.
- Gamma is Delta change per one spot unit.
- Vega is price change per absolute volatility unit `1.00`, rather than per percentage point.
- Rho and dividend rho are changes per absolute rate unit `1.00`, rather than per basis point.
- Theta is price change per calendar year.

Display scaling belongs to a caller or future presentation layer.

## 3. Analytical Black-Scholes-Merton formulas

For positive \(T\) and \(\sigma\), define

\[
d_1=\frac{\log(S/K)+(r-q+\tfrac12\sigma^2)T}{\sigma\sqrt T},
\qquad
d_2=d_1-\sigma\sqrt T.
\]

Let \(N\) be the standard-normal CDF and \(\phi\) its density. Call and put Delta are

\[
\Delta_C=e^{-qT}N(d_1),
\qquad
\Delta_P=e^{-qT}\bigl(N(d_1)-1\bigr).
\]

Gamma and Vega are common to calls and puts:

\[
\Gamma=\frac{e^{-qT}\phi(d_1)}{S\sigma\sqrt T},
\qquad
\mathcal V=Se^{-qT}\phi(d_1)\sqrt T.
\]

Rate rhos are

\[
\rho_C=KTe^{-rT}N(d_2),
\qquad
\rho_P=-KTe^{-rT}N(-d_2).
\]

Dividend rhos are

\[
\rho_{q,C}=-STe^{-qT}N(d_1),
\qquad
\rho_{q,P}=STe^{-qT}N(-d_1).
\]

With calendar-time sign, Theta is

\[
\Theta_C=-\frac{Se^{-qT}\phi(d_1)\sigma}{2\sqrt T}
          +qSe^{-qT}N(d_1)-rKe^{-rT}N(d_2),
\]

\[
\Theta_P=-\frac{Se^{-qT}\phi(d_1)\sigma}{2\sqrt T}
          -qSe^{-qT}N(-d_1)+rKe^{-rT}N(-d_2).
\]

The analytical implementation reuses the pricing implementation's finite \(d_1,d_2\) and
discounted-present-value helpers, preventing formula drift between price and Greek calculations.

## 4. Analytical boundary conventions

Some exact boundaries contain payoff kinks or distributional derivatives. Phase 3 returns explicit
finite conventions rather than NaN or infinity.

### Exact expiry, \(T=0\)

- Call Delta is 1 above strike, 0 below strike, and 0.5 exactly at strike.
- Put Delta is 0 above strike, -1 below strike, and -0.5 exactly at strike.
- Gamma is reported as 0. At the strike the mathematical object is distributional; 0 is the API's
  finite point convention, not a claim that the limiting Gamma is zero.
- Vega, Theta, Rho, and dividend rho are 0.
- `d1` and `d2` are unavailable (`None`).

The analytical at-strike Delta convention is the symmetric derivative of the payoff kink.
Pathwise estimators use the strict Phase 1 ITM rules and therefore contribute zero at exact
equality. This distinction is intentional and documented in their result interpretation.

### Exact zero volatility, \(\sigma=0\), with \(T>0\)

Let

\[
A=Se^{-qT}, \qquad B=Ke^{-rT}.
\]

The deterministic call is \((A-B)^+\), and the deterministic put is \((B-A)^+\). Strictly inside
the deterministic exercise region, Greeks are obtained by differentiating that active branch;
outside it they are zero. At \(A=B\), branch derivatives use half weight, Gamma is 0, and Vega uses
the right-hand volatility limit

\[
\mathcal V_{0+}=A\sqrt T\,\phi(0).
\]

Away from the forward kink, the zero-volatility Vega convention is 0. A positive volatility-time
product that underflows numerically uses the same deterministic policy.

Very small positive \(T\) or \(\sigma\) still uses the closed form when finite. Deep moneyness,
negative rates, and positive or negative continuous dividend yields are supported.

## 5. Finite-difference formulas and bump policy

`GreekBumpConfig` has these defaults:

| Field | Default | Unit |
|---|---:|---|
| `spot_relative_bump` | `1e-3` | fraction of spot |
| `volatility_absolute_bump` | `1e-4` | absolute volatility |
| `rate_absolute_bump` | `1e-4` | absolute rate |
| `dividend_yield_absolute_bump` | `1e-4` | absolute yield |
| `maturity_absolute_bump` | `1/3650` | years, about 0.1 day |

The spot increment is \(h_S=S\times\text{spot_relative_bump}\), so it scales with the asset. The
relative bump is constrained to \((0,1)\), keeping both spot states positive.

Where both states are valid, central formulas are

\[
\Delta\approx\frac{V(S+h_S)-V(S-h_S)}{2h_S},
\]

\[
\Gamma\approx\frac{V(S+h_S)-2V(S)+V(S-h_S)}{h_S^2},
\]

\[
\mathcal V\approx\frac{V(\sigma+h_\sigma)-V(\sigma-h_\sigma)}{2h_\sigma},
\]

\[
\rho\approx\frac{V(r+h_r)-V(r-h_r)}{2h_r},
\qquad
\rho_q\approx\frac{V(q+h_q)-V(q-h_q)}{2h_q},
\]

and, applying the calendar-time sign,

\[
\Theta\approx-\frac{V(T+h_T)-V(T-h_T)}{2h_T}.
\]

If \(\sigma-h_\sigma<0\), Vega uses \([V(\sigma+h)-V(\sigma)]/h\). If \(T-h_T<0\), Theta uses
\(-[V(T+h)-V(T)]/h\). The result reports `FORWARD` and leaves `downward_price` unavailable. These
fallbacks never construct an invalid market or contract.

Larger bumps increase truncation error. Very small bumps amplify floating-point cancellation and,
without driver coupling, simulation noise. Defaults are conservative general-purpose values;
research experiments should inspect bump stability rather than assume that a smaller bump is
always better.

## 6. Common Random Numbers

Finite differences generate one explicit standard-normal driver vector from the configured
`SeedSequence`/`PCG64` generator. Every base, upward, and downward valuation for the request reuses
that same immutable vector. Antithetic modes generate the vector once using the existing interleaved
`Z,-Z` policy, including the exact odd singleton.

CRN makes path \(i\) in every bumped state respond to the same random shock. The price difference
therefore reflects the parameter change rather than two unrelated Monte Carlo errors. Seed reuse is
an explicit driver-level operation in `monte_carlo_greeks.py`; it does not depend on accidentally
constructing multiple generators with the same seed.

The existing price engine continues to construct and own its generator normally. The new
`terminal_prices_from_normals` GBM function separates deterministic terminal evaluation from driver
generation so sensitivity engines can own and reuse drivers without changing Phase 1/2 behavior.

## 7. Direct finite-difference observations and uncertainty

The estimator is constructed before aggregation. For example, Delta path observations are

\[
D_i=\frac{Y_i(S+h_S)-Y_i(S-h_S)}{2h_S},
\]

and Gamma observations are

\[
G_i=\frac{Y_i(S+h_S)-2Y_i(S)+Y_i(S-h_S)}{h_S^2}.
\]

Vega, Theta, Rho, and dividend-rho observations use the analogous linear combinations, including
the Theta sign. The reported Greek is the weighted mean of these observations. Its estimator
variance is calculated directly from them, and standard error is the square root of that variance.
The implementation never propagates two bumped-price errors as though bumped prices were
independent.

For `CONTROL_VARIATE`, each bumped state constructs

\[
X=e^{-rT}S_T, \qquad E[X]=S_0e^{-qT},
\]

estimates its state-specific beta, and produces adjusted independent observations before the
finite-difference combination. The expectation is recomputed when spot, maturity, rate, dividend
yield, or another state component changes; an original-state expectation is never incorrectly
reused. `ANTITHETIC_CONTROL_VARIATE` performs those transformations at pair-cluster level.

The confidence interval is the normal approximation

\[
\widehat g\mathbin{\pm}z_{(1+c)/2}\widehat{\operatorname{SE}}(\widehat g).
\]

## 8. Antithetic clusters and odd counts

For an even antithetic sample, path-level Greek contributions are averaged inside each pair and SE
uses the independent pair averages. For odd \(N=2M+1\), the final driver is a singleton. Pair
clusters retain weight 2, the singleton retains weight 1, and the estimate is

\[
\widehat g=\frac{\sum_j w_j C_j}{N}.
\]

With \(G\) independent clusters, the finite-sample estimator variance is

\[
\widehat{\operatorname{Var}}(\widehat g)=
\frac{G}{G-1}\frac{1}{N^2}
\sum_j\left[w_j(C_j-\widehat g)\right]^2.
\]

This is the same Phase 1/2 cluster rule used for antithetic prices and control variates. It applies
to finite-difference and pathwise Greek observations. `num_simulations` always remains the exact
requested number of terminal outcomes.

## 9. Pathwise Delta

Under terminal GBM,

\[
S_T=S_0\exp\left((r-q-\tfrac12\sigma^2)T+\sigma\sqrt T Z\right),
\qquad
\frac{\partial S_T}{\partial S_0}=\frac{S_T}{S_0}.
\]

Using strict ITM indicators, the discounted call contribution is

\[
D_i^C=e^{-rT}\mathbf 1_{\{S_T>K\}}\frac{S_T}{S_0},
\]

and the put contribution is

\[
D_i^P=-e^{-rT}\mathbf 1_{\{S_T<K\}}\frac{S_T}{S_0}.
\]

Equality at strike contributes zero, consistent with Phase 1's ITM convention.

## 10. Pathwise Vega

Differentiating terminal GBM with respect to absolute volatility gives

\[
\frac{\partial S_T}{\partial\sigma}
=S_T\left(-\sigma T+\sqrt T Z\right).
\]

The call contribution is

\[
\mathcal V_i^C=e^{-rT}\mathbf 1_{\{S_T>K\}}S_T
\left(-\sigma T+\sqrt T Z\right),
\]

and the put uses the negative strict-ITM indicator. Contributions are vectorized and their
independent path or pair clusters define the reported mean, SE, and confidence interval.

Pathwise differentiation is not applied to Gamma. The payoff's second derivative is concentrated
at the strike, so naively differentiating the strict indicator would produce an invalid estimator.
Pathwise Theta, Rho, and dividend rho are also deferred because Phase 3 already provides rigorously
coupled finite differences for them.

## 11. Greek comparison diagnostics

The comparison engine selects the analytical field for one requested Greek and evaluates every
stochastic method supported by the current variance configuration. Delta and Vega include pathwise
results for `NONE` or `ANTITHETIC`; every Greek includes finite difference. Control-variate
configurations omit pathwise results rather than silently dropping the requested variance mode.

For stochastic estimate \(\widehat g\), analytical value \(g\), and positive SE,

\[
z=\frac{\widehat g-g}{\operatorname{SE}(\widehat g)}.
\]

If SE is zero, z-score is unavailable. Relative error is unavailable when the analytical benchmark
is zero. All other diagnostics must remain finite.

## 12. Sensitivity engine

The one-dimensional sensitivity engine varies one of `SPOT`, `VOLATILITY`, `MATURITY`,
`RISK_FREE_RATE`, or `DIVIDEND_YIELD`. At each requested value it returns Black-Scholes price and
all six analytical Greeks. It preserves input order and reports the original state's base value.

Spot values must be positive; volatility and maturity values must be non-negative; rates and
dividend yields may be negative. Every value must be finite. The library has no hidden grid limit,
while the HTTP layer defaults to at most 100 points through `MAX_SENSITIVITY_POINTS`.

The engine returns structured values for later visualization but creates no plots or frontend.

## 13. API and resource boundaries

Phase 3 adds:

- `POST /api/greeks/black-scholes`
- `POST /api/greeks/monte-carlo`
- `POST /api/analytics/greeks/compare`
- `POST /api/analytics/sensitivity`

Monte Carlo Greek endpoints reuse `MAX_API_SIMULATIONS`; the sensitivity endpoint uses
`MAX_SENSITIVITY_POINTS`. Requests and responses are strict typed Pydantic models. Simulation
drivers and observation arrays never cross the API boundary. INFO logs contain request method,
Greek names, simulation count, variance mode, runtime, or grid size; DEBUG logs may contain bump
configuration but never arrays.

## 14. Numerical and statistical validation

Tests use published/independently calculated Black-Scholes values, put-call Greek relationships,
and analytical price finite differences. They distinguish exact expiry from very small positive
maturity and exercise the zero-volatility conventions. Direct reconstruction tests verify CRN
path differences, Gamma cluster variance, pathwise contributions, antithetic pairs, and odd
singleton weights. Fixed-seed stochastic assertions use the estimator's own SE plus a small
finite-difference truncation allowance. Representative tests confirm that CRN substantially lowers
Delta noise relative to independently simulated bumps without asserting universal dominance.

## 15. Deferred work

Phase 3 adds no implied-volatility solver, volatility surface, market data, tree, early exercise,
path-dependent payoff, multi-asset process, stochastic-volatility or jump model, calibration,
portfolio aggregation, risk measure, parallelism, GPU execution, quasi-Monte Carlo, or frontend.
Future phases can add model-specific analytical sensitivities, likelihood-ratio estimators,
automatic differentiation, pathwise methods for smooth exotics, and presentation scaling through
new modules that consume the typed Greek results. These are extension directions, not Phase 3
implementations.
