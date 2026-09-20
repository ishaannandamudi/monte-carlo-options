# Phase 16 portfolio risk and scenarios

## 1. Portfolio architecture

`Portfolio` is an immutable Pydantic aggregate with an optional identifier, base currency,
metadata, and a nonempty ordered position list. The base currency is a reporting label: Phase 16
does not perform FX conversion. Position order is preserved; grouping outputs are key-sorted.

## 2. Position architecture

The position union is discriminated by `instrument_class`. It contains typed variants for vanilla
European, vanilla American, Asian, digital, barrier, and basket options. Every position has a
unique identifier, a finite nonzero quantity, optional explicit model, explicit market snapshot,
model configuration, optional market-data provenance reference, and metadata. Positive quantities
are long and negative quantities are short.

## 3. Model dispatch

European vanilla defaults to Black-Scholes; American vanilla defaults to American CRR. Geometric
Asians and baskets default to their analytical engines, digitals to their analytical engine, and
arithmetic Asians, barriers, and arithmetic baskets to Monte Carlo. Explicit vanilla models pass
through the Phase 15 compatibility registry. Exotic compatibility is checked against the existing
instrument-specific engines. An incompatible explicit selection fails that position; no fallback
model is substituted.

## 4. Partial failures

Each position produces `SUCCESS` or `FAILED`, including an exception type and message on failure.
Portfolio completeness is `COMPLETE`, `PARTIAL`, or `FAILED`. With `allow_partial=true`, totals
contain successful positions and the completeness/count fields make omissions explicit. With
`allow_partial=false`, any position failure raises `PortfolioRiskError`.

## 5. Greek conventions

For value `V(S,sigma,r,q,T)`, the canonical values are Delta `dV/dS`, Gamma `d²V/dS²`, Vega
`dV/dsigma`, Rho `dV/dr`, and calendar Theta `-dV/dT`. Vega and Rho use raw absolute units and
Theta uses years. BSM uses Phase 3 analytical Greeks. GBM Monte Carlo uses Phase 3 CRN finite
differences. Other engines use request-local central differences with the same position seed.
Metrics that have no defensible scalar definition remain `null` and appear in
`unavailable_metrics`.

## 6. Net and gross Greeks

For each available Greek `g_i`, net exposure is `sum(g_i)` and gross exposure is
`sum(abs(g_i))`. Missing values are not inserted as zeros. Netting can therefore reduce net risk
while gross risk remains positive.

## 7. Dollar and scaled Greeks

- `dollar_delta = Delta*S`, the linear P&L for a hypothetical 100% proportional spot move.
- `delta_1pct = Delta*S*0.01`.
- `gamma_1pct = 0.5*Gamma*(0.01*S)^2`.
- `vega_1vol_point = Vega*0.01`.
- `rho_1pct = Rho*0.01`.
- `theta_1day = Theta/365`.

Each unit sensitivity is multiplied by signed quantity before portfolio aggregation.

## 8. Underlying aggregation

Single-asset positions map directly to `underlying_id`. Basket per-asset Delta is computed by
central spot bumps using the same Monte Carlo seed on both sides. Basket value and scenario P&L are
allocated across asset identifiers by contractual weights solely for reporting and exact group
sum identities. This allocation is not a risk attribution theorem.

## 9. Expiry aggregation

Exact year-fraction maturities are serialized with a stable 12-significant-digit representation
and sorted lexically. The aggregation contains value, supported Greeks, scaled Greeks, position
count, and missing metric names.

## 10. Option-type aggregation

Calls and puts group by their actual payoff orientation. Asian, digital, barrier, and basket
contracts retain their call/put orientation; instrument class remains a separate dimension.

## 11. Instrument-class and model aggregation

Instrument groups use `VANILLA_EUROPEAN`, `VANILLA_AMERICAN`, `ASIAN`, `DIGITAL`, `BARRIER`, and
`BASKET`. Model groups use the resolved pricing engine, including whether a default was selected.

## 12. Multi-asset exposures

A basket has a map of per-asset Deltas rather than a misleading scalar Delta. Scalar basket Gamma,
per-asset Vega, correlation Greeks, and the cross-Gamma matrix are intentionally unavailable.
Parallel basket Rho and calendar Theta are finite-differenced when the engine supports repricing.

## 13. Scenario architecture

`MarketScenario` is a sparse immutable state transformation. Fields cover global relative and
absolute spot moves, global absolute and relative volatility moves, parallel rate/dividend moves,
elapsed years, per-asset spot maps, Heston parameters, and basket correlation. A zero-valued or
omitted field leaves that factor unchanged.

## 14. Spot shocks

For each single asset, `S' = S*(1+s_rel)+s_abs`; per-underlying maps take precedence over global
values. The same rule is applied separately to every basket asset. A nonpositive result fails the
affected position.

## 15. Volatility shocks

For constant-volatility engines, `sigma' = sigma*(1+v_rel)+v_abs`. Negative volatility is rejected.
For opt-in surface BSM, the base volatility is first queried at the shocked maturity and fixed
strike, then the shock is applied. This is an explicit sticky-strike convention.

## 16. Rate and dividend shocks

Rates and continuous dividend yields receive additive parallel shifts in raw units. Thus 25 basis
points is `0.0025`. Negative rates and yields remain valid.

## 17. Time decay

Elapsed time `dt` gives `T'=max(T-dt,0)`. Full repricing uses the new contract maturity and reaches
the existing expiry-payoff policy at zero. It never passes a negative maturity to an engine.

## 18. Heston scenarios

Heston scenarios explicitly override or add shifts to `v0`, `theta`, `kappa`, `sigma_v`, and
`rho`. A per-position full parameter object can represent an alternate calibration or an ensemble
member. Generic volatility shocks do not map to Heston parameters and produce an explanatory
warning. Phase 14 calibration results may supply the base parameter object; calibration is never
run inside portfolio valuation.

## 19. Correlation scenarios

Baskets accept a position-specific matrix override or an additive off-diagonal shift. The Phase 12
correlation validator checks symmetry, bounds, unit diagonal, and positive semidefiniteness. A
matrix requiring PSD repair is rejected for scenario work instead of being silently changed.

## 20. Full repricing

The canonical scenario result changes the typed market/model state and invokes the assigned engine
again. Position P&L is `V_i' - V_i`; portfolio P&L is the compensated sum of comparable successful
position P&Ls. For partial results, base and scenario portfolio values cover the same successful
subset, so `scenario_value-base_value=full_repricing_pnl` remains true.
Every contribution also serializes typed base and shocked market snapshots: asset-keyed spots,
volatilities and dividend yields, the parallel rate, remaining maturity, optional Heston parameters,
and an optional basket correlation matrix. Live timestamps are not required.

## 21. Greek approximation

For a supported single-asset position,

```text
dV = Delta*dS + 0.5*Gamma*dS^2 + Vega*dsigma + Theta*dt + Rho*dr.
```

Position Greeks already include signed quantity. Basket approximation uses the sum of per-asset
Delta terms and omits cross-Gamma. Dividend, correlation, and Heston parameter moves have no term
in this approximation, so the approximation is explicitly unavailable for those scenarios.

## 22. Approximation limitations

The local expansion omits vanna, volga/vomma, charm, cross-Gamma, rate/volatility interactions,
correlation effects, and all higher orders. Digital and barrier outputs carry a discontinuous-payoff
flag. A large residual is a diagnostic, not a pricing failure.

## 23. Approximation diagnostics

`approximation_residual = full_repricing_pnl - greek_approximation_pnl`. Relative error divides by
`max(abs(full_repricing_pnl),1e-12)` when full P&L is nonzero. A configurable threshold emits
`LARGE_GREEK_APPROXIMATION_ERROR`; missing terms emit `MISSING_GREEKS`.

## 24. Scenario grids and cubes

`ScenarioGridAxis` supplies an ordered axis and values. The engine forms an `itertools.product` in
declared axis order, which directly supports one-dimensional grids, spot-by-volatility grids, and
bounded higher-dimensional cubes. Duplicate axes are rejected. Results are a flat ordered cell
list plus shape, avoiding large nested payloads.

## 25. Stress scenarios

A stress request is an ordered list of uniquely named `MarketScenario` objects. Names label user
assumptions; the library does not hardcode universal sell-off or volatility-spike magnitudes.

## 26. Contribution analysis

Each scenario cell retains position base/scenario value and P&L. Underlying and instrument-class
contributions are compensated sums. Their complete-portfolio sums equal portfolio scenario P&L
within floating-point tolerance.

## 27. P&L explain

P&L explain returns Delta, Gamma, Vega, Theta, and Rho contributions at position and portfolio
levels. The terms use base-state Greeks and actual state differences. The explain is available
without timestamps because elapsed time is an explicit year fraction.

## 28. Residual and higher-order terms

`unexplained_or_higher_order = full_reprice_pnl - explained_total`, so
`explained_total + unexplained_or_higher_order` equals full P&L. The residual can contain valid
nonlinear, cross-factor, discontinuity, and numerical effects and is not universally an error.

## 29. Model-risk integration

The model-risk endpoint applies a named reference and alternate compatible Phase 15 models to the
same vanilla positions. It reports total and position differences. The spread is descriptive and
has no probability interpretation. Exotic incompatibilities remain structured failures.

## 30. RNG and Common Random Numbers

Stochastic seeds are derived as `SeedSequence([root_seed, SHA256(position_id,model)])`, making them
independent of evaluation order. A scenario keeps position identity and model, so base and shocked
valuations reuse the same stream where the engine supports it. Finite-difference Greeks use the
same rule, and Phase 3 GBM Greeks reuse the driver vector directly. No CRN is forced across models.

## 31. Monte Carlo uncertainty

Position values retain base and scenario standard errors and confidence intervals where available.
The current portfolio layer does not retain paired path observations, so it reports P&L standard
error as unavailable rather than combining correlated errors incorrectly. A warning explains this
when CRN is active.

## 32. Reproducibility and ordering

Models are immutable, PCG64 is reached through the existing engines, identifiers are unique, child
seeds are stable, input position order is retained, and groups are sorted. Same inputs and root seed
reproduce numerical results on the same supported numerical environment. Runtime fields naturally
vary.

## 33. Request-local reuse and integrations

Deterministic identical position inputs reuse prices and unit Greeks within a request. Scenario
grids and stress sets compute the base valuation and Greeks once. There is no persistent cache.
`market_data_reference` records Phase 5 snapshot provenance while markets remain explicit; no live
fetch happens during risk calculation. Phase 6 surfaces are used only with
`use_surface_volatility=true`. Phase 14 calibration results supply parameters without recalibration.

## 34. Resource limits

The API defaults are 1,000 positions, 1,000 coordinate values, 2,500 grid cells, 100 stress
scenarios, and seven alternate models. It also enforces every existing engine's simulation, node,
monitoring, tree, and asset cap before numerical work. Library functions remain usable for explicit
research workloads outside HTTP.

## 35. Performance and memory

Analytical repeated inputs benefit from request-local reuse. Results retain only scalar prices,
Greeks, diagnostics, and contributions; raw paths never enter portfolio responses. Grid memory is
approximately proportional to positions plus returned cells. Local timing data appears in
`benchmarks/phase16-portfolio-risk-scenarios.md` and is descriptive rather than a latency promise.

## 36. Known state limitations and Phase 17 boundary

Barrier scenarios assume the existing current hit-state convention and do not invent historical
hits. Asian time shifts do not model prior fixings or partially observed averages. Phase 16 does
not implement historical, parametric, or Monte Carlo VaR, Expected Shortfall, VaR backtesting,
capital, portfolio optimization, live risk streaming, or HFT integration. Those remain Phase 17 or
later consumers of this portfolio boundary.
