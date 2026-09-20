# Phase 17 VaR, Expected Shortfall, and portfolio market risk

## 1. Scope and architecture

Phase 17 is a consumer of the Phase 16 portfolio boundary. Typed contracts live in
`backend/app/models/risk.py`; estimators and covariance checks live in
`backend/app/risk/statistics.py`; factor mapping and the three risk engines live in
`backend/app/risk/engine.py`; exception tests live in `backend/app/risk/backtesting.py`; and
`backend/app/api/risk.py` applies HTTP resource limits before dispatch. Phase 17 does not replace a
pricing engine, recalibrate Heston, or introduce a second portfolio representation.

## 2. Loss convention

Every method uses `PnL = V_horizon - V_today` and `Loss = -PnL`. Positive loss is adverse. VaR,
ES, worst scenarios, contributions, and backtest exceptions all use loss units. Scenario engines
continue to return P&L; the risk boundary performs the sign conversion exactly once.

## 3. Value at Risk

`VaR_alpha` is the alpha quantile of loss. Empirical methods call `numpy.quantile` with a stored,
explicit method. The default is `linear`; `lower`, `higher`, `midpoint`, and `nearest` are available.
Parametric normal VaR is `mu_L + z_alpha sigma_L`. VaR may be negative when modeled gains dominate;
the engine does not silently floor a valid result at zero.

## 4. Expected Shortfall

Historical and Monte Carlo ES average the worst exact mass `N(1-alpha)`. Sort losses descending,
give the first `floor(N(1-alpha))` observations unit weight, and give the next observation the
fractional remainder. When the mass is below one, the worst observation receives unit weight. The
same indices and weights define empirical ES contributions. Normal ES is
`mu_L + sigma_L phi(z_alpha)/(1-alpha)`.

## 5. Confidence levels

Any finite `0 < alpha < 1` is accepted; 0.99 is the API/model default. Common 0.95, 0.975, 0.99,
and 0.995 values require no special cases. Confidence level is separate from a pricing engine's
Monte Carlo confidence interval.

## 6. Horizons

`RiskHorizon` stores positive trading days and trading days per year, defaulting to 1 and 252. Its
year fraction is used only when `include_time_passage=true`. Parametric and factor Monte Carlo IID
scaling use `mean_h=h*mean_1` and `cov_h=h*cov_1`. Historical shocks are daily observations replayed
without automatic calendar or square-root scaling; a multi-day historical request states this in a
warning.

## 7. Historical observations and scenarios

An observation has a factor identifier, finite change, optional source timestamp, and bounded typed
metadata. `HistoricalRiskScenario` combines simultaneous observations under one authoritative
timestamp. Factor identifiers cannot repeat within a scenario. `HistoricalRiskDataset` requires at
least two scenarios in strict chronological order with unique timestamps and records its sampling
frequency. Same-date shocks are never independently resampled.

## 8. Factor mapping

Mappings bind factor IDs to spot relative returns, absolute volatility/rate/dividend changes,
explicit Heston parameter changes, basket equicorrelation changes, or a custom Gaussian exposure.
Spot, volatility, and dividend factors require an underlying; Heston/correlation factors require a
position. Delta-mode spot exposure is `Delta*S`; volatility and rate exposures are raw Phase 16
Vega and Rho. Basket spot factors use directly bumped per-asset Delta. Explicit per-position linear
and diagonal-Gamma exposures support well-defined user factors without pretending Phase 16 has a
Greek it does not expose.

## 9. Missing-factor policy

The default `ERROR` policy rejects a dated scenario missing any mapped factor. `ZERO_SHOCK` fills
only a known missing mapping with zero. `DROP_SCENARIO` drops the complete timestamp so dependence
among the remaining factors is preserved. Unmapped observations are errors; they are never ignored
as if they were known. A post-filter empty dataset is invalid.

## 10. Historical full repricing

Each retained row becomes one `MarketScenario`. Per-underlying spot, volatility, and dividend
changes, the rate change, explicit Heston overrides, and explicit basket-correlation shifts are
passed to Phase 16 `run_scenario`. The base valuation and Greeks are reused across rows. Phase 16
validates positive spots/volatilities, Heston domains, and every shocked correlation matrix. The
full-repricing P&L is authoritative; maximum observed local-approximation error is diagnostic.

## 11. Historical Greek approximation

DELTA uses `sum_i exposure_i*x_i`. DELTA_GAMMA_DIAGONAL adds
`0.5*sum_i gamma_exposure_i*x_i^2`. Optional time passage adds each position's calendar Theta times
the risk year fraction. Vanna, volga, charm, cross-Gamma, rate/volatility interactions, and other
cross terms are omitted and disclosed. Dividend, Heston, and correlation approximations require
explicit exposures because Phase 16 does not supply their canonical Greeks.

## 12. Empirical distribution output

Reports include sample count, mean and sample standard deviation of P&L/loss, extrema, median,
selected loss quantiles, tail size, skewness, excess kurtosis, and Jarque-Bera diagnostics when the
sample supports them. Worst-scenario output is bounded and contains timestamp/index, loss, P&L,
mapped shocks, and available position losses. Raw distributions do not cross the API.

## 13. Covariance inputs and estimation

`CovarianceInput` stores ordered unique factor IDs, a symmetric covariance matrix, and optional
daily means. Input order may differ from mapping order; the engine reorders exactly by identifier.
Alternatively, complete historical scenario rows produce ordinary sample means and sample
covariance with `ddof=1`. EWMA and shrinkage are intentionally absent.

## 14. PSD validation

Covariance shape, finiteness, symmetry, and eigenvalues are checked. Material eigenvalues below
`-1e-10 * max(1, matrix scale)` are rejected. Only tiny floating-point-negative eigenvalues are
clipped during preparation/factorization. Diagnostics report the minimum effective eigenvalue,
rank, full-rank condition number when defined, source, factor order, and sample count.

## 15. Delta-normal risk

For portfolio factor exposure `w`, daily mean `m`, and covariance `Sigma`, P&L mean is
`w'm + theta_passage`; P&L variance is `w'Sigma w`. Loss mean is its negative. The output contains
analytical normal VaR and ES and a theoretical distribution summary; unavailable sample extrema,
tail counts, and normality-test results remain null/zero rather than fabricated.

## 16. Marginal and component VaR

For factor `j`, `dVaR/dw_j = -m_j + z*(Sigma w)_j/sigma` when `sigma>0`; the variance term is zero
when risk is deterministic. Factor component VaR multiplies by `w_j`. Position component VaR uses
the position exposure vector and its mapped mean/Theta contribution. Position components sum to
portfolio VaR, including nonzero mean and optional time passage. The same Euler construction is
reported for normal ES.

## 17. Incremental VaR

When requested, parametric incremental VaR equals portfolio VaR minus VaR after removing one
position's exposure, mean, and Theta contribution. Empirical incremental VaR is available when a
historical/full position P&L matrix exists and applies the configured quantile to the leave-one-out
loss vector. The API caps contribution position counts before this repeated work.

## 18. Delta-Gamma diagonal risk

Phase 16 has scalar Gamma for single-asset positions and no basket cross-Gamma matrix. Phase 17
therefore labels its second-order method `DELTA_GAMMA_DIAGONAL`. Spot-return Gamma exposure is
`Gamma*S^2`; custom factors may supply explicit diagonal exposure. No result is described as a full
quadratic portfolio approximation.

## 19. Monte Carlo factor model

Monte Carlo uses the prepared mean and covariance and an eigen square root, including singular PSD
matrices. MULTIVARIATE_NORMAL is required baseline. MULTIVARIATE_STUDENT_T divides Gaussian draws
by a chi-square scale and multiplies by `sqrt(df-2)`, so the supplied covariance remains the finite
covariance when `df>2`. Generic factor simulation does not silently invoke Heston.

## 20. RNG, antithetic draws, and reproducibility

An isolated PCG64 `Generator` owns each run, seeded through `SeedSequence`. No global NumPy random
state is used. Same request, seed, batch size, and numerical environment reproduce shocks and risk
statistics. Antithetic mode interleaves `z,-z`; Student-t pairs share the scale. Odd batches retain
one deterministic singleton, so requested simulation count is exact. Antithetics are not claimed
to improve a tail quantile universally.

## 21. Simulation batching and memory

Factor shocks are generated in configured chunks. Greek modes compute batch P&L vectorially and
retain only the O(N) portfolio P&L vector needed for exact quantiles. If tail details or ES
contributions are requested, the identical factor stream is regenerated and only selected tail
rows are retained. Full repricing iterates scenarios while reusing Phase 16 base state. Its optional
position matrix is allowed only under stricter API scenario-position limits.

## 22. Monte Carlo valuation modes

DELTA and DELTA_GAMMA_DIAGONAL use the exposure matrices described above. FULL_REVALUATION converts
every generated factor row to a Phase 16 scenario and reprices the portfolio. Heston parameters and
basket correlations change only through explicitly mapped factors. Full revaluation rejects custom
Gaussian shocks because no market-state transformation exists.

## 23. ES contributions

For empirical methods, each position contribution is its weighted mean loss over exactly the
portfolio ES tail. Therefore position contributions sum to portfolio ES, including fractional tail
mass. For normal risk, position ES contribution is the analytical Euler contribution. Position
results also contain mark-to-model value, value fraction where net value is nonzero, selected
linear exposure, and available component/incremental measures.

## 24. Grouping and concentration

Position risk is aggregated by underlying, instrument class, and selected pricing model. A basket
contribution is allocated equally across its underlying reporting keys because Phase 17 has no
unique scalar allocation of tail loss; directly mapped basket Delta remains asset-specific.
Concentration reports largest absolute component-VaR/ES contributors, absolute-contribution HHI,
and the top five position IDs. These are descriptive quantities, not risk ratings.

## 25. Stress integration

`/api/risk/report` first computes the selected probabilistic risk method, then invokes Phase 16
named stresses. Stress P&L/loss is stored in `stress_losses`; scenario count, VaR, ES, tail, and
contributions are unchanged. Stress outcomes are never appended to a historical or simulated loss
sample.

## 26. Backtest exception policy

`VaRBacktestObservation` stores date, predicted VaR, realized P&L, and optional predicted ES. Loss
is derived as negative P&L. An exception occurs only for `Loss > VaR`; equality is explicitly not an
exception. Backtest dates must be unique and chronological.

## 27. Kupiec proportion-of-failures test

The null exception probability is `1-alpha`. The likelihood-ratio statistic compares the null
binomial log likelihood with the maximum-likelihood observed exception rate and uses chi-square one
degree of freedom for its descriptive p-value. `scipy.special.xlogy` makes zero and all-exception
boundaries finite without adding arbitrary probabilities.

## 28. Christoffersen independence and conditional coverage

Transitions 00, 01, 10, and 11 are counted in date order. The independence statistic compares one
unconditional transition probability with separate post-nonexception and post-exception
probabilities and reports a chi-square-one p-value. The conditional-coverage statistic is Kupiec
plus independence with a chi-square-two p-value. Degenerate zero-count transition rows remain
stable.

## 29. Traffic-light diagnostic

Observed exceptions are compared with expected count plus/minus two binomial standard deviations
and labeled BELOW_EXPECTED, WITHIN_EXPECTED_RANGE, or ABOVE_EXPECTED. This is a configurable-model
diagnostic convention in spirit, not a Basel regulatory classification or capital decision.

## 30. Normality and uncertainty limitations

Empirical skew, excess kurtosis, and Jarque-Bera results describe the realized sample and never
select a model automatically. Phase 17 does not yet bootstrap VaR/ES confidence intervals. Pricing
Monte Carlo uncertainty and market-risk distribution uncertainty remain distinct. Student-t adds a
specified heavy-tail family but is not a fitted extreme-value model.

## 31. API surface

Dedicated routes are `/api/risk/historical`, `/parametric`, and `/monte-carlo`. `/portfolio`
dispatches a unified typed request, `/contributions` requires contributions, `/backtest` evaluates
exceptions, and `/report` adds separate stresses. Validation and numerical-domain failures return
HTTP 422 through the existing application handler. Response models never expose raw loss arrays.

## 32. Resource safeguards

Environment settings cap historical scenarios, factor simulations, risk positions,
scenario-position full-repricing evaluations, contribution positions, backtest observations, and
tail output. Limits are checked before scenario generation and matrix allocation. Existing nested
path, tree, LSM, exotic, basket, Heston, and portfolio caps are reapplied with scenario repetition
counts. Direct library calls remain flexible for controlled research.

## 33. Computational complexity

Delta and diagonal Delta-Gamma P&L construction is O(NF + NP) in factor and selected-position
work, with O(N) retained loss memory outside requested tail state. Parametric variance is O(F^2)
after covariance validation/factorization. Historical or Monte Carlo full repricing is O(NP) times
the assigned pricing engines. Exact covariance eigenvalidation is O(F^3); this is why very wide
factor models require deliberate resource planning.

## 34. Tests and validation

Deterministic tests cover loss signs, explicit quantiles, fractional ES, normal formulas, zero and
single-factor risk, supplied/estimated covariance, correlation, horizon scaling, full/Greek
historical replay, missing-factor policies, Heston/correlation shocks, simulation reproducibility,
Student-t and antithetic behavior, Euler/incremental/ES identities, grouping, Kupiec,
Christoffersen, strict equality, stress separation, all APIs, serialization, and every limit.
Performance tests are opt-in. All Phase 1-16 regression tests remain unchanged.

## 35. Deferred work and Phase 18 boundary

Phase 17 deliberately omits filtered historical simulation, covariance shrinkage, full surface
PCA, cross-Gamma, advanced ES regulatory backtests, EVT fitting, jump diffusion, regime switching,
QMC, distributed/GPU simulation, portfolio optimization, live HFT/broker risk, and a frontend.
Future work can consume typed factors, loss distributions, and backtests without rewriting pricing
or Phase 16 scenarios. No Phase 18 functionality is implemented here.
