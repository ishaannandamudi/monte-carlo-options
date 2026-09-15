# Phase 2 variance-reduction design

## 1. Scope and motivation

Phase 2 reduces the sampling variance of the Phase 1 European-option Monte Carlo estimator. It
adds a discounted-stock control variate and a method that combines that control with antithetic
sampling. It does not change the contract, market, payoff, stochastic-process, Black-Scholes, RNG,
or HTTP/library boundaries established in Phase 1.

The supported methods are:

- `NONE`
- `ANTITHETIC`
- `CONTROL_VARIATE`
- `ANTITHETIC_CONTROL_VARIATE`

Variance reduction matters because the Monte Carlo error at a fixed path count depends on the
variance of the observations used by the estimator. Lowering that variance gives a narrower
confidence interval for the same simulation budget. It does not guarantee a better realized error
for every finite sample.

## 2. Plain discounted-payoff estimator

For terminal stock value \(S_{T,i}\), define the discounted payoff

\[
Y_i = e^{-rT}h(S_{T,i}),
\]

where \(h(s)=\max(s-K,0)\) for a call and \(h(s)=\max(K-s,0)\) for a put. With \(N\) independent
paths, the plain estimator is

\[
\widehat V_{\mathrm{plain}}=\bar Y=\frac{1}{N}\sum_{i=1}^{N}Y_i.
\]

Its estimated variance and standard error are

\[
\widehat{\operatorname{Var}}(\bar Y)=\frac{s_Y^2}{N},
\qquad
\widehat{\operatorname{SE}}(\bar Y)=\sqrt{\widehat{\operatorname{Var}}(\bar Y)},
\]

where \(s_Y^2\) uses Bessel's correction (`ddof=1`). All prices and uncertainty calculations use
discounted observations.

## 3. Antithetic estimator

For each primary normal draw \(Z_j\), antithetic simulation evaluates both \(Z_j\) and \(-Z_j\).
For an even path count \(N=2M\), the independent estimator observations are pair averages

\[
A_j^Y=\frac{Y_j^+ + Y_j^-}{2}, \qquad j=1,\ldots,M.
\]

The price is the mean of the \(M\) pair averages. The standard error is their Bessel-corrected
sample standard deviation divided by \(\sqrt M\). The two values inside a pair are correlated and
are never counted as independent observations.

Phase 1 owns this sampling rule. Phase 2 reuses it unchanged for `ANTITHETIC` and extends the same
independent-cluster representation to the combined estimator.

## 4. Control-variable selection and known expectation

The control variable is discounted terminal stock:

\[
X_i=e^{-rT}S_{T,i}.
\]

Under the risk-neutral GBM model,

\[
E[S_T]=S_0e^{(r-q)T},
\]

so its known discounted expectation is

\[
\mu_X=E[X]=e^{-rT}E[S_T]=S_0e^{-qT}.
\]

This control is useful because the option payoff and terminal stock are correlated, while
\(\mu_X\) is available exactly under the same model. The implementation computes both discount
factors with the Phase 1 finite-number guards.

## 5. Estimated beta and adjusted estimator

For independent payoff observations \(Y_i\) and matching control observations \(X_i\), the
sample-optimal coefficient is

\[
\widehat\beta=
\frac{\widehat{\operatorname{Cov}}(Y,X)}
     {\widehat{\operatorname{Var}}(X)}.
\]

Both numerator and denominator use the same Bessel correction, so the correction cancels in the
ratio. Beta is estimated from the current simulation rather than hard-coded. Each adjusted
observation is

\[
Y_i^{cv}=Y_i-\widehat\beta(X_i-\mu_X),
\]

and the primary returned price for a control method is

\[
\widehat V_{cv}=\frac{1}{N}\sum_{i=1}^{N}Y_i^{cv}.
\]

The result retains the unadjusted price and uncertainty so callers can inspect the realized
effect of the transformation.

## 6. Adjusted variance and standard error

For `CONTROL_VARIATE`, paths are independent clusters of size one. The adjusted estimator variance
is

\[
\widehat{\operatorname{Var}}(\widehat V_{cv})=
\frac{s_{Y^{cv}}^2}{N},
\]

and the reported standard error is its square root. The fields `raw_estimator_variance` and
`adjusted_estimator_variance` are variances of the corresponding **price estimators**, so they equal
`raw_standard_error ** 2` and `adjusted_standard_error ** 2`. They are not the individual-path
payoff variances. `payoff_variance` continues to expose the Phase 1 Bessel-corrected variance of
individual discounted payoffs.

Confidence intervals use the selected estimator and its standard error:

\[
\widehat V \mathbin{\pm}
\Phi^{-1}\!\left(\frac{1+c}{2}\right)\widehat{\operatorname{SE}}(\widehat V),
\]

where \(c\) is the requested confidence level. Black-Scholes comparison z-scores likewise use the
adjusted standard error whenever a control method is selected.

## 7. Combined antithetic and control treatment

For `ANTITHETIC_CONTROL_VARIATE`, the implementation first forms matching pair averages for both
variables:

\[
A_j^Y=\frac{Y_j^+ + Y_j^-}{2},
\qquad
A_j^X=\frac{X_j^+ + X_j^-}{2}.
\]

It then estimates beta from the independent pairs \((A_j^Y,A_j^X)\), applies

\[
A_j^{cv}=A_j^Y-\widehat\beta(A_j^X-\mu_X),
\]

and computes the price and standard error from those adjusted independent clusters. Beta and
uncertainty are never calculated by pretending that the two paths within a pair are independent.

For even \(N=2M\), this reduces exactly to ordinary sample covariance, variance, and standard error
over the \(M\) pair averages.

## 8. Odd path counts and weighted clusters

An odd request \(N=2M+1\) returns exactly \(N\) terminal outcomes. The first \(2M\) form antithetic
pairs; the last independently drawn primary normal remains a singleton. Let cluster \(j\) have
mean \(C_j\) and size \(w_j\), where pair weights are 2 and the singleton weight is 1. The estimator
is the path-weighted mean

\[
\widehat\mu=\frac{\sum_j w_j C_j}{N}.
\]

For \(G\) independent clusters, the finite-sample cluster variance used by both Phase 1
antithetic uncertainty and Phase 2 control calculations is

\[
\widehat{\operatorname{Var}}(\widehat\mu)=
\frac{G}{G-1}\frac{1}{N^2}
\sum_{j=1}^{G}\left[w_j(C_j-\widehat\mu)\right]^2.
\]

For two matched variables, their estimator covariance replaces the squared score with the product
of their weighted centered scores. The combined method uses the ratio of this cluster covariance
to the corresponding cluster control variance for beta. Adjustment is then evaluated with the same
weights. Thus the singleton contributes one path to the estimate and one correctly weighted
independent cluster to beta and uncertainty.

At least two independent clusters are required to estimate stochastic uncertainty. Smaller
stochastic requests raise a clear `StatisticalError`; deterministic maturity-zero or
volatility-zero requests remain exact even with one path.

## 9. Variance-reduction ratio

The reported ratio is

\[
\mathrm{VRR}=
\frac{\widehat{\operatorname{Var}}(\widehat V_{raw})}
     {\widehat{\operatorname{Var}}(\widehat V_{adjusted})}.
\]

Because both variances refer to the same path budget and clustering, this is also the ratio of
their squared standard errors. A value above one indicates lower estimated variance in that
sample; one indicates no change; below one indicates worse sampled variance. No implementation or
test assumes universal improvement. If adjusted variance is zero, the ratio is returned as `None`
instead of infinity.

## 10. Fixed-budget method comparison

`compare_variance_reduction_methods` evaluates all four modes for the same contract, market,
confidence level, path count, and root seed. `NONE` is the baseline.

The comparison uses controlled common root randomness:

- `NONE` and `CONTROL_VARIATE` construct identical `PCG64` streams and therefore receive the same
  path normals.
- `ANTITHETIC` and `ANTITHETIC_CONTROL_VARIATE` construct identical streams and therefore receive
  the same primary normals, antithetic partners, and odd singleton when present.

This isolates each within-family transformation from an unrelated seed change. Plain and
antithetic methods consume different numbers of primary draws, so the comparison does not claim a
one-to-one pairing across those two families.

For method estimator variance \(v_m\) and baseline variance \(v_0\), the reported diagnostics are

\[
\text{relative variance}=v_m/v_0,
\qquad
\text{relative efficiency}=v_0/v_m.
\]

Runtime is reported separately and is not folded into those statistical quantities. If a required
denominator is zero, the corresponding ratio is unavailable rather than infinite.

## 11. Convergence experiments

The Phase 1 convergence engine accepts every Phase 2 method. It derives one deterministic child
seed per path count from `SeedSequence(root_seed)`, so different convergence points are independent
and the complete experiment is reproducible. Control methods contribute their adjusted price and
adjusted standard error to each convergence point.

For ordinary finite-variance Monte Carlo,

\[
\operatorname{SE}(\widehat V_N)=O(N^{-1/2}).
\]

Control and antithetic transformations primarily reduce the variance constant. They do not change
the IID Monte Carlo order. Consequently the fitted regression

\[
\log(\operatorname{SE})=a+b\log(N)
\]

still has a theoretical slope near \(-1/2\) under ordinary conditions.

## 12. Numerical edge cases

- **Zero maturity:** terminal stock equals spot, payoff is intrinsic value, beta is zero, and raw
  and adjusted uncertainty are zero.
- **Zero volatility:** the risk-neutral terminal stock and payoff are deterministic. The same
  beta-zero fallback applies.
- **Zero sampled control variance:** beta is set to zero. The adjusted estimator equals the raw
  estimator, and the variance-reduction ratio is one when raw variance is positive.
- **Zero adjusted variance:** the standard error and confidence-interval width are zero; VRR is
  unavailable to avoid an infinite serialized value.
- **Tiny time or volatility and extreme moneyness:** calculations remain vectorized and use the
  observed finite variance. A constant sampled payoff is valid. A statistically unidentifiable
  control coefficient falls back to zero.
- **Negative rates and dividends:** both are supported. The control expectation remains
  \(S_0e^{-qT}\), and all discounts use the supplied continuous rates.
- **Non-finite derived values:** covariance, variance, beta, adjusted observations, estimates,
  standard errors, terminal values, and discount factors are checked. An unrecoverable failure
  raises a quantitative domain exception instead of returning NaN or infinity.

## 13. Module and API boundaries

- `simulation/gbm.py` owns terminal draws and antithetic normal construction.
- `pricing/variance_reduction.py` owns independent-cluster construction, weighted estimator
  variance/covariance, the discounted-stock control, and the control transformation.
- `pricing/monte_carlo.py` owns payoff discounting, chooses the configured transformation, and
  assembles the typed pricing result.
- `analytics/variance_reduction.py` owns fixed-budget method comparisons.
- `analytics/convergence.py` remains the convergence service for every supported method.
- `api/analytics.py` applies HTTP resource limits and delegates to the analytics services.

The API returns typed scalar diagnostics and never exposes terminal-price, payoff, control, or
adjusted-observation arrays. `batch_size` remains a validated, result-neutral hint; streaming is
deferred until it can preserve covariance, pairing, singleton weights, and exact RNG behavior.

## 14. Validation strategy

Small deterministic arrays verify covariance, control variance, beta, adjustment, estimator mean,
standard error, VRR, pair-level calculations, and odd-cluster weights. Fixed-seed pricing tests
cover calls, puts, dividends, negative rates, deterministic limits, tiny parameters, extreme
moneyness, reproducibility, finite outputs, and statistical agreement with Black-Scholes. Selected
vanilla cases check that control variants reduce sampled estimator variance without treating that
outcome as universal. API tests cover all enum values, result serialization, comparison output,
invalid methods, and resource limits.

## 15. Explicitly deferred methods

Phase 2 does not implement Sobol or other quasi-Monte Carlo sequences, stratification, importance
sampling, moment matching, common-random-number portfolio studies, parallel simulation, GPU
execution, path-dependent controls, Greeks, new payoffs, or new stochastic processes. A future
method must define its dependence structure and uncertainty estimator before joining the existing
configuration and comparison interfaces. Those additions do not require replacing the Phase 1
contract, market, process, payoff, pricing, or RNG foundations.

Phase 3 builds Common-Random-Number finite differences and pathwise sensitivities on these same
independent estimator clusters. Its formulas and boundary conventions are documented in
[phase3-greeks.md](phase3-greeks.md).
