# Phase 12: Correlated Multi-Asset GBM and Basket Options

## 1. Module boundaries

`models/multi_asset.py` owns immutable market inputs and correlation diagnostics.
`simulation/correlation.py` validates and factors correlation matrices.
`simulation/multi_asset_gbm.py` transforms independent PCG64 normals and evaluates exact GBM
states. `payoffs/basket.py` reduces terminal states to basket values and payoffs.
`pricing/basket.py` owns analytical and Monte Carlo estimators. `analytics/basket.py` and
`analytics/multi_asset.py` run reproducible experiments. `api/basket.py` only validates HTTP work
budgets and delegates to these library services.

This separation keeps correlation, stochastic evolution, payoff definition, estimator statistics,
and transport concerns independently testable.

## 2. Multi-asset market model

`MultiAssetMarketEnvironment` contains `N` positive spots, one finite continuously compounded
risk-free rate, `N` finite continuous dividend yields, `N` nonnegative volatilities, an `N x N`
correlation matrix, and optional unique asset identifiers. Contract strike, maturity, option type,
and weights remain in `BasketOptionContract`. The two vector dimensions must agree when a pricing
service is called.

## 3. Correlation requirements and PSD validation

A submitted matrix must be finite, nonempty, square, symmetric to `1e-12`, have a unit diagonal to
`1e-12`, and contain entries in `[-1,1]` to `1e-12`. Validation symmetrizes and clamps only values
already within those representation tolerances. It then uses `numpy.linalg.eigh` on the symmetric
matrix.

The positive-semidefinite policy is explicit:

- a minimum eigenvalue below `-1e-10` is materially invalid and is rejected;
- an eigenvalue in `[-1e-10,0)` is treated as floating-point noise, clipped to zero, reconstructed,
  and renormalized to a unit diagonal;
- the model records the original minimum eigenvalue, whether repair occurred, and the Frobenius
  norm of the adjustment.

This is a small numerical repair, not a general nearest-correlation algorithm. Large invalidities
are never silently changed.

## 4. Cholesky and eigen factorization

Positive-definite matrices use a lower Cholesky factor `L` satisfying `C = L L^T`. Valid singular
PSD matrices use the symmetric eigendecomposition

```text
C = Q Lambda Q^T,       L = Q sqrt(max(Lambda, 0)).
```

For independent row vectors `epsilon`, correlated drivers are `Z = epsilon L^T`. Diagnostics expose
`CHOLESKY` or `EIGEN`, effective rank (eigenvalues above `1e-12 max(1,lambda_max)`), minimum and
maximum eigenvalues, and a condition number only for full-rank matrices. Perfect correlation is
therefore a supported singular model, not a factorization failure.

## 5. Correlated Gaussian generation

All randomness comes from an explicitly owned `numpy.random.Generator(PCG64(SeedSequence(seed)))`.
The independent array is generated first and correlation is applied only along its final asset
axis. No module uses global `numpy.random` state. Empirical correlation diagnostics return target,
sample, absolute-error matrices, maximum absolute error, Frobenius error, and factorization
metadata; raw draws are never returned by the API.

## 6. Terminal correlated GBM

For asset `i`, Phase 12 evaluates

```text
S_i(T) = S_i(0) exp[(r-q_i-0.5 sigma_i^2)T + sigma_i sqrt(T) Z_i].
```

The terminal convention is `(num_simulations, num_assets)`. Drift, diffusion, exponential, and
spot multiplication are vectorized. Terminal basket pricing allocates this matrix and never a path
cube, so its main state cost is `O(paths * assets)`. Exponential underflow, overflow, NaN, or
infinity raises a domain error rather than entering a result.

At `T=0`, every row is the spot vector. If all volatilities are zero, each component is
`S_i(0) exp[(r-q_i)T]`; neither branch consumes RNG state. Assets with individual zero volatility
remain deterministic while other components are simulated. Negative rates and per-asset negative
or positive dividend yields require no special branch.

## 7. Full paths at arbitrary times

Full paths use `(num_simulations, num_time_points, num_assets)`, where the first time point is time
zero. Callers supply strictly increasing, strictly positive future observation times. For each
actual interval `Delta t`, the exact transition is

```text
Delta log S_i = (r-q_i-0.5 sigma_i^2) Delta t
              + sigma_i sqrt(Delta t) Z_i.
```

The engine does not round observations onto a regular grid. It is an internal research primitive;
terminal basket pricing does not call it.

## 8. Multivariate antithetic policy

For a complete independent Gaussian tensor `epsilon`, the next path receives `-epsilon` before
either vector is multiplied by the correlation factor. Linearity gives `L(-epsilon)=-(L epsilon)`,
preserving every cross-asset correlation and the intended path pairing. Paths are interleaved as
`Z,-Z`. An odd requested count returns exactly that count, with a final independent singleton.

For standard antithetic basket pricing, estimator uncertainty uses independent pair clusters and
the existing weighted odd-singleton sandwich formula. Individual paired payoffs are never counted
as independent observations.

## 9. Basket contract and weight convention

`BasketOptionContract` is European and fixed strike. It holds `CALL` or `PUT`, positive strike,
nonnegative maturity, `ARITHMETIC` or `GEOMETRIC`, and one nonnegative weight per asset. At least one
weight must be positive. Weights must sum to one within `1e-10`; accepted rounding drift is
normalized exactly to one. Negative weights are excluded because a geometric basket with signed
weights does not have the selected portfolio semantics.

## 10. Arithmetic and geometric baskets

The arithmetic terminal value and payoff are

```text
B_A = sum_i w_i S_i(T)
call = max(B_A-K,0),     put = max(K-B_A,0).
```

The geometric value is evaluated in log space:

```text
log B_G = sum_i w_i log S_i(T),     B_G = exp(log B_G).
```

Payoff helpers are vectorized and do not discount. Equality to strike is not in the money.

## 11. Geometric basket distribution and analytical price

Under the correlated GBM model, `log B_G` is normal with

```text
mu_G = sum_i w_i [log S_i(0) + (r-q_i-0.5 sigma_i^2)T]

v_G = T sum_i sum_j w_i w_j sigma_i sigma_j rho_ij.
```

Let `m = exp(mu_G + v_G/2)`. For positive `v_G`, define

```text
d2 = (mu_G-log K)/sqrt(v_G),       d1 = d2 + sqrt(v_G).
```

The prices are computed directly from the terminal lognormal law:

```text
call = exp(-rT) [m N(d1) - K N(d2)]
put  = exp(-rT) [K N(-d2) - m N(-d1)].
```

This is not ordinary Black-Scholes applied to a fictitious traded basket. When `v_G=0`, the engine
discounts the deterministic payoff. With one asset and weight one, the formula algebraically and
numerically reduces to the existing dividend-paying Black-Scholes-Merton price.

## 12. Basket Monte Carlo estimator

The engine simulates terminal assets, constructs `B_A` or `B_G`, applies the European payoff, and
forms observations

```text
Y = exp(-rT) payoff(B_T).
```

The estimate, variance, standard error, and confidence interval all refer to discounted payoff
observations. Results also include seed, requested path count, method, asset count, normalized
weights, basket mean and sample standard deviation, strict ITM probability, runtimes, deterministic
status, and factorization diagnostics. Arrays are internal only.

## 13. Geometric control variate

For an arithmetic basket, use

```text
Y = exp(-rT) arithmetic_payoff,
X = exp(-rT) geometric_payoff,
E[X] = exact geometric analytical price.
```

At the independent-observation level,

```text
beta_hat = Cov(Y,X)/Var(X),
Y_cv = Y - beta_hat (X-E[X]).
```

If control variance is zero, `beta_hat=0`. Results retain raw and adjusted prices, standard errors,
estimator variances, beta, sample control mean, known analytical mean, and the raw-to-adjusted
variance ratio. Control sampling is rejected for an already geometric payoff.

For `ANTITHETIC_GEOMETRIC_CONTROL`, both `Y` and `X` are averaged over identical `Z,-Z` clusters
before covariance, beta, adjustment, and SE are estimated. An odd singleton keeps one-path weight
through the existing cluster estimator.

## 14. Deterministic boundaries

At zero maturity, arithmetic and geometric baskets are formed from initial spots and intrinsic
value is returned with zero SE and no RNG draw. When every positive-weight component has zero
volatility, exact risk-neutral terminal values are used once, again with zero SE. A geometric
basket can also have `v_G=0` through singular correlation cancellation; its analytical formula
uses the deterministic branch.

## 15. Comparisons and convergence

Arithmetic method comparison runs `NONE`, `ANTITHETIC`, `GEOMETRIC_CONTROL`, and
`ANTITHETIC_GEOMETRIC_CONTROL` with a common root seed. Geometric comparison runs plain and
antithetic Monte Carlo and reports the exact analytical reference and absolute errors.

Path-count convergence sorts unique positive counts and derives independent deterministic child
seeds with `SeedSequence(root_seed).spawn`. It reports price, SE, interval width, runtime, and the
analytical error when available. With at least two positive-SE points, least squares estimates `b`
in `log(SE)=a+b log(N)`; otherwise the slope is unavailable.

## 16. Correlation and weight sensitivity

Correlation sensitivity builds a validated equicorrelation matrix for each requested `rho`. For
`N` assets the admissible interval is `[-1/(N-1),1]`; the two-asset form is the usual rho sweep.
Every point reuses the same seed, so the underlying independent normals are Common Random Numbers
(CRN) before their scenario-specific factor transform. Results report price, SE, basket mean,
basket variance, and runtime. The service does not claim universal price monotonicity.

Two-asset weight sensitivity maps each requested `w_1 in [0,1]` to `(w_1,1-w_1)` and likewise
reuses the same seed. Endpoints reduce to single-asset payoffs. More general portfolio scenario
machinery is intentionally deferred.

## 17. Resource safety and computational cost

The HTTP layer applies independent settings without changing library mathematics:

```text
MAX_MULTI_ASSET_COUNT=100
MAX_MULTI_ASSET_PATH_NODES=25000000
MAX_BASKET_CONVERGENCE_POINTS=20
MAX_BASKET_SENSITIVITY_POINTS=50
```

It also retains the global per-run simulation cap. Comparison and sensitivity endpoints charge
their complete number of scenarios against the path-node budget. Correlation factorization costs
`O(N^3)` time and `O(N^2)` storage. Terminal simulation costs `O(paths*N^2)` for a dense factor and
`O(paths*N)` state memory. Full paths cost `O(paths*times*N)` state memory and must be requested
explicitly.

## 18. API surface

- `POST /api/pricing/basket`
- `POST /api/pricing/basket/geometric-analytical`
- `POST /api/analytics/basket/compare`
- `POST /api/analytics/basket/convergence`
- `POST /api/analytics/basket/correlation-sensitivity`
- `POST /api/analytics/basket/weight-sensitivity`
- `POST /api/analytics/multi-asset/correlation`

Invalid dimensions, weights, correlations, method combinations, and work budgets produce 4xx
responses. No endpoint serializes normal draws, terminal matrices, or paths.

## 19. Validation strategy

Tests cover input dimensions and finiteness, material PSD rejection, tiny repair, Cholesky/eigen
orientation, singular perfect correlation, valid negative correlation, empirical correlations,
terminal moments, arbitrary-time transitions, antithetic pairing, payoff vectors, analytical
single-asset reduction, geometric MC confidence checks, every arithmetic variance method,
pair-cluster controls, deterministic cases, comparisons, convergence, sensitivities, API limits,
and serialization. Large asset/path cases are marked `performance` and remain opt-in.

## 20. Deliberately deferred work and Phase 13 boundary

Basket Deltas were optional in the specification and are deferred. Phase 3's finite-difference
architecture is single-asset; introducing a partial adapter without vector bump/error semantics
would create an unstable public contract. A later portfolio-Greeks phase can add vector Deltas and
CRN uncertainty coherently. Per-asset Vega and cross-Gamma are also deferred.

Phase 12 contains no Heston process, stochastic variance, stochastic correlation, local
volatility, jumps, copulas, QMC, multiprocessing, or GPU code. Phase 13 may use the same explicit
RNG ownership, factor diagnostics, final-axis factor transform, and process/payoff separation, but
it must add Heston dynamics without changing these constant-volatility basket semantics.
