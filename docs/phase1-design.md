# Phase 1 Architecture and Numerical Design

## 1. Scope and design goal

Phase 1 establishes a small quantitative kernel whose behavior is explicit, reproducible, and
testable. It prices European calls and puts under risk-neutral GBM with Monte Carlo and
Black-Scholes-Merton. The architecture separates the object being priced, market assumptions, the
stochastic process, payoff transformation, estimator, analytical benchmark, experiments, and HTTP
delivery.

The separation is deliberately narrow. It creates stable seams for later research without adding
unused trees, path containers, calibration graphs, portfolio objects, or execution backends.

## 2. Module boundaries

| Module | Responsibility | Explicitly does not own |
|---|---|---|
| `models.option` | Contract enum and validated continuous contract inputs | Rates, volatility, simulation behavior |
| `models.market` | Constant risk-free rate, volatility, dividend yield | Contract and payoff terms |
| `models.simulation` | Path count, seed, PCG64 choice, variance reduction, CI level | Hidden global defaults |
| `models.results` | Immutable finite API/library results | Raw simulation arrays |
| `simulation.base` | Terminal-process interface | Payoff and pricing rules |
| `simulation.gbm` | Risk-neutral terminal GBM state | Discounting and payoff evaluation |
| `simulation.random` | Generator construction and child seed derivation | Global NumPy state |
| `payoffs.vanilla` | Vectorized call/put payoff and intrinsic value | Discounting and sampling |
| `analytics.statistics` | Sample moments, SE, antithetic SE, normal CI | Stochastic simulation |
| `pricing.monte_carlo` | Discounted-payoff estimator and diagnostics | HTTP resource policy |
| `pricing.black_scholes` | Analytical value under identical assumptions | Monte Carlo correction |
| `pricing.comparison` | Error and z-score diagnostics | Model calibration |
| `analytics.convergence` | Independent repeated experiments and slope fit | API limits |
| `api` and `main` | Validation composition, allocation limits, logs, HTTP | Quantitative formulas |

## 3. Contract and market separation

`OptionContract` holds `option_type`, the retained `EUROPEAN` exercise style, spot, strike, and
maturity. `MarketEnvironment` holds the continuously compounded risk-free rate, volatility, and
continuous dividend yield. Volatility is non-negative; the two rates may be negative.

This boundary matters because a contract is stable across market scenarios, while a market
environment can be replaced without mutating contract terms. Both models reject unknown fields,
NaN, and infinity. Spot and strike are positive, and maturity is non-negative. Values are
continuous quantitative inputs rather than accounting amounts.

## 4. Stochastic-process abstraction

`StochasticProcess.simulate_terminal(contract, market, config, rng)` returns a one-dimensional
`float64` array with exactly the configured number of observations. `GBMProcess` implements:

```text
S_T = S_0 exp((r - q - sigma^2/2)T + sigma sqrt(T) Z)
```

The process neither sees payoff type nor performs discounting. A later terminal Heston or jump
process could satisfy the same interface. Full-path work will add a separate path-oriented
interface so terminal-only code stays simple and existing processes remain valid.

GBM exposes the validation moment:

```text
E[S_T] = S_0 exp((r - q)T)
```

Tests compare a fixed-seed sample mean with this value using a tolerance measured in sample
standard errors.

## 5. Payoff and pricing abstractions

Payoffs are vectorized pure transformations:

```text
call = max(S_T - K, 0)
put  = max(K - S_T, 0)
```

Equality at the strike has zero payoff and is not in the money. Discounting is excluded from this
layer.

The pricing boundary maps a contract and market, plus engine-specific configuration, to a typed
result. `BlackScholesPricer` is an analytical `Pricer`; `MonteCarloPricer` additionally requires
explicit simulation configuration and accepts a stochastic process through constructor injection.
This is enough composition for Phase 1. Future tree or regression engines can implement their own
configuration types while preserving the contract/market/result boundary.

## 6. RNG ownership and seed policy

Each standalone Monte Carlo call owns a newly constructed `numpy.random.Generator` backed
explicitly by `numpy.random.PCG64`. PCG64 receives a user seed through NumPy's `SeedSequence`
machinery. No function calls `numpy.random.seed` or a module-level sampling function. This prevents
unrelated code from advancing a shared stream.

The same contract, market, simulation configuration, seed, implementation, and supported numerical
environment produce exactly the same terminal vector and quantitative result fields. `runtime_ms`
is an observation of wall-clock execution and is intentionally not reproducible.

For convergence, the engine constructs `SeedSequence(root_seed)` and calls `spawn(point_count)`.
Each child generates a `uint64` seed used by that point's PCG64 generator. Point order is therefore
part of the experiment definition. Repeating the same ordered counts and root seed reproduces every
stream; different points do not reuse prefixes from one generator. The hierarchy also supplies a
clean deterministic stream-derivation pattern for later parallel work, although Phase 1 does not
run in parallel.

## 7. Monte Carlo estimator

For each terminal value, the pricer forms

```text
X_i = exp(-rT) payoff(S_T_i)
price = (1/N) sum_i X_i
```

`payoff_variance` is the Bessel-corrected sample variance of individual discounted `X_i` values,
and `discounted_payoff_std` is its square root. For plain independent Monte Carlo:

```text
SE = sample_std(X, ddof=1) / sqrt(N)
```

Terminal mean, terminal sample standard deviation, and strict ITM frequency are descriptive
simulation diagnostics. Raw arrays are local temporaries and never appear in result models or API
responses.

A stochastic singleton has no degree of freedom for estimating dispersion, so pricing raises a
clear `StatisticalError` instead of returning NaN or false zero uncertainty. A genuinely
deterministic GBM case (`T = 0` or `sigma = 0`) has exact zero uncertainty even with one outcome.
The API defaults to 100,000 outcomes.

## 8. Antithetic estimator treatment

The GBM generator places pair members consecutively:

```text
Z_1, -Z_1, Z_2, -Z_2, ...
```

For `N = 2m`, define discounted pair averages

```text
Y_j = (X_(2j-1) + X_(2j)) / 2
```

The price is exactly `mean(Y)`. Pairs are independent even though observations within a pair are
correlated. The estimator uncertainty is therefore

```text
SE^2 = sample_variance(Y, ddof=1) / m
```

Computing `sample_std(X)/sqrt(N)` here would incorrectly assume all antithetic payoffs are
independent and is explicitly forbidden by the implementation and direct tests.

For `N = 2m + 1`, Phase 1 draws `m + 1` independent primary normals. The first `m` produce complete
pairs; the last primary draw produces one terminal outcome. Exactly `N` outcomes are returned, and
the estimator remains the mean of all outcomes. Let cluster `j` have total discounted payoff `T_j`,
size `n_j` (two for a pair and one for the final singleton), and let `M = m + 1`. With
`mu_hat = sum(T_j)/N`, the finite-sample independent-cluster estimate is

```text
u_j  = T_j - n_j mu_hat
SE^2 = [M/(M-1)] sum_j u_j^2 / N^2
```

For equal pair sizes this reduces algebraically to the pair-average formula above. For an odd count
it preserves the singleton's one-path weight rather than pretending that a singleton mean and a
pair mean have equal contributions. Fewer than two independent clusters have no observable
between-cluster dispersion and raise a clear `StatisticalError`, unless the GBM case is genuinely
deterministic and therefore has exact zero uncertainty.

Antithetic variance reduction is payoff- and parameter-dependent. A representative vanilla case
is tested for reduction, but the suite does not assert universal dominance.

## 9. Confidence intervals and comparison diagnostics

For requested confidence level `c`, SciPy supplies

```text
z = NormalPPF((1 + c)/2)
CI = price +/- z SE
```

Zero SE returns `[price, price]` directly. All interval components must be finite.

Comparison computes absolute error and, when the analytical price is nonzero, relative error. If SE
is positive, it reports `(MC - BS)/SE`; otherwise z-score is unavailable. Analytical zero also
produces unavailable relative error instead of division by zero. Confidence containment is an
inclusive comparison against the Monte Carlo interval.

## 10. Black-Scholes benchmark role and edge policies

Black-Scholes-Merton uses continuous dividend yield and SciPy's normal CDF. It is the analytical
truth under Phase 1 assumptions and validates simulation bias, estimator scaling, put-call parity,
and convergence. It never adjusts the Monte Carlo result.

Numerical branches are exact rather than tolerance hacks:

- `T = 0`: return reusable intrinsic value.
- `sigma = 0`: return the discounted deterministic risk-neutral terminal payoff, equivalently
  `max(S exp(-qT) - K exp(-rT), 0)` for a call and the reverse for a put.
- Positive `sigma*sqrt(T)` that underflows to representational zero uses the same deterministic
  limit rather than dividing by zero.
- Extremely small positive time or volatility otherwise uses the formula directly.
- Negative rates and negative dividend yields use the same continuous-compounding equations.
- Deep ITM/OTM CDF values may saturate at zero or one, which is the stable limiting behavior.
- Derived exponentials, terminal values, and results are checked. Overflow, non-finite values, and
  terminal underflow to non-positive values raise `NumericalError`.

## 11. Convergence experiment architecture

`run_convergence` computes Black-Scholes once, derives one child seed per requested path count, and
runs ordinary library pricing for each point. A point records path count, child seed, Monte Carlo and
analytical prices, absolute and safe relative error, SE, CI width, and runtime.

The only Python loop is over experiment points; individual paths remain vectorized NumPy operations.
The slope function filters non-positive or non-finite SE values, then fits
`log(SE) = a + b log(N)` with `numpy.polyfit`. It returns the slope and number of usable points. With
fewer than two usable points, slope is unavailable.

## 12. API and library separation

The library accepts any validated positive path count permitted by available memory. FastAPI owns
deployment safety through `MAX_API_SIMULATIONS` and `MAX_CONVERGENCE_POINTS`. Every convergence
point is checked against the simulation limit. Mathematical inputs, including confidence level and
seed, remain explicit request fields and are never taken from hidden environment configuration.

Pydantic composes nested request validation and rejects extra fields. Quantitative domain failures
become HTTP 422 responses with clear details. The API logs request summaries, simulation counts,
variance-reduction modes, convergence start/completion, and runtime without logging arrays.

The optional `batch_size` field is validated but is an inert execution hint in Phase 1. The engine
uses one vectorized allocation, so the field cannot alter RNG consumption or numerical results.
Later memory optimization can honor it only after preserving pairing and deterministic stream
equivalence.

## 13. Validation strategy

Deterministic tests cover input rejection, payoff vectors, equality at strike, maturity zero,
volatility zero, negative rates, dividends, and resource limits. Numerical tests cover known
Black-Scholes call/put benchmarks, put-call parity, deep moneyness, tiny time/volatility, same- and
different-seed behavior, strict positivity, and theoretical GBM mean.

Statistical tests use fixed seeds and errors scaled by the estimator's own standard error. Direct
fixtures distinguish pair-level antithetic SE from the incorrect individual-path formula. Ordinary
tests use moderate vector sizes and complete in CI-friendly time.

## 14. Intentionally deferred abstractions

Phase 1 contains no market-data adapters, volatility surfaces, calibration protocols, path payoff
state, early-exercise policy, regression layer, tree lattice, portfolio aggregate, correlated factor
model, compute scheduler, cache, persistence, or frontend. Creating these without a concrete phase
would make ownership less clear and force premature interface commitments.

## 15. Future extension paths

- **Greeks and implied volatility:** consume existing typed pricers and results through new analysis
  services; do not put sensitivities into GBM or payoff functions.
- **Full-path simulation and exotics:** add a path-process interface and path-aware payoff protocol.
  Terminal processes and vanilla vector payoffs remain unchanged.
- **Binomial trees:** implement a new pricing engine over the same contract and market models. Add
  exercise policy only when American contracts enter scope.
- **Longstaff-Schwartz:** combine full paths, an explicit exercise policy, and a regression service;
  it need not alter analytical or terminal Monte Carlo engines.
- **Heston and jump models:** implement process-specific parameter models and stochastic-process
  implementations. A terminal-only implementation may plug directly into `MonteCarloPricer`; a
  path implementation uses the future path interface.
- **Multi-asset and basket work:** introduce asset-vector contract and market models when their
  dimensional requirements are known, while reusing RNG stream ownership and statistical helpers.
- **Advanced variance reduction and quasi-Monte Carlo:** add explicit configuration enum values and
  estimator implementations, each with uncertainty rules appropriate to its dependence structure.
- **Parallel or GPU execution:** derive deterministic SeedSequence children and preserve estimator
  clusters. No concurrency assumption is embedded in Phase 1.

These are integration directions, not Phase 1 implementations.
