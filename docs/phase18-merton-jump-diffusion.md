# Phase 18: Merton jump diffusion

## 1. Scope and module boundaries

Phase 18 adds discontinuous returns to the existing constant-market European option foundation.
`models/merton.py` owns validated parameters, controls, and serialized results;
`simulation/merton.py` owns state generation; `pricing/merton.py` owns payoff valuation and Greeks;
`analytics/merton.py` owns experiments; and `api/merton.py` only validates resource budgets and
delegates. The model does not include stochastic volatility, regime changes, or calibration state.

## 2. Risk-neutral model

Log jump size is `Y ~ N(mu_j, sigma_j²)` and the multiplicative stock jump is `J=exp(Y)`. With an
independent Brownian motion `W` and Poisson process `N` of intensity `lambda`,

```text
dS_t/S_(t-) = (r-q-lambda*kappa_J)dt + sigma dW_t + (J-1)dN_t
kappa_J = exp(mu_j + 0.5 sigma_j²) - 1.
```

The drift compensation is part of every simulation and analytical term. It gives the risk-neutral
identity `E[S_T]=S_0 exp((r-q)T)`. `MarketEnvironment.volatility` is diffusion volatility;
`MertonJumpParameters` contains only `jump_intensity`, `jump_mean`, and `jump_volatility`.

## 3. Parameter and numerical policy

Intensity and jump volatility are finite and nonnegative; jump mean is any finite real number.
The model computes the compensator with `expm1` and rejects values outside floating-point range.
Rates and dividend yields may be negative. Derived exponential arrays pass through the shared
finite numerical helpers. No valid request silently returns NaN or infinity.

At zero maturity, both pricers return exact intrinsic value, simulation consumes no random draws,
and jump counts are zero. With zero intensity, analytical pricing calls the existing BSM function
directly. Zero jump volatility produces the deterministic jump size `exp(mu_j)`. Zero diffusion
volatility does not remove Poisson or jump-size randomness.

## 4. Conditional terminal law

Conditional on `N_T=n`,

```text
log S_T ~ Normal(m_n, v_n)
m_n = log S_0 + (r-q-lambda*kappa_J-0.5 sigma²)T + n mu_j
v_n = sigma²T + n sigma_j².
```

This identity is shared conceptually by simulation tests and the analytical mixture. It avoids an
ambiguous effective-rate or effective-dividend parameterization.

## 5. Terminal and full-path simulation

Terminal simulation draws a vector of Poisson counts, a vector of diffusion normals when needed,
and a vector of aggregate-jump normals when needed. Conditional aggregate log jump is
`n*mu_j + sqrt(n)*sigma_j*Z_j`. Thus memory is O(paths), and there is no Python loop over paths or
jumps. `terminal_prices_from_drivers` accepts caller-owned counts and both normal arrays for exact
unit tests.

Optional full paths have shape `(paths, steps+1)`. Each regular interval draws its exact Poisson
count, so counts greater than one are preserved. One loop advances time; all paths are vectorized.
The returned interval-count matrix exists for research diagnostics and tests. European pricing uses
terminal simulation and does not allocate full paths.

Merton Phase 18 does not label its draws antithetic. A symmetric transformation for Poisson counts
is not canonical, and pairing only Gaussian components can create a clustered estimator whose
benefit depends on the payoff and sharing policy. The baseline therefore uses independent exact
draws and a conventional independent-path standard error.

## 6. Conditional-lognormal pricing and mixture

`price_lognormal_terminal_distribution` discounts a call or put expectation directly from log mean
and log variance. At zero log variance it discounts the deterministic intrinsic payoff. Otherwise
it uses the standard truncated-normal moments. Merton price is

```text
V = sum(n >= 0) PoissonPMF(n; lambda*T) * V_lognormal(m_n, v_n).
```

SciPy evaluates Poisson probabilities without factorial overflow. The requested cumulative upper
tail tolerance selects the last included count through the Poisson quantile. `max_terms` is a hard
safety cap. Results report `terms_used`, exact survival-function omitted mass, and whether that mass
meets tolerance. A deliberately small cap returns a finite partial sum with `converged=false`.

The default `1e-12` omitted-mass tolerance makes call/put parity accurate to the same truncation
scale. At zero intensity, direct BSM dispatch gives an exact library-level limit rather than a
numerically close mixture.

## 7. Monte Carlo estimator and diagnostics

The Monte Carlo estimator is the mean discounted vanilla payoff. Its Bessel-corrected sample
standard deviation divided by `sqrt(paths)` gives standard error; the existing normal-quantile
helper produces the requested confidence interval. Result statistics include terminal mean and
standard deviation, strict ITM probability, expected and sample jump-count moments, zero/positive
jump fractions, maximum observed count, seed, scheme, and timing. Arrays are never serialized.

Jump diagnostics compare sample count mean, sample variance, and zero-event frequency with
`lambda*T`, `lambda*T`, and `exp(-lambda*T)`. These are distribution summaries, distinct from
pricing-estimator uncertainty.

## 8. Research analytics

Comparison reports analytical, Monte Carlo, BSM, and optional Heston prices without ranking the
models. MC error includes a z-score only when SE is positive. Path convergence derives independent,
reproducible child seeds from a root `SeedSequence` and estimates the log(SE)-on-log(N) slope from
positive-SE points. Series convergence varies tail tolerance and exposes term and omitted-mass
behavior.

Smile analytics price call slices and invoke the Phase 4 BSM IV solver, returning each inversion
status rather than manufacturing a volatility after failure. Short-dated jump smiles are a model
output whose direction depends on jump mean, dispersion, intensity, strike, and maturity.
Sensitivity sweeps support intensity, jump mean, jump volatility, and diffusion volatility.

## 9. Greeks and calibration boundary

Analytical-price finite differences provide Delta, Gamma, diffusion Vega, calendar-time Theta, and
Rho using Phase 3 unit conventions. Separate fields name derivatives with respect to intensity,
jump mean, and jump volatility; they are never called standard Vega. Near nonnegative boundaries,
the implementation uses a forward difference.

`merton_price_objective` returns raw and square-root-weighted price residuals plus their squared
sum. This is enough for a later bounded optimizer to reuse stable prices. Phase 18 does not add a
second calibration framework, infer jumps from returns, or modify Phase 14 Heston calibration.

## 10. Model comparison and market fit

Phase 15 registers `MERTON_ANALYTICAL` and `MERTON_MONTE_CARLO`. Both support European calls, puts,
continuous dividends, and jump diffusion; neither claims American exercise. The analytical engine
advertises standard Greeks, and the Monte Carlo engine reports uncertainty. Missing explicit Merton
parameters produce a structured exclusion. Cross-model diagnostics include analytical-versus-MC
and BSM-versus-Merton differences.

Market fit reuses Phase 14's normalized dataset preparation, observation weights, BSM IV inversion,
and Phase 15 expiry/moneyness/smile residual summaries. The configured flat BSM volatility becomes
Merton diffusion volatility. The fit is descriptive and does not optimize Merton parameters.

## 11. Portfolio and risk integration

Phase 16 European vanilla positions may select either Merton engine and carry explicit Merton
parameters, simulation controls, and series controls. Generic finite differences aggregate the five
standard Greeks; their Vega changes diffusion volatility only. `MertonScenario` accepts additive
intensity, mean, and jump-volatility shifts or a complete override. Per-position overrides support
heterogeneous books. Parameter scenarios force full repricing and deliberately mark the ordinary
Greek P&L approximation unavailable.

`realized_jump_log_size` is a deterministic spot stress `S_new=S_old*exp(jump_log_size)`. It is not
a Poisson draw or a parameter change. Phase 17 adds explicit position-targeted mappings for the
three Merton parameters. Full revaluation uses supplied histories; Greek modes require explicit
exposures. There is no automatic jump identification, return classification, or calibration.

## 12. API resource controls

The pricing API exposes analytical and Monte Carlo routes. Analytics expose comparison, path and
series convergence, smile, sensitivity, jump diagnostics, and Greeks. Limits are environment
controlled: `MAX_MERTON_PATHS`, `MAX_MERTON_SERIES_TERMS`,
`MAX_MERTON_CONVERGENCE_POINTS`, `MAX_MERTON_SMILE_POINTS`,
`MAX_MERTON_SENSITIVITY_POINTS`, and `MAX_MERTON_PATH_NODES`. Full paths remain a library research
operation, but the node cap documents the allowed future HTTP budget. Model-comparison and
portfolio routes enforce the same path and term caps before allocation.

## 13. Performance and reproducibility

All generators are caller-owned NumPy `Generator(PCG64)` instances. Same inputs and seed reproduce
terminal spots, jump counts, prices, and diagnostics on the supported numerical environment.
Different convergence points use deterministic independent child streams. Analytical work scales
with the required Poisson terms; terminal Monte Carlo scales linearly in paths with several
one-dimensional arrays. The benchmark report records local timings and approximate array memory.

## 14. Deferred work and Phase 19 boundary

Phase 18 implements only Merton's independent lognormal compound-Poisson jumps with constant
diffusion volatility. Bates, Heston-with-jumps, Kou, Variance Gamma, CGMY, NIG, stochastic rates,
local/rough volatility, QMC, parallel/GPU execution, and automated Merton calibration are absent.
Regime switching and hidden Markov models belong to Phase 19 or a later explicit specification and
require no replacement of the parameter, process, pricing, comparison, or portfolio interfaces
introduced here.
