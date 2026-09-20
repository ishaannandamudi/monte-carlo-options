# Phase 19: Markov chains and Gaussian hidden Markov models

## Purpose and scope

Phase 19 adds a research layer for latent changes in historical return and factor behavior. It
supports finite Markov chains, Gaussian hidden Markov models (HMMs), out-of-sample filtering,
regime forecasts, and regime-conditioned risk. An HMM fitted to historical observations describes
the real-world probability measure, **P**. Its state means are not substituted for risk-neutral
pricing drifts, and this phase does not create switching Heston, switching Merton, or an
arbitrage-free regime-switching derivative pricer.

The core implementation is in `backend/app/regime`. Public, JSON-safe contracts are in
`backend/app/models/regime.py`; the HTTP adapter is in `backend/app/api/regime.py`. The numerical
layer owns NumPy arrays. Public models own validation and serialization.

## Finite Markov chains

`MarkovChain` accepts a finite `K x K` transition matrix for `K >= 2` and optional initial
probabilities. Entries must be finite and nonnegative, and each row must sum to one within
`1e-10`. The stationary solver augments `P.T - I` with the unit-sum equation and solves by least
squares. It reports the nullity of `P.T - I`, whether the stationary distribution is unique, and
whether every state communicates with every other state. Reducible chains therefore never gain a
false uniqueness claim.

`n_step_transition` uses matrix powers. Forecasts compute `p_t P^h`. State persistence is `P_ii`,
and expected duration is `1 / (1 - P_ii)`. An absorbing state's public duration is `null`, with an
explicit `absorbing=true`, because JSON has no portable infinity. Row entropy is descriptive.

## Observation and parameter contracts

`HMMObservationDataset` stores an ordered `T x D` finite matrix, optional ordered timestamps, and
optional unique feature names. Phase 19 rejects missing or non-finite values; preprocessing and
imputation remain outside the HMM. Both univariate and multivariate data are supported.

`GaussianHMMParameters` contains `K`, `D`, initial probabilities, transition probabilities, state
means, covariance matrices, covariance type, and any explicit standardization transform.
`DIAGONAL` is the robust default. `FULL` accepts symmetric positive-semidefinite covariance and the
numerical layer applies the configured eigenvalue floor during fitting.

Gaussian emissions use the standard multivariate log density. Diagonal covariance avoids matrix
solves. Full covariance uses a symmetric eigensystem floor, a stable log determinant, and a linear
solve. Materially indefinite covariance is rejected.

## Forward, backward, smoothing, and decoding

The forward recursion computes log alpha values with `scipy.special.logsumexp`. Its normalized
rows are filtered probabilities: row `t` uses observations only through `t`. The backward
recursion computes log beta from the end of the sample. Normalized `log(alpha) + log(beta)` gives
smoothed probabilities, which use the entire sample and can change when future observations
change.

Viterbi decoding stores only the dynamic-programming scores and predecessor table. NumPy's first
maximum gives deterministic lowest-index tie resolution. It returns the most likely state path and
its log probability.

The one-step filter first predicts with `p P`, then multiplies by the new Gaussian emission in log
space and normalizes. This is an in-process helper and endpoint; Phase 19 contains no streaming
service.

## Baum-Welch estimation

Each EM iteration performs:

1. forward/backward inference;
2. posterior state probabilities (`gamma`);
3. log-normalized adjacent-state probabilities (`xi`);
4. updates of initial probabilities, transition rows, means, and covariance;
5. a new log-likelihood evaluation.

The implementation stores `O(TK)` alpha, beta, filtered, and smoothed arrays. It accumulates the
expected transition counts into one `K x K` matrix instead of retaining an `O(TK^2)` xi tensor.

Convergence uses `delta_log_likelihood <= tolerance * (1 + abs(previous_log_likelihood))`, which
combines useful absolute and relative scaling. A decrease larger than numerical tolerance stops
the start, retains the previous iterate, and records a diagnostic message. The returned likelihood
history contains accepted nondecreasing iterates.

Every updated variance is floored at `min_variance`; full covariance eigenvalues receive the same
floor. Transition entries receive `transition_floor` and rows are renormalized. If a state's
posterior mass is at or below `occupancy_floor * T`, its previous mean and covariance are retained.
This avoids inventing estimates from an empty state while allowing later EM iterations to recover
through nonzero transition probabilities.

## Initialization, multi-start, and labels

`QUANTILE` deterministically partitions observations by the first feature. `RANDOM` chooses seeded
data centers with PCG64 and assigns nearest centers. No global NumPy random state or extra clustering
dependency is used. `SeedSequence` derives an independent deterministic seed for each start. The
first start uses the requested method; later starts use random centers. The fit with the greatest
final log likelihood wins, and every start's seed, convergence, iteration count, likelihood, and
message is returned.

State names carry no market interpretation. After fitting, the default canonical order sorts by
first-feature mean; the alternative sorts by first-feature volatility. Initial probabilities,
transition rows and columns, means, covariance, filtered/smoothed probabilities, and decoded paths
all follow that permutation.

## State and transition diagnostics

Each state summary reports posterior occupancy, posterior mass, Viterbi observation count, mean,
standard deviation, covariance, persistence, duration, stationary probability, and
posterior-weighted first-feature skewness and excess kurtosis. Min/max values use observations
decoded to that state. Volatility is descriptive and states are never assigned subjective names.

The transition report contains `P`, persistence, duration, stationary probabilities, row entropy,
irreducibility, and stationary uniqueness.

## Information criteria and model order

For `K` states and `D` features the free parameter count is:

```text
(K - 1)                         initial probabilities
+ K (K - 1)                     transition probabilities
+ K D                           means
+ K D                           diagonal covariance
```

For full covariance, the last term is `K D (D + 1) / 2`. The result computes
`AIC = 2p - 2 logL` and `BIC = p log(T) - 2 logL`. Model-order comparison fits each caller-supplied
state count with an independent child seed and reports all criteria; it does not declare a true
number of regimes.

## Forecasting, standardization, and walk-forward analysis

For each requested horizon, the forecast returns `p P^h` and the corresponding Gaussian-mixture
mean and covariance:

```text
mu = sum_i p_i mu_i
Sigma = sum_i p_i (Sigma_i + mu_i mu_i') - mu mu'
```

Optional fit standardization stores the training feature mean and standard deviation in the fitted
parameters. Decode and one-step filter apply those same stored values. The separate
`standardize_dataset` helper makes this transform visible to callers. Emission parameters remain in
standardized coordinates when that option is selected; state summaries, forecasts, and regime-risk
moments are converted back to the original feature units.

The bounded walk-forward helper fits only observations before the scoring time, supports expanding
or rolling training windows, and returns out-of-sample filtered probabilities and observation log
likelihood. Refit cadence is explicit. It never uses a future observation when producing a filtered
probability.

## Regime-conditioned risk and existing analytics

`regime_conditioned_risk` accepts factor-aligned Delta exposures and optional diagonal Gamma
exposures. It reports each state's conditional risk. Delta-only conditional VaR and ES use the exact
normal linear-exposure formulas. Diagonal Delta-Gamma conditional risk uses seeded simulation.

Mixture risk samples a state from the supplied probability vector, samples a factor vector from
that state's Gaussian emission, evaluates P&L, converts with the Phase 17 convention
`Loss = -PnL`, and applies the existing empirical VaR/ES estimator. It does not average conditional
VaRs. This preserves tail nonlinearity and serializes `PnL`, `Loss`, `VaR`, and `ES` with existing
signs. Full portfolio repricing can be performed upstream to produce state-contingent P&L samples;
the Phase 19 bounded endpoint supports Delta and diagonal Delta-Gamma factor maps.

The Phase 15 integration helper groups an existing market-fit residual series by externally
inferred state assignments. It does not infer labels from pricing errors or automatically refit
Heston/Merton. The Phase 16 stress-context helper attaches state probabilities to unchanged named
stress output; regime labels never rescale deterministic stress values.

## Real-world switching GBM research helper

`MarkovSwitchingGBMConfig` is deliberately separate from pricing market assumptions. At every step
the simulator transitions the state and draws a conditional log return from that state's specified
real-world drift and volatility. It returns paths and latent states for research. It is not used by
any derivative pricer.

## API and output bounds

The endpoints are:

- `POST /api/regime/hmm/fit`
- `POST /api/regime/hmm/decode`
- `POST /api/regime/hmm/forecast`
- `POST /api/regime/hmm/model-order`
- `POST /api/regime/hmm/filter`
- `POST /api/regime/risk`
- `POST /api/regime/markov/diagnostics`

Environment settings cap observations, features, states, iterations, starts, model-order
candidates, returned probability points, and regime-risk simulations before allocation. Public
responses contain Python lists, never NumPy arrays. Probability output can be final-only, full, or
thinned; full/thinned API output is deterministically downsampled to the configured response cap.
Decode paths remain bounded by the observation cap.

## Limitations and Phase 20 boundary

Phase 19 provides Gaussian emissions only; Student-t emissions were optional and are deferred.
There is no missing-data likelihood, covariate-dependent transition model, Bayesian inference,
production streaming, trading signal, semantic state naming, or automatic regime-specific model
calibration. Exact absorbing transitions are supported by Markov diagnostics and inference;
fitting applies a positive transition floor.

Phase 20 may add its own explicitly specified work. Phase 19 does not add Sobol or other
quasi-Monte Carlo methods, GPU or multiprocessing execution, HFT integration, deep learning,
switching Heston/Merton, rough volatility, or a frontend.
