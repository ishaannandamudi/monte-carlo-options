# Phase 20: quasi-Monte Carlo numerical integration

## Scope and architecture

Phase 20 adds a numerical sampling layer; it does not add a financial model. Existing
pseudorandom APIs still default to PCG64. New QMC endpoints explicitly select pseudorandom,
deterministic Sobol, or scrambled Sobol sampling and inject their normal drivers into the existing
GBM and correlated-GBM transition functions.

Public contracts live in `backend/app/models/qmc.py`. Sobol, transformations, Brownian path
construction, and diagnostics live in `backend/app/simulation/qmc.py`. Product estimators are in
`backend/app/pricing/qmc.py`; convergence research is in `backend/app/analytics/qmc.py`; HTTP
resource checks are in `backend/app/api/qmc.py`.

## Monte Carlo and low-discrepancy integration

Ordinary Monte Carlo samples independent random points and supplies classical path-level standard
errors for integrable payoffs. QMC replaces those points with a deterministic low-discrepancy net
that fills the integration cube more evenly. Phase 20 uses `scipy.stats.qmc.Sobol`, relying on the
direction numbers and scrambling supplied by the installed SciPy version rather than maintaining a
local Sobol implementation.

The supported dependency range is SciPy `>=1.13,<2`; validation respects SciPy Sobol's 21,201
dimension ceiling and its default 30-bit maximum sequence length. Benchmark values record the
installed numerical environment and should not be treated as bitwise cross-version guarantees.

Deterministic Sobol is useful for integration and convergence research, but one deterministic net
does not define a classical sampling variance. Its `standard_error` and confidence interval fields
are therefore `null`.

Randomized QMC uses independent seeded Sobol scrambles. For replicate estimates
`theta_1,...,theta_R`, the reported estimate and uncertainty are

```text
theta_hat = mean(theta_r)
SE(theta_hat) = sample_std(theta_r, ddof=1) / sqrt(R).
```

At least two scrambles are required. Confidence intervals use a normal quantile and this
replicate-level SE. The implementation processes scrambles sequentially and retains only estimates
and compact diagnostics, not every replicate's paths.

## Configuration and reproducibility

`QMCConfig` records method, dimension, requested samples, scramble flag, seed, randomization count,
skip, power-of-two policy, inverse-normal transform, path construction, and confidence level.
Its default remains one PCG64 pseudorandom sample set; Sobol or scrambled Sobol must be selected.
`SeedSequence` children provide distinct reproducible scramble seeds. Equal complete requests
produce identical points, estimates, and reproducibility signatures on the supported SciPy/NumPy
environment. The signature includes contract, market, and complete QMC configuration, so Phase 15
compute-frontier records distinguish numerical methods without registering QMC as new dynamics.

Sobol balance properties are strongest at `N=2^m`. Policies are explicit:

- `REQUIRE_POWER_OF_TWO` rejects other counts;
- `ROUND_UP` records the request but uses the next power of two;
- `ALLOW_ARBITRARY` preserves the exact count and accepts weaker net guarantees.

With no skip and a power-of-two count, generation uses `random_base2` exactly once. With explicit
fast-forward or arbitrary counts it uses `Sobol.fast_forward` followed by one `random` call. Skip is
useful for Sobol sequence research and is not described as an independent randomization;
pseudorandom configurations reject a nonzero Sobol offset.

## Uniform-to-normal transformation

Sobol points lie in `[0,1)`, including zero in an unscrambled net. Before applying
`scipy.special.ndtri`, coordinates are clipped to
`[eps, 1-eps]`, where `eps = numpy.finfo(float64).eps`. This prevents infinite normal values while
changing only floating-point endpoint representations. Generated normal matrices always have shape
`(effective_samples, dimension)`.

## Dimension mapping and terminal pricing

European terminal GBM uses dimension one. A terminal `A`-asset basket uses dimension `A`; the
independent normal matrix is passed to the existing Phase 12 correlation factorization and terminal
GBM transition. Phase 20 does not duplicate correlation repair or factorization.

That asset-axis mapping is the only multi-factor QMC mapping in Phase 20. A later Heston extension
would require `2 * time_steps` coordinates plus an explicit interleaved or factor-major layout for
variance and independent spot drivers. Phase 20 does not guess that ordering or apply a scalar
Brownian bridge across the correlated Heston factors.

European calls and puts use the Black-Scholes-Merton price as their reference. Geometric baskets
use the existing exact lognormal basket reference. Arithmetic baskets may accept a clearly labeled
caller-supplied numerical reference. Arithmetic baskets can use the existing geometric payoff as a
control variate, but RQMC uncertainty still comes only from adjusted replicate estimates.

## Path-dependent construction

For monitoring times `0 < t_1 < ... < t_M`, STANDARD construction maps Sobol coordinate `i`
chronologically to the standardized increment over `(t_(i-1), t_i]`.

BROWNIAN_BRIDGE assigns coordinate zero to `W_T`, then recursively fills the observation time
nearest each interval midpoint. Given assigned endpoints, it uses

```text
E[W_t | W_l,W_r] = ((t-l)/(r-l)) W_r + ((r-t)/(r-l)) W_l
Var[W_t | W_l,W_r] = (t-l)(r-t)/(r-l).
```

The deterministic terminal-first coordinate order is public diagnostic metadata. Brownian values
are differenced and divided by `sqrt(delta_t)` before reuse by the existing exact GBM transition.
This QMC path-construction bridge is separate from Phase 11's barrier-crossing correction, which
conditions on already simulated endpoints.

PCA construction forms `C_ij=min(t_i,t_j)`, sorts its eigenvalues descending, and maps
`W = Q sqrt(Lambda) Z`. Eigenvector signs are canonicalized for reproducibility. Diagnostics return
the deterministic coordinate order and cumulative variance concentration for the supplied
observation times (or a normalized equally spaced grid in generic sampling diagnostics). This is a practical effective-dimension
diagnostic; it is not claimed to be ANOVA effective dimension.

## Asian pricing

Discrete arithmetic and geometric Asians reuse their Phase 10 monitoring schedules, exact GBM
observation transition, average functions, and payoff functions. STANDARD, BROWNIAN_BRIDGE, and PCA
are supported. Geometric Asians use the exact analytical discrete-average price as a reference.
Arithmetic Asians can use the geometric option as a within-replicate control variate and may carry
a caller-supplied numerical reference label.

## Convergence and compute frontiers

European convergence compares PCG64 pseudorandom Monte Carlo, deterministic Sobol, and scrambled
Sobol for `2^minimum_power` through `2^maximum_power`. Every point reports estimate, analytical
error, relative error, valid SE where available, runtime, and
`absolute_error * sqrt(runtime_seconds)`. The last metric is descriptive, not a universal optimizer.

Log-log least squares estimates error slopes from positive errors and SE slopes from available
positive SE values. No product is required to attain a hard-coded theoretical rate. The same points
form the Phase 15-style compute frontier, with QMC represented as sampling metadata on risk-neutral
GBM rather than a new financial model.

## Diagnostics, memory, and safeguards

Sampling diagnostics report bounds, mean, normal moments, scramble seed provenance, bounded first
points, and SciPy centered discrepancy only through 4,096 samples, 32 dimensions, and 65,536
coordinates. Larger requests omit discrepancy to avoid its much greater work.
API settings cap samples, dimension, replicates, total `samples * dimension * replicates`, and
convergence points before numerical allocation. Replicates are sequential. A complete individual
Sobol net is generated at once so chunk boundaries cannot restart or corrupt sequence semantics.

## Limitations

Smooth expectation problems often benefit from QMC, but improvement is problem and dimension
dependent. Digital and barrier payoffs are discontinuous, and VaR/ES are tail quantiles; Phase 20
does not promise improved convergence for them. The existing Phase 17 pseudorandom risk simulator
is unchanged.

Heston FTE QMC was considered but deferred because its two-factor time-step driver interface would
require a separate simulation implementation or invasive process changes. Barrier QMC and digital
QMC are also deferred rather than duplicating their established pricing and bridge-correction
logic. There is no QMC Merton inversion, multiprocessing, GPU, or distributed implementation.

## Phase 21 boundary

Phase 20 returns reproducible signatures and numerical diagnostics in each response but creates no
experiment database, artifact lineage, run registry, or tracking service. Those belong to Phase 21
only when separately specified.
