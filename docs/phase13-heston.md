# Phase 13: Heston stochastic volatility

## Scope and module boundaries

Phase 13 adds risk-neutral Heston simulation, European Monte Carlo pricing, stable
characteristic-function pricing, and research diagnostics. It reuses `OptionContract` for spot,
strike, maturity, payoff direction, and exercise style, and `MarketEnvironment` for the continuously
compounded risk-free rate and continuous dividend yield. The market object's constant-volatility
field is ignored by Heston engines; variance-process assumptions live only in `HestonParameters`.

- `models/heston.py` owns parameter, configuration, enum, and result schemas.
- `simulation/heston.py` owns drivers, FTE, QE-M, terminal simulation, and full paths.
- `pricing/heston.py` owns discounted-payoff Monte Carlo estimation.
- `pricing/heston_analytical.py` owns the characteristic function and Fourier inversion.
- `analytics/heston.py` owns comparisons, convergence, variance diagnostics, smiles, and sweeps.
- `api/heston.py` is a bounded HTTP adapter and contains no numerical algorithm.

Calibration is deliberately absent. Parameters supplied to this phase are already risk-neutral
parameters under the pricing measure `Q`.

## Model and parameters

Under `Q`, with constant `r` and `q`,

```text
dS_t = (r-q) S_t dt + sqrt(v_t) S_t dW_t^S
dv_t = kappa(theta-v_t) dt + sigma_v sqrt(v_t) dW_t^v
corr(dW_t^S, dW_t^v) = rho.
```

`v0` is initial variance, `kappa` mean-reversion speed, `theta` long-run variance, `sigma_v`
volatility of variance, and `rho` the instantaneous spot/variance Brownian correlation. The first
four are finite and nonnegative; `rho` is finite and lies in `[-1,1]`. Boundary values are valid.

The Feller diagnostic is

```text
feller_satisfied = 2 kappa theta >= sigma_v^2
feller_ratio     = 2 kappa theta / sigma_v^2.
```

The ratio is unavailable when `sigma_v=0`. A violation is reported and never rejected because
useful market parameter sets often violate this sufficient strict-positivity condition.

## Brownian drivers and reproducibility

Each step draws independent PCG64 normals `Z1,Z2` and sets

```text
Z_v = Z1
Z_s = rho Z1 + sqrt(max(1-rho^2,0)) Z2.
```

This orientation is tested empirically, including exact `rho=+1` and `rho=-1`. The square-root
guard prevents a tiny negative argument at a correlation endpoint. Callers own an isolated
`numpy.random.Generator`; global NumPy random state is never used. A complete input and seed
therefore reproduces terminal states and prices on the same supported numerical environment.

With antithetic sampling, paths are interleaved as `(Z1,Z2),(-Z1,-Z2)`. QE also pairs `U` with
`1-U`. An odd request returns exactly the requested number of paths and leaves the last draw as an
independent singleton. Reported Monte Carlo SE uses independent pair clusters, plus the existing
weighted singleton treatment for odd counts. Antithetic sampling is available for both schemes but
is not claimed to reduce variance for every parameter set.

## Full-truncation Euler

The canonical FTE convention retains the raw variance state. Given `v_plus=max(v,0)`,

```text
log S_next = log S + (r-q-0.5 v_plus) dt + sqrt(v_plus dt) Z_s
v_next     = v + kappa(theta-v_plus) dt + sigma_v sqrt(v_plus dt) Z_v.
```

The raw `v_next` may be negative. The following step applies its positive part in every drift or
diffusion coefficient, so no negative square root occurs. Terminal option results expose a
nonnegative effective variance, while full-path research diagnostics retain raw FTE states and
report the fraction of negative proposals. This storage distinction is intentional.

## Andersen QE-M

For current nonnegative variance `v` and `kappa>0`, the exact CIR conditional moments are

```text
e   = exp(-kappa dt)
m   = theta + (v-theta)e
s^2 = v sigma_v^2 e(1-e)/kappa
      + theta sigma_v^2 (1-e)^2/(2 kappa)
psi = s^2/m^2.
```

`-expm1(-kappa dt)` evaluates `1-e` accurately. At `kappa=0`, the continuous limit is
`m=v` and `s^2=sigma_v^2 v dt`; this makes QE valid rather than rejecting the boundary.

For `psi <= 1.5`, the quadratic-Gaussian branch uses

```text
b^2    = 2/psi - 1 + sqrt((2/psi)(2/psi-1))
a      = m/(1+b^2)
v_next = a(sqrt(b^2)+Z_v)^2.
```

For `psi > 1.5`, the point-mass/exponential branch uses

```text
p    = (psi-1)/(psi+1)
beta = (1-p)/m
v_next = 0                                      when U <= p
v_next = log((1-p)/(1-U))/beta                  when U > p.
```

Both branches are tested against their target conditional mean and variance. Zero conditional
variance is handled exactly.

The spot update is Andersen's martingale-corrected QE-M update with `gamma1=gamma2=1/2`. Define

```text
R  = rho/sigma_v
K1 = 0.5 dt(kappa R - 0.5) - R
K2 = 0.5 dt(kappa R - 0.5) + R
K3 = K4 = 0.5 dt(1-rho^2)
A  = K2 + 0.5 K4.
```

The implementation evaluates the selected QE branch's conditional log-MGF `log M_vnext(A)` and
sets

```text
K0* = -(K1+0.5K3)v - log M_vnext(A)
log S_next = log S + (r-q)dt + K0* + K1 v + K2 v_next
             + sqrt(K3 v + K4 v_next) Z2.
```

This makes the conditional gross return a martingale. If the required MGF does not exist for an
extreme coarse step, the engine raises a clear error and directs the caller to increase the step
count. It never emits an invalid value. When `sigma_v=0`, both schemes use the exact deterministic
variance transition and exact integrated variance over each step, avoiding division by zero.

## Terminal and full-path simulation

European pricing keeps only current log spot and variance plus step drivers. Memory is `O(paths)`;
there is no hidden path matrix. Diagnostic simulation returns float64 arrays with shapes
`(paths, steps+1)` for spot and variance. Its state memory is approximately
`16 * paths * (steps+1)` bytes before temporary vectors. Uniform time steps are used.

At `T=0`, every path equals the initial spot and variance. The deterministic `sigma_v=0` variance
path is

```text
v(t) = theta + (v0-theta) exp(-kappa t),
```

with the `kappa=0` limit `v(t)=v0`. Valid finite outputs are checked after updates and exponentiation.

## Monte Carlo estimator

Calls and puts use the existing vectorized terminal payoff, discounted by `exp(-rT)`. Price, SE,
normal confidence interval, strict ITM probability, terminal spot/variance summaries, runtimes,
scheme, seed, and Feller diagnostics are returned. Plain SE treats paths as independent.
Antithetic SE treats pair averages as independent clusters. Discretization bias is separate from
the reported sampling error and is measured against the analytical price.

## Characteristic function and branch stability

For complex `u`, define

```text
beta = kappa - rho sigma_v i u
d    = sqrt(beta^2 + sigma_v^2(u^2+i u)), with Re(d) >= 0
g    = (beta-d)/(beta+d)
E    = exp(-dT)
C    = i u(r-q)T + (kappa theta/sigma_v^2)
       [(beta-d)T - 2 log((1-gE)/(1-g))]
D    = ((beta-d)/sigma_v^2)(1-E)/(1-gE)
phi(u) = exp(i u log(S0) + C + D v0).
```

This is the branch-consistent Little Heston Trap representation. Choosing the square-root branch
with nonnegative real part and using the reciprocal-stable `g` form avoids the discontinuous
logarithm behavior of the naive representation. Every complex result and denominator is checked.

When variance is deterministic, the engine bypasses the stochastic formula. It computes

```text
I = integral_0^T v(t)dt
  = theta T + (v0-theta)(1-exp(-kappa T))/kappa
```

with the `kappa=0` limit `v0 T`, then applies Black-Scholes with effective volatility
`sqrt(I/T)`. This gives exact `sigma_v=0` behavior and a stable near-zero vol-of-vol limit.
The limit switch compares `sigma_v*sqrt(T)` with the variance-volatility scale using a
fourth-root-machine-epsilon conditioning boundary; this follows the `sigma_v^2` cancellation in
the characteristic-function coefficients rather than applying a price-domain epsilon.

## Fourier price

The implementation uses Heston's `P1/P2` inversion:

```text
P1 = 1/2 + (1/pi) integral_0^infinity Re[e^(-iu log K) phi(u-i)/(iu E[S_T])] du
P2 = 1/2 + (1/pi) integral_0^infinity Re[e^(-iu log K) phi(u)/(iu)] du
C  = S exp(-qT) P1 - K exp(-rT) P2.
```

SciPy adaptive quadrature evaluates finite segments. Integration may stop after two quiet tail
segments beyond the first three; otherwise it continues to the configured ceiling, 400 by default,
and rejects a materially unconverged tail. The result records summed quadrature error estimates,
function evaluations, the used bound, and runtime. Puts use parity
`P=C-S exp(-qT)+K exp(-rT)`, preserving parity to floating-point precision.

The offline benchmark `S=K=100`, `T=1`, `r=5%`, `q=0`, `v0=theta=0.04`, `kappa=2`,
`sigma_v=0.3`, `rho=-0.7` gives call value `10.3942185523`, consistent with the hard-coded
reference. Exact expiry returns intrinsic value.

## Research analytics

Scheme comparison evaluates FTE and QE-M with the same configuration and seed against one
analytical reference. It reports bias, absolute/relative error, sampling SE, z-score, runtime, and
variance diagnostics. A z-score measures sampling-scale disagreement; it does not include time-step
bias.

Path-count convergence sorts positive unique counts and assigns deterministic independent children
from `SeedSequence(root_seed)`. Time-step convergence keeps path count and seed fixed to expose the
combined discretization and same-seed sampling behavior. Neither analysis assumes monotonic error.

Full-path diagnostics compare the sample variance mean at every time to

```text
E[v_t] = theta + (v0-theta) exp(-kappa t),
```

and report near-zero fractions, terminal quantiles, min/max, and FTE negative-proposal frequency.
The terminal spot validation target is `E[S_T]=S0 exp((r-q)T)`.

Smile research prices sorted unique call strikes, then uses the Phase 4 solver to return `K/S`,
`log(K/F)`, price, implied volatility, and inversion status. A representative negative `rho`
produces the familiar downward equity skew, but the code does not enforce that as a universal
invariant. Parameter sensitivity sweeps exactly one of `v0`, `kappa`, `theta`, `sigma_v`, or `rho`
and reprices analytically. Changing `sigma_v` commonly changes smile curvature. These are scenario
sweeps, not calibration and not trading Greeks.

## HTTP safeguards and limitations

The API caps paths at 500,000, time steps at 2,000, cumulative requested path-step nodes at
25 million, convergence points at 20, smile points at 200, and sensitivity points at 100 by
default. Comparison accounts for both simulation runs, and convergence accounts for every point.
Full paths and driver arrays never cross the API boundary. Only European contracts are accepted.

The model assumes constant `r`, constant `q`, continuous trading, frictionless markets, a CIR
variance diffusion, no jumps, and no stochastic rates. FTE and QE-M have finite-step behavior;
their MC interval measures sampling uncertainty only. Phase 13 does not implement American or
path-dependent Heston payoffs, Heston Greeks, control variates, QMC, parallelism, or calibration.
Phase 14 may fit these same risk-neutral parameters to observed surfaces without replacing the
parameter, characteristic-function, analytical-pricing, simulation, or result boundaries.
