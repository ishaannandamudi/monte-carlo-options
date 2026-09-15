# Monte Carlo Options

Monte Carlo Options is the quantitative foundation for a long-term derivatives-research platform.
The current implementation prices European calls and puts under constant-market risk-neutral
geometric Brownian motion (GBM), with a Black-Scholes-Merton benchmark, statistically sound
uncertainty diagnostics, advanced variance reduction, and first-class Greeks.

The inputs are continuous floating-point model values. They are not accounting ledger values or
currency-safe decimal amounts.

## Current capabilities

- Validated European call and put contracts and separate constant market assumptions
- Risk-neutral GBM terminal simulation with continuous dividend yield
- Plain Monte Carlo and antithetic variates
- Discounted-terminal-stock control variates and combined antithetic plus control variates
- Discounted-payoff sample variance, standard deviation, standard error, and configurable normal
  confidence intervals
- Exact pair-level uncertainty treatment for even antithetic samples and an explicit weighted
  cluster treatment for odd samples
- Black-Scholes-Merton analytical pricing with dividends
- Monte Carlo versus analytical error, z-score, and interval-containment diagnostics
- Independently seeded, reproducible convergence experiments and empirical standard-error slopes
- Fixed-budget comparisons across all four variance-reduction modes
- Analytical Delta, Gamma, Vega, calendar-time Theta, Rho, and dividend Rho
- Common-Random-Number finite-difference Greeks with direct estimator uncertainty
- Pathwise Delta and Vega with plain or antithetic sampling
- Analytical-versus-stochastic Greek comparisons and one-dimensional sensitivity grids
- Thin FastAPI endpoints with configurable resource limits
- Deterministic, numerical, statistical, and API tests

The current scope deliberately excludes implied volatility, market data, path-dependent or American
options, trees, calibration, stochastic volatility, jumps, multi-asset models, portfolio risk,
parallelism, a GPU backend, quasi-Monte Carlo, and a frontend.

## Mathematical model

The terminal stock price under the risk-neutral measure is

```text
S_T = S_0 exp((r - q - 0.5 sigma^2)T + sigma sqrt(T) Z),  Z ~ N(0, 1)
```

The pricer evaluates the vectorized call payoff `max(S_T - K, 0)` or put payoff
`max(K - S_T, 0)`, then discounts every payoff by `exp(-rT)`. The price is the sample mean of those
discounted payoffs. All reported estimator uncertainty uses discounted values.

Black-Scholes-Merton supplies the exact analytical benchmark under the same assumptions. It is
used for model validation and convergence diagnostics, not to alter Monte Carlo estimates.

## Antithetic uncertainty

With antithetic sampling, each normal `Z` is followed by `-Z`. `num_simulations` always means total
terminal outcomes. An odd request returns exactly the requested count: complete pairs come first,
and one final independent positive-side draw is retained as a singleton.

For an even count, discounted payoffs are averaged within each pair. The reported standard error is
the sample standard deviation of those independent pair averages divided by the square root of the
number of pairs. Individual antithetic payoffs are correlated and are never treated as independent
for this calculation.

For an odd count, the final singleton remains in the path-weighted price. A finite-sample
independent-cluster variance estimate accounts for pair clusters of size two and the singleton
cluster of size one. The full formula and rationale are in
[docs/phase1-design.md](docs/phase1-design.md).

Antithetic sampling often lowers variance for vanilla options, but it is not guaranteed to improve
every payoff and parameter set.

## Control variates

The control variable is discounted terminal stock:

```text
X = exp(-rT) S_T
E[X] = S_0 exp(-qT)
```

For discounted payoff `Y`, the sample estimates

```text
beta_hat = Cov(Y, X) / Var(X)
Y_cv = Y - beta_hat (X - E[X])
```

The returned price and standard error come from `Y_cv`. Results retain the raw price and raw SE,
estimated beta, sample and known control means, raw and adjusted estimator variances, and the
variance-reduction ratio `raw estimator variance / adjusted estimator variance`. The two estimator
variances equal their corresponding standard errors squared; `payoff_variance` remains the variance
of individual discounted payoff values. A ratio above one indicates improvement for that sample;
no universal improvement is assumed. If control variance is zero, beta falls back to zero and the
adjusted estimate equals the raw estimate.

For `ANTITHETIC_CONTROL_VARIATE`, payoff and control values are both averaged within independent
antithetic clusters before beta is estimated or adjustment and SE are calculated. Odd counts retain
the singleton's one-path weight. See the explicit derivation in
[docs/phase2-variance-reduction.md](docs/phase2-variance-reduction.md).

## Greeks and unit conventions

For value `V(S, sigma, r, q, T)`, the library reports:

- Delta: `dV/dS`
- Gamma: `d²V/dS²`
- Vega: `dV/dsigma`
- Rho: `dV/dr`
- Dividend Rho: `dV/dq`
- Theta: `-dV/dT`, the change in value as calendar time passes

Values use raw model units. Vega is per absolute volatility unit `1.00`; rate sensitivities are per
absolute rate unit `1.00`; Theta is per year. The library does not silently scale Vega to one
percentage point or Rho to basis points.

Black-Scholes-Merton provides closed-form values for all six Greeks. Exact expiry uses an explicit
finite convention: call/put Delta at the strike is `0.5`/`-0.5`, while Gamma, Vega, Theta, Rho, and
dividend Rho are zero. Zero volatility differentiates the active deterministic discounted-payoff
branch; at the deterministic forward kink it uses half-weight branch derivatives and the right-hand
Vega limit. See [docs/phase3-greeks.md](docs/phase3-greeks.md) for every formula and boundary rule.

## Monte Carlo Greeks

Finite differences support Delta, Gamma, Vega, Theta, Rho, and dividend Rho under every existing
variance-reduction mode. Spot uses a relative bump; all other bumps use raw absolute model units.
Central differences are used when both states are valid. Vega near zero volatility and Theta near
expiry use a forward fallback so no invalid state is constructed.

The engine explicitly generates one PCG64 standard-normal driver vector and reuses it for the base,
upward, and downward valuations. These Common Random Numbers (CRN) make each per-driver Greek
observation a difference under the same shock. Standard errors come directly from those difference
observations, including antithetic pair clusters and an odd singleton. They are not formed by
treating bumped prices as independent.

Control-variate finite differences recompute the correct discounted-stock expectation for every
bumped state before combining its adjusted observations. The default bumps are:

```text
spot_relative_bump             = 1e-3
volatility_absolute_bump       = 1e-4
rate_absolute_bump             = 1e-4
dividend_yield_absolute_bump   = 1e-4
maturity_absolute_bump         = 1/3650 years
```

Large bumps have more truncation error; very small bumps can suffer cancellation and simulation
noise. Smaller is not universally better, so research callers should inspect bump stability.

Pathwise estimators are implemented for Delta and Vega. They differentiate terminal GBM and use
strict ITM indicators at the payoff kink. Pathwise sampling supports `NONE` and `ANTITHETIC`, with
SE calculated from independent paths or pair clusters. Pathwise Gamma is intentionally absent
because naively differentiating the payoff indicator does not produce a valid second-derivative
estimator.

## Reproducibility and convergence

Every pricing call constructs an isolated `numpy.random.Generator` backed explicitly by `PCG64`.
The code never uses NumPy's global random state. Identical model inputs, simulation configuration,
seed, and supported numerical environment reproduce all quantitative outputs exactly; measured
wall-clock runtime naturally varies.

A convergence experiment creates `SeedSequence(root_seed)` and spawns one deterministic child per
simulation count. Points therefore use independent, reproducible streams rather than prefixes of
one shared stream. Valid positive standard errors are fitted by ordinary least squares:

```text
log(SE) = a + b log(N)
```

For all four methods, the expected asymptotic slope remains approximately `b = -0.5`. Variance
reduction lowers the error's constant factor; it does not change the IID Monte Carlo order. If fewer
than two positive-SE points exist, the slope is returned as unavailable rather than raising an
error.

Method comparisons use the same root seed for all modes. `NONE` and `CONTROL_VARIATE` therefore
share path normals, while `ANTITHETIC` and `ANTITHETIC_CONTROL_VARIATE` share primary normals and
pairing. Relative variance is method estimator variance divided by the `NONE` variance; relative
efficiency is its reciprocal. Runtime is reported separately and is not mixed into either metric.

## Repository structure

```text
backend/app/
├── analytics/     # convergence, estimator comparisons, and sensitivity grids
├── api/           # FastAPI request models, limits, and thin adapters
├── core/          # environment settings, logging, errors, numerical guards
├── models/        # contracts, Greek/config enums, and typed result models
├── payoffs/       # vectorized vanilla payoff and intrinsic value helpers
├── pricing/       # prices, Greeks, comparisons, and estimator transformations
└── simulation/    # GBM, reusable CRN drivers, and PCG64/SeedSequence ownership
backend/tests/     # deterministic, numerical, statistical, and HTTP tests
benchmarks/        # reserved for opt-in quantitative benchmarks
docs/              # architecture and numerical methodology
experiments/       # reserved for explicit research experiments
```

## Installation

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The runtime dependencies are FastAPI, Pydantic, NumPy, SciPy, and Uvicorn. The requirements file
also installs pytest, Ruff, mypy, and HTTPX for development and API testing.

## Library example

```python
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.models.simulation import SimulationConfig, VarianceReduction
from backend.app.pricing.comparison import compare_prices

contract = OptionContract(
    option_type=OptionType.CALL,
    spot=100.0,
    strike=100.0,
    maturity_years=1.0,
)
market = MarketEnvironment(
    risk_free_rate=0.05,
    volatility=0.20,
    dividend_yield=0.01,
)
simulation = SimulationConfig(
    num_simulations=100_000,
    seed=42,
    variance_reduction=VarianceReduction.ANTITHETIC_CONTROL_VARIATE,
    confidence_level=0.95,
)

comparison = compare_prices(contract, market, simulation)
print(comparison.model_dump())
```

Analytical and pathwise Greeks:

```python
from backend.app.models.greeks import Greek
from backend.app.pricing.black_scholes_greeks import black_scholes_greeks
from backend.app.pricing.monte_carlo_greeks import pathwise_greeks

analytical = black_scholes_greeks(contract, market)
pathwise = pathwise_greeks(
    contract,
    market,
    simulation.model_copy(update={"variance_reduction": VarianceReduction.ANTITHETIC}),
    [Greek.DELTA, Greek.VEGA],
)
```

`batch_size` is validated as an optional execution hint. The current engine evaluates a single
vector; changing `batch_size` does not change the random stream or quantitative output.

## API

Start the service from the repository root:

```bash
.venv/bin/uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Monte Carlo pricing:

```bash
curl -X POST http://127.0.0.1:8000/api/pricing/monte-carlo \
  -H 'Content-Type: application/json' \
  -d '{
    "contract": {
      "option_type": "CALL",
      "exercise_style": "EUROPEAN",
      "spot": 100.0,
      "strike": 100.0,
      "maturity_years": 1.0
    },
    "market": {
      "risk_free_rate": 0.05,
      "volatility": 0.2,
      "dividend_yield": 0.01
    },
    "simulation": {
      "num_simulations": 100000,
      "seed": 42,
      "variance_reduction": "ANTITHETIC_CONTROL_VARIATE",
      "random_engine": "PCG64",
      "confidence_level": 0.95
    }
  }'
```

Available endpoints:

- `GET /health`
- `POST /api/pricing/monte-carlo`
- `POST /api/pricing/black-scholes`
- `POST /api/pricing/compare`
- `POST /api/analytics/convergence`
- `POST /api/analytics/variance-reduction`
- `POST /api/greeks/black-scholes`
- `POST /api/greeks/monte-carlo`
- `POST /api/analytics/greeks/compare`
- `POST /api/analytics/sensitivity`

The convergence body replaces `simulation` with `simulation_counts`, `root_seed`,
`variance_reduction`, and `confidence_level`. Generated OpenAPI documentation is available at
`/docs` while the service is running.

Compare every variance-reduction method with one fixed budget:

```bash
curl -X POST http://127.0.0.1:8000/api/analytics/variance-reduction \
  -H 'Content-Type: application/json' \
  -d '{
    "contract": {
      "option_type": "CALL", "exercise_style": "EUROPEAN",
      "spot": 100.0, "strike": 100.0, "maturity_years": 1.0
    },
    "market": {
      "risk_free_rate": 0.05, "volatility": 0.2, "dividend_yield": 0.01
    },
    "num_simulations": 100000, "root_seed": 42, "confidence_level": 0.95
  }'
```

Request CRN finite-difference Greeks by sending the same contract, market, and simulation objects
to `/api/greeks/monte-carlo`, together with:

```json
{
  "greeks": ["DELTA", "GAMMA", "VEGA", "THETA", "RHO", "DIVIDEND_RHO"],
  "estimator_method": "FINITE_DIFFERENCE"
}
```

The optional `bump_config` object overrides individual bump defaults. `PATHWISE` accepts Delta and
Vega with `NONE` or `ANTITHETIC` variance reduction. Unsupported combinations return HTTP 422.

API allocation limits come from environment variables and do not constrain direct library use:

```text
MAX_API_SIMULATIONS=5000000
MAX_CONVERGENCE_POINTS=20
MAX_SENSITIVITY_POINTS=100
LOG_LEVEL=INFO
```

Invalid input and unsafe allocation requests return HTTP 422 with a clear detail. Raw terminal
prices and payoff vectors never cross the API boundary. A stochastic run also needs enough
independent replication to estimate uncertainty: at least two paths for plain Monte Carlo and at
least two antithetic clusters. Smaller stochastic requests fail clearly instead of reporting false
zero uncertainty; deterministic cases remain exact.

## Tests and quality checks

```bash
.venv/bin/pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy backend/app
```

The normal suite uses fixed seeds, moderate sample sizes, analytical identities, deterministic edge
cases, theoretical GBM moments, direct pair/control/CRN calculations, analytical Greek identities,
and statistically scaled Monte Carlo tolerances such as `abs(estimate - benchmark) <= k * SE`.
Resource-intensive checks belong under the `performance` marker and are excluded from ordinary
pytest by project configuration.

## Model assumptions and limitations

- European exercise only
- Frictionless market and continuous trading as assumed by Black-Scholes
- Risk-neutral GBM with lognormal stock prices
- Constant volatility
- Constant continuously compounded risk-free rate, which may be negative
- Constant continuous dividend yield, which may be negative
- No jumps or stochastic volatility
- No transaction costs, liquidity effects, discrete hedging, or model calibration
- Terminal-only simulation; no path-dependent state
- Normal-approximation confidence intervals
- Greeks use raw model units and the calendar-time Theta convention
- Pathwise differentiation is limited to Delta and Vega

At `T = 0`, both engines return intrinsic value. At `sigma = 0`, both use the discounted
deterministic risk-neutral terminal payoff. Derived values outside finite floating-point range raise
a quantitative domain error rather than returning NaN or infinity.

## Roadmap

Future phases can add full paths, additional variance-reduction techniques, implied volatility,
market option chains, binomial trees, American exercise and Longstaff-Schwartz, Asian and barrier
payoffs, correlated assets, Heston volatility and calibration, jump diffusion, regime switching,
portfolio aggregation and risk, performance work, quasi-Monte Carlo, and eventually an interactive
dashboard. Those features are architectural consumers of the existing contract, process, payoff,
pricing, RNG, cluster, Greek, and result boundaries; none are implemented here.
