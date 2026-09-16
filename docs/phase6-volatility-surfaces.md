# Phase 6 volatility smiles and surfaces

## Scope and module boundaries

Phase 6 turns independent Phase 4/5 implied-volatility observations into a queryable research
surface. It does not calibrate a parametric model and it never repairs market data. The boundary is:

```text
normalized Phase 5 chain
  -> Phase 5 quote enrichment and IV inversion
  -> deterministic Phase 6 filtering and one-point-per-strike selection
  -> immutable smile slices
  -> immutable multi-maturity surface
  -> interpolation, descriptive analytics, and diagnostic-only arbitrage checks
```

`models/volatility_surface.py` contains serializable Pydantic contracts and policies.
`volatility_surface/core.py` owns coordinates, construction, interpolation, and queries.
`volatility_surface/synthetic.py` generates deterministic research fixtures.
`analytics/volatility_surface.py` owns shape metrics and static-arbitrage diagnostics. The API module
only validates resource limits and dispatches to these services. SciPy interpolator instances are
created for a query and are never serialized or stored in a model.

## Market IV points and coordinates

`VolatilityPoint` preserves strike, maturity, IV, option side, optional market price and bid/ask,
source, expiration, and quality metadata. It also stores each supported coordinate:

```text
moneyness          = K / S
forward moneyness  = K / F(T)
k                  = log(K / F(T))
F(T)               = S exp((r - q)T)
```

Strike and both moneyness ratios are positive. Maturity is strictly positive, IV is non-negative,
and all numeric inputs must be finite. `k = log(K/F)` is the canonical coordinate. It centers every
maturity on its own forward and separates surface shape from the deterministic carry shift. Phase 6
uses one flat continuously compounded risk-free rate and one flat continuous dividend yield. Yield
curves and discrete dividends remain deferred.

## Total implied variance

Total implied variance is first-class:

```text
w(k,T) = sigma(k,T)^2 T
sigma(k,T) = sqrt(w(k,T) / T)
```

The conversion helpers require `T > 0`, `sigma >= 0`, and `w >= 0`, and reject NaN, infinity, and
overflow. Smile interpolation and maturity interpolation both operate on `w`. Using total variance
across time matches the additive scale of diffusion variance and makes the calendar diagnostic
direct: at fixed `k`, decreasing `w` is a potential calendar-arbitrage signal. This is still an
interpolation convention, not a proof that an interpolated surface is arbitrage-free.

## Smile construction

A `VolatilitySmile` contains one maturity, spot, maturity-specific forward, flat carry assumptions,
expiration, ordered points, and construction metadata. The model always sorts points by `k` and
rejects duplicate canonical coordinates. It verifies that every point has the same maturity and
forward as its slice.

Market construction accepts `SmileConstructionConfig`. A quote is excluded when IV recovery
failed, maturity or forward is invalid, IV lies outside the configured range, the relative spread
is missing or above the configured maximum, freshness is required and the quote is stale/future,
or European price-bound validity is required and failed. Exclusions are counted by reason. An
expiration with fewer than `minimum_points` is reported and no synthetic point or interpolation is
fabricated.

The default `OTM_ONLY` policy chooses:

- a put for `k < -atm_log_moneyness_tolerance`;
- a call for `k > +atm_log_moneyness_tolerance`;
- one quote near `k = 0` by tightest relative spread, then call, then freshest timestamp, then
  lexicographic provider symbol.

`CALLS_ONLY`, `PUTS_ONLY`, and `ALL_VALID` are explicit alternatives. At any duplicate strike the
same deterministic ranking chooses one observation. Thus input ordering never changes a smile.
The construction metadata records the OTM, ATM, duplicate, and minimum-point policies.

`build_surface_from_option_chain` is the complete normalized-chain path. It calls the existing
Phase 5 enrichment service, applies these policies, and creates the surface. `build_market_surface`
accepts an already enriched chain when the caller has reused that work.

## Strike interpolation and extrapolation

`LINEAR` uses piecewise-linear interpolation in `(k,w)`. `PCHIP` uses SciPy's shape-preserving
piecewise cubic Hermite interpolator in `(k,w)`. PCHIP avoids oscillations typical of high-degree
global polynomials, but shape preservation alone does not guarantee option-price convexity.

An exact `k` node returns the observed IV and its directly calculated total variance. An interior
query interpolates `w` and converts it back through `sqrt(w/T)`. Negative or non-finite output is a
domain error.

Wing extrapolation defaults to `NONE`; an out-of-domain query fails with the observed bounds.
`FLAT_VOL` is the only enabled wing alternative: it uses the nearest endpoint volatility. Phase 6
does not infer an asymptotic wing slope. The query result states the domain, method, exact/interior
status, and extrapolation use.

## Multi-expiration surface and two-dimensional query

`VolatilitySurface` sorts slices by positive maturity and rejects duplicate maturities. All slices
must share spot, risk-free rate, and dividend yield. It stores only the points and configuration.

For requested `(T,K)`:

1. Calculate `F(T) = S exp((r-q)T)` and `k = log(K/F(T))`. A direct `k` request skips this conversion.
2. If `T` matches a slice, query that smile. Exact market nodes remain exact.
3. Otherwise identify adjacent maturities `T1 < T < T2`.
4. Query both smiles at the same canonical `k`, producing `w1` and `w2`.
5. Interpolate total variance:

   ```text
   lambda = (T - T1) / (T2 - T1)
   w(T)   = w1 + lambda (w2 - w1)
   sigma  = sqrt(w(T) / T)
   ```

6. Return both bounding maturities, both extrapolation flags, and both interpolation methods.

Maturity extrapolation defaults to `NONE`. Explicit `FLAT_VOL` maturity extrapolation uses the
nearest slice's queried volatility and recomputes `w = sigma^2 T` at the requested maturity.

## Static-arbitrage diagnostics

Diagnostics report structured `ArbitrageIssue` objects. They do not alter quotes, smiles, or the
surface.

### Strike monotonicity

For each surface smile, Phase 6 reconstructs European call prices at every point with the existing
Black-Scholes-Merton pricer and that point's IV. Adjacent prices must satisfy
`C(K[i+1]) <= C(K[i]) + tolerance`. The reusable strike-slice checker also supports puts and checks
`P(K[i+1]) >= P(K[i]) - tolerance`.

### Butterfly convexity

On an irregular strike grid, equal-spacing second differences are wrong. Phase 6 computes adjacent
secant slopes:

```text
s[i] = (V(K[i+1]) - V(K[i])) / (K[i+1] - K[i])
```

Convexity requires `s[i+1] >= s[i]` within tolerance. A decrease creates a
`BUTTERFLY_CONVEXITY` issue containing the middle strike and both slopes.

### Calendar total variance

The default calendar grid contains 21 evenly spaced `k` values over the common domain shared by
all smiles. A caller may supply a finite grid. At every `k`, adjacent maturities must satisfy:

```text
w(k,T[i+1]) >= w(k,T[i]) - tolerance
```

A decrease creates a `CALENDAR_TOTAL_VARIANCE` issue with the coordinate, maturity, values, and
tolerance. No common domain produces an `INSUFFICIENT_DATA` issue. A configured point outside a
slice with no extrapolation produces `EXTRAPOLATION_REQUIRED`.

The quality summary reports smile and point counts, maturity and `k` ranges, average points per
smile, construction exclusions, issue counts, and common-domain width divided by global-domain
width as descriptive interpolation coverage.

## ATM term structure and local shape

ATM-forward means `k = 0`, equivalently `K = F(T)`. The term-structure service queries each smile
at zero and returns unsmoothed IV, total variance, and exact/interpolated status. Maturities whose
smiles do not cover ATM under the configured extrapolation policy are listed as excluded.

For bump `h` (default `0.01` log-moneyness), local descriptive metrics are:

```text
skew      = [sigma(h) - sigma(-h)] / (2h)
curvature = [sigma(h) - 2 sigma(0) + sigma(-h)] / h^2
```

These are coordinate-based derivatives. They are not 25-delta risk reversals, calibrated model
parameters, or stable estimates when the slice lacks `[-h,h]` support.

## Synthetic validation

Deterministic helpers generate flat, linear-skew, and quadratic smiles:

```text
sigma(k,T) = base + skew k + curvature k^2 + term_slope T
```

They reject negative generated volatility. Tests cover exact-node recovery, LINEAR and PCHIP,
unsorted normalization, duplicate rejection, explicit extrapolation, flat-surface round trips,
total-variance maturity interpolation, increasing and decreasing calendar variance, intentional
irregular-grid butterfly violations, ATM term structure, and finite-difference skew/curvature.
Phase 5 fake market chains validate quote filtering, OTM side selection, ATM choice, duplicate
resolution, insufficient expirations, and end-to-end queryability without network access.

## API and resource policy

The six Phase 6 endpoints build/query smiles, build/query surfaces, run diagnostics, and extract ATM
term structure. Inputs and outputs are Pydantic models. Limits apply to smiles per surface, points
per smile, normalized chain contracts, and total diagnostic grid work. The direct library remains
available for larger deliberate research runs.

## Limitations and deferred work

The surface is an interpolation of European Black-Scholes implied volatilities under flat carry
assumptions. U.S. listed options may be American-style; the Phase 5 approximation marker remains
material. Midpoint IV is not an executable volatility, and Phase 6 does not model discrete
dividends, settlement conventions, or microstructure uncertainty. Optional bid/ask IV fields exist
for future enrichment, but midpoint IV drives construction. No weighting or fitting is performed.

SVI, arbitrage repair, Heston and calibration, local volatility and Dupire, yield curves, American
pricing, full-path/exotic pricing, jumps, regimes, and portfolio risk are deliberately deferred.
Those future models can consume the canonical points and total-variance queries without replacing
the Phase 1–5 contract, pricing, RNG, IV, or market-data foundations.
