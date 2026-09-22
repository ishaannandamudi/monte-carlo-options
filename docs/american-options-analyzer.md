# American Options Analyzer

## 1. User workflow

The default application route is a ticker-first American equity-option workflow. A user enters a
symbol such as `AAPL`; the backend fetches the underlying quote, available expirations, and one
bounded option chain, then returns market prices, BSM-equivalent implied volatilities, independent
fair-value volatility estimates, American CRR values, Greeks, liquidity fields, and diagnostics.
The browser never receives the Massive credential and performs no pricing mathematics.

`POST /api/options/analyze` accepts an optional expiration and bounded advanced overrides. With no
overrides, all automatic assumptions are returned in `assumptions`. Snapshot mode normally displays
approximately ten calls and ten puts around the money; Basic mode displays its bounded near-ATM
sample. The user can change expiration, filter the returned rows by side, sort the table, refresh
the cached analysis, and inspect a contract detail panel. Existing Phase 22 research pages remain
under `/research` and their original deep links.

## 2. Market-data source and freshness

The service reuses the Phase 5 `MarketDataProvider` and normalized `UnderlyingQuote`,
`OptionInstrument`, `OptionQuote`, and `OptionChainSnapshot` models. Production configuration uses
Massive; tests use the deterministic in-memory provider and never require network access.

Massive access can be snapshot or end-of-day according to the account and feed. The UI reports the
detected data mode, price source, as-of time, calculated age, and explicit warnings. It makes no
real-time claim. Provider authentication, entitlement, symbol, rate-limit, timeout,
malformed-response, and network errors retain distinct safe HTTP translations.

### Massive plan-aware data modes

The provider always tries the existing high-quality snapshot path first. A successful snapshot
uses the stock snapshot and option-chain snapshot, preserving bid, ask, midpoint, sizes, latest
trade, open interest, and provider analytics. A `401` means the credential is absent or rejected
and becomes `ProviderAuthenticationError`; it never triggers fallback. A `403` means the credential
authenticated but the plan excludes that endpoint and becomes `ProviderEntitlementError`. Only
this typed entitlement failure activates fallback. Network, timeout, malformed-data, rate-limit,
and server failures remain errors.

The Basic path uses three endpoint classes verified with the configured account:

- `/v2/aggs/ticker/{symbol}/prev` for the adjusted underlying previous-day OHLC bar;
- `/v3/reference/options/contracts` for active contract discovery near the requested expiration
  and underlying close; and
- `/v2/aggs/ticker/{optionsTicker}/prev` for each bounded option previous-day OHLC bar.

The underlying reference is the daily close and is labeled `END_OF_DAY_CLOSE`; open, high, low,
volume, VWAP, date, and timestamp are retained when supplied. Bid, ask, midpoint, quote size, live
trade, open interest, and provider IV/Greeks remain null when the EOD endpoints do not supply them.
Option `market_reference_price` and `comparison_price` are the EOD close and are never called a
midpoint. The result returns `data_mode`, `price_as_of`, `underlying_price_source`,
`option_price_source`, and provider capability metadata.

The valuation clock is the underlying bar's 4:00 p.m. America/New_York close converted to UTC.
Option bars carry their own trading date. Matching dates are evaluated together; a mismatch is
visible as `DATA_DATE_MISMATCH` and a result warning. EOD IV is inverted from the close and labeled
`EOD_BSM_EQUIVALENT_IV`. The same leave-one-out smile hierarchy remains in force when enough EOD
neighbors exist; a sparse set reaches the documented ATM or own-IV fallback. CRR remains the
authoritative American pricer and uses the EOD underlying close, EOD valuation time, remaining
maturity, configured rates/dividends, and independent fair-value volatility where available.

Massive Basic permits five calls per minute on the tested account. A cold fallback request spends
at most one denied snapshot probe, one underlying bar request, one bounded reference request, and
two option-bar requests. `BASIC_ANALYSIS_CONTRACT_LIMIT=2` therefore returns the closest ATM call
and put when both sides are requested, or two contracts of the requested side. The provider caches
the observed snapshot capability, underlying bar, reference contracts, and option bars for
`MARKET_DATA_CACHE_TTL_SECONDS=60`; authentication errors are never cached. Reference discovery is
limited to one Basic page in a configurable window around the target expiration. The response and
UI state this limited coverage explicitly rather than implying a complete live chain.

## 3. Expiration selection

When the request omits `expiration`, the analyzer removes expirations on or before the current UTC
date and selects the future date closest to `DEFAULT_ANALYSIS_TARGET_DAYS` (45 by default). An
equal-distance tie selects the earlier date. This avoids same-day expiration and does not drift to
an arbitrary long maturity. A requested expiration must be in the provider list. The response
returns the target, selected day count, selected expiration, and a bounded list of alternatives.

## 4. Chain filtering

The analyzer first applies the configured percentage window around the underlying price. It ranks
contracts deterministically by usable midpoint, timestamp consistency, relative spread, distance
from ATM, liquidity (`volume + open interest`), and provider symbol. It retains a bounded neighbor
pool for smile construction, then chooses the displayed contracts with equal call/put quotas when
both sides are requested. Any unused quota can be filled by the other side. Output ordering is
side, strike, then provider symbol.

The snapshot default is 20 displayed contracts with a 15% strike window. The provider snapshot and
display limits are checked before IV and tree loops. Basic mode applies its two-contract provider
cap before local analytics, regardless of the larger requested display limit.

## 5. Market price and implied volatility

The market midpoint is `(bid + ask) / 2` only for a non-negative, non-crossed two-sided quote. An
eligible last price is used only when `allow_last_price_fallback` is explicitly true. Fair-value
gap remains a comparison with the midpoint, so it is unavailable for a last-only observation.

Phase 4's bounded solver recovers implied volatility from the selected market price after European
no-arbitrage checks. Because that inversion uses BSM while listed US equity options are treated as
American, snapshot values are labeled **BSM-equivalent** and EOD values are labeled
**EOD BSM-equivalent**. Provider IV is retained only as comparison
metadata. Failed inversion becomes a typed status and quality flag; it never injects NaN or
infinity and does not fail the whole chain.

## 6. Fair-value volatility and smile interpolation

The volatility used for fair value follows this hierarchy for every target contract:

1. `SMILE_INTERPOLATED`: PCHIP interpolation in total variance using valid same-side,
   same-expiration neighboring strikes.
2. `ATM_REFERENCE`: nearest valid independent neighboring contract when interpolation is not
   supported.
3. `CONTRACT_IV_FALLBACK`: the target's own IV only when no independent neighbor exists.
4. `UNAVAILABLE`: no usable volatility.

The interpolation is leave-one-out: it excludes both the target provider symbol and every contract
at the target strike. It also excludes stale/future, crossed/negative, expired, arbitrage-invalid,
failed-IV, and overly wide-spread observations. Interpolation is allowed only inside the remaining
smile domain; there is no wing extrapolation. This prevents the circular calculation “market price
→ own IV → model price” from being presented as independent fair value. A contract-IV fallback is
marked non-independent and low data quality.

## 7. American CRR fair value and Greeks

The authoritative fair-value model is the existing Phase 7/8 Cox-Ross-Rubinstein American tree,
using the fair-value volatility above, the returned continuously compounded risk-free rate,
continuous dividend yield, and configured tree steps. American exercise is applied at each node.
The result includes a BSM European reference and early-exercise premium for diagnostics.

Delta, Gamma, and annual calendar Theta come directly from the tree's first levels. Daily Theta is
annual Theta divided by 365. Vega and Rho are central bump-and-revalue derivatives of the same
American CRR engine using bumps of `1e-4`; Vega per volatility point is raw Vega times `0.01`.
Runtime fields separate base CRR pricing from the additional Greek repricings.

## 8. Fair-value gap

For a positive market midpoint:

```text
fair_value_gap = CRR American fair value - market midpoint
fair_value_gap_pct = fair_value_gap / market midpoint
```

A positive number means model value exceeds midpoint; a negative number means model value is below
midpoint. These are descriptive model differences. The application does not label contracts as
under/overvalued and provides no buy or sell recommendation.

In EOD mode the same formulas use `EOD close` in place of `market midpoint`. The response's
`comparison_price_source` removes any ambiguity between the two cases.

## 9. Liquidity and data quality

Every contract retains bid, ask, sizes, last, absolute and relative spread, volume, and open
interest. Quality flags reuse Phase 5 issues and add:

- `WIDE_SPREAD`: midpoint-relative spread exceeds
  `DEFAULT_ANALYSIS_MAX_RELATIVE_SPREAD` (50% by default).
- `NO_RECORDED_VOLUME`: volume is absent or zero.
- `NO_OPEN_INTEREST`: open interest is absent or zero.
- `ILLIQUID`: spread is wide, or both volume and open interest are absent/zero.
- `END_OF_DAY_DATA`, `SNAPSHOT_UNAVAILABLE_FOR_PLAN`, `NO_LIVE_BID_ASK`, and
  `LOW_PROVIDER_RATE_LIMIT`: expected capability constraints in Basic mode.
- `DATA_DATE_MISMATCH`: the option and underlying daily bars do not share a trading date.

The descriptive analysis-quality rules are exact:

- `LOW_DATA_QUALITY` when fair value is unavailable, fair volatility is non-independent, or the
  quote is crossed, negative, stale, future-dated, missing a market price, or too wide.
- `MODERATE_DATA_QUALITY` when otherwise usable but marked illiquid or using `ATM_REFERENCE`.
- `HIGH_DATA_QUALITY` for the remaining independently interpolated, priced observations.

These labels describe input quality only. Warning flags also report BSM-equivalent IV,
insufficient interpolation neighbors, non-independent fallback, assumed American exercise when
provider metadata differs, and failed CRR evaluation.

## 10. Automatic assumptions

The project does not add an unverified rate or dividend feed. Defaults come from
`DEFAULT_ANALYSIS_RISK_FREE_RATE` (4%) and `DEFAULT_ANALYSIS_DIVIDEND_YIELD` (0%) and appear in the
response with `CONFIGURED_DEFAULT`. Request overrides appear as `REQUEST_OVERRIDE`. The response
also exposes target days, strike window, quote-age threshold, maximum spread, contract limit,
last-price policy, tree steps, and valuation time.

## 11. Resource limits and caching

`MAX_OPTION_CHAIN_CONTRACTS`, `MAX_ANALYSIS_CONTRACTS`, `MAX_ANALYSIS_EXPIRATIONS`, and the existing
`MAX_BINOMIAL_STEPS` bound provider normalization, response size, and numerical work. Pydantic
validates all positive counts, ages, and windows before provider/tree operations. TanStack Query
caches a result for two minutes by the complete request object: symbol, expiration, and every
assumption. Refresh explicitly refetches the same key. The backend's short provider cache remains
authoritative for Basic rate-limit compliance even when the browser explicitly refreshes.

## 12. API and frontend behavior

The request/response is strictly typed in Python and TypeScript. The response never includes raw
provider payloads or full numerical trees. The default page uses an accessible ticker label,
semantic sortable table, call/put group, labeled expiration selector, keyboard buttons, textual
warnings, an accessible two-series SVG smile plus hidden data table, and a responsive horizontal
chain scroller. Advanced assumptions are collapsed initially. Loading text describes the atomic
backend operation without claiming measured per-stage progress.

## 13. Limitations

- Market IV is a BSM-equivalent approximation, not American IV.
- Fair volatility is a local single-expiration estimate rather than a calibrated forecasting
  model. Sparse chains can fall back to ATM or the target IV.
- Rates and dividends are configured defaults unless overridden; corporate actions and discrete
  dividends are not fetched.
- CRR convergence depends on tree steps. Finite-difference Vega/Rho require four additional tree
  values per contract and dominate CPU cost.
- Quote quality cannot make delayed, sparse, or illiquid data current or executable.
- Basic mode is a two-contract near-ATM research sample, not a complete option chain. A contract
  without a previous-day qualifying trade can have no aggregate bar and may be omitted.
- Values are research diagnostics, not trade recommendations, broker quotes, or execution signals.

## 14. Advanced research terminal

The complete Phase 1–22 terminal remains available at `/research`, with Pricing, Volatility,
Models, Portfolio, Risk, Regimes, QMC, and Experiments below it. Existing top-level deep links such
as `/pricing` remain valid. The analyzer composes the established provider, IV, smile, CRR, and
Greek services; it does not replace or weaken their standalone APIs.
