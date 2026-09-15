# Phase 5 market-data design

## 1. Scope

Phase 5 turns external option snapshots into typed, provider-independent research inputs. It adds
normalization, quote-quality diagnostics, deterministic chain analytics, Phase 4 implied-volatility
enrichment, Phase 3 analytical Greeks, liquidity fields, smile extraction, and put-call parity.

It does not construct or repair a volatility surface, price American exercise, infer dividends,
build a yield curve, or add any Phase 6 model.

## 2. Provider abstraction

`MarketDataProvider` is a synchronous protocol with three operations:

```text
get_underlying_quote(symbol) -> UnderlyingQuote
get_option_chain(symbol, expiration=None) -> OptionChainSnapshot
list_option_expirations(symbol) -> list[date]
```

Analytics and API response models depend only on these normalized return types. Provider request
URLs, enum spellings, abbreviated field names, pagination, and timestamp units stay inside the
adapter. `FakeMarketDataProvider` serves normalized in-memory models for deterministic tests and
local development.

## 3. Concrete provider: Massive

The concrete adapter targets the Massive REST API, formerly Polygon.io. It uses:

- `/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}` for an underlying snapshot;
- `/v3/snapshot/options/{symbol}` for paginated option-chain snapshots;
- `/v3/reference/options/contracts` for paginated expiration discovery.

The standard library `urllib` transport avoids adding a provider SDK. Credentials and transport
settings come from `MARKET_DATA_API_KEY`, `MARKET_DATA_BASE_URL`, and
`MARKET_DATA_TIMEOUT_SECONDS`. The application can start without a key; the adapter raises a typed
authentication error only when a live request is attempted. API keys are added to requests inside
the transport and are never logged.

## 4. Provider field audit

The adapter maps Massive fields as follows:

| Massive field | Internal field | Interpretation |
|---|---|---|
| `ticker.ticker` | `UnderlyingQuote.symbol` | normalized underlying symbol |
| `ticker.lastQuote.p` | `UnderlyingQuote.bid` | stock NBBO bid |
| `ticker.lastQuote.P` | `UnderlyingQuote.ask` | stock NBBO ask |
| `ticker.lastQuote.t` | `UnderlyingQuote.timestamp` | quote nanoseconds since Unix epoch |
| `ticker.lastTrade.p` | `UnderlyingQuote.last` | most recent reported trade |
| `ticker.prevDay.c` | `previous_close` | previous daily close |
| `ticker.day.v` | `volume` | current daily volume |
| `details.contract_type` | `OptionType` | `call`/`put` normalized to project enum |
| `details.strike_price` | `OptionInstrument.strike` | positive strike |
| `details.expiration_date` | `OptionInstrument.expiration` | date mapped by documented convention |
| `details.exercise_style` | `MarketExerciseStyle` | American/European/Bermudan/unknown |
| `details.shares_per_contract` | `multiplier` | optional contract multiplier |
| `details.ticker` | `provider_symbol` | opaque provider contract identifier |
| `last_quote.bid`, `ask` | `OptionQuote.bid`, `ask` | option NBBO sides |
| `last_quote.bid_size`, `ask_size` | quote sizes | optional quoted sizes |
| `last_quote.last_updated` | `OptionQuote.timestamp` | quote update nanoseconds |
| `last_trade.price` | `OptionQuote.last` | optional last trade price |
| `last_trade.sip_timestamp` | `last_trade_timestamp` | trade SIP nanoseconds |
| `day.volume` | `volume` | optional current daily volume |
| `open_interest` | `open_interest` | prior-day open contracts |
| `implied_volatility` | `provider_implied_volatility` | reference only |
| `greeks` | `provider_greeks` | reference only |

Massive `break_even_price`, FMV, quote `timeframe`, raw exchange/condition codes, raw pagination,
and the full response are not exposed because Phase 5 does not assign them canonical analytical
semantics. Provider IV and Greeks are preserved for audits but never replace project calculations.
The adapter relies on explicit `details` fields and treats the OCC-style ticker as opaque, so core
analytics do not assume a symbol encoding.

## 5. Normalized models

`UnderlyingQuote` contains symbol, optional bid/ask/last, derived midpoint, timestamp and timestamp
source, provider source, previous close, and volume. Underlying prices must be non-negative.

`OptionInstrument` contains normalized underlying, `OptionType`, positive strike, UTC expiration,
opaque provider symbol, reported exercise style, and optional positive multiplier.

`OptionQuote` contains the instrument, optional price sides and last trade, derived midpoint,
optional sizes/volume/open interest, UTC timestamps, provider source, and optional provider
reference analytics. Negative option price fields remain representable so quality analytics can
report them rather than losing the contract.

`OptionChainSnapshot` combines the normalized underlying quote, retrieval time, source, option
quotes, and available expiration dates. Raw payloads are excluded.

All models are immutable, reject unknown fields, and reject NaN or infinity.

## 6. Symbol policy

Underlying symbols are stripped and uppercased. Provider suffixes are not inferred or rewritten.
Provider option symbols are retained verbatim as identity/audit fields. The Massive adapter uses
explicit contract details and therefore does not need OCC parsing; no fixed-width symbol assumption
leaks into analytics.

## 7. Timestamp normalization

All internal datetimes must have a timezone. Model validators convert them to UTC. Massive integer
timestamps are interpreted according to its documented nanosecond fields.

`QuoteTimestampSource` distinguishes `QUOTE`, `TRADE`, and `SNAPSHOT`. Option timestamps prefer
`last_quote.last_updated`, then the last-trade SIP timestamp, then the explicit retrieval time.
This makes fallback semantics visible. Underlying timestamps prefer the last quote and otherwise
use the provider snapshot update time. A missing or impossible required provider timestamp raises a
malformed-response error rather than creating a naive datetime.

## 8. Expiration and maturity convention

Massive option snapshots provide an expiration date without a time. Phase 5 maps that date to
4:00 p.m. `America/New_York`, then converts to UTC. This is a research valuation convention aligned
with the regular option trading close; it is not a claim about legal exercise, settlement, or every
index product's exact cutoff.

Maturity uses ACT/365 elapsed seconds:

\[
T=\frac{\max(t_{expiration}-t_{valuation},0)}{365\times24\times3600}.
\]

Expired observations remain in results with `EXPIRED`; negative maturity is never constructed.

## 9. Midpoint and spread policy

A midpoint exists only when bid and ask are both present, both non-negative, and ask is at least
bid:

\[
M=\frac{bid+ask}{2}.
\]

Missing sides and crossed markets produce no midpoint. They are preserved without silently fixing
the quote. For a valid two-sided quote:

\[
spread_{abs}=ask-bid,
\qquad spread_{rel}=\frac{ask-bid}{M}
\]

when \(M>0\). Relative spread is unavailable at a zero midpoint.

## 10. Quote-quality issues

An empty issue list means valid. Structured issues are:

- `MISSING_BID`, `MISSING_ASK`
- `NEGATIVE_PRICE`, `CROSSED_MARKET`
- `STALE`, `FUTURE_TIMESTAMP`, `EXPIRED`
- `PRICE_BELOW_LOWER_BOUND`, `PRICE_ABOVE_UPPER_BOUND`
- `NO_MARKET_PRICE`, `IV_UNAVAILABLE`

A quote may have several issues. The engine never collapses these into one vague boolean and does
not destructively discard bad observations.

## 11. Staleness

Staleness compares UTC valuation time with normalized quote timestamp. An age strictly greater than
`max_quote_age_seconds` is stale; a negative age is `FUTURE_TIMESTAMP`. The application default is
`MAX_QUOTE_AGE_SECONDS=900`, while a request can supply an explicit `QuoteQualityConfig`.

## 12. Market-price selection

Analytics first choose a fresh valid midpoint. If no midpoint exists, a non-negative last trade can
be used only when `allow_last_price=true`. Stale, future-dated, crossed, negative, and expired
quotes cannot use either source. Missing one side does not prevent explicit last-trade fallback.
`market_price_source` records `MIDPOINT` or `LAST`.

The underlying spot uses its valid midpoint, then its normalized last trade. A chain without either
cannot support spot-dependent analytics and raises a clear normalized validation error.

## 13. No-arbitrage integration

For every selected option market price, Phase 5 constructs the existing European `OptionContract`
and calls Phase 4 `european_option_price_bounds`. It applies the request's
`arbitrage_tolerance` around those bounds. Prices below or above are flagged; the quote is retained,
and IV solving is skipped. No bound formula is duplicated and no quote is repaired.

## 14. Implied-volatility and Greek enrichment

Eligible prices call Phase 4 `solve_implied_volatility` with explicit flat continuously compounded
rate, flat continuous dividend yield, and the configured solver policy. Solver failure becomes
`IV_UNAVAILABLE` plus a typed IV status on that quote; it never aborts the chain.

When IV is available, Phase 5 constructs `MarketEnvironment` with the recovered value and calls the
Phase 3 `black_scholes_greeks`. Provider IV or Greeks are never canonical. The result also includes
forward price, \(K/S\) moneyness, \(\log(K/F)\), bounds, spread, and maturity.

## 15. Deterministic chain summary and sorting

Output sorts by expiration, option type, strike, then provider symbol. It does not inherit provider
page order. Summary counts are factual: total contracts, empty-issue valid quotes, invalid quotes,
quotes with midpoints, quotes with recovered IV, bound violations, stale quotes, calls, puts, and
unique expirations.

`MAX_OPTION_CHAIN_CONTRACTS=5000` bounds raw-chain responses and supplied-chain analytics in the
HTTP layer. Oversized chains are rejected, not truncated. Direct library analytics have no hidden
size limit.

## 16. Smile extraction and filters

Smile extraction requires one exact UTC expiration and one option type. It returns strike, selected
price, recovered IV, moneyness coordinates, bid, ask, and issues. Calls and puts remain separate.
There is no smoothing, interpolation, extrapolation, or strike repair.

Reusable filters support expiration, call/put type, strike range, moneyness range, valid midpoint,
available IV, and freshness. Filter output keeps deterministic chain ordering.

## 17. Put-call parity

Fresh valid call and put midpoints are indexed by `(expiration, strike)`, avoiding a quadratic
search. For each pair:

\[
D_{obs}=C_{mid}-P_{mid},
\qquad D_{theory}=Se^{-qT}-Ke^{-rT},
\]

\[
residual=D_{obs}-D_{theory}.
\]

Relative residual is `residual / abs(theoretical_difference)` when that denominator is meaningful.
A violation occurs when

\[
|residual| > \epsilon_{abs}+\epsilon_{rel}|D_{theory}|.
\]

Unmatched or ineligible contracts are counted and omitted from pair results. Prices are diagnosed,
never altered.

## 18. Provider failures and APIs

The adapter translates authentication, rate limit, symbol-not-found, malformed response, timeout,
network, and provider-server failures into provider-independent exceptions. HTTP mapping is 401,
429, 404, 502, or 504 as appropriate. Error text contains no key or raw payload.

Phase 5 endpoints are:

- `GET /api/market-data/underlying/{symbol}`
- `GET /api/market-data/options/{symbol}` with optional `expiration`
- `GET /api/market-data/options/{symbol}/expirations`
- `POST /api/analytics/options/chain`
- `POST /api/analytics/options/smile`
- `POST /api/analytics/options/put-call-parity`

Analytics POST bodies accept a normalized chain snapshot, optional valuation time, explicit flat
rate/yield, optional quality configuration, and Phase 4 IV solver configuration. This keeps live
retrieval separate from deterministic analytics and makes saved snapshots reproducible.

## 19. Offline testing

Normal tests never require credentials or live entitlements. Static Massive-shaped JSON fixtures
cover calls, puts, multiple strikes/expirations, missing sides, crossed quotes, quote/trade times,
sizes, volume, open interest, provider reference analytics, and American exercise style. Adapter
transport is monkeypatched; `FakeMarketDataProvider` drives API tests. An autouse pytest fixture
replaces socket connection attempts with an assertion, so accidental network access fails the suite.

Tests cover timezone validation, ACT/365, midpoint and spreads, multiple simultaneous issues,
staleness and future times, price selection, Phase 4 bounds, synthetic IV/Greek recovery, partial
chain failure, summaries, smile identity, filters, parity under negative rates/dividends, resource
limits, serialization, and provider error translation.

## 20. Real-market limitations and deferred work

U.S. listed equity and ETF options are generally American-style. The raw instrument preserves the
provider's actual style. Phase 5 still reports European Black-Scholes IV as a market convention and
sets `european_bsm_approximation=true` whenever the style is not known to be European. Early
exercise and discrete dividends are not modeled.

Quotes may be stale or crossed, midpoint may not execute, last trades may be old, and provider
plans may deliver delayed or incomplete fields. Flat rate and dividend yield are simplifying
inputs. Index, ETF, settlement, multiplier, and exercise conventions can differ; Phase 5 preserves
reported metadata without inventing a full taxonomy.

Volatility-surface interpolation, SVI, arbitrage repair, local volatility, American pricing, yield
curves, dividend forecasting, and calibration are deliberately deferred. The normalized models and
enriched quote results are inputs for that future work, not implementations of it.
