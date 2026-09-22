"""Quote quality, chain enrichment, smile, filtering, and parity analytics."""

import math
from collections.abc import Sequence
from datetime import UTC, datetime

from backend.app.core.exceptions import MarketDataValidationError, QuantitativeError
from backend.app.core.numerics import finite_exp
from backend.app.models.market import MarketEnvironment
from backend.app.models.market_data import (
    MarketExerciseStyle,
    MarketPriceSource,
    OptionChainSnapshot,
    OptionQuote,
    ParityStatus,
    QuoteIssue,
    QuoteQualityConfig,
    UnderlyingQuote,
)
from backend.app.models.option import OptionContract, OptionType
from backend.app.models.results import (
    EnrichedOptionChain,
    EnrichedOptionQuote,
    MarketSmilePoint,
    MarketSmileResult,
    OptionChainSummary,
    PutCallParityPoint,
    PutCallParityResult,
)
from backend.app.models.volatility import (
    ImpliedVolatilityConfig,
    ImpliedVolatilityMarket,
    ImpliedVolatilityObservation,
    ImpliedVolatilityStatus,
)
from backend.app.pricing.black_scholes_greeks import black_scholes_greeks
from backend.app.pricing.implied_volatility import (
    european_option_price_bounds,
    forward_price,
    implied_volatility_failure_status,
    solve_implied_volatility,
)

_SECONDS_PER_ACT_365_YEAR = 365.0 * 24.0 * 60.0 * 60.0


def normalize_utc(value: datetime, *, field_name: str = "timestamp") -> datetime:
    """Require timezone awareness and return a UTC datetime."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise MarketDataValidationError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


def act_365_maturity_years(expiration: datetime, valuation_time: datetime) -> float:
    """Return non-negative ACT/365 time using exact elapsed seconds."""

    expiration_utc = normalize_utc(expiration, field_name="expiration")
    valuation_utc = normalize_utc(valuation_time, field_name="valuation_time")
    seconds = (expiration_utc - valuation_utc).total_seconds()
    return max(seconds, 0.0) / _SECONDS_PER_ACT_365_YEAR


def is_two_sided(quote: OptionQuote) -> bool:
    """Return whether bid/ask form a non-negative, non-crossed market."""

    return quote.midpoint is not None


def quote_quality_issues(
    quote: OptionQuote,
    valuation_time: datetime,
    config: QuoteQualityConfig,
) -> list[QuoteIssue]:
    """Return deterministic, potentially multiple quote-quality issues."""

    valuation_utc = normalize_utc(valuation_time, field_name="valuation_time")
    issues: list[QuoteIssue] = []
    if quote.bid is None:
        issues.append(QuoteIssue.MISSING_BID)
    if quote.ask is None:
        issues.append(QuoteIssue.MISSING_ASK)
    if any(
        value is not None and value < 0.0
        for value in (quote.bid, quote.ask, quote.last, quote.market_reference_price)
    ):
        issues.append(QuoteIssue.NEGATIVE_PRICE)
    if quote.bid is not None and quote.ask is not None and quote.ask < quote.bid:
        issues.append(QuoteIssue.CROSSED_MARKET)
    age_seconds = (valuation_utc - quote.timestamp).total_seconds()
    if age_seconds < 0.0:
        issues.append(QuoteIssue.FUTURE_TIMESTAMP)
    elif age_seconds > config.max_quote_age_seconds:
        issues.append(QuoteIssue.STALE)
    if quote.instrument.expiration < valuation_utc:
        issues.append(QuoteIssue.EXPIRED)
    return issues


def is_fresh(issues: Sequence[QuoteIssue]) -> bool:
    """Return whether issues contain no stale or future timestamp condition."""

    return QuoteIssue.STALE not in issues and QuoteIssue.FUTURE_TIMESTAMP not in issues


def select_market_price(
    quote: OptionQuote,
    issues: Sequence[QuoteIssue],
    config: QuoteQualityConfig,
) -> tuple[float | None, MarketPriceSource | None]:
    """Choose a fresh midpoint, or an explicitly enabled consistent last trade."""

    blockers = {
        QuoteIssue.CROSSED_MARKET,
        QuoteIssue.NEGATIVE_PRICE,
        QuoteIssue.STALE,
        QuoteIssue.FUTURE_TIMESTAMP,
        QuoteIssue.EXPIRED,
    }
    if any(issue in blockers for issue in issues):
        return None, None
    if (
        quote.market_price_source is MarketPriceSource.END_OF_DAY_CLOSE
        and quote.market_reference_price is not None
    ):
        return quote.market_reference_price, MarketPriceSource.END_OF_DAY_CLOSE
    if quote.midpoint is not None:
        return quote.midpoint, MarketPriceSource.MIDPOINT
    if config.allow_last_price and quote.last is not None and quote.last >= 0.0:
        return quote.last, MarketPriceSource.LAST
    return None, None


def bid_ask_spread(quote: OptionQuote) -> tuple[float | None, float | None]:
    """Return absolute and midpoint-relative spread for a valid two-sided quote."""

    if quote.midpoint is None or quote.bid is None or quote.ask is None:
        return None, None
    absolute = quote.ask - quote.bid
    relative = absolute / quote.midpoint if quote.midpoint > 0.0 else None
    return absolute, relative


def _underlying_price(quote: UnderlyingQuote) -> float:
    if quote.reference_price is not None:
        return quote.reference_price
    if quote.midpoint is not None:
        return quote.midpoint
    if quote.last is not None:
        return quote.last
    raise MarketDataValidationError(
        f"underlying {quote.symbol} has neither a valid midpoint nor last price"
    )


def _append_once(issues: list[QuoteIssue], issue: QuoteIssue) -> None:
    if issue not in issues:
        issues.append(issue)


def enrich_option_quote(
    quote: OptionQuote,
    *,
    spot: float,
    valuation_time: datetime,
    risk_free_rate: float,
    dividend_yield: float,
    quality_config: QuoteQualityConfig,
    iv_config: ImpliedVolatilityConfig,
) -> EnrichedOptionQuote:
    """Enrich one quote without allowing a bad observation to fail its chain."""

    valuation_utc = normalize_utc(valuation_time, field_name="valuation_time")
    issues = quote_quality_issues(quote, valuation_utc, quality_config)
    market_price, market_price_source = select_market_price(quote, issues, quality_config)
    if market_price is None:
        _append_once(issues, QuoteIssue.NO_MARKET_PRICE)
    maturity = act_365_maturity_years(quote.instrument.expiration, valuation_utc)
    forward = forward_price(spot, maturity, risk_free_rate, dividend_yield)
    moneyness = quote.instrument.strike / spot
    log_moneyness = math.log(quote.instrument.strike / forward)
    contract = OptionContract(
        option_type=quote.instrument.option_type,
        spot=spot,
        strike=quote.instrument.strike,
        maturity_years=maturity,
    )
    iv_market = ImpliedVolatilityMarket(
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
    )
    bounds = european_option_price_bounds(contract, iv_market)
    if market_price is not None:
        if market_price < bounds.lower - quality_config.arbitrage_tolerance:
            _append_once(issues, QuoteIssue.PRICE_BELOW_LOWER_BOUND)
        if market_price > bounds.upper + quality_config.arbitrage_tolerance:
            _append_once(issues, QuoteIssue.PRICE_ABOVE_UPPER_BOUND)

    implied_volatility = None
    iv_status = None
    analytical_greeks = None
    bound_issues = {
        QuoteIssue.PRICE_BELOW_LOWER_BOUND,
        QuoteIssue.PRICE_ABOVE_UPPER_BOUND,
    }
    if any(issue in bound_issues for issue in issues):
        iv_status = ImpliedVolatilityStatus.ARBITRAGE_VIOLATION
        _append_once(issues, QuoteIssue.IV_UNAVAILABLE)
    elif market_price is not None:
        try:
            iv_result = solve_implied_volatility(
                ImpliedVolatilityObservation(
                    contract=contract,
                    market=iv_market,
                    observed_price=market_price,
                ),
                iv_config,
            )
            iv_status = iv_result.status
            implied_volatility = iv_result.implied_volatility
            if implied_volatility is None:
                _append_once(issues, QuoteIssue.IV_UNAVAILABLE)
            else:
                analytical_greeks = black_scholes_greeks(
                    contract,
                    MarketEnvironment(
                        risk_free_rate=risk_free_rate,
                        volatility=implied_volatility,
                        dividend_yield=dividend_yield,
                    ),
                )
        except QuantitativeError as exc:
            iv_status = implied_volatility_failure_status(exc)
            _append_once(issues, QuoteIssue.IV_UNAVAILABLE)

    absolute_spread, relative_spread = bid_ask_spread(quote)
    return EnrichedOptionQuote(
        quote=quote,
        market_price=market_price,
        market_price_source=market_price_source,
        maturity_years=maturity,
        forward_price=forward,
        moneyness=moneyness,
        log_moneyness=log_moneyness,
        price_lower_bound=bounds.lower,
        price_upper_bound=bounds.upper,
        implied_volatility=implied_volatility,
        iv_status=iv_status,
        analytical_greeks=analytical_greeks,
        quality_issues=issues,
        absolute_spread=absolute_spread,
        relative_spread=relative_spread,
        european_bsm_approximation=(
            quote.instrument.exercise_style is not MarketExerciseStyle.EUROPEAN
        ),
    )


def _quote_sort_key(quote: OptionQuote) -> tuple[datetime, str, float, str]:
    instrument = quote.instrument
    return (
        instrument.expiration,
        instrument.option_type.value,
        instrument.strike,
        instrument.provider_symbol,
    )


def enrich_option_chain(
    snapshot: OptionChainSnapshot,
    *,
    valuation_time: datetime | None = None,
    risk_free_rate: float,
    dividend_yield: float,
    quality_config: QuoteQualityConfig | None = None,
    iv_config: ImpliedVolatilityConfig | None = None,
) -> EnrichedOptionChain:
    """Return expiration/type/strike-sorted analytics and factual summary counts."""

    valuation_utc = normalize_utc(
        valuation_time or snapshot.retrieved_at,
        field_name="valuation_time",
    )
    quality = quality_config or QuoteQualityConfig()
    solver = iv_config or ImpliedVolatilityConfig()
    spot = _underlying_price(snapshot.underlying_quote)
    enriched = [
        enrich_option_quote(
            quote,
            spot=spot,
            valuation_time=valuation_utc,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            quality_config=quality,
            iv_config=solver,
        )
        for quote in sorted(snapshot.quotes, key=_quote_sort_key)
    ]
    arbitrage_issues = {
        QuoteIssue.PRICE_BELOW_LOWER_BOUND,
        QuoteIssue.PRICE_ABOVE_UPPER_BOUND,
    }
    valid_quotes = sum(not item.quality_issues for item in enriched)
    summary = OptionChainSummary(
        total_contracts=len(enriched),
        valid_quotes=valid_quotes,
        invalid_quotes=len(enriched) - valid_quotes,
        quotes_with_midpoint=sum(item.quote.midpoint is not None for item in enriched),
        quotes_with_recovered_iv=sum(item.implied_volatility is not None for item in enriched),
        arbitrage_violations=sum(
            any(issue in arbitrage_issues for issue in item.quality_issues) for item in enriched
        ),
        stale_quotes=sum(QuoteIssue.STALE in item.quality_issues for item in enriched),
        calls_count=sum(item.quote.instrument.option_type is OptionType.CALL for item in enriched),
        puts_count=sum(item.quote.instrument.option_type is OptionType.PUT for item in enriched),
        expiration_count=len({item.quote.instrument.expiration for item in enriched}),
    )
    return EnrichedOptionChain(
        underlying_symbol=snapshot.underlying_symbol,
        underlying_quote=snapshot.underlying_quote,
        valuation_time=valuation_utc,
        source=snapshot.source,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        quotes=enriched,
        summary=summary,
    )


def extract_market_smile(
    chain: EnrichedOptionChain,
    *,
    expiration: datetime,
    option_type: OptionType,
) -> MarketSmileResult:
    """Return one call or put slice without smoothing or interpolation."""

    expiration_utc = normalize_utc(expiration, field_name="expiration")
    selected = [
        item
        for item in chain.quotes
        if item.quote.instrument.expiration == expiration_utc
        and item.quote.instrument.option_type is option_type
    ]
    points = [
        MarketSmilePoint(
            strike=item.quote.instrument.strike,
            option_type=option_type,
            market_price=item.market_price,
            implied_volatility=item.implied_volatility,
            moneyness=item.moneyness,
            log_moneyness=item.log_moneyness,
            bid=item.quote.bid,
            ask=item.quote.ask,
            quality_issues=item.quality_issues,
        )
        for item in selected
    ]
    return MarketSmileResult(
        underlying_symbol=chain.underlying_symbol,
        expiration=expiration_utc,
        option_type=option_type,
        points=points,
    )


def analyze_put_call_parity(
    snapshot: OptionChainSnapshot,
    *,
    valuation_time: datetime | None = None,
    risk_free_rate: float,
    dividend_yield: float,
    quality_config: QuoteQualityConfig | None = None,
) -> PutCallParityResult:
    """Compare matched fresh call/put midpoints against European parity."""

    valuation_utc = normalize_utc(
        valuation_time or snapshot.retrieved_at,
        field_name="valuation_time",
    )
    quality = quality_config or QuoteQualityConfig()
    spot = _underlying_price(snapshot.underlying_quote)
    matched: dict[tuple[datetime, float], dict[OptionType, OptionQuote]] = {}
    all_keys: dict[tuple[datetime, float], int] = {}
    for quote in sorted(snapshot.quotes, key=_quote_sort_key):
        key = (quote.instrument.expiration, quote.instrument.strike)
        all_keys[key] = all_keys.get(key, 0) + 1
        issues = quote_quality_issues(quote, valuation_utc, quality)
        if quote.midpoint is None or issues:
            continue
        matched.setdefault(key, {}).setdefault(quote.instrument.option_type, quote)

    points: list[PutCallParityPoint] = []
    paired_keys: set[tuple[datetime, float]] = set()
    for (expiration, strike), sides in sorted(matched.items()):
        call = sides.get(OptionType.CALL)
        put = sides.get(OptionType.PUT)
        if call is None or put is None or call.midpoint is None or put.midpoint is None:
            continue
        paired_keys.add((expiration, strike))
        maturity = act_365_maturity_years(expiration, valuation_utc)
        theoretical = spot * finite_exp(
            -dividend_yield * maturity,
            context="parity dividend discount",
        ) - strike * finite_exp(
            -risk_free_rate * maturity,
            context="parity risk-free discount",
        )
        observed = call.midpoint - put.midpoint
        residual = observed - theoretical
        relative = residual / abs(theoretical) if abs(theoretical) > 1e-15 else None
        tolerance = quality.parity_absolute_tolerance + (
            quality.parity_relative_tolerance * abs(theoretical)
        )
        parity_status = (
            ParityStatus.WITHIN_TOLERANCE if abs(residual) <= tolerance else ParityStatus.VIOLATION
        )
        points.append(
            PutCallParityPoint(
                strike=strike,
                expiration=expiration,
                maturity_years=maturity,
                call_price=call.midpoint,
                put_price=put.midpoint,
                theoretical_difference=theoretical,
                observed_difference=observed,
                residual=residual,
                relative_residual=relative,
                status=parity_status,
            )
        )
    unmatched_contracts = sum(count for key, count in all_keys.items() if key not in paired_keys)
    return PutCallParityResult(
        underlying_symbol=snapshot.underlying_symbol,
        valuation_time=valuation_utc,
        points=points,
        matched_pairs=len(points),
        violations=sum(point.status is ParityStatus.VIOLATION for point in points),
        unmatched_contracts=unmatched_contracts,
    )


def filter_enriched_quotes(
    quotes: Sequence[EnrichedOptionQuote],
    *,
    expiration: datetime | None = None,
    option_type: OptionType | None = None,
    min_strike: float | None = None,
    max_strike: float | None = None,
    min_moneyness: float | None = None,
    max_moneyness: float | None = None,
    valid_midpoint_only: bool = False,
    iv_available_only: bool = False,
    fresh_only: bool = False,
) -> list[EnrichedOptionQuote]:
    """Apply simple deterministic Phase 5 filters to enriched quotes."""

    expiration_utc = None if expiration is None else normalize_utc(expiration)
    selected: list[EnrichedOptionQuote] = []
    for item in quotes:
        instrument = item.quote.instrument
        if expiration_utc is not None and instrument.expiration != expiration_utc:
            continue
        if option_type is not None and instrument.option_type is not option_type:
            continue
        if min_strike is not None and instrument.strike < min_strike:
            continue
        if max_strike is not None and instrument.strike > max_strike:
            continue
        if min_moneyness is not None and (item.moneyness is None or item.moneyness < min_moneyness):
            continue
        if max_moneyness is not None and (item.moneyness is None or item.moneyness > max_moneyness):
            continue
        if valid_midpoint_only and item.quote.midpoint is None:
            continue
        if iv_available_only and item.implied_volatility is None:
            continue
        if fresh_only and not is_fresh(item.quality_issues):
            continue
        selected.append(item)
    return sorted(selected, key=lambda item: _quote_sort_key(item.quote))


def is_arbitrage_valid(item: EnrichedOptionQuote) -> bool:
    """Return whether an enriched quote has no Phase 4 price-bound issue."""

    return not any(
        issue
        in {
            QuoteIssue.PRICE_BELOW_LOWER_BOUND,
            QuoteIssue.PRICE_ABOVE_UPPER_BOUND,
        }
        for issue in item.quality_issues
    )
