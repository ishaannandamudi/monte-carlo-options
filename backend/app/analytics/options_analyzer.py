"""High-level American equity-options analysis over existing quantitative engines."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from time import perf_counter

from backend.app.analytics.market_data import enrich_option_chain
from backend.app.core.config import Settings
from backend.app.core.exceptions import MarketDataValidationError, QuantitativeError
from backend.app.market_data.base import MarketDataProvider
from backend.app.models.binomial import (
    AmericanBinomialPricingResult,
    BinomialTreeConfig,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.market_data import (
    ComparisonPriceSource,
    MarketDataMode,
    MarketExerciseStyle,
    MarketPriceSource,
    OptionChainSnapshot,
    OptionQuote,
    QuoteIssue,
    QuoteQualityConfig,
    UnderlyingQuote,
)
from backend.app.models.option import ExerciseStyle, OptionContract, OptionType
from backend.app.models.options_analyzer import (
    AnalysisAssumptions,
    AnalysisDataQuality,
    AnalysisOptionType,
    AnalyzedOptionContract,
    AnalyzerSmilePoint,
    AnalyzerTiming,
    AnalyzerUnderlyingQuote,
    FairValueVolatilitySource,
    OptionsAnalysisRequest,
    OptionsAnalysisResult,
    OptionsAnalysisSummary,
)
from backend.app.models.results import EnrichedOptionChain, EnrichedOptionQuote
from backend.app.models.volatility import ImpliedVolatilityConfig
from backend.app.models.volatility_surface import (
    ExtrapolationPolicy,
    SmileInterpolationConfig,
    SmileInterpolationMethod,
)
from backend.app.payoffs.vanilla import intrinsic_value
from backend.app.pricing.binomial import binomial_price
from backend.app.volatility_surface.core import build_smile, make_volatility_point, query_smile

_VOLATILITY_BUMP = 1e-4
_RATE_BUMP = 1e-4


@dataclass(frozen=True)
class _FairVolatilityEstimate:
    value: float | None
    source: FairValueVolatilitySource
    independent: bool
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _AmericanAnalytics:
    result: AmericanBinomialPricingResult
    vega: float
    rho: float
    price_runtime_ms: float
    greeks_runtime_ms: float
    runtime_ms: float


def select_default_expiration(
    expirations: Sequence[date],
    valuation_date: date,
    target_days: int,
) -> date:
    """Select the future expiration closest to the configured target deterministically."""

    if target_days <= 0:
        raise ValueError("target expiration days must be positive")
    future = sorted({expiration for expiration in expirations if expiration > valuation_date})
    if not future:
        raise MarketDataValidationError("no future option expiration is available for this symbol")
    return min(
        future,
        key=lambda expiration: (
            abs((expiration - valuation_date).days - target_days),
            expiration,
        ),
    )


def _underlying_price(quote: UnderlyingQuote) -> float:
    price = quote.reference_price
    if price is None:
        price = quote.midpoint if quote.midpoint is not None else quote.last
    if price is None or not math.isfinite(price) or price <= 0.0:
        raise MarketDataValidationError(
            f"underlying {quote.symbol} has neither a valid midpoint nor last price"
        )
    return price


def _raw_quote_rank(
    quote: OptionQuote,
    spot: float,
    valuation_time: datetime,
) -> tuple[object, ...]:
    age = (valuation_time - quote.timestamp).total_seconds()
    midpoint_missing = quote.midpoint is None and quote.market_reference_price is None
    relative_spread = (
        math.inf
        if quote.midpoint is None or quote.midpoint <= 0.0 or quote.bid is None or quote.ask is None
        else (quote.ask - quote.bid) / quote.midpoint
    )
    liquidity = (quote.volume or 0.0) + float(quote.open_interest or 0)
    return (
        midpoint_missing,
        age < 0.0,
        relative_spread,
        abs(math.log(quote.instrument.strike / spot)),
        -liquidity,
        quote.instrument.provider_symbol,
    )


def _requested_side(option_type: AnalysisOptionType, quote: OptionQuote) -> bool:
    if option_type is AnalysisOptionType.BOTH:
        return True
    return quote.instrument.option_type.value == option_type.value


def bounded_source_quotes(
    quotes: Sequence[OptionQuote],
    *,
    spot: float,
    expiration: date,
    option_type: AnalysisOptionType,
    strike_window_pct: float,
    max_contracts: int,
    valuation_time: datetime,
) -> list[OptionQuote]:
    """Bound enrichment work while retaining liquid near-ATM neighbors on both sides."""

    lower = spot * (1.0 - strike_window_pct)
    upper = spot * (1.0 + strike_window_pct)
    candidates = [
        quote
        for quote in quotes
        if quote.instrument.expiration.date() == expiration
        and lower <= quote.instrument.strike <= upper
        and _requested_side(option_type, quote)
    ]
    if not candidates:
        raise MarketDataValidationError(
            "no option contracts fall inside the requested expiration and strike window"
        )
    per_side_limit = max(6, max_contracts * 2)
    selected: list[OptionQuote] = []
    for side in (OptionType.CALL, OptionType.PUT):
        if option_type is not AnalysisOptionType.BOTH and side.value != option_type.value:
            continue
        side_quotes = [quote for quote in candidates if quote.instrument.option_type is side]
        selected.extend(
            sorted(side_quotes, key=lambda quote: _raw_quote_rank(quote, spot, valuation_time))[
                :per_side_limit
            ]
        )
    return sorted(
        selected,
        key=lambda quote: (
            quote.instrument.option_type.value,
            quote.instrument.strike,
            quote.instrument.provider_symbol,
        ),
    )


def _enriched_rank(item: EnrichedOptionQuote) -> tuple[object, ...]:
    issues = set(item.quality_issues)
    liquidity = (item.quote.volume or 0.0) + float(item.quote.open_interest or 0)
    return (
        item.market_price is None,
        QuoteIssue.STALE in issues or QuoteIssue.FUTURE_TIMESTAMP in issues,
        item.relative_spread if item.relative_spread is not None else math.inf,
        abs(item.log_moneyness or 0.0),
        -liquidity,
        item.quote.instrument.provider_symbol,
    )


def select_display_contracts(
    quotes: Sequence[EnrichedOptionQuote],
    option_type: AnalysisOptionType,
    max_contracts: int,
) -> list[EnrichedOptionQuote]:
    """Return a balanced, deterministic, liquidity-aware call/put selection."""

    if max_contracts <= 0:
        raise ValueError("max contracts must be positive")
    calls = sorted(
        (item for item in quotes if item.quote.instrument.option_type is OptionType.CALL),
        key=_enriched_rank,
    )
    puts = sorted(
        (item for item in quotes if item.quote.instrument.option_type is OptionType.PUT),
        key=_enriched_rank,
    )
    if option_type is AnalysisOptionType.CALL:
        selected = calls[:max_contracts]
    elif option_type is AnalysisOptionType.PUT:
        selected = puts[:max_contracts]
    else:
        call_quota = (max_contracts + 1) // 2
        put_quota = max_contracts // 2
        selected = calls[:call_quota] + puts[:put_quota]
        used_symbols = {item.quote.instrument.provider_symbol for item in selected}
        if len(selected) < max_contracts:
            remainder = sorted(
                (
                    item
                    for item in (*calls[call_quota:], *puts[put_quota:])
                    if item.quote.instrument.provider_symbol not in used_symbols
                ),
                key=_enriched_rank,
            )
            selected.extend(remainder[: max_contracts - len(selected)])
    return sorted(
        selected,
        key=lambda item: (
            item.quote.instrument.option_type.value,
            item.quote.instrument.strike,
            item.quote.instrument.provider_symbol,
        ),
    )


def _eligible_volatility_point(item: EnrichedOptionQuote, maximum_spread: float) -> bool:
    blockers = {
        QuoteIssue.CROSSED_MARKET,
        QuoteIssue.NEGATIVE_PRICE,
        QuoteIssue.STALE,
        QuoteIssue.FUTURE_TIMESTAMP,
        QuoteIssue.EXPIRED,
        QuoteIssue.PRICE_BELOW_LOWER_BOUND,
        QuoteIssue.PRICE_ABOVE_UPPER_BOUND,
    }
    return (
        item.implied_volatility is not None
        and item.forward_price is not None
        and not any(issue in blockers for issue in item.quality_issues)
        and (
            item.market_price_source is MarketPriceSource.END_OF_DAY_CLOSE
            or (item.relative_spread is not None and item.relative_spread <= maximum_spread)
        )
    )


def fair_value_volatility(
    target: EnrichedOptionQuote,
    candidates: Sequence[EnrichedOptionQuote],
    *,
    spot: float,
    risk_free_rate: float,
    dividend_yield: float,
    maximum_spread: float,
) -> _FairVolatilityEstimate:
    """Estimate fair-value volatility without reusing the target when neighbors exist."""

    target_instrument = target.quote.instrument
    target_coordinate = target.log_moneyness
    maturity = target.maturity_years
    forward = target.forward_price
    if target_coordinate is None or forward is None or maturity <= 0.0:
        return _FairVolatilityEstimate(
            value=None,
            source=FairValueVolatilitySource.UNAVAILABLE,
            independent=False,
            warnings=("FAIR_VALUE_VOLATILITY_UNAVAILABLE",),
        )

    same_side_by_strike: dict[float, EnrichedOptionQuote] = {}
    all_independent: list[EnrichedOptionQuote] = []
    for item in candidates:
        instrument = item.quote.instrument
        if instrument.provider_symbol == target_instrument.provider_symbol:
            continue
        if math.isclose(instrument.strike, target_instrument.strike, abs_tol=1e-12):
            continue
        if not _eligible_volatility_point(item, maximum_spread):
            continue
        all_independent.append(item)
        if instrument.option_type is target_instrument.option_type:
            existing = same_side_by_strike.get(instrument.strike)
            if existing is None or _enriched_rank(item) < _enriched_rank(existing):
                same_side_by_strike[instrument.strike] = item

    same_side = sorted(
        same_side_by_strike.values(),
        key=lambda item: item.quote.instrument.strike,
    )
    if len(same_side) >= 2:
        lower = min(item.log_moneyness or 0.0 for item in same_side)
        upper = max(item.log_moneyness or 0.0 for item in same_side)
        if lower <= target_coordinate <= upper:
            points = [
                make_volatility_point(
                    strike=item.quote.instrument.strike,
                    maturity_years=maturity,
                    implied_volatility=float(item.implied_volatility),
                    option_type=item.quote.instrument.option_type,
                    spot=spot,
                    forward=forward,
                    market_price=item.market_price,
                    bid=item.quote.bid,
                    ask=item.quote.ask,
                    source=item.quote.source.value,
                    expiration=item.quote.instrument.expiration,
                )
                for item in same_side
                if item.implied_volatility is not None
            ]
            try:
                smile = build_smile(
                    points=points,
                    maturity_years=maturity,
                    spot=spot,
                    risk_free_rate=risk_free_rate,
                    dividend_yield=dividend_yield,
                    underlying_symbol=target_instrument.underlying_symbol,
                    expiration=target_instrument.expiration,
                    minimum_points=2,
                    source="LEAVE_ONE_OUT_MARKET_SMILE",
                ).smile
                result = query_smile(
                    smile,
                    log_moneyness=target_coordinate,
                    config=SmileInterpolationConfig(
                        method=SmileInterpolationMethod.PCHIP,
                        extrapolation=ExtrapolationPolicy.NONE,
                    ),
                )
                return _FairVolatilityEstimate(
                    value=result.implied_volatility,
                    source=FairValueVolatilitySource.SMILE_INTERPOLATED,
                    independent=True,
                )
            except QuantitativeError:
                pass

    if all_independent:
        atm = min(
            all_independent,
            key=lambda item: (
                abs(item.log_moneyness or 0.0),
                _enriched_rank(item),
            ),
        )
        return _FairVolatilityEstimate(
            value=atm.implied_volatility,
            source=FairValueVolatilitySource.ATM_REFERENCE,
            independent=True,
            warnings=("INSUFFICIENT_NEIGHBORS_FOR_INTERPOLATION",),
        )

    if target.implied_volatility is not None:
        return _FairVolatilityEstimate(
            value=target.implied_volatility,
            source=FairValueVolatilitySource.CONTRACT_IV_FALLBACK,
            independent=False,
            warnings=("FAIR_VALUE_VOLATILITY_NOT_INDEPENDENT",),
        )
    return _FairVolatilityEstimate(
        value=None,
        source=FairValueVolatilitySource.UNAVAILABLE,
        independent=False,
        warnings=("FAIR_VALUE_VOLATILITY_UNAVAILABLE",),
    )


def _american_result(
    contract: OptionContract,
    market: MarketEnvironment,
    tree: BinomialTreeConfig,
) -> AmericanBinomialPricingResult:
    result = binomial_price(contract, market, tree)
    if not isinstance(result, AmericanBinomialPricingResult):
        raise RuntimeError("American analyzer unexpectedly received a European tree result")
    return result


def _american_analytics(
    contract: OptionContract,
    market: MarketEnvironment,
    tree: BinomialTreeConfig,
) -> _AmericanAnalytics:
    started = perf_counter()
    base = _american_result(contract, market, tree)
    price_completed = perf_counter()
    up_vol = _american_result(
        contract,
        market.model_copy(update={"volatility": market.volatility + _VOLATILITY_BUMP}),
        tree,
    ).price
    if market.volatility >= _VOLATILITY_BUMP:
        down_vol = _american_result(
            contract,
            market.model_copy(update={"volatility": market.volatility - _VOLATILITY_BUMP}),
            tree,
        ).price
        vega = (up_vol - down_vol) / (2.0 * _VOLATILITY_BUMP)
    else:
        vega = (up_vol - base.price) / _VOLATILITY_BUMP
    up_rate = _american_result(
        contract,
        market.model_copy(update={"risk_free_rate": market.risk_free_rate + _RATE_BUMP}),
        tree,
    ).price
    down_rate = _american_result(
        contract,
        market.model_copy(update={"risk_free_rate": market.risk_free_rate - _RATE_BUMP}),
        tree,
    ).price
    rho = (up_rate - down_rate) / (2.0 * _RATE_BUMP)
    if not math.isfinite(vega) or not math.isfinite(rho):
        raise MarketDataValidationError("American finite-difference Greeks are non-finite")
    completed = perf_counter()
    return _AmericanAnalytics(
        result=base,
        vega=vega,
        rho=rho,
        price_runtime_ms=(price_completed - started) * 1_000.0,
        greeks_runtime_ms=(completed - price_completed) * 1_000.0,
        runtime_ms=(completed - started) * 1_000.0,
    )


def _quality_flags(
    item: EnrichedOptionQuote,
    maximum_spread: float,
    valuation_time: datetime,
) -> list[str]:
    flags = [issue.value for issue in item.quality_issues]
    if item.relative_spread is not None and item.relative_spread > maximum_spread:
        flags.append("WIDE_SPREAD")
    if not item.quote.volume:
        flags.append("NO_RECORDED_VOLUME")
    if not item.quote.open_interest:
        flags.append("NO_OPEN_INTEREST")
    if (not item.quote.volume and not item.quote.open_interest) or "WIDE_SPREAD" in flags:
        flags.append("ILLIQUID")
    if item.quote.market_data_mode is MarketDataMode.END_OF_DAY:
        flags.extend(
            [
                "END_OF_DAY_DATA",
                "SNAPSHOT_UNAVAILABLE_FOR_PLAN",
                "NO_LIVE_BID_ASK",
                "LOW_PROVIDER_RATE_LIMIT",
            ]
        )
        if item.quote.data_date is not None and item.quote.data_date != valuation_time.date():
            flags.append("DATA_DATE_MISMATCH")
    return list(dict.fromkeys(flags))


def _analysis_quality(
    quality_flags: Sequence[str],
    fair_volatility: _FairVolatilityEstimate,
    fair_value: float | None,
) -> AnalysisDataQuality:
    low = {
        QuoteIssue.CROSSED_MARKET.value,
        QuoteIssue.NEGATIVE_PRICE.value,
        QuoteIssue.STALE.value,
        QuoteIssue.FUTURE_TIMESTAMP.value,
        QuoteIssue.NO_MARKET_PRICE.value,
        "WIDE_SPREAD",
    }
    if (
        fair_value is None
        or not fair_volatility.independent
        or any(flag in low for flag in quality_flags)
    ):
        return AnalysisDataQuality.LOW_DATA_QUALITY
    if (
        "ILLIQUID" in quality_flags
        or fair_volatility.source is FairValueVolatilitySource.ATM_REFERENCE
    ):
        return AnalysisDataQuality.MODERATE_DATA_QUALITY
    return AnalysisDataQuality.HIGH_DATA_QUALITY


def _analyze_contract(
    item: EnrichedOptionQuote,
    fair_volatility: _FairVolatilityEstimate,
    *,
    spot: float,
    valuation_time: datetime,
    risk_free_rate: float,
    dividend_yield: float,
    tree_steps: int,
    maximum_spread: float,
) -> AnalyzedOptionContract:
    quote = item.quote
    instrument = quote.instrument
    eod_mode = quote.market_data_mode is MarketDataMode.END_OF_DAY
    iv_method = "EOD_BSM_EQUIVALENT_IV" if eod_mode else "BSM_EQUIVALENT"
    warnings = [iv_method, *fair_volatility.warnings]
    if instrument.exercise_style is not MarketExerciseStyle.AMERICAN:
        warnings.append("AMERICAN_EXERCISE_ASSUMED_FOR_EQUITY_WORKFLOW")
    analytics: _AmericanAnalytics | None = None
    if fair_volatility.value is not None:
        try:
            analytics = _american_analytics(
                OptionContract(
                    option_type=instrument.option_type,
                    exercise_style=ExerciseStyle.AMERICAN,
                    spot=spot,
                    strike=instrument.strike,
                    maturity_years=item.maturity_years,
                ),
                MarketEnvironment(
                    risk_free_rate=risk_free_rate,
                    volatility=fair_volatility.value,
                    dividend_yield=dividend_yield,
                ),
                BinomialTreeConfig(steps=tree_steps),
            )
        except QuantitativeError:
            warnings.append("AMERICAN_PRICING_FAILED")
    fair_value = None if analytics is None else analytics.result.price
    market_midpoint = quote.midpoint
    market_reference_price = item.market_price
    comparison_price = item.market_price if eod_mode else market_midpoint
    if comparison_price is None:
        comparison_source = None
    elif item.market_price_source is MarketPriceSource.END_OF_DAY_CLOSE:
        comparison_source = ComparisonPriceSource.END_OF_DAY_CLOSE
    elif item.market_price_source is MarketPriceSource.MIDPOINT:
        comparison_source = ComparisonPriceSource.BID_ASK_MIDPOINT
    elif item.market_price_source is MarketPriceSource.LAST:
        comparison_source = ComparisonPriceSource.LAST_TRADE
    else:
        comparison_source = None
    gap = None if fair_value is None or comparison_price is None else fair_value - comparison_price
    gap_pct = (
        None
        if gap is None or comparison_price is None or comparison_price <= 0.0
        else gap / comparison_price
    )
    intrinsic = intrinsic_value(instrument.option_type, spot, instrument.strike)
    market_reference = item.market_price
    time_value = None if market_reference is None else max(market_reference - intrinsic, 0.0)
    quality_flags = _quality_flags(item, maximum_spread, valuation_time)
    quote_age = (valuation_time - quote.timestamp).total_seconds()
    return AnalyzedOptionContract(
        provider_symbol=instrument.provider_symbol,
        option_type=instrument.option_type,
        strike=instrument.strike,
        expiration=instrument.expiration.date(),
        provider_exercise_style=instrument.exercise_style,
        bid=quote.bid,
        ask=quote.ask,
        market_midpoint=market_midpoint,
        market_price_source=item.market_price_source,
        market_reference_price=market_reference_price,
        comparison_price=comparison_price,
        comparison_price_source=comparison_source,
        last=quote.last,
        bid_size=quote.bid_size,
        ask_size=quote.ask_size,
        volume=quote.volume,
        open_interest=quote.open_interest,
        quote_timestamp=quote.timestamp,
        quote_age_seconds=quote_age,
        days_to_expiration=max((instrument.expiration.date() - valuation_time.date()).days, 0),
        time_to_expiration_years=item.maturity_years,
        implied_volatility=item.implied_volatility,
        implied_volatility_status=item.iv_status,
        implied_volatility_method=iv_method,
        provider_implied_volatility=quote.provider_implied_volatility,
        fair_value_volatility=fair_volatility.value,
        fair_value_volatility_source=fair_volatility.source,
        fair_value_volatility_independent=fair_volatility.independent,
        fair_value=fair_value,
        fair_value_gap=gap,
        fair_value_gap_pct=gap_pct,
        delta=None if analytics is None else analytics.result.delta,
        gamma=None if analytics is None else analytics.result.gamma,
        theta=None if analytics is None else analytics.result.theta,
        theta_daily=(
            None
            if analytics is None or analytics.result.theta is None
            else analytics.result.theta / 365.0
        ),
        vega=None if analytics is None else analytics.vega,
        vega_per_vol_point=None if analytics is None else analytics.vega * 0.01,
        rho=None if analytics is None else analytics.rho,
        intrinsic_value=intrinsic,
        time_value=time_value,
        moneyness=instrument.strike / spot,
        log_moneyness=float(item.log_moneyness or 0.0),
        spread=item.absolute_spread,
        spread_pct=item.relative_spread,
        european_reference_value=(
            None if analytics is None else analytics.result.black_scholes_european_price
        ),
        early_exercise_premium=(
            None if analytics is None else analytics.result.early_exercise_premium
        ),
        tree_steps=tree_steps,
        crr_pricing_runtime_ms=None if analytics is None else analytics.price_runtime_ms,
        greeks_runtime_ms=None if analytics is None else analytics.greeks_runtime_ms,
        model_runtime_ms=None if analytics is None else analytics.runtime_ms,
        analysis_quality=_analysis_quality(quality_flags, fair_volatility, fair_value),
        quality_flags=quality_flags,
        warning_flags=list(dict.fromkeys(warnings)),
        market_data_mode=quote.market_data_mode,
        eod_open=quote.eod_open,
        eod_high=quote.eod_high,
        eod_low=quote.eod_low,
        eod_close=quote.eod_close,
        eod_vwap=quote.eod_vwap,
        data_date=quote.data_date,
    )


def _bounded_expirations(
    expirations: Sequence[date], selected: date, limit: int
) -> tuple[list[date], bool]:
    ordered = sorted(set(expirations))
    if len(ordered) <= limit:
        return ordered, False
    bounded = ordered[:limit]
    if selected not in bounded:
        bounded[-1] = selected
        bounded.sort()
    return bounded, True


def analyze_american_options(
    request: OptionsAnalysisRequest,
    *,
    provider: MarketDataProvider,
    settings: Settings,
) -> OptionsAnalysisResult:
    """Fetch, clean, interpolate, and price one bounded American equity-option chain."""

    started = perf_counter()
    if request.max_contracts > settings.max_analysis_contracts:
        raise MarketDataValidationError(
            f"max_contracts exceeds configured limit of {settings.max_analysis_contracts}"
        )
    tree_steps = request.tree_steps or settings.default_analysis_tree_steps
    if tree_steps > settings.max_binomial_steps:
        raise MarketDataValidationError(
            f"tree_steps exceeds configured binomial limit of {settings.max_binomial_steps}"
        )
    target_days = request.target_expiration_days or settings.default_analysis_target_days
    strike_window = request.strike_window_pct or settings.default_analysis_strike_window_pct
    risk_free_rate = (
        settings.default_analysis_risk_free_rate
        if request.risk_free_rate is None
        else request.risk_free_rate
    )
    dividend_yield = (
        settings.default_analysis_dividend_yield
        if request.dividend_yield is None
        else request.dividend_yield
    )
    max_quote_age = request.max_quote_age_seconds or settings.max_quote_age_seconds
    maximum_spread = settings.default_analysis_max_relative_spread

    market_started = perf_counter()
    underlying_quote = provider.get_underlying_quote(request.symbol)
    valuation_date = (
        underlying_quote.data_date
        if underlying_quote.market_data_mode is MarketDataMode.END_OF_DAY
        and underlying_quote.data_date is not None
        else datetime.now(UTC).date()
    )
    target_date = request.expiration or (valuation_date + timedelta(days=target_days))
    expirations = provider.list_option_expirations(
        request.symbol,
        target_date=target_date,
        strike_window_pct=strike_window,
    )
    if request.expiration is not None:
        if request.expiration not in expirations:
            raise MarketDataValidationError(
                f"expiration {request.expiration.isoformat()} is unavailable for {request.symbol}"
            )
        selected_expiration = request.expiration
    else:
        selected_expiration = select_default_expiration(expirations, valuation_date, target_days)
    requested_side = (
        None
        if request.option_type is AnalysisOptionType.BOTH
        else OptionType(request.option_type.value)
    )
    snapshot = provider.get_option_chain(
        request.symbol,
        selected_expiration,
        option_type=requested_side,
        max_contracts=request.max_contracts,
    )
    if len(snapshot.quotes) > settings.max_option_chain_contracts:
        raise MarketDataValidationError(
            "option chain exceeds configured limit of "
            f"{settings.max_option_chain_contracts} contracts"
        )
    valuation_time = snapshot.retrieved_at.astimezone(UTC)
    underlying_quote = snapshot.underlying_quote
    spot = _underlying_price(underlying_quote)
    source_quotes = bounded_source_quotes(
        snapshot.quotes,
        spot=spot,
        expiration=selected_expiration,
        option_type=request.option_type,
        strike_window_pct=strike_window,
        max_contracts=request.max_contracts,
        valuation_time=valuation_time,
    )
    bounded_snapshot = OptionChainSnapshot(
        underlying_symbol=snapshot.underlying_symbol,
        underlying_quote=underlying_quote,
        retrieved_at=valuation_time,
        source=snapshot.source,
        quotes=source_quotes,
        available_expirations=[selected_expiration],
        market_data_mode=snapshot.market_data_mode,
        capabilities=snapshot.capabilities,
    )
    market_data_ms = (perf_counter() - market_started) * 1_000.0

    iv_started = perf_counter()
    enriched: EnrichedOptionChain = enrich_option_chain(
        bounded_snapshot,
        valuation_time=valuation_time,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        quality_config=QuoteQualityConfig(
            max_quote_age_seconds=max_quote_age,
            allow_last_price=request.allow_last_price_fallback,
        ),
        iv_config=ImpliedVolatilityConfig(),
    )
    implied_volatility_ms = (perf_counter() - iv_started) * 1_000.0
    selected = select_display_contracts(enriched.quotes, request.option_type, request.max_contracts)
    if not selected:
        raise MarketDataValidationError("no contracts remain after bounded chain filtering")

    smile_started = perf_counter()
    fair_volatilities = {
        item.quote.instrument.provider_symbol: fair_value_volatility(
            item,
            enriched.quotes,
            spot=spot,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            maximum_spread=maximum_spread,
        )
        for item in selected
    }
    smile_construction_ms = (perf_counter() - smile_started) * 1_000.0

    pricing_started = perf_counter()
    analyzed = [
        _analyze_contract(
            item,
            fair_volatilities[item.quote.instrument.provider_symbol],
            spot=spot,
            valuation_time=valuation_time,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            tree_steps=tree_steps,
            maximum_spread=maximum_spread,
        )
        for item in selected
    ]
    pricing_and_greeks_ms = (perf_counter() - pricing_started) * 1_000.0
    bounded_expirations, expirations_truncated = _bounded_expirations(
        expirations, selected_expiration, settings.max_analysis_expirations
    )
    atm_strike = min(
        (item.quote.instrument.strike for item in selected),
        key=lambda strike: (abs(strike - spot), strike),
    )
    underlying_age = (valuation_time - underlying_quote.timestamp).total_seconds()
    warnings: list[str] = []
    if snapshot.market_data_mode is MarketDataMode.END_OF_DAY:
        warnings.append(
            "Using Massive end-of-day data because snapshot access is not included in the "
            "current provider plan."
        )
        warnings.append(
            f"Massive Basic analysis is limited to {snapshot.capabilities.basic_contract_limit} "
            "near-ATM contract observations per cold request."
        )
    elif underlying_age > max_quote_age:
        warnings.append(
            "Delayed/stale market data: the underlying quote exceeds the age threshold."
        )
    if underlying_age < 0.0:
        warnings.append("The underlying quote timestamp is later than the provider retrieval time.")
    if expirations_truncated:
        warnings.append(
            f"Available expirations were truncated to {settings.max_analysis_expirations} entries."
        )
    if any(QuoteIssue.STALE.value in contract.quality_flags for contract in analyzed):
        warnings.append("Delayed/stale market data is present in the selected option contracts.")
    if any("DATA_DATE_MISMATCH" in contract.quality_flags for contract in analyzed):
        warnings.append(
            "One or more option bars are not aligned with the underlying end-of-day date."
        )
    if not any(contract.fair_value_volatility_independent for contract in analyzed):
        warnings.append(
            "No independent neighboring volatility estimate was available; "
            "fair-value gaps are low-confidence diagnostics."
        )
    total_ms = (perf_counter() - started) * 1_000.0
    return OptionsAnalysisResult(
        symbol=request.symbol,
        data_source=snapshot.source,
        data_mode=snapshot.market_data_mode,
        price_as_of=valuation_time,
        underlying_price_source=underlying_quote.price_source,
        option_price_source=(
            ComparisonPriceSource.END_OF_DAY_CLOSE
            if snapshot.market_data_mode is MarketDataMode.END_OF_DAY
            else next(
                (
                    contract.comparison_price_source
                    for contract in analyzed
                    if contract.comparison_price_source is not None
                ),
                ComparisonPriceSource.BID_ASK_MIDPOINT,
            )
        ),
        provider_capabilities=snapshot.capabilities,
        underlying=AnalyzerUnderlyingQuote(
            price=spot,
            bid=underlying_quote.bid,
            ask=underlying_quote.ask,
            last=underlying_quote.last,
            timestamp=underlying_quote.timestamp,
            data_age_seconds=underlying_age,
            source=underlying_quote.source,
            price_source=underlying_quote.price_source,
            open=underlying_quote.open,
            high=underlying_quote.high,
            low=underlying_quote.low,
            close=underlying_quote.close,
            vwap=underlying_quote.vwap,
            volume=underlying_quote.volume,
            data_date=underlying_quote.data_date,
        ),
        selected_expiration=selected_expiration,
        available_expirations=bounded_expirations,
        assumptions=AnalysisAssumptions(
            risk_free_rate=risk_free_rate,
            risk_free_rate_source=(
                "CONFIGURED_DEFAULT" if request.risk_free_rate is None else "REQUEST_OVERRIDE"
            ),
            dividend_yield=dividend_yield,
            dividend_yield_source=(
                "CONFIGURED_DEFAULT" if request.dividend_yield is None else "REQUEST_OVERRIDE"
            ),
            valuation_time=valuation_time,
            implied_volatility_method=(
                "EOD_BSM_EQUIVALENT_IV"
                if snapshot.market_data_mode is MarketDataMode.END_OF_DAY
                else "BSM_EQUIVALENT"
            ),
            tree_steps=tree_steps,
            target_expiration_days=target_days,
            selected_expiration_days=max((selected_expiration - valuation_time.date()).days, 0),
            strike_window_pct=strike_window,
            max_contracts=request.max_contracts,
            max_quote_age_seconds=max_quote_age,
            allow_last_price_fallback=request.allow_last_price_fallback,
            maximum_relative_spread=maximum_spread,
        ),
        summary=OptionsAnalysisSummary(
            contracts_analyzed=len(analyzed),
            calls=sum(contract.option_type is OptionType.CALL for contract in analyzed),
            puts=sum(contract.option_type is OptionType.PUT for contract in analyzed),
            atm_strike=atm_strike,
            source_contracts_considered=len(source_quotes),
            contracts_with_market_iv=sum(
                contract.implied_volatility is not None for contract in analyzed
            ),
            contracts_with_independent_fair_vol=sum(
                contract.fair_value_volatility_independent for contract in analyzed
            ),
        ),
        contracts=analyzed,
        smile=[
            AnalyzerSmilePoint(
                provider_symbol=contract.provider_symbol,
                option_type=contract.option_type,
                strike=contract.strike,
                log_moneyness=contract.log_moneyness,
                market_implied_volatility=contract.implied_volatility,
                fair_value_volatility=contract.fair_value_volatility,
                fair_value_volatility_source=contract.fair_value_volatility_source,
            )
            for contract in analyzed
        ],
        warnings=warnings,
        timings=AnalyzerTiming(
            market_data_ms=market_data_ms,
            implied_volatility_ms=implied_volatility_ms,
            smile_construction_ms=smile_construction_ms,
            crr_pricing_ms=sum(contract.crr_pricing_runtime_ms or 0.0 for contract in analyzed),
            greeks_ms=sum(contract.greeks_runtime_ms or 0.0 for contract in analyzed),
            pricing_and_greeks_ms=pricing_and_greeks_ms,
            total_ms=total_ms,
        ),
    )
