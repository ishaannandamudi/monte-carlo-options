from datetime import UTC, datetime, timedelta

import pytest

from backend.app.analytics.market_data import (
    analyze_put_call_parity,
    enrich_option_chain,
    extract_market_smile,
    filter_enriched_quotes,
    is_arbitrage_valid,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.market_data import (
    MarketDataSource,
    MarketExerciseStyle,
    OptionChainSnapshot,
    OptionInstrument,
    OptionQuote,
    ParityStatus,
    QuoteIssue,
    QuoteQualityConfig,
    UnderlyingQuote,
)
from backend.app.models.option import OptionContract, OptionType
from backend.app.pricing.black_scholes import black_scholes_price

VALUATION = datetime(2026, 1, 1, 21, tzinfo=UTC)
EXPIRATION = VALUATION + timedelta(days=365)


def option_quote(
    option_type: OptionType,
    strike: float,
    midpoint: float,
    *,
    expiration: datetime = EXPIRATION,
    timestamp: datetime = VALUATION,
    provider_symbol_suffix: str = "",
) -> OptionQuote:
    return OptionQuote(
        instrument=OptionInstrument(
            underlying_symbol="XYZ",
            option_type=option_type,
            strike=strike,
            expiration=expiration,
            provider_symbol=f"O:XYZ-{expiration.date()}-{option_type.value}-{strike}{provider_symbol_suffix}",
            exercise_style=MarketExerciseStyle.AMERICAN,
            multiplier=100,
        ),
        bid=midpoint - 0.01,
        ask=midpoint + 0.01,
        last=midpoint,
        bid_size=10,
        ask_size=12,
        volume=100,
        open_interest=500,
        timestamp=timestamp,
        source=MarketDataSource.FAKE,
    )


def theoretical_quote(
    option_type: OptionType,
    strike: float,
    volatility: float,
    *,
    rate: float = 0.03,
    dividend: float = 0.01,
    expiration: datetime = EXPIRATION,
) -> OptionQuote:
    maturity = (expiration - VALUATION).total_seconds() / (365 * 24 * 3600)
    contract = OptionContract(
        option_type=option_type,
        spot=100.0,
        strike=strike,
        maturity_years=maturity,
    )
    price = black_scholes_price(
        contract,
        MarketEnvironment(
            risk_free_rate=rate,
            volatility=volatility,
            dividend_yield=dividend,
        ),
    )
    return option_quote(option_type, strike, price, expiration=expiration)


def snapshot(quotes: list[OptionQuote]) -> OptionChainSnapshot:
    return OptionChainSnapshot(
        underlying_symbol="XYZ",
        underlying_quote=UnderlyingQuote(
            symbol="XYZ",
            bid=99.9,
            ask=100.1,
            last=100.0,
            timestamp=VALUATION,
            source=MarketDataSource.FAKE,
        ),
        retrieved_at=VALUATION,
        source=MarketDataSource.FAKE,
        quotes=quotes,
        available_expirations=sorted({quote.instrument.expiration.date() for quote in quotes}),
    )


def test_chain_enrichment_recovers_iv_greeks_spreads_and_sorts() -> None:
    quotes = [
        theoretical_quote(OptionType.PUT, 110.0, 0.24),
        theoretical_quote(OptionType.CALL, 100.0, 0.20),
        theoretical_quote(OptionType.PUT, 100.0, 0.20),
        theoretical_quote(OptionType.CALL, 90.0, 0.23),
    ]
    result = enrich_option_chain(
        snapshot(list(reversed(quotes))),
        valuation_time=VALUATION,
        risk_free_rate=0.03,
        dividend_yield=0.01,
    )
    identities = [
        (item.quote.instrument.option_type, item.quote.instrument.strike) for item in result.quotes
    ]
    assert identities == [
        (OptionType.CALL, 90.0),
        (OptionType.CALL, 100.0),
        (OptionType.PUT, 100.0),
        (OptionType.PUT, 110.0),
    ]
    expected_vols = [0.23, 0.20, 0.20, 0.24]
    for item, expected in zip(result.quotes, expected_vols, strict=True):
        assert item.implied_volatility == pytest.approx(expected, abs=1e-8)
        assert item.analytical_greeks is not None
        assert item.analytical_greeks.vega > 0.0
        assert item.maturity_years == pytest.approx(1.0)
        assert item.absolute_spread == pytest.approx(0.02)
        assert item.relative_spread is not None
        assert item.european_bsm_approximation is True
        assert item.quality_issues == []
    assert result.summary.total_contracts == 4
    assert result.summary.valid_quotes == 4
    assert result.summary.quotes_with_recovered_iv == 4
    assert result.summary.calls_count == 2
    assert result.summary.puts_count == 2
    assert result.summary.expiration_count == 1


def test_bad_quotes_are_preserved_and_do_not_fail_chain() -> None:
    valid = theoretical_quote(OptionType.CALL, 100.0, 0.2)
    crossed = OptionQuote(
        instrument=valid.instrument.model_copy(
            update={"strike": 105.0, "provider_symbol": "cross"}
        ),
        bid=3.0,
        ask=2.0,
        last=2.5,
        timestamp=VALUATION,
        source=MarketDataSource.FAKE,
    )
    impossible = option_quote(OptionType.CALL, 110.0, 120.0)
    stale = option_quote(
        OptionType.PUT,
        90.0,
        1.0,
        timestamp=VALUATION - timedelta(hours=1),
    )
    result = enrich_option_chain(
        snapshot([valid, crossed, impossible, stale]),
        valuation_time=VALUATION,
        risk_free_rate=0.03,
        dividend_yield=0.01,
        quality_config=QuoteQualityConfig(max_quote_age_seconds=60),
    )
    by_strike = {item.quote.instrument.strike: item for item in result.quotes}
    assert QuoteIssue.CROSSED_MARKET in by_strike[105.0].quality_issues
    assert by_strike[105.0].market_price is None
    assert QuoteIssue.PRICE_ABOVE_UPPER_BOUND in by_strike[110.0].quality_issues
    assert QuoteIssue.IV_UNAVAILABLE in by_strike[110.0].quality_issues
    assert is_arbitrage_valid(by_strike[110.0]) is False
    assert QuoteIssue.STALE in by_strike[90.0].quality_issues
    assert result.summary.total_contracts == 4
    assert result.summary.invalid_quotes == 3
    assert result.summary.arbitrage_violations == 1
    assert result.summary.stale_quotes == 1
    assert result.summary.quotes_with_recovered_iv == 1


def test_explicit_last_price_can_enrich_one_sided_quote() -> None:
    theoretical = theoretical_quote(OptionType.CALL, 100.0, 0.2)
    one_sided = OptionQuote(
        instrument=theoretical.instrument,
        ask=theoretical.ask,
        last=theoretical.last,
        timestamp=VALUATION,
        source=MarketDataSource.FAKE,
    )
    result = enrich_option_chain(
        snapshot([one_sided]),
        valuation_time=VALUATION,
        risk_free_rate=0.03,
        dividend_yield=0.01,
        quality_config=QuoteQualityConfig(allow_last_price=True),
    )
    enriched = result.quotes[0]
    assert enriched.market_price == theoretical.last
    assert enriched.market_price_source is not None
    assert enriched.implied_volatility == pytest.approx(0.2, abs=1e-8)
    assert QuoteIssue.MISSING_BID in enriched.quality_issues


def test_smile_extraction_keeps_calls_and_puts_distinct() -> None:
    chain = enrich_option_chain(
        snapshot(
            [
                theoretical_quote(OptionType.CALL, 90.0, 0.23),
                theoretical_quote(OptionType.PUT, 90.0, 0.23),
                theoretical_quote(OptionType.CALL, 100.0, 0.20),
            ]
        ),
        valuation_time=VALUATION,
        risk_free_rate=0.03,
        dividend_yield=0.01,
    )
    smile = extract_market_smile(
        chain,
        expiration=EXPIRATION,
        option_type=OptionType.CALL,
    )
    assert [point.strike for point in smile.points] == [90.0, 100.0]
    assert all(point.option_type is OptionType.CALL for point in smile.points)


@pytest.mark.parametrize(("rate", "dividend"), [(0.03, 0.01), (-0.02, 0.03)])
def test_put_call_parity_clean_distorted_and_unmatched(
    rate: float,
    dividend: float,
) -> None:
    call_90 = theoretical_quote(OptionType.CALL, 90.0, 0.25, rate=rate, dividend=dividend)
    put_90 = theoretical_quote(OptionType.PUT, 90.0, 0.25, rate=rate, dividend=dividend)
    call_100 = theoretical_quote(OptionType.CALL, 100.0, 0.25, rate=rate, dividend=dividend)
    put_100_base = theoretical_quote(OptionType.PUT, 100.0, 0.25, rate=rate, dividend=dividend)
    put_100 = option_quote(OptionType.PUT, 100.0, put_100_base.midpoint + 1.0)
    unmatched_call = theoretical_quote(OptionType.CALL, 110.0, 0.25, rate=rate, dividend=dividend)
    result = analyze_put_call_parity(
        snapshot([unmatched_call, put_100, call_90, call_100, put_90]),
        valuation_time=VALUATION,
        risk_free_rate=rate,
        dividend_yield=dividend,
        quality_config=QuoteQualityConfig(
            parity_absolute_tolerance=1e-8,
            parity_relative_tolerance=1e-8,
        ),
    )
    assert result.matched_pairs == 2
    assert result.unmatched_contracts == 1
    assert result.points[0].strike == 90.0
    assert result.points[0].residual == pytest.approx(0.0, abs=1e-12)
    assert result.points[0].status is ParityStatus.WITHIN_TOLERANCE
    assert result.points[1].residual == pytest.approx(-1.0, abs=1e-12)
    assert result.points[1].status is ParityStatus.VIOLATION
    assert result.violations == 1


def test_filters_are_deterministic_across_all_supported_axes() -> None:
    second_expiration = EXPIRATION + timedelta(days=30)
    stale = theoretical_quote(OptionType.CALL, 120.0, 0.25, expiration=second_expiration)
    stale = stale.model_copy(update={"timestamp": VALUATION - timedelta(hours=1)})
    chain = enrich_option_chain(
        snapshot(
            [
                stale,
                theoretical_quote(OptionType.PUT, 100.0, 0.2),
                theoretical_quote(OptionType.CALL, 90.0, 0.23),
                theoretical_quote(OptionType.CALL, 100.0, 0.2),
            ]
        ),
        valuation_time=VALUATION,
        risk_free_rate=0.03,
        dividend_yield=0.01,
        quality_config=QuoteQualityConfig(max_quote_age_seconds=60),
    )
    selected = filter_enriched_quotes(
        chain.quotes,
        expiration=EXPIRATION,
        option_type=OptionType.CALL,
        min_strike=90.0,
        max_strike=105.0,
        min_moneyness=0.85,
        max_moneyness=1.05,
        valid_midpoint_only=True,
        iv_available_only=True,
        fresh_only=True,
    )
    assert [item.quote.instrument.strike for item in selected] == [90.0, 100.0]
