from datetime import UTC, datetime, timedelta, timezone

import pytest

from backend.app.analytics.market_data import (
    act_365_maturity_years,
    bid_ask_spread,
    quote_quality_issues,
    select_market_price,
)
from backend.app.core.exceptions import MarketDataValidationError
from backend.app.models.market_data import (
    MarketDataSource,
    MarketPriceSource,
    OptionInstrument,
    OptionQuote,
    QuoteIssue,
    QuoteQualityConfig,
)
from backend.app.models.option import OptionType

VALUATION = datetime(2026, 1, 1, 12, tzinfo=UTC)
EXPIRATION = datetime(2027, 1, 1, 12, tzinfo=UTC)


def quote(
    *,
    bid: float | None = 1.0,
    ask: float | None = 1.2,
    last: float | None = 1.1,
    timestamp: datetime = VALUATION,
    expiration: datetime = EXPIRATION,
) -> OptionQuote:
    return OptionQuote(
        instrument=OptionInstrument(
            underlying_symbol="XYZ",
            option_type=OptionType.CALL,
            strike=100.0,
            expiration=expiration,
            provider_symbol="O:XYZ",
        ),
        bid=bid,
        ask=ask,
        last=last,
        timestamp=timestamp,
        source=MarketDataSource.FAKE,
    )


def test_act_365_maturity_positive_exact_expiry_expired_and_timezone() -> None:
    assert act_365_maturity_years(EXPIRATION, VALUATION) == 1.0
    assert act_365_maturity_years(VALUATION, VALUATION) == 0.0
    assert act_365_maturity_years(VALUATION - timedelta(seconds=1), VALUATION) == 0.0
    eastern = timezone(timedelta(hours=-5))
    assert (
        act_365_maturity_years(
            EXPIRATION.astimezone(eastern),
            VALUATION.astimezone(eastern),
        )
        == 1.0
    )
    with pytest.raises(MarketDataValidationError, match="timezone-aware"):
        act_365_maturity_years(datetime(2027, 1, 1), VALUATION)


def test_quality_detects_fresh_stale_future_and_expired_quotes() -> None:
    config = QuoteQualityConfig(max_quote_age_seconds=60)
    assert quote_quality_issues(quote(), VALUATION, config) == []
    assert QuoteIssue.STALE in quote_quality_issues(
        quote(timestamp=VALUATION - timedelta(seconds=61)), VALUATION, config
    )
    assert QuoteIssue.FUTURE_TIMESTAMP in quote_quality_issues(
        quote(timestamp=VALUATION + timedelta(seconds=1)), VALUATION, config
    )
    assert QuoteIssue.EXPIRED in quote_quality_issues(
        quote(expiration=VALUATION - timedelta(seconds=1)), VALUATION, config
    )


def test_quality_reports_missing_crossed_negative_and_multiple_issues() -> None:
    config = QuoteQualityConfig(max_quote_age_seconds=60)
    missing = quote_quality_issues(quote(bid=None), VALUATION, config)
    assert missing == [QuoteIssue.MISSING_BID]

    issues = quote_quality_issues(
        quote(
            bid=2.0,
            ask=-1.0,
            timestamp=VALUATION - timedelta(seconds=120),
            expiration=VALUATION - timedelta(seconds=1),
        ),
        VALUATION,
        config,
    )
    assert issues == [
        QuoteIssue.NEGATIVE_PRICE,
        QuoteIssue.CROSSED_MARKET,
        QuoteIssue.STALE,
        QuoteIssue.EXPIRED,
    ]


def test_midpoint_and_explicit_last_price_selection() -> None:
    config = QuoteQualityConfig()
    market_price, source = select_market_price(quote(), [], config)
    assert market_price == pytest.approx(1.1)
    assert source is MarketPriceSource.MIDPOINT

    one_sided = quote(bid=None, ask=1.2, last=1.15)
    issues = quote_quality_issues(one_sided, VALUATION, config)
    assert select_market_price(one_sided, issues, config) == (None, None)
    assert select_market_price(
        one_sided,
        issues,
        QuoteQualityConfig(allow_last_price=True),
    ) == (1.15, MarketPriceSource.LAST)


def test_last_price_is_not_used_for_stale_or_crossed_quote() -> None:
    config = QuoteQualityConfig(allow_last_price=True, max_quote_age_seconds=60)
    stale = quote(bid=None, timestamp=VALUATION - timedelta(seconds=61))
    stale_issues = quote_quality_issues(stale, VALUATION, config)
    assert select_market_price(stale, stale_issues, config) == (None, None)
    crossed = quote(bid=2.0, ask=1.0)
    crossed_issues = quote_quality_issues(crossed, VALUATION, config)
    assert select_market_price(crossed, crossed_issues, config) == (None, None)


def test_spread_absolute_relative_zero_and_invalid_cases() -> None:
    absolute, relative = bid_ask_spread(quote(bid=1.0, ask=1.2))
    assert absolute == pytest.approx(0.2)
    assert relative == pytest.approx(0.2 / 1.1)
    assert bid_ask_spread(quote(bid=0.0, ask=0.0)) == (0.0, None)
    assert bid_ask_spread(quote(bid=None)) == (None, None)
    assert bid_ask_spread(quote(bid=2.0, ask=1.0)) == (None, None)
