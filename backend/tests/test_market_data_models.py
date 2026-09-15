from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from backend.app.models.market_data import (
    MarketDataSource,
    MarketExerciseStyle,
    OptionChainSnapshot,
    OptionInstrument,
    OptionQuote,
    UnderlyingQuote,
)
from backend.app.models.option import OptionType


def instrument() -> OptionInstrument:
    return OptionInstrument(
        underlying_symbol=" xyz ",
        option_type=OptionType.CALL,
        strike=100.0,
        expiration=datetime(2027, 1, 15, 21, tzinfo=UTC),
        provider_symbol="O:XYZ270115C00100000",
        exercise_style=MarketExerciseStyle.AMERICAN,
        multiplier=100,
    )


def test_underlying_quote_normalizes_symbol_timezone_and_midpoint() -> None:
    quote = UnderlyingQuote(
        symbol=" xyz ",
        bid=99.0,
        ask=101.0,
        last=100.0,
        timestamp=datetime(2026, 1, 1, 12, tzinfo=UTC) + timedelta(hours=1),
        source=MarketDataSource.FAKE,
    )
    assert quote.symbol == "XYZ"
    assert quote.midpoint == 100.0
    assert quote.timestamp.tzinfo is UTC


def test_underlying_crossed_market_has_no_midpoint() -> None:
    quote = UnderlyingQuote(
        symbol="XYZ",
        bid=101.0,
        ask=100.0,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        source=MarketDataSource.FAKE,
    )
    assert quote.midpoint is None


def test_option_quote_midpoint_policy_preserves_bad_prices() -> None:
    valid = OptionQuote(
        instrument=instrument(),
        bid=1.0,
        ask=1.2,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        source=MarketDataSource.FAKE,
    )
    missing = valid.model_copy(update={"bid": None, "midpoint": None})
    crossed = OptionQuote(
        instrument=instrument(),
        bid=2.0,
        ask=1.0,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        source=MarketDataSource.FAKE,
    )
    negative = OptionQuote(
        instrument=instrument(),
        bid=-1.0,
        ask=1.0,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        source=MarketDataSource.FAKE,
    )
    assert valid.midpoint == pytest.approx(1.1)
    assert missing.midpoint is None
    assert crossed.midpoint is None
    assert negative.midpoint is None
    assert negative.bid == -1.0


@pytest.mark.parametrize(
    "model_values",
    [
        {
            "underlying_symbol": "XYZ",
            "option_type": "CALL",
            "strike": 0.0,
            "expiration": datetime(2027, 1, 1, tzinfo=UTC),
            "provider_symbol": "bad",
        },
        {
            "underlying_symbol": "XYZ",
            "option_type": "CALL",
            "strike": 100.0,
            "expiration": datetime(2027, 1, 1),
            "provider_symbol": "bad",
        },
    ],
)
def test_instrument_rejects_bad_strike_and_naive_expiration(
    model_values: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        OptionInstrument.model_validate(model_values)


def test_quotes_and_snapshots_reject_naive_timestamps() -> None:
    with pytest.raises(ValidationError):
        OptionQuote(
            instrument=instrument(),
            bid=1.0,
            ask=1.1,
            timestamp=datetime(2026, 1, 1),
            source=MarketDataSource.FAKE,
        )
    underlying = UnderlyingQuote(
        symbol="XYZ",
        last=100.0,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        source=MarketDataSource.FAKE,
    )
    with pytest.raises(ValidationError):
        OptionChainSnapshot(
            underlying_symbol="XYZ",
            underlying_quote=underlying,
            retrieved_at=datetime(2026, 1, 1),
            source=MarketDataSource.FAKE,
            quotes=[],
        )


def test_underlying_rejects_negative_prices() -> None:
    with pytest.raises(ValidationError):
        UnderlyingQuote(
            symbol="XYZ",
            bid=-1.0,
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            source=MarketDataSource.FAKE,
        )
