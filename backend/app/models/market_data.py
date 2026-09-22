"""Provider-independent normalized option-market data models."""

from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.app.models.option import OptionType


class MarketDataSource(StrEnum):
    """Normalized provider identity without leaking provider response fields."""

    MASSIVE = "MASSIVE"
    FAKE = "FAKE"


class MarketDataMode(StrEnum):
    """Capability mode used to obtain one internally consistent dataset."""

    SNAPSHOT = "SNAPSHOT"
    END_OF_DAY = "END_OF_DAY"


class UnderlyingPriceSource(StrEnum):
    """Source of the underlying reference price used by analytics."""

    SNAPSHOT_QUOTE = "SNAPSHOT_QUOTE"
    END_OF_DAY_CLOSE = "END_OF_DAY_CLOSE"


class ComparisonPriceSource(StrEnum):
    """Source of the option observation compared with model fair value."""

    BID_ASK_MIDPOINT = "BID_ASK_MIDPOINT"
    LAST_TRADE = "LAST_TRADE"
    END_OF_DAY_CLOSE = "END_OF_DAY_CLOSE"


class MarketExerciseStyle(StrEnum):
    """Exercise style reported for a real-market instrument."""

    AMERICAN = "AMERICAN"
    EUROPEAN = "EUROPEAN"
    BERMUDAN = "BERMUDAN"
    UNKNOWN = "UNKNOWN"


class QuoteIssue(StrEnum):
    """Structured data-quality and analytical issues attached to a quote."""

    MISSING_BID = "MISSING_BID"
    MISSING_ASK = "MISSING_ASK"
    CROSSED_MARKET = "CROSSED_MARKET"
    NEGATIVE_PRICE = "NEGATIVE_PRICE"
    STALE = "STALE"
    FUTURE_TIMESTAMP = "FUTURE_TIMESTAMP"
    EXPIRED = "EXPIRED"
    PRICE_BELOW_LOWER_BOUND = "PRICE_BELOW_LOWER_BOUND"
    PRICE_ABOVE_UPPER_BOUND = "PRICE_ABOVE_UPPER_BOUND"
    IV_UNAVAILABLE = "IV_UNAVAILABLE"
    NO_MARKET_PRICE = "NO_MARKET_PRICE"


class MarketPriceSource(StrEnum):
    """Source chosen for an analytical market price."""

    MIDPOINT = "MIDPOINT"
    LAST = "LAST"
    END_OF_DAY_CLOSE = "END_OF_DAY_CLOSE"


class QuoteTimestampSource(StrEnum):
    """Meaning of the normalized quote timestamp."""

    QUOTE = "QUOTE"
    TRADE = "TRADE"
    SNAPSHOT = "SNAPSHOT"
    END_OF_DAY = "END_OF_DAY"


class ParityStatus(StrEnum):
    """Put-call parity diagnostic outcome."""

    WITHIN_TOLERANCE = "WITHIN_TOLERANCE"
    VIOLATION = "VIOLATION"


NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]


def _normalize_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    if not normalized:
        raise ValueError("symbol must not be empty")
    return normalized


def _normalize_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


class MarketDataModel(BaseModel):
    """Immutable finite-data policy for normalized market-data objects."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ProviderCapabilities(MarketDataModel):
    """Provider features observed for the current request/cached capability window."""

    snapshot_available: bool
    end_of_day_available: bool
    basic_contract_limit: PositiveInt | None = None


class UnderlyingQuote(MarketDataModel):
    """Normalized underlying NBBO/trade snapshot."""

    symbol: str
    bid: NonNegativeFloat | None = None
    ask: NonNegativeFloat | None = None
    last: NonNegativeFloat | None = None
    midpoint: NonNegativeFloat | None = None
    timestamp: datetime
    timestamp_source: QuoteTimestampSource = QuoteTimestampSource.QUOTE
    source: MarketDataSource
    previous_close: NonNegativeFloat | None = None
    volume: NonNegativeFloat | None = None
    market_data_mode: MarketDataMode = MarketDataMode.SNAPSHOT
    price_source: UnderlyingPriceSource = UnderlyingPriceSource.SNAPSHOT_QUOTE
    reference_price: PositiveFloat | None = None
    open: NonNegativeFloat | None = None
    high: NonNegativeFloat | None = None
    low: NonNegativeFloat | None = None
    close: NonNegativeFloat | None = None
    vwap: NonNegativeFloat | None = None
    data_date: date | None = None

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return _normalize_symbol(value)

    @field_validator("timestamp")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        return _normalize_utc(value)

    @model_validator(mode="after")
    def derive_midpoint(self) -> Self:
        midpoint = None
        if self.bid is not None and self.ask is not None and self.ask >= self.bid:
            midpoint = 0.5 * (self.bid + self.ask)
        object.__setattr__(self, "midpoint", midpoint)
        if self.reference_price is None:
            object.__setattr__(self, "reference_price", midpoint or self.last)
        return self


class OptionInstrument(MarketDataModel):
    """Provider-independent identity for one listed option contract."""

    underlying_symbol: str
    option_type: OptionType
    strike: PositiveFloat
    expiration: datetime
    provider_symbol: str
    exercise_style: MarketExerciseStyle = MarketExerciseStyle.UNKNOWN
    multiplier: PositiveInt | None = None

    @field_validator("underlying_symbol")
    @classmethod
    def normalize_underlying_symbol(cls, value: str) -> str:
        return _normalize_symbol(value)

    @field_validator("provider_symbol")
    @classmethod
    def validate_provider_symbol(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("provider_symbol must not be empty")
        return normalized

    @field_validator("expiration")
    @classmethod
    def normalize_expiration(cls, value: datetime) -> datetime:
        return _normalize_utc(value)


class OptionQuote(MarketDataModel):
    """Normalized option quote; invalid price relationships remain inspectable."""

    instrument: OptionInstrument
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    midpoint: NonNegativeFloat | None = None
    bid_size: NonNegativeInt | None = None
    ask_size: NonNegativeInt | None = None
    volume: NonNegativeFloat | None = None
    open_interest: NonNegativeInt | None = None
    timestamp: datetime
    timestamp_source: QuoteTimestampSource = QuoteTimestampSource.QUOTE
    last_trade_timestamp: datetime | None = None
    source: MarketDataSource
    provider_implied_volatility: NonNegativeFloat | None = None
    provider_greeks: dict[str, float] | None = None
    market_data_mode: MarketDataMode = MarketDataMode.SNAPSHOT
    market_reference_price: NonNegativeFloat | None = None
    market_price_source: MarketPriceSource | None = None
    eod_open: NonNegativeFloat | None = None
    eod_high: NonNegativeFloat | None = None
    eod_low: NonNegativeFloat | None = None
    eod_close: NonNegativeFloat | None = None
    eod_vwap: NonNegativeFloat | None = None
    data_date: date | None = None

    @field_validator("timestamp", "last_trade_timestamp")
    @classmethod
    def normalize_timestamps(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _normalize_utc(value)

    @model_validator(mode="after")
    def derive_midpoint(self) -> Self:
        midpoint = None
        if (
            self.bid is not None
            and self.ask is not None
            and self.bid >= 0.0
            and self.ask >= self.bid
        ):
            midpoint = 0.5 * (self.bid + self.ask)
        object.__setattr__(self, "midpoint", midpoint)
        return self


class OptionChainSnapshot(MarketDataModel):
    """One normalized provider snapshot without raw response payloads."""

    underlying_symbol: str
    underlying_quote: UnderlyingQuote
    retrieved_at: datetime
    source: MarketDataSource
    quotes: list[OptionQuote]
    available_expirations: list[date] = Field(default_factory=list)
    market_data_mode: MarketDataMode = MarketDataMode.SNAPSHOT
    capabilities: ProviderCapabilities = Field(
        default_factory=lambda: ProviderCapabilities(
            snapshot_available=True,
            end_of_day_available=False,
        )
    )

    @field_validator("underlying_symbol")
    @classmethod
    def normalize_underlying_symbol(cls, value: str) -> str:
        return _normalize_symbol(value)

    @field_validator("retrieved_at")
    @classmethod
    def normalize_retrieval_timestamp(cls, value: datetime) -> datetime:
        return _normalize_utc(value)


class QuoteQualityConfig(MarketDataModel):
    """Explicit quote eligibility, age, arbitrage, and parity tolerances."""

    max_quote_age_seconds: PositiveFloat = 900.0
    allow_last_price: bool = False
    arbitrage_tolerance: NonNegativeFloat = 1e-8
    parity_absolute_tolerance: NonNegativeFloat = 0.05
    parity_relative_tolerance: NonNegativeFloat = 0.01
