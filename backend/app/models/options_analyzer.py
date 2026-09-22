"""Typed request and response models for the American-options analyzer."""

from datetime import date, datetime
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.app.models.market_data import (
    MarketDataSource,
    MarketExerciseStyle,
    MarketPriceSource,
)
from backend.app.models.option import ExerciseStyle, OptionType
from backend.app.models.volatility import ImpliedVolatilityStatus

PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]


class AnalyzerModel(BaseModel):
    """Immutable finite-data policy shared by analyzer contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class AnalysisOptionType(StrEnum):
    """Option side requested by the simplified chain workflow."""

    CALL = "CALL"
    PUT = "PUT"
    BOTH = "BOTH"


class FairValueVolatilitySource(StrEnum):
    """Provenance of the volatility used for model fair value."""

    SMILE_INTERPOLATED = "SMILE_INTERPOLATED"
    ATM_REFERENCE = "ATM_REFERENCE"
    CONTRACT_IV_FALLBACK = "CONTRACT_IV_FALLBACK"
    UNAVAILABLE = "UNAVAILABLE"


class AnalysisDataQuality(StrEnum):
    """Descriptive input-data quality, never an investment-confidence label."""

    HIGH_DATA_QUALITY = "HIGH_DATA_QUALITY"
    MODERATE_DATA_QUALITY = "MODERATE_DATA_QUALITY"
    LOW_DATA_QUALITY = "LOW_DATA_QUALITY"


class OptionsAnalysisRequest(AnalyzerModel):
    """One bounded ticker analysis request with visible automatic assumptions."""

    symbol: str
    expiration: date | None = None
    option_type: AnalysisOptionType = AnalysisOptionType.BOTH
    max_contracts: PositiveInt = 20
    strike_window_pct: float | None = Field(default=None, gt=0.0, le=2.0)
    target_expiration_days: PositiveInt | None = None
    risk_free_rate: float | None = None
    dividend_yield: float | None = None
    tree_steps: PositiveInt | None = None
    max_quote_age_seconds: PositiveFloat | None = None
    allow_last_price_fallback: bool = False

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("ticker symbol must not be empty")
        if len(normalized) > 15 or not normalized[0].isalpha():
            raise ValueError(
                "ticker symbol must begin with a letter and contain at most 15 characters"
            )
        if any(not (character.isalnum() or character in {".", "-"}) for character in normalized):
            raise ValueError("ticker symbol may contain only letters, numbers, dot, and hyphen")
        return normalized


class AnalyzerUnderlyingQuote(AnalyzerModel):
    """Underlying market observation used by the analysis."""

    price: PositiveFloat
    bid: NonNegativeFloat | None = None
    ask: NonNegativeFloat | None = None
    last: NonNegativeFloat | None = None
    timestamp: datetime
    data_age_seconds: float
    source: MarketDataSource


class AnalysisAssumptions(AnalyzerModel):
    """Every automatic or overridden assumption used by the workflow."""

    risk_free_rate: float
    risk_free_rate_source: Literal["CONFIGURED_DEFAULT", "REQUEST_OVERRIDE"]
    dividend_yield: float
    dividend_yield_source: Literal["CONFIGURED_DEFAULT", "REQUEST_OVERRIDE"]
    valuation_time: datetime
    pricing_model: Literal["CRR_AMERICAN"] = "CRR_AMERICAN"
    implied_volatility_method: Literal["BSM_EQUIVALENT"] = "BSM_EQUIVALENT"
    tree_steps: PositiveInt
    target_expiration_days: PositiveInt
    selected_expiration_days: int = Field(ge=0)
    strike_window_pct: PositiveFloat
    max_contracts: PositiveInt
    max_quote_age_seconds: PositiveFloat
    allow_last_price_fallback: bool
    maximum_relative_spread: PositiveFloat


class OptionsAnalysisSummary(AnalyzerModel):
    """Compact factual counts for the selected bounded chain."""

    contracts_analyzed: int = Field(ge=0)
    calls: int = Field(ge=0)
    puts: int = Field(ge=0)
    atm_strike: PositiveFloat
    source_contracts_considered: int = Field(ge=0)
    contracts_with_market_iv: int = Field(ge=0)
    contracts_with_independent_fair_vol: int = Field(ge=0)


class AnalyzerSmilePoint(AnalyzerModel):
    """Market and fair-value volatility at one displayed contract."""

    provider_symbol: str
    option_type: OptionType
    strike: PositiveFloat
    log_moneyness: float
    market_implied_volatility: NonNegativeFloat | None = None
    fair_value_volatility: NonNegativeFloat | None = None
    fair_value_volatility_source: FairValueVolatilitySource


class AnalyzerTiming(AnalyzerModel):
    """Separated local timing for provider, IV, smile, and tree work."""

    market_data_ms: NonNegativeFloat
    implied_volatility_ms: NonNegativeFloat
    smile_construction_ms: NonNegativeFloat
    crr_pricing_ms: NonNegativeFloat
    greeks_ms: NonNegativeFloat
    pricing_and_greeks_ms: NonNegativeFloat
    total_ms: NonNegativeFloat


class AnalyzedOptionContract(AnalyzerModel):
    """One market contract with American value, Greeks, liquidity, and diagnostics."""

    provider_symbol: str
    option_type: OptionType
    strike: PositiveFloat
    expiration: date
    exercise_style: Literal[ExerciseStyle.AMERICAN] = ExerciseStyle.AMERICAN
    provider_exercise_style: MarketExerciseStyle
    bid: float | None = None
    ask: float | None = None
    market_midpoint: NonNegativeFloat | None = None
    market_price_source: MarketPriceSource | None = None
    last: float | None = None
    bid_size: int | None = Field(default=None, ge=0)
    ask_size: int | None = Field(default=None, ge=0)
    volume: NonNegativeFloat | None = None
    open_interest: int | None = Field(default=None, ge=0)
    quote_timestamp: datetime
    quote_age_seconds: float
    days_to_expiration: int = Field(ge=0)
    time_to_expiration_years: NonNegativeFloat
    implied_volatility: NonNegativeFloat | None = None
    implied_volatility_status: ImpliedVolatilityStatus | None = None
    implied_volatility_method: Literal["BSM_EQUIVALENT"] = "BSM_EQUIVALENT"
    provider_implied_volatility: NonNegativeFloat | None = None
    fair_value_volatility: NonNegativeFloat | None = None
    fair_value_volatility_source: FairValueVolatilitySource
    fair_value_volatility_independent: bool
    fair_value: NonNegativeFloat | None = None
    fair_value_gap: float | None = None
    fair_value_gap_pct: float | None = None
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    theta_daily: float | None = None
    vega: float | None = None
    vega_per_vol_point: float | None = None
    rho: float | None = None
    intrinsic_value: NonNegativeFloat
    time_value: NonNegativeFloat | None = None
    moneyness: PositiveFloat
    log_moneyness: float
    spread: NonNegativeFloat | None = None
    spread_pct: NonNegativeFloat | None = None
    european_reference_value: NonNegativeFloat | None = None
    early_exercise_premium: NonNegativeFloat | None = None
    tree_steps: PositiveInt
    crr_pricing_runtime_ms: NonNegativeFloat | None = None
    greeks_runtime_ms: NonNegativeFloat | None = None
    model_runtime_ms: NonNegativeFloat | None = None
    analysis_quality: AnalysisDataQuality
    quality_flags: list[str]
    warning_flags: list[str]


class OptionsAnalysisResult(AnalyzerModel):
    """Complete bounded result for the ticker-only American-options workflow."""

    symbol: str
    data_source: MarketDataSource
    underlying: AnalyzerUnderlyingQuote
    selected_expiration: date
    available_expirations: list[date]
    assumptions: AnalysisAssumptions
    summary: OptionsAnalysisSummary
    contracts: list[AnalyzedOptionContract]
    smile: list[AnalyzerSmilePoint]
    warnings: list[str]
    timings: AnalyzerTiming

    @model_validator(mode="after")
    def validate_summary_counts(self) -> Self:
        if self.summary.contracts_analyzed != len(self.contracts):
            raise ValueError("contracts_analyzed must match serialized contracts")
        return self
