"""Typed volatility-smile, surface, interpolation, and diagnostic models."""

import math
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.app.models.option import OptionType

PositiveFloat = Annotated[float, Field(gt=0.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
PositiveInt = Annotated[int, Field(gt=0)]


class SurfaceModel(BaseModel):
    """Immutable finite-data policy for volatility research objects."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class VolatilityCoordinate(StrEnum):
    """Supported strike coordinates; log-forward-moneyness is canonical."""

    STRIKE = "STRIKE"
    MONEYNESS = "MONEYNESS"
    FORWARD_MONEYNESS = "FORWARD_MONEYNESS"
    LOG_FORWARD_MONEYNESS = "LOG_FORWARD_MONEYNESS"


class SmileSelectionPolicy(StrEnum):
    """Option-side policy used to select one market observation per strike."""

    OTM_ONLY = "OTM_ONLY"
    CALLS_ONLY = "CALLS_ONLY"
    PUTS_ONLY = "PUTS_ONLY"
    ALL_VALID = "ALL_VALID"


class SmileInterpolationMethod(StrEnum):
    """Shape-safe one-dimensional smile interpolation methods."""

    LINEAR = "LINEAR"
    PCHIP = "PCHIP"


class ExtrapolationPolicy(StrEnum):
    """Explicit wing or maturity behavior outside observed support."""

    NONE = "NONE"
    FLAT_VOL = "FLAT_VOL"


class MaturityInterpolationMethod(StrEnum):
    """Supported surface term interpolation policy."""

    TOTAL_VARIANCE_LINEAR = "TOTAL_VARIANCE_LINEAR"


class ArbitrageIssueType(StrEnum):
    """Structured, diagnostic-only volatility-surface issue categories."""

    STRIKE_MONOTONICITY = "STRIKE_MONOTONICITY"
    BUTTERFLY_CONVEXITY = "BUTTERFLY_CONVEXITY"
    CALENDAR_TOTAL_VARIANCE = "CALENDAR_TOTAL_VARIANCE"
    NEGATIVE_TOTAL_VARIANCE = "NEGATIVE_TOTAL_VARIANCE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    EXTRAPOLATION_REQUIRED = "EXTRAPOLATION_REQUIRED"
    INTERPOLATION_ARTIFACT = "INTERPOLATION_ARTIFACT"


class VolatilityPointQuality(SurfaceModel):
    """Traceable market-quality metadata without provider-specific payloads."""

    quality_issues: list[str] = Field(default_factory=list)
    relative_spread: NonNegativeFloat | None = None
    quote_timestamp: datetime | None = None
    provider_symbol: str | None = None
    is_fresh: bool = True
    arbitrage_valid: bool = True
    bid_implied_volatility: NonNegativeFloat | None = None
    ask_implied_volatility: NonNegativeFloat | None = None

    @field_validator("quote_timestamp")
    @classmethod
    def normalize_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("quote_timestamp must be timezone-aware")
        return value.astimezone(UTC)


class VolatilityPoint(SurfaceModel):
    """One normalized implied-volatility observation in all supported coordinates."""

    strike: PositiveFloat
    maturity_years: PositiveFloat
    implied_volatility: NonNegativeFloat
    option_type: OptionType
    market_price: NonNegativeFloat | None = None
    bid: NonNegativeFloat | None = None
    ask: NonNegativeFloat | None = None
    forward_price: PositiveFloat
    moneyness: PositiveFloat
    forward_moneyness: PositiveFloat
    log_moneyness: float
    source: str | None = None
    expiration: datetime | None = None
    quality: VolatilityPointQuality = Field(default_factory=VolatilityPointQuality)

    @field_validator("expiration")
    @classmethod
    def normalize_expiration(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("expiration must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_forward_coordinates(self) -> Self:
        expected_forward_moneyness = self.strike / self.forward_price
        expected_log_moneyness = math.log(expected_forward_moneyness)
        if not math.isclose(
            self.forward_moneyness,
            expected_forward_moneyness,
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError("forward_moneyness must equal strike / forward_price")
        if not math.isclose(
            self.log_moneyness,
            expected_log_moneyness,
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError("log_moneyness must equal log(strike / forward_price)")
        return self


class SmileConstructionConfig(SurfaceModel):
    """Explicit quote selection and cleaning policy for market smiles."""

    selection_policy: SmileSelectionPolicy = SmileSelectionPolicy.OTM_ONLY
    maximum_relative_spread: NonNegativeFloat | None = 0.50
    require_fresh_quote: bool = True
    require_arbitrage_valid: bool = True
    min_implied_volatility: NonNegativeFloat = 0.0
    max_implied_volatility: PositiveFloat = 5.0
    minimum_points: PositiveInt = 3
    atm_log_moneyness_tolerance: NonNegativeFloat = 1e-8

    @model_validator(mode="after")
    def validate_volatility_range(self) -> Self:
        if self.max_implied_volatility < self.min_implied_volatility:
            raise ValueError("max_implied_volatility must be at least min_implied_volatility")
        return self


class SmileInterpolationConfig(SurfaceModel):
    """Serializable smile and maturity interpolation/extrapolation policy."""

    method: SmileInterpolationMethod = SmileInterpolationMethod.PCHIP
    extrapolation: ExtrapolationPolicy = ExtrapolationPolicy.NONE
    maturity_method: MaturityInterpolationMethod = MaturityInterpolationMethod.TOTAL_VARIANCE_LINEAR
    maturity_extrapolation: ExtrapolationPolicy = ExtrapolationPolicy.NONE


class SmileConstructionMetadata(SurfaceModel):
    """Policies and provenance retained with a constructed smile."""

    selection_policy: SmileSelectionPolicy = SmileSelectionPolicy.ALL_VALID
    minimum_points: PositiveInt = 1
    accepted_points: int = Field(ge=0)
    excluded_points: int = Field(ge=0, default=0)
    atm_policy: str = "TIGHTEST_RELATIVE_SPREAD_THEN_CALL"
    duplicate_policy: str = "PREFERRED_SIDE_SPREAD_FRESHNESS_SYMBOL"
    source: str | None = None


class SmileConstructionDiagnostics(SurfaceModel):
    """Auditable counts from quote filtering and slice construction."""

    total_quotes_considered: int = Field(ge=0)
    quotes_excluded: int = Field(ge=0)
    iv_points_accepted: int = Field(ge=0)
    smiles_created: int = Field(ge=0)
    expirations_rejected: int = Field(ge=0)
    exclusion_reasons: dict[str, int] = Field(default_factory=dict)
    rejected_expirations: list[datetime] = Field(default_factory=list)


class VolatilitySmile(SurfaceModel):
    """One maturity slice sorted by canonical log-forward-moneyness."""

    underlying_symbol: str | None = None
    valuation_timestamp: datetime | None = None
    expiration: datetime | None = None
    maturity_years: PositiveFloat
    spot: PositiveFloat
    forward: PositiveFloat
    risk_free_rate: float
    dividend_yield: float
    points: list[VolatilityPoint]
    coordinate_convention: VolatilityCoordinate = VolatilityCoordinate.LOG_FORWARD_MONEYNESS
    construction_metadata: SmileConstructionMetadata

    @field_validator("valuation_timestamp", "expiration")
    @classmethod
    def normalize_timestamps(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("smile timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def sort_and_validate_points(self) -> Self:
        if not self.points:
            raise ValueError("a volatility smile requires at least one point")
        points = sorted(self.points, key=lambda point: (point.log_moneyness, point.strike))
        for left, right in zip(points, points[1:], strict=False):
            if math.isclose(
                left.log_moneyness,
                right.log_moneyness,
                rel_tol=0.0,
                abs_tol=1e-14,
            ):
                raise ValueError("volatility smile coordinates must be unique")
        for point in points:
            if not math.isclose(point.maturity_years, self.maturity_years, abs_tol=1e-12):
                raise ValueError("all smile points must share the smile maturity")
            if not math.isclose(point.forward_price, self.forward, rel_tol=1e-10):
                raise ValueError("all smile points must share the smile forward")
        object.__setattr__(self, "points", points)
        return self


class SmileQueryResult(SurfaceModel):
    """A traceable exact, interpolated, or explicitly extrapolated smile value."""

    requested_strike: PositiveFloat | None = None
    log_moneyness: float
    implied_volatility: NonNegativeFloat
    total_variance: NonNegativeFloat
    interpolation_method: SmileInterpolationMethod
    exact_observation: bool
    interpolated: bool
    extrapolation_used: bool
    domain_lower: float
    domain_upper: float
    diagnostics: list[str] = Field(default_factory=list)


class VolatilitySurface(SurfaceModel):
    """Serializable multi-expiration surface without live interpolator objects."""

    underlying_symbol: str | None = None
    valuation_timestamp: datetime | None = None
    spot: PositiveFloat
    risk_free_rate: float
    dividend_yield: float
    rate_assumption: str = "FLAT_CONTINUOUSLY_COMPOUNDED"
    dividend_assumption: str = "FLAT_CONTINUOUS_YIELD"
    smiles: list[VolatilitySmile]
    maturities: list[PositiveFloat] = Field(default_factory=list)
    coordinate_convention: VolatilityCoordinate = VolatilityCoordinate.LOG_FORWARD_MONEYNESS
    interpolation_config: SmileInterpolationConfig = Field(default_factory=SmileInterpolationConfig)
    construction_diagnostics: SmileConstructionDiagnostics

    @field_validator("valuation_timestamp")
    @classmethod
    def normalize_valuation_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("valuation_timestamp must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def sort_and_validate_smiles(self) -> Self:
        if not self.smiles:
            raise ValueError("a volatility surface requires at least one smile")
        smiles = sorted(self.smiles, key=lambda smile: smile.maturity_years)
        maturities = [smile.maturity_years for smile in smiles]
        if any(
            math.isclose(a, b, rel_tol=0.0, abs_tol=1e-14)
            for a, b in zip(maturities, maturities[1:], strict=False)
        ):
            raise ValueError("surface smile maturities must be unique")
        for smile in smiles:
            if not math.isclose(smile.spot, self.spot, rel_tol=1e-10):
                raise ValueError("all smiles must share the surface spot")
            if not math.isclose(smile.risk_free_rate, self.risk_free_rate, abs_tol=1e-12):
                raise ValueError("all smiles must share the surface risk-free rate")
            if not math.isclose(smile.dividend_yield, self.dividend_yield, abs_tol=1e-12):
                raise ValueError("all smiles must share the surface dividend yield")
        object.__setattr__(self, "smiles", smiles)
        object.__setattr__(self, "maturities", maturities)
        return self


class SurfaceQueryResult(SurfaceModel):
    """A two-dimensional surface query with both interpolation stages exposed."""

    requested_strike: PositiveFloat | None = None
    requested_maturity: PositiveFloat
    forward: PositiveFloat
    log_moneyness: float
    implied_volatility: NonNegativeFloat
    total_variance: NonNegativeFloat
    lower_maturity: PositiveFloat
    upper_maturity: PositiveFloat
    smile_interpolation_method: SmileInterpolationMethod
    maturity_interpolation_method: MaturityInterpolationMethod
    exact_maturity: bool
    exact_observation: bool
    smile_extrapolation_used: bool
    maturity_extrapolation_used: bool
    diagnostics: list[str] = Field(default_factory=list)


class ArbitrageIssue(SurfaceModel):
    """One potential static-arbitrage or surface-data issue with local context."""

    issue_type: ArbitrageIssueType
    maturity: PositiveFloat | None = None
    strike: PositiveFloat | None = None
    log_moneyness: float | None = None
    observed_values: list[float] = Field(default_factory=list)
    tolerance: NonNegativeFloat
    message: str


class SurfaceQualitySummary(SurfaceModel):
    """Descriptive coverage and issue counts for a constructed surface."""

    smile_count: int = Field(ge=0)
    total_iv_points: int = Field(ge=0)
    minimum_maturity: PositiveFloat | None = None
    maximum_maturity: PositiveFloat | None = None
    minimum_log_moneyness: float | None = None
    maximum_log_moneyness: float | None = None
    average_points_per_smile: NonNegativeFloat
    excluded_point_count: int = Field(ge=0)
    arbitrage_issue_count: int = Field(ge=0)
    calendar_issue_count: int = Field(ge=0)
    butterfly_issue_count: int = Field(ge=0)
    interpolation_coverage: float = Field(ge=0.0, le=1.0)


class SurfaceDiagnosticConfig(SurfaceModel):
    """Finite diagnostic grids and numerical tolerances."""

    log_moneyness_grid: list[float] | None = None
    price_tolerance: NonNegativeFloat = 1e-8
    convexity_tolerance: NonNegativeFloat = 1e-8
    calendar_tolerance: NonNegativeFloat = 1e-10
    default_grid_size: PositiveInt = 21


class SurfaceDiagnosticsResult(SurfaceModel):
    """All diagnostic-only issues and their descriptive summary."""

    issues: list[ArbitrageIssue]
    quality_summary: SurfaceQualitySummary
    strike_checks: int = Field(ge=0)
    calendar_checks: int = Field(ge=0)


class ATMMaturityPoint(SurfaceModel):
    """ATM-forward volatility and total variance for one maturity."""

    maturity_years: PositiveFloat
    implied_volatility: NonNegativeFloat
    total_variance: NonNegativeFloat
    exact_observation: bool
    interpolated: bool


class ATMTermStructureResult(SurfaceModel):
    """Unsmoothed ATM-forward term structure."""

    points: list[ATMMaturityPoint]
    excluded_maturities: list[float] = Field(default_factory=list)


class SmileAnalyticsPoint(SurfaceModel):
    """Coordinate-based ATM, local skew, and curvature descriptors."""

    maturity_years: PositiveFloat
    atm_implied_volatility: NonNegativeFloat
    atm_exact_observation: bool
    skew_dvol_dlog_moneyness: float | None = None
    curvature_d2vol_dlog_moneyness2: float | None = None
    bump_size: PositiveFloat


class SmileComparisonResult(SurfaceModel):
    """Comparable local smile descriptors ordered across maturities."""

    points: list[SmileAnalyticsPoint]


class SmileBuildResult(SurfaceModel):
    """A constructed smile and its local construction diagnostics."""

    smile: VolatilitySmile
    diagnostics: SmileConstructionDiagnostics


class SurfaceBuildResult(SurfaceModel):
    """A constructed surface with descriptive quality information."""

    surface: VolatilitySurface
    quality_summary: SurfaceQualitySummary
    diagnostics: SmileConstructionDiagnostics


def metadata_dict(**values: Any) -> dict[str, Any]:
    """Small typed-friendly helper used by callers that assemble metadata."""

    return values
