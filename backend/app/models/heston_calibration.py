"""Typed Heston calibration datasets, controls, diagnostics, and results."""

import math
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.app.models.heston import HestonIntegrationConfig, HestonParameters
from backend.app.models.option import OptionType

PositiveFloat = Annotated[float, Field(gt=0.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]


class CalibrationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class HestonCalibrationTarget(StrEnum):
    PRICE = "PRICE"
    IMPLIED_VOLATILITY = "IMPLIED_VOLATILITY"
    TOTAL_VARIANCE = "TOTAL_VARIANCE"


class CalibrationWeighting(StrEnum):
    UNIFORM = "UNIFORM"
    VEGA = "VEGA"
    INVERSE_VEGA = "INVERSE_VEGA"
    BID_ASK = "BID_ASK"
    CUSTOM = "CUSTOM"


class HestonCalibrationOptimizer(StrEnum):
    LEAST_SQUARES = "LEAST_SQUARES"
    DIFFERENTIAL_EVOLUTION = "DIFFERENTIAL_EVOLUTION"
    HYBRID = "HYBRID"


class CalibrationRobustLoss(StrEnum):
    LINEAR = "LINEAR"
    SOFT_L1 = "SOFT_L1"
    HUBER = "HUBER"


class CalibrationAccuracy(StrEnum):
    FAST = "FAST"
    STANDARD = "STANDARD"
    HIGH_ACCURACY = "HIGH_ACCURACY"


class CalibrationSelectionPolicy(StrEnum):
    ALL = "ALL"
    OTM_ONLY = "OTM_ONLY"


class CalibrationMoneynessBucket(StrEnum):
    DEEP_OTM = "DEEP_OTM"
    OTM = "OTM"
    ATM = "ATM"
    ITM = "ITM"


class CalibrationStage(StrEnum):
    LOCAL = "LOCAL"
    GLOBAL = "GLOBAL"
    GLOBAL_REFINEMENT = "GLOBAL_REFINEMENT"


class CalibrationHoldoutPolicy(StrEnum):
    EXPIRY = "EXPIRY"
    EVERY_NTH_STRIKE = "EVERY_NTH_STRIKE"
    WINGS = "WINGS"


class HestonCalibrationObservation(CalibrationModel):
    strike: PositiveFloat
    maturity_years: PositiveFloat
    option_type: OptionType
    observed_price: NonNegativeFloat | None = None
    observed_implied_volatility: NonNegativeFloat | None = None
    bid: NonNegativeFloat | None = None
    ask: NonNegativeFloat | None = None
    mid: NonNegativeFloat | None = None
    weight: NonNegativeFloat | None = None
    source: str | None = None
    quote_timestamp: datetime | None = None
    quality_issues: list[str] = Field(default_factory=list)
    quote_is_fresh: bool | None = None
    arbitrage_valid: bool | None = None

    @field_validator("quote_timestamp")
    @classmethod
    def normalize_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("quote_timestamp must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("source")
    @classmethod
    def normalize_source(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class HestonCalibrationDataset(CalibrationModel):
    spot: PositiveFloat
    risk_free_rate: float
    dividend_yield: float = 0.0
    observations: list[HestonCalibrationObservation] = Field(min_length=1)
    valuation_timestamp: datetime | None = None
    underlying_symbol: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("valuation_timestamp")
    @classmethod
    def normalize_valuation_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("valuation_timestamp must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("underlying_symbol")
    @classmethod
    def normalize_symbol(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().upper()
        return normalized or None


class CalibrationParameterRange(CalibrationModel):
    lower: float
    upper: float

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        if self.lower >= self.upper:
            raise ValueError("calibration bound lower must be strictly below upper")
        return self


class HestonCalibrationBounds(CalibrationModel):
    v0: CalibrationParameterRange = Field(
        default_factory=lambda: CalibrationParameterRange(lower=1e-6, upper=2.0)
    )
    kappa: CalibrationParameterRange = Field(
        default_factory=lambda: CalibrationParameterRange(lower=1e-4, upper=20.0)
    )
    theta: CalibrationParameterRange = Field(
        default_factory=lambda: CalibrationParameterRange(lower=1e-6, upper=2.0)
    )
    sigma_v: CalibrationParameterRange = Field(
        default_factory=lambda: CalibrationParameterRange(lower=1e-4, upper=5.0)
    )
    rho: CalibrationParameterRange = Field(
        default_factory=lambda: CalibrationParameterRange(lower=-0.999, upper=0.999)
    )

    @model_validator(mode="after")
    def validate_heston_domain(self) -> Self:
        for name in ("v0", "kappa", "theta", "sigma_v"):
            if getattr(self, name).lower < 0.0:
                raise ValueError(f"{name} calibration lower bound must be non-negative")
        if self.rho.lower < -1.0 or self.rho.upper > 1.0:
            raise ValueError("rho calibration bounds must lie in [-1, 1]")
        return self


class HestonCalibrationFilters(CalibrationModel):
    minimum_log_moneyness: float | None = None
    maximum_log_moneyness: float | None = None
    minimum_strike_to_spot: PositiveFloat | None = None
    maximum_strike_to_spot: PositiveFloat | None = None
    minimum_price: NonNegativeFloat | None = None
    maximum_relative_spread: NonNegativeFloat | None = None
    allowed_maturities: list[PositiveFloat] | None = None
    exclude_stale: bool = True
    exclude_crossed: bool = True
    exclude_missing_mid: bool = False
    exclude_arbitrage_invalid: bool = True

    @model_validator(mode="after")
    def validate_ranges(self) -> Self:
        if (
            self.minimum_log_moneyness is not None
            and self.maximum_log_moneyness is not None
            and self.minimum_log_moneyness > self.maximum_log_moneyness
        ):
            raise ValueError("minimum_log_moneyness must not exceed maximum_log_moneyness")
        if (
            self.minimum_strike_to_spot is not None
            and self.maximum_strike_to_spot is not None
            and self.minimum_strike_to_spot > self.maximum_strike_to_spot
        ):
            raise ValueError("minimum_strike_to_spot must not exceed maximum_strike_to_spot")
        if self.allowed_maturities is not None and not self.allowed_maturities:
            raise ValueError("allowed_maturities must be nonempty when supplied")
        return self


class HestonCalibrationConfig(CalibrationModel):
    target: HestonCalibrationTarget = HestonCalibrationTarget.IMPLIED_VOLATILITY
    weighting: CalibrationWeighting = CalibrationWeighting.UNIFORM
    bounds: HestonCalibrationBounds = Field(default_factory=HestonCalibrationBounds)
    optimizer: HestonCalibrationOptimizer = HestonCalibrationOptimizer.LEAST_SQUARES
    accuracy: CalibrationAccuracy = CalibrationAccuracy.STANDARD
    max_iterations: PositiveInt = 100
    tolerance: PositiveFloat = 1e-8
    multi_start_count: PositiveInt = 1
    random_seed: NonNegativeInt = 42
    feller_penalty_weight: NonNegativeFloat = 0.0
    expiry_balancing: bool = False
    selection_policy: CalibrationSelectionPolicy = CalibrationSelectionPolicy.OTM_ONLY
    filters: HestonCalibrationFilters = Field(default_factory=HestonCalibrationFilters)
    local_refinement: bool = True
    robust_loss: CalibrationRobustLoss = CalibrationRobustLoss.LINEAR
    return_residuals: bool = True
    max_returned_residuals: PositiveInt = 500
    return_start_diagnostics: bool = True
    max_returned_starts: PositiveInt = 20
    identifiability_diagnostics: bool = True
    f_scale: PositiveFloat = 1.0
    spread_floor: PositiveFloat = 0.01
    vega_floor: PositiveFloat = 1e-4
    failure_penalty: PositiveFloat = 1_000.0
    differential_evolution_population: PositiveInt = 5
    initial_parameters: HestonParameters | None = None
    regularization_weight: NonNegativeFloat = 0.0
    regularization_parameters: HestonParameters | None = None
    integration: HestonIntegrationConfig | None = None

    @model_validator(mode="after")
    def validate_dependencies(self) -> Self:
        if self.regularization_weight > 0.0 and self.regularization_parameters is None:
            raise ValueError("positive regularization_weight requires regularization_parameters")
        return self


class CalibrationExclusion(CalibrationModel):
    observation_index: NonNegativeInt
    strike: PositiveFloat
    maturity_years: PositiveFloat
    option_type: OptionType
    reason: str


class HestonCalibrationResidual(CalibrationModel):
    observation_index: NonNegativeInt
    strike: PositiveFloat
    maturity_years: PositiveFloat
    option_type: OptionType
    market_price: NonNegativeFloat | None
    model_price: NonNegativeFloat | None
    market_implied_volatility: NonNegativeFloat | None
    model_implied_volatility: NonNegativeFloat | None
    raw_residual: float
    weighted_residual: float
    weight: NonNegativeFloat
    moneyness: PositiveFloat
    log_moneyness: float
    target: HestonCalibrationTarget
    evaluation_failed: bool = False
    failure_message: str | None = None


class HestonCalibrationMetrics(CalibrationModel):
    observation_count: NonNegativeInt
    rmse: NonNegativeFloat
    mae: NonNegativeFloat
    weighted_rmse: NonNegativeFloat
    max_abs_error: NonNegativeFloat
    average_iv_error: float | None = None
    objective_contribution: NonNegativeFloat


class HestonExpiryCalibrationDiagnostics(HestonCalibrationMetrics):
    maturity_years: PositiveFloat


class HestonMoneynessCalibrationDiagnostics(HestonCalibrationMetrics):
    bucket: CalibrationMoneynessBucket
    lower_signed_moneyness: float | None
    upper_signed_moneyness: float | None


class HestonIdentifiabilityDiagnostics(CalibrationModel):
    jacobian_rank: NonNegativeInt
    parameter_count: PositiveInt = 5
    singular_values: list[NonNegativeFloat]
    condition_number: NonNegativeFloat | None
    locally_full_rank: bool
    covariance_available: bool
    local_covariance: list[list[float]] | None = None
    local_correlation: list[list[float]] | None = None
    note: str


class HestonCalibrationStartResult(CalibrationModel):
    start_index: NonNegativeInt
    stage: CalibrationStage
    initial_parameters: HestonParameters
    final_parameters: HestonParameters
    objective_value: NonNegativeFloat
    cost: NonNegativeFloat
    optimality: NonNegativeFloat | None = None
    converged: bool
    status: int
    message: str
    iterations: NonNegativeInt
    evaluations: NonNegativeInt
    jacobian_evaluations: NonNegativeInt | None = None
    runtime_ms: NonNegativeFloat


class HestonCalibrationResult(CalibrationModel):
    calibrated_parameters: HestonParameters
    objective_value: NonNegativeFloat
    cost: NonNegativeFloat
    optimality: NonNegativeFloat | None
    weighted_rmse: NonNegativeFloat
    rmse: NonNegativeFloat
    mae: NonNegativeFloat
    max_abs_error: NonNegativeFloat
    target: HestonCalibrationTarget
    weighting: CalibrationWeighting
    optimizer: HestonCalibrationOptimizer
    converged: bool
    status: int
    message: str
    iterations: NonNegativeInt
    evaluations: NonNegativeInt
    jacobian_evaluations: NonNegativeInt | None
    pricing_evaluations: NonNegativeInt
    failed_evaluations: NonNegativeInt
    cache_hits: NonNegativeInt
    runtime_ms: NonNegativeFloat
    prices_per_second: NonNegativeFloat
    feller_satisfied: bool
    feller_ratio: NonNegativeFloat | None
    penalty_value: NonNegativeFloat
    regularization_value: NonNegativeFloat
    observations_used: PositiveInt
    observations_excluded: NonNegativeInt
    exclusion_reasons: dict[str, int]
    initial_parameters: HestonParameters
    best_start_index: NonNegativeInt
    active_bounds: list[str]
    residual_summary: HestonCalibrationMetrics
    residuals: list[HestonCalibrationResidual]
    exclusions: list[CalibrationExclusion]
    per_expiry: list[HestonExpiryCalibrationDiagnostics]
    per_moneyness: list[HestonMoneynessCalibrationDiagnostics]
    identifiability: HestonIdentifiabilityDiagnostics | None
    starts: list[HestonCalibrationStartResult]


class HestonCalibrationDiagnosticsResult(CalibrationModel):
    parameters: HestonParameters
    target: HestonCalibrationTarget
    weighting: CalibrationWeighting
    metrics: HestonCalibrationMetrics
    penalty_value: NonNegativeFloat
    residuals: list[HestonCalibrationResidual]
    per_expiry: list[HestonExpiryCalibrationDiagnostics]
    per_moneyness: list[HestonMoneynessCalibrationDiagnostics]
    identifiability: HestonIdentifiabilityDiagnostics | None
    observations_used: PositiveInt
    observations_excluded: NonNegativeInt
    exclusions: list[CalibrationExclusion]
    failed_evaluations: NonNegativeInt


class HestonFittedSurfacePoint(CalibrationModel):
    strike: PositiveFloat
    maturity_years: PositiveFloat
    option_type: OptionType
    market_price: NonNegativeFloat | None
    heston_price: NonNegativeFloat
    market_implied_volatility: NonNegativeFloat | None
    heston_implied_volatility: NonNegativeFloat | None
    price_residual: float | None
    implied_volatility_residual: float | None
    moneyness: PositiveFloat
    log_moneyness: float


class HestonATMFitPoint(CalibrationModel):
    maturity_years: PositiveFloat
    strike: PositiveFloat
    market_implied_volatility: NonNegativeFloat | None
    heston_implied_volatility: NonNegativeFloat | None
    residual: float | None


class HestonSkewFitPoint(CalibrationModel):
    maturity_years: PositiveFloat
    market_skew: float | None
    heston_skew: float | None
    residual: float | None
    observation_count: NonNegativeInt


class HestonFittedSurfaceResult(CalibrationModel):
    parameters: HestonParameters
    points: list[HestonFittedSurfacePoint]
    atm_term_structure: list[HestonATMFitPoint]
    skew_by_expiry: list[HestonSkewFitPoint]
    metrics: HestonCalibrationMetrics
    observations_excluded: NonNegativeInt
    exclusions: list[CalibrationExclusion]


class HestonCalibrationStabilityRun(CalibrationModel):
    run_index: NonNegativeInt
    seed: NonNegativeInt
    parameters: HestonParameters
    objective_value: NonNegativeFloat
    converged: bool
    evaluations: NonNegativeInt
    runtime_ms: NonNegativeFloat


class HestonCalibrationStabilityResult(CalibrationModel):
    runs: list[HestonCalibrationStabilityRun]
    best_run_index: NonNegativeInt
    worst_run_index: NonNegativeInt
    objective_mean: NonNegativeFloat
    objective_std: NonNegativeFloat
    parameter_means: dict[str, float]
    parameter_standard_deviations: dict[str, NonNegativeFloat]


class HestonCalibrationHoldoutConfig(CalibrationModel):
    policy: CalibrationHoldoutPolicy
    maturities: list[PositiveFloat] = Field(default_factory=list)
    every_nth_strike: PositiveInt = 3
    wing_log_moneyness: PositiveFloat = 0.15

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        if self.policy is CalibrationHoldoutPolicy.EXPIRY and not self.maturities:
            raise ValueError("expiry holdout requires at least one maturity")
        return self


class HestonCalibrationValidationResult(CalibrationModel):
    calibration: HestonCalibrationResult
    policy: CalibrationHoldoutPolicy
    training_observations: PositiveInt
    validation_observations: PositiveInt
    validation_metrics: HestonCalibrationMetrics
    validation_residuals: list[HestonCalibrationResidual]


def calibration_metadata(**values: Any) -> dict[str, Any]:
    return values


def finite_parameter_vector(values: list[float]) -> bool:
    return len(values) == 5 and all(math.isfinite(value) for value in values)
