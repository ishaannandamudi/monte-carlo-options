"""Typed digital-option contracts, configurations, and pricing results."""

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.models.option import ExerciseStyle, OptionType
from backend.app.models.simulation import ConfidenceLevel, NonNegativeInt, PositiveInt, RandomEngine

PositiveFloat = Annotated[float, Field(gt=0.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
Fraction = Annotated[float, Field(ge=0.0, le=1.0)]


class DigitalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class DigitalType(StrEnum):
    CASH_OR_NOTHING = "CASH_OR_NOTHING"
    ASSET_OR_NOTHING = "ASSET_OR_NOTHING"


class DigitalPricingMethod(StrEnum):
    ANALYTICAL = "ANALYTICAL"
    MONTE_CARLO = "MONTE_CARLO"


class DigitalVarianceReduction(StrEnum):
    NONE = "NONE"
    ANTITHETIC = "ANTITHETIC"

    @property
    def uses_antithetic(self) -> bool:
        return self is self.ANTITHETIC


class DigitalOptionContract(DigitalModel):
    option_type: OptionType
    spot: PositiveFloat
    strike: PositiveFloat
    maturity_years: NonNegativeFloat
    digital_type: DigitalType = DigitalType.CASH_OR_NOTHING
    cash_payout: NonNegativeFloat | None = None
    exercise_style: Literal[ExerciseStyle.EUROPEAN] = ExerciseStyle.EUROPEAN

    @model_validator(mode="before")
    @classmethod
    def default_cash_payout(cls, value: object) -> object:
        if isinstance(value, cls) or not isinstance(value, dict):
            return value
        data = dict(value)
        kind = data.get("digital_type", DigitalType.CASH_OR_NOTHING)
        if kind in {DigitalType.CASH_OR_NOTHING, DigitalType.CASH_OR_NOTHING.value}:
            data.setdefault("cash_payout", 1.0)
        return data

    @model_validator(mode="after")
    def validate_payout(self) -> Self:
        if self.digital_type is DigitalType.CASH_OR_NOTHING and self.cash_payout is None:
            raise ValueError("cash_payout is required for CASH_OR_NOTHING")
        if self.digital_type is DigitalType.ASSET_OR_NOTHING and self.cash_payout is not None:
            raise ValueError("cash_payout is not used for ASSET_OR_NOTHING")
        return self


class DigitalSimulationConfig(DigitalModel):
    num_simulations: PositiveInt = 100_000
    seed: NonNegativeInt = 42
    variance_reduction: DigitalVarianceReduction = DigitalVarianceReduction.ANTITHETIC
    random_engine: RandomEngine = RandomEngine.PCG64
    confidence_level: ConfidenceLevel = 0.95


class DigitalAnalyticalResult(DigitalModel):
    method: Literal[DigitalPricingMethod.ANALYTICAL] = DigitalPricingMethod.ANALYTICAL
    price: NonNegativeFloat
    digital_type: DigitalType
    option_type: OptionType
    cash_payout: NonNegativeFloat | None
    probability_itm: Fraction
    deterministic: bool
    runtime_ms: NonNegativeFloat


class DigitalMonteCarloResult(DigitalModel):
    method: Literal[DigitalPricingMethod.MONTE_CARLO] = DigitalPricingMethod.MONTE_CARLO
    price: NonNegativeFloat
    standard_error: NonNegativeFloat
    estimator_variance: NonNegativeFloat
    confidence_interval_low: float
    confidence_interval_high: float
    confidence_level: float = Field(gt=0.0, lt=1.0)
    digital_type: DigitalType
    option_type: OptionType
    cash_payout: NonNegativeFloat | None
    probability_itm: Fraction
    num_simulations: PositiveInt
    seed: NonNegativeInt
    variance_reduction: DigitalVarianceReduction
    runtime_ms: NonNegativeFloat
    deterministic: bool
    analytical_price: NonNegativeFloat
    absolute_error: NonNegativeFloat


class DigitalComparison(DigitalModel):
    monte_carlo: DigitalMonteCarloResult
    analytical: DigitalAnalyticalResult
    absolute_error: NonNegativeFloat
    relative_error: NonNegativeFloat | None
    z_score: float | None
    analytical_inside_confidence_interval: bool


class DigitalConvergencePoint(DigitalModel):
    num_simulations: PositiveInt
    seed: NonNegativeInt
    monte_carlo_price: NonNegativeFloat
    analytical_price: NonNegativeFloat
    absolute_error: NonNegativeFloat
    relative_error: NonNegativeFloat | None
    standard_error: NonNegativeFloat
    confidence_interval_width: NonNegativeFloat
    runtime_ms: NonNegativeFloat


class DigitalConvergenceResult(DigitalModel):
    analytical_price: NonNegativeFloat
    root_seed: NonNegativeInt
    variance_reduction: DigitalVarianceReduction
    points: list[DigitalConvergencePoint]
