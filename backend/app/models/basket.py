"""Typed contracts, configurations, and results for European basket options."""

import math
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.models.multi_asset import CorrelationFactorizationDiagnostics
from backend.app.models.option import ExerciseStyle, OptionType
from backend.app.models.simulation import ConfidenceLevel, NonNegativeInt, PositiveInt, RandomEngine

PositiveFloat = Annotated[float, Field(gt=0.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
Fraction = Annotated[float, Field(ge=0.0, le=1.0)]
WEIGHT_SUM_TOLERANCE = 1e-10


class BasketModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class BasketType(StrEnum):
    ARITHMETIC = "ARITHMETIC"
    GEOMETRIC = "GEOMETRIC"


class BasketVarianceReduction(StrEnum):
    NONE = "NONE"
    ANTITHETIC = "ANTITHETIC"
    GEOMETRIC_CONTROL = "GEOMETRIC_CONTROL"
    ANTITHETIC_GEOMETRIC_CONTROL = "ANTITHETIC_GEOMETRIC_CONTROL"

    @property
    def uses_antithetic(self) -> bool:
        return self in {self.ANTITHETIC, self.ANTITHETIC_GEOMETRIC_CONTROL}

    @property
    def uses_geometric_control(self) -> bool:
        return self in {self.GEOMETRIC_CONTROL, self.ANTITHETIC_GEOMETRIC_CONTROL}


class BasketOptionContract(BasketModel):
    """European fixed-strike option on a normalized nonnegative weighted basket."""

    option_type: OptionType
    strike: PositiveFloat
    maturity_years: NonNegativeFloat
    weights: tuple[NonNegativeFloat, ...] = Field(min_length=1)
    basket_type: BasketType
    exercise_style: Literal[ExerciseStyle.EUROPEAN] = ExerciseStyle.EUROPEAN

    @model_validator(mode="after")
    def validate_and_normalize_weights(self) -> Self:
        total = math.fsum(self.weights)
        if total <= 0.0:
            raise ValueError("at least one basket weight must be positive")
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=WEIGHT_SUM_TOLERANCE):
            raise ValueError("basket weights must sum to one")
        leading = tuple(weight / total for weight in self.weights[:-1])
        normalized = (*leading, 1.0 - math.fsum(leading))
        object.__setattr__(self, "weights", normalized)
        return self


class BasketSimulationConfig(BasketModel):
    num_simulations: PositiveInt = 100_000
    seed: NonNegativeInt = 42
    variance_reduction: BasketVarianceReduction = BasketVarianceReduction.ANTITHETIC
    random_engine: RandomEngine = RandomEngine.PCG64
    confidence_level: ConfidenceLevel = 0.95


class GeometricBasketAnalyticalResult(BasketModel):
    price: NonNegativeFloat
    mean_log_basket: float
    variance_log_basket: NonNegativeFloat
    expected_geometric_basket: PositiveFloat
    num_assets: PositiveInt
    weights: tuple[NonNegativeFloat, ...]
    deterministic: bool


class BasketPricingResult(BasketModel):
    price: float
    standard_error: NonNegativeFloat
    estimator_variance: NonNegativeFloat
    confidence_interval_low: float
    confidence_interval_high: float
    confidence_level: ConfidenceLevel
    num_simulations: PositiveInt
    seed: NonNegativeInt
    num_assets: PositiveInt
    basket_type: BasketType
    weights: tuple[NonNegativeFloat, ...]
    variance_reduction: BasketVarianceReduction
    runtime_ms: NonNegativeFloat
    simulation_runtime_ms: NonNegativeFloat
    mean_basket_value: PositiveFloat
    basket_std: NonNegativeFloat
    probability_itm: Fraction
    deterministic: bool
    factorization: CorrelationFactorizationDiagnostics
    raw_price: NonNegativeFloat | None = None
    adjusted_price: float | None = None
    raw_standard_error: NonNegativeFloat | None = None
    adjusted_standard_error: NonNegativeFloat | None = None
    raw_estimator_variance: NonNegativeFloat | None = None
    adjusted_estimator_variance: NonNegativeFloat | None = None
    control_variate_beta: float | None = None
    control_sample_mean: NonNegativeFloat | None = None
    analytical_geometric_price: NonNegativeFloat | None = None
    variance_reduction_ratio: NonNegativeFloat | None = None


class BasketComparisonPoint(BasketModel):
    method: BasketVarianceReduction
    price: float
    standard_error: NonNegativeFloat
    estimator_variance: NonNegativeFloat
    confidence_interval_width: NonNegativeFloat
    runtime_ms: NonNegativeFloat
    analytical_reference: NonNegativeFloat | None = None
    absolute_error: NonNegativeFloat | None = None
    variance_reduction_ratio: NonNegativeFloat | None = None


class BasketComparisonResult(BasketModel):
    basket_type: BasketType
    num_simulations: PositiveInt
    seed: NonNegativeInt
    num_assets: PositiveInt
    points: list[BasketComparisonPoint]


class BasketConvergencePoint(BasketModel):
    num_simulations: PositiveInt
    seed: NonNegativeInt
    price: float
    standard_error: NonNegativeFloat
    confidence_interval_width: NonNegativeFloat
    runtime_ms: NonNegativeFloat
    analytical_reference: NonNegativeFloat | None = None
    absolute_error: NonNegativeFloat | None = None
    variance_reduction_ratio: NonNegativeFloat | None = None


class BasketConvergenceResult(BasketModel):
    root_seed: NonNegativeInt
    variance_reduction: BasketVarianceReduction
    basket_type: BasketType
    points: list[BasketConvergencePoint]
    empirical_standard_error_slope: float | None
    slope_point_count: int = Field(ge=0)


class CorrelationSensitivityPoint(BasketModel):
    rho: float = Field(ge=-1.0, le=1.0)
    price: float
    standard_error: NonNegativeFloat
    mean_basket_value: PositiveFloat
    basket_variance: NonNegativeFloat
    runtime_ms: NonNegativeFloat


class CorrelationSensitivityResult(BasketModel):
    seed: NonNegativeInt
    num_assets: PositiveInt
    points: list[CorrelationSensitivityPoint]


class WeightSensitivityPoint(BasketModel):
    weights: tuple[NonNegativeFloat, NonNegativeFloat]
    price: float
    standard_error: NonNegativeFloat
    mean_basket_value: PositiveFloat
    basket_variance: NonNegativeFloat
    runtime_ms: NonNegativeFloat


class WeightSensitivityResult(BasketModel):
    seed: NonNegativeInt
    points: list[WeightSensitivityPoint]
