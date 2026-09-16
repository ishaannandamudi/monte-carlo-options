"""Typed contracts, configurations, and results for fixed-strike Asian options."""

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.models.monitoring import MonitoringSchedule
from backend.app.models.option import ExerciseStyle, OptionType
from backend.app.models.simulation import ConfidenceLevel, NonNegativeInt, PositiveInt, RandomEngine

PositiveFloat = Annotated[float, Field(gt=0.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
Fraction = Annotated[float, Field(ge=0.0, le=1.0)]


class AsianModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class AveragingType(StrEnum):
    ARITHMETIC = "ARITHMETIC"
    GEOMETRIC = "GEOMETRIC"


class AsianVarianceReduction(StrEnum):
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


class AsianConvergenceMode(StrEnum):
    PATH_COUNT = "PATH_COUNT"
    MONITORING_COUNT = "MONITORING_COUNT"


class AsianOptionContract(AsianModel):
    """European fixed-strike option on a discrete stock-price average."""

    option_type: OptionType
    spot: PositiveFloat
    strike: PositiveFloat
    maturity_years: NonNegativeFloat
    averaging_type: AveragingType
    averaging_schedule: MonitoringSchedule = Field(default_factory=MonitoringSchedule)
    include_initial_spot_in_average: bool = False
    exercise_style: Literal[ExerciseStyle.EUROPEAN] = ExerciseStyle.EUROPEAN

    @model_validator(mode="after")
    def validate_schedule(self) -> Self:
        self.averaging_schedule.validate_for_contract(
            self.maturity_years,
            include_initial_spot=self.include_initial_spot_in_average,
        )
        return self

    @property
    def observation_times(self) -> tuple[float, ...]:
        return self.averaging_schedule.resolve(
            self.maturity_years,
            include_initial_spot=self.include_initial_spot_in_average,
        )

    @property
    def future_monitoring_times(self) -> tuple[float, ...]:
        return tuple(time for time in self.observation_times if time > 0.0)

    @property
    def monitoring_count(self) -> int:
        if self.maturity_years == 0.0:
            return 1
        if self.averaging_schedule.monitoring_count is not None:
            return self.averaging_schedule.monitoring_count + int(
                self.include_initial_spot_in_average
            )
        assert self.averaging_schedule.monitoring_times is not None
        has_zero = self.averaging_schedule.monitoring_times[0] == 0.0
        return len(self.averaging_schedule.monitoring_times) + int(
            self.include_initial_spot_in_average and not has_zero
        )


class AsianSimulationConfig(AsianModel):
    num_simulations: PositiveInt = 100_000
    seed: NonNegativeInt = 42
    variance_reduction: AsianVarianceReduction = AsianVarianceReduction.ANTITHETIC
    random_engine: RandomEngine = RandomEngine.PCG64
    confidence_level: ConfidenceLevel = 0.95


class GeometricAsianAnalyticalResult(AsianModel):
    price: NonNegativeFloat
    mean_log_average: float
    variance_log_average: NonNegativeFloat
    expected_geometric_average: PositiveFloat
    monitoring_count: PositiveInt
    include_initial_spot_in_average: bool
    deterministic: bool


class AsianPricingResult(AsianModel):
    price: float
    standard_error: NonNegativeFloat
    estimator_variance: NonNegativeFloat
    confidence_interval_low: float
    confidence_interval_high: float
    confidence_level: float = Field(gt=0.0, lt=1.0)
    num_simulations: PositiveInt
    seed: NonNegativeInt
    variance_reduction: AsianVarianceReduction
    averaging_type: AveragingType
    monitoring_count: PositiveInt
    future_monitoring_count: int = Field(ge=0)
    include_initial_spot_in_average: bool
    runtime_ms: NonNegativeFloat
    simulation_runtime_ms: NonNegativeFloat
    mean_average: PositiveFloat
    average_std: NonNegativeFloat
    probability_itm: Fraction
    deterministic: bool
    raw_price: NonNegativeFloat | None = None
    adjusted_price: float | None = None
    raw_standard_error: NonNegativeFloat | None = None
    adjusted_standard_error: NonNegativeFloat | None = None
    raw_estimator_variance: NonNegativeFloat | None = None
    adjusted_estimator_variance: NonNegativeFloat | None = None
    control_variate_beta: float | None = None
    control_sample_mean: NonNegativeFloat | None = None
    analytical_geometric_control_price: NonNegativeFloat | None = None
    variance_reduction_ratio: NonNegativeFloat | None = None


class AsianComparisonPoint(AsianModel):
    method: AsianVarianceReduction
    price: float
    standard_error: NonNegativeFloat
    estimator_variance: NonNegativeFloat
    confidence_interval_width: NonNegativeFloat
    runtime_ms: NonNegativeFloat
    analytical_reference: NonNegativeFloat | None = None
    absolute_error: NonNegativeFloat | None = None
    variance_reduction_ratio: NonNegativeFloat | None = None


class AsianComparisonResult(AsianModel):
    averaging_type: AveragingType
    num_simulations: PositiveInt
    seed: NonNegativeInt
    monitoring_count: PositiveInt
    points: list[AsianComparisonPoint]


class AsianConvergencePoint(AsianModel):
    num_simulations: PositiveInt
    monitoring_count: PositiveInt
    seed: NonNegativeInt
    price: float
    standard_error: NonNegativeFloat
    confidence_interval_width: NonNegativeFloat
    runtime_ms: NonNegativeFloat
    analytical_reference: NonNegativeFloat | None = None
    absolute_error: NonNegativeFloat | None = None


class AsianConvergenceResult(AsianModel):
    mode: AsianConvergenceMode
    root_seed: NonNegativeInt
    variance_reduction: AsianVarianceReduction
    points: list[AsianConvergencePoint]
