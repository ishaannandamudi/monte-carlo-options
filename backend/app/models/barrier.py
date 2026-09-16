"""Typed contracts, configurations, and results for single-barrier options."""

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.models.monitoring import MonitoringSchedule
from backend.app.models.option import ExerciseStyle, OptionType
from backend.app.models.simulation import ConfidenceLevel, NonNegativeInt, PositiveInt, RandomEngine

PositiveFloat = Annotated[float, Field(gt=0.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class BarrierModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class BarrierDirection(StrEnum):
    UP = "UP"
    DOWN = "DOWN"


class BarrierStyle(StrEnum):
    IN = "IN"
    OUT = "OUT"


class BarrierMonitoringMode(StrEnum):
    DISCRETE = "DISCRETE"
    BROWNIAN_BRIDGE = "BROWNIAN_BRIDGE"


class RebateTiming(StrEnum):
    AT_HIT = "AT_HIT"
    AT_EXPIRY = "AT_EXPIRY"


class BarrierVarianceReduction(StrEnum):
    NONE = "NONE"
    ANTITHETIC = "ANTITHETIC"

    @property
    def uses_antithetic(self) -> bool:
        return self is self.ANTITHETIC


class BarrierConvergenceMode(StrEnum):
    PATH_COUNT = "PATH_COUNT"
    MONITORING_COUNT = "MONITORING_COUNT"


class BarrierOptionContract(BarrierModel):
    """European fixed-strike option with one up or down barrier."""

    option_type: OptionType
    barrier_direction: BarrierDirection
    barrier_style: BarrierStyle
    spot: PositiveFloat
    strike: PositiveFloat
    barrier: PositiveFloat
    maturity_years: NonNegativeFloat
    monitoring_schedule: MonitoringSchedule = Field(default_factory=MonitoringSchedule)
    monitoring_mode: BarrierMonitoringMode = BarrierMonitoringMode.DISCRETE
    rebate: NonNegativeFloat = 0.0
    rebate_timing: RebateTiming = RebateTiming.AT_EXPIRY
    exercise_style: Literal[ExerciseStyle.EUROPEAN] = ExerciseStyle.EUROPEAN

    @model_validator(mode="after")
    def validate_barrier_terms(self) -> Self:
        self.monitoring_schedule.validate_for_contract(
            self.maturity_years,
            include_initial_spot=True,
        )
        if self.barrier_style is BarrierStyle.IN and self.rebate_timing is RebateTiming.AT_HIT:
            raise ValueError("knock-in rebates are supported only AT_EXPIRY on no hit")
        if (
            self.monitoring_mode is BarrierMonitoringMode.BROWNIAN_BRIDGE
            and self.rebate > 0.0
            and self.rebate_timing is RebateTiming.AT_HIT
        ):
            raise ValueError(
                "Brownian-bridge survival weighting does not support positive AT_HIT rebates"
            )
        return self

    @property
    def monitoring_times(self) -> tuple[float, ...]:
        """Return scheduled future barrier observations, excluding the mandatory t=0 check."""

        return tuple(
            time
            for time in self.monitoring_schedule.resolve(
                self.maturity_years,
                include_initial_spot=True,
            )
            if time > 0.0
        )

    @property
    def monitoring_count(self) -> int:
        """Return future scheduled observations without allocating a regular schedule."""

        if self.maturity_years == 0.0:
            return 0
        if self.monitoring_schedule.monitoring_count is not None:
            return self.monitoring_schedule.monitoring_count
        assert self.monitoring_schedule.monitoring_times is not None
        return sum(time > 0.0 for time in self.monitoring_schedule.monitoring_times)

    @property
    def simulation_times(self) -> tuple[float, ...]:
        """Return future barrier endpoints plus maturity for terminal payoff evaluation."""

        if self.maturity_years == 0.0:
            return ()
        return tuple(sorted({*self.monitoring_times, self.maturity_years}))

    @property
    def simulation_state_count(self) -> int:
        """Return future stored states, including maturity, without expanding regular times."""

        if self.maturity_years == 0.0:
            return 0
        if self.monitoring_schedule.monitoring_count is not None:
            return self.monitoring_schedule.monitoring_count
        assert self.monitoring_schedule.monitoring_times is not None
        future = tuple(time for time in self.monitoring_schedule.monitoring_times if time > 0.0)
        return len(future) + int(self.maturity_years not in future)

    @property
    def initially_hit(self) -> bool:
        if self.barrier_direction is BarrierDirection.DOWN:
            return self.spot <= self.barrier
        return self.spot >= self.barrier


class BarrierSimulationConfig(BarrierModel):
    num_simulations: PositiveInt = 100_000
    seed: NonNegativeInt = 42
    variance_reduction: BarrierVarianceReduction = BarrierVarianceReduction.ANTITHETIC
    random_engine: RandomEngine = RandomEngine.PCG64
    confidence_level: ConfidenceLevel = 0.95


class BarrierAnalyticalResult(BarrierModel):
    price: NonNegativeFloat
    barrier_direction: BarrierDirection
    barrier_style: BarrierStyle
    barrier: PositiveFloat
    rebate: NonNegativeFloat
    initially_hit: bool
    deterministic: bool
    runtime_ms: NonNegativeFloat


class BarrierPricingResult(BarrierModel):
    price: NonNegativeFloat
    standard_error: NonNegativeFloat
    estimator_variance: NonNegativeFloat
    confidence_interval_low: float
    confidence_interval_high: float
    confidence_level: float = Field(gt=0.0, lt=1.0)
    num_simulations: PositiveInt
    seed: NonNegativeInt
    variance_reduction: BarrierVarianceReduction
    barrier_direction: BarrierDirection
    barrier_style: BarrierStyle
    barrier: PositiveFloat
    monitoring_mode: BarrierMonitoringMode
    monitoring_count: int = Field(ge=0)
    simulation_state_count: int = Field(ge=0)
    rebate: NonNegativeFloat
    rebate_timing: RebateTiming
    hit_probability: Probability
    no_hit_probability: Probability
    activation_probability: Probability
    mean_hit_time: NonNegativeFloat | None
    median_hit_time: NonNegativeFloat | None
    initially_hit: bool
    deterministic: bool
    survival_probability_weighting: bool
    runtime_ms: NonNegativeFloat
    simulation_runtime_ms: NonNegativeFloat
    analytical_continuous_price: NonNegativeFloat | None = None


class BarrierComparisonResult(BarrierModel):
    discrete: BarrierPricingResult
    brownian_bridge: BarrierPricingResult
    analytical_continuous: BarrierAnalyticalResult | None
    bridge_minus_discrete: float
    hit_probability_difference: float
    analytical_bridge_error: NonNegativeFloat | None
    discrete_parity_residual: float | None
    bridge_parity_residual: float | None


class BarrierParityResult(BarrierModel):
    knock_in_price: NonNegativeFloat
    knock_out_price: NonNegativeFloat
    vanilla_price: NonNegativeFloat
    parity_residual: float
    monitoring_mode: BarrierMonitoringMode
    num_simulations: PositiveInt
    seed: NonNegativeInt


class BarrierConvergencePoint(BarrierModel):
    num_simulations: PositiveInt
    monitoring_count: int = Field(ge=0)
    seed: NonNegativeInt
    discrete_price: NonNegativeFloat
    bridge_price: NonNegativeFloat
    discrete_standard_error: NonNegativeFloat
    bridge_standard_error: NonNegativeFloat
    discrete_hit_probability: Probability
    bridge_hit_probability: Probability
    analytical_continuous_price: NonNegativeFloat | None
    bridge_absolute_error: NonNegativeFloat | None
    runtime_ms: NonNegativeFloat


class BarrierConvergenceResult(BarrierModel):
    mode: BarrierConvergenceMode
    root_seed: NonNegativeInt
    points: list[BarrierConvergencePoint]


class BarrierLevelPoint(BarrierModel):
    barrier: PositiveFloat
    discrete_price: NonNegativeFloat
    bridge_price: NonNegativeFloat
    discrete_standard_error: NonNegativeFloat
    bridge_standard_error: NonNegativeFloat
    discrete_hit_probability: Probability
    bridge_hit_probability: Probability
    bridge_minus_discrete: float


class BarrierLevelSensitivityResult(BarrierModel):
    levels: list[BarrierLevelPoint]
    num_simulations: PositiveInt
    root_seed: NonNegativeInt
