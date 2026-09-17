"""Typed Heston parameters, simulation controls, and research results."""

import math
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from backend.app.models.simulation import ConfidenceLevel, NonNegativeInt, PositiveInt, RandomEngine
from backend.app.models.volatility import ImpliedVolatilityStatus

NonNegativeFloat = Annotated[float, Field(ge=0.0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]
Fraction = Annotated[float, Field(ge=0.0, le=1.0)]


class HestonModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class HestonSimulationScheme(StrEnum):
    FULL_TRUNCATION_EULER = "FULL_TRUNCATION_EULER"
    QUADRATIC_EXPONENTIAL = "QUADRATIC_EXPONENTIAL"


class HestonAnalyticalMethod(StrEnum):
    LITTLE_TRAP_FOURIER = "LITTLE_TRAP_FOURIER"
    DETERMINISTIC_VARIANCE_LIMIT = "DETERMINISTIC_VARIANCE_LIMIT"
    INTRINSIC = "INTRINSIC"


class HestonConvergenceMode(StrEnum):
    PATH_COUNT = "PATH_COUNT"
    TIME_STEP = "TIME_STEP"


class HestonParameterName(StrEnum):
    V0 = "v0"
    KAPPA = "kappa"
    THETA = "theta"
    SIGMA_V = "sigma_v"
    RHO = "rho"


class HestonParameters(HestonModel):
    """Risk-neutral CIR variance-process parameters."""

    v0: NonNegativeFloat
    kappa: NonNegativeFloat
    theta: NonNegativeFloat
    sigma_v: NonNegativeFloat
    rho: float = Field(ge=-1.0, le=1.0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def feller_satisfied(self) -> bool:
        return 2.0 * self.kappa * self.theta >= self.sigma_v * self.sigma_v

    @computed_field  # type: ignore[prop-decorator]
    @property
    def feller_ratio(self) -> float | None:
        if self.sigma_v == 0.0:
            return None
        return 2.0 * self.kappa * self.theta / (self.sigma_v * self.sigma_v)


class HestonSimulationConfig(HestonModel):
    num_simulations: PositiveInt = 100_000
    num_time_steps: PositiveInt = 100
    seed: NonNegativeInt = 42
    scheme: HestonSimulationScheme = HestonSimulationScheme.FULL_TRUNCATION_EULER
    antithetic: bool = True
    random_engine: RandomEngine = RandomEngine.PCG64
    confidence_level: ConfidenceLevel = 0.95


class HestonIntegrationConfig(HestonModel):
    upper_bound: PositiveFloat = 400.0
    segment_width: PositiveFloat = 20.0
    absolute_tolerance: PositiveFloat = 1e-9
    relative_tolerance: PositiveFloat = 1e-8
    tail_tolerance: PositiveFloat = 1e-8
    quadrature_limit: PositiveInt = 200

    @model_validator(mode="after")
    def validate_segments(self) -> "HestonIntegrationConfig":
        if self.segment_width > self.upper_bound:
            raise ValueError("segment_width must not exceed upper_bound")
        return self


class HestonMonteCarloResult(HestonModel):
    price: NonNegativeFloat
    standard_error: NonNegativeFloat
    estimator_variance: NonNegativeFloat
    confidence_interval_low: float
    confidence_interval_high: float
    confidence_level: ConfidenceLevel
    num_simulations: PositiveInt
    num_time_steps: PositiveInt
    seed: NonNegativeInt
    scheme: HestonSimulationScheme
    antithetic: bool
    runtime_ms: NonNegativeFloat
    simulation_runtime_ms: NonNegativeFloat
    mean_terminal_spot: PositiveFloat
    std_terminal_spot: NonNegativeFloat
    mean_terminal_variance: NonNegativeFloat
    std_terminal_variance: NonNegativeFloat
    min_terminal_variance: NonNegativeFloat
    max_terminal_variance: NonNegativeFloat
    negative_variance_proposal_fraction: Fraction | None
    probability_itm: Fraction
    feller_satisfied: bool
    feller_ratio: NonNegativeFloat | None


class HestonAnalyticalResult(HestonModel):
    price: NonNegativeFloat
    integration_error: NonNegativeFloat
    integration_evaluations: int = Field(ge=0)
    integration_upper_bound: NonNegativeFloat
    runtime_ms: NonNegativeFloat
    method: HestonAnalyticalMethod
    feller_satisfied: bool
    feller_ratio: NonNegativeFloat | None


class HestonComparisonPoint(HestonModel):
    scheme: HestonSimulationScheme
    price: NonNegativeFloat
    standard_error: NonNegativeFloat
    analytical_price: NonNegativeFloat
    absolute_error: NonNegativeFloat
    relative_error: NonNegativeFloat | None
    z_score: float | None
    runtime_ms: NonNegativeFloat
    mean_terminal_variance: NonNegativeFloat
    negative_variance_proposal_fraction: Fraction | None


class HestonComparisonResult(HestonModel):
    analytical: HestonAnalyticalResult
    points: list[HestonComparisonPoint]
    feller_satisfied: bool
    feller_ratio: NonNegativeFloat | None


class HestonConvergencePoint(HestonModel):
    num_simulations: PositiveInt
    num_time_steps: PositiveInt
    seed: NonNegativeInt
    scheme: HestonSimulationScheme
    price: NonNegativeFloat
    standard_error: NonNegativeFloat
    analytical_price: NonNegativeFloat
    bias: float
    absolute_error: NonNegativeFloat
    runtime_ms: NonNegativeFloat


class HestonConvergenceResult(HestonModel):
    mode: HestonConvergenceMode
    root_seed: NonNegativeInt
    analytical_price: NonNegativeFloat
    points: list[HestonConvergencePoint]


class HestonVarianceTimePoint(HestonModel):
    time_years: NonNegativeFloat
    simulated_mean_variance: float
    theoretical_mean_variance: NonNegativeFloat
    fraction_near_zero: Fraction


class HestonVarianceDiagnosticsResult(HestonModel):
    scheme: HestonSimulationScheme
    num_simulations: PositiveInt
    num_time_steps: PositiveInt
    seed: NonNegativeInt
    points: list[HestonVarianceTimePoint]
    terminal_variance_mean: float
    terminal_variance_std: NonNegativeFloat
    terminal_variance_min: float
    terminal_variance_max: float
    terminal_variance_quantiles: dict[str, float]
    negative_variance_proposal_fraction: Fraction | None
    feller_satisfied: bool
    feller_ratio: NonNegativeFloat | None
    runtime_ms: NonNegativeFloat


class HestonSmilePoint(HestonModel):
    strike: PositiveFloat
    moneyness: PositiveFloat
    log_moneyness: float
    heston_price: NonNegativeFloat
    implied_volatility: NonNegativeFloat | None
    inversion_status: ImpliedVolatilityStatus


class HestonSmileResult(HestonModel):
    maturity_years: NonNegativeFloat
    points: list[HestonSmilePoint]
    feller_satisfied: bool
    feller_ratio: NonNegativeFloat | None


class HestonSensitivityPoint(HestonModel):
    parameter_value: float
    price: NonNegativeFloat
    implied_volatility: NonNegativeFloat | None
    inversion_status: ImpliedVolatilityStatus
    runtime_ms: NonNegativeFloat


class HestonSensitivityResult(HestonModel):
    parameter: HestonParameterName
    base_value: NonNegativeFloat | float
    points: list[HestonSensitivityPoint]


class HestonDriverCorrelationResult(HestonModel):
    target_correlation: float = Field(ge=-1.0, le=1.0)
    empirical_correlation: float = Field(ge=-1.0, le=1.0)
    absolute_error: NonNegativeFloat
    num_simulations: PositiveInt
    seed: NonNegativeInt


def require_finite_parameter_value(parameter: HestonParameterName, value: float) -> None:
    """Validate a sensitivity value before constructing a copied parameter model."""

    if not math.isfinite(value):
        raise ValueError(f"{parameter.value} sensitivity values must be finite")
