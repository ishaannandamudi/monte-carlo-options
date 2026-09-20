"""Typed Merton jump-diffusion parameters, controls, and research results."""

import math
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from backend.app.models.option import OptionType
from backend.app.models.simulation import ConfidenceLevel, NonNegativeInt, PositiveInt, RandomEngine
from backend.app.models.volatility import ImpliedVolatilityStatus

NonNegativeFloat = Annotated[float, Field(ge=0.0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]
Fraction = Annotated[float, Field(ge=0.0, le=1.0)]


class MertonModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class MertonSimulationScheme(StrEnum):
    COMPOUND_POISSON_EXACT = "COMPOUND_POISSON_EXACT"


class MertonAnalyticalMethod(StrEnum):
    POISSON_MIXTURE = "POISSON_MIXTURE"
    BLACK_SCHOLES_LIMIT = "BLACK_SCHOLES_LIMIT"
    INTRINSIC = "INTRINSIC"


class MertonParameterName(StrEnum):
    JUMP_INTENSITY = "jump_intensity"
    JUMP_MEAN = "jump_mean"
    JUMP_VOLATILITY = "jump_volatility"
    DIFFUSION_VOLATILITY = "diffusion_volatility"


class MertonJumpParameters(MertonModel):
    """Compound-Poisson parameters for log jump size ``Y ~ N(mu_j, sigma_j²)``."""

    jump_intensity: NonNegativeFloat
    jump_mean: float
    jump_volatility: NonNegativeFloat

    @property
    def jump_compensator(self) -> float:
        """Return ``E[exp(Y)-1]`` with an explicit finite-range check."""

        exponent = self.jump_mean + 0.5 * self.jump_volatility * self.jump_volatility
        try:
            value = math.expm1(exponent)
        except OverflowError as exc:
            raise ValueError("Merton jump compensator exceeds floating-point range") from exc
        if not math.isfinite(value):
            raise ValueError("Merton jump compensator must be finite")
        return value


class MertonSeriesConfig(MertonModel):
    tail_probability_tolerance: PositiveFloat = Field(default=1e-12, lt=1.0)
    max_terms: PositiveInt = 10_000


class MertonSimulationConfig(MertonModel):
    num_simulations: PositiveInt = 100_000
    seed: NonNegativeInt = 42
    confidence_level: ConfidenceLevel = 0.95
    random_engine: RandomEngine = RandomEngine.PCG64


class MertonPathSimulationConfig(MertonSimulationConfig):
    num_time_steps: PositiveInt = 100


class MertonAnalyticalResult(MertonModel):
    price: NonNegativeFloat
    terms_used: PositiveInt
    omitted_poisson_mass: Fraction
    runtime_ms: NonNegativeFloat
    converged: bool
    jump_compensator: float
    expected_jump_count: NonNegativeFloat
    method: MertonAnalyticalMethod


class MertonMonteCarloResult(MertonModel):
    price: NonNegativeFloat
    standard_error: NonNegativeFloat
    estimator_variance: NonNegativeFloat
    confidence_interval: tuple[float, float]
    confidence_interval_low: float
    confidence_interval_high: float
    confidence_level: ConfidenceLevel
    num_simulations: PositiveInt
    seed: NonNegativeInt
    runtime_ms: NonNegativeFloat
    simulation_runtime_ms: NonNegativeFloat
    mean_terminal_spot: PositiveFloat
    std_terminal_spot: NonNegativeFloat
    probability_itm: Fraction
    expected_jump_count: NonNegativeFloat
    mean_jump_count: NonNegativeFloat
    jump_count_variance: NonNegativeFloat
    zero_jump_fraction: Fraction
    at_least_one_jump_fraction: Fraction
    max_observed_jumps: NonNegativeInt
    scheme: MertonSimulationScheme


class MertonJumpDiagnosticsResult(MertonModel):
    expected_jump_count: NonNegativeFloat
    theoretical_jump_count_variance: NonNegativeFloat
    theoretical_zero_jump_probability: Fraction
    theoretical_at_least_one_jump_probability: Fraction
    mean_simulated_jump_count: NonNegativeFloat
    simulated_jump_count_variance: NonNegativeFloat
    zero_jump_fraction: Fraction
    at_least_one_jump_fraction: Fraction
    max_observed_jumps: NonNegativeInt
    num_simulations: PositiveInt
    seed: NonNegativeInt
    runtime_ms: NonNegativeFloat


class MertonComparisonResult(MertonModel):
    analytical: MertonAnalyticalResult
    monte_carlo: MertonMonteCarloResult
    black_scholes_price: NonNegativeFloat
    heston_price: NonNegativeFloat | None = None
    signed_error: float
    absolute_error: NonNegativeFloat
    relative_error: NonNegativeFloat | None
    z_score: float | None
    black_scholes_difference: float


class MertonConvergencePoint(MertonModel):
    num_simulations: PositiveInt
    seed: NonNegativeInt
    price: NonNegativeFloat
    standard_error: NonNegativeFloat
    analytical_price: NonNegativeFloat
    signed_error: float
    absolute_error: NonNegativeFloat
    runtime_ms: NonNegativeFloat


class MertonConvergenceResult(MertonModel):
    root_seed: NonNegativeInt
    analytical_price: NonNegativeFloat
    points: list[MertonConvergencePoint]
    standard_error_slope: float | None
    slope_points_used: NonNegativeInt


class MertonSeriesConvergencePoint(MertonModel):
    tolerance: PositiveFloat
    terms_used: PositiveInt
    price: NonNegativeFloat
    omitted_poisson_mass: Fraction
    converged: bool
    runtime_ms: NonNegativeFloat


class MertonSeriesConvergenceResult(MertonModel):
    points: list[MertonSeriesConvergencePoint]


class MertonSmilePoint(MertonModel):
    strike: PositiveFloat
    moneyness: PositiveFloat
    log_moneyness: float
    merton_price: NonNegativeFloat
    implied_volatility: NonNegativeFloat | None
    inversion_status: ImpliedVolatilityStatus


class MertonSmileResult(MertonModel):
    maturity_years: NonNegativeFloat
    points: list[MertonSmilePoint]


class MertonSensitivityPoint(MertonModel):
    parameter_value: float
    price: NonNegativeFloat
    implied_volatility: NonNegativeFloat | None
    inversion_status: ImpliedVolatilityStatus
    runtime_ms: NonNegativeFloat


class MertonSensitivityResult(MertonModel):
    parameter: MertonParameterName
    base_value: float
    points: list[MertonSensitivityPoint]


class MertonGreekBumps(MertonModel):
    relative_spot_bump: float = Field(default=1e-3, gt=0.0, lt=0.5)
    volatility_bump: PositiveFloat = 1e-4
    maturity_bump: PositiveFloat = 1e-4
    rate_bump: PositiveFloat = 1e-4
    jump_intensity_bump: PositiveFloat = 1e-4
    jump_mean_bump: PositiveFloat = 1e-4
    jump_volatility_bump: PositiveFloat = 1e-4


class MertonGreekResult(MertonModel):
    price: NonNegativeFloat
    delta: float
    gamma: float
    diffusion_vega: float
    theta: float
    rho: float
    jump_intensity_sensitivity: float
    jump_mean_sensitivity: float
    jump_vol_sensitivity: float


class MertonObjectiveObservation(MertonModel):
    strike: PositiveFloat
    maturity_years: NonNegativeFloat
    option_type: OptionType
    observed_price: NonNegativeFloat
    weight: NonNegativeFloat = 1.0


class MertonObjectiveResult(MertonModel):
    residuals: list[float]
    weighted_residuals: list[float]
    sum_squared_error: NonNegativeFloat
