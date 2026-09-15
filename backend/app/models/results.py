"""Serializable quantitative result contracts."""

from pydantic import BaseModel, ConfigDict, Field

from backend.app.models.simulation import VarianceReduction


class ResultModel(BaseModel):
    """Shared finite, immutable result-model policy."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class MonteCarloResult(ResultModel):
    """Monte Carlo estimate and diagnostics without raw path data."""

    price: float = Field(ge=0.0)
    standard_error: float = Field(ge=0.0)
    payoff_variance: float = Field(ge=0.0)
    discounted_payoff_std: float = Field(ge=0.0)
    confidence_interval_low: float
    confidence_interval_high: float
    confidence_level: float = Field(gt=0.0, lt=1.0)
    num_simulations: int = Field(gt=0)
    seed: int = Field(ge=0)
    variance_reduction: VarianceReduction
    runtime_ms: float = Field(ge=0.0)
    probability_itm: float = Field(ge=0.0, le=1.0)
    mean_terminal_price: float = Field(gt=0.0)
    terminal_price_std: float = Field(ge=0.0)


class BlackScholesResult(ResultModel):
    """Analytical benchmark result."""

    price: float = Field(ge=0.0)


class PricingComparison(ResultModel):
    """Monte Carlo estimate compared with its analytical benchmark."""

    monte_carlo: MonteCarloResult
    black_scholes_price: float = Field(ge=0.0)
    absolute_error: float = Field(ge=0.0)
    relative_error: float | None
    z_score: float | None
    black_scholes_inside_confidence_interval: bool


class ConvergencePoint(ResultModel):
    """One independently seeded convergence observation."""

    num_simulations: int = Field(gt=0)
    monte_carlo_price: float = Field(ge=0.0)
    black_scholes_price: float = Field(ge=0.0)
    absolute_error: float = Field(ge=0.0)
    relative_error: float | None
    standard_error: float = Field(ge=0.0)
    confidence_interval_width: float = Field(ge=0.0)
    runtime_ms: float = Field(ge=0.0)
    seed: int = Field(ge=0)


class ConvergenceResult(ResultModel):
    """Full reproducible convergence experiment."""

    black_scholes_price: float = Field(ge=0.0)
    points: list[ConvergencePoint]
    estimated_standard_error_slope: float | None
    slope_points_used: int = Field(ge=0)
    root_seed: int = Field(ge=0)
    variance_reduction: VarianceReduction
    confidence_level: float = Field(gt=0.0, lt=1.0)
