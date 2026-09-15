"""Serializable quantitative result contracts."""

from pydantic import BaseModel, ConfigDict, Field

from backend.app.models.simulation import VarianceReduction


class ResultModel(BaseModel):
    """Shared finite, immutable result-model policy."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class MonteCarloResult(ResultModel):
    """Monte Carlo estimate and diagnostics without raw path data."""

    price: float
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
    raw_price: float | None = Field(default=None, ge=0.0)
    raw_standard_error: float | None = Field(default=None, ge=0.0)
    adjusted_standard_error: float | None = Field(default=None, ge=0.0)
    control_variate_beta: float | None = None
    control_variate_mean: float | None = Field(default=None, gt=0.0)
    control_variate_expected_mean: float | None = Field(default=None, gt=0.0)
    raw_estimator_variance: float | None = Field(default=None, ge=0.0)
    adjusted_estimator_variance: float | None = Field(default=None, ge=0.0)
    variance_reduction_ratio: float | None = Field(default=None, ge=0.0)


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
    monte_carlo_price: float
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


class VarianceReductionComparisonPoint(ResultModel):
    """One method in a fixed-budget variance-reduction comparison."""

    method: VarianceReduction
    price: float
    black_scholes_price: float = Field(ge=0.0)
    absolute_error: float = Field(ge=0.0)
    standard_error: float = Field(ge=0.0)
    estimator_variance: float = Field(ge=0.0)
    raw_estimator_variance: float | None = Field(default=None, ge=0.0)
    adjusted_estimator_variance: float | None = Field(default=None, ge=0.0)
    variance_reduction_ratio: float | None = Field(default=None, ge=0.0)
    relative_variance: float | None = Field(default=None, ge=0.0)
    relative_efficiency: float | None = Field(default=None, ge=0.0)
    confidence_interval_width: float = Field(ge=0.0)
    runtime_ms: float = Field(ge=0.0)
    control_variate_beta: float | None = None


class VarianceReductionComparisonResult(ResultModel):
    """All supported methods evaluated at one fixed total path budget."""

    baseline_method: VarianceReduction
    black_scholes_price: float = Field(ge=0.0)
    num_simulations: int = Field(gt=0)
    root_seed: int = Field(ge=0)
    confidence_level: float = Field(gt=0.0, lt=1.0)
    points: list[VarianceReductionComparisonPoint]
