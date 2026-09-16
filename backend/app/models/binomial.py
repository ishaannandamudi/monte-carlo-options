"""Typed Cox-Ross-Rubinstein tree configuration and result contracts."""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.app.models.option import ExerciseStyle, OptionType

PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]


class BinomialModel(BaseModel):
    """Immutable finite-data policy for binomial-tree objects."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class TreeModel(StrEnum):
    """Supported lattice parameterizations."""

    CRR = "CRR"


class StepParity(StrEnum):
    """Parity label used to make CRR oscillation diagnostics explicit."""

    ODD = "ODD"
    EVEN = "EVEN"


class ExerciseRegionDirection(StrEnum):
    """Stock-price side represented by an early-exercise boundary."""

    LOW_STOCK = "LOW_STOCK"
    HIGH_STOCK = "HIGH_STOCK"


class DeterministicBoundaryCase(StrEnum):
    """Tree-degenerate cases handled without constructing a lattice."""

    ZERO_MATURITY = "ZERO_MATURITY"
    ZERO_VOLATILITY = "ZERO_VOLATILITY"


class BinomialTreeConfig(BinomialModel):
    """Explicit CRR resolution and probability-roundoff policy."""

    steps: PositiveInt = 500
    model: TreeModel = TreeModel.CRR
    probability_tolerance: float = Field(default=1e-12, ge=0.0, le=1e-6)
    arbitrage_tolerance: float = Field(default=1e-10, ge=0.0, le=1e-4)
    exercise_tolerance: float = Field(default=1e-12, ge=0.0, le=1e-4)


class CRRParameters(BinomialModel):
    """One validated CRR time-step parameterization."""

    steps: PositiveInt
    dt: float = Field(gt=0.0)
    up_factor: float = Field(gt=0.0)
    down_factor: float = Field(gt=0.0)
    risk_neutral_growth: float = Field(gt=0.0)
    risk_neutral_probability: float = Field(ge=0.0, le=1.0)
    discount_factor: float = Field(gt=0.0)


class BinomialPricingResult(BinomialModel):
    """CRR price, native Greeks, parameters, and analytical diagnostics."""

    price: NonNegativeFloat
    steps: PositiveInt
    model: TreeModel
    dt: float | None = Field(default=None, gt=0.0)
    up_factor: float | None = Field(default=None, gt=0.0)
    down_factor: float | None = Field(default=None, gt=0.0)
    risk_neutral_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    discount_factor: float | None = Field(default=None, gt=0.0)
    runtime_ms: NonNegativeFloat
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    black_scholes_price: NonNegativeFloat
    signed_error: float
    absolute_error: NonNegativeFloat
    relative_error: float | None
    price_lower_bound: NonNegativeFloat
    price_upper_bound: NonNegativeFloat
    deterministic_boundary_case: DeterministicBoundaryCase | None = None


class ExerciseBoundaryPoint(BinomialModel):
    """One observed early-exercise frontier on a discrete tree level."""

    step: int = Field(ge=0)
    time_from_valuation: NonNegativeFloat
    time_to_maturity: float = Field(gt=0.0)
    boundary_stock_price: float = Field(gt=0.0)
    option_type: OptionType
    exercise_region_direction: ExerciseRegionDirection
    number_of_exercise_nodes: PositiveInt
    region_contiguous: bool


class AmericanBinomialPricingResult(BinomialModel):
    """American CRR value with same-tree European and exercise diagnostics."""

    price: NonNegativeFloat
    exercise_style: Literal[ExerciseStyle.AMERICAN]
    steps: PositiveInt
    model: TreeModel
    dt: float | None = Field(default=None, gt=0.0)
    up_factor: float | None = Field(default=None, gt=0.0)
    down_factor: float | None = Field(default=None, gt=0.0)
    risk_neutral_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    discount_factor: float | None = Field(default=None, gt=0.0)
    runtime_ms: NonNegativeFloat
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    european_tree_price: NonNegativeFloat
    black_scholes_european_price: NonNegativeFloat
    early_exercise_premium: NonNegativeFloat
    early_exercise_occurred: bool
    earliest_exercise_step: int | None = Field(default=None, ge=0)
    earliest_exercise_time: NonNegativeFloat | None = None
    earliest_boundary_price: float | None = Field(default=None, gt=0.0)
    exercise_node_count: int = Field(ge=0)
    boundary_available: bool
    non_contiguous_level_count: int = Field(ge=0)
    deterministic_boundary_case: DeterministicBoundaryCase | None = None


class ExerciseBoundaryResult(BinomialModel):
    """Ordered O(N)-sized early-exercise frontier and summary diagnostics."""

    exercise_style: Literal[ExerciseStyle.AMERICAN]
    option_type: OptionType
    steps: PositiveInt
    model: TreeModel
    points: list[ExerciseBoundaryPoint]
    early_exercise_occurred: bool
    earliest_exercise_step: int | None = Field(default=None, ge=0)
    earliest_exercise_time: NonNegativeFloat | None = None
    earliest_boundary_price: float | None = Field(default=None, gt=0.0)
    exercise_node_count: int = Field(ge=0)
    non_contiguous_level_count: int = Field(ge=0)
    runtime_ms: NonNegativeFloat


class TreeGreekComparison(BinomialModel):
    """One tree-native Greek measured against its analytical value."""

    tree_value: float | None
    analytical_value: float
    signed_error: float | None
    absolute_error: NonNegativeFloat | None
    relative_error: float | None


class BinomialComparison(BinomialModel):
    """One CRR result and Black-Scholes price/Greek benchmarks."""

    binomial: BinomialPricingResult
    black_scholes_price: NonNegativeFloat
    signed_error: float
    absolute_error: NonNegativeFloat
    relative_error: float | None
    delta: TreeGreekComparison
    gamma: TreeGreekComparison
    theta: TreeGreekComparison


class AmericanTreeGreekDifference(BinomialModel):
    """American minus same-tree European Greek diagnostic."""

    american_value: float | None
    european_value: float | None
    difference: float | None


class AmericanBinomialComparison(BinomialModel):
    """American CRR result compared with same-tree and analytical European values."""

    american: AmericanBinomialPricingResult
    european: BinomialPricingResult
    black_scholes_european_price: NonNegativeFloat
    early_exercise_premium: NonNegativeFloat
    delta: AmericanTreeGreekDifference
    gamma: AmericanTreeGreekDifference
    theta: AmericanTreeGreekDifference


class BinomialConvergencePoint(BinomialModel):
    """One independently evaluated CRR step count."""

    steps: PositiveInt
    step_parity: StepParity
    tree_price: NonNegativeFloat
    black_scholes_price: NonNegativeFloat
    signed_error: float
    absolute_error: NonNegativeFloat
    relative_error: float | None
    runtime_ms: NonNegativeFloat
    delta: float | None
    delta_error: float | None
    gamma: float | None
    gamma_error: float | None
    theta: float | None
    theta_error: float | None


class OddEvenDiagnostic(BinomialModel):
    """Neighboring step estimates retained without altering canonical prices."""

    lower_steps: PositiveInt
    upper_steps: PositiveInt
    lower_parity: StepParity
    upper_parity: StepParity
    price_difference: float
    adjacent_average: NonNegativeFloat
    average_absolute_error: NonNegativeFloat


class BinomialConvergenceResult(BinomialModel):
    """Ordered CRR convergence observations and adjacent-parity diagnostics."""

    model: TreeModel
    black_scholes_price: NonNegativeFloat
    analytical_delta: float
    analytical_gamma: NonNegativeFloat
    analytical_theta: float
    points: list[BinomialConvergencePoint]
    odd_even_diagnostics: list[OddEvenDiagnostic]


class AmericanBinomialConvergencePoint(BinomialModel):
    """One American CRR resolution with same-tree European diagnostics."""

    steps: PositiveInt
    step_parity: StepParity
    american_price: NonNegativeFloat
    european_tree_price: NonNegativeFloat
    early_exercise_premium: NonNegativeFloat
    delta: float | None
    gamma: float | None
    theta: float | None
    runtime_ms: NonNegativeFloat
    exercise_node_count: int = Field(ge=0)
    earliest_exercise_time: NonNegativeFloat | None = None


class AmericanBinomialConvergenceResult(BinomialModel):
    """American CRR stabilization diagnostics without a claimed closed form."""

    model: TreeModel
    black_scholes_european_price: NonNegativeFloat
    points: list[AmericanBinomialConvergencePoint]
