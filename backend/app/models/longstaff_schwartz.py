"""Typed LSM configurations, frozen exercise policies, and compact research results."""

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.models.binomial import AmericanBinomialPricingResult
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import ExerciseStyle, OptionContract, OptionType
from backend.app.models.simulation import VarianceReduction

PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
Fraction = Annotated[float, Field(ge=0.0, le=1.0)]


class LSMModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class BasisFamily(StrEnum):
    POWER = "POWER"
    LAGUERRE = "LAGUERRE"


class LSMEvaluationMode(StrEnum):
    IN_SAMPLE = "IN_SAMPLE"
    OUT_OF_SAMPLE = "OUT_OF_SAMPLE"


class RegressionFallback(StrEnum):
    NO_ITM_PATHS = "NO_ITM_PATHS"
    INSUFFICIENT_PATHS = "INSUFFICIENT_PATHS"
    RANK_DEFICIENT = "RANK_DEFICIENT"
    SOLVER_FAILURE = "SOLVER_FAILURE"


class LSMConvergenceMode(StrEnum):
    PATH_COUNT = "PATH_COUNT"
    TIME_STEPS = "TIME_STEPS"


class LongstaffSchwartzConfig(LSMModel):
    """Explicit simulation, regression, stopping, and diagnostic choices."""

    num_paths: PositiveInt = 25_000
    num_time_steps: PositiveInt = 50
    seed: int = Field(default=42, ge=0)
    regression_paths: PositiveInt | None = None
    evaluation_mode: LSMEvaluationMode = LSMEvaluationMode.OUT_OF_SAMPLE
    variance_reduction: Literal[VarianceReduction.NONE, VarianceReduction.ANTITHETIC] = (
        VarianceReduction.ANTITHETIC
    )
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    basis_family: BasisFamily = BasisFamily.POWER
    basis_degree: int = Field(default=2, ge=0, le=5)
    min_regression_paths: PositiveInt = 20
    exercise_tolerance: float = Field(default=1e-12, ge=0.0, le=1e-4)
    exercise_steps: tuple[PositiveInt, ...] | None = None
    min_boundary_observations: PositiveInt = 5
    condition_warning_threshold: float = Field(default=1e8, gt=1.0)
    store_regression_diagnostics: bool = False
    store_boundary_diagnostics: bool = True
    max_diagnostic_dates: PositiveInt = 100

    @model_validator(mode="after")
    def validate_schedule_and_sample(self) -> Self:
        if self.exercise_steps is not None:
            if not self.exercise_steps or self.exercise_steps[-1] != self.num_time_steps:
                raise ValueError("exercise_steps must include maturity as its final index")
            if tuple(sorted(set(self.exercise_steps))) != self.exercise_steps:
                raise ValueError("exercise_steps must be sorted and unique")
        if (
            self.evaluation_mode is LSMEvaluationMode.IN_SAMPLE
            and self.regression_paths is not None
        ):
            raise ValueError("regression_paths is only used in OUT_OF_SAMPLE mode")
        return self

    @property
    def schedule(self) -> tuple[int, ...]:
        return self.exercise_steps or tuple(range(1, self.num_time_steps + 1))

    @property
    def training_path_count(self) -> int:
        return self.regression_paths or self.num_paths

    @property
    def total_simulated_paths(self) -> int:
        if self.evaluation_mode is LSMEvaluationMode.OUT_OF_SAMPLE:
            return self.training_path_count + self.num_paths
        return self.num_paths


class LSMRegressionDiagnostic(LSMModel):
    """One frozen date policy; a fallback means continue every candidate."""

    time_index: PositiveInt
    time: NonNegativeFloat
    basis_family: BasisFamily
    basis_degree: int = Field(ge=0, le=5)
    normalization: Literal["S_OVER_K"] = "S_OVER_K"
    itm_path_count: int = Field(ge=0)
    regression_path_count: int = Field(ge=0)
    basis_term_count: PositiveInt
    matrix_rank: int | None = Field(default=None, ge=0)
    condition_number: NonNegativeFloat | None = None
    condition_warning: bool = False
    residual_sum_squares: NonNegativeFloat | None = None
    fallback_used: bool
    fallback_reason: RegressionFallback | None = None
    coefficients: tuple[float, ...] | None = None


class LSMPolicy(LSMModel):
    """Reusable policy fitted only on training data, including its root decision."""

    contract: OptionContract
    market: MarketEnvironment
    num_time_steps: PositiveInt
    exercise_steps: tuple[int, ...]
    exercise_tolerance: NonNegativeFloat
    dates: tuple[LSMRegressionDiagnostic, ...]
    immediate_exercise: bool
    training_continuation_price: NonNegativeFloat
    no_early_exercise_theorem: bool = False


class LSMBoundaryPoint(LSMModel):
    time_index: int = Field(ge=0)
    time: NonNegativeFloat
    time_to_maturity: float = Field(gt=0.0)
    estimated_boundary_stock: float = Field(gt=0.0)
    option_type: OptionType
    exercise_region_direction: Literal["LOW_STOCK", "HIGH_STOCK"]
    exercise_observations: PositiveInt
    sample_support: Fraction


class LongstaffSchwartzResult(LSMModel):
    price: NonNegativeFloat
    standard_error: NonNegativeFloat
    estimator_variance: NonNegativeFloat
    confidence_interval_low: float
    confidence_interval_high: float
    confidence_level: float
    num_paths: PositiveInt
    regression_paths: PositiveInt
    num_time_steps: PositiveInt
    seed: int = Field(ge=0)
    training_seed: int | None = Field(default=None, ge=0)
    valuation_seed: int | None = Field(default=None, ge=0)
    option_type: OptionType
    exercise_style: Literal[ExerciseStyle.AMERICAN] = ExerciseStyle.AMERICAN
    variance_reduction: VarianceReduction
    evaluation_mode: LSMEvaluationMode
    basis_family: BasisFamily
    basis_degree: int
    runtime_ms: NonNegativeFloat
    simulation_runtime_ms: NonNegativeFloat
    regression_runtime_ms: NonNegativeFloat
    valuation_runtime_ms: NonNegativeFloat
    approximate_path_array_bytes: int = Field(ge=0)
    early_exercise_occurred: bool
    immediate_exercise: bool
    immediate_exercise_fraction: Fraction
    early_exercise_fraction: Fraction
    maturity_exercise_fraction: Fraction
    zero_cashflow_fraction: Fraction
    mean_exercise_time: NonNegativeFloat | None
    median_exercise_time: NonNegativeFloat | None
    regression_count: int = Field(ge=0)
    regression_fallback_count: int = Field(ge=0)
    condition_warning_count: int = Field(ge=0)
    max_condition_number: NonNegativeFloat | None
    training_continuation_price: NonNegativeFloat
    intrinsic_value: NonNegativeFloat
    below_intrinsic: bool
    deterministic: bool
    no_early_exercise_theorem: bool
    regression_diagnostics: tuple[LSMRegressionDiagnostic, ...] = ()
    boundary_diagnostics: tuple[LSMBoundaryPoint, ...] = ()
    regression_diagnostics_truncated: bool = False
    boundary_diagnostics_truncated: bool = False
    uncertainty_note: str


class LSMBoundaryComparisonPoint(LSMModel):
    """Descriptive boundary difference at an exactly shared nonterminal date."""

    time: NonNegativeFloat
    lsm_time_index: int = Field(ge=0)
    crr_step: int = Field(ge=0)
    lsm_boundary_stock: float = Field(gt=0.0)
    crr_boundary_stock: float = Field(gt=0.0)
    signed_difference: float
    absolute_difference: NonNegativeFloat
    exercise_observations: PositiveInt
    sample_support: Fraction


class LSMComparison(LSMModel):
    longstaff_schwartz: LongstaffSchwartzResult
    crr_american: AmericanBinomialPricingResult
    signed_error: float
    absolute_error: NonNegativeFloat
    relative_error: NonNegativeFloat | None
    normalized_error: float | None
    european_reference_price: NonNegativeFloat
    lsm_early_exercise_premium: float
    crr_early_exercise_premium: NonNegativeFloat
    european_reference_below_confidence_high: bool
    boundary_comparison: tuple[LSMBoundaryComparisonPoint, ...] | None = None


class LSMResearchPoint(LSMModel):
    num_paths: PositiveInt
    regression_paths: PositiveInt
    num_time_steps: PositiveInt
    seed: int
    basis_family: BasisFamily
    basis_degree: int
    price: NonNegativeFloat
    standard_error: NonNegativeFloat
    confidence_interval_low: float
    confidence_interval_high: float
    crr_price: NonNegativeFloat
    absolute_error: NonNegativeFloat
    runtime_ms: NonNegativeFloat
    early_exercise_fraction: Fraction
    regression_count: int
    fallback_count: int
    condition_warning_count: int
    max_condition_number: NonNegativeFloat | None


class LSMConvergenceResult(LSMModel):
    mode: LSMConvergenceMode
    root_seed: int
    crr_steps: PositiveInt
    points: list[LSMResearchPoint]


class LSMBasisComparisonResult(LSMModel):
    seed: int
    crr_steps: PositiveInt
    points: list[LSMResearchPoint]
