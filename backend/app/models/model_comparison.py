"""Typed contracts for cross-model pricing, fit, stability, and model-risk research."""

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.models.binomial import BinomialTreeConfig
from backend.app.models.heston import (
    HestonIntegrationConfig,
    HestonParameters,
    HestonSimulationConfig,
)
from backend.app.models.heston_calibration import (
    HestonCalibrationConfig,
    HestonCalibrationDataset,
    HestonCalibrationHoldoutConfig,
    HestonCalibrationResult,
    HestonCalibrationStabilityResult,
)
from backend.app.models.longstaff_schwartz import LongstaffSchwartzConfig
from backend.app.models.market import MarketEnvironment
from backend.app.models.merton import (
    MertonJumpParameters,
    MertonSeriesConfig,
    MertonSimulationConfig,
)
from backend.app.models.option import OptionContract
from backend.app.models.simulation import SimulationConfig
from backend.app.models.volatility_surface import VolatilitySurface

PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]


class ComparisonModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class PricingModel(StrEnum):
    BLACK_SCHOLES = "BLACK_SCHOLES"
    MONTE_CARLO_GBM = "MONTE_CARLO_GBM"
    CRR_BINOMIAL = "CRR_BINOMIAL"
    HESTON_ANALYTICAL = "HESTON_ANALYTICAL"
    HESTON_MONTE_CARLO = "HESTON_MONTE_CARLO"
    CRR_AMERICAN = "CRR_AMERICAN"
    LONGSTAFF_SCHWARTZ = "LONGSTAFF_SCHWARTZ"
    MERTON_ANALYTICAL = "MERTON_ANALYTICAL"
    MERTON_MONTE_CARLO = "MERTON_MONTE_CARLO"


class ModelCostClass(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"


class ModelRunStatus(StrEnum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    EXCLUDED = "EXCLUDED"


class ComparisonMetric(StrEnum):
    PRICE = "PRICE"
    GREEKS = "GREEKS"
    IMPLIED_VOLATILITY = "IMPLIED_VOLATILITY"
    RUNTIME = "RUNTIME"


class PriceRangeNormalization(StrEnum):
    MEDIAN_PRICE = "MEDIAN_PRICE"
    SPOT = "SPOT"
    REFERENCE_PRICE = "REFERENCE_PRICE"


class CommonRandomNumberPolicy(StrEnum):
    INDEPENDENT_BY_MODEL = "INDEPENDENT_BY_MODEL"
    RELATED_GBM_ONLY = "RELATED_GBM_ONLY"


class ModelRiskFlag(StrEnum):
    LARGE_PRICE_DISPERSION = "LARGE_PRICE_DISPERSION"
    LARGE_GREEK_DISPERSION = "LARGE_GREEK_DISPERSION"
    LARGE_IV_DISPERSION = "LARGE_IV_DISPERSION"
    CALIBRATION_UNSTABLE = "CALIBRATION_UNSTABLE"
    MODEL_OUTSIDE_BID_ASK = "MODEL_OUTSIDE_BID_ASK"
    HIGH_MONTE_CARLO_UNCERTAINTY = "HIGH_MONTE_CARLO_UNCERTAINTY"
    POOR_OUT_OF_SAMPLE_FIT = "POOR_OUT_OF_SAMPLE_FIT"
    NUMERICAL_CONSISTENCY_WARNING = "NUMERICAL_CONSISTENCY_WARNING"


class ComputeBudgetType(StrEnum):
    PATHS = "PATHS"
    TREE_STEPS = "TREE_STEPS"
    TIME_STEPS = "TIME_STEPS"
    PATH_TIME_NODES = "PATH_TIME_NODES"


class StabilityInput(StrEnum):
    SPOT = "SPOT"
    VOLATILITY_OR_V0 = "VOLATILITY_OR_V0"
    RISK_FREE_RATE = "RISK_FREE_RATE"
    DIVIDEND_YIELD = "DIVIDEND_YIELD"
    MATURITY = "MATURITY"


class ModelCapabilities(ComparisonModel):
    model: PricingModel
    supports_european: bool
    supports_american: bool
    supports_call: bool = True
    supports_put: bool = True
    supports_dividends: bool = True
    supports_stochastic_volatility: bool
    supports_jump_diffusion: bool = False
    supports_closed_form_or_semi_analytical: bool
    supports_monte_carlo: bool
    supports_greeks: bool
    requires_calibration: bool
    deterministic: bool
    typical_cost: ModelCostClass
    state_variables: PositiveInt
    stochastic_factors: int = Field(ge=0)
    calibrated_parameter_count: int = Field(ge=0)
    numerical_integration_required: bool
    simulation_required: bool
    early_exercise_capability: bool


class DispersionThresholds(ComparisonModel):
    max_relative_price_dispersion: NonNegativeFloat = 0.10
    max_delta_dispersion: NonNegativeFloat = 0.15
    max_iv_dispersion: NonNegativeFloat = 0.05
    max_relative_monte_carlo_se: NonNegativeFloat = 0.05
    max_calibration_parameter_cv: NonNegativeFloat = 0.50
    max_out_of_sample_rmse: NonNegativeFloat = 1.0
    numerical_consistency_tolerance: NonNegativeFloat = 0.02


class HestonGreekBumps(ComparisonModel):
    relative_spot_bump: float = Field(default=1e-3, gt=0.0, lt=0.5)
    rate_bump: float = Field(default=1e-4, gt=0.0)
    maturity_bump: float = Field(default=1e-4, gt=0.0)


class ModelComparisonConfig(ComparisonModel):
    selected_models: list[PricingModel] = Field(
        default_factory=lambda: [
            PricingModel.BLACK_SCHOLES,
            PricingModel.MONTE_CARLO_GBM,
            PricingModel.CRR_BINOMIAL,
        ],
        min_length=1,
    )
    reference_model: PricingModel | None = PricingModel.BLACK_SCHOLES
    metrics: list[ComparisonMetric] = Field(
        default_factory=lambda: [ComparisonMetric.PRICE, ComparisonMetric.RUNTIME]
    )
    include_greeks: bool = False
    include_iv: bool = True
    include_pairwise: bool = True
    include_market_fit: bool = False
    normalization: PriceRangeNormalization = PriceRangeNormalization.MEDIAN_PRICE
    common_random_numbers: CommonRandomNumberPolicy = CommonRandomNumberPolicy.RELATED_GBM_ONLY
    thresholds: DispersionThresholds = Field(default_factory=DispersionThresholds)
    heston_greek_bumps: HestonGreekBumps = Field(default_factory=HestonGreekBumps)
    runtime_repetitions: PositiveInt = 1
    deterministic_seed: int = Field(default=42, ge=0)
    max_result_points: PositiveInt = 500
    relative_price_floor: float = Field(default=1e-8, gt=0.0)

    @model_validator(mode="after")
    def validate_models(self) -> Self:
        if len(set(self.selected_models)) != len(self.selected_models):
            raise ValueError("selected_models must not contain duplicates")
        return self


class ModelComparisonRequest(ComparisonModel):
    contract: OptionContract
    market: MarketEnvironment
    config: ModelComparisonConfig = Field(default_factory=ModelComparisonConfig)
    heston_parameters: HestonParameters | None = None
    heston_calibration_result: HestonCalibrationResult | None = None
    gbm_monte_carlo: SimulationConfig = Field(default_factory=SimulationConfig)
    tree: BinomialTreeConfig = Field(default_factory=BinomialTreeConfig)
    heston_monte_carlo: HestonSimulationConfig = Field(default_factory=HestonSimulationConfig)
    heston_integration: HestonIntegrationConfig = Field(default_factory=HestonIntegrationConfig)
    longstaff_schwartz: LongstaffSchwartzConfig = Field(default_factory=LongstaffSchwartzConfig)
    merton_parameters: MertonJumpParameters | None = None
    merton_monte_carlo: MertonSimulationConfig = Field(default_factory=MertonSimulationConfig)
    merton_series: MertonSeriesConfig = Field(default_factory=MertonSeriesConfig)


class ModelExclusion(ComparisonModel):
    model: PricingModel
    status: ModelRunStatus = ModelRunStatus.EXCLUDED
    reason: str


class ModelCalibrationMetadata(ComparisonModel):
    calibration_required: bool
    calibration_runtime_ms: NonNegativeFloat | None = None
    objective_evaluations: int | None = Field(default=None, ge=0)
    observations: int | None = Field(default=None, ge=0)
    prices_per_second: NonNegativeFloat | None = None
    fitted_rmse: NonNegativeFloat | None = None
    source: str | None = None


class ModelGreekValues(ComparisonModel):
    delta: float | None = None
    gamma: float | None = None
    vega: float | None = None
    theta: float | None = None
    rho: float | None = None
    heston_vega_interpretation: str | None = None


class ModelOutput(ComparisonModel):
    model: PricingModel
    status: ModelRunStatus
    price: NonNegativeFloat | None = None
    runtime_ms: NonNegativeFloat | None = None
    deterministic: bool
    standard_error: NonNegativeFloat | None = None
    confidence_interval_low: float | None = None
    confidence_interval_high: float | None = None
    greeks: ModelGreekValues = Field(default_factory=ModelGreekValues)
    implied_volatility: NonNegativeFloat | None = None
    implied_volatility_residual: float | None = None
    calibration: ModelCalibrationMetadata | None = None
    paths: int | None = Field(default=None, gt=0)
    time_steps: int | None = Field(default=None, gt=0)
    tree_steps: int | None = Field(default=None, gt=0)
    early_exercise_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    error_type: str | None = None
    message: str | None = None
    warnings: list[str] = Field(default_factory=list)


class ScalarDispersion(ComparisonModel):
    count: int = Field(ge=0)
    minimum: float | None = None
    maximum: float | None = None
    range: NonNegativeFloat | None = None
    mean: float | None = None
    median: float | None = None
    standard_deviation: NonNegativeFloat | None = None


class PriceDispersion(ScalarDispersion):
    relative_range: NonNegativeFloat | None = None
    normalization: PriceRangeNormalization
    normalization_value: NonNegativeFloat | None = None
    interpretation: str = "Descriptive variation across model prices; not a confidence interval."


class PairwiseMatrix(ComparisonModel):
    models: list[PricingModel]
    signed_differences: list[list[float]]
    relative_differences: list[list[float]]
    relative_denominator: str


class ReferenceDeviation(ComparisonModel):
    model: PricingModel
    reference_model: PricingModel
    signed_difference: float
    absolute_difference: NonNegativeFloat
    relative_difference: float


class GreekDispersion(ComparisonModel):
    greek: str
    dispersion: ScalarDispersion
    models: list[PricingModel]
    pairwise_differences: list[list[float]]


class CrossModelDiagnostic(ComparisonModel):
    name: str
    left_model: PricingModel
    right_model: PricingModel
    signed_difference: float
    absolute_difference: NonNegativeFloat
    z_score: float | None = None
    runtime_ratio: NonNegativeFloat | None = None


class ExperimentRecord(ComparisonModel):
    experiment_identifier: str
    timestamp: datetime
    input_signature: str
    model_configs: dict[str, Any]
    total_runtime_ms: NonNegativeFloat


class ModelComparisonResult(ComparisonModel):
    contract: OptionContract
    market: MarketEnvironment
    outputs: list[ModelOutput]
    exclusions: list[ModelExclusion]
    price_dispersion: PriceDispersion
    pairwise: PairwiseMatrix | None
    implied_volatility_pairwise: PairwiseMatrix | None
    reference_deviations: list[ReferenceDeviation]
    greek_dispersion: list[GreekDispersion]
    consistency_diagnostics: list[CrossModelDiagnostic]
    flags: list[ModelRiskFlag]
    successful_models: list[PricingModel]
    failed_models: list[PricingModel]
    excluded_models: list[PricingModel]
    experiment: ExperimentRecord


class FitMetrics(ComparisonModel):
    observation_count: int = Field(ge=0)
    rmse: NonNegativeFloat | None = None
    mae: NonNegativeFloat | None = None
    weighted_rmse: NonNegativeFloat | None = None
    normalized_rmse: NonNegativeFloat | None = None
    normalized_mae: NonNegativeFloat | None = None
    max_absolute_error: NonNegativeFloat | None = None
    median_absolute_error: NonNegativeFloat | None = None
    bias: float | None = None
    bid_ask_observation_count: int = Field(ge=0)
    fraction_inside_bid_ask: float | None = Field(default=None, ge=0.0, le=1.0)
    average_distance_outside_spread: NonNegativeFloat | None = None
    maximum_distance_outside_spread: NonNegativeFloat | None = None
    iv_observation_count: int = Field(ge=0)
    iv_rmse: NonNegativeFloat | None = None
    iv_mae: NonNegativeFloat | None = None
    max_iv_error: NonNegativeFloat | None = None
    iv_bias: float | None = None


class MarketFitPoint(ComparisonModel):
    model: PricingModel
    observation_index: int = Field(ge=0)
    strike: float = Field(gt=0.0)
    maturity_years: float = Field(gt=0.0)
    moneyness: float = Field(gt=0.0)
    log_moneyness: float
    option_type: str
    market_price: NonNegativeFloat
    model_price: NonNegativeFloat | None
    price_residual: float | None
    normalized_price_residual: float | None
    market_implied_volatility: NonNegativeFloat | None
    model_implied_volatility: NonNegativeFloat | None
    implied_volatility_residual: float | None
    bid: NonNegativeFloat | None
    ask: NonNegativeFloat | None
    spread_distance: NonNegativeFloat | None
    weight: NonNegativeFloat
    status: ModelRunStatus
    message: str | None = None


class GroupFitMetrics(ComparisonModel):
    label: str
    maturity_years: float | None = Field(default=None, gt=0.0)
    metrics: FitMetrics


class SmileFitMetrics(ComparisonModel):
    maturity_years: float = Field(gt=0.0)
    market_atm_iv: NonNegativeFloat | None
    model_atm_iv: NonNegativeFloat | None
    atm_iv_residual: float | None
    market_skew: float | None
    model_skew: float | None
    skew_residual: float | None
    market_curvature: float | None
    model_curvature: float | None
    curvature_residual: float | None


class ModelMarketFit(ComparisonModel):
    model: PricingModel
    status: ModelRunStatus
    metrics: FitMetrics
    per_expiry: list[GroupFitMetrics]
    per_moneyness: list[GroupFitMetrics]
    per_metadata_group: list[GroupFitMetrics]
    smile_fit: list[SmileFitMetrics]
    residuals: list[MarketFitPoint]
    calibration: ModelCalibrationMetadata | None = None
    runtime_ms: NonNegativeFloat
    message: str | None = None


class MarketFitResult(ComparisonModel):
    models: list[ModelMarketFit]
    observations_used: int = Field(ge=0)
    observations_excluded: int = Field(ge=0)
    exclusion_reasons: dict[str, int]
    input_signature: str
    warnings: list[str]


class MarketFitRequest(ComparisonModel):
    dataset: HestonCalibrationDataset
    selected_models: list[PricingModel] = Field(min_length=1)
    heston_parameters: HestonParameters | None = None
    heston_calibration_result: HestonCalibrationResult | None = None
    merton_parameters: MertonJumpParameters | None = None
    black_scholes_volatility: float = Field(default=0.20, ge=0.0)
    use_observed_iv_for_black_scholes: bool = False
    preprocessing: HestonCalibrationConfig = Field(default_factory=HestonCalibrationConfig)
    tree: BinomialTreeConfig = Field(default_factory=BinomialTreeConfig)
    integration: HestonIntegrationConfig = Field(default_factory=HestonIntegrationConfig)
    gbm_monte_carlo: SimulationConfig = Field(
        default_factory=lambda: SimulationConfig(num_simulations=10_000)
    )
    heston_monte_carlo: HestonSimulationConfig = Field(
        default_factory=lambda: HestonSimulationConfig(num_simulations=5_000, num_time_steps=25)
    )
    merton_monte_carlo: MertonSimulationConfig = Field(
        default_factory=lambda: MertonSimulationConfig(num_simulations=10_000)
    )
    merton_series: MertonSeriesConfig = Field(default_factory=MertonSeriesConfig)
    max_returned_residuals: PositiveInt = 500
    relative_price_floor: float = Field(default=1e-8, gt=0.0)
    observation_labels: dict[int, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_models(self) -> Self:
        if len(set(self.selected_models)) != len(self.selected_models):
            raise ValueError("selected_models must not contain duplicates")
        if any(index < 0 for index in self.observation_labels):
            raise ValueError("observation-label indices must be non-negative")
        if any(index >= len(self.dataset.observations) for index in self.observation_labels):
            raise ValueError("observation-label index is outside the dataset")
        if any(not label.strip() for label in self.observation_labels.values()):
            raise ValueError("observation labels must be nonempty")
        return self


class OutOfSampleModelFit(ComparisonModel):
    model: PricingModel
    training_metrics: FitMetrics
    validation_metrics: FitMetrics


class OutOfSampleComparisonResult(ComparisonModel):
    holdout: HestonCalibrationHoldoutConfig
    training_observations: PositiveInt
    validation_observations: PositiveInt
    models: list[OutOfSampleModelFit]


class ModelComplexityFit(ComparisonModel):
    model: PricingModel
    state_variables: PositiveInt
    stochastic_factors: int = Field(ge=0)
    calibrated_parameter_count: int = Field(ge=0)
    calibration_required: bool
    calibration_runtime_ms: NonNegativeFloat | None
    pricing_runtime_ms: NonNegativeFloat | None
    fit_rmse: NonNegativeFloat | None
    out_of_sample_rmse: NonNegativeFloat | None


class ComputeFrontierRequest(ComparisonModel):
    comparison: ModelComparisonRequest
    model: PricingModel
    budgets: list[PositiveInt] = Field(min_length=1)
    reference_model: PricingModel
    budget_type: ComputeBudgetType | None = None

    @model_validator(mode="after")
    def normalize_budgets(self) -> Self:
        if len(set(self.budgets)) != len(self.budgets):
            raise ValueError("compute-frontier budgets must be unique")
        object.__setattr__(self, "budgets", sorted(self.budgets))
        allowed = {
            PricingModel.MONTE_CARLO_GBM: {ComputeBudgetType.PATHS},
            PricingModel.CRR_BINOMIAL: {ComputeBudgetType.TREE_STEPS},
            PricingModel.CRR_AMERICAN: {ComputeBudgetType.TREE_STEPS},
            PricingModel.HESTON_MONTE_CARLO: {
                ComputeBudgetType.PATHS,
                ComputeBudgetType.TIME_STEPS,
            },
            PricingModel.LONGSTAFF_SCHWARTZ: {
                ComputeBudgetType.PATHS,
                ComputeBudgetType.TIME_STEPS,
            },
            PricingModel.MERTON_MONTE_CARLO: {ComputeBudgetType.PATHS},
        }
        if self.model not in allowed:
            raise ValueError("selected model has no compute-frontier budget")
        if self.budget_type is not None and self.budget_type not in allowed[self.model]:
            raise ValueError("budget_type is incompatible with the selected model")
        return self


class ComputeFrontierPoint(ComparisonModel):
    budget: PositiveInt
    budget_type: ComputeBudgetType
    price: NonNegativeFloat
    reference_price: NonNegativeFloat
    absolute_error: NonNegativeFloat
    relative_error: NonNegativeFloat
    standard_error: NonNegativeFloat | None
    confidence_interval_width: NonNegativeFloat | None
    runtime_ms: NonNegativeFloat
    approximate_memory_bytes: int | None = Field(default=None, ge=0)
    seed: int | None = Field(default=None, ge=0)


class ComputeFrontierResult(ComparisonModel):
    model: PricingModel
    reference_model: PricingModel
    natural_convergence_variable: ComputeBudgetType
    points: list[ComputeFrontierPoint]
    input_signature: str


class StabilityPerturbation(ComparisonModel):
    input: StabilityInput
    relative_bump: float = Field(default=1e-3, gt=0.0, lt=0.5)
    absolute_bump: float | None = Field(default=None, gt=0.0)


class StabilityPoint(ComparisonModel):
    model: PricingModel
    input: StabilityInput
    base_price: NonNegativeFloat
    downward_price: NonNegativeFloat | None
    upward_price: NonNegativeFloat
    downward_change: float | None
    upward_change: float
    central_sensitivity: float | None
    status: ModelRunStatus
    message: str | None = None


class ModelStabilityResult(ComparisonModel):
    points: list[StabilityPoint]
    input_signature: str


class CalibrationEnsembleMember(ComparisonModel):
    index: int = Field(ge=0)
    parameters: HestonParameters
    price: NonNegativeFloat
    implied_volatility: NonNegativeFloat | None
    provenance: str


class CalibrationEnsembleResult(ComparisonModel):
    members: list[CalibrationEnsembleMember]
    price_dispersion: ScalarDispersion
    implied_volatility_dispersion: ScalarDispersion


class CalibrationStabilityValuation(ComparisonModel):
    parameter_standard_deviations: dict[str, NonNegativeFloat]
    price_dispersion: ScalarDispersion
    implied_volatility_dispersion: ScalarDispersion
    ensemble: CalibrationEnsembleResult


class ParameterUncertaintyResult(ComparisonModel):
    parameter_order: list[str]
    price_gradient: list[float]
    approximate_price_variance: NonNegativeFloat
    approximate_price_standard_deviation: NonNegativeFloat
    interpretation: str = (
        "Local first-order delta-method approximation; it is not a confidence interval."
    )


class UncertaintyDecomposition(ComparisonModel):
    monte_carlo_sampling: dict[str, float]
    model_dispersion: PriceDispersion
    calibration_dispersion: ScalarDispersion | None
    market_spread: dict[str, float]
    numerical_discretization: dict[str, float]
    note: str = (
        "Sampling, numerical, calibration, model, and market-spread uncertainty are "
        "reported separately and are not combined into a synthetic interval."
    )


class ModelRiskRequest(ComparisonModel):
    comparison: ModelComparisonRequest
    market_fit: MarketFitRequest | None = None
    holdout: HestonCalibrationHoldoutConfig | None = None
    calibration_stability: HestonCalibrationStabilityResult | None = None
    calibration_ensemble: list[HestonParameters] = Field(default_factory=list)
    local_parameter_covariance: list[list[float]] | None = None


class ModelRiskReport(ComparisonModel):
    comparison: ModelComparisonResult
    market_fit: MarketFitResult | None
    out_of_sample: OutOfSampleComparisonResult | None
    calibration_stability: CalibrationStabilityValuation | None
    parameter_uncertainty: ParameterUncertaintyResult | None
    uncertainty: UncertaintyDecomposition
    complexity_vs_fit: list[ModelComplexityFit]
    flags: list[ModelRiskFlag]
    warnings: list[str]


class ConsistencyCheck(ComparisonModel):
    name: str
    left_model: PricingModel
    right_model: PricingModel
    left_price: NonNegativeFloat | None
    right_price: NonNegativeFloat | None
    absolute_difference: NonNegativeFloat | None
    tolerance: NonNegativeFloat
    passed: bool
    z_score: float | None = None
    message: str


class ConsistencyRequest(ComparisonModel):
    contract: OptionContract
    market: MarketEnvironment
    simulation: SimulationConfig = Field(
        default_factory=lambda: SimulationConfig(num_simulations=50_000)
    )
    tree: BinomialTreeConfig = Field(default_factory=BinomialTreeConfig)
    heston_monte_carlo: HestonSimulationConfig = Field(
        default_factory=lambda: HestonSimulationConfig(num_simulations=20_000, num_time_steps=50)
    )
    heston_integration: HestonIntegrationConfig = Field(default_factory=HestonIntegrationConfig)
    tolerance: float = Field(default=0.05, ge=0.0)
    seed: int = Field(default=42, ge=0)


class ConsistencyResult(ComparisonModel):
    checks: list[ConsistencyCheck]
    flags: list[ModelRiskFlag]
    input_signature: str


class SurfaceComparisonRequest(ComparisonModel):
    surface: VolatilitySurface
    selected_models: list[PricingModel] = Field(
        default_factory=lambda: [PricingModel.BLACK_SCHOLES, PricingModel.HESTON_ANALYTICAL]
    )
    heston_parameters: HestonParameters | None = None
    heston_calibration_result: HestonCalibrationResult | None = None
    merton_parameters: MertonJumpParameters | None = None
    tree: BinomialTreeConfig = Field(default_factory=BinomialTreeConfig)
    integration: HestonIntegrationConfig = Field(default_factory=HestonIntegrationConfig)
    gbm_monte_carlo: SimulationConfig = Field(
        default_factory=lambda: SimulationConfig(num_simulations=10_000)
    )
    heston_monte_carlo: HestonSimulationConfig = Field(
        default_factory=lambda: HestonSimulationConfig(num_simulations=5_000, num_time_steps=25)
    )
    merton_monte_carlo: MertonSimulationConfig = Field(
        default_factory=lambda: MertonSimulationConfig(num_simulations=5_000)
    )
    merton_series: MertonSeriesConfig = Field(default_factory=MertonSeriesConfig)
    max_returned_residuals: PositiveInt = 500

    @model_validator(mode="after")
    def validate_models(self) -> Self:
        if len(set(self.selected_models)) != len(self.selected_models):
            raise ValueError("selected_models must not contain duplicates")
        return self
