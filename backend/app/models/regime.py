"""Public models for finite Markov chains and Gaussian hidden Markov models."""

from __future__ import annotations

import math
from datetime import datetime
from enum import StrEnum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class RegimeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class CovarianceType(StrEnum):
    DIAGONAL = "DIAGONAL"
    FULL = "FULL"


class HMMInitialization(StrEnum):
    QUANTILE = "QUANTILE"
    RANDOM = "RANDOM"


class CanonicalOrdering(StrEnum):
    FIRST_FEATURE_MEAN = "FIRST_FEATURE_MEAN"
    VOLATILITY = "VOLATILITY"


class ProbabilityOutputMode(StrEnum):
    FINAL_ONLY = "FINAL_ONLY"
    THINNED = "THINNED"
    FULL = "FULL"


class RegimeRiskValuationMode(StrEnum):
    DELTA = "DELTA"
    DELTA_GAMMA_DIAGONAL = "DELTA_GAMMA_DIAGONAL"


class WalkForwardWindow(StrEnum):
    EXPANDING = "EXPANDING"
    ROLLING = "ROLLING"


class MarkovChain(RegimeModel):
    transition_matrix: list[list[float]]
    initial_probabilities: list[float] | None = None

    @model_validator(mode="after")
    def validate_chain(self) -> MarkovChain:
        k = len(self.transition_matrix)
        if k < 2:
            raise ValueError("a Markov chain requires at least two states")
        for row in self.transition_matrix:
            if len(row) != k:
                raise ValueError("transition_matrix must be square")
            if any(not math.isfinite(value) or value < 0.0 for value in row):
                raise ValueError("transition probabilities must be finite and nonnegative")
            if not math.isclose(sum(row), 1.0, rel_tol=0.0, abs_tol=1e-10):
                raise ValueError("each transition row must sum to one")
        if self.initial_probabilities is not None:
            if len(self.initial_probabilities) != k:
                raise ValueError("initial_probabilities length must match the state count")
            if any(not math.isfinite(value) or value < 0.0 for value in self.initial_probabilities):
                raise ValueError("initial probabilities must be finite and nonnegative")
            if not math.isclose(sum(self.initial_probabilities), 1.0, rel_tol=0.0, abs_tol=1e-10):
                raise ValueError("initial probabilities must sum to one")
        return self

    @property
    def num_states(self) -> int:
        return len(self.transition_matrix)


class MarkovStateDiagnostic(RegimeModel):
    state: int = Field(ge=0)
    persistence: float = Field(ge=0.0, le=1.0)
    expected_duration: float | None = Field(default=None, gt=0.0)
    absorbing: bool
    stationary_probability: float = Field(ge=0.0, le=1.0)


class MarkovDiagnostics(RegimeModel):
    transition_matrix: list[list[float]]
    stationary_distribution: list[float]
    stationary_distribution_unique: bool
    irreducible: bool
    stationary_nullity: int = Field(ge=1)
    states: list[MarkovStateDiagnostic]
    row_entropies: list[float]


class MarkovDiagnosticsRequest(RegimeModel):
    chain: MarkovChain
    horizons: list[int] = Field(default_factory=lambda: [1])
    current_probabilities: list[float] | None = None

    @model_validator(mode="after")
    def validate_request(self) -> MarkovDiagnosticsRequest:
        if not self.horizons or any(value < 0 for value in self.horizons):
            raise ValueError("horizons must contain nonnegative integers")
        if self.current_probabilities is not None:
            _validate_probability_vector(
                self.current_probabilities, self.chain.num_states, "current_probabilities"
            )
        return self


class ProbabilityForecast(RegimeModel):
    horizon: int = Field(ge=0)
    probabilities: list[float]


class MarkovDiagnosticsResult(RegimeModel):
    diagnostics: MarkovDiagnostics
    forecasts: list[ProbabilityForecast]


class HMMObservationDataset(RegimeModel):
    observations: list[list[float]]
    timestamps: list[datetime] | None = None
    feature_names: list[str] | None = None

    @model_validator(mode="after")
    def validate_dataset(self) -> HMMObservationDataset:
        if not self.observations:
            raise ValueError("observations must not be empty")
        dimension = len(self.observations[0])
        if dimension < 1:
            raise ValueError("observations must contain at least one feature")
        for row in self.observations:
            if len(row) != dimension:
                raise ValueError("observations must form a rectangular matrix")
            if any(not math.isfinite(value) for value in row):
                raise ValueError("observations must be finite; missing values are unsupported")
        if self.timestamps is not None and len(self.timestamps) != len(self.observations):
            raise ValueError("timestamps length must match observations")
        if self.feature_names is not None:
            if len(self.feature_names) != dimension:
                raise ValueError("feature_names length must match the feature dimension")
            if len(set(self.feature_names)) != dimension or any(
                not name.strip() for name in self.feature_names
            ):
                raise ValueError("feature_names must be unique and non-empty")
        return self

    @property
    def observation_count(self) -> int:
        return len(self.observations)

    @property
    def feature_dimension(self) -> int:
        return len(self.observations[0])


class GaussianHMMParameters(RegimeModel):
    num_states: int = Field(ge=2)
    feature_dimension: int = Field(ge=1)
    covariance_type: CovarianceType = CovarianceType.DIAGONAL
    initial_probabilities: list[float]
    transition_matrix: list[list[float]]
    means: list[list[float]]
    covariances: list[list[list[float]]]
    standardization_mean: list[float] | None = None
    standardization_std: list[float] | None = None

    @model_validator(mode="after")
    def validate_parameters(self) -> GaussianHMMParameters:
        k, d = self.num_states, self.feature_dimension
        _validate_probability_vector(self.initial_probabilities, k, "initial_probabilities")
        MarkovChain(
            transition_matrix=self.transition_matrix,
            initial_probabilities=self.initial_probabilities,
        )
        if len(self.means) != k or any(len(row) != d for row in self.means):
            raise ValueError("means must have shape (num_states, feature_dimension)")
        if any(not math.isfinite(value) for row in self.means for value in row):
            raise ValueError("means must be finite")
        if len(self.covariances) != k:
            raise ValueError("covariances must contain one matrix per state")
        for matrix in self.covariances:
            if len(matrix) != d or any(len(row) != d for row in matrix):
                raise ValueError("each covariance must be feature_dimension square")
            if any(not math.isfinite(value) for row in matrix for value in row):
                raise ValueError("covariances must be finite")
            for i in range(d):
                if matrix[i][i] <= 0.0:
                    raise ValueError("covariance diagonal entries must be positive")
                for j in range(d):
                    if not math.isclose(matrix[i][j], matrix[j][i], abs_tol=1e-10):
                        raise ValueError("covariance matrices must be symmetric")
                    if (
                        self.covariance_type is CovarianceType.DIAGONAL
                        and i != j
                        and not math.isclose(matrix[i][j], 0.0, abs_tol=1e-12)
                    ):
                        raise ValueError("diagonal covariance matrices cannot have cross terms")
            eigenvalues = np.linalg.eigvalsh(np.asarray(matrix, dtype=np.float64))
            scale = max(1.0, float(np.max(np.abs(matrix))))
            if float(eigenvalues[0]) < -1e-10 * scale:
                raise ValueError("covariance matrices must be positive semidefinite")
        if (self.standardization_mean is None) != (self.standardization_std is None):
            raise ValueError("standardization mean and std must be supplied together")
        if self.standardization_mean is not None and self.standardization_std is not None:
            if len(self.standardization_mean) != d or len(self.standardization_std) != d:
                raise ValueError("standardization vectors must match feature_dimension")
            if any(value <= 0.0 for value in self.standardization_std):
                raise ValueError("standardization standard deviations must be positive")
        return self


class HMMFitConfig(RegimeModel):
    num_states: int = Field(default=2, ge=2)
    covariance_type: CovarianceType = CovarianceType.DIAGONAL
    max_iterations: int = Field(default=200, gt=0)
    tolerance: float = Field(default=1e-6, gt=0.0)
    min_variance: float = Field(default=1e-6, gt=0.0)
    num_starts: int = Field(default=3, gt=0)
    seed: int = Field(default=42, ge=0)
    initialization: HMMInitialization = HMMInitialization.QUANTILE
    canonical_ordering: CanonicalOrdering = CanonicalOrdering.FIRST_FEATURE_MEAN
    occupancy_floor: float = Field(default=1e-6, gt=0.0, lt=1.0)
    transition_floor: float = Field(default=1e-10, gt=0.0, lt=0.5)
    return_probability_series: bool = False
    probability_output_mode: ProbabilityOutputMode = ProbabilityOutputMode.FINAL_ONLY
    probability_stride: int = Field(default=1, gt=0)
    standardize: bool = False

    @model_validator(mode="after")
    def align_probability_policy(self) -> HMMFitConfig:
        if (
            self.return_probability_series
            and self.probability_output_mode is ProbabilityOutputMode.FINAL_ONLY
        ):
            object.__setattr__(self, "probability_output_mode", ProbabilityOutputMode.FULL)
        return self


class HMMProbabilityPoint(RegimeModel):
    observation_index: int = Field(ge=0)
    timestamp: datetime | None = None
    filtered_probabilities: list[float]
    smoothed_probabilities: list[float]


class HMMStartSummary(RegimeModel):
    start_index: int = Field(ge=0)
    seed: int = Field(ge=0)
    initialization: HMMInitialization
    converged: bool
    iterations: int = Field(ge=0)
    log_likelihood: float
    message: str | None = None


class HMMStateSummary(RegimeModel):
    state: int = Field(ge=0)
    occupancy_probability: float = Field(ge=0.0, le=1.0)
    mean: list[float]
    standard_deviation: list[float]
    covariance: list[list[float]]
    persistence: float = Field(ge=0.0, le=1.0)
    expected_duration: float | None = Field(default=None, gt=0.0)
    stationary_probability: float = Field(ge=0.0, le=1.0)
    posterior_mass: float = Field(ge=0.0)
    empirical_observation_count: int = Field(ge=0)
    weighted_skewness: float | None = None
    weighted_excess_kurtosis: float | None = None
    minimum_first_feature: float | None = None
    maximum_first_feature: float | None = None


class HMMFitResult(RegimeModel):
    parameters: GaussianHMMParameters
    converged: bool
    iterations: int = Field(ge=0)
    log_likelihood: float
    likelihood_history: list[float]
    parameter_count: int = Field(gt=0)
    aic: float
    bic: float
    start_summaries: list[HMMStartSummary]
    state_summaries: list[HMMStateSummary]
    transition_diagnostics: MarkovDiagnostics
    filtered_final_probabilities: list[float]
    smoothed_final_probabilities: list[float]
    probability_series: list[HMMProbabilityPoint]
    probability_series_truncated: bool
    runtime_ms: float = Field(ge=0.0)


class HMMFitRequest(RegimeModel):
    dataset: HMMObservationDataset
    config: HMMFitConfig = Field(default_factory=HMMFitConfig)


class HMMDecodeRequest(RegimeModel):
    dataset: HMMObservationDataset
    parameters: GaussianHMMParameters
    include_probabilities: bool = False
    probability_stride: int = Field(default=1, gt=0)


class HMMDecodeResult(RegimeModel):
    state_sequence: list[int]
    path_log_probability: float
    log_likelihood: float
    probability_series: list[HMMProbabilityPoint]
    probability_series_truncated: bool


class HMMForecastRequest(RegimeModel):
    parameters: GaussianHMMParameters
    current_probabilities: list[float]
    horizons: list[int] = Field(default_factory=lambda: [1])

    @model_validator(mode="after")
    def validate_forecast(self) -> HMMForecastRequest:
        _validate_probability_vector(
            self.current_probabilities, self.parameters.num_states, "current_probabilities"
        )
        if not self.horizons or any(value < 1 for value in self.horizons):
            raise ValueError("forecast horizons must be positive")
        return self


class HMMMomentForecast(RegimeModel):
    horizon: int = Field(ge=1)
    state_probabilities: list[float]
    expected_mean: list[float]
    covariance: list[list[float]]


class HMMForecastResult(RegimeModel):
    forecasts: list[HMMMomentForecast]


class HMMFilterRequest(RegimeModel):
    parameters: GaussianHMMParameters
    prior_probabilities: list[float]
    observation: list[float]

    @model_validator(mode="after")
    def validate_filter(self) -> HMMFilterRequest:
        _validate_probability_vector(
            self.prior_probabilities, self.parameters.num_states, "prior_probabilities"
        )
        if len(self.observation) != self.parameters.feature_dimension:
            raise ValueError("observation dimension must match parameters")
        if any(not math.isfinite(value) for value in self.observation):
            raise ValueError("observation must be finite")
        return self


class HMMFilterResult(RegimeModel):
    predicted_probabilities: list[float]
    filtered_probabilities: list[float]
    observation_log_likelihood: float


class HMMModelOrderRequest(RegimeModel):
    dataset: HMMObservationDataset
    state_counts: list[int]
    config: HMMFitConfig = Field(default_factory=HMMFitConfig)

    @model_validator(mode="after")
    def validate_counts(self) -> HMMModelOrderRequest:
        if not self.state_counts or any(value < 2 for value in self.state_counts):
            raise ValueError("state_counts must contain values of at least two")
        if len(set(self.state_counts)) != len(self.state_counts):
            raise ValueError("state_counts must not contain duplicates")
        return self


class HMMModelOrderPoint(RegimeModel):
    num_states: int = Field(ge=2)
    log_likelihood: float
    aic: float
    bic: float
    converged: bool
    runtime_ms: float = Field(ge=0.0)


class HMMModelOrderResult(RegimeModel):
    points: list[HMMModelOrderPoint]


class HMMWalkForwardConfig(RegimeModel):
    initial_training_observations: int = Field(ge=2)
    refit_interval: int = Field(default=20, gt=0)
    window: WalkForwardWindow = WalkForwardWindow.EXPANDING
    rolling_window_observations: int | None = Field(default=None, ge=2)
    fit: HMMFitConfig = Field(default_factory=HMMFitConfig)

    @model_validator(mode="after")
    def validate_window(self) -> HMMWalkForwardConfig:
        if self.window is WalkForwardWindow.ROLLING:
            if self.rolling_window_observations is None:
                raise ValueError("rolling_window_observations is required for rolling analysis")
            if self.rolling_window_observations < self.initial_training_observations:
                raise ValueError("rolling window cannot be shorter than initial training data")
        return self


class HMMWalkForwardPoint(RegimeModel):
    observation_index: int = Field(ge=0)
    timestamp: datetime | None = None
    filtered_probabilities: list[float]
    observation_log_likelihood: float
    refitted: bool


class HMMWalkForwardResult(RegimeModel):
    points: list[HMMWalkForwardPoint]
    total_out_of_sample_log_likelihood: float
    refit_count: int = Field(ge=1)


class RegimeRiskRequest(RegimeModel):
    parameters: GaussianHMMParameters
    state_probabilities: list[float]
    factor_ids: list[str]
    delta_exposures: list[float]
    diagonal_gamma_exposures: list[float] | None = None
    valuation_mode: RegimeRiskValuationMode = RegimeRiskValuationMode.DELTA
    confidence_level: float = Field(default=0.99, gt=0.0, lt=1.0)
    num_simulations: int = Field(default=100_000, gt=0)
    seed: int = Field(default=42, ge=0)

    @model_validator(mode="after")
    def validate_risk(self) -> RegimeRiskRequest:
        k, d = self.parameters.num_states, self.parameters.feature_dimension
        _validate_probability_vector(self.state_probabilities, k, "state_probabilities")
        if len(self.factor_ids) != d or len(set(self.factor_ids)) != d:
            raise ValueError("factor_ids must be unique and match the feature dimension")
        if len(self.delta_exposures) != d or any(
            not math.isfinite(value) for value in self.delta_exposures
        ):
            raise ValueError("delta_exposures must be finite and match the feature dimension")
        if self.valuation_mode is RegimeRiskValuationMode.DELTA_GAMMA_DIAGONAL and (
            self.diagonal_gamma_exposures is None or len(self.diagonal_gamma_exposures) != d
        ):
            raise ValueError("diagonal gamma exposures are required for Delta-Gamma mode")
        if self.diagonal_gamma_exposures is not None and any(
            not math.isfinite(value) for value in self.diagonal_gamma_exposures
        ):
            raise ValueError("diagonal gamma exposures must be finite")
        return self


class StateConditionalRisk(RegimeModel):
    state: int = Field(ge=0)
    probability: float = Field(ge=0.0, le=1.0)
    value_at_risk: float
    expected_shortfall: float
    pnl_mean: float
    pnl_standard_deviation: float = Field(ge=0.0)
    mean: list[float]
    covariance: list[list[float]]


class RegimeRiskResult(RegimeModel):
    confidence_level: float = Field(gt=0.0, lt=1.0)
    valuation_mode: RegimeRiskValuationMode
    state_probabilities: list[float]
    state_risks: list[StateConditionalRisk]
    mixture_value_at_risk: float
    mixture_expected_shortfall: float
    mixture_pnl_mean: float
    mixture_pnl_standard_deviation: float = Field(ge=0.0)
    num_simulations: int = Field(gt=0)
    seed: int = Field(ge=0)


class RegimeGroupedMetric(RegimeModel):
    state: int = Field(ge=0)
    observation_count: int = Field(ge=0)
    rmse: float | None = Field(default=None, ge=0.0)
    mae: float | None = Field(default=None, ge=0.0)
    bias: float | None = None


class RegimeAwareStressContext(RegimeModel):
    """Regime probabilities shown alongside unchanged named stress results."""

    state_probabilities: list[float]
    stress_result: dict[str, object]


class MarkovSwitchingGBMConfig(RegimeModel):
    initial_spot: float = Field(gt=0.0)
    drifts: list[float]
    volatilities: list[float]
    transition_matrix: list[list[float]]
    initial_probabilities: list[float]
    num_steps: int = Field(gt=0)
    num_paths: int = Field(gt=0)
    time_step: float = Field(gt=0.0)
    seed: int = Field(default=42, ge=0)

    @model_validator(mode="after")
    def validate_switching(self) -> MarkovSwitchingGBMConfig:
        chain = MarkovChain(
            transition_matrix=self.transition_matrix,
            initial_probabilities=self.initial_probabilities,
        )
        if len(self.drifts) != chain.num_states or len(self.volatilities) != chain.num_states:
            raise ValueError("drifts and volatilities must match the number of states")
        if any(not math.isfinite(value) for value in self.drifts):
            raise ValueError("drifts must be finite")
        if any(not math.isfinite(value) or value < 0.0 for value in self.volatilities):
            raise ValueError("volatilities must be finite and nonnegative")
        return self


def _validate_probability_vector(values: list[float], size: int, name: str) -> None:
    if len(values) != size:
        raise ValueError(f"{name} length must match the state count")
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError(f"{name} must be finite and nonnegative")
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError(f"{name} must sum to one")
