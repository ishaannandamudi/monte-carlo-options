"""Typed portfolio VaR, Expected Shortfall, contribution, and backtest contracts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.models.portfolio import (
    InstrumentClass,
    MarketScenario,
    Portfolio,
    PortfolioAnalysisConfig,
    PortfolioPricingModel,
)

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]


class RiskModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class RiskMethod(StrEnum):
    HISTORICAL = "HISTORICAL"
    PARAMETRIC_NORMAL = "PARAMETRIC_NORMAL"
    MONTE_CARLO = "MONTE_CARLO"


class RiskValuationMode(StrEnum):
    FULL_REVALUATION = "FULL_REVALUATION"
    DELTA = "DELTA"
    DELTA_GAMMA_DIAGONAL = "DELTA_GAMMA_DIAGONAL"


class RiskDistribution(StrEnum):
    MULTIVARIATE_NORMAL = "MULTIVARIATE_NORMAL"
    MULTIVARIATE_STUDENT_T = "MULTIVARIATE_STUDENT_T"


class MissingFactorPolicy(StrEnum):
    ERROR = "ERROR"
    ZERO_SHOCK = "ZERO_SHOCK"
    DROP_SCENARIO = "DROP_SCENARIO"


class RiskFactorType(StrEnum):
    SPOT_RETURN = "SPOT_RETURN"
    VOLATILITY_CHANGE = "VOLATILITY_CHANGE"
    RATE_CHANGE = "RATE_CHANGE"
    DIVIDEND_YIELD_CHANGE = "DIVIDEND_YIELD_CHANGE"
    HESTON_V0_CHANGE = "HESTON_V0_CHANGE"
    HESTON_THETA_CHANGE = "HESTON_THETA_CHANGE"
    HESTON_KAPPA_CHANGE = "HESTON_KAPPA_CHANGE"
    HESTON_SIGMA_V_CHANGE = "HESTON_SIGMA_V_CHANGE"
    HESTON_RHO_CHANGE = "HESTON_RHO_CHANGE"
    MERTON_JUMP_INTENSITY_CHANGE = "MERTON_JUMP_INTENSITY_CHANGE"
    MERTON_JUMP_MEAN_CHANGE = "MERTON_JUMP_MEAN_CHANGE"
    MERTON_JUMP_VOLATILITY_CHANGE = "MERTON_JUMP_VOLATILITY_CHANGE"
    CORRELATION_CHANGE = "CORRELATION_CHANGE"
    CUSTOM_GAUSSIAN = "CUSTOM_GAUSSIAN"


class CovarianceSource(StrEnum):
    USER_SUPPLIED = "USER_SUPPLIED"
    HISTORICAL_SAMPLE = "HISTORICAL_SAMPLE"


class BacktestTrafficLight(StrEnum):
    BELOW_EXPECTED = "BELOW_EXPECTED"
    WITHIN_EXPECTED_RANGE = "WITHIN_EXPECTED_RANGE"
    ABOVE_EXPECTED = "ABOVE_EXPECTED"


class RiskHorizon(RiskModel):
    trading_days: PositiveInt = 1
    trading_days_per_year: PositiveInt = 252

    @property
    def year_fraction(self) -> float:
        return self.trading_days / self.trading_days_per_year


class HistoricalRiskFactorObservation(RiskModel):
    timestamp: datetime | None = None
    factor_id: str = Field(min_length=1)
    value: FiniteFloat
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class HistoricalRiskScenario(RiskModel):
    timestamp: datetime
    observations: list[HistoricalRiskFactorObservation] = Field(min_length=1)
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique_factors(self) -> Self:
        identifiers = [item.factor_id for item in self.observations]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("historical scenario factor_id values must be unique")
        return self


class HistoricalRiskDataset(RiskModel):
    scenarios: list[HistoricalRiskScenario] = Field(min_length=2)
    sampling_frequency: str = Field(default="DAILY", min_length=1)

    @model_validator(mode="after")
    def chronological_and_unique(self) -> Self:
        timestamps = [item.timestamp for item in self.scenarios]
        if timestamps != sorted(timestamps):
            raise ValueError("historical scenarios must be in chronological order")
        if len(timestamps) != len(set(timestamps)):
            raise ValueError("historical scenario timestamps must be unique")
        return self

    @property
    def start_timestamp(self) -> datetime:
        return self.scenarios[0].timestamp

    @property
    def end_timestamp(self) -> datetime:
        return self.scenarios[-1].timestamp


class RiskFactorMapping(RiskModel):
    factor_id: str = Field(min_length=1)
    factor_type: RiskFactorType
    underlying_id: str | None = None
    position_id: str | None = None
    explicit_position_exposures: dict[str, FiniteFloat] = Field(default_factory=dict)
    explicit_position_gamma_exposures: dict[str, FiniteFloat] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_target(self) -> Self:
        underlying_types = {
            RiskFactorType.SPOT_RETURN,
            RiskFactorType.VOLATILITY_CHANGE,
            RiskFactorType.DIVIDEND_YIELD_CHANGE,
        }
        position_types = {
            RiskFactorType.HESTON_V0_CHANGE,
            RiskFactorType.HESTON_THETA_CHANGE,
            RiskFactorType.HESTON_KAPPA_CHANGE,
            RiskFactorType.HESTON_SIGMA_V_CHANGE,
            RiskFactorType.HESTON_RHO_CHANGE,
            RiskFactorType.MERTON_JUMP_INTENSITY_CHANGE,
            RiskFactorType.MERTON_JUMP_MEAN_CHANGE,
            RiskFactorType.MERTON_JUMP_VOLATILITY_CHANGE,
            RiskFactorType.CORRELATION_CHANGE,
        }
        if self.factor_type in underlying_types and self.underlying_id is None:
            raise ValueError(f"{self.factor_type.value} requires underlying_id")
        if self.factor_type in position_types and self.position_id is None:
            raise ValueError(f"{self.factor_type.value} requires position_id")
        if (
            self.factor_type is RiskFactorType.CUSTOM_GAUSSIAN
            and not self.explicit_position_exposures
        ):
            raise ValueError("CUSTOM_GAUSSIAN requires explicit_position_exposures")
        return self


class CovarianceInput(RiskModel):
    factor_ids: list[str] = Field(min_length=1)
    covariance_matrix: tuple[tuple[FiniteFloat, ...], ...]
    mean_changes: tuple[FiniteFloat, ...] | None = None

    @model_validator(mode="after")
    def validate_covariance(self) -> Self:
        count = len(self.factor_ids)
        if len(set(self.factor_ids)) != count:
            raise ValueError("covariance factor_ids must be unique")
        matrix = np.asarray(self.covariance_matrix, dtype=np.float64)
        if matrix.shape != (count, count):
            raise ValueError("covariance_matrix dimensions must match factor_ids")
        if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12):
            raise ValueError("covariance_matrix must be symmetric")
        scale = max(1.0, float(np.max(np.abs(matrix))))
        minimum = float(np.linalg.eigvalsh(matrix)[0])
        if minimum < -1e-10 * scale:
            raise ValueError("covariance_matrix must be positive semidefinite")
        if self.mean_changes is not None and len(self.mean_changes) != count:
            raise ValueError("mean_changes length must match factor_ids")
        return self


class PortfolioRiskConfig(RiskModel):
    method: RiskMethod
    confidence_level: float = Field(default=0.99, gt=0.0, lt=1.0)
    horizon: RiskHorizon = Field(default_factory=RiskHorizon)
    valuation_mode: RiskValuationMode = RiskValuationMode.FULL_REVALUATION
    root_seed: NonNegativeInt = 42
    simulations: PositiveInt = 100_000
    distribution: RiskDistribution = RiskDistribution.MULTIVARIATE_NORMAL
    student_t_degrees_of_freedom: float = Field(default=5.0, gt=2.0)
    antithetic: bool = False
    batch_size: PositiveInt = 10_000
    missing_factor_policy: MissingFactorPolicy = MissingFactorPolicy.ERROR
    include_contributions: bool = True
    include_tail_scenarios: bool = True
    max_tail_scenarios: PositiveInt = 10
    include_incremental_var: bool = False
    iid_horizon_scaling: bool = True
    include_time_passage: bool = False
    quantile_method: Literal["linear", "lower", "higher", "midpoint", "nearest"] = "linear"

    @model_validator(mode="after")
    def method_consistency(self) -> Self:
        if self.method is RiskMethod.PARAMETRIC_NORMAL:
            if self.valuation_mode is not RiskValuationMode.DELTA:
                raise ValueError("PARAMETRIC_NORMAL supports DELTA valuation only")
            if self.distribution is not RiskDistribution.MULTIVARIATE_NORMAL:
                raise ValueError("PARAMETRIC_NORMAL requires MULTIVARIATE_NORMAL")
        if self.include_incremental_var and not self.include_contributions:
            raise ValueError("include_incremental_var requires include_contributions")
        return self


class BasePortfolioRiskRequest(RiskModel):
    portfolio: Portfolio
    factor_mappings: list[RiskFactorMapping] = Field(min_length=1)
    config: PortfolioRiskConfig
    portfolio_config: PortfolioAnalysisConfig = Field(default_factory=PortfolioAnalysisConfig)

    @model_validator(mode="after")
    def unique_factor_mappings(self) -> Self:
        identifiers = [item.factor_id for item in self.factor_mappings]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("factor_mappings factor_id values must be unique")
        positions = {item.position_id for item in self.portfolio.positions}
        for mapping in self.factor_mappings:
            if mapping.position_id is not None and mapping.position_id not in positions:
                raise ValueError(f"unknown mapped position_id: {mapping.position_id}")
            if not set(mapping.explicit_position_exposures).issubset(positions):
                raise ValueError("explicit_position_exposures contains unknown position_id")
            if not set(mapping.explicit_position_gamma_exposures).issubset(positions):
                raise ValueError("explicit_position_gamma_exposures contains unknown position_id")
        return self


class HistoricalRiskRequest(BasePortfolioRiskRequest):
    dataset: HistoricalRiskDataset

    @model_validator(mode="after")
    def historical_method(self) -> Self:
        if self.config.method is not RiskMethod.HISTORICAL:
            raise ValueError("historical request config.method must be HISTORICAL")
        return self


class ParametricRiskRequest(BasePortfolioRiskRequest):
    covariance: CovarianceInput | None = None
    historical_dataset: HistoricalRiskDataset | None = None

    @model_validator(mode="after")
    def parametric_inputs(self) -> Self:
        if self.config.method is not RiskMethod.PARAMETRIC_NORMAL:
            raise ValueError("parametric request config.method must be PARAMETRIC_NORMAL")
        if (self.covariance is None) == (self.historical_dataset is None):
            raise ValueError("provide exactly one of covariance or historical_dataset")
        return self


class MonteCarloRiskRequest(BasePortfolioRiskRequest):
    covariance: CovarianceInput | None = None
    historical_dataset: HistoricalRiskDataset | None = None

    @model_validator(mode="after")
    def monte_carlo_inputs(self) -> Self:
        if self.config.method is not RiskMethod.MONTE_CARLO:
            raise ValueError("Monte Carlo request config.method must be MONTE_CARLO")
        if (self.covariance is None) == (self.historical_dataset is None):
            raise ValueError("provide exactly one of covariance or historical_dataset")
        return self


class PortfolioRiskRequest(RiskModel):
    portfolio: Portfolio
    factor_mappings: list[RiskFactorMapping] = Field(min_length=1)
    config: PortfolioRiskConfig
    portfolio_config: PortfolioAnalysisConfig = Field(default_factory=PortfolioAnalysisConfig)
    historical_dataset: HistoricalRiskDataset | None = None
    covariance: CovarianceInput | None = None

    @model_validator(mode="after")
    def validate_method_inputs(self) -> Self:
        BasePortfolioRiskRequest(
            portfolio=self.portfolio,
            factor_mappings=self.factor_mappings,
            config=self.config,
            portfolio_config=self.portfolio_config,
        )
        if self.config.method is RiskMethod.HISTORICAL:
            if self.historical_dataset is None or self.covariance is not None:
                raise ValueError("HISTORICAL requires historical_dataset and no covariance")
        elif (self.covariance is None) == (self.historical_dataset is None):
            raise ValueError("provide exactly one covariance source")
        return self


class PositionTailContribution(RiskModel):
    position_id: str
    loss: float


class WorstRiskScenario(RiskModel):
    scenario_index: NonNegativeInt
    timestamp: datetime | None = None
    pnl: float
    loss: float
    factor_shocks: dict[str, float] = Field(default_factory=dict)
    position_contributions: list[PositionTailContribution] = Field(default_factory=list)


class LossDistributionSummary(RiskModel):
    pnl_mean: float
    pnl_standard_deviation: float = Field(ge=0.0)
    minimum_pnl: float | None
    maximum_pnl: float | None
    loss_mean: float
    loss_standard_deviation: float = Field(ge=0.0)
    median_loss: float
    worst_loss: float | None
    best_pnl: float | None
    skewness: float
    excess_kurtosis: float
    loss_quantiles: dict[str, float]
    tail_count: NonNegativeInt
    jarque_bera_statistic: float | None = Field(default=None, ge=0.0)
    jarque_bera_p_value: float | None = Field(default=None, ge=0.0, le=1.0)


class FactorRiskContribution(RiskModel):
    factor_id: str
    exposure: float
    gamma_exposure: float | None = None
    marginal_var: float | None = None
    component_var: float | None = None
    component_es: float | None = None


class PositionRiskContribution(RiskModel):
    position_id: str
    underlying_keys: list[str]
    instrument_class: InstrumentClass
    model: PortfolioPricingModel
    market_value: float | None
    market_value_fraction: float | None = None
    selected_greek_exposure: float | None
    component_var: float | None = None
    incremental_var: float | None = None
    expected_shortfall_contribution: float | None = None


class GroupedRiskContribution(RiskModel):
    key: str
    component_var: float | None = None
    expected_shortfall_contribution: float | None = None
    position_count: PositiveInt


class RiskConcentration(RiskModel):
    largest_component_var_position: str | None = None
    largest_expected_shortfall_contributor: str | None = None
    component_var_absolute_hhi: float | None = Field(default=None, ge=0.0, le=1.0)
    expected_shortfall_absolute_hhi: float | None = Field(default=None, ge=0.0, le=1.0)
    top_component_var_positions: list[str] = Field(default_factory=list)
    top_expected_shortfall_positions: list[str] = Field(default_factory=list)


class RiskContributionSummary(RiskModel):
    factors: list[FactorRiskContribution] = Field(default_factory=list)
    positions: list[PositionRiskContribution] = Field(default_factory=list)
    by_underlying: list[GroupedRiskContribution] = Field(default_factory=list)
    by_instrument_class: list[GroupedRiskContribution] = Field(default_factory=list)
    by_model: list[GroupedRiskContribution] = Field(default_factory=list)
    component_var_sum: float | None = None
    expected_shortfall_contribution_sum: float | None = None
    concentration: RiskConcentration


class CovarianceDiagnostics(RiskModel):
    source: CovarianceSource
    factor_ids: list[str]
    observation_count: int | None = Field(default=None, ge=2)
    minimum_eigenvalue: float
    effective_rank: NonNegativeInt
    condition_number: float | None = Field(default=None, ge=1.0)
    daily_means: list[float]


class StressLossPoint(RiskModel):
    name: str
    pnl: float | None
    loss: float | None


class PortfolioRiskReport(RiskModel):
    portfolio_id: str | None
    base_value: float | None
    value_at_risk: float
    expected_shortfall: float
    confidence_level: float = Field(gt=0.0, lt=1.0)
    horizon: RiskHorizon
    method: RiskMethod
    valuation_mode: RiskValuationMode
    scenario_count: NonNegativeInt
    simulations: int | None = Field(default=None, gt=0)
    distribution: RiskDistribution | None = None
    distribution_summary: LossDistributionSummary
    worst_scenarios: list[WorstRiskScenario] = Field(default_factory=list)
    contributions: RiskContributionSummary | None = None
    covariance_diagnostics: CovarianceDiagnostics | None = None
    historical_start: datetime | None = None
    historical_end: datetime | None = None
    sampling_frequency: str | None = None
    stress_losses: list[StressLossPoint] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    runtime_ms: float = Field(ge=0.0)


class VaRBacktestObservation(RiskModel):
    date: datetime
    predicted_var: float
    realized_pnl: float
    predicted_es: float | None = None

    @property
    def loss(self) -> float:
        return -self.realized_pnl


class VaRBacktestRequest(RiskModel):
    confidence_level: float = Field(gt=0.0, lt=1.0)
    observations: list[VaRBacktestObservation] = Field(min_length=2)

    @model_validator(mode="after")
    def unique_ordered_dates(self) -> Self:
        dates = [item.date for item in self.observations]
        if dates != sorted(dates) or len(dates) != len(set(dates)):
            raise ValueError("backtest dates must be unique and chronological")
        return self


class VaRBacktestResult(RiskModel):
    observation_count: PositiveInt
    exception_count: NonNegativeInt
    exception_rate: float = Field(ge=0.0, le=1.0)
    expected_exception_rate: float = Field(gt=0.0, lt=1.0)
    exception_dates: list[datetime]
    exceptions: list[bool]
    kupiec_statistic: float = Field(ge=0.0)
    kupiec_p_value: float = Field(ge=0.0, le=1.0)
    christoffersen_statistic: float = Field(ge=0.0)
    christoffersen_p_value: float = Field(ge=0.0, le=1.0)
    conditional_coverage_statistic: float = Field(ge=0.0)
    conditional_coverage_p_value: float = Field(ge=0.0, le=1.0)
    transitions_00: NonNegativeInt
    transitions_01: NonNegativeInt
    transitions_10: NonNegativeInt
    transitions_11: NonNegativeInt
    traffic_light: BacktestTrafficLight
    equality_policy: str = "An exception occurs only when loss is strictly greater than VaR."


class PortfolioRiskReportRequest(PortfolioRiskRequest):
    stress_scenarios: list[MarketScenario] = Field(default_factory=list)

    @model_validator(mode="after")
    def named_unique_stresses(self) -> Self:
        names = [item.name for item in self.stress_scenarios]
        if any(name is None for name in names) or len(names) != len(set(names)):
            raise ValueError("stress scenarios must have unique names")
        return self


def model_name_for_position(position: object) -> PortfolioPricingModel:
    """Return the explicit/default model without importing the portfolio engine."""

    from backend.app.models.portfolio import (  # local import avoids a broad model cycle
        AsianPosition,
        BarrierPosition,
        BasketPosition,
        DigitalPosition,
        VanillaPosition,
        basket_default_model,
    )

    if isinstance(position, VanillaPosition):
        if position.model is not None:
            return position.model
        return (
            PortfolioPricingModel.BLACK_SCHOLES
            if position.instrument.exercise_style.value == "EUROPEAN"
            else PortfolioPricingModel.CRR_AMERICAN
        )
    if isinstance(position, AsianPosition):
        return position.model or (
            PortfolioPricingModel.GEOMETRIC_ASIAN_ANALYTICAL
            if position.instrument.averaging_type.value == "GEOMETRIC"
            else PortfolioPricingModel.ASIAN_MONTE_CARLO
        )
    if isinstance(position, DigitalPosition):
        return position.model or PortfolioPricingModel.DIGITAL_ANALYTICAL
    if isinstance(position, BarrierPosition):
        return position.model or PortfolioPricingModel.BARRIER_MONTE_CARLO
    if isinstance(position, BasketPosition):
        return position.model or basket_default_model(position)
    raise TypeError("unsupported portfolio position")
