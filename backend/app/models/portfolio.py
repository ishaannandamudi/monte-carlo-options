"""Typed portfolio valuation, exposure, scenario, and P&L-explain contracts."""

from enum import StrEnum
from typing import Annotated, Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.models.asian import AsianOptionContract, AsianSimulationConfig
from backend.app.models.barrier import BarrierOptionContract, BarrierSimulationConfig
from backend.app.models.basket import BasketOptionContract, BasketSimulationConfig, BasketType
from backend.app.models.binomial import BinomialTreeConfig
from backend.app.models.digital import DigitalOptionContract, DigitalSimulationConfig
from backend.app.models.heston import (
    HestonIntegrationConfig,
    HestonParameters,
    HestonSimulationConfig,
)
from backend.app.models.heston_calibration import HestonCalibrationResult
from backend.app.models.longstaff_schwartz import LongstaffSchwartzConfig
from backend.app.models.market import MarketEnvironment
from backend.app.models.multi_asset import MultiAssetMarketEnvironment
from backend.app.models.option import ExerciseStyle, OptionContract, OptionType
from backend.app.models.simulation import SimulationConfig
from backend.app.models.volatility_surface import VolatilitySurface

MetadataValue: TypeAlias = str | int | float | bool | None
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]


class PortfolioModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def ignore_serialized_heston_computed_fields(cls, value: object) -> object:
        """Permit round trips of existing Heston models with computed diagnostics."""

        def clean(item: object) -> object:
            if isinstance(item, dict):
                return {
                    key: clean(child)
                    for key, child in item.items()
                    if key not in {"feller_satisfied", "feller_ratio"}
                }
            if isinstance(item, list):
                return [clean(child) for child in item]
            if isinstance(item, tuple):
                return tuple(clean(child) for child in item)
            return item

        return clean(value)


class InstrumentClass(StrEnum):
    VANILLA_EUROPEAN = "VANILLA_EUROPEAN"
    VANILLA_AMERICAN = "VANILLA_AMERICAN"
    ASIAN = "ASIAN"
    DIGITAL = "DIGITAL"
    BARRIER = "BARRIER"
    BASKET = "BASKET"


class PortfolioPricingModel(StrEnum):
    BLACK_SCHOLES = "BLACK_SCHOLES"
    MONTE_CARLO_GBM = "MONTE_CARLO_GBM"
    CRR_BINOMIAL = "CRR_BINOMIAL"
    CRR_AMERICAN = "CRR_AMERICAN"
    LONGSTAFF_SCHWARTZ = "LONGSTAFF_SCHWARTZ"
    HESTON_ANALYTICAL = "HESTON_ANALYTICAL"
    HESTON_MONTE_CARLO = "HESTON_MONTE_CARLO"
    ASIAN_MONTE_CARLO = "ASIAN_MONTE_CARLO"
    GEOMETRIC_ASIAN_ANALYTICAL = "GEOMETRIC_ASIAN_ANALYTICAL"
    DIGITAL_ANALYTICAL = "DIGITAL_ANALYTICAL"
    DIGITAL_MONTE_CARLO = "DIGITAL_MONTE_CARLO"
    BARRIER_ANALYTICAL = "BARRIER_ANALYTICAL"
    BARRIER_MONTE_CARLO = "BARRIER_MONTE_CARLO"
    GEOMETRIC_BASKET_ANALYTICAL = "GEOMETRIC_BASKET_ANALYTICAL"
    BASKET_MONTE_CARLO = "BASKET_MONTE_CARLO"


class PortfolioCompleteness(StrEnum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


class PositionStatus(StrEnum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class ScenarioFlag(StrEnum):
    LARGE_GREEK_APPROXIMATION_ERROR = "LARGE_GREEK_APPROXIMATION_ERROR"
    MISSING_GREEKS = "MISSING_GREEKS"
    DISCONTINUOUS_PAYOFF = "DISCONTINUOUS_PAYOFF"
    MODEL_CHANGED = "MODEL_CHANGED"
    MC_NOISE_MATERIAL = "MC_NOISE_MATERIAL"


class PortfolioRiskFlag(StrEnum):
    LARGE_NET_DELTA = "LARGE_NET_DELTA"
    LARGE_GROSS_VEGA = "LARGE_GROSS_VEGA"
    CONCENTRATED_UNDERLYING_EXPOSURE = "CONCENTRATED_UNDERLYING_EXPOSURE"


class ScenarioAxis(StrEnum):
    SPOT_RELATIVE = "SPOT_RELATIVE"
    SPOT_ABSOLUTE = "SPOT_ABSOLUTE"
    VOLATILITY_ABSOLUTE = "VOLATILITY_ABSOLUTE"
    VOLATILITY_RELATIVE = "VOLATILITY_RELATIVE"
    RATE_ABSOLUTE = "RATE_ABSOLUTE"
    DIVIDEND_YIELD_ABSOLUTE = "DIVIDEND_YIELD_ABSOLUTE"
    TIME_YEARS = "TIME_YEARS"


class _PositionBase(PortfolioModel):
    position_id: str = Field(min_length=1)
    quantity: float
    model: PortfolioPricingModel | None = None
    market_data_reference: str | None = None
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def nonzero_quantity(self) -> Self:
        if self.quantity == 0.0:
            raise ValueError("quantity must be nonzero")
        return self


class VanillaPosition(_PositionBase):
    instrument_class: Literal[
        InstrumentClass.VANILLA_EUROPEAN,
        InstrumentClass.VANILLA_AMERICAN,
    ]
    underlying_id: str = Field(min_length=1)
    instrument: OptionContract
    market: MarketEnvironment
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    tree: BinomialTreeConfig = Field(default_factory=BinomialTreeConfig)
    longstaff_schwartz: LongstaffSchwartzConfig = Field(default_factory=LongstaffSchwartzConfig)
    heston_parameters: HestonParameters | None = None
    heston_calibration_result: HestonCalibrationResult | None = None
    heston_simulation: HestonSimulationConfig = Field(default_factory=HestonSimulationConfig)
    heston_integration: HestonIntegrationConfig = Field(default_factory=HestonIntegrationConfig)
    volatility_surface: VolatilitySurface | None = None
    use_surface_volatility: bool = False

    @model_validator(mode="after")
    def consistent_exercise_style(self) -> Self:
        expected = (
            ExerciseStyle.EUROPEAN
            if self.instrument_class is InstrumentClass.VANILLA_EUROPEAN
            else ExerciseStyle.AMERICAN
        )
        if self.instrument.exercise_style is not expected:
            raise ValueError("instrument exercise style does not match instrument_class")
        if self.use_surface_volatility:
            if self.volatility_surface is None:
                raise ValueError("use_surface_volatility requires volatility_surface")
            if expected is not ExerciseStyle.EUROPEAN:
                raise ValueError("surface-based BSM valuation supports European options only")
            if self.model not in {None, PortfolioPricingModel.BLACK_SCHOLES}:
                raise ValueError("surface volatility is currently supported only by BLACK_SCHOLES")
        return self


class AsianPosition(_PositionBase):
    instrument_class: Literal[InstrumentClass.ASIAN] = InstrumentClass.ASIAN
    underlying_id: str = Field(min_length=1)
    instrument: AsianOptionContract
    market: MarketEnvironment
    simulation: AsianSimulationConfig = Field(default_factory=AsianSimulationConfig)


class DigitalPosition(_PositionBase):
    instrument_class: Literal[InstrumentClass.DIGITAL] = InstrumentClass.DIGITAL
    underlying_id: str = Field(min_length=1)
    instrument: DigitalOptionContract
    market: MarketEnvironment
    simulation: DigitalSimulationConfig = Field(default_factory=DigitalSimulationConfig)


class BarrierPosition(_PositionBase):
    instrument_class: Literal[InstrumentClass.BARRIER] = InstrumentClass.BARRIER
    underlying_id: str = Field(min_length=1)
    instrument: BarrierOptionContract
    market: MarketEnvironment
    simulation: BarrierSimulationConfig = Field(default_factory=BarrierSimulationConfig)


class BasketPosition(_PositionBase):
    instrument_class: Literal[InstrumentClass.BASKET] = InstrumentClass.BASKET
    instrument: BasketOptionContract
    market: MultiAssetMarketEnvironment
    simulation: BasketSimulationConfig = Field(default_factory=BasketSimulationConfig)

    @model_validator(mode="after")
    def require_asset_ids(self) -> Self:
        if self.market.asset_ids is None:
            raise ValueError("basket portfolio positions require explicit market asset_ids")
        return self


PortfolioPosition = Annotated[
    VanillaPosition | AsianPosition | DigitalPosition | BarrierPosition | BasketPosition,
    Field(discriminator="instrument_class"),
]


class Portfolio(PortfolioModel):
    portfolio_id: str | None = None
    positions: list[PortfolioPosition] = Field(min_length=1)
    base_currency: str = Field(default="USD", min_length=1)
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique_position_ids(self) -> Self:
        identifiers = [position.position_id for position in self.positions]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("position_id values must be unique")
        return self


class PortfolioAnalysisConfig(PortfolioModel):
    root_seed: NonNegativeInt = 42
    allow_partial: bool = True
    approximation_error_threshold: float = Field(default=0.10, ge=0.0)
    mc_noise_materiality_threshold: float = Field(default=0.05, ge=0.0)
    large_net_delta_threshold: float | None = Field(default=None, ge=0.0)
    large_gross_vega_threshold: float | None = Field(default=None, ge=0.0)
    underlying_concentration_hhi_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class PositionValuationResult(PortfolioModel):
    position_id: str
    instrument_class: InstrumentClass
    model: PortfolioPricingModel
    quantity: float
    status: PositionStatus
    unit_price: float | None = None
    position_value: float | None = None
    standard_error: float | None = Field(default=None, ge=0.0)
    confidence_interval_low: float | None = None
    confidence_interval_high: float | None = None
    error_type: str | None = None
    message: str | None = None
    warnings: list[str] = Field(default_factory=list)


class PortfolioValuationResult(PortfolioModel):
    portfolio_id: str | None
    base_currency: str
    total_value: float | None
    gross_long_value: float
    gross_short_value: float
    position_count: PositiveInt
    successful_position_count: NonNegativeInt
    failed_position_count: NonNegativeInt
    completeness: PortfolioCompleteness
    positions: list[PositionValuationResult]
    runtime_ms: float = Field(ge=0.0)


class GreekVector(PortfolioModel):
    delta: float | None = None
    gamma: float | None = None
    vega: float | None = None
    theta: float | None = None
    rho: float | None = None


class ScaledGreekVector(PortfolioModel):
    dollar_delta: float | None = None
    delta_1pct: float | None = None
    gamma_1pct: float | None = None
    vega_1vol_point: float | None = None
    theta_1day: float | None = None
    rho_1pct: float | None = None


class NetGrossExposure(PortfolioModel):
    net: float | None
    gross: float | None


class NetGrossGreeks(PortfolioModel):
    delta: NetGrossExposure
    gamma: NetGrossExposure
    vega: NetGrossExposure
    theta: NetGrossExposure
    rho: NetGrossExposure


class PositionGreeksResult(PortfolioModel):
    position_id: str
    status: PositionStatus
    model: PortfolioPricingModel
    quantity: float
    unit_greeks: GreekVector
    position_greeks: GreekVector
    scaled_position_greeks: ScaledGreekVector
    per_asset_delta: dict[str, float] = Field(default_factory=dict)
    unavailable_metrics: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PositionMarketSnapshot(PortfolioModel):
    spots: dict[str, float]
    volatilities: dict[str, float]
    risk_free_rate: float
    dividend_yields: dict[str, float]
    maturity_years: float = Field(ge=0.0)
    heston_parameters: HestonParameters | None = None
    correlation_matrix: tuple[tuple[float, ...], ...] | None = None


class ExposureGroup(PortfolioModel):
    key: str
    market_value: float
    position_count: PositiveInt
    greeks: GreekVector
    scaled_greeks: ScaledGreekVector
    missing_metrics: list[str] = Field(default_factory=list)


class ConcentrationContributor(PortfolioModel):
    position_id: str | None
    absolute_contribution: float = Field(ge=0.0)


class ConcentrationMetrics(PortfolioModel):
    largest_absolute_delta: ConcentrationContributor | None
    largest_absolute_vega: ConcentrationContributor | None
    largest_absolute_market_value: ConcentrationContributor | None
    underlying_delta_hhi: float | None = Field(default=None, ge=0.0, le=1.0)


class PortfolioGreeksResult(PortfolioModel):
    portfolio_id: str | None
    valuation: PortfolioValuationResult
    net_greeks: GreekVector
    gross_greeks: GreekVector
    net_gross: NetGrossGreeks
    scaled_net_greeks: ScaledGreekVector
    scaled_gross_greeks: ScaledGreekVector
    positions: list[PositionGreeksResult]
    by_underlying: list[ExposureGroup]
    by_expiry: list[ExposureGroup]
    by_option_type: list[ExposureGroup]
    by_instrument_class: list[ExposureGroup]
    by_model: list[ExposureGroup]
    concentration: ConcentrationMetrics
    unavailable_metrics: list[str] = Field(default_factory=list)
    risk_flags: list[PortfolioRiskFlag] = Field(default_factory=list)
    runtime_ms: float = Field(ge=0.0)


class HestonScenario(PortfolioModel):
    v0_shift: float | None = None
    theta_shift: float | None = None
    kappa_shift: float | None = None
    sigma_v_shift: float | None = None
    rho_shift: float | None = None
    overrides: HestonParameters | None = None

    @model_validator(mode="after")
    def exclusive_override_or_shifts(self) -> Self:
        shifts = (
            self.v0_shift,
            self.theta_shift,
            self.kappa_shift,
            self.sigma_v_shift,
            self.rho_shift,
        )
        if self.overrides is not None and any(value is not None for value in shifts):
            raise ValueError("choose Heston parameter overrides or additive shifts, not both")
        return self


class MarketScenario(PortfolioModel):
    name: str | None = None
    spot_relative_shift: float | None = None
    spot_absolute_shift: float | None = None
    volatility_absolute_shift: float | None = None
    volatility_relative_shift: float | None = None
    rate_absolute_shift: float | None = None
    dividend_yield_shift: float | None = None
    time_shift_years: float | None = Field(default=None, ge=0.0)
    per_underlying_spot_relative_shifts: dict[str, float] = Field(default_factory=dict)
    per_underlying_spot_absolute_shifts: dict[str, float] = Field(default_factory=dict)
    per_underlying_volatility_absolute_shifts: dict[str, float] = Field(default_factory=dict)
    per_underlying_dividend_yield_shifts: dict[str, float] = Field(default_factory=dict)
    heston: HestonScenario | None = None
    heston_parameters_by_position: dict[str, HestonParameters] = Field(default_factory=dict)
    correlation_matrix_by_position: dict[str, tuple[tuple[float, ...], ...]] = Field(
        default_factory=dict
    )
    equicorrelation_shift_by_position: dict[str, float] = Field(default_factory=dict)


class PositionScenarioContribution(PortfolioModel):
    position_id: str
    base_value: float | None
    scenario_value: float | None
    full_repricing_pnl: float | None
    greek_approximation_pnl: float | None
    approximation_residual: float | None
    relative_approximation_error: float | None
    base_standard_error: float | None = Field(default=None, ge=0.0)
    scenario_standard_error: float | None = Field(default=None, ge=0.0)
    pnl_standard_error: float | None = Field(default=None, ge=0.0)
    base_market: PositionMarketSnapshot | None = None
    scenario_market: PositionMarketSnapshot | None = None
    flags: list[ScenarioFlag] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ScenarioContributionGroup(PortfolioModel):
    key: str
    pnl: float
    position_count: PositiveInt


class PortfolioScenarioResult(PortfolioModel):
    portfolio_id: str | None
    scenario: MarketScenario
    base_value: float | None
    scenario_value: float | None
    full_repricing_pnl: float | None
    greek_approximation_pnl: float | None
    approximation_residual: float | None
    relative_approximation_error: float | None
    contributions: list[PositionScenarioContribution]
    by_underlying: list[ScenarioContributionGroup]
    by_instrument_class: list[ScenarioContributionGroup]
    completeness: PortfolioCompleteness
    flags: list[ScenarioFlag] = Field(default_factory=list)
    runtime_ms: float = Field(ge=0.0)


class ScenarioGridAxis(PortfolioModel):
    axis: ScenarioAxis
    values: list[float] = Field(min_length=1)


class ScenarioGridCell(PortfolioModel):
    coordinates: dict[str, float]
    scenario: MarketScenario
    portfolio_value: float | None
    full_repricing_pnl: float | None
    greek_approximation_pnl: float | None
    approximation_residual: float | None
    completeness: PortfolioCompleteness


class ScenarioGridResult(PortfolioModel):
    portfolio_id: str | None
    axes: list[ScenarioGridAxis]
    shape: list[PositiveInt]
    cells: list[ScenarioGridCell]
    runtime_ms: float = Field(ge=0.0)


class StressScenarioResult(PortfolioModel):
    name: str
    result: PortfolioScenarioResult


class PortfolioStressResult(PortfolioModel):
    portfolio_id: str | None
    scenarios: list[StressScenarioResult]
    runtime_ms: float = Field(ge=0.0)


class PnLExplainTerms(PortfolioModel):
    delta: float | None
    gamma: float | None
    vega: float | None
    theta: float | None
    rho: float | None
    explained_total: float | None
    unexplained_or_higher_order: float | None


class PositionPnLExplain(PortfolioModel):
    position_id: str
    full_reprice_pnl: float | None
    terms: PnLExplainTerms
    base_market: PositionMarketSnapshot | None = None
    scenario_market: PositionMarketSnapshot | None = None


class PortfolioPnLExplainResult(PortfolioModel):
    portfolio_id: str | None
    scenario: MarketScenario
    total_full_reprice_pnl: float | None
    terms: PnLExplainTerms
    positions: list[PositionPnLExplain]
    completeness: PortfolioCompleteness
    runtime_ms: float = Field(ge=0.0)


class PortfolioModelRiskPosition(PortfolioModel):
    position_id: str
    reference_value: float | None
    alternate_value: float | None
    difference: float | None
    status: PositionStatus
    message: str | None = None


class PortfolioModelRiskPoint(PortfolioModel):
    model: PortfolioPricingModel
    portfolio_value: float | None
    difference_from_reference: float | None
    contributions: list[PortfolioModelRiskPosition]
    completeness: PortfolioCompleteness


class PortfolioModelRiskResult(PortfolioModel):
    portfolio_id: str | None
    reference_model: PortfolioPricingModel
    reference_value: float | None
    alternatives: list[PortfolioModelRiskPoint]
    interpretation: str = (
        "Model-price spread is descriptive and is not a probabilistic risk measure."
    )
    runtime_ms: float = Field(ge=0.0)


class PortfolioRequest(PortfolioModel):
    portfolio: Portfolio
    config: PortfolioAnalysisConfig = Field(default_factory=PortfolioAnalysisConfig)


class PortfolioScenarioRequest(PortfolioRequest):
    scenario: MarketScenario


class ScenarioGridRequest(PortfolioRequest):
    axes: list[ScenarioGridAxis] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_axes(self) -> Self:
        axes = [axis.axis for axis in self.axes]
        if len(axes) != len(set(axes)):
            raise ValueError("scenario grid axes must be unique")
        return self


class PortfolioStressRequest(PortfolioRequest):
    scenarios: list[MarketScenario] = Field(min_length=1)

    @model_validator(mode="after")
    def named_scenarios(self) -> Self:
        if any(not scenario.name for scenario in self.scenarios):
            raise ValueError("every stress scenario must have a name")
        names = [scenario.name for scenario in self.scenarios]
        if len(names) != len(set(names)):
            raise ValueError("stress scenario names must be unique")
        return self


class PortfolioPnLExplainRequest(PortfolioScenarioRequest):
    pass


class PortfolioModelRiskRequest(PortfolioRequest):
    reference_model: PortfolioPricingModel
    alternate_models: list[PortfolioPricingModel] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_alternates(self) -> Self:
        if len(set(self.alternate_models)) != len(self.alternate_models):
            raise ValueError("alternate_models must be unique")
        if self.reference_model in self.alternate_models:
            raise ValueError("reference_model must not appear in alternate_models")
        return self


def position_option_type(position: PortfolioPosition) -> OptionType:
    return position.instrument.option_type


def position_maturity(position: PortfolioPosition) -> float:
    return position.instrument.maturity_years


def basket_default_model(position: BasketPosition) -> PortfolioPricingModel:
    if position.instrument.basket_type is BasketType.GEOMETRIC:
        return PortfolioPricingModel.GEOMETRIC_BASKET_ANALYTICAL
    return PortfolioPricingModel.BASKET_MONTE_CARLO
