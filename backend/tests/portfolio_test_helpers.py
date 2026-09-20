"""Small deterministic portfolio fixtures shared by Phase 16 tests."""

from backend.app.models.asian import AsianOptionContract, AveragingType
from backend.app.models.barrier import (
    BarrierDirection,
    BarrierMonitoringMode,
    BarrierOptionContract,
    BarrierStyle,
)
from backend.app.models.basket import BasketOptionContract, BasketType
from backend.app.models.binomial import BinomialTreeConfig
from backend.app.models.digital import DigitalOptionContract
from backend.app.models.heston import HestonParameters
from backend.app.models.market import MarketEnvironment
from backend.app.models.multi_asset import MultiAssetMarketEnvironment
from backend.app.models.option import ExerciseStyle, OptionContract, OptionType
from backend.app.models.portfolio import (
    AsianPosition,
    BarrierPosition,
    BasketPosition,
    DigitalPosition,
    InstrumentClass,
    Portfolio,
    PortfolioPricingModel,
    VanillaPosition,
)


def market(volatility: float = 0.2) -> MarketEnvironment:
    return MarketEnvironment(risk_free_rate=0.03, volatility=volatility, dividend_yield=0.01)


def vanilla(
    position_id: str = "vanilla",
    quantity: float = 1.0,
    *,
    underlying_id: str = "SPY",
    option_type: OptionType = OptionType.CALL,
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN,
    model: PortfolioPricingModel | None = None,
) -> VanillaPosition:
    instrument_class = (
        InstrumentClass.VANILLA_EUROPEAN
        if exercise_style is ExerciseStyle.EUROPEAN
        else InstrumentClass.VANILLA_AMERICAN
    )
    return VanillaPosition(
        position_id=position_id,
        quantity=quantity,
        instrument_class=instrument_class,
        underlying_id=underlying_id,
        instrument=OptionContract(
            option_type=option_type,
            exercise_style=exercise_style,
            spot=100.0,
            strike=100.0,
            maturity_years=1.0,
        ),
        market=market(),
        model=model,
        heston_parameters=HestonParameters(v0=0.04, kappa=1.5, theta=0.04, sigma_v=0.35, rho=-0.6),
    )


def mixed_portfolio() -> Portfolio:
    multi_market = MultiAssetMarketEnvironment(
        spots=(100.0, 80.0),
        risk_free_rate=0.03,
        dividend_yields=(0.01, 0.02),
        volatilities=(0.2, 0.25),
        correlation_matrix=((1.0, 0.3), (0.3, 1.0)),
        asset_ids=("AAA", "BBB"),
    )
    return Portfolio(
        portfolio_id="mixed",
        positions=[
            vanilla(),
            vanilla(
                "american",
                option_type=OptionType.PUT,
                exercise_style=ExerciseStyle.AMERICAN,
            ).model_copy(update={"tree": BinomialTreeConfig(steps=50)}),
            AsianPosition(
                position_id="asian",
                quantity=2.0,
                underlying_id="SPY",
                instrument=AsianOptionContract(
                    option_type=OptionType.CALL,
                    spot=100.0,
                    strike=100.0,
                    maturity_years=1.0,
                    averaging_type=AveragingType.GEOMETRIC,
                ),
                market=market(),
            ),
            DigitalPosition(
                position_id="digital",
                quantity=10.0,
                underlying_id="SPY",
                instrument=DigitalOptionContract(
                    option_type=OptionType.CALL,
                    spot=100.0,
                    strike=100.0,
                    maturity_years=1.0,
                ),
                market=market(),
            ),
            BarrierPosition(
                position_id="barrier",
                quantity=1.0,
                underlying_id="SPY",
                model=PortfolioPricingModel.BARRIER_ANALYTICAL,
                instrument=BarrierOptionContract(
                    option_type=OptionType.CALL,
                    barrier_direction=BarrierDirection.UP,
                    barrier_style=BarrierStyle.OUT,
                    spot=100.0,
                    strike=100.0,
                    barrier=130.0,
                    maturity_years=1.0,
                    monitoring_mode=BarrierMonitoringMode.BROWNIAN_BRIDGE,
                ),
                market=market(),
            ),
            BasketPosition(
                position_id="basket",
                quantity=1.0,
                instrument=BasketOptionContract(
                    option_type=OptionType.CALL,
                    strike=90.0,
                    maturity_years=1.0,
                    weights=(0.6, 0.4),
                    basket_type=BasketType.GEOMETRIC,
                ),
                market=multi_market,
            ),
        ],
    )
