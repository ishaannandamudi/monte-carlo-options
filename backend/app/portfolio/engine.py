"""Reusable portfolio valuation, exposure, scenario, and P&L-explain engine."""

from __future__ import annotations

import hashlib
import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import TypeVar, cast

import numpy as np

from backend.app.core.exceptions import PortfolioRiskError
from backend.app.model_comparison.registry import compatibility_exclusion
from backend.app.models.asian import AveragingType
from backend.app.models.barrier import BarrierMonitoringMode
from backend.app.models.basket import BasketType
from backend.app.models.greeks import Greek, GreekBumpConfig, GreekMethod
from backend.app.models.heston import HestonParameters
from backend.app.models.market import MarketEnvironment
from backend.app.models.model_comparison import PricingModel
from backend.app.models.multi_asset import MultiAssetMarketEnvironment
from backend.app.models.option import ExerciseStyle
from backend.app.models.portfolio import (
    AsianPosition,
    BarrierPosition,
    BasketPosition,
    ConcentrationContributor,
    ConcentrationMetrics,
    DigitalPosition,
    ExposureGroup,
    GreekVector,
    MarketScenario,
    NetGrossExposure,
    NetGrossGreeks,
    PnLExplainTerms,
    Portfolio,
    PortfolioAnalysisConfig,
    PortfolioCompleteness,
    PortfolioGreeksResult,
    PortfolioModelRiskPoint,
    PortfolioModelRiskPosition,
    PortfolioModelRiskRequest,
    PortfolioModelRiskResult,
    PortfolioPnLExplainRequest,
    PortfolioPnLExplainResult,
    PortfolioPosition,
    PortfolioPricingModel,
    PortfolioRequest,
    PortfolioRiskFlag,
    PortfolioScenarioRequest,
    PortfolioScenarioResult,
    PortfolioStressRequest,
    PortfolioStressResult,
    PortfolioValuationResult,
    PositionGreeksResult,
    PositionMarketSnapshot,
    PositionPnLExplain,
    PositionScenarioContribution,
    PositionStatus,
    PositionValuationResult,
    ScaledGreekVector,
    ScenarioAxis,
    ScenarioContributionGroup,
    ScenarioFlag,
    ScenarioGridCell,
    ScenarioGridRequest,
    ScenarioGridResult,
    StressScenarioResult,
    VanillaPosition,
    basket_default_model,
    position_maturity,
    position_option_type,
)
from backend.app.pricing.asian import asian_monte_carlo_price, geometric_asian_analytical_price
from backend.app.pricing.barrier import (
    barrier_monte_carlo_price,
    continuous_barrier_analytical_price,
)
from backend.app.pricing.basket import (
    basket_monte_carlo_price,
    geometric_basket_analytical_price,
)
from backend.app.pricing.binomial import binomial_price
from backend.app.pricing.black_scholes import black_scholes_price
from backend.app.pricing.black_scholes_greeks import black_scholes_greeks
from backend.app.pricing.digital import digital_analytical_price, digital_monte_carlo_price
from backend.app.pricing.heston import heston_monte_carlo_price
from backend.app.pricing.heston_analytical import heston_analytical_price
from backend.app.pricing.longstaff_schwartz import longstaff_schwartz_price
from backend.app.pricing.monte_carlo import MonteCarloPricer
from backend.app.pricing.monte_carlo_greeks import monte_carlo_greeks
from backend.app.simulation.correlation import prepare_correlation_matrix
from backend.app.volatility_surface.core import query_surface

PositionT = TypeVar("PositionT", bound=PortfolioPosition)
GREEK_NAMES = ("delta", "gamma", "vega", "theta", "rho")


@dataclass(frozen=True)
class _Price:
    value: float
    model: PortfolioPricingModel
    standard_error: float | None = None
    confidence_low: float | None = None
    confidence_high: float | None = None
    warnings: tuple[str, ...] = ()


def _seed(root_seed: int, position_id: str, model: PortfolioPricingModel) -> int:
    """Derive an order-independent PCG64-compatible child seed from stable identity."""

    digest = hashlib.sha256(f"{position_id}\0{model.value}".encode()).digest()
    identity = np.frombuffer(digest[:16], dtype=np.uint32).tolist()
    sequence = np.random.SeedSequence([root_seed, *identity])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _default_model(position: PortfolioPosition) -> PortfolioPricingModel:
    if isinstance(position, VanillaPosition):
        return (
            PortfolioPricingModel.BLACK_SCHOLES
            if position.instrument.exercise_style is ExerciseStyle.EUROPEAN
            else PortfolioPricingModel.CRR_AMERICAN
        )
    if isinstance(position, AsianPosition):
        return (
            PortfolioPricingModel.GEOMETRIC_ASIAN_ANALYTICAL
            if position.instrument.averaging_type is AveragingType.GEOMETRIC
            else PortfolioPricingModel.ASIAN_MONTE_CARLO
        )
    if isinstance(position, DigitalPosition):
        return PortfolioPricingModel.DIGITAL_ANALYTICAL
    if isinstance(position, BarrierPosition):
        return PortfolioPricingModel.BARRIER_MONTE_CARLO
    return basket_default_model(position)


def _resolved_model(position: PortfolioPosition) -> PortfolioPricingModel:
    return position.model or _default_model(position)


def _deterministic_model(model: PortfolioPricingModel) -> bool:
    return model not in {
        PortfolioPricingModel.MONTE_CARLO_GBM,
        PortfolioPricingModel.LONGSTAFF_SCHWARTZ,
        PortfolioPricingModel.HESTON_MONTE_CARLO,
        PortfolioPricingModel.ASIAN_MONTE_CARLO,
        PortfolioPricingModel.DIGITAL_MONTE_CARLO,
        PortfolioPricingModel.BARRIER_MONTE_CARLO,
        PortfolioPricingModel.BASKET_MONTE_CARLO,
    }


def _cache_key(position: PortfolioPosition, model: PortfolioPricingModel) -> str:
    payload = position.model_dump_json(
        exclude={"position_id", "quantity", "metadata", "market_data_reference"},
        exclude_computed_fields=True,
    )
    return f"{model.value}:{payload}"


def _heston_parameters(position: VanillaPosition) -> HestonParameters:
    if position.heston_parameters is not None:
        return position.heston_parameters
    if position.heston_calibration_result is not None:
        return position.heston_calibration_result.calibrated_parameters
    raise PortfolioRiskError("Heston pricing requires parameters or a calibration result")


def _effective_vanilla_market(
    position: VanillaPosition, *, maturity_years: float | None = None
) -> MarketEnvironment:
    if not position.use_surface_volatility:
        return position.market
    assert position.volatility_surface is not None
    maturity = position.instrument.maturity_years if maturity_years is None else maturity_years
    if maturity == 0.0:
        return position.market
    query = query_surface(
        position.volatility_surface,
        maturity_years=maturity,
        strike=position.instrument.strike,
    )
    return position.market.model_copy(update={"volatility": query.implied_volatility})


def _require_model(position: PortfolioPosition, model: PortfolioPricingModel) -> None:
    if isinstance(position, VanillaPosition):
        try:
            registered_model = PricingModel(model.value)
        except ValueError as exc:
            raise PortfolioRiskError(
                f"model {model.value} is incompatible with {position.instrument_class.value}"
            ) from exc
        exclusion = compatibility_exclusion(
            registered_model,
            position.instrument,
            heston_parameters_available=(
                position.heston_parameters is not None
                or position.heston_calibration_result is not None
            ),
        )
        if exclusion is not None:
            raise PortfolioRiskError(exclusion.reason)
        return
    elif isinstance(position, AsianPosition):
        allowed = {
            PortfolioPricingModel.ASIAN_MONTE_CARLO,
            PortfolioPricingModel.GEOMETRIC_ASIAN_ANALYTICAL,
        }
    elif isinstance(position, DigitalPosition):
        allowed = {
            PortfolioPricingModel.DIGITAL_ANALYTICAL,
            PortfolioPricingModel.DIGITAL_MONTE_CARLO,
        }
    elif isinstance(position, BarrierPosition):
        allowed = {
            PortfolioPricingModel.BARRIER_ANALYTICAL,
            PortfolioPricingModel.BARRIER_MONTE_CARLO,
        }
    else:
        allowed = {
            PortfolioPricingModel.GEOMETRIC_BASKET_ANALYTICAL,
            PortfolioPricingModel.BASKET_MONTE_CARLO,
        }
    if model not in allowed:
        raise PortfolioRiskError(
            f"model {model.value} is incompatible with {position.instrument_class.value}"
        )


def _price_position(
    position: PortfolioPosition,
    root_seed: int,
    *,
    model_override: PortfolioPricingModel | None = None,
) -> _Price:
    model = model_override or _resolved_model(position)
    _require_model(position, model)
    child_seed = _seed(root_seed, position.position_id, model)

    if isinstance(position, VanillaPosition):
        contract, market = position.instrument, _effective_vanilla_market(position)
        if model is PortfolioPricingModel.BLACK_SCHOLES:
            return _Price(black_scholes_price(contract, market), model)
        if model is PortfolioPricingModel.MONTE_CARLO_GBM:
            mc_result = MonteCarloPricer().price(
                contract, market, position.simulation.model_copy(update={"seed": child_seed})
            )
            return _Price(
                mc_result.price,
                model,
                mc_result.standard_error,
                mc_result.confidence_interval_low,
                mc_result.confidence_interval_high,
            )
        if model in {PortfolioPricingModel.CRR_BINOMIAL, PortfolioPricingModel.CRR_AMERICAN}:
            tree_result = binomial_price(contract, market, position.tree)
            return _Price(tree_result.price, model)
        if model is PortfolioPricingModel.LONGSTAFF_SCHWARTZ:
            lsm_result = longstaff_schwartz_price(
                contract,
                market,
                position.longstaff_schwartz.model_copy(update={"seed": child_seed}),
            )
            return _Price(
                lsm_result.price,
                model,
                lsm_result.standard_error,
                lsm_result.confidence_interval_low,
                lsm_result.confidence_interval_high,
            )
        parameters = _heston_parameters(position)
        if model is PortfolioPricingModel.HESTON_ANALYTICAL:
            heston_analytical_result = heston_analytical_price(
                contract, market, parameters, position.heston_integration
            )
            return _Price(heston_analytical_result.price, model)
        heston_mc_result = heston_monte_carlo_price(
            contract,
            market,
            parameters,
            position.heston_simulation.model_copy(update={"seed": child_seed}),
        )
        return _Price(
            heston_mc_result.price,
            model,
            heston_mc_result.standard_error,
            heston_mc_result.confidence_interval_low,
            heston_mc_result.confidence_interval_high,
        )

    if isinstance(position, AsianPosition):
        if model is PortfolioPricingModel.GEOMETRIC_ASIAN_ANALYTICAL:
            if position.instrument.averaging_type is not AveragingType.GEOMETRIC:
                raise PortfolioRiskError("geometric Asian analytical model requires GEOMETRIC")
            geometric_result = geometric_asian_analytical_price(
                position.instrument, position.market
            )
            return _Price(geometric_result.price, model)
        asian_result = asian_monte_carlo_price(
            position.instrument,
            position.market,
            position.simulation.model_copy(update={"seed": child_seed}),
        )
        return _Price(
            asian_result.price,
            model,
            asian_result.standard_error,
            asian_result.confidence_interval_low,
            asian_result.confidence_interval_high,
        )

    if isinstance(position, DigitalPosition):
        if model is PortfolioPricingModel.DIGITAL_ANALYTICAL:
            digital_result = digital_analytical_price(position.instrument, position.market)
            return _Price(digital_result.price, model)
        digital_mc_result = digital_monte_carlo_price(
            position.instrument,
            position.market,
            position.simulation.model_copy(update={"seed": child_seed}),
        )
        return _Price(
            digital_mc_result.price,
            model,
            digital_mc_result.standard_error,
            digital_mc_result.confidence_interval_low,
            digital_mc_result.confidence_interval_high,
        )

    if isinstance(position, BarrierPosition):
        if model is PortfolioPricingModel.BARRIER_ANALYTICAL:
            if position.instrument.monitoring_mode is not BarrierMonitoringMode.BROWNIAN_BRIDGE:
                raise PortfolioRiskError(
                    "barrier analytical model requires BROWNIAN_BRIDGE continuous monitoring"
                )
            barrier_analytical_result = continuous_barrier_analytical_price(
                position.instrument, position.market
            )
            return _Price(barrier_analytical_result.price, model)
        barrier_result = barrier_monte_carlo_price(
            position.instrument,
            position.market,
            position.simulation.model_copy(update={"seed": child_seed}),
        )
        return _Price(
            barrier_result.price,
            model,
            barrier_result.standard_error,
            barrier_result.confidence_interval_low,
            barrier_result.confidence_interval_high,
        )

    if model is PortfolioPricingModel.GEOMETRIC_BASKET_ANALYTICAL:
        if position.instrument.basket_type is not BasketType.GEOMETRIC:
            raise PortfolioRiskError("geometric basket analytical model requires GEOMETRIC")
        basket_analytical_result = geometric_basket_analytical_price(
            position.instrument, position.market
        )
        return _Price(basket_analytical_result.price, model)
    basket_result = basket_monte_carlo_price(
        position.instrument,
        position.market,
        position.simulation.model_copy(update={"seed": child_seed}),
    )
    return _Price(
        basket_result.price,
        model,
        basket_result.standard_error,
        basket_result.confidence_interval_low,
        basket_result.confidence_interval_high,
    )


def _completeness(successes: int, failures: int) -> PortfolioCompleteness:
    if failures == 0:
        return PortfolioCompleteness.COMPLETE
    if successes == 0:
        return PortfolioCompleteness.FAILED
    return PortfolioCompleteness.PARTIAL


def price_portfolio(request: PortfolioRequest) -> PortfolioValuationResult:
    """Value each position independently and aggregate successful signed values."""

    started = perf_counter()
    results: list[PositionValuationResult] = []
    signed_values: list[tuple[float, float]] = []
    deterministic_cache: dict[str, _Price] = {}
    for position in request.portfolio.positions:
        model = _resolved_model(position)
        try:
            key = _cache_key(position, model)
            if _deterministic_model(model) and key in deterministic_cache:
                observation = deterministic_cache[key]
            else:
                observation = _price_position(position, request.config.root_seed)
                if _deterministic_model(model):
                    deterministic_cache[key] = observation
            value = position.quantity * observation.value
            signed_values.append((position.quantity, value))
            results.append(
                PositionValuationResult(
                    position_id=position.position_id,
                    instrument_class=position.instrument_class,
                    model=observation.model,
                    quantity=position.quantity,
                    status=PositionStatus.SUCCESS,
                    unit_price=observation.value,
                    position_value=value,
                    standard_error=(
                        None
                        if observation.standard_error is None
                        else abs(position.quantity) * observation.standard_error
                    ),
                    confidence_interval_low=(
                        None
                        if observation.confidence_low is None
                        else min(
                            position.quantity * observation.confidence_low,
                            position.quantity * cast(float, observation.confidence_high),
                        )
                    ),
                    confidence_interval_high=(
                        None
                        if observation.confidence_high is None
                        else max(
                            position.quantity * cast(float, observation.confidence_low),
                            position.quantity * observation.confidence_high,
                        )
                    ),
                    warnings=list(observation.warnings),
                )
            )
        except Exception as exc:  # position failure isolation is an explicit policy
            results.append(
                PositionValuationResult(
                    position_id=position.position_id,
                    instrument_class=position.instrument_class,
                    model=model,
                    quantity=position.quantity,
                    status=PositionStatus.FAILED,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            )
    successes, failures = len(signed_values), len(results) - len(signed_values)
    if failures and not request.config.allow_partial:
        raise PortfolioRiskError("portfolio contains failed positions and allow_partial is false")
    total = math.fsum(value for _, value in signed_values) if signed_values else None
    gross_long = math.fsum(value for quantity, value in signed_values if quantity > 0.0)
    gross_short = math.fsum(-value for quantity, value in signed_values if quantity < 0.0)
    return PortfolioValuationResult(
        portfolio_id=request.portfolio.portfolio_id,
        base_currency=request.portfolio.base_currency,
        total_value=total,
        gross_long_value=gross_long,
        gross_short_value=gross_short,
        position_count=len(results),
        successful_position_count=successes,
        failed_position_count=failures,
        completeness=_completeness(successes, failures),
        positions=results,
        runtime_ms=(perf_counter() - started) * 1_000.0,
    )


def _replace_single(
    position: PositionT,
    *,
    spot: float | None = None,
    volatility: float | None = None,
    rate: float | None = None,
    dividend: float | None = None,
    maturity: float | None = None,
) -> PositionT:
    if isinstance(position, BasketPosition):
        raise PortfolioRiskError("single-asset replacement requested for basket")
    contract_updates: dict[str, float] = {}
    market_updates: dict[str, float] = {}
    if spot is not None:
        contract_updates["spot"] = spot
    if maturity is not None:
        contract_updates["maturity_years"] = maturity
    if volatility is not None:
        market_updates["volatility"] = volatility
    if rate is not None:
        market_updates["risk_free_rate"] = rate
    if dividend is not None:
        market_updates["dividend_yield"] = dividend
    return cast(
        PositionT,
        position.model_copy(
            update={
                "instrument": position.instrument.model_copy(update=contract_updates),
                "market": position.market.model_copy(update=market_updates),
            }
        ),
    )


def _finite_difference_greeks(
    position: PortfolioPosition, config: PortfolioAnalysisConfig
) -> tuple[GreekVector, dict[str, float], list[str]]:
    model = _resolved_model(position)
    if isinstance(position, VanillaPosition) and model is PortfolioPricingModel.BLACK_SCHOLES:
        analytical = black_scholes_greeks(position.instrument, _effective_vanilla_market(position))
        return (
            GreekVector(
                delta=analytical.delta,
                gamma=analytical.gamma,
                vega=analytical.vega,
                theta=analytical.theta,
                rho=analytical.rho,
            ),
            {},
            [],
        )
    if isinstance(position, VanillaPosition) and model is PortfolioPricingModel.MONTE_CARLO_GBM:
        simulation = position.simulation.model_copy(
            update={"seed": _seed(config.root_seed, position.position_id, model)}
        )
        mc_greeks = monte_carlo_greeks(
            position.instrument,
            position.market,
            simulation,
            [Greek.DELTA, Greek.GAMMA, Greek.VEGA, Greek.THETA, Greek.RHO],
            GreekMethod.FINITE_DIFFERENCE,
            GreekBumpConfig(),
        )
        values = {point.greek.value.lower(): point.estimate for point in mc_greeks.results}
        return GreekVector(**values), {}, []

    base = _price_position(position, config.root_seed).value
    warnings: list[str] = []
    unavailable: list[str] = []

    def price(item: PortfolioPosition) -> float:
        return _price_position(item, config.root_seed).value

    if isinstance(position, BasketPosition):
        assert position.market.asset_ids is not None
        per_asset: dict[str, float] = {}
        for index, identifier in enumerate(position.market.asset_ids):
            spot = position.market.spots[index]
            bump = max(spot * 1e-4, 1e-8)
            up_spots = list(position.market.spots)
            down_spots = list(position.market.spots)
            up_spots[index] += bump
            down_spots[index] -= bump
            up = position.model_copy(
                update={"market": position.market.model_copy(update={"spots": tuple(up_spots)})}
            )
            down = position.model_copy(
                update={"market": position.market.model_copy(update={"spots": tuple(down_spots)})}
            )
            per_asset[identifier] = (price(up) - price(down)) / (2.0 * bump)
        rate_bump = 1e-4
        up_rate = position.model_copy(
            update={
                "market": position.market.model_copy(
                    update={"risk_free_rate": position.market.risk_free_rate + rate_bump}
                )
            }
        )
        down_rate = position.model_copy(
            update={
                "market": position.market.model_copy(
                    update={"risk_free_rate": position.market.risk_free_rate - rate_bump}
                )
            }
        )
        rho = (price(up_rate) - price(down_rate)) / (2.0 * rate_bump)
        theta = _theta_difference(position, config, price)
        unavailable.extend(["delta", "gamma", "vega"])
        return GreekVector(theta=theta, rho=rho), per_asset, unavailable

    spot = position.instrument.spot
    spot_bump = max(spot * 1e-4, 1e-8)
    up_spot = _replace_single(position, spot=spot + spot_bump)
    down_spot = _replace_single(position, spot=spot - spot_bump)
    up_price, down_price = price(up_spot), price(down_spot)
    delta = (up_price - down_price) / (2.0 * spot_bump)
    gamma = (up_price - 2.0 * base + down_price) / (spot_bump * spot_bump)

    rate_bump = 1e-4
    rho = (
        price(_replace_single(position, rate=position.market.risk_free_rate + rate_bump))
        - price(_replace_single(position, rate=position.market.risk_free_rate - rate_bump))
    ) / (2.0 * rate_bump)
    theta = _theta_difference(position, config, price)

    if isinstance(position, VanillaPosition) and model in {
        PortfolioPricingModel.HESTON_ANALYTICAL,
        PortfolioPricingModel.HESTON_MONTE_CARLO,
    }:
        vega = None
        unavailable.append("vega")
        warnings.append("Heston has no single canonical volatility Vega; use parameter scenarios")
    else:
        vol_bump = 1e-4
        up_vol = position.market.volatility + vol_bump
        if position.market.volatility >= vol_bump:
            vega = (
                price(_replace_single(position, volatility=up_vol))
                - price(_replace_single(position, volatility=position.market.volatility - vol_bump))
            ) / (2.0 * vol_bump)
        else:
            vega = (price(_replace_single(position, volatility=up_vol)) - base) / vol_bump
    return GreekVector(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho), {}, (unavailable)


def _theta_difference(
    position: PortfolioPosition,
    config: PortfolioAnalysisConfig,
    price: Callable[[PortfolioPosition], float],
) -> float | None:
    maturity = position_maturity(position)
    if maturity == 0.0:
        return None
    bump = min(1.0 / 3650.0, maturity)
    try:
        longer = _replace_maturity(position, maturity + bump)
        if maturity > bump:
            shorter = _replace_maturity(position, maturity - bump)
            return -(price(longer) - price(shorter)) / (2.0 * bump)
        return -(price(longer) - price(position)) / bump
    except Exception:
        return None


def _replace_maturity(position: PositionT, maturity: float) -> PositionT:
    return cast(
        PositionT,
        position.model_copy(
            update={
                "instrument": position.instrument.model_copy(
                    update={"maturity_years": max(maturity, 0.0)}
                )
            }
        ),
    )


def _scaled(greeks: GreekVector, spot: float | None) -> ScaledGreekVector:
    return ScaledGreekVector(
        dollar_delta=(None if greeks.delta is None or spot is None else greeks.delta * spot),
        delta_1pct=(None if greeks.delta is None or spot is None else greeks.delta * spot * 0.01),
        gamma_1pct=(
            None
            if greeks.gamma is None or spot is None
            else 0.5 * greeks.gamma * (0.01 * spot) ** 2
        ),
        vega_1vol_point=None if greeks.vega is None else greeks.vega * 0.01,
        theta_1day=None if greeks.theta is None else greeks.theta / 365.0,
        rho_1pct=None if greeks.rho is None else greeks.rho * 0.01,
    )


def _multiply(greeks: GreekVector, quantity: float) -> GreekVector:
    return GreekVector(
        **{
            name: None if getattr(greeks, name) is None else quantity * getattr(greeks, name)
            for name in GREEK_NAMES
        }
    )


def _net_gross(vectors: list[GreekVector]) -> tuple[GreekVector, GreekVector, NetGrossGreeks]:
    net: dict[str, float | None] = {}
    gross: dict[str, float | None] = {}
    pairs: dict[str, NetGrossExposure] = {}
    for name in GREEK_NAMES:
        values = [
            cast(float, getattr(vector, name))
            for vector in vectors
            if getattr(vector, name) is not None
        ]
        net[name] = math.fsum(values) if values else None
        gross[name] = math.fsum(abs(value) for value in values) if values else None
        pairs[name] = NetGrossExposure(net=net[name], gross=gross[name])
    return GreekVector(**net), GreekVector(**gross), NetGrossGreeks(**pairs)


@dataclass
class _GroupAccumulator:
    market_value: float = 0.0
    position_ids: set[str] | None = None
    values: dict[str, list[float]] | None = None
    scaled: dict[str, list[float]] | None = None

    def __post_init__(self) -> None:
        self.position_ids = set()
        self.values = {name: [] for name in GREEK_NAMES}
        self.scaled = {
            name: []
            for name in (
                "dollar_delta",
                "delta_1pct",
                "gamma_1pct",
                "vega_1vol_point",
                "theta_1day",
                "rho_1pct",
            )
        }


def _group_results(
    portfolio: Portfolio,
    valuation: PortfolioValuationResult,
    greek_results: list[PositionGreeksResult],
    dimension: str,
) -> list[ExposureGroup]:
    valuation_by_id = {item.position_id: item for item in valuation.positions}
    greek_by_id = {item.position_id: item for item in greek_results}
    groups: dict[str, _GroupAccumulator] = {}

    def add(
        key: str,
        position_id: str,
        market_value: float,
        vector: GreekVector,
        scaled: ScaledGreekVector,
    ) -> None:
        acc = groups.setdefault(key, _GroupAccumulator())
        acc.market_value += market_value
        assert acc.position_ids is not None and acc.values is not None and acc.scaled is not None
        acc.position_ids.add(position_id)
        for name in GREEK_NAMES:
            value = getattr(vector, name)
            if value is not None:
                acc.values[name].append(value)
        for name in acc.scaled:
            value = getattr(scaled, name)
            if value is not None:
                acc.scaled[name].append(value)

    for position in portfolio.positions:
        value_result = valuation_by_id[position.position_id]
        greek_result = greek_by_id[position.position_id]
        if value_result.position_value is None:
            continue
        if dimension == "underlying" and isinstance(position, BasketPosition):
            assert position.market.asset_ids is not None
            for index, identifier in enumerate(position.market.asset_ids):
                delta = greek_result.per_asset_delta.get(identifier)
                delta_vector = GreekVector(delta=delta)
                delta_scaled = ScaledGreekVector(
                    dollar_delta=(None if delta is None else delta * position.market.spots[index]),
                    delta_1pct=(
                        None if delta is None else delta * position.market.spots[index] * 0.01
                    ),
                )
                add(
                    identifier,
                    position.position_id,
                    value_result.position_value * position.instrument.weights[index],
                    delta_vector,
                    delta_scaled,
                )
            continue
        if dimension == "underlying":
            key = cast(
                VanillaPosition | AsianPosition | DigitalPosition | BarrierPosition,
                position,
            ).underlying_id
        elif dimension == "expiry":
            key = f"{position_maturity(position):.12g}"
        elif dimension == "option_type":
            key = position_option_type(position).value
        elif dimension == "instrument_class":
            key = position.instrument_class.value
        else:
            key = greek_result.model.value
        add(
            key,
            position.position_id,
            value_result.position_value,
            greek_result.position_greeks,
            greek_result.scaled_position_greeks,
        )

    output: list[ExposureGroup] = []
    for key in sorted(groups):
        acc = groups[key]
        assert acc.position_ids is not None and acc.values is not None and acc.scaled is not None
        greek_values = {
            name: math.fsum(values) if values else None for name, values in acc.values.items()
        }
        scaled_values = {
            name: math.fsum(values) if values else None for name, values in acc.scaled.items()
        }
        output.append(
            ExposureGroup(
                key=key,
                market_value=acc.market_value,
                position_count=len(acc.position_ids),
                greeks=GreekVector(**greek_values),
                scaled_greeks=ScaledGreekVector(**scaled_values),
                missing_metrics=[name for name, values in acc.values.items() if not values],
            )
        )
    return output


def _largest(
    identifiers: Sequence[str], values: Sequence[float | None]
) -> ConcentrationContributor | None:
    available = [
        (identifier, abs(value))
        for identifier, value in zip(identifiers, values, strict=True)
        if value is not None
    ]
    if not available:
        return None
    identifier, amount = max(available, key=lambda item: (item[1], item[0]))
    return ConcentrationContributor(position_id=identifier, absolute_contribution=amount)


def portfolio_greeks(request: PortfolioRequest) -> PortfolioGreeksResult:
    """Compute position Greeks, canonical scaling, and deterministic exposure groupings."""

    started = perf_counter()
    valuation = price_portfolio(request)
    value_by_id = {item.position_id: item for item in valuation.positions}
    results: list[PositionGreeksResult] = []
    deterministic_greek_cache: dict[str, tuple[GreekVector, dict[str, float], list[str]]] = {}
    for position in request.portfolio.positions:
        model = _resolved_model(position)
        if value_by_id[position.position_id].status is PositionStatus.FAILED:
            results.append(
                PositionGreeksResult(
                    position_id=position.position_id,
                    status=PositionStatus.FAILED,
                    model=model,
                    quantity=position.quantity,
                    unit_greeks=GreekVector(),
                    position_greeks=GreekVector(),
                    scaled_position_greeks=ScaledGreekVector(),
                    unavailable_metrics=list(GREEK_NAMES),
                    warnings=["pricing failed; Greeks unavailable"],
                )
            )
            continue
        try:
            key = _cache_key(position, model)
            cached = deterministic_greek_cache.get(key) if _deterministic_model(model) else None
            if cached is None:
                unit, per_asset, unavailable = _finite_difference_greeks(position, request.config)
                if _deterministic_model(model):
                    deterministic_greek_cache[key] = (unit, per_asset, unavailable)
            else:
                unit, per_asset, unavailable = cached
            scaled_unit = _scaled(
                unit,
                None if isinstance(position, BasketPosition) else position.instrument.spot,
            )
            results.append(
                PositionGreeksResult(
                    position_id=position.position_id,
                    status=PositionStatus.SUCCESS,
                    model=model,
                    quantity=position.quantity,
                    unit_greeks=unit,
                    position_greeks=_multiply(unit, position.quantity),
                    scaled_position_greeks=ScaledGreekVector(
                        **{
                            name: (
                                None
                                if getattr(scaled_unit, name) is None
                                else position.quantity * getattr(scaled_unit, name)
                            )
                            for name in ScaledGreekVector.model_fields
                        }
                    ),
                    per_asset_delta={
                        key: position.quantity * value for key, value in sorted(per_asset.items())
                    },
                    unavailable_metrics=unavailable,
                    warnings=(
                        ["Digital payoff is discontinuous near strike"]
                        if isinstance(position, DigitalPosition)
                        else ["Barrier payoff is discontinuous near the barrier"]
                        if isinstance(position, BarrierPosition)
                        else []
                    ),
                )
            )
        except Exception as exc:
            if not request.config.allow_partial:
                raise PortfolioRiskError(
                    f"Greek evaluation failed for {position.position_id}"
                ) from exc
            results.append(
                PositionGreeksResult(
                    position_id=position.position_id,
                    status=PositionStatus.FAILED,
                    model=model,
                    quantity=position.quantity,
                    unit_greeks=GreekVector(),
                    position_greeks=GreekVector(),
                    scaled_position_greeks=ScaledGreekVector(),
                    unavailable_metrics=list(GREEK_NAMES),
                    warnings=[f"Greek evaluation failed: {type(exc).__name__}: {exc}"],
                )
            )

    vectors = [item.position_greeks for item in results if item.status is PositionStatus.SUCCESS]
    net, gross, pairs = _net_gross(vectors)
    scaled_vectors = [
        item.scaled_position_greeks for item in results if item.status is PositionStatus.SUCCESS
    ]
    scaled_net_values: dict[str, float | None] = {}
    scaled_gross_values: dict[str, float | None] = {}
    for name in ScaledGreekVector.model_fields:
        values = [
            cast(float, getattr(item, name))
            for item in scaled_vectors
            if getattr(item, name) is not None
        ]
        scaled_net_values[name] = math.fsum(values) if values else None
        scaled_gross_values[name] = math.fsum(abs(value) for value in values) if values else None

    by_underlying = _group_results(request.portfolio, valuation, results, "underlying")
    identifiers = [item.position_id for item in results]
    market_values = [value_by_id[identifier].position_value for identifier in identifiers]
    delta_abs = [
        abs(group.greeks.delta) for group in by_underlying if group.greeks.delta is not None
    ]
    delta_total = math.fsum(delta_abs)
    hhi = (
        math.fsum((value / delta_total) ** 2 for value in delta_abs) if delta_total > 0.0 else None
    )
    concentration = ConcentrationMetrics(
        largest_absolute_delta=_largest(
            identifiers, [item.position_greeks.delta for item in results]
        ),
        largest_absolute_vega=_largest(
            identifiers, [item.position_greeks.vega for item in results]
        ),
        largest_absolute_market_value=_largest(identifiers, market_values),
        underlying_delta_hhi=hhi,
    )
    flags: list[PortfolioRiskFlag] = []
    if (
        request.config.large_net_delta_threshold is not None
        and net.delta is not None
        and abs(net.delta) > request.config.large_net_delta_threshold
    ):
        flags.append(PortfolioRiskFlag.LARGE_NET_DELTA)
    if (
        request.config.large_gross_vega_threshold is not None
        and gross.vega is not None
        and gross.vega > request.config.large_gross_vega_threshold
    ):
        flags.append(PortfolioRiskFlag.LARGE_GROSS_VEGA)
    if (
        request.config.underlying_concentration_hhi_threshold is not None
        and hhi is not None
        and hhi > request.config.underlying_concentration_hhi_threshold
    ):
        flags.append(PortfolioRiskFlag.CONCENTRATED_UNDERLYING_EXPOSURE)
    unavailable = [
        name
        for name in GREEK_NAMES
        if any(getattr(item.position_greeks, name) is None for item in results)
    ]
    return PortfolioGreeksResult(
        portfolio_id=request.portfolio.portfolio_id,
        valuation=valuation,
        net_greeks=net,
        gross_greeks=gross,
        net_gross=pairs,
        scaled_net_greeks=ScaledGreekVector(**scaled_net_values),
        scaled_gross_greeks=ScaledGreekVector(**scaled_gross_values),
        positions=results,
        by_underlying=by_underlying,
        by_expiry=_group_results(request.portfolio, valuation, results, "expiry"),
        by_option_type=_group_results(request.portfolio, valuation, results, "option_type"),
        by_instrument_class=_group_results(
            request.portfolio, valuation, results, "instrument_class"
        ),
        by_model=_group_results(request.portfolio, valuation, results, "model"),
        concentration=concentration,
        unavailable_metrics=unavailable,
        risk_flags=flags,
        runtime_ms=(perf_counter() - started) * 1_000.0,
    )


def _apply_heston_scenario(
    parameters: HestonParameters, position_id: str, scenario: MarketScenario
) -> HestonParameters:
    alternate = scenario.heston_parameters_by_position.get(position_id)
    if alternate is not None:
        return alternate
    if scenario.heston is None:
        return parameters
    if scenario.heston.overrides is not None:
        return scenario.heston.overrides
    updates = {
        "v0": parameters.v0 + (scenario.heston.v0_shift or 0.0),
        "theta": parameters.theta + (scenario.heston.theta_shift or 0.0),
        "kappa": parameters.kappa + (scenario.heston.kappa_shift or 0.0),
        "sigma_v": parameters.sigma_v + (scenario.heston.sigma_v_shift or 0.0),
        "rho": parameters.rho + (scenario.heston.rho_shift or 0.0),
    }
    return HestonParameters.model_validate(updates)


def _scenario_single(position: PositionT, scenario: MarketScenario) -> PositionT:
    assert not isinstance(position, BasketPosition)
    relative = scenario.per_underlying_spot_relative_shifts.get(
        position.underlying_id, scenario.spot_relative_shift or 0.0
    )
    absolute = scenario.per_underlying_spot_absolute_shifts.get(
        position.underlying_id, scenario.spot_absolute_shift or 0.0
    )
    spot = position.instrument.spot * (1.0 + relative) + absolute
    if spot <= 0.0:
        raise PortfolioRiskError(f"scenario makes spot nonpositive for {position.position_id}")
    maturity = max(position.instrument.maturity_years - (scenario.time_shift_years or 0.0), 0.0)
    model = _resolved_model(position)
    is_heston = model in {
        PortfolioPricingModel.HESTON_ANALYTICAL,
        PortfolioPricingModel.HESTON_MONTE_CARLO,
    }
    volatility = (
        _effective_vanilla_market(position, maturity_years=maturity).volatility
        if isinstance(position, VanillaPosition)
        else position.market.volatility
    )
    if not is_heston:
        volatility = volatility * (1.0 + (scenario.volatility_relative_shift or 0.0)) + (
            scenario.volatility_absolute_shift or 0.0
        )
        if volatility < 0.0:
            raise PortfolioRiskError(
                f"scenario makes volatility negative for {position.position_id}"
            )
    updated_position: PortfolioPosition = _replace_single(
        position,
        spot=spot,
        volatility=volatility,
        rate=position.market.risk_free_rate + (scenario.rate_absolute_shift or 0.0),
        dividend=position.market.dividend_yield + (scenario.dividend_yield_shift or 0.0),
        maturity=maturity,
    )
    if isinstance(updated_position, VanillaPosition) and updated_position.use_surface_volatility:
        updated_position = updated_position.model_copy(update={"use_surface_volatility": False})
    if isinstance(updated_position, VanillaPosition) and is_heston:
        parameters = _apply_heston_scenario(
            _heston_parameters(updated_position), updated_position.position_id, scenario
        )
        return cast(
            PositionT,
            updated_position.model_copy(update={"heston_parameters": parameters}),
        )
    return cast(PositionT, updated_position)


def _scenario_basket(position: BasketPosition, scenario: MarketScenario) -> BasketPosition:
    assert position.market.asset_ids is not None
    spots: list[float] = []
    for identifier, spot in zip(position.market.asset_ids, position.market.spots, strict=True):
        relative = scenario.per_underlying_spot_relative_shifts.get(
            identifier, scenario.spot_relative_shift or 0.0
        )
        absolute = scenario.per_underlying_spot_absolute_shifts.get(
            identifier, scenario.spot_absolute_shift or 0.0
        )
        shocked = spot * (1.0 + relative) + absolute
        if shocked <= 0.0:
            raise PortfolioRiskError(
                f"scenario makes basket spot nonpositive for asset {identifier}"
            )
        spots.append(shocked)
    volatilities = tuple(
        volatility * (1.0 + (scenario.volatility_relative_shift or 0.0))
        + (scenario.volatility_absolute_shift or 0.0)
        for volatility in position.market.volatilities
    )
    if any(volatility < 0.0 for volatility in volatilities):
        raise PortfolioRiskError("scenario makes a basket volatility negative")
    matrix = position.market.correlation_matrix
    if position.position_id in scenario.correlation_matrix_by_position:
        matrix = scenario.correlation_matrix_by_position[position.position_id]
    if position.position_id in scenario.equicorrelation_shift_by_position:
        shift = scenario.equicorrelation_shift_by_position[position.position_id]
        matrix = tuple(
            tuple(value if row == column else value + shift for column, value in enumerate(line))
            for row, line in enumerate(matrix)
        )
    prepared = prepare_correlation_matrix(matrix)
    if prepared.repaired:
        raise PortfolioRiskError("scenario correlation matrix requires PSD repair")
    effective_matrix = tuple(tuple(float(value) for value in row) for row in prepared.matrix)
    market = MultiAssetMarketEnvironment(
        spots=tuple(spots),
        risk_free_rate=position.market.risk_free_rate + (scenario.rate_absolute_shift or 0.0),
        dividend_yields=tuple(
            value + (scenario.dividend_yield_shift or 0.0)
            for value in position.market.dividend_yields
        ),
        volatilities=volatilities,
        correlation_matrix=effective_matrix,
        asset_ids=position.market.asset_ids,
    )
    instrument = position.instrument.model_copy(
        update={
            "maturity_years": max(
                position.instrument.maturity_years - (scenario.time_shift_years or 0.0), 0.0
            )
        }
    )
    return position.model_copy(update={"instrument": instrument, "market": market})


def apply_scenario(position: PositionT, scenario: MarketScenario) -> PositionT:
    if isinstance(position, BasketPosition):
        return cast(PositionT, _scenario_basket(position, scenario))
    return _scenario_single(position, scenario)


def _market_snapshot(position: PortfolioPosition) -> PositionMarketSnapshot:
    if isinstance(position, BasketPosition):
        assert position.market.asset_ids is not None
        identifiers = position.market.asset_ids
        return PositionMarketSnapshot(
            spots=dict(zip(identifiers, position.market.spots, strict=True)),
            volatilities=dict(zip(identifiers, position.market.volatilities, strict=True)),
            risk_free_rate=position.market.risk_free_rate,
            dividend_yields=dict(zip(identifiers, position.market.dividend_yields, strict=True)),
            maturity_years=position.instrument.maturity_years,
            correlation_matrix=position.market.correlation_matrix,
        )
    market = (
        _effective_vanilla_market(position)
        if isinstance(position, VanillaPosition)
        else position.market
    )
    heston_parameters = None
    if isinstance(position, VanillaPosition):
        heston_parameters = position.heston_parameters
        if heston_parameters is None and position.heston_calibration_result is not None:
            heston_parameters = position.heston_calibration_result.calibrated_parameters
    return PositionMarketSnapshot(
        spots={position.underlying_id: position.instrument.spot},
        volatilities={position.underlying_id: market.volatility},
        risk_free_rate=market.risk_free_rate,
        dividend_yields={position.underlying_id: market.dividend_yield},
        maturity_years=position.instrument.maturity_years,
        heston_parameters=heston_parameters,
    )


def _safe_market_snapshot(position: PortfolioPosition) -> PositionMarketSnapshot | None:
    try:
        return _market_snapshot(position)
    except Exception:
        return None


def _position_approximation(
    position: PortfolioPosition,
    result: PositionGreeksResult,
    scenario: MarketScenario,
) -> tuple[float | None, list[ScenarioFlag]]:
    flags: list[ScenarioFlag] = []
    greeks = result.position_greeks
    terms: list[float] = []
    if isinstance(position, BasketPosition):
        assert position.market.asset_ids is not None
        for identifier, spot in zip(position.market.asset_ids, position.market.spots, strict=True):
            relative = scenario.per_underlying_spot_relative_shifts.get(
                identifier, scenario.spot_relative_shift or 0.0
            )
            absolute = scenario.per_underlying_spot_absolute_shifts.get(
                identifier, scenario.spot_absolute_shift or 0.0
            )
            delta = result.per_asset_delta.get(identifier)
            if (relative != 0.0 or absolute != 0.0) and delta is None:
                flags.append(ScenarioFlag.MISSING_GREEKS)
                return None, flags
            if delta is not None:
                terms.append(delta * (spot * relative + absolute))
    else:
        relative = scenario.per_underlying_spot_relative_shifts.get(
            position.underlying_id, scenario.spot_relative_shift or 0.0
        )
        absolute = scenario.per_underlying_spot_absolute_shifts.get(
            position.underlying_id, scenario.spot_absolute_shift or 0.0
        )
        d_spot = position.instrument.spot * relative + absolute
        if d_spot != 0.0:
            if greeks.delta is None or greeks.gamma is None:
                flags.append(ScenarioFlag.MISSING_GREEKS)
                return None, flags
            terms.extend([greeks.delta * d_spot, 0.5 * greeks.gamma * d_spot * d_spot])

    model = _resolved_model(position)
    is_heston = model in {
        PortfolioPricingModel.HESTON_ANALYTICAL,
        PortfolioPricingModel.HESTON_MONTE_CARLO,
    }
    d_vol = 0.0
    if not is_heston:
        if isinstance(position, BasketPosition):
            if scenario.volatility_absolute_shift or scenario.volatility_relative_shift:
                flags.append(ScenarioFlag.MISSING_GREEKS)
                return None, flags
        else:
            d_vol = (scenario.volatility_absolute_shift or 0.0) + (
                position.market.volatility * (scenario.volatility_relative_shift or 0.0)
            )
    if d_vol != 0.0:
        if greeks.vega is None:
            flags.append(ScenarioFlag.MISSING_GREEKS)
            return None, flags
        terms.append(greeks.vega * d_vol)
    if scenario.rate_absolute_shift:
        if greeks.rho is None:
            flags.append(ScenarioFlag.MISSING_GREEKS)
            return None, flags
        terms.append(greeks.rho * scenario.rate_absolute_shift)
    if scenario.time_shift_years:
        if greeks.theta is None:
            flags.append(ScenarioFlag.MISSING_GREEKS)
            return None, flags
        terms.append(greeks.theta * scenario.time_shift_years)
    if scenario.dividend_yield_shift:
        flags.append(ScenarioFlag.MISSING_GREEKS)
        return None, flags
    if scenario.heston is not None or scenario.heston_parameters_by_position:
        flags.append(ScenarioFlag.MISSING_GREEKS)
        return None, flags
    if scenario.correlation_matrix_by_position or scenario.equicorrelation_shift_by_position:
        flags.append(ScenarioFlag.MISSING_GREEKS)
        return None, flags
    return math.fsum(terms), flags


def _scenario_group(
    portfolio: Portfolio,
    contributions: list[PositionScenarioContribution],
    dimension: str,
) -> list[ScenarioContributionGroup]:
    positions = {position.position_id: position for position in portfolio.positions}
    values: dict[str, list[float]] = {}
    identifiers: dict[str, set[str]] = {}
    for contribution in contributions:
        if contribution.full_repricing_pnl is None:
            continue
        position = positions[contribution.position_id]
        if dimension == "instrument_class":
            keys = [(position.instrument_class.value, contribution.full_repricing_pnl)]
        elif isinstance(position, BasketPosition):
            assert position.market.asset_ids is not None
            keys = [
                (identifier, contribution.full_repricing_pnl * weight)
                for identifier, weight in zip(
                    position.market.asset_ids, position.instrument.weights, strict=True
                )
            ]
        else:
            keys = [(position.underlying_id, contribution.full_repricing_pnl)]
        for key, value in keys:
            values.setdefault(key, []).append(value)
            identifiers.setdefault(key, set()).add(position.position_id)
    return [
        ScenarioContributionGroup(
            key=key, pnl=math.fsum(values[key]), position_count=len(identifiers[key])
        )
        for key in sorted(values)
    ]


def run_scenario(
    request: PortfolioScenarioRequest,
    *,
    _base: PortfolioValuationResult | None = None,
    _greeks: PortfolioGreeksResult | None = None,
) -> PortfolioScenarioResult:
    """Fully reprice a shocked portfolio and compare it with a local Greek approximation."""

    started = perf_counter()
    base_request = PortfolioRequest(portfolio=request.portfolio, config=request.config)
    base = _base or price_portfolio(base_request)
    greeks = _greeks or portfolio_greeks(base_request)
    base_by_id = {item.position_id: item for item in base.positions}
    greek_by_id = {item.position_id: item for item in greeks.positions}
    contributions: list[PositionScenarioContribution] = []
    deterministic_scenario_cache: dict[str, _Price] = {}
    for position in request.portfolio.positions:
        base_item = base_by_id[position.position_id]
        flags: list[ScenarioFlag] = []
        if isinstance(position, (DigitalPosition, BarrierPosition)):
            flags.append(ScenarioFlag.DISCONTINUOUS_PAYOFF)
        try:
            shocked = apply_scenario(position, request.scenario)
            model = _resolved_model(shocked)
            key = _cache_key(shocked, model)
            if _deterministic_model(model) and key in deterministic_scenario_cache:
                shocked_price = deterministic_scenario_cache[key]
            else:
                shocked_price = _price_position(shocked, request.config.root_seed)
                if _deterministic_model(model):
                    deterministic_scenario_cache[key] = shocked_price
            scenario_value = shocked.quantity * shocked_price.value
            full_pnl = (
                None
                if base_item.position_value is None
                else scenario_value - base_item.position_value
            )
            approximation, approximation_flags = _position_approximation(
                position, greek_by_id[position.position_id], request.scenario
            )
            flags.extend(approximation_flags)
            residual = (
                None if full_pnl is None or approximation is None else full_pnl - approximation
            )
            if residual is None or full_pnl is None or full_pnl == 0.0:
                relative = None
            else:
                relative = abs(residual) / max(abs(full_pnl), 1e-12)
            if relative is not None and relative > request.config.approximation_error_threshold:
                flags.append(ScenarioFlag.LARGE_GREEK_APPROXIMATION_ERROR)
            base_se = base_item.standard_error
            scenario_se = (
                None
                if shocked_price.standard_error is None
                else abs(shocked.quantity) * shocked_price.standard_error
            )
            warnings: list[str] = []
            model = _resolved_model(position)
            if model in {
                PortfolioPricingModel.HESTON_ANALYTICAL,
                PortfolioPricingModel.HESTON_MONTE_CARLO,
            } and (
                request.scenario.volatility_absolute_shift
                or request.scenario.volatility_relative_shift
            ):
                warnings.append(
                    "generic volatility shifts do not alter Heston parameters; use heston shifts"
                )
            if base_se is not None:
                warnings.append(
                    "CRN is used by stable per-position seeds; paired P&L SE is unavailable"
                )
            if (
                full_pnl is not None
                and (base_se is not None or scenario_se is not None)
                and max(base_se or 0.0, scenario_se or 0.0)
                > request.config.mc_noise_materiality_threshold * max(abs(full_pnl), 1e-12)
            ):
                flags.append(ScenarioFlag.MC_NOISE_MATERIAL)
            contributions.append(
                PositionScenarioContribution(
                    position_id=position.position_id,
                    base_value=base_item.position_value,
                    scenario_value=scenario_value,
                    full_repricing_pnl=full_pnl,
                    greek_approximation_pnl=approximation,
                    approximation_residual=residual,
                    relative_approximation_error=relative,
                    base_standard_error=base_se,
                    scenario_standard_error=scenario_se,
                    pnl_standard_error=None,
                    base_market=_safe_market_snapshot(position),
                    scenario_market=_safe_market_snapshot(shocked),
                    flags=list(dict.fromkeys(flags)),
                    warnings=warnings,
                )
            )
        except Exception as exc:
            contributions.append(
                PositionScenarioContribution(
                    position_id=position.position_id,
                    base_value=base_item.position_value,
                    scenario_value=None,
                    full_repricing_pnl=None,
                    greek_approximation_pnl=None,
                    approximation_residual=None,
                    relative_approximation_error=None,
                    base_market=_safe_market_snapshot(position),
                    scenario_market=None,
                    flags=list(dict.fromkeys([*flags, ScenarioFlag.MISSING_GREEKS])),
                    warnings=[f"scenario repricing failed: {type(exc).__name__}: {exc}"],
                )
            )
    successful = [item for item in contributions if item.full_repricing_pnl is not None]
    failures = len(contributions) - len(successful)
    if failures and not request.config.allow_partial:
        raise PortfolioRiskError("scenario contains failed positions and allow_partial is false")
    full = (
        math.fsum(cast(float, item.full_repricing_pnl) for item in successful)
        if successful
        else None
    )
    portfolio_scenario_value = (
        math.fsum(cast(float, item.scenario_value) for item in successful) if successful else None
    )
    comparable_base_value = (
        math.fsum(cast(float, item.base_value) for item in successful) if successful else None
    )
    approximations = [item.greek_approximation_pnl for item in successful]
    approximate = (
        math.fsum(cast(float, value) for value in approximations)
        if successful and all(value is not None for value in approximations)
        else None
    )
    residual = None if full is None or approximate is None else full - approximate
    if residual is None or full is None or full == 0.0:
        relative = None
    else:
        relative = abs(residual) / max(abs(full), 1e-12)
    all_flags = list(dict.fromkeys(flag for item in contributions for flag in item.flags))
    return PortfolioScenarioResult(
        portfolio_id=request.portfolio.portfolio_id,
        scenario=request.scenario,
        base_value=comparable_base_value,
        scenario_value=portfolio_scenario_value,
        full_repricing_pnl=full,
        greek_approximation_pnl=approximate,
        approximation_residual=residual,
        relative_approximation_error=relative,
        contributions=contributions,
        by_underlying=_scenario_group(request.portfolio, contributions, "underlying"),
        by_instrument_class=_scenario_group(request.portfolio, contributions, "instrument_class"),
        completeness=_completeness(len(successful), failures),
        flags=all_flags,
        runtime_ms=(perf_counter() - started) * 1_000.0,
    )


def _axis_scenario(axis: ScenarioAxis, value: float) -> dict[str, float]:
    mapping = {
        ScenarioAxis.SPOT_RELATIVE: "spot_relative_shift",
        ScenarioAxis.SPOT_ABSOLUTE: "spot_absolute_shift",
        ScenarioAxis.VOLATILITY_ABSOLUTE: "volatility_absolute_shift",
        ScenarioAxis.VOLATILITY_RELATIVE: "volatility_relative_shift",
        ScenarioAxis.RATE_ABSOLUTE: "rate_absolute_shift",
        ScenarioAxis.DIVIDEND_YIELD_ABSOLUTE: "dividend_yield_shift",
        ScenarioAxis.TIME_YEARS: "time_shift_years",
    }
    return {mapping[axis]: value}


def run_scenario_grid(request: ScenarioGridRequest) -> ScenarioGridResult:
    """Evaluate a bounded N-dimensional Cartesian grid in stable product order."""

    started = perf_counter()
    cells: list[ScenarioGridCell] = []
    base_request = PortfolioRequest(portfolio=request.portfolio, config=request.config)
    base = price_portfolio(base_request)
    greeks = portfolio_greeks(base_request)
    for coordinates in itertools.product(*(axis.values for axis in request.axes)):
        updates: dict[str, float] = {}
        labels: dict[str, float] = {}
        for axis, value in zip(request.axes, coordinates, strict=True):
            updates.update(_axis_scenario(axis.axis, value))
            labels[axis.axis.value] = value
        scenario = MarketScenario(**updates)
        result = run_scenario(
            PortfolioScenarioRequest(
                portfolio=request.portfolio, config=request.config, scenario=scenario
            ),
            _base=base,
            _greeks=greeks,
        )
        cells.append(
            ScenarioGridCell(
                coordinates=labels,
                scenario=scenario,
                portfolio_value=result.scenario_value,
                full_repricing_pnl=result.full_repricing_pnl,
                greek_approximation_pnl=result.greek_approximation_pnl,
                approximation_residual=result.approximation_residual,
                completeness=result.completeness,
            )
        )
    return ScenarioGridResult(
        portfolio_id=request.portfolio.portfolio_id,
        axes=request.axes,
        shape=[len(axis.values) for axis in request.axes],
        cells=cells,
        runtime_ms=(perf_counter() - started) * 1_000.0,
    )


def run_stress_scenarios(request: PortfolioStressRequest) -> PortfolioStressResult:
    started = perf_counter()
    base_request = PortfolioRequest(portfolio=request.portfolio, config=request.config)
    base = price_portfolio(base_request)
    greeks = portfolio_greeks(base_request)
    results = [
        StressScenarioResult(
            name=cast(str, scenario.name),
            result=run_scenario(
                PortfolioScenarioRequest(
                    portfolio=request.portfolio, config=request.config, scenario=scenario
                ),
                _base=base,
                _greeks=greeks,
            ),
        )
        for scenario in request.scenarios
    ]
    return PortfolioStressResult(
        portfolio_id=request.portfolio.portfolio_id,
        scenarios=results,
        runtime_ms=(perf_counter() - started) * 1_000.0,
    )


def _explain_terms(
    position: PortfolioPosition,
    greek_result: PositionGreeksResult,
    scenario: MarketScenario,
    full_pnl: float | None,
) -> PnLExplainTerms:
    greeks = greek_result.position_greeks
    if isinstance(position, BasketPosition):
        delta_terms: list[float] = []
        has_spot_shock = False
        assert position.market.asset_ids is not None
        for identifier, spot in zip(position.market.asset_ids, position.market.spots, strict=True):
            delta = greek_result.per_asset_delta.get(identifier)
            relative = scenario.per_underlying_spot_relative_shifts.get(
                identifier, scenario.spot_relative_shift or 0.0
            )
            absolute = scenario.per_underlying_spot_absolute_shifts.get(
                identifier, scenario.spot_absolute_shift or 0.0
            )
            has_spot_shock = has_spot_shock or relative != 0.0 or absolute != 0.0
            if delta is not None:
                delta_terms.append(delta * (spot * relative + absolute))
        delta_term = (
            math.fsum(delta_terms) if delta_terms else (0.0 if not has_spot_shock else None)
        )
        gamma_term = None if has_spot_shock else 0.0
        has_vol_shock = bool(
            scenario.volatility_relative_shift or scenario.volatility_absolute_shift
        )
        vega_term = None if has_vol_shock else 0.0
    else:
        relative = scenario.per_underlying_spot_relative_shifts.get(
            position.underlying_id, scenario.spot_relative_shift or 0.0
        )
        d_spot = position.instrument.spot * relative + (
            scenario.per_underlying_spot_absolute_shifts.get(
                position.underlying_id, scenario.spot_absolute_shift or 0.0
            )
        )
        delta_term = (
            0.0 if d_spot == 0.0 else None if greeks.delta is None else greeks.delta * d_spot
        )
        gamma_term = (
            0.0
            if d_spot == 0.0
            else None
            if greeks.gamma is None
            else 0.5 * greeks.gamma * d_spot * d_spot
        )
        d_vol = position.market.volatility * (scenario.volatility_relative_shift or 0.0) + (
            scenario.volatility_absolute_shift or 0.0
        )
        vega_term = 0.0 if d_vol == 0.0 else None if greeks.vega is None else greeks.vega * d_vol
    d_time = scenario.time_shift_years or 0.0
    theta_term = 0.0 if d_time == 0.0 else None if greeks.theta is None else greeks.theta * d_time
    d_rate = scenario.rate_absolute_shift or 0.0
    rho_term = 0.0 if d_rate == 0.0 else None if greeks.rho is None else greeks.rho * d_rate
    relevant = [delta_term, gamma_term, vega_term, theta_term, rho_term]
    explained = (
        math.fsum(cast(float, value) for value in relevant)
        if all(value is not None for value in relevant)
        else None
    )
    residual = None if full_pnl is None or explained is None else full_pnl - explained
    return PnLExplainTerms(
        delta=delta_term,
        gamma=gamma_term,
        vega=vega_term,
        theta=theta_term,
        rho=rho_term,
        explained_total=explained,
        unexplained_or_higher_order=residual,
    )


def explain_portfolio_pnl(request: PortfolioPnLExplainRequest) -> PortfolioPnLExplainResult:
    """Decompose full-repricing P&L into local Greek terms and an exact residual."""

    started = perf_counter()
    scenario = run_scenario(
        PortfolioScenarioRequest(
            portfolio=request.portfolio, config=request.config, scenario=request.scenario
        )
    )
    greeks = portfolio_greeks(PortfolioRequest(portfolio=request.portfolio, config=request.config))
    greek_by_id = {item.position_id: item for item in greeks.positions}
    contribution_by_id = {item.position_id: item for item in scenario.contributions}
    positions: list[PositionPnLExplain] = []
    for position in request.portfolio.positions:
        contribution = contribution_by_id[position.position_id]
        positions.append(
            PositionPnLExplain(
                position_id=position.position_id,
                full_reprice_pnl=contribution.full_repricing_pnl,
                terms=_explain_terms(
                    position,
                    greek_by_id[position.position_id],
                    request.scenario,
                    contribution.full_repricing_pnl,
                ),
                base_market=contribution.base_market,
                scenario_market=contribution.scenario_market,
            )
        )
    aggregate: dict[str, float | None] = {}
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        values = [getattr(item.terms, name) for item in positions]
        aggregate[name] = (
            math.fsum(cast(float, value) for value in values)
            if values and all(value is not None for value in values)
            else None
        )
    explained_values = list(aggregate.values())
    explained = (
        math.fsum(cast(float, value) for value in explained_values)
        if all(value is not None for value in explained_values)
        else None
    )
    residual = (
        None
        if scenario.full_repricing_pnl is None or explained is None
        else scenario.full_repricing_pnl - explained
    )
    terms = PnLExplainTerms(
        **aggregate,
        explained_total=explained,
        unexplained_or_higher_order=residual,
    )
    return PortfolioPnLExplainResult(
        portfolio_id=request.portfolio.portfolio_id,
        scenario=request.scenario,
        total_full_reprice_pnl=scenario.full_repricing_pnl,
        terms=terms,
        positions=positions,
        completeness=scenario.completeness,
        runtime_ms=(perf_counter() - started) * 1_000.0,
    )


def _portfolio_with_model(portfolio: Portfolio, model: PortfolioPricingModel) -> Portfolio:
    return portfolio.model_copy(
        update={
            "positions": [
                position.model_copy(update={"model": model}) for position in portfolio.positions
            ]
        }
    )


def portfolio_model_risk(request: PortfolioModelRiskRequest) -> PortfolioModelRiskResult:
    """Revalue compatible positions under alternate models and report descriptive spread."""

    started = perf_counter()
    reference = price_portfolio(
        PortfolioRequest(
            portfolio=_portfolio_with_model(request.portfolio, request.reference_model),
            config=request.config,
        )
    )
    reference_by_id = {item.position_id: item for item in reference.positions}
    alternatives: list[PortfolioModelRiskPoint] = []
    for model in request.alternate_models:
        alternate = price_portfolio(
            PortfolioRequest(
                portfolio=_portfolio_with_model(request.portfolio, model), config=request.config
            )
        )
        contributions: list[PortfolioModelRiskPosition] = []
        for item in alternate.positions:
            base = reference_by_id[item.position_id]
            difference = (
                None
                if base.position_value is None or item.position_value is None
                else item.position_value - base.position_value
            )
            contributions.append(
                PortfolioModelRiskPosition(
                    position_id=item.position_id,
                    reference_value=base.position_value,
                    alternate_value=item.position_value,
                    difference=difference,
                    status=item.status,
                    message=item.message,
                )
            )
        alternatives.append(
            PortfolioModelRiskPoint(
                model=model,
                portfolio_value=alternate.total_value,
                difference_from_reference=(
                    None
                    if reference.total_value is None or alternate.total_value is None
                    else alternate.total_value - reference.total_value
                ),
                contributions=contributions,
                completeness=alternate.completeness,
            )
        )
    return PortfolioModelRiskResult(
        portfolio_id=request.portfolio.portfolio_id,
        reference_model=request.reference_model,
        reference_value=reference.total_value,
        alternatives=alternatives,
        runtime_ms=(perf_counter() - started) * 1_000.0,
    )
