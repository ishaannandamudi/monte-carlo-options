"""Thin, bounded HTTP adapters for portfolio valuation and scenario analytics."""

import math
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.core.config import Settings, get_settings
from backend.app.models.portfolio import (
    AsianPosition,
    BarrierPosition,
    BasketPosition,
    DigitalPosition,
    PortfolioGreeksResult,
    PortfolioModelRiskRequest,
    PortfolioModelRiskResult,
    PortfolioPnLExplainRequest,
    PortfolioPnLExplainResult,
    PortfolioPricingModel,
    PortfolioRequest,
    PortfolioScenarioRequest,
    PortfolioScenarioResult,
    PortfolioStressRequest,
    PortfolioStressResult,
    PortfolioValuationResult,
    ScenarioGridRequest,
    ScenarioGridResult,
    VanillaPosition,
)
from backend.app.portfolio.engine import (
    explain_portfolio_pnl,
    portfolio_greeks,
    portfolio_model_risk,
    price_portfolio,
    run_scenario,
    run_scenario_grid,
    run_stress_scenarios,
)

router = APIRouter(prefix="/api/portfolio", tags=["portfolio-risk"])


def _reject(detail: str) -> None:
    raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail)


def _enforce_portfolio(
    request: PortfolioRequest, settings: Settings, *, repetitions: int = 1
) -> None:
    positions = request.portfolio.positions
    if len(positions) > settings.max_portfolio_positions:
        _reject("portfolio position count exceeds API limit")
    general_paths = 0
    lsm_nodes = 0
    asian_nodes = 0
    barrier_nodes = 0
    basket_nodes = 0
    heston_nodes = 0
    for position in positions:
        if isinstance(position, VanillaPosition):
            if position.simulation.num_simulations > settings.max_api_simulations:
                _reject("portfolio GBM simulation count exceeds API limit")
            if position.tree.steps > settings.max_binomial_steps:
                _reject("portfolio tree steps exceed API limit")
            if position.longstaff_schwartz.num_paths > settings.max_lsm_paths:
                _reject("portfolio LSM paths exceed API limit")
            if position.longstaff_schwartz.num_time_steps > settings.max_lsm_time_steps:
                _reject("portfolio LSM time steps exceed API limit")
            if (
                position.longstaff_schwartz.total_simulated_paths
                * position.longstaff_schwartz.num_time_steps
                > settings.max_lsm_total_path_nodes
            ):
                _reject("portfolio LSM path-time nodes exceed API limit")
            if position.heston_simulation.num_simulations > settings.max_heston_paths:
                _reject("portfolio Heston paths exceed API limit")
            if position.heston_simulation.num_time_steps > settings.max_heston_time_steps:
                _reject("portfolio Heston time steps exceed API limit")
            if (
                position.heston_simulation.num_simulations
                * position.heston_simulation.num_time_steps
                > settings.max_heston_total_path_nodes
            ):
                _reject("portfolio Heston path-time nodes exceed API limit")
            model = position.model or (
                PortfolioPricingModel.BLACK_SCHOLES
                if position.instrument.exercise_style.value == "EUROPEAN"
                else PortfolioPricingModel.CRR_AMERICAN
            )
            if model is PortfolioPricingModel.MONTE_CARLO_GBM:
                general_paths += position.simulation.num_simulations
            elif model is PortfolioPricingModel.LONGSTAFF_SCHWARTZ:
                lsm_nodes += (
                    position.longstaff_schwartz.total_simulated_paths
                    * position.longstaff_schwartz.num_time_steps
                )
            elif model is PortfolioPricingModel.HESTON_MONTE_CARLO:
                heston_nodes += (
                    position.heston_simulation.num_simulations
                    * position.heston_simulation.num_time_steps
                )
        elif isinstance(position, AsianPosition):
            if position.simulation.num_simulations > settings.max_api_simulations:
                _reject("portfolio Asian simulation count exceeds API limit")
            if position.instrument.monitoring_count > settings.max_asian_monitoring_points:
                _reject("portfolio Asian monitoring count exceeds API limit")
            if (
                position.simulation.num_simulations * position.instrument.monitoring_count
                > settings.max_asian_total_path_nodes
            ):
                _reject("portfolio Asian path nodes exceed API limit")
            model = position.model or (
                PortfolioPricingModel.GEOMETRIC_ASIAN_ANALYTICAL
                if position.instrument.averaging_type.value == "GEOMETRIC"
                else PortfolioPricingModel.ASIAN_MONTE_CARLO
            )
            if model is PortfolioPricingModel.ASIAN_MONTE_CARLO:
                asian_nodes += (
                    position.simulation.num_simulations * position.instrument.monitoring_count
                )
        elif isinstance(position, BarrierPosition):
            if position.simulation.num_simulations > settings.max_api_simulations:
                _reject("portfolio barrier simulation count exceeds API limit")
            if position.instrument.simulation_state_count > settings.max_barrier_monitoring_points:
                _reject("portfolio barrier monitoring count exceeds API limit")
            if (
                position.simulation.num_simulations * position.instrument.simulation_state_count
                > settings.max_barrier_total_path_nodes
            ):
                _reject("portfolio barrier path nodes exceed API limit")
            model = position.model or PortfolioPricingModel.BARRIER_MONTE_CARLO
            if model is PortfolioPricingModel.BARRIER_MONTE_CARLO:
                barrier_nodes += (
                    position.simulation.num_simulations * position.instrument.simulation_state_count
                )
        elif isinstance(position, BasketPosition):
            if position.market.num_assets > settings.max_multi_asset_count:
                _reject("portfolio basket asset count exceeds API limit")
            if position.simulation.num_simulations > settings.max_api_simulations:
                _reject("portfolio basket simulation count exceeds API limit")
            if (
                position.simulation.num_simulations * position.market.num_assets
                > settings.max_multi_asset_path_nodes
            ):
                _reject("portfolio basket path nodes exceed API limit")
            model = position.model or (
                PortfolioPricingModel.GEOMETRIC_BASKET_ANALYTICAL
                if position.instrument.basket_type.value == "GEOMETRIC"
                else PortfolioPricingModel.BASKET_MONTE_CARLO
            )
            if model is PortfolioPricingModel.BASKET_MONTE_CARLO:
                basket_nodes += position.simulation.num_simulations * position.market.num_assets
        elif isinstance(position, DigitalPosition):
            if position.simulation.num_simulations > settings.max_api_simulations:
                _reject("portfolio digital simulation count exceeds API limit")
            model = position.model or PortfolioPricingModel.DIGITAL_ANALYTICAL
            if model is PortfolioPricingModel.DIGITAL_MONTE_CARLO:
                general_paths += position.simulation.num_simulations
    if general_paths * repetitions > settings.max_api_simulations:
        _reject("aggregate portfolio simulation paths exceed API limit")
    if lsm_nodes * repetitions > settings.max_lsm_total_path_nodes:
        _reject("aggregate portfolio LSM path-time nodes exceed API limit")
    if asian_nodes * repetitions > settings.max_asian_total_path_nodes:
        _reject("aggregate portfolio Asian path nodes exceed API limit")
    if barrier_nodes * repetitions > settings.max_barrier_total_path_nodes:
        _reject("aggregate portfolio barrier path nodes exceed API limit")
    if basket_nodes * repetitions > settings.max_multi_asset_path_nodes:
        _reject("aggregate portfolio basket path nodes exceed API limit")
    if heston_nodes * repetitions > settings.max_heston_total_path_nodes:
        _reject("aggregate portfolio Heston path-time nodes exceed API limit")


def _grid_size(request: ScenarioGridRequest) -> int:
    return math.prod(len(axis.values) for axis in request.axes)


@router.post("/value", response_model=PortfolioValuationResult)
def value_portfolio(
    request: PortfolioRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> PortfolioValuationResult:
    _enforce_portfolio(request, settings)
    return price_portfolio(request)


@router.post("/greeks", response_model=PortfolioGreeksResult)
def greeks_portfolio(
    request: PortfolioRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> PortfolioGreeksResult:
    _enforce_portfolio(request, settings, repetitions=6)
    return portfolio_greeks(request)


@router.post("/scenario", response_model=PortfolioScenarioResult)
def scenario_portfolio(
    request: PortfolioScenarioRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> PortfolioScenarioResult:
    _enforce_portfolio(request, settings, repetitions=7)
    return run_scenario(request)


@router.post("/scenario-grid", response_model=ScenarioGridResult)
def scenario_grid_portfolio(
    request: ScenarioGridRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> ScenarioGridResult:
    points = _grid_size(request)
    if sum(len(axis.values) for axis in request.axes) > settings.max_portfolio_scenario_points:
        _reject("portfolio scenario coordinate count exceeds API limit")
    if points > settings.max_portfolio_grid_points:
        _reject("portfolio grid point count exceeds API limit")
    _enforce_portfolio(request, settings, repetitions=points + 6)
    return run_scenario_grid(request)


@router.post("/stress", response_model=PortfolioStressResult)
def stress_portfolio(
    request: PortfolioStressRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> PortfolioStressResult:
    if len(request.scenarios) > settings.max_portfolio_stress_scenarios:
        _reject("portfolio stress scenario count exceeds API limit")
    if len(request.scenarios) > settings.max_portfolio_scenario_points:
        _reject("portfolio scenario count exceeds API limit")
    _enforce_portfolio(request, settings, repetitions=len(request.scenarios) + 6)
    return run_stress_scenarios(request)


@router.post("/pnl-explain", response_model=PortfolioPnLExplainResult)
def pnl_explain_portfolio(
    request: PortfolioPnLExplainRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> PortfolioPnLExplainResult:
    _enforce_portfolio(request, settings, repetitions=14)
    return explain_portfolio_pnl(request)


@router.post("/model-risk", response_model=PortfolioModelRiskResult)
def model_risk_portfolio(
    request: PortfolioModelRiskRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> PortfolioModelRiskResult:
    if len(request.alternate_models) > settings.max_portfolio_model_risk_models:
        _reject("portfolio model-risk model count exceeds API limit")
    _enforce_portfolio(request, settings, repetitions=len(request.alternate_models) + 1)
    return portfolio_model_risk(request)
