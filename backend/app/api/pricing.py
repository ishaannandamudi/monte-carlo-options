"""Thin HTTP adapters for the Phase 1 pricing services."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict

from backend.app.core.config import Settings, get_settings
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract
from backend.app.models.results import BlackScholesResult, MonteCarloResult, PricingComparison
from backend.app.models.simulation import SimulationConfig
from backend.app.pricing.black_scholes import BlackScholesPricer
from backend.app.pricing.comparison import compare_prices
from backend.app.pricing.monte_carlo import MonteCarloPricer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/pricing", tags=["pricing"])


class RequestModel(BaseModel):
    """Strict request-body policy."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class PricingRequest(RequestModel):
    contract: OptionContract
    market: MarketEnvironment
    simulation: SimulationConfig


class BlackScholesRequest(RequestModel):
    contract: OptionContract
    market: MarketEnvironment


def _enforce_simulation_limit(simulations: int, settings: Settings) -> None:
    if simulations > settings.max_api_simulations:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(f"num_simulations exceeds API limit of {settings.max_api_simulations}"),
        )


@router.post("/monte-carlo", response_model=MonteCarloResult)
def monte_carlo_price(
    request: PricingRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> MonteCarloResult:
    """Price one option by risk-neutral GBM Monte Carlo."""

    _enforce_simulation_limit(request.simulation.num_simulations, settings)
    logger.info(
        "pricing_request engine=monte_carlo option_type=%s simulations=%d variance_reduction=%s",
        request.contract.option_type.value,
        request.simulation.num_simulations,
        request.simulation.variance_reduction.value,
    )
    return MonteCarloPricer().price(request.contract, request.market, request.simulation)


@router.post("/black-scholes", response_model=BlackScholesResult)
def analytical_price(request: BlackScholesRequest) -> BlackScholesResult:
    """Price one option with the analytical Black-Scholes-Merton formula."""

    logger.info(
        "pricing_request engine=black_scholes option_type=%s",
        request.contract.option_type.value,
    )
    return BlackScholesPricer().price(request.contract, request.market)


@router.post("/compare", response_model=PricingComparison)
def compare_price(
    request: PricingRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> PricingComparison:
    """Compare Monte Carlo with the analytical benchmark."""

    _enforce_simulation_limit(request.simulation.num_simulations, settings)
    logger.info(
        "pricing_request engine=comparison option_type=%s simulations=%d variance_reduction=%s",
        request.contract.option_type.value,
        request.simulation.num_simulations,
        request.simulation.variance_reduction.value,
    )
    return compare_prices(request.contract, request.market, request.simulation)
