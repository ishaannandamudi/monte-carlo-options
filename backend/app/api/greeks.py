"""Thin HTTP adapters for analytical and Monte Carlo Greek estimates."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from backend.app.api.pricing import _enforce_simulation_limit
from backend.app.core.config import Settings, get_settings
from backend.app.models.greeks import Greek, GreekBumpConfig, GreekMethod
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract
from backend.app.models.results import AnalyticalGreeksResult, MonteCarloGreeksResult
from backend.app.models.simulation import SimulationConfig
from backend.app.pricing.black_scholes_greeks import black_scholes_greeks
from backend.app.pricing.monte_carlo_greeks import monte_carlo_greeks

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/greeks", tags=["greeks"])


class GreekRequestModel(BaseModel):
    """Strict Phase 3 request-body policy."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class AnalyticalGreeksRequest(GreekRequestModel):
    contract: OptionContract
    market: MarketEnvironment


class MonteCarloGreeksRequest(GreekRequestModel):
    contract: OptionContract
    market: MarketEnvironment
    simulation: SimulationConfig
    greeks: list[Greek] = Field(min_length=1)
    estimator_method: GreekMethod = GreekMethod.FINITE_DIFFERENCE
    bump_config: GreekBumpConfig = Field(default_factory=GreekBumpConfig)


@router.post("/black-scholes", response_model=AnalyticalGreeksResult)
def analytical_greeks(request: AnalyticalGreeksRequest) -> AnalyticalGreeksResult:
    """Return all raw-unit analytical Black-Scholes-Merton Greeks."""

    logger.info(
        "greek_request method=ANALYTICAL option_type=%s",
        request.contract.option_type.value,
    )
    return black_scholes_greeks(request.contract, request.market)


@router.post("/monte-carlo", response_model=MonteCarloGreeksResult)
def stochastic_greeks(
    request: MonteCarloGreeksRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> MonteCarloGreeksResult:
    """Return requested CRN finite-difference or supported pathwise Greeks."""

    _enforce_simulation_limit(request.simulation.num_simulations, settings)
    logger.info(
        "greek_request method=%s greeks=%s simulations=%d variance_reduction=%s",
        request.estimator_method.value,
        ",".join(greek.value for greek in request.greeks),
        request.simulation.num_simulations,
        request.simulation.variance_reduction.value,
    )
    return monte_carlo_greeks(
        request.contract,
        request.market,
        request.simulation,
        request.greeks,
        request.estimator_method,
        request.bump_config,
    )
