"""Thin HTTP adapters for analytical and Monte Carlo digital pricing."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from backend.app.analytics.digital import compare_digital_prices, run_digital_convergence
from backend.app.core.config import Settings, get_settings
from backend.app.models.digital import (
    DigitalAnalyticalResult,
    DigitalComparison,
    DigitalConvergenceResult,
    DigitalMonteCarloResult,
    DigitalOptionContract,
    DigitalPricingMethod,
    DigitalSimulationConfig,
)
from backend.app.models.market import MarketEnvironment
from backend.app.pricing.digital import digital_analytical_price, digital_monte_carlo_price

pricing_router = APIRouter(prefix="/api/pricing", tags=["digital-pricing"])
analytics_router = APIRouter(prefix="/api/analytics/digital", tags=["digital-analytics"])


class DigitalPricingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    contract: DigitalOptionContract
    market: MarketEnvironment
    method: DigitalPricingMethod = DigitalPricingMethod.ANALYTICAL
    simulation: DigitalSimulationConfig = Field(default_factory=DigitalSimulationConfig)


class DigitalResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    contract: DigitalOptionContract
    market: MarketEnvironment
    simulation: DigitalSimulationConfig = Field(default_factory=DigitalSimulationConfig)


class DigitalConvergenceRequest(DigitalResearchRequest):
    simulation_counts: list[Annotated[int, Field(gt=0)]] = Field(min_length=1)


def _enforce_counts(counts: list[int], settings: Settings) -> None:
    if any(count > settings.max_api_simulations for count in counts):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "digital simulation count exceeds API limit",
        )


@pricing_router.post(
    "/digital",
    response_model=DigitalAnalyticalResult | DigitalMonteCarloResult,
)
def price_digital(
    request: DigitalPricingRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> DigitalAnalyticalResult | DigitalMonteCarloResult:
    if request.method is DigitalPricingMethod.ANALYTICAL:
        return digital_analytical_price(request.contract, request.market)
    _enforce_counts([request.simulation.num_simulations], settings)
    return digital_monte_carlo_price(request.contract, request.market, request.simulation)


@analytics_router.post("/compare", response_model=DigitalComparison)
def compare_digital(
    request: DigitalResearchRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> DigitalComparison:
    _enforce_counts([request.simulation.num_simulations], settings)
    return compare_digital_prices(request.contract, request.market, request.simulation)


@analytics_router.post("/convergence", response_model=DigitalConvergenceResult)
def converge_digital(
    request: DigitalConvergenceRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> DigitalConvergenceResult:
    if len(request.simulation_counts) > settings.max_exotic_convergence_points:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "digital convergence point count exceeds API limit",
        )
    _enforce_counts(request.simulation_counts, settings)
    return run_digital_convergence(
        request.contract,
        request.market,
        request.simulation,
        request.simulation_counts,
    )
