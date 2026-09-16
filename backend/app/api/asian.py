"""Thin, resource-bounded HTTP adapters for Asian pricing and research."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from backend.app.analytics.asian import (
    asian_convergence_inputs,
    compare_asian_methods,
    run_asian_convergence,
)
from backend.app.core.config import Settings, get_settings
from backend.app.core.exceptions import ExoticPricingError
from backend.app.models.asian import (
    AsianComparisonResult,
    AsianConvergenceMode,
    AsianConvergenceResult,
    AsianOptionContract,
    AsianPricingResult,
    AsianSimulationConfig,
    AveragingType,
    GeometricAsianAnalyticalResult,
)
from backend.app.models.market import MarketEnvironment
from backend.app.pricing.asian import (
    asian_monte_carlo_price,
    geometric_asian_analytical_price,
)

pricing_router = APIRouter(prefix="/api/pricing/asian", tags=["asian-pricing"])
analytics_router = APIRouter(prefix="/api/analytics/asian", tags=["asian-analytics"])


class AsianRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    contract: AsianOptionContract
    market: MarketEnvironment
    simulation: AsianSimulationConfig = Field(default_factory=AsianSimulationConfig)


class GeometricAsianRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    contract: AsianOptionContract
    market: MarketEnvironment


class AsianConvergenceRequest(AsianRequest):
    mode: AsianConvergenceMode
    values: list[Annotated[int, Field(gt=0)]] = Field(min_length=1)


def _enforce_asian_work(
    inputs: list[tuple[AsianOptionContract, AsianSimulationConfig]],
    settings: Settings,
) -> None:
    nodes = 0
    for contract, config in inputs:
        if contract.monitoring_count > settings.max_asian_monitoring_points:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "Asian monitoring count exceeds API limit",
            )
        if config.num_simulations > settings.max_api_simulations:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "Asian simulation count exceeds API limit",
            )
        nodes += config.num_simulations * contract.monitoring_count
    if nodes > settings.max_asian_total_path_nodes:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Asian total path nodes exceed API limit",
        )


def _enforce_monitoring_count(contract: AsianOptionContract, settings: Settings) -> None:
    if contract.monitoring_count > settings.max_asian_monitoring_points:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Asian monitoring count exceeds API limit",
        )


@pricing_router.post("", response_model=AsianPricingResult)
def price_asian(
    request: AsianRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> AsianPricingResult:
    _enforce_asian_work([(request.contract, request.simulation)], settings)
    return asian_monte_carlo_price(request.contract, request.market, request.simulation)


@pricing_router.post("/geometric-analytical", response_model=GeometricAsianAnalyticalResult)
def price_geometric_asian(
    request: GeometricAsianRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> GeometricAsianAnalyticalResult:
    _enforce_monitoring_count(request.contract, settings)
    if request.contract.averaging_type is not AveragingType.GEOMETRIC:
        raise ExoticPricingError("geometric-analytical pricing requires averaging_type=GEOMETRIC")
    return geometric_asian_analytical_price(request.contract, request.market)


@analytics_router.post("/compare", response_model=AsianComparisonResult)
def compare_asian(
    request: AsianRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> AsianComparisonResult:
    method_count = 4 if request.contract.averaging_type is AveragingType.ARITHMETIC else 2
    _enforce_asian_work(
        [(request.contract, request.simulation)] * method_count,
        settings,
    )
    return compare_asian_methods(request.contract, request.market, request.simulation)


@analytics_router.post("/convergence", response_model=AsianConvergenceResult)
def converge_asian(
    request: AsianConvergenceRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> AsianConvergenceResult:
    if len(request.values) > settings.max_exotic_convergence_points:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Asian convergence point count exceeds API limit",
        )
    inputs = asian_convergence_inputs(
        request.contract,
        request.simulation,
        request.values,
        request.mode,
    )
    _enforce_asian_work(inputs, settings)
    return run_asian_convergence(
        request.contract,
        request.market,
        request.simulation,
        request.values,
        request.mode,
    )
