"""Thin resource-bounded HTTP adapters for barrier pricing and research."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from backend.app.analytics.barrier import (
    barrier_convergence_inputs,
    barrier_parity,
    compare_barrier_methods,
    run_barrier_convergence,
    run_barrier_level_sensitivity,
)
from backend.app.core.config import Settings, get_settings
from backend.app.models.barrier import (
    BarrierAnalyticalResult,
    BarrierComparisonResult,
    BarrierConvergenceMode,
    BarrierConvergenceResult,
    BarrierLevelSensitivityResult,
    BarrierOptionContract,
    BarrierParityResult,
    BarrierPricingResult,
    BarrierSimulationConfig,
)
from backend.app.models.market import MarketEnvironment
from backend.app.pricing.barrier import (
    barrier_monte_carlo_price,
    continuous_barrier_analytical_price,
)

pricing_router = APIRouter(prefix="/api/pricing/barrier", tags=["barrier-pricing"])
analytics_router = APIRouter(prefix="/api/analytics/barrier", tags=["barrier-analytics"])


class BarrierRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    contract: BarrierOptionContract
    market: MarketEnvironment
    simulation: BarrierSimulationConfig = Field(default_factory=BarrierSimulationConfig)


class BarrierAnalyticalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    contract: BarrierOptionContract
    market: MarketEnvironment


class BarrierConvergenceRequest(BarrierRequest):
    mode: BarrierConvergenceMode
    values: list[Annotated[int, Field(gt=0)]] = Field(min_length=1)


class BarrierLevelSensitivityRequest(BarrierRequest):
    barrier_levels: list[Annotated[float, Field(gt=0.0)]] = Field(min_length=1)


def _enforce_contract(contract: BarrierOptionContract, settings: Settings) -> None:
    if contract.monitoring_count > settings.max_barrier_monitoring_points:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "barrier monitoring count exceeds API limit",
        )


def _enforce_work(
    inputs: list[tuple[BarrierOptionContract, BarrierSimulationConfig, int]],
    settings: Settings,
) -> None:
    nodes = 0
    for contract, config, runs in inputs:
        _enforce_contract(contract, settings)
        if config.num_simulations > settings.max_api_simulations:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "barrier simulation count exceeds API limit",
            )
        nodes += config.num_simulations * (contract.simulation_state_count + 1) * runs
    if nodes > settings.max_barrier_total_path_nodes:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "barrier total path nodes exceed API limit",
        )


@pricing_router.post("", response_model=BarrierPricingResult)
def price_barrier(
    request: BarrierRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> BarrierPricingResult:
    _enforce_work([(request.contract, request.simulation, 1)], settings)
    return barrier_monte_carlo_price(request.contract, request.market, request.simulation)


@pricing_router.post("/analytical", response_model=BarrierAnalyticalResult)
def price_barrier_analytical(
    request: BarrierAnalyticalRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> BarrierAnalyticalResult:
    _enforce_contract(request.contract, settings)
    return continuous_barrier_analytical_price(request.contract, request.market)


@analytics_router.post("/compare", response_model=BarrierComparisonResult)
def compare_barrier(
    request: BarrierRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> BarrierComparisonResult:
    run_count = 6 if request.contract.rebate == 0.0 else 2
    _enforce_work([(request.contract, request.simulation, run_count)], settings)
    return compare_barrier_methods(request.contract, request.market, request.simulation)


@analytics_router.post("/convergence", response_model=BarrierConvergenceResult)
def converge_barrier(
    request: BarrierConvergenceRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> BarrierConvergenceResult:
    if len(request.values) > settings.max_barrier_convergence_points:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "barrier convergence point count exceeds API limit",
        )
    inputs = barrier_convergence_inputs(
        request.contract,
        request.simulation,
        request.values,
        request.mode,
    )
    _enforce_work([(contract, config, 2) for contract, config in inputs], settings)
    return run_barrier_convergence(
        request.contract,
        request.market,
        request.simulation,
        request.values,
        request.mode,
    )


@analytics_router.post("/parity", response_model=BarrierParityResult)
def check_barrier_parity(
    request: BarrierRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> BarrierParityResult:
    _enforce_work([(request.contract, request.simulation, 2)], settings)
    return barrier_parity(request.contract, request.market, request.simulation)


@analytics_router.post("/sensitivity", response_model=BarrierLevelSensitivityResult)
def barrier_level_sensitivity(
    request: BarrierLevelSensitivityRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> BarrierLevelSensitivityResult:
    if len(request.barrier_levels) > settings.max_barrier_convergence_points:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "barrier sensitivity point count exceeds API limit",
        )
    _enforce_work(
        [(request.contract, request.simulation, 2 * len(request.barrier_levels))],
        settings,
    )
    return run_barrier_level_sensitivity(
        request.contract,
        request.market,
        request.simulation,
        request.barrier_levels,
    )
