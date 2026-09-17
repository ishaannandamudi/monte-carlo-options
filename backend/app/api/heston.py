"""Resource-bounded HTTP adapters for Heston pricing and research."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from backend.app.analytics.heston import (
    compare_heston_schemes,
    heston_convergence_configs,
    heston_implied_volatility_smile,
    heston_parameter_sensitivity,
    heston_variance_diagnostics,
    run_heston_convergence,
)
from backend.app.core.config import Settings, get_settings
from backend.app.models.heston import (
    HestonAnalyticalResult,
    HestonComparisonResult,
    HestonConvergenceMode,
    HestonConvergenceResult,
    HestonIntegrationConfig,
    HestonMonteCarloResult,
    HestonParameterName,
    HestonParameters,
    HestonSensitivityResult,
    HestonSimulationConfig,
    HestonSmileResult,
    HestonVarianceDiagnosticsResult,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract
from backend.app.models.volatility import ImpliedVolatilityConfig
from backend.app.pricing.heston import heston_monte_carlo_price
from backend.app.pricing.heston_analytical import heston_analytical_price

pricing_router = APIRouter(prefix="/api/pricing/heston", tags=["heston-pricing"])
analytics_router = APIRouter(prefix="/api/analytics/heston", tags=["heston-analytics"])


class HestonRequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class HestonMonteCarloRequest(HestonRequestModel):
    contract: OptionContract
    market: MarketEnvironment
    parameters: HestonParameters
    simulation: HestonSimulationConfig = Field(default_factory=HestonSimulationConfig)


class HestonAnalyticalRequest(HestonRequestModel):
    contract: OptionContract
    market: MarketEnvironment
    parameters: HestonParameters
    integration: HestonIntegrationConfig = Field(default_factory=HestonIntegrationConfig)


class HestonComparisonRequest(HestonMonteCarloRequest):
    integration: HestonIntegrationConfig = Field(default_factory=HestonIntegrationConfig)


class HestonConvergenceRequest(HestonComparisonRequest):
    mode: HestonConvergenceMode
    values: list[Annotated[int, Field(gt=0)]] = Field(min_length=1)


class HestonSmileRequest(HestonAnalyticalRequest):
    strikes: list[Annotated[float, Field(gt=0.0)]] = Field(min_length=1)
    implied_volatility: ImpliedVolatilityConfig = Field(default_factory=ImpliedVolatilityConfig)


class HestonSensitivityRequest(HestonAnalyticalRequest):
    parameter: HestonParameterName
    values: list[float] = Field(min_length=1)
    implied_volatility: ImpliedVolatilityConfig = Field(default_factory=ImpliedVolatilityConfig)


class HestonVarianceDiagnosticsRequest(HestonMonteCarloRequest):
    near_zero_tolerance: Annotated[float, Field(ge=0.0)] = 1e-8


def _enforce_heston_work(configs: list[HestonSimulationConfig], settings: Settings) -> None:
    total_nodes = 0
    for config in configs:
        if config.num_simulations > settings.max_heston_paths:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "Heston path count exceeds API limit",
            )
        if config.num_time_steps > settings.max_heston_time_steps:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "Heston time-step count exceeds API limit",
            )
        total_nodes += config.num_simulations * config.num_time_steps
    if total_nodes > settings.max_heston_total_path_nodes:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Heston total path nodes exceed API limit",
        )


@pricing_router.post("/monte-carlo", response_model=HestonMonteCarloResult)
def price_heston_monte_carlo(
    request: HestonMonteCarloRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> HestonMonteCarloResult:
    _enforce_heston_work([request.simulation], settings)
    return heston_monte_carlo_price(
        request.contract, request.market, request.parameters, request.simulation
    )


@pricing_router.post("/analytical", response_model=HestonAnalyticalResult)
def price_heston_analytical(
    request: HestonAnalyticalRequest,
) -> HestonAnalyticalResult:
    return heston_analytical_price(
        request.contract, request.market, request.parameters, request.integration
    )


@analytics_router.post("/compare", response_model=HestonComparisonResult)
def compare_heston(
    request: HestonComparisonRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> HestonComparisonResult:
    _enforce_heston_work([request.simulation, request.simulation], settings)
    return compare_heston_schemes(
        request.contract,
        request.market,
        request.parameters,
        request.simulation,
        request.integration,
    )


@analytics_router.post("/convergence", response_model=HestonConvergenceResult)
def converge_heston(
    request: HestonConvergenceRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> HestonConvergenceResult:
    if len(request.values) > settings.max_heston_convergence_points:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Heston convergence point count exceeds API limit",
        )
    configs = heston_convergence_configs(request.simulation, request.values, request.mode)
    _enforce_heston_work(configs, settings)
    return run_heston_convergence(
        request.contract,
        request.market,
        request.parameters,
        request.simulation,
        request.values,
        request.mode,
        request.integration,
    )


@analytics_router.post("/smile", response_model=HestonSmileResult)
def smile_heston(
    request: HestonSmileRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> HestonSmileResult:
    if len(request.strikes) > settings.max_heston_smile_points:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Heston smile point count exceeds API limit",
        )
    return heston_implied_volatility_smile(
        request.contract,
        request.market,
        request.parameters,
        request.strikes,
        request.integration,
        request.implied_volatility,
    )


@analytics_router.post("/sensitivity", response_model=HestonSensitivityResult)
def sensitivity_heston(
    request: HestonSensitivityRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> HestonSensitivityResult:
    if len(request.values) > settings.max_heston_sensitivity_points:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Heston sensitivity point count exceeds API limit",
        )
    return heston_parameter_sensitivity(
        request.contract,
        request.market,
        request.parameters,
        request.parameter,
        request.values,
        request.integration,
        request.implied_volatility,
    )


@analytics_router.post("/variance-diagnostics", response_model=HestonVarianceDiagnosticsResult)
def variance_diagnostics_heston(
    request: HestonVarianceDiagnosticsRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> HestonVarianceDiagnosticsResult:
    _enforce_heston_work([request.simulation], settings)
    return heston_variance_diagnostics(
        request.contract,
        request.market,
        request.parameters,
        request.simulation,
        near_zero_tolerance=request.near_zero_tolerance,
    )
