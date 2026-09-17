"""Thin, resource-bounded HTTP adapters for multi-asset and basket research."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from backend.app.analytics.basket import (
    basket_convergence_inputs,
    compare_basket_methods,
    run_basket_convergence,
    run_correlation_sensitivity,
    run_weight_sensitivity,
)
from backend.app.analytics.multi_asset import empirical_correlation_diagnostics
from backend.app.core.config import Settings, get_settings
from backend.app.models.basket import (
    BasketComparisonResult,
    BasketConvergenceResult,
    BasketOptionContract,
    BasketPricingResult,
    BasketSimulationConfig,
    BasketType,
    CorrelationSensitivityResult,
    GeometricBasketAnalyticalResult,
    WeightSensitivityResult,
)
from backend.app.models.multi_asset import (
    CorrelationDiagnosticConfig,
    EmpiricalCorrelationResult,
    MultiAssetMarketEnvironment,
)
from backend.app.pricing.basket import (
    basket_monte_carlo_price,
    geometric_basket_analytical_price,
)

pricing_router = APIRouter(prefix="/api/pricing/basket", tags=["basket-pricing"])
analytics_router = APIRouter(prefix="/api/analytics/basket", tags=["basket-analytics"])
multi_asset_router = APIRouter(prefix="/api/analytics/multi-asset", tags=["multi-asset-analytics"])


class BasketRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    contract: BasketOptionContract
    market: MultiAssetMarketEnvironment
    simulation: BasketSimulationConfig = Field(default_factory=BasketSimulationConfig)


class GeometricBasketRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    contract: BasketOptionContract
    market: MultiAssetMarketEnvironment


class BasketConvergenceRequest(BasketRequest):
    simulation_counts: list[Annotated[int, Field(gt=0)]] = Field(min_length=1)


class CorrelationSensitivityRequest(BasketRequest):
    rho_values: list[float] = Field(min_length=1)


class WeightSensitivityRequest(BasketRequest):
    first_asset_weights: list[Annotated[float, Field(ge=0.0, le=1.0)]] = Field(min_length=1)


class CorrelationDiagnosticRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    market: MultiAssetMarketEnvironment
    diagnostic: CorrelationDiagnosticConfig = Field(default_factory=CorrelationDiagnosticConfig)


def _enforce_assets(market: MultiAssetMarketEnvironment, settings: Settings) -> None:
    if market.num_assets > settings.max_multi_asset_count:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "multi-asset count exceeds API limit",
        )


def _enforce_work(
    market: MultiAssetMarketEnvironment,
    simulation_counts: list[int],
    settings: Settings,
) -> None:
    _enforce_assets(market, settings)
    if any(count > settings.max_api_simulations for count in simulation_counts):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "basket simulation count exceeds API limit",
        )
    total_nodes = market.num_assets * sum(simulation_counts)
    if total_nodes > settings.max_multi_asset_path_nodes:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "multi-asset path nodes exceed API limit",
        )


@pricing_router.post("", response_model=BasketPricingResult)
def price_basket(
    request: BasketRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> BasketPricingResult:
    _enforce_work(request.market, [request.simulation.num_simulations], settings)
    return basket_monte_carlo_price(request.contract, request.market, request.simulation)


@pricing_router.post("/geometric-analytical", response_model=GeometricBasketAnalyticalResult)
def price_geometric_basket(
    request: GeometricBasketRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> GeometricBasketAnalyticalResult:
    _enforce_assets(request.market, settings)
    if request.contract.basket_type is not BasketType.GEOMETRIC:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "geometric analytical pricing requires basket_type=GEOMETRIC",
        )
    return geometric_basket_analytical_price(request.contract, request.market)


@analytics_router.post("/compare", response_model=BasketComparisonResult)
def compare_basket(
    request: BasketRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> BasketComparisonResult:
    method_count = 4 if request.contract.basket_type is BasketType.ARITHMETIC else 2
    _enforce_work(
        request.market,
        [request.simulation.num_simulations] * method_count,
        settings,
    )
    return compare_basket_methods(request.contract, request.market, request.simulation)


@analytics_router.post("/convergence", response_model=BasketConvergenceResult)
def converge_basket(
    request: BasketConvergenceRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> BasketConvergenceResult:
    if len(request.simulation_counts) > settings.max_basket_convergence_points:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "basket convergence point count exceeds API limit",
        )
    point_configs = basket_convergence_inputs(request.simulation, request.simulation_counts)
    _enforce_work(
        request.market,
        [config.num_simulations for config in point_configs],
        settings,
    )
    return run_basket_convergence(
        request.contract,
        request.market,
        request.simulation,
        request.simulation_counts,
    )


@analytics_router.post("/correlation-sensitivity", response_model=CorrelationSensitivityResult)
def correlation_sensitivity(
    request: CorrelationSensitivityRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> CorrelationSensitivityResult:
    if len(request.rho_values) > settings.max_basket_sensitivity_points:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "basket sensitivity point count exceeds API limit",
        )
    _enforce_work(
        request.market,
        [request.simulation.num_simulations] * len(request.rho_values),
        settings,
    )
    return run_correlation_sensitivity(
        request.contract,
        request.market,
        request.simulation,
        request.rho_values,
    )


@analytics_router.post("/weight-sensitivity", response_model=WeightSensitivityResult)
def weight_sensitivity(
    request: WeightSensitivityRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> WeightSensitivityResult:
    if len(request.first_asset_weights) > settings.max_basket_sensitivity_points:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "basket sensitivity point count exceeds API limit",
        )
    _enforce_work(
        request.market,
        [request.simulation.num_simulations] * len(request.first_asset_weights),
        settings,
    )
    return run_weight_sensitivity(
        request.contract,
        request.market,
        request.simulation,
        request.first_asset_weights,
    )


@multi_asset_router.post("/correlation", response_model=EmpiricalCorrelationResult)
def diagnose_correlation(
    request: CorrelationDiagnosticRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> EmpiricalCorrelationResult:
    _enforce_work(request.market, [request.diagnostic.num_simulations], settings)
    return empirical_correlation_diagnostics(request.market, request.diagnostic)
