"""Resource-bounded HTTP adapters for Merton jump-diffusion research."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from backend.app.analytics.merton import (
    compare_merton,
    merton_implied_volatility_smile,
    merton_jump_diagnostics,
    merton_parameter_sensitivity,
    run_merton_convergence,
    run_merton_series_convergence,
)
from backend.app.core.config import Settings, get_settings
from backend.app.models.heston import HestonIntegrationConfig, HestonParameters
from backend.app.models.market import MarketEnvironment
from backend.app.models.merton import (
    MertonAnalyticalResult,
    MertonComparisonResult,
    MertonConvergenceResult,
    MertonGreekBumps,
    MertonGreekResult,
    MertonJumpDiagnosticsResult,
    MertonJumpParameters,
    MertonMonteCarloResult,
    MertonParameterName,
    MertonSensitivityResult,
    MertonSeriesConfig,
    MertonSeriesConvergenceResult,
    MertonSimulationConfig,
    MertonSmileResult,
)
from backend.app.models.option import OptionContract
from backend.app.models.volatility import ImpliedVolatilityConfig
from backend.app.pricing.merton import (
    merton_analytical_greeks,
    merton_analytical_price,
    merton_monte_carlo_price,
)

pricing_router = APIRouter(prefix="/api/pricing/merton", tags=["merton-pricing"])
analytics_router = APIRouter(prefix="/api/analytics/merton", tags=["merton-analytics"])


class MertonRequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class MertonAnalyticalRequest(MertonRequestModel):
    contract: OptionContract
    market: MarketEnvironment
    parameters: MertonJumpParameters
    series: MertonSeriesConfig = Field(default_factory=MertonSeriesConfig)


class MertonMonteCarloRequest(MertonRequestModel):
    contract: OptionContract
    market: MarketEnvironment
    parameters: MertonJumpParameters
    simulation: MertonSimulationConfig = Field(default_factory=MertonSimulationConfig)


class MertonComparisonRequest(MertonMonteCarloRequest):
    series: MertonSeriesConfig = Field(default_factory=MertonSeriesConfig)
    heston_parameters: HestonParameters | None = None
    heston_integration: HestonIntegrationConfig = Field(default_factory=HestonIntegrationConfig)


class MertonConvergenceRequest(MertonAnalyticalRequest):
    path_counts: list[Annotated[int, Field(gt=0)]] = Field(min_length=1)
    root_seed: int = Field(default=42, ge=0)
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)


class MertonSeriesConvergenceRequest(MertonAnalyticalRequest):
    tolerances: list[Annotated[float, Field(gt=0.0, lt=1.0)]] = Field(min_length=1)


class MertonSmileRequest(MertonAnalyticalRequest):
    strikes: list[Annotated[float, Field(gt=0.0)]] = Field(min_length=1)
    implied_volatility: ImpliedVolatilityConfig = Field(default_factory=ImpliedVolatilityConfig)


class MertonSensitivityRequest(MertonAnalyticalRequest):
    parameter: MertonParameterName
    values: list[float] = Field(min_length=1)
    implied_volatility: ImpliedVolatilityConfig = Field(default_factory=ImpliedVolatilityConfig)


class MertonGreeksRequest(MertonAnalyticalRequest):
    bumps: MertonGreekBumps = Field(default_factory=MertonGreekBumps)


def _enforce_series(config: MertonSeriesConfig, settings: Settings) -> None:
    if config.max_terms > settings.max_merton_series_terms:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Merton analytical series terms exceed API limit",
        )


def _enforce_paths(config: MertonSimulationConfig, settings: Settings) -> None:
    if config.num_simulations > settings.max_merton_paths:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Merton path count exceeds API limit",
        )


@pricing_router.post("/analytical", response_model=MertonAnalyticalResult)
def price_merton_analytical(
    request: MertonAnalyticalRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> MertonAnalyticalResult:
    _enforce_series(request.series, settings)
    return merton_analytical_price(
        request.contract, request.market, request.parameters, request.series
    )


@pricing_router.post("/monte-carlo", response_model=MertonMonteCarloResult)
def price_merton_monte_carlo(
    request: MertonMonteCarloRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> MertonMonteCarloResult:
    _enforce_paths(request.simulation, settings)
    return merton_monte_carlo_price(
        request.contract, request.market, request.parameters, request.simulation
    )


@analytics_router.post("/compare", response_model=MertonComparisonResult)
def compare_merton_endpoint(
    request: MertonComparisonRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> MertonComparisonResult:
    _enforce_paths(request.simulation, settings)
    _enforce_series(request.series, settings)
    return compare_merton(
        request.contract,
        request.market,
        request.parameters,
        request.simulation,
        request.series,
        request.heston_parameters,
        request.heston_integration,
    )


@analytics_router.post("/convergence", response_model=MertonConvergenceResult)
def converge_merton(
    request: MertonConvergenceRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> MertonConvergenceResult:
    if len(request.path_counts) > settings.max_merton_convergence_points:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Merton convergence point count exceeds API limit",
        )
    if max(request.path_counts) > settings.max_merton_paths:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Merton convergence path count exceeds API limit",
        )
    _enforce_series(request.series, settings)
    return run_merton_convergence(
        request.contract,
        request.market,
        request.parameters,
        request.path_counts,
        request.root_seed,
        request.confidence_level,
        request.series,
    )


@analytics_router.post("/series-convergence", response_model=MertonSeriesConvergenceResult)
def converge_merton_series(
    request: MertonSeriesConvergenceRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> MertonSeriesConvergenceResult:
    if len(request.tolerances) > settings.max_merton_convergence_points:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Merton series convergence point count exceeds API limit",
        )
    _enforce_series(request.series, settings)
    return run_merton_series_convergence(
        request.contract,
        request.market,
        request.parameters,
        request.tolerances,
        request.series.max_terms,
    )


@analytics_router.post("/smile", response_model=MertonSmileResult)
def smile_merton(
    request: MertonSmileRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> MertonSmileResult:
    if len(request.strikes) > settings.max_merton_smile_points:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Merton smile point count exceeds API limit",
        )
    _enforce_series(request.series, settings)
    return merton_implied_volatility_smile(
        request.contract,
        request.market,
        request.parameters,
        request.strikes,
        request.series,
        request.implied_volatility,
    )


@analytics_router.post("/sensitivity", response_model=MertonSensitivityResult)
def sensitivity_merton(
    request: MertonSensitivityRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> MertonSensitivityResult:
    if len(request.values) > settings.max_merton_sensitivity_points:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Merton sensitivity point count exceeds API limit",
        )
    _enforce_series(request.series, settings)
    return merton_parameter_sensitivity(
        request.contract,
        request.market,
        request.parameters,
        request.parameter,
        request.values,
        request.series,
        request.implied_volatility,
    )


@analytics_router.post("/jump-diagnostics", response_model=MertonJumpDiagnosticsResult)
def diagnostics_merton(
    request: MertonMonteCarloRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> MertonJumpDiagnosticsResult:
    _enforce_paths(request.simulation, settings)
    return merton_jump_diagnostics(
        request.contract, request.market, request.parameters, request.simulation
    )


@analytics_router.post("/greeks", response_model=MertonGreekResult)
def greeks_merton(
    request: MertonGreeksRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> MertonGreekResult:
    _enforce_series(request.series, settings)
    return merton_analytical_greeks(
        request.contract,
        request.market,
        request.parameters,
        request.series,
        request.bumps,
    )
