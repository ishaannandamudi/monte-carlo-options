"""Resource-bounded HTTP adapters for Phase 20 quasi-Monte Carlo research."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.analytics.qmc import compare_qmc_methods, qmc_convergence
from backend.app.core.config import Settings, get_settings
from backend.app.models.qmc import (
    PowerOfTwoPolicy,
    QMCAsianRequest,
    QMCBasketRequest,
    QMCComparisonRequest,
    QMCComparisonResult,
    QMCConfig,
    QMCConvergenceRequest,
    QMCConvergenceResult,
    QMCEuropeanRequest,
    QMCPricingResult,
    SamplingDiagnosticsRequest,
    SamplingDiagnosticsResult,
)
from backend.app.pricing.qmc import price_asian_qmc, price_basket_qmc, price_european_qmc
from backend.app.simulation.qmc import sampling_diagnostics

pricing_router = APIRouter(prefix="/api/pricing/qmc", tags=["qmc-pricing"])
analytics_router = APIRouter(prefix="/api/analytics/qmc", tags=["qmc-analytics"])


def _reject(message: str) -> None:
    raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, message)


def _enforce_config(config: QMCConfig, settings: Settings) -> None:
    if config.effective_num_samples > settings.max_qmc_samples:
        _reject("QMC sample count exceeds API limit")
    if config.dimension > settings.max_qmc_dimension:
        _reject("QMC dimension exceeds API limit")
    if config.num_randomizations > settings.max_qmc_replicates:
        _reject("QMC replicate count exceeds API limit")
    coordinates = config.effective_num_samples * config.dimension * config.num_randomizations
    if coordinates > settings.max_qmc_total_coordinates:
        _reject("QMC total coordinate count exceeds API limit")


@pricing_router.post("/european", response_model=QMCPricingResult)
def european(
    request: QMCEuropeanRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> QMCPricingResult:
    _enforce_config(request.qmc, settings)
    return price_european_qmc(request)


@pricing_router.post("/asian", response_model=QMCPricingResult)
def asian(
    request: QMCAsianRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> QMCPricingResult:
    _enforce_config(request.qmc, settings)
    if len(request.contract.future_monitoring_times) > settings.max_asian_monitoring_points:
        _reject("QMC Asian monitoring count exceeds API limit")
    return price_asian_qmc(request)


@pricing_router.post("/basket", response_model=QMCPricingResult)
def basket(
    request: QMCBasketRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> QMCPricingResult:
    _enforce_config(request.qmc, settings)
    if request.market.num_assets > settings.max_multi_asset_count:
        _reject("QMC basket asset count exceeds API limit")
    return price_basket_qmc(request)


@analytics_router.post("/sampling", response_model=SamplingDiagnosticsResult)
def diagnostics(
    request: SamplingDiagnosticsRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> SamplingDiagnosticsResult:
    _enforce_config(request.config, settings)
    if request.max_output_points > settings.max_qmc_diagnostic_output_points:
        _reject("QMC diagnostic output count exceeds API limit")
    return sampling_diagnostics(request)


@analytics_router.post("/convergence", response_model=QMCConvergenceResult)
def convergence(
    request: QMCConvergenceRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> QMCConvergenceResult:
    count = request.maximum_power - request.minimum_power + 1
    if count > settings.max_qmc_convergence_points:
        _reject("QMC convergence point count exceeds API limit")
    maximum_samples = 1 << request.maximum_power
    if maximum_samples > settings.max_qmc_samples:
        _reject("QMC convergence sample count exceeds API limit")
    if request.randomized_replicates > settings.max_qmc_replicates:
        _reject("QMC convergence replicate count exceeds API limit")
    total_coordinates = sum(
        1 << power for power in range(request.minimum_power, request.maximum_power + 1)
    ) * (request.randomized_replicates + 2)
    if total_coordinates > settings.max_qmc_total_coordinates:
        _reject("QMC convergence coordinate count exceeds API limit")
    return qmc_convergence(request)


@analytics_router.post("/compare", response_model=QMCComparisonResult)
def compare(
    request: QMCComparisonRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> QMCComparisonResult:
    effective_samples = (
        1 << (request.num_samples - 1).bit_length()
        if request.power_of_two_policy is PowerOfTwoPolicy.ROUND_UP
        else request.num_samples
    )
    if effective_samples > settings.max_qmc_samples:
        _reject("QMC comparison sample count exceeds API limit")
    if request.randomized_replicates > settings.max_qmc_replicates:
        _reject("QMC comparison replicate count exceeds API limit")
    total_coordinates = (
        request.num_samples + effective_samples + effective_samples * request.randomized_replicates
    )
    if total_coordinates > settings.max_qmc_total_coordinates:
        _reject("QMC comparison coordinate count exceeds API limit")
    return compare_qmc_methods(request)
