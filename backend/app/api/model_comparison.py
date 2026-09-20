"""Resource-bounded HTTP adapters for Phase 15 model-comparison research."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.analytics.model_comparison import (
    build_model_risk_report,
    compare_models,
    compare_surface,
    compute_frontier,
    evaluate_market_fit,
    run_consistency_suite,
)
from backend.app.core.config import Settings, get_settings
from backend.app.model_comparison.registry import capability_registry
from backend.app.models.model_comparison import (
    ComputeBudgetType,
    ComputeFrontierRequest,
    ComputeFrontierResult,
    ConsistencyRequest,
    ConsistencyResult,
    MarketFitRequest,
    MarketFitResult,
    ModelCapabilities,
    ModelComparisonRequest,
    ModelComparisonResult,
    ModelRiskReport,
    ModelRiskRequest,
    PricingModel,
    SurfaceComparisonRequest,
)

router = APIRouter(prefix="/api/analytics/models", tags=["model-comparison"])


def _reject(message: str) -> None:
    raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, message)


def _enforce_comparison(request: ModelComparisonRequest, settings: Settings) -> None:
    if len(request.config.selected_models) > settings.max_model_comparison_models:
        _reject("model comparison count exceeds API limit")
    if request.config.runtime_repetitions > settings.max_model_comparison_runtime_repetitions:
        _reject("model comparison runtime repetitions exceed API limit")
    if request.config.max_result_points > settings.max_model_comparison_surface_points:
        _reject("model comparison result-size limit exceeds API limit")
    if request.gbm_monte_carlo.num_simulations > settings.max_api_simulations:
        _reject("GBM Monte Carlo simulations exceed API limit")
    if request.tree.steps > settings.max_binomial_steps:
        _reject("tree steps exceed API limit")
    if request.heston_monte_carlo.num_simulations > settings.max_heston_paths:
        _reject("Heston paths exceed API limit")
    if request.heston_monte_carlo.num_time_steps > settings.max_heston_time_steps:
        _reject("Heston time steps exceed API limit")
    if (
        request.heston_monte_carlo.num_simulations * request.heston_monte_carlo.num_time_steps
        > settings.max_heston_total_path_nodes
    ):
        _reject("Heston path-time nodes exceed API limit")
    selected = set(request.config.selected_models)
    if (
        PricingModel.MERTON_MONTE_CARLO in selected
        and request.merton_monte_carlo.num_simulations > settings.max_merton_paths
    ):
        _reject("Merton paths exceed API limit")
    if (
        PricingModel.MERTON_ANALYTICAL in selected
        and request.merton_series.max_terms > settings.max_merton_series_terms
    ):
        _reject("Merton analytical series terms exceed API limit")
    lsm = request.longstaff_schwartz
    if lsm.num_paths > settings.max_lsm_paths or lsm.training_path_count > settings.max_lsm_paths:
        _reject("Longstaff-Schwartz paths exceed API limit")
    if lsm.num_time_steps > settings.max_lsm_time_steps:
        _reject("Longstaff-Schwartz time steps exceed API limit")
    if lsm.total_simulated_paths * lsm.num_time_steps > settings.max_lsm_total_path_nodes:
        _reject("Longstaff-Schwartz path-time nodes exceed API limit")


def _enforce_market_fit(request: MarketFitRequest, settings: Settings) -> None:
    if len(request.selected_models) > settings.max_model_comparison_models:
        _reject("model comparison count exceeds API limit")
    if len(request.dataset.observations) > settings.max_model_comparison_observations:
        _reject("model comparison observation count exceeds API limit")
    if request.max_returned_residuals > settings.max_model_comparison_surface_points:
        _reject("model comparison residual count exceeds API limit")
    if request.gbm_monte_carlo.num_simulations > settings.max_api_simulations:
        _reject("GBM Monte Carlo simulations exceed API limit")
    if request.heston_monte_carlo.num_simulations > settings.max_heston_paths:
        _reject("Heston paths exceed API limit")
    if request.heston_monte_carlo.num_time_steps > settings.max_heston_time_steps:
        _reject("Heston time steps exceed API limit")
    selected = set(request.selected_models)
    if (
        PricingModel.MERTON_MONTE_CARLO in selected
        and request.merton_monte_carlo.num_simulations > settings.max_merton_paths
    ):
        _reject("Merton paths exceed API limit")
    if (
        PricingModel.MERTON_ANALYTICAL in selected
        and request.merton_series.max_terms > settings.max_merton_series_terms
    ):
        _reject("Merton analytical series terms exceed API limit")
    observations = len(request.dataset.observations)
    if (
        PricingModel.MONTE_CARLO_GBM in request.selected_models
        and observations * request.gbm_monte_carlo.num_simulations > settings.max_api_simulations
    ):
        _reject("aggregate GBM market-fit paths exceed API limit")
    if (
        PricingModel.HESTON_MONTE_CARLO in request.selected_models
        and observations
        * request.heston_monte_carlo.num_simulations
        * request.heston_monte_carlo.num_time_steps
        > settings.max_heston_total_path_nodes
    ):
        _reject("aggregate Heston market-fit path-time nodes exceed API limit")
    if (
        PricingModel.MERTON_MONTE_CARLO in request.selected_models
        and observations * request.merton_monte_carlo.num_simulations > settings.max_merton_paths
    ):
        _reject("aggregate Merton market-fit paths exceed API limit")


@router.get("/registry", response_model=list[ModelCapabilities])
def model_registry_endpoint() -> list[ModelCapabilities]:
    return list(capability_registry())


@router.post("/compare", response_model=ModelComparisonResult)
def compare_models_endpoint(
    request: ModelComparisonRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> ModelComparisonResult:
    _enforce_comparison(request, settings)
    return compare_models(request)


@router.post("/market-fit", response_model=MarketFitResult)
def market_fit_endpoint(
    request: MarketFitRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> MarketFitResult:
    _enforce_market_fit(request, settings)
    return evaluate_market_fit(request)


@router.post("/risk", response_model=ModelRiskReport)
def model_risk_endpoint(
    request: ModelRiskRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> ModelRiskReport:
    _enforce_comparison(request.comparison, settings)
    if request.market_fit is not None:
        _enforce_market_fit(request.market_fit, settings)
    if len(request.calibration_ensemble) > settings.max_heston_calibration_starts:
        _reject("calibration ensemble size exceeds API limit")
    return build_model_risk_report(request)


@router.post("/compute-frontier", response_model=ComputeFrontierResult)
def compute_frontier_endpoint(
    request: ComputeFrontierRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> ComputeFrontierResult:
    _enforce_comparison(request.comparison, settings)
    if len(request.budgets) > settings.max_model_comparison_experiment_points:
        _reject("model comparison experiment point count exceeds API limit")
    budget_type = request.budget_type
    if request.model is PricingModel.MONTE_CARLO_GBM:
        maximum = settings.max_api_simulations
    elif request.model is PricingModel.HESTON_MONTE_CARLO:
        if budget_type is ComputeBudgetType.TIME_STEPS:
            maximum = settings.max_heston_time_steps
            nodes = max(request.budgets) * request.comparison.heston_monte_carlo.num_simulations
        else:
            maximum = settings.max_heston_paths
            nodes = max(request.budgets) * request.comparison.heston_monte_carlo.num_time_steps
        if nodes > settings.max_heston_total_path_nodes:
            _reject("compute-frontier Heston path-time nodes exceed API limit")
    elif request.model is PricingModel.MERTON_MONTE_CARLO:
        maximum = settings.max_merton_paths
    elif request.model is PricingModel.LONGSTAFF_SCHWARTZ:
        lsm = request.comparison.longstaff_schwartz
        if budget_type is ComputeBudgetType.TIME_STEPS:
            maximum = settings.max_lsm_time_steps
            frontier_lsm = lsm.model_copy(update={"num_time_steps": max(request.budgets)})
        else:
            maximum = settings.max_lsm_paths
            frontier_lsm = lsm.model_copy(update={"num_paths": max(request.budgets)})
        if (
            frontier_lsm.total_simulated_paths * frontier_lsm.num_time_steps
            > settings.max_lsm_total_path_nodes
        ):
            _reject("compute-frontier LSM path-time nodes exceed API limit")
    else:
        maximum = settings.max_binomial_steps
    if max(request.budgets) > maximum:
        _reject("compute-frontier budget exceeds API limit")
    return compute_frontier(request)


@router.post("/consistency", response_model=ConsistencyResult)
def consistency_endpoint(
    request: ConsistencyRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> ConsistencyResult:
    if request.simulation.num_simulations > settings.max_api_simulations:
        _reject("consistency GBM simulations exceed API limit")
    if request.tree.steps > settings.max_binomial_steps:
        _reject("consistency tree steps exceed API limit")
    if request.heston_monte_carlo.num_simulations > settings.max_heston_paths:
        _reject("consistency Heston paths exceed API limit")
    if request.heston_monte_carlo.num_time_steps > settings.max_heston_time_steps:
        _reject("consistency Heston time steps exceed API limit")
    if (
        request.heston_monte_carlo.num_simulations * request.heston_monte_carlo.num_time_steps
        > settings.max_heston_total_path_nodes
    ):
        _reject("consistency Heston path-time nodes exceed API limit")
    return run_consistency_suite(request)


@router.post("/surface-comparison", response_model=MarketFitResult)
def surface_comparison_endpoint(
    request: SurfaceComparisonRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> MarketFitResult:
    if len(request.selected_models) > settings.max_model_comparison_models:
        _reject("model comparison count exceeds API limit")
    point_count = sum(len(smile.points) for smile in request.surface.smiles)
    if point_count > settings.max_model_comparison_surface_points:
        _reject("model comparison surface point count exceeds API limit")
    if request.max_returned_residuals > settings.max_model_comparison_surface_points:
        _reject("model comparison residual count exceeds API limit")
    if request.heston_monte_carlo.num_simulations > settings.max_heston_paths:
        _reject("Heston paths exceed API limit")
    if request.heston_monte_carlo.num_time_steps > settings.max_heston_time_steps:
        _reject("Heston time steps exceed API limit")
    selected = set(request.selected_models)
    if (
        PricingModel.MERTON_MONTE_CARLO in selected
        and request.merton_monte_carlo.num_simulations > settings.max_merton_paths
    ):
        _reject("Merton paths exceed API limit")
    if (
        PricingModel.MERTON_ANALYTICAL in selected
        and request.merton_series.max_terms > settings.max_merton_series_terms
    ):
        _reject("Merton analytical series terms exceed API limit")
    if (
        PricingModel.MONTE_CARLO_GBM in request.selected_models
        and point_count * request.gbm_monte_carlo.num_simulations > settings.max_api_simulations
    ):
        _reject("aggregate GBM surface-comparison paths exceed API limit")
    if (
        PricingModel.HESTON_MONTE_CARLO in request.selected_models
        and point_count
        * request.heston_monte_carlo.num_simulations
        * request.heston_monte_carlo.num_time_steps
        > settings.max_heston_total_path_nodes
    ):
        _reject("aggregate Heston surface-comparison path-time nodes exceed API limit")
    if (
        PricingModel.MERTON_MONTE_CARLO in request.selected_models
        and point_count * request.merton_monte_carlo.num_simulations > settings.max_merton_paths
    ):
        _reject("aggregate Merton surface-comparison paths exceed API limit")
    return compare_surface(request)
