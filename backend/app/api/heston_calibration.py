"""Resource-bounded HTTP adapters for Heston calibration and fit research."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from backend.app.analytics.heston_calibration import (
    analyze_heston_calibration_stability,
    validate_heston_calibration,
)
from backend.app.calibration.datasets import calibration_dataset_from_surface
from backend.app.calibration.heston import (
    calibrate_heston,
    evaluate_heston_calibration,
    generate_fitted_heston_surface,
)
from backend.app.core.config import Settings, get_settings
from backend.app.models.heston import HestonParameters
from backend.app.models.heston_calibration import (
    HestonCalibrationConfig,
    HestonCalibrationDataset,
    HestonCalibrationDiagnosticsResult,
    HestonCalibrationHoldoutConfig,
    HestonCalibrationResult,
    HestonCalibrationStabilityResult,
    HestonCalibrationValidationResult,
    HestonFittedSurfaceResult,
)
from backend.app.models.volatility_surface import VolatilitySurface

calibration_router = APIRouter(prefix="/api/calibration/heston", tags=["heston-calibration"])
analytics_router = APIRouter(
    prefix="/api/analytics/heston/calibration", tags=["heston-calibration-analytics"]
)


class CalibrationRequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class HestonCalibrationRequest(CalibrationRequestModel):
    dataset: HestonCalibrationDataset
    config: HestonCalibrationConfig = Field(default_factory=HestonCalibrationConfig)


class HestonSurfaceCalibrationRequest(CalibrationRequestModel):
    surface: VolatilitySurface
    config: HestonCalibrationConfig = Field(default_factory=HestonCalibrationConfig)


class HestonCalibrationDiagnosticsRequest(HestonCalibrationRequest):
    parameters: HestonParameters


class HestonCalibrationStabilityRequest(HestonCalibrationRequest):
    run_count: Annotated[int, Field(gt=0)] = 3


class HestonCalibrationValidationRequest(HestonCalibrationRequest):
    holdout: HestonCalibrationHoldoutConfig


def _enforce_config(config: HestonCalibrationConfig, settings: Settings) -> None:
    if config.multi_start_count > settings.max_heston_calibration_starts:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Heston calibration start count exceeds API limit",
        )
    if config.max_iterations > settings.max_heston_calibration_iterations:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Heston calibration iteration count exceeds API limit",
        )
    if config.max_returned_residuals > settings.max_heston_calibration_diagnostic_points:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Heston calibration diagnostic point count exceeds API limit",
        )


def _enforce_dataset(
    dataset: HestonCalibrationDataset,
    config: HestonCalibrationConfig,
    settings: Settings,
) -> None:
    if len(dataset.observations) > settings.max_heston_calibration_observations:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Heston calibration observation count exceeds API limit",
        )
    _enforce_config(config, settings)


@calibration_router.post("", response_model=HestonCalibrationResult)
def calibrate_heston_endpoint(
    request: HestonCalibrationRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> HestonCalibrationResult:
    _enforce_dataset(request.dataset, request.config, settings)
    return calibrate_heston(request.dataset, request.config)


@calibration_router.post("/surface", response_model=HestonCalibrationResult)
def calibrate_heston_surface_endpoint(
    request: HestonSurfaceCalibrationRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> HestonCalibrationResult:
    point_count = sum(len(smile.points) for smile in request.surface.smiles)
    if point_count > settings.max_heston_calibration_surface_points:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Heston calibration surface point count exceeds API limit",
        )
    dataset = calibration_dataset_from_surface(request.surface)
    _enforce_dataset(dataset, request.config, settings)
    return calibrate_heston(dataset, request.config)


@analytics_router.post("/diagnostics", response_model=HestonCalibrationDiagnosticsResult)
def heston_calibration_diagnostics_endpoint(
    request: HestonCalibrationDiagnosticsRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> HestonCalibrationDiagnosticsResult:
    _enforce_dataset(request.dataset, request.config, settings)
    return evaluate_heston_calibration(request.dataset, request.parameters, request.config)


@analytics_router.post("/fitted-surface", response_model=HestonFittedSurfaceResult)
def heston_fitted_surface_endpoint(
    request: HestonCalibrationDiagnosticsRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> HestonFittedSurfaceResult:
    _enforce_dataset(request.dataset, request.config, settings)
    if len(request.dataset.observations) > settings.max_heston_calibration_surface_points:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Heston fitted-surface point count exceeds API limit",
        )
    return generate_fitted_heston_surface(request.dataset, request.parameters, request.config)


@analytics_router.post("/stability", response_model=HestonCalibrationStabilityResult)
def heston_calibration_stability_endpoint(
    request: HestonCalibrationStabilityRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> HestonCalibrationStabilityResult:
    _enforce_dataset(request.dataset, request.config, settings)
    if request.run_count > settings.max_heston_calibration_starts:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Heston calibration stability run count exceeds API limit",
        )
    return analyze_heston_calibration_stability(request.dataset, request.config, request.run_count)


@analytics_router.post("/validate", response_model=HestonCalibrationValidationResult)
def heston_calibration_validation_endpoint(
    request: HestonCalibrationValidationRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> HestonCalibrationValidationResult:
    _enforce_dataset(request.dataset, request.config, settings)
    return validate_heston_calibration(request.dataset, request.config, request.holdout)
