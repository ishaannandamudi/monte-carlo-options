"""Thin HTTP adapters for quantitative analytics experiments."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from backend.app.analytics.convergence import run_convergence
from backend.app.analytics.greeks import compare_greek_estimators
from backend.app.analytics.sensitivity import run_sensitivity
from backend.app.analytics.variance_reduction import compare_variance_reduction_methods
from backend.app.api.pricing import _enforce_simulation_limit
from backend.app.core.config import Settings, get_settings
from backend.app.models.greeks import Greek, GreekBumpConfig, SensitivityParameter
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract
from backend.app.models.results import (
    ConvergenceResult,
    GreekComparisonResult,
    SensitivityResult,
    VarianceReductionComparisonResult,
)
from backend.app.models.simulation import (
    ConfidenceLevel,
    NonNegativeInt,
    PositiveInt,
    SimulationConfig,
    VarianceReduction,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/analytics", tags=["analytics"])


class ConvergenceRequest(BaseModel):
    """Explicit convergence experiment request."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    contract: OptionContract
    market: MarketEnvironment
    simulation_counts: list[Annotated[int, Field(gt=0)]] = Field(min_length=1)
    root_seed: NonNegativeInt = 42
    variance_reduction: VarianceReduction = VarianceReduction.ANTITHETIC
    confidence_level: ConfidenceLevel = 0.95


class VarianceReductionRequest(BaseModel):
    """One fixed-budget experiment across every supported method."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    contract: OptionContract
    market: MarketEnvironment
    num_simulations: PositiveInt = 100_000
    root_seed: NonNegativeInt = 42
    confidence_level: ConfidenceLevel = 0.95


class GreekComparisonRequest(BaseModel):
    """One Greek compared across its supported stochastic estimators."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    contract: OptionContract
    market: MarketEnvironment
    simulation: SimulationConfig
    greek: Greek
    bump_config: GreekBumpConfig = Field(default_factory=GreekBumpConfig)


class SensitivityRequest(BaseModel):
    """One-dimensional analytical price and Greek grid."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    contract: OptionContract
    market: MarketEnvironment
    parameter: SensitivityParameter
    values: list[float] = Field(min_length=1)


@router.post("/convergence", response_model=ConvergenceResult)
def convergence_analysis(
    request: ConvergenceRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> ConvergenceResult:
    """Run bounded, independently seeded convergence points."""

    if len(request.simulation_counts) > settings.max_convergence_points:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"simulation_counts exceeds API limit of {settings.max_convergence_points} points"
            ),
        )
    for count in request.simulation_counts:
        _enforce_simulation_limit(count, settings)

    logger.info(
        "convergence_request points=%d variance_reduction=%s",
        len(request.simulation_counts),
        request.variance_reduction.value,
    )
    return run_convergence(
        request.contract,
        request.market,
        request.simulation_counts,
        root_seed=request.root_seed,
        variance_reduction=request.variance_reduction,
        confidence_level=request.confidence_level,
    )


@router.post("/variance-reduction", response_model=VarianceReductionComparisonResult)
def variance_reduction_analysis(
    request: VarianceReductionRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> VarianceReductionComparisonResult:
    """Compare all four methods under one bounded total path count."""

    _enforce_simulation_limit(request.num_simulations, settings)
    logger.info("variance_reduction_request simulations=%d", request.num_simulations)
    return compare_variance_reduction_methods(
        request.contract,
        request.market,
        request.num_simulations,
        root_seed=request.root_seed,
        confidence_level=request.confidence_level,
    )


@router.post("/greeks/compare", response_model=GreekComparisonResult)
def greek_comparison_analysis(
    request: GreekComparisonRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> GreekComparisonResult:
    """Compare one Greek with its analytical Black-Scholes benchmark."""

    _enforce_simulation_limit(request.simulation.num_simulations, settings)
    logger.info(
        "greek_comparison_request greek=%s simulations=%d variance_reduction=%s",
        request.greek.value,
        request.simulation.num_simulations,
        request.simulation.variance_reduction.value,
    )
    return compare_greek_estimators(
        request.contract,
        request.market,
        request.simulation,
        request.greek,
        request.bump_config,
    )


@router.post("/sensitivity", response_model=SensitivityResult)
def sensitivity_analysis(
    request: SensitivityRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> SensitivityResult:
    """Return a bounded analytical sensitivity grid."""

    if len(request.values) > settings.max_sensitivity_points:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"values exceeds API limit of {settings.max_sensitivity_points} points",
        )
    logger.info(
        "sensitivity_request parameter=%s points=%d",
        request.parameter.value,
        len(request.values),
    )
    return run_sensitivity(
        request.contract,
        request.market,
        request.parameter,
        request.values,
    )
