"""Thin HTTP adapter for convergence experiments."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from backend.app.analytics.convergence import run_convergence
from backend.app.api.pricing import _enforce_simulation_limit
from backend.app.core.config import Settings, get_settings
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract
from backend.app.models.results import ConvergenceResult
from backend.app.models.simulation import ConfidenceLevel, NonNegativeInt, VarianceReduction

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
