"""FastAPI adapters for CRR pricing, comparison, and convergence."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from backend.app.analytics.binomial import (
    compare_binomial_to_black_scholes,
    run_binomial_convergence,
)
from backend.app.core.config import Settings, get_settings
from backend.app.models.binomial import (
    BinomialComparison,
    BinomialConvergenceResult,
    BinomialPricingResult,
    BinomialTreeConfig,
    TreeModel,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract
from backend.app.pricing.binomial import binomial_price

logger = logging.getLogger(__name__)
pricing_router = APIRouter(prefix="/api/pricing", tags=["pricing"])
analytics_router = APIRouter(prefix="/api/analytics/binomial", tags=["binomial"])


class BinomialRequestModel(BaseModel):
    """Strict finite request policy for Phase 7 endpoints."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class BinomialPricingRequest(BinomialRequestModel):
    contract: OptionContract
    market: MarketEnvironment
    tree: BinomialTreeConfig = Field(default_factory=BinomialTreeConfig)


class BinomialConvergenceRequest(BinomialRequestModel):
    contract: OptionContract
    market: MarketEnvironment
    step_counts: list[Annotated[int, Field(gt=0)]] = Field(min_length=1)
    model: TreeModel = TreeModel.CRR
    probability_tolerance: float = Field(default=1e-12, ge=0.0, le=1e-6)
    arbitrage_tolerance: float = Field(default=1e-10, ge=0.0, le=1e-4)


def _enforce_step_limit(steps: int, settings: Settings) -> None:
    if steps > settings.max_binomial_steps:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"steps exceeds API limit of {settings.max_binomial_steps}",
        )


@pricing_router.post("/binomial", response_model=BinomialPricingResult)
def price_binomial(
    request: BinomialPricingRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> BinomialPricingResult:
    """Price one European option with a bounded CRR tree."""

    _enforce_step_limit(request.tree.steps, settings)
    logger.info(
        "pricing_request engine=binomial model=%s option_type=%s steps=%d",
        request.tree.model.value,
        request.contract.option_type.value,
        request.tree.steps,
    )
    return binomial_price(request.contract, request.market, request.tree)


@analytics_router.post("/compare", response_model=BinomialComparison)
def compare_binomial(
    request: BinomialPricingRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> BinomialComparison:
    """Compare CRR price and native Greeks with Black-Scholes values."""

    _enforce_step_limit(request.tree.steps, settings)
    logger.info(
        "binomial_comparison_request model=%s steps=%d",
        request.tree.model.value,
        request.tree.steps,
    )
    return compare_binomial_to_black_scholes(request.contract, request.market, request.tree)


@analytics_router.post("/convergence", response_model=BinomialConvergenceResult)
def binomial_convergence(
    request: BinomialConvergenceRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> BinomialConvergenceResult:
    """Run a bounded deterministic CRR step-convergence experiment."""

    if len(request.step_counts) > settings.max_binomial_convergence_points:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "step_counts exceeds API limit of "
                f"{settings.max_binomial_convergence_points} points"
            ),
        )
    for steps in request.step_counts:
        _enforce_step_limit(steps, settings)
    logger.info(
        "binomial_convergence_request model=%s points=%d",
        request.model.value,
        len(request.step_counts),
    )
    return run_binomial_convergence(
        request.contract,
        request.market,
        request.step_counts,
        model=request.model,
        probability_tolerance=request.probability_tolerance,
        arbitrage_tolerance=request.arbitrage_tolerance,
    )
