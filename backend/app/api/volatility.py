"""Thin HTTP adapters for implied-volatility and synthetic-smile services."""

import logging
from typing import Annotated, Self

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.analytics.volatility import (
    compare_implied_volatility_solvers,
    run_volatility_smile,
    solve_implied_volatility_batch,
)
from backend.app.core.config import Settings, get_settings
from backend.app.models.option import OptionContract, OptionType
from backend.app.models.results import (
    ImpliedVolatilityBatchResult,
    ImpliedVolatilityResult,
    ImpliedVolatilitySolverComparisonResult,
    VolatilitySmileResult,
)
from backend.app.models.volatility import (
    ImpliedVolatilityConfig,
    ImpliedVolatilityMarket,
    ImpliedVolatilityObservation,
)
from backend.app.pricing.implied_volatility import solve_implied_volatility

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/volatility", tags=["volatility"])
analytics_router = APIRouter(prefix="/api/analytics/volatility", tags=["analytics"])

NonNegativeFloat = Annotated[float, Field(ge=0.0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]


class VolatilityRequestModel(BaseModel):
    """Strict request-body policy for Phase 4 endpoints."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ImpliedVolatilityInputs(VolatilityRequestModel):
    """One observed option with known rates and unknown volatility."""

    contract: OptionContract
    risk_free_rate: float
    dividend_yield: float = 0.0
    observed_price: NonNegativeFloat

    def to_observation(self) -> ImpliedVolatilityObservation:
        return ImpliedVolatilityObservation(
            contract=self.contract,
            market=ImpliedVolatilityMarket(
                risk_free_rate=self.risk_free_rate,
                dividend_yield=self.dividend_yield,
            ),
            observed_price=self.observed_price,
        )


class ImpliedVolatilityRequest(ImpliedVolatilityInputs):
    """One price inversion and its explicit solver policy."""

    solver_config: ImpliedVolatilityConfig = Field(default_factory=ImpliedVolatilityConfig)


class ImpliedVolatilityBatchRequest(VolatilityRequestModel):
    """Ordered observations solved under one common policy."""

    observations: list[ImpliedVolatilityInputs] = Field(min_length=1)
    solver_config: ImpliedVolatilityConfig = Field(default_factory=ImpliedVolatilityConfig)


class VolatilitySmileRequest(VolatilityRequestModel):
    """One synthetic single-maturity strike slice."""

    spot: PositiveFloat
    maturity_years: NonNegativeFloat
    risk_free_rate: float
    dividend_yield: float = 0.0
    option_type: OptionType
    strikes: list[PositiveFloat] = Field(min_length=1)
    option_prices: list[NonNegativeFloat] = Field(min_length=1)
    solver_config: ImpliedVolatilityConfig = Field(default_factory=ImpliedVolatilityConfig)

    @model_validator(mode="after")
    def validate_matching_lengths(self) -> Self:
        if len(self.strikes) != len(self.option_prices):
            raise ValueError("strikes and option_prices must have matching lengths")
        return self


class SolverComparisonRequest(ImpliedVolatilityInputs):
    """One observation and common controls for all three explicit solvers."""

    solver_config: ImpliedVolatilityConfig = Field(default_factory=ImpliedVolatilityConfig)


@router.post("/implied", response_model=ImpliedVolatilityResult)
def implied_volatility(request: ImpliedVolatilityRequest) -> ImpliedVolatilityResult:
    """Recover one European Black-Scholes implied volatility."""

    logger.info(
        "iv_request option_type=%s solver=%s",
        request.contract.option_type.value,
        request.solver_config.method.value,
    )
    result = solve_implied_volatility(request.to_observation(), request.solver_config)
    logger.info(
        "iv_complete requested=%s used=%s status=%s iterations=%d runtime_ms=%.6f",
        result.method_requested.value,
        result.method_used.value if result.method_used is not None else "NONE",
        result.status.value,
        result.iterations,
        result.runtime_ms,
    )
    return result


@router.post("/implied/batch", response_model=ImpliedVolatilityBatchResult)
def implied_volatility_batch(
    request: ImpliedVolatilityBatchRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> ImpliedVolatilityBatchResult:
    """Recover an ordered batch while preserving individual failures."""

    size = len(request.observations)
    if size > settings.max_iv_batch_size:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"observations exceeds API limit of {settings.max_iv_batch_size} items",
        )
    logger.info("iv_batch_request size=%d solver=%s", size, request.solver_config.method.value)
    result = solve_implied_volatility_batch(
        [item.to_observation() for item in request.observations],
        request.solver_config,
    )
    logger.info(
        "iv_batch_complete size=%d successes=%d failures=%d",
        result.total_count,
        result.success_count,
        result.failure_count,
    )
    return result


@analytics_router.post("/smile", response_model=VolatilitySmileResult)
def volatility_smile(
    request: VolatilitySmileRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> VolatilitySmileResult:
    """Recover a bounded synthetic smile slice in caller-supplied order."""

    size = len(request.strikes)
    if size > settings.max_smile_points:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"strikes exceeds API limit of {settings.max_smile_points} points",
        )
    logger.info("volatility_smile_request size=%d", size)
    result = run_volatility_smile(
        spot=request.spot,
        maturity_years=request.maturity_years,
        risk_free_rate=request.risk_free_rate,
        dividend_yield=request.dividend_yield,
        option_type=request.option_type,
        strikes=request.strikes,
        option_prices=request.option_prices,
        config=request.solver_config,
    )
    logger.info("volatility_smile_complete size=%d", len(result.points))
    return result


@analytics_router.post(
    "/solver-comparison",
    response_model=ImpliedVolatilitySolverComparisonResult,
)
def solver_comparison(request: SolverComparisonRequest) -> ImpliedVolatilitySolverComparisonResult:
    """Compare Brent, bisection, and Newton on one observed price."""

    logger.info("iv_solver_comparison_request option_type=%s", request.contract.option_type.value)
    result = compare_implied_volatility_solvers(
        request.to_observation(),
        request.solver_config,
    )
    logger.info(
        "iv_solver_comparison_complete successes=%d",
        sum(point.success for point in result.points),
    )
    return result
