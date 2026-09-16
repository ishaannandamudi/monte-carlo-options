"""Bounded HTTP adapters for American LSM pricing and research."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import Field

from backend.app.analytics.longstaff_schwartz import (
    basis_configs,
    compare_longstaff_schwartz,
    compare_lsm_bases,
    convergence_configs,
    run_lsm_convergence,
)
from backend.app.core.config import Settings, get_settings
from backend.app.models.binomial import BinomialTreeConfig
from backend.app.models.longstaff_schwartz import (
    BasisFamily,
    LongstaffSchwartzConfig,
    LongstaffSchwartzResult,
    LSMBasisComparisonResult,
    LSMComparison,
    LSMConvergenceMode,
    LSMConvergenceResult,
    LSMModel,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract
from backend.app.pricing.longstaff_schwartz import longstaff_schwartz_price

pricing_router = APIRouter(prefix="/api/pricing", tags=["pricing"])
analytics_router = APIRouter(
    prefix="/api/analytics/longstaff-schwartz", tags=["longstaff-schwartz"]
)


class LSMRequest(LSMModel):
    contract: OptionContract
    market: MarketEnvironment
    lsm: LongstaffSchwartzConfig = Field(default_factory=LongstaffSchwartzConfig)


class LSMBenchmarkRequest(LSMRequest):
    tree: BinomialTreeConfig = Field(default_factory=lambda: BinomialTreeConfig(steps=1_000))


class LSMComparisonRequest(LSMBenchmarkRequest):
    include_boundary_comparison: bool = False


class LSMConvergenceRequest(LSMBenchmarkRequest):
    mode: LSMConvergenceMode
    values: list[Annotated[int, Field(gt=0)]] = Field(min_length=1)


class LSMBasisRequest(LSMBenchmarkRequest):
    families: list[BasisFamily] = Field(
        default_factory=lambda: [BasisFamily.POWER, BasisFamily.LAGUERRE], min_length=1
    )
    degrees: list[Annotated[int, Field(ge=0, le=5)]] = Field(
        default_factory=lambda: [1, 2, 3, 4], min_length=1
    )


def _limits(
    configs: list[LongstaffSchwartzConfig],
    settings: Settings,
    tree: BinomialTreeConfig | None = None,
) -> None:
    if len(configs) > settings.max_lsm_convergence_points:
        raise HTTPException(422, "LSM experiment exceeds API convergence-point limit")
    if tree is not None and tree.steps > settings.max_binomial_steps:
        raise HTTPException(422, "tree steps exceeds API binomial step limit")
    nodes = 0
    for config in configs:
        if max(config.num_paths, config.training_path_count) > settings.max_lsm_paths:
            raise HTTPException(422, "LSM path count exceeds API limit")
        if config.num_time_steps > settings.max_lsm_time_steps:
            raise HTTPException(422, "LSM time-step count exceeds API limit")
        nodes += config.total_simulated_paths * (config.num_time_steps + 1)
    if nodes > settings.max_lsm_total_path_nodes:
        raise HTTPException(
            422, "LSM total path nodes exceeds API limit (training + valuation across all runs)"
        )


def _diagnostic_cap(config: LongstaffSchwartzConfig, settings: Settings) -> LongstaffSchwartzConfig:
    return config.model_copy(
        update={
            "max_diagnostic_dates": min(
                config.max_diagnostic_dates, settings.max_lsm_diagnostic_dates
            )
        }
    )


@pricing_router.post("/longstaff-schwartz", response_model=LongstaffSchwartzResult)
def price_lsm(
    request: LSMRequest, settings: Annotated[Settings, Depends(get_settings)]
) -> LongstaffSchwartzResult:
    config = _diagnostic_cap(request.lsm, settings)
    _limits([config], settings)
    return longstaff_schwartz_price(request.contract, request.market, config)


@analytics_router.post("/compare", response_model=LSMComparison)
def compare_lsm(
    request: LSMComparisonRequest, settings: Annotated[Settings, Depends(get_settings)]
) -> LSMComparison:
    config = _diagnostic_cap(request.lsm, settings)
    _limits([config], settings, request.tree)
    return compare_longstaff_schwartz(
        request.contract,
        request.market,
        config,
        request.tree,
        include_boundary_comparison=request.include_boundary_comparison,
    )


@analytics_router.post("/convergence", response_model=LSMConvergenceResult)
def convergence_lsm(
    request: LSMConvergenceRequest, settings: Annotated[Settings, Depends(get_settings)]
) -> LSMConvergenceResult:
    # Check size before constructing schedules or child seeds for large input lists.
    if len(request.values) > settings.max_lsm_convergence_points:
        raise HTTPException(422, "LSM experiment exceeds API convergence-point limit")
    config = _diagnostic_cap(request.lsm, settings)
    _limits(convergence_configs(config, request.values, request.mode), settings, request.tree)
    return run_lsm_convergence(
        request.contract, request.market, config, request.values, request.mode, request.tree
    )


@analytics_router.post("/basis-comparison", response_model=LSMBasisComparisonResult)
def bases_lsm(
    request: LSMBasisRequest, settings: Annotated[Settings, Depends(get_settings)]
) -> LSMBasisComparisonResult:
    if len(request.families) * len(request.degrees) > settings.max_lsm_convergence_points:
        raise HTTPException(422, "LSM experiment exceeds API convergence-point limit")
    config = _diagnostic_cap(request.lsm, settings)
    _limits(basis_configs(config, request.families, request.degrees), settings, request.tree)
    return compare_lsm_bases(
        request.contract, request.market, config, request.families, request.degrees, request.tree
    )
