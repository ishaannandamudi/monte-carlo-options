"""Thin HTTP adapters for volatility-smile and surface research services."""

import logging
from datetime import datetime
from typing import Annotated, Self

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.analytics.volatility_surface import (
    atm_forward_term_structure,
    diagnose_surface,
    surface_quality_summary,
)
from backend.app.core.config import Settings, get_settings
from backend.app.core.exceptions import VolatilitySurfaceError
from backend.app.models.market_data import OptionChainSnapshot, QuoteQualityConfig
from backend.app.models.results import EnrichedOptionChain
from backend.app.models.volatility import ImpliedVolatilityConfig
from backend.app.models.volatility_surface import (
    ATMTermStructureResult,
    SmileBuildResult,
    SmileConstructionConfig,
    SmileInterpolationConfig,
    SmileQueryResult,
    SurfaceBuildResult,
    SurfaceDiagnosticConfig,
    SurfaceDiagnosticsResult,
    SurfaceQueryResult,
    VolatilityPoint,
    VolatilitySmile,
    VolatilitySurface,
)
from backend.app.volatility_surface.core import (
    build_market_surface,
    build_smile,
    build_surface,
    build_surface_from_option_chain,
    construct_market_smiles,
    query_smile,
    query_surface,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/analytics/volatility", tags=["volatility-surface"])


class APIModel(BaseModel):
    """Strict finite request model."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class SmileBuildRequest(APIModel):
    """Build one smile from normalized points or one enriched-chain expiration."""

    points: list[VolatilityPoint] | None = None
    enriched_chain: EnrichedOptionChain | None = None
    underlying_symbol: str | None = None
    valuation_timestamp: datetime | None = None
    expiration: datetime | None = None
    maturity_years: float | None = Field(default=None, gt=0.0)
    spot: float | None = Field(default=None, gt=0.0)
    risk_free_rate: float | None = None
    dividend_yield: float | None = None
    construction_config: SmileConstructionConfig = Field(default_factory=SmileConstructionConfig)

    @model_validator(mode="after")
    def validate_mode(self) -> Self:
        if (self.points is None) == (self.enriched_chain is None):
            raise ValueError("provide exactly one of points or enriched_chain")
        if self.enriched_chain is not None and self.expiration is None:
            raise ValueError("expiration is required when building from an enriched chain")
        if self.points is not None and any(
            value is None
            for value in (
                self.maturity_years,
                self.spot,
                self.risk_free_rate,
                self.dividend_yield,
            )
        ):
            raise ValueError(
                "maturity_years, spot, risk_free_rate, and dividend_yield are required with points"
            )
        return self


class SmileQueryRequest(APIModel):
    """Query one serialized smile at exactly one strike coordinate."""

    smile: VolatilitySmile
    strike: float | None = Field(default=None, gt=0.0)
    log_moneyness: float | None = None
    interpolation_config: SmileInterpolationConfig = Field(default_factory=SmileInterpolationConfig)


class SurfaceBuildRequest(APIModel):
    """Build from smiles, an enriched chain, or a normalized Phase 5 chain."""

    smiles: list[VolatilitySmile] | None = None
    enriched_chain: EnrichedOptionChain | None = None
    option_chain: OptionChainSnapshot | None = None
    valuation_time: datetime | None = None
    risk_free_rate: float | None = None
    dividend_yield: float = 0.0
    quote_quality_config: QuoteQualityConfig | None = None
    iv_solver_config: ImpliedVolatilityConfig = Field(default_factory=ImpliedVolatilityConfig)
    construction_config: SmileConstructionConfig = Field(default_factory=SmileConstructionConfig)
    interpolation_config: SmileInterpolationConfig = Field(default_factory=SmileInterpolationConfig)

    @model_validator(mode="after")
    def validate_mode(self) -> Self:
        modes = sum(
            value is not None for value in (self.smiles, self.enriched_chain, self.option_chain)
        )
        if modes != 1:
            raise ValueError("provide exactly one of smiles, enriched_chain, or option_chain")
        if self.option_chain is not None and self.risk_free_rate is None:
            raise ValueError("risk_free_rate is required with option_chain")
        return self


class SurfaceQueryRequest(APIModel):
    """Query a serialized surface at one maturity and one strike coordinate."""

    surface: VolatilitySurface
    maturity_years: float = Field(gt=0.0)
    strike: float | None = Field(default=None, gt=0.0)
    log_moneyness: float | None = None
    interpolation_config: SmileInterpolationConfig | None = None


class SurfaceDiagnosticsRequest(APIModel):
    """Run finite grid diagnostics against a serialized surface."""

    surface: VolatilitySurface
    diagnostic_config: SurfaceDiagnosticConfig = Field(default_factory=SurfaceDiagnosticConfig)


class ATMTermStructureRequest(APIModel):
    """Extract unsmoothed ATM-forward values from a surface."""

    surface: VolatilitySurface


def _enforce_smile_limit(point_count: int, settings: Settings) -> None:
    if point_count > settings.max_smile_points:
        raise VolatilitySurfaceError(
            f"smile has {point_count} points; API limit is {settings.max_smile_points}"
        )


def _enforce_surface_limits(smiles: list[VolatilitySmile], settings: Settings) -> None:
    if len(smiles) > settings.max_surface_smiles:
        raise VolatilitySurfaceError(
            f"surface has {len(smiles)} smiles; API limit is {settings.max_surface_smiles}"
        )
    for smile in smiles:
        _enforce_smile_limit(len(smile.points), settings)


@router.post("/smile/build", response_model=SmileBuildResult)
def smile_build(
    request: SmileBuildRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> SmileBuildResult:
    """Construct one deterministic smile from normalized or market observations."""

    if request.points is not None:
        _enforce_smile_limit(len(request.points), settings)
        if (
            request.maturity_years is None
            or request.spot is None
            or request.risk_free_rate is None
            or request.dividend_yield is None
        ):
            raise VolatilitySurfaceError("direct point smile metadata is incomplete")
        return build_smile(
            points=request.points,
            maturity_years=request.maturity_years,
            spot=request.spot,
            risk_free_rate=request.risk_free_rate,
            dividend_yield=request.dividend_yield,
            underlying_symbol=request.underlying_symbol,
            valuation_timestamp=request.valuation_timestamp,
            expiration=request.expiration,
            minimum_points=request.construction_config.minimum_points,
        )
    chain = request.enriched_chain
    if chain is None or request.expiration is None:
        raise VolatilitySurfaceError("market smile request is incomplete")
    if len(chain.quotes) > settings.max_option_chain_contracts:
        raise VolatilitySurfaceError("enriched chain exceeds API contract limit")
    smiles, diagnostics = construct_market_smiles(chain, request.construction_config)
    selected = [smile for smile in smiles if smile.expiration == request.expiration]
    if not selected:
        raise VolatilitySurfaceError("requested expiration did not produce an eligible smile")
    _enforce_smile_limit(len(selected[0].points), settings)
    return SmileBuildResult(smile=selected[0], diagnostics=diagnostics)


@router.post("/smile/query", response_model=SmileQueryResult)
def smile_query(
    request: SmileQueryRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> SmileQueryResult:
    """Query a smile in total-variance space."""

    _enforce_smile_limit(len(request.smile.points), settings)
    return query_smile(
        request.smile,
        strike=request.strike,
        log_moneyness=request.log_moneyness,
        config=request.interpolation_config,
    )


@router.post("/surface/build", response_model=SurfaceBuildResult)
def surface_build(
    request: SurfaceBuildRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> SurfaceBuildResult:
    """Build a maturity-sorted surface and summarize interpolation coverage."""

    logger.info("volatility_surface_build_start")
    if request.smiles is not None:
        _enforce_surface_limits(request.smiles, settings)
        surface = build_surface(
            request.smiles,
            interpolation_config=request.interpolation_config,
        )
        diagnostics = surface.construction_diagnostics
    elif request.enriched_chain is not None:
        chain = request.enriched_chain
        if len(chain.quotes) > settings.max_option_chain_contracts:
            raise VolatilitySurfaceError("enriched chain exceeds API contract limit")
        surface, diagnostics = build_market_surface(
            chain,
            construction_config=request.construction_config,
            interpolation_config=request.interpolation_config,
        )
        _enforce_surface_limits(surface.smiles, settings)
    else:
        raw_chain = request.option_chain
        if raw_chain is None or request.risk_free_rate is None:
            raise VolatilitySurfaceError("normalized market surface request is incomplete")
        if len(raw_chain.quotes) > settings.max_option_chain_contracts:
            raise VolatilitySurfaceError("option chain exceeds API contract limit")
        surface, diagnostics = build_surface_from_option_chain(
            raw_chain,
            valuation_time=request.valuation_time,
            risk_free_rate=request.risk_free_rate,
            dividend_yield=request.dividend_yield,
            quote_quality_config=request.quote_quality_config,
            iv_config=request.iv_solver_config,
            construction_config=request.construction_config,
            interpolation_config=request.interpolation_config,
        )
        _enforce_surface_limits(surface.smiles, settings)
    quality = surface_quality_summary(surface)
    logger.info(
        "volatility_surface_build_complete smiles=%d points=%d excluded=%d",
        quality.smile_count,
        quality.total_iv_points,
        quality.excluded_point_count,
    )
    return SurfaceBuildResult(
        surface=surface,
        quality_summary=quality,
        diagnostics=diagnostics,
    )


@router.post("/surface/query", response_model=SurfaceQueryResult)
def surface_query(
    request: SurfaceQueryRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> SurfaceQueryResult:
    """Evaluate one two-dimensional surface point."""

    _enforce_surface_limits(request.surface.smiles, settings)
    return query_surface(
        request.surface,
        maturity_years=request.maturity_years,
        strike=request.strike,
        log_moneyness=request.log_moneyness,
        config=request.interpolation_config,
    )


@router.post("/surface/diagnostics", response_model=SurfaceDiagnosticsResult)
def surface_diagnostics(
    request: SurfaceDiagnosticsRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> SurfaceDiagnosticsResult:
    """Run static-arbitrage diagnostics without repairing the surface."""

    _enforce_surface_limits(request.surface.smiles, settings)
    grid_size = (
        len(request.diagnostic_config.log_moneyness_grid)
        if request.diagnostic_config.log_moneyness_grid is not None
        else request.diagnostic_config.default_grid_size
    )
    work = grid_size * len(request.surface.smiles)
    if work > settings.max_diagnostic_grid_points:
        raise VolatilitySurfaceError(
            f"diagnostic grid has {work} points; API limit is {settings.max_diagnostic_grid_points}"
        )
    return diagnose_surface(request.surface, request.diagnostic_config)


@router.post("/surface/atm-term-structure", response_model=ATMTermStructureResult)
def surface_atm_term_structure(
    request: ATMTermStructureRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> ATMTermStructureResult:
    """Return ATM-forward IV and total variance at each surface maturity."""

    _enforce_surface_limits(request.surface.smiles, settings)
    return atm_forward_term_structure(request.surface)
