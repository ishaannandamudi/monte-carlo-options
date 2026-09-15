"""Thin HTTP adapters for normalized option-chain analytics."""

import logging
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from backend.app.analytics.market_data import (
    analyze_put_call_parity,
    enrich_option_chain,
    extract_market_smile,
)
from backend.app.api.market_data import _enforce_chain_limit
from backend.app.core.config import Settings, get_settings
from backend.app.models.market_data import OptionChainSnapshot, QuoteQualityConfig
from backend.app.models.option import OptionType
from backend.app.models.results import (
    EnrichedOptionChain,
    MarketSmileResult,
    PutCallParityResult,
)
from backend.app.models.volatility import ImpliedVolatilityConfig

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/analytics/options", tags=["analytics"])


class OptionAnalyticsRequest(BaseModel):
    """A supplied normalized chain and explicit flat-market assumptions."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    chain: OptionChainSnapshot
    valuation_time: datetime | None = None
    risk_free_rate: float
    dividend_yield: float = 0.0
    quality_config: QuoteQualityConfig | None = None
    iv_solver_config: ImpliedVolatilityConfig = Field(default_factory=ImpliedVolatilityConfig)


class MarketSmileRequest(OptionAnalyticsRequest):
    """A single expiration and option type extracted from an enriched chain."""

    expiration: datetime
    option_type: OptionType


def _quality(request: OptionAnalyticsRequest, settings: Settings) -> QuoteQualityConfig:
    return request.quality_config or QuoteQualityConfig(
        max_quote_age_seconds=settings.max_quote_age_seconds
    )


@router.post("/chain", response_model=EnrichedOptionChain)
def enriched_chain(
    request: OptionAnalyticsRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> EnrichedOptionChain:
    """Validate, enrich, and deterministically sort a supplied normalized chain."""

    _enforce_chain_limit(len(request.chain.quotes), settings)
    logger.info(
        "chain_enrichment_start symbol=%s quotes=%d",
        request.chain.underlying_symbol,
        len(request.chain.quotes),
    )
    result = enrich_option_chain(
        request.chain,
        valuation_time=request.valuation_time,
        risk_free_rate=request.risk_free_rate,
        dividend_yield=request.dividend_yield,
        quality_config=_quality(request, settings),
        iv_config=request.iv_solver_config,
    )
    logger.info(
        "chain_enrichment_complete symbol=%s quotes=%d iv_success=%d invalid=%d",
        result.underlying_symbol,
        result.summary.total_contracts,
        result.summary.quotes_with_recovered_iv,
        result.summary.invalid_quotes,
    )
    return result


@router.post("/smile", response_model=MarketSmileResult)
def market_smile(
    request: MarketSmileRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> MarketSmileResult:
    """Return one unsmoothed call or put smile slice."""

    _enforce_chain_limit(len(request.chain.quotes), settings)
    chain = enrich_option_chain(
        request.chain,
        valuation_time=request.valuation_time,
        risk_free_rate=request.risk_free_rate,
        dividend_yield=request.dividend_yield,
        quality_config=_quality(request, settings),
        iv_config=request.iv_solver_config,
    )
    return extract_market_smile(
        chain,
        expiration=request.expiration,
        option_type=request.option_type,
    )


@router.post("/put-call-parity", response_model=PutCallParityResult)
def put_call_parity(
    request: OptionAnalyticsRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> PutCallParityResult:
    """Evaluate midpoint parity for fresh matched calls and puts."""

    _enforce_chain_limit(len(request.chain.quotes), settings)
    logger.info(
        "parity_analysis_start symbol=%s quotes=%d",
        request.chain.underlying_symbol,
        len(request.chain.quotes),
    )
    result = analyze_put_call_parity(
        request.chain,
        valuation_time=request.valuation_time,
        risk_free_rate=request.risk_free_rate,
        dividend_yield=request.dividend_yield,
        quality_config=_quality(request, settings),
    )
    logger.info(
        "parity_analysis_complete symbol=%s pairs=%d violations=%d",
        result.underlying_symbol,
        result.matched_pairs,
        result.violations,
    )
    return result
