"""Thin API adapter for ticker-only American-options analysis."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends

from backend.app.analytics.options_analyzer import analyze_american_options
from backend.app.core.config import Settings, get_settings
from backend.app.market_data.base import MarketDataProvider
from backend.app.market_data.provider import get_market_data_provider
from backend.app.models.options_analyzer import OptionsAnalysisRequest, OptionsAnalysisResult

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/options", tags=["options-analyzer"])


@router.post("/analyze", response_model=OptionsAnalysisResult)
def analyze_options(
    request: OptionsAnalysisRequest,
    provider: Annotated[MarketDataProvider, Depends(get_market_data_provider)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> OptionsAnalysisResult:
    """Fetch and analyze one bounded real-market American option chain."""

    logger.info(
        "american_options_analysis symbol=%s expiration=%s max_contracts=%d",
        request.symbol,
        request.expiration.isoformat() if request.expiration is not None else "AUTO",
        request.max_contracts,
    )
    result = analyze_american_options(request, provider=provider, settings=settings)
    logger.info(
        "american_options_analysis_complete symbol=%s expiration=%s contracts=%d runtime_ms=%.3f",
        result.symbol,
        result.selected_expiration.isoformat(),
        result.summary.contracts_analyzed,
        result.timings.total_ms,
    )
    return result
