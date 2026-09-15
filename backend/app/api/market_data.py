"""Thin normalized market-data provider endpoints."""

import logging
from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.app.core.config import Settings, get_settings
from backend.app.market_data.base import MarketDataProvider
from backend.app.market_data.provider import get_market_data_provider
from backend.app.models.market_data import OptionChainSnapshot, UnderlyingQuote
from backend.app.models.results import OptionExpirationsResult

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/market-data", tags=["market-data"])


def _enforce_chain_limit(count: int, settings: Settings) -> None:
    if count > settings.max_option_chain_contracts:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"option chain exceeds API limit of {settings.max_option_chain_contracts} contracts"
            ),
        )


@router.get("/underlying/{symbol}", response_model=UnderlyingQuote)
def underlying_quote(
    symbol: str,
    provider: Annotated[MarketDataProvider, Depends(get_market_data_provider)],
) -> UnderlyingQuote:
    """Return a provider-independent underlying quote."""

    logger.info("provider_request type=underlying symbol=%s", symbol.upper())
    return provider.get_underlying_quote(symbol)


@router.get("/options/{symbol}/expirations", response_model=OptionExpirationsResult)
def option_expirations(
    symbol: str,
    provider: Annotated[MarketDataProvider, Depends(get_market_data_provider)],
) -> OptionExpirationsResult:
    """Return sorted normalized option expiration dates."""

    logger.info("provider_request type=expirations symbol=%s", symbol.upper())
    expirations = provider.list_option_expirations(symbol)
    return OptionExpirationsResult(
        underlying_symbol=symbol.strip().upper(),
        expirations=expirations,
    )


@router.get("/options/{symbol}", response_model=OptionChainSnapshot)
def option_chain(
    symbol: str,
    provider: Annotated[MarketDataProvider, Depends(get_market_data_provider)],
    settings: Annotated[Settings, Depends(get_settings)],
    expiration: Annotated[date | None, Query()] = None,
) -> OptionChainSnapshot:
    """Return a bounded normalized chain without provider payload fields."""

    logger.info(
        "provider_request type=option_chain symbol=%s expiration=%s",
        symbol.upper(),
        expiration.isoformat() if expiration is not None else "ALL",
    )
    snapshot = provider.get_option_chain(symbol, expiration)
    _enforce_chain_limit(len(snapshot.quotes), settings)
    logger.info(
        "provider_normalized symbol=%s quotes=%d",
        snapshot.underlying_symbol,
        len(snapshot.quotes),
    )
    return snapshot
