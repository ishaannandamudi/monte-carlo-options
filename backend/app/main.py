"""FastAPI application factory and service-level error handling."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from backend.app.api.analytics import router as analytics_router
from backend.app.api.binomial import analytics_router as binomial_analytics_router
from backend.app.api.binomial import pricing_router as binomial_pricing_router
from backend.app.api.greeks import router as greeks_router
from backend.app.api.market_data import router as market_data_router
from backend.app.api.options_analytics import router as options_analytics_router
from backend.app.api.pricing import router as pricing_router
from backend.app.api.volatility import analytics_router as volatility_analytics_router
from backend.app.api.volatility import router as volatility_router
from backend.app.api.volatility_surface import router as volatility_surface_router
from backend.app.core.config import get_settings
from backend.app.core.exceptions import (
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderSymbolNotFoundError,
    ProviderTimeoutError,
    QuantitativeError,
)
from backend.app.core.logging import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("service_startup service=monte-carlo-options")
    yield


app = FastAPI(
    title="Monte Carlo Options",
    version="0.7.0",
    description="European option research with Monte Carlo, analytical, and CRR pricing",
    lifespan=lifespan,
)
app.include_router(pricing_router)
app.include_router(analytics_router)
app.include_router(greeks_router)
app.include_router(volatility_router)
app.include_router(volatility_analytics_router)
app.include_router(market_data_router)
app.include_router(options_analytics_router)
app.include_router(volatility_surface_router)
app.include_router(binomial_pricing_router)
app.include_router(binomial_analytics_router)


@app.exception_handler(ProviderError)
async def provider_error_handler(_: Request, exc: ProviderError) -> JSONResponse:
    """Translate normalized provider failures without exposing credentials or payloads."""

    if isinstance(exc, ProviderAuthenticationError):
        status_code = 401
    elif isinstance(exc, ProviderRateLimitError):
        status_code = 429
    elif isinstance(exc, ProviderSymbolNotFoundError):
        status_code = 404
    elif isinstance(exc, ProviderTimeoutError):
        status_code = 504
    else:
        status_code = 502
    logger.warning("provider_error type=%s error=%s", type(exc).__name__, exc)
    return JSONResponse(status_code=status_code, content={"detail": str(exc)})


@app.exception_handler(QuantitativeError)
async def quantitative_error_handler(_: Request, exc: QuantitativeError) -> JSONResponse:
    """Return clear non-success JSON for representational or statistical failures."""

    logger.warning("quantitative_error error=%s", exc)
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/health")
def health() -> dict[str, str]:
    """Return a dependency-free liveness response."""

    return {"status": "ok", "service": "monte-carlo-options"}
