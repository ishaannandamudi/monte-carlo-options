"""FastAPI application factory and service-level error handling."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from backend.app.api.analytics import router as analytics_router
from backend.app.api.greeks import router as greeks_router
from backend.app.api.pricing import router as pricing_router
from backend.app.core.config import get_settings
from backend.app.core.exceptions import QuantitativeError
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
    version="0.3.0",
    description="European option pricing, variance reduction, Greeks, and sensitivity API",
    lifespan=lifespan,
)
app.include_router(pricing_router)
app.include_router(analytics_router)
app.include_router(greeks_router)


@app.exception_handler(QuantitativeError)
async def quantitative_error_handler(_: Request, exc: QuantitativeError) -> JSONResponse:
    """Return clear non-success JSON for representational or statistical failures."""

    logger.warning("quantitative_error error=%s", exc)
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/health")
def health() -> dict[str, str]:
    """Return a dependency-free liveness response."""

    return {"status": "ok", "service": "monte-carlo-options"}
