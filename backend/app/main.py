"""FastAPI application factory and service-level error handling."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app import __version__
from backend.app.api.analytics import router as analytics_router
from backend.app.api.asian import analytics_router as asian_analytics_router
from backend.app.api.asian import pricing_router as asian_pricing_router
from backend.app.api.barrier import analytics_router as barrier_analytics_router
from backend.app.api.barrier import pricing_router as barrier_pricing_router
from backend.app.api.basket import analytics_router as basket_analytics_router
from backend.app.api.basket import multi_asset_router
from backend.app.api.basket import pricing_router as basket_pricing_router
from backend.app.api.binomial import analytics_router as binomial_analytics_router
from backend.app.api.binomial import pricing_router as binomial_pricing_router
from backend.app.api.digital import analytics_router as digital_analytics_router
from backend.app.api.digital import pricing_router as digital_pricing_router
from backend.app.api.experiments import get_experiment_store
from backend.app.api.experiments import router as experiments_router
from backend.app.api.greeks import router as greeks_router
from backend.app.api.heston import analytics_router as heston_analytics_router
from backend.app.api.heston import pricing_router as heston_pricing_router
from backend.app.api.heston_calibration import (
    analytics_router as heston_calibration_analytics_router,
)
from backend.app.api.heston_calibration import calibration_router as heston_calibration_router
from backend.app.api.longstaff_schwartz import analytics_router as lsm_analytics_router
from backend.app.api.longstaff_schwartz import pricing_router as lsm_pricing_router
from backend.app.api.market_data import router as market_data_router
from backend.app.api.merton import analytics_router as merton_analytics_router
from backend.app.api.merton import pricing_router as merton_pricing_router
from backend.app.api.model_comparison import router as model_comparison_router
from backend.app.api.options_analytics import router as options_analytics_router
from backend.app.api.options_analyzer import router as options_analyzer_router
from backend.app.api.portfolio import router as portfolio_router
from backend.app.api.pricing import router as pricing_router
from backend.app.api.qmc import analytics_router as qmc_analytics_router
from backend.app.api.qmc import pricing_router as qmc_pricing_router
from backend.app.api.regime import router as regime_router
from backend.app.api.risk import router as risk_router
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
    get_experiment_store()
    logger.info("service_startup service=monte-carlo-options")
    yield


app = FastAPI(
    title="Monte Carlo Options",
    version=__version__,
    description=(
        "European, American, path-dependent, basket, Heston, and Merton option research with "
        "analytical pricing, Monte Carlo, calibration, model comparison, portfolio risk, and "
        "latent-regime, quasi-Monte Carlo, and reproducible experiment analytics"
    ),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(get_settings().cors_allowed_origins),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(pricing_router)
app.include_router(analytics_router)
app.include_router(greeks_router)
app.include_router(volatility_router)
app.include_router(volatility_analytics_router)
app.include_router(market_data_router)
app.include_router(options_analytics_router)
app.include_router(options_analyzer_router)
app.include_router(volatility_surface_router)
app.include_router(binomial_pricing_router)
app.include_router(binomial_analytics_router)
app.include_router(lsm_pricing_router)
app.include_router(lsm_analytics_router)
app.include_router(asian_pricing_router)
app.include_router(asian_analytics_router)
app.include_router(digital_pricing_router)
app.include_router(digital_analytics_router)
app.include_router(barrier_pricing_router)
app.include_router(barrier_analytics_router)
app.include_router(basket_pricing_router)
app.include_router(basket_analytics_router)
app.include_router(multi_asset_router)
app.include_router(heston_pricing_router)
app.include_router(heston_analytics_router)
app.include_router(merton_pricing_router)
app.include_router(merton_analytics_router)
app.include_router(heston_calibration_router)
app.include_router(heston_calibration_analytics_router)
app.include_router(model_comparison_router)
app.include_router(portfolio_router)
app.include_router(risk_router)
app.include_router(regime_router)
app.include_router(qmc_pricing_router)
app.include_router(qmc_analytics_router)
app.include_router(experiments_router)


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


@app.get("/version")
def version() -> dict[str, str]:
    """Return stable service identity and package version metadata."""

    return {"service": "monte-carlo-options", "version": __version__}
