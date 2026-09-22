"""Application provider construction from environment-backed settings."""

from functools import lru_cache

from backend.app.core.config import get_settings
from backend.app.market_data.base import MarketDataProvider
from backend.app.market_data.massive import MassiveMarketDataProvider


@lru_cache
def get_market_data_provider() -> MarketDataProvider:
    """Return the configured normalized provider without performing a request."""

    settings = get_settings()
    return MassiveMarketDataProvider(
        api_key=settings.market_data_api_key,
        base_url=settings.market_data_base_url,
        timeout_seconds=settings.market_data_timeout_seconds,
        max_contracts=settings.max_option_chain_contracts,
        basic_contract_limit=settings.basic_analysis_contract_limit,
        cache_ttl_seconds=settings.market_data_cache_ttl_seconds,
        reference_window_days=settings.market_data_reference_window_days,
    )
