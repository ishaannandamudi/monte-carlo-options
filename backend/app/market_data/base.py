"""Provider-independent market-data interface."""

from datetime import date
from typing import Protocol

from backend.app.models.market_data import OptionChainSnapshot, UnderlyingQuote


class MarketDataProvider(Protocol):
    """Normalized synchronous provider contract used by application services."""

    def get_underlying_quote(self, symbol: str) -> UnderlyingQuote:
        """Return one normalized underlying quote."""

    def get_option_chain(
        self,
        symbol: str,
        expiration: date | None = None,
    ) -> OptionChainSnapshot:
        """Return a normalized option-chain snapshot."""

    def list_option_expirations(self, symbol: str) -> list[date]:
        """Return sorted unique available expiration dates."""
