"""Deterministic in-memory market-data provider for tests and local development."""

from datetime import date

from backend.app.core.exceptions import ProviderSymbolNotFoundError
from backend.app.models.market_data import OptionChainSnapshot, UnderlyingQuote


class FakeMarketDataProvider:
    """Serve caller-supplied normalized models without any network transport."""

    def __init__(
        self,
        *,
        underlying_quotes: dict[str, UnderlyingQuote],
        option_chains: dict[str, OptionChainSnapshot],
    ) -> None:
        self._underlying_quotes = {key.upper(): value for key, value in underlying_quotes.items()}
        self._option_chains = {key.upper(): value for key, value in option_chains.items()}

    def get_underlying_quote(self, symbol: str) -> UnderlyingQuote:
        normalized = symbol.strip().upper()
        try:
            return self._underlying_quotes[normalized]
        except KeyError as exc:
            raise ProviderSymbolNotFoundError(f"symbol {normalized!r} is unavailable") from exc

    def get_option_chain(
        self,
        symbol: str,
        expiration: date | None = None,
    ) -> OptionChainSnapshot:
        normalized = symbol.strip().upper()
        try:
            snapshot = self._option_chains[normalized]
        except KeyError as exc:
            raise ProviderSymbolNotFoundError(
                f"option chain for {normalized!r} is unavailable"
            ) from exc
        if expiration is None:
            return snapshot
        quotes = [
            quote for quote in snapshot.quotes if quote.instrument.expiration.date() == expiration
        ]
        return snapshot.model_copy(update={"quotes": quotes})

    def list_option_expirations(self, symbol: str) -> list[date]:
        snapshot = self.get_option_chain(symbol)
        if snapshot.available_expirations:
            return sorted(set(snapshot.available_expirations))
        return sorted({quote.instrument.expiration.date() for quote in snapshot.quotes})
