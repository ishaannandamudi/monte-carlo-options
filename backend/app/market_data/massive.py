"""Massive (formerly Polygon.io) REST adapter and response normalization."""

import json
from datetime import UTC, date, datetime, time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from backend.app.core.exceptions import (
    MarketDataValidationError,
    ProviderAuthenticationError,
    ProviderMalformedResponseError,
    ProviderNetworkError,
    ProviderRateLimitError,
    ProviderServerError,
    ProviderSymbolNotFoundError,
    ProviderTimeoutError,
)
from backend.app.models.market_data import (
    MarketDataSource,
    MarketExerciseStyle,
    OptionChainSnapshot,
    OptionInstrument,
    OptionQuote,
    QuoteTimestampSource,
    UnderlyingQuote,
)
from backend.app.models.option import OptionType

_NEW_YORK = ZoneInfo("America/New_York")


def _normalized_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    if not normalized:
        raise ProviderSymbolNotFoundError("symbol must not be empty")
    return normalized


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProviderMalformedResponseError(f"Massive {context} must be an object")
    return value


def _optional_mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProviderMalformedResponseError(f"Massive {context} must be numeric")
    return float(value)


def _optional_number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    numeric = _optional_number(value)
    return None if numeric is None else int(numeric)


def _timestamp_from_nanoseconds(value: object, context: str) -> datetime:
    nanoseconds = _number(value, context)
    try:
        return datetime.fromtimestamp(nanoseconds / 1_000_000_000.0, tz=UTC)
    except (OverflowError, OSError, ValueError) as exc:
        raise ProviderMalformedResponseError(f"Massive {context} is not a valid timestamp") from exc


def _expiration_at_market_close(value: object) -> datetime:
    if not isinstance(value, str):
        raise ProviderMalformedResponseError("Massive expiration_date must be a YYYY-MM-DD string")
    try:
        expiration_date = date.fromisoformat(value)
    except ValueError as exc:
        raise ProviderMalformedResponseError("Massive expiration_date is invalid") from exc
    local_close = datetime.combine(expiration_date, time(16, 0), tzinfo=_NEW_YORK)
    return local_close.astimezone(UTC)


def _option_type(value: object) -> OptionType:
    if value == "call":
        return OptionType.CALL
    if value == "put":
        return OptionType.PUT
    raise ProviderMalformedResponseError(f"unsupported Massive contract_type {value!r}")


def _exercise_style(value: object) -> MarketExerciseStyle:
    mapping = {
        "american": MarketExerciseStyle.AMERICAN,
        "european": MarketExerciseStyle.EUROPEAN,
        "bermudan": MarketExerciseStyle.BERMUDAN,
    }
    if not isinstance(value, str):
        return MarketExerciseStyle.UNKNOWN
    return mapping.get(value, MarketExerciseStyle.UNKNOWN)


class MassiveMarketDataProvider:
    """Normalize Massive REST snapshots into provider-independent models."""

    source = MarketDataSource.MASSIVE

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str = "https://api.massive.com",
        timeout_seconds: float = 10.0,
        max_contracts: int | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/") + "/"
        self._timeout_seconds = timeout_seconds
        self._max_contracts = max_contracts

    def _request_json(
        self,
        path_or_url: str,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if not self._api_key:
            raise ProviderAuthenticationError("MARKET_DATA_API_KEY is required for Massive")
        url = (
            path_or_url
            if path_or_url.startswith(("http://", "https://"))
            else urljoin(self._base_url, path_or_url.lstrip("/"))
        )
        parsed = urlparse(url)
        query_values: dict[str, str] = {}
        if parsed.query:
            query_values.update(dict(parse_qsl(parsed.query)))
        if params:
            query_values.update(params)
        query_values["apiKey"] = self._api_key
        url = urlunparse(parsed._replace(query=urlencode(query_values)))
        request = Request(url, headers={"Accept": "application/json"})
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:  # noqa: S310
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code in {401, 403}:
                raise ProviderAuthenticationError("Massive rejected provider credentials") from exc
            if exc.code == 404:
                raise ProviderSymbolNotFoundError(
                    "Massive could not find the requested symbol"
                ) from exc
            if exc.code == 429:
                raise ProviderRateLimitError("Massive rate limit exceeded") from exc
            if exc.code >= 500:
                raise ProviderServerError(f"Massive server returned HTTP {exc.code}") from exc
            raise ProviderNetworkError(f"Massive request returned HTTP {exc.code}") from exc
        except TimeoutError as exc:
            raise ProviderTimeoutError("Massive request timed out") from exc
        except URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise ProviderTimeoutError("Massive request timed out") from exc
            raise ProviderNetworkError("Massive network request failed") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProviderMalformedResponseError("Massive returned malformed JSON") from exc
        return _mapping(payload, "response")

    def get_underlying_quote(self, symbol: str) -> UnderlyingQuote:
        normalized = _normalized_symbol(symbol)
        payload = self._request_json(f"/v2/snapshot/locale/us/markets/stocks/tickers/{normalized}")
        ticker = _mapping(payload.get("ticker"), "ticker snapshot")
        quote = _optional_mapping(ticker.get("lastQuote") or ticker.get("last_quote"))
        trade = _optional_mapping(ticker.get("lastTrade") or ticker.get("last_trade"))
        previous = _optional_mapping(ticker.get("prevDay") or ticker.get("prev_day"))
        day = _optional_mapping(ticker.get("day"))
        timestamp_value = quote.get("t", quote.get("last_updated", ticker.get("updated")))
        if timestamp_value is None:
            raise ProviderMalformedResponseError("Massive underlying snapshot has no timestamp")
        timestamp_source = (
            QuoteTimestampSource.QUOTE
            if quote.get("t", quote.get("last_updated")) is not None
            else QuoteTimestampSource.SNAPSHOT
        )
        try:
            return UnderlyingQuote(
                symbol=str(ticker.get("ticker", normalized)),
                bid=_optional_number(quote.get("p", quote.get("bid"))),
                ask=_optional_number(quote.get("P", quote.get("ask"))),
                last=_optional_number(trade.get("p", trade.get("price"))),
                timestamp=_timestamp_from_nanoseconds(
                    timestamp_value, "underlying quote timestamp"
                ),
                timestamp_source=timestamp_source,
                source=self.source,
                previous_close=_optional_number(previous.get("c", previous.get("close"))),
                volume=_optional_number(day.get("v", day.get("volume"))),
            )
        except ValidationError as exc:
            raise ProviderMalformedResponseError(
                "Massive underlying snapshot failed normalized validation"
            ) from exc

    def _normalize_option_quote(
        self,
        raw: object,
        underlying_symbol: str,
        retrieved_at: datetime,
    ) -> OptionQuote:
        item = _mapping(raw, "option snapshot item")
        details = _mapping(item.get("details"), "option details")
        last_quote = _optional_mapping(item.get("last_quote"))
        last_trade = _optional_mapping(item.get("last_trade"))
        day = _optional_mapping(item.get("day"))
        trade_timestamp_value = last_trade.get("sip_timestamp")
        trade_timestamp = (
            None
            if trade_timestamp_value is None
            else _timestamp_from_nanoseconds(trade_timestamp_value, "option trade timestamp")
        )
        quote_timestamp_value = last_quote.get("last_updated")
        if quote_timestamp_value is not None:
            quote_timestamp = _timestamp_from_nanoseconds(
                quote_timestamp_value, "option quote timestamp"
            )
            timestamp_source = QuoteTimestampSource.QUOTE
        elif trade_timestamp is not None:
            quote_timestamp = trade_timestamp
            timestamp_source = QuoteTimestampSource.TRADE
        else:
            quote_timestamp = retrieved_at
            timestamp_source = QuoteTimestampSource.SNAPSHOT
        provider_symbol = details.get("ticker")
        if not isinstance(provider_symbol, str):
            raise ProviderMalformedResponseError("Massive option ticker is missing")
        multiplier_value = details.get("shares_per_contract")
        multiplier = _optional_int(multiplier_value)
        raw_greeks = item.get("greeks")
        provider_greeks = None
        if isinstance(raw_greeks, dict):
            provider_greeks = {
                str(key): float(value)
                for key, value in raw_greeks.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
        try:
            instrument = OptionInstrument(
                underlying_symbol=underlying_symbol,
                option_type=_option_type(details.get("contract_type")),
                strike=_number(details.get("strike_price"), "strike_price"),
                expiration=_expiration_at_market_close(details.get("expiration_date")),
                provider_symbol=provider_symbol,
                exercise_style=_exercise_style(details.get("exercise_style")),
                multiplier=multiplier,
            )
            return OptionQuote(
                instrument=instrument,
                bid=_optional_number(last_quote.get("bid")),
                ask=_optional_number(last_quote.get("ask")),
                last=_optional_number(last_trade.get("price")),
                bid_size=_optional_int(last_quote.get("bid_size")),
                ask_size=_optional_int(last_quote.get("ask_size")),
                volume=_optional_number(day.get("volume")),
                open_interest=_optional_int(item.get("open_interest")),
                timestamp=quote_timestamp,
                timestamp_source=timestamp_source,
                last_trade_timestamp=trade_timestamp,
                source=self.source,
                provider_implied_volatility=_optional_number(item.get("implied_volatility")),
                provider_greeks=provider_greeks,
            )
        except ValidationError as exc:
            raise ProviderMalformedResponseError(
                "Massive option snapshot failed normalized validation"
            ) from exc

    def get_option_chain(
        self,
        symbol: str,
        expiration: date | None = None,
    ) -> OptionChainSnapshot:
        normalized = _normalized_symbol(symbol)
        retrieved_at = datetime.now(UTC)
        params = {"limit": "250", "sort": "ticker", "order": "asc"}
        if expiration is not None:
            params["expiration_date"] = expiration.isoformat()
        url = f"/v3/snapshot/options/{normalized}"
        quotes: list[OptionQuote] = []
        seen_urls: set[str] = set()
        while url:
            if url in seen_urls:
                raise ProviderMalformedResponseError("Massive pagination repeated next_url")
            seen_urls.add(url)
            payload = self._request_json(url, params)
            params = {}
            raw_results = payload.get("results", [])
            if not isinstance(raw_results, list):
                raise ProviderMalformedResponseError("Massive option results must be a list")
            quotes.extend(
                self._normalize_option_quote(item, normalized, retrieved_at) for item in raw_results
            )
            if self._max_contracts is not None and len(quotes) > self._max_contracts:
                raise MarketDataValidationError(
                    "option chain exceeds configured provider limit of "
                    f"{self._max_contracts} contracts"
                )
            next_url = payload.get("next_url")
            if next_url is not None and not isinstance(next_url, str):
                raise ProviderMalformedResponseError("Massive next_url must be a string")
            url = next_url or ""
        underlying = self.get_underlying_quote(normalized)
        expirations = sorted({quote.instrument.expiration.date() for quote in quotes})
        return OptionChainSnapshot(
            underlying_symbol=normalized,
            underlying_quote=underlying,
            retrieved_at=retrieved_at,
            source=self.source,
            quotes=quotes,
            available_expirations=expirations,
        )

    def list_option_expirations(self, symbol: str) -> list[date]:
        normalized = _normalized_symbol(symbol)
        params = {
            "underlying_ticker": normalized,
            "expired": "false",
            "limit": "1000",
            "sort": "expiration_date",
            "order": "asc",
        }
        url = "/v3/reference/options/contracts"
        expirations: set[date] = set()
        seen_urls: set[str] = set()
        while url:
            if url in seen_urls:
                raise ProviderMalformedResponseError("Massive pagination repeated next_url")
            seen_urls.add(url)
            payload = self._request_json(url, params)
            params = {}
            raw_results = payload.get("results", [])
            if not isinstance(raw_results, list):
                raise ProviderMalformedResponseError("Massive contract results must be a list")
            for raw in raw_results:
                contract = _mapping(raw, "contract reference")
                expiration_value = contract.get("expiration_date")
                if not isinstance(expiration_value, str):
                    raise ProviderMalformedResponseError("Massive contract expiration is missing")
                try:
                    expirations.add(date.fromisoformat(expiration_value))
                except ValueError as exc:
                    raise ProviderMalformedResponseError(
                        "Massive contract expiration is invalid"
                    ) from exc
            next_url = payload.get("next_url")
            if next_url is not None and not isinstance(next_url, str):
                raise ProviderMalformedResponseError("Massive next_url must be a string")
            url = next_url or ""
        return sorted(expirations)
