"""Massive REST adapter with paid-snapshot and Basic end-of-day modes."""

import json
import math
from datetime import UTC, date, datetime, time, timedelta
from time import monotonic
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from backend.app.core.exceptions import (
    MarketDataValidationError,
    ProviderAuthenticationError,
    ProviderEntitlementError,
    ProviderMalformedResponseError,
    ProviderNetworkError,
    ProviderRateLimitError,
    ProviderServerError,
    ProviderSymbolNotFoundError,
    ProviderTimeoutError,
)
from backend.app.models.market_data import (
    MarketDataMode,
    MarketDataSource,
    MarketExerciseStyle,
    MarketPriceSource,
    OptionChainSnapshot,
    OptionInstrument,
    OptionQuote,
    ProviderCapabilities,
    QuoteTimestampSource,
    UnderlyingPriceSource,
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
    result = float(value)
    if not math.isfinite(result):
        raise ProviderMalformedResponseError(f"Massive {context} must be finite")
    return result


def _optional_number(value: object) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _optional_int(value: object) -> int | None:
    numeric = _optional_number(value)
    return None if numeric is None else int(numeric)


def _timestamp_from_nanoseconds(value: object, context: str) -> datetime:
    nanoseconds = _number(value, context)
    try:
        return datetime.fromtimestamp(nanoseconds / 1_000_000_000.0, tz=UTC)
    except (OverflowError, OSError, ValueError) as exc:
        raise ProviderMalformedResponseError(f"Massive {context} is not a valid timestamp") from exc


def _aggregate_date(value: object, context: str) -> date:
    milliseconds = _number(value, context)
    try:
        instant = datetime.fromtimestamp(milliseconds / 1_000.0, tz=UTC)
    except (OverflowError, OSError, ValueError) as exc:
        raise ProviderMalformedResponseError(f"Massive {context} is not a valid timestamp") from exc
    return instant.astimezone(_NEW_YORK).date()


def _market_close(value: date) -> datetime:
    return datetime.combine(value, time(16, 0), tzinfo=_NEW_YORK).astimezone(UTC)


def _expiration_at_market_close(value: object) -> datetime:
    if not isinstance(value, str):
        raise ProviderMalformedResponseError("Massive expiration_date must be a YYYY-MM-DD string")
    try:
        expiration_date = date.fromisoformat(value)
    except ValueError as exc:
        raise ProviderMalformedResponseError("Massive expiration_date is invalid") from exc
    return _market_close(expiration_date)


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
    """Normalize Massive data and fall back only on snapshot entitlement failures."""

    source = MarketDataSource.MASSIVE

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str = "https://api.massive.com",
        timeout_seconds: float = 10.0,
        max_contracts: int | None = None,
        basic_contract_limit: int = 2,
        cache_ttl_seconds: float = 60.0,
        reference_window_days: int = 14,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/") + "/"
        self._timeout_seconds = timeout_seconds
        self._max_contracts = max_contracts
        self._basic_contract_limit = basic_contract_limit
        self._cache_ttl_seconds = cache_ttl_seconds
        self._reference_window_days = reference_window_days
        self._snapshot_capability: tuple[float, bool] | None = None
        self._underlying_cache: dict[tuple[str, MarketDataMode], tuple[float, UnderlyingQuote]] = {}
        self._option_bar_cache: dict[str, tuple[float, OptionQuote]] = {}
        self._reference_cache: dict[str, tuple[float, list[OptionInstrument]]] = {}

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
        query_values = dict(parse_qsl(parsed.query))
        if params:
            query_values.update(params)
        query_values["apiKey"] = self._api_key
        request = Request(
            urlunparse(parsed._replace(query=urlencode(query_values))),
            headers={"Accept": "application/json"},
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:  # noqa: S310
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code == 401:
                raise ProviderAuthenticationError("Massive rejected provider credentials.") from exc
            if exc.code == 403:
                raise ProviderEntitlementError(
                    "Massive authenticated successfully, but the current plan does not include "
                    "this endpoint."
                ) from exc
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

    def _snapshot_available(self) -> bool | None:
        if self._snapshot_capability is None:
            return None
        expires_at, available = self._snapshot_capability
        if monotonic() >= expires_at:
            self._snapshot_capability = None
            return None
        return available

    def _remember_snapshot_capability(self, available: bool) -> None:
        self._snapshot_capability = (monotonic() + self._cache_ttl_seconds, available)

    def _cached_underlying(self, symbol: str, mode: MarketDataMode) -> UnderlyingQuote | None:
        cached = self._underlying_cache.get((symbol, mode))
        if cached is None or monotonic() >= cached[0]:
            self._underlying_cache.pop((symbol, mode), None)
            return None
        return cached[1]

    def _store_underlying(self, quote: UnderlyingQuote) -> UnderlyingQuote:
        self._underlying_cache[(quote.symbol, quote.market_data_mode)] = (
            monotonic() + self._cache_ttl_seconds,
            quote,
        )
        return quote

    def _normalize_underlying_snapshot(
        self, payload: dict[str, Any], normalized: str
    ) -> UnderlyingQuote:
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

    def _previous_bar(self, provider_symbol: str, context: str) -> dict[str, Any]:
        payload = self._request_json(
            f"/v2/aggs/ticker/{provider_symbol}/prev",
            {"adjusted": "true"},
        )
        results = payload.get("results", [])
        if not isinstance(results, list):
            raise ProviderMalformedResponseError(f"Massive {context} results must be a list")
        if not results:
            raise ProviderSymbolNotFoundError(
                f"Massive returned no previous-day aggregate for {context}"
            )
        return _mapping(results[0], f"{context} aggregate")

    def _get_underlying_eod(self, normalized: str) -> UnderlyingQuote:
        cached = self._cached_underlying(normalized, MarketDataMode.END_OF_DAY)
        if cached is not None:
            return cached
        bar = self._previous_bar(normalized, f"underlying {normalized}")
        data_date = _aggregate_date(bar.get("t"), "underlying aggregate timestamp")
        close = _number(bar.get("c"), "underlying aggregate close")
        try:
            quote = UnderlyingQuote(
                symbol=normalized,
                timestamp=_market_close(data_date),
                timestamp_source=QuoteTimestampSource.END_OF_DAY,
                source=self.source,
                volume=_optional_number(bar.get("v")),
                market_data_mode=MarketDataMode.END_OF_DAY,
                price_source=UnderlyingPriceSource.END_OF_DAY_CLOSE,
                reference_price=close,
                open=_optional_number(bar.get("o")),
                high=_optional_number(bar.get("h")),
                low=_optional_number(bar.get("l")),
                close=close,
                vwap=_optional_number(bar.get("vw")),
                data_date=data_date,
            )
        except ValidationError as exc:
            raise ProviderMalformedResponseError(
                "Massive underlying aggregate failed normalized validation"
            ) from exc
        return self._store_underlying(quote)

    def get_underlying_quote(self, symbol: str) -> UnderlyingQuote:
        normalized = _normalized_symbol(symbol)
        if self._snapshot_available() is False:
            return self._get_underlying_eod(normalized)
        cached = self._cached_underlying(normalized, MarketDataMode.SNAPSHOT)
        if cached is not None:
            return cached
        try:
            payload = self._request_json(
                f"/v2/snapshot/locale/us/markets/stocks/tickers/{normalized}"
            )
        except ProviderEntitlementError:
            self._remember_snapshot_capability(False)
            return self._get_underlying_eod(normalized)
        quote = self._normalize_underlying_snapshot(payload, normalized)
        self._remember_snapshot_capability(True)
        return self._store_underlying(quote)

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
                multiplier=_optional_int(details.get("shares_per_contract")),
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

    def _normalize_reference_contract(
        self, raw: object, underlying_symbol: str
    ) -> OptionInstrument:
        item = _mapping(raw, "contract reference")
        provider_symbol = item.get("ticker")
        if not isinstance(provider_symbol, str):
            raise ProviderMalformedResponseError("Massive option ticker is missing")
        try:
            return OptionInstrument(
                underlying_symbol=underlying_symbol,
                option_type=_option_type(item.get("contract_type")),
                strike=_number(item.get("strike_price"), "strike_price"),
                expiration=_expiration_at_market_close(item.get("expiration_date")),
                provider_symbol=provider_symbol,
                exercise_style=_exercise_style(item.get("exercise_style")),
                multiplier=_optional_int(item.get("shares_per_contract")),
            )
        except ValidationError as exc:
            raise ProviderMalformedResponseError(
                "Massive contract reference failed normalized validation"
            ) from exc

    def _reference_contracts(
        self,
        normalized: str,
        *,
        target_date: date | None,
        strike_window_pct: float | None,
    ) -> list[OptionInstrument]:
        basic_mode = self._snapshot_available() is False
        target_key = target_date.isoformat() if target_date is not None else "ALL"
        cache_key = f"{normalized}:{basic_mode}:{target_key}:{strike_window_pct}"
        cached = self._reference_cache.get(cache_key)
        if cached is not None and monotonic() < cached[0]:
            return cached[1]
        params = {
            "underlying_ticker": normalized,
            "expired": "false",
            "limit": "1000",
            "sort": "expiration_date",
            "order": "asc",
        }
        if basic_mode:
            center = target_date or (datetime.now(_NEW_YORK).date() + timedelta(days=45))
            params["expiration_date.gte"] = (
                center - timedelta(days=self._reference_window_days)
            ).isoformat()
            params["expiration_date.lte"] = (
                center + timedelta(days=self._reference_window_days)
            ).isoformat()
            underlying = self._cached_underlying(normalized, MarketDataMode.END_OF_DAY)
            if underlying is not None and underlying.reference_price is not None:
                window = max(strike_window_pct or 0.15, 0.15)
                params["strike_price.gte"] = f"{underlying.reference_price * (1.0 - window):.8g}"
                params["strike_price.lte"] = f"{underlying.reference_price * (1.0 + window):.8g}"
        url = "/v3/reference/options/contracts"
        contracts: list[OptionInstrument] = []
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
            contracts.extend(
                self._normalize_reference_contract(raw, normalized) for raw in raw_results
            )
            next_url = payload.get("next_url")
            if next_url is not None and not isinstance(next_url, str):
                raise ProviderMalformedResponseError("Massive next_url must be a string")
            url = "" if basic_mode else (next_url or "")
        if not contracts:
            raise ProviderSymbolNotFoundError(
                f"Massive returned no active option contracts for {normalized}"
            )
        self._reference_cache[cache_key] = (
            monotonic() + self._cache_ttl_seconds,
            contracts,
        )
        return contracts

    def list_option_expirations(
        self,
        symbol: str,
        target_date: date | None = None,
        strike_window_pct: float | None = None,
    ) -> list[date]:
        normalized = _normalized_symbol(symbol)
        contracts = self._reference_contracts(
            normalized,
            target_date=target_date,
            strike_window_pct=strike_window_pct,
        )
        return sorted({contract.expiration.date() for contract in contracts})

    def _cached_reference_candidates(
        self, normalized: str, expiration: date
    ) -> list[OptionInstrument]:
        now = monotonic()
        candidates: dict[str, OptionInstrument] = {}
        for cache_key, (expires_at, contracts) in self._reference_cache.items():
            if expires_at <= now or not cache_key.startswith(f"{normalized}:"):
                continue
            for contract in contracts:
                if contract.expiration.date() == expiration:
                    candidates[contract.provider_symbol] = contract
        return list(candidates.values())

    def _select_basic_contracts(
        self,
        candidates: list[OptionInstrument],
        *,
        spot: float,
        option_type: OptionType | None,
        max_contracts: int | None,
    ) -> list[OptionInstrument]:
        limit = min(max_contracts or self._basic_contract_limit, self._basic_contract_limit)

        def rank(item: OptionInstrument) -> tuple[float, float, str]:
            return (
                abs(math.log(item.strike / spot)),
                item.strike,
                item.provider_symbol,
            )

        if option_type is not None:
            return sorted(
                (item for item in candidates if item.option_type is option_type), key=rank
            )[:limit]
        calls = sorted(
            (item for item in candidates if item.option_type is OptionType.CALL), key=rank
        )
        puts = sorted((item for item in candidates if item.option_type is OptionType.PUT), key=rank)
        selected: list[OptionInstrument] = []
        while len(selected) < limit and (calls or puts):
            for side in (calls, puts):
                if side and len(selected) < limit:
                    selected.append(side.pop(0))
        return selected

    def _get_option_eod(self, instrument: OptionInstrument) -> OptionQuote:
        cached = self._option_bar_cache.get(instrument.provider_symbol)
        if cached is not None and monotonic() < cached[0]:
            return cached[1]
        bar = self._previous_bar(instrument.provider_symbol, instrument.provider_symbol)
        data_date = _aggregate_date(bar.get("t"), "option aggregate timestamp")
        close = _number(bar.get("c"), "option aggregate close")
        try:
            quote = OptionQuote(
                instrument=instrument,
                timestamp=_market_close(data_date),
                timestamp_source=QuoteTimestampSource.END_OF_DAY,
                source=self.source,
                volume=_optional_number(bar.get("v")),
                market_data_mode=MarketDataMode.END_OF_DAY,
                market_reference_price=close,
                market_price_source=MarketPriceSource.END_OF_DAY_CLOSE,
                eod_open=_optional_number(bar.get("o")),
                eod_high=_optional_number(bar.get("h")),
                eod_low=_optional_number(bar.get("l")),
                eod_close=close,
                eod_vwap=_optional_number(bar.get("vw")),
                data_date=data_date,
            )
        except ValidationError as exc:
            raise ProviderMalformedResponseError(
                "Massive option aggregate failed normalized validation"
            ) from exc
        self._option_bar_cache[instrument.provider_symbol] = (
            monotonic() + self._cache_ttl_seconds,
            quote,
        )
        return quote

    def _get_eod_option_chain(
        self,
        normalized: str,
        expiration: date | None,
        option_type: OptionType | None,
        max_contracts: int | None,
    ) -> OptionChainSnapshot:
        underlying = self._get_underlying_eod(normalized)
        if expiration is None:
            raise MarketDataValidationError(
                "an expiration is required for Basic end-of-day option analysis"
            )
        candidates = self._cached_reference_candidates(normalized, expiration)
        if not candidates:
            self._reference_contracts(
                normalized,
                target_date=expiration,
                strike_window_pct=0.15,
            )
            candidates = self._cached_reference_candidates(normalized, expiration)
        selected = self._select_basic_contracts(
            candidates,
            spot=float(underlying.reference_price or 0.0),
            option_type=option_type,
            max_contracts=max_contracts,
        )
        if not selected:
            raise MarketDataValidationError(
                "no Basic-compatible option contracts matched the requested side"
            )
        quotes: list[OptionQuote] = []
        for instrument in selected:
            try:
                quotes.append(self._get_option_eod(instrument))
            except ProviderSymbolNotFoundError:
                continue
        if not quotes:
            raise MarketDataValidationError(
                "Massive Basic returned no previous-day bars for the bounded option set"
            )
        return OptionChainSnapshot(
            underlying_symbol=normalized,
            underlying_quote=underlying,
            retrieved_at=underlying.timestamp,
            source=self.source,
            quotes=quotes,
            available_expirations=sorted({contract.expiration.date() for contract in candidates}),
            market_data_mode=MarketDataMode.END_OF_DAY,
            capabilities=ProviderCapabilities(
                snapshot_available=False,
                end_of_day_available=True,
                basic_contract_limit=self._basic_contract_limit,
            ),
        )

    def get_option_chain(
        self,
        symbol: str,
        expiration: date | None = None,
        option_type: OptionType | None = None,
        max_contracts: int | None = None,
    ) -> OptionChainSnapshot:
        normalized = _normalized_symbol(symbol)
        if self._snapshot_available() is False:
            return self._get_eod_option_chain(normalized, expiration, option_type, max_contracts)
        retrieved_at = datetime.now(UTC)
        params = {"limit": "250", "sort": "ticker", "order": "asc"}
        if expiration is not None:
            params["expiration_date"] = expiration.isoformat()
        url = f"/v3/snapshot/options/{normalized}"
        quotes: list[OptionQuote] = []
        seen_urls: set[str] = set()
        try:
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
                    self._normalize_option_quote(item, normalized, retrieved_at)
                    for item in raw_results
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
        except ProviderEntitlementError:
            self._remember_snapshot_capability(False)
            return self._get_eod_option_chain(normalized, expiration, option_type, max_contracts)
        self._remember_snapshot_capability(True)
        underlying = self.get_underlying_quote(normalized)
        return OptionChainSnapshot(
            underlying_symbol=normalized,
            underlying_quote=underlying,
            retrieved_at=retrieved_at,
            source=self.source,
            quotes=quotes,
            available_expirations=sorted({quote.instrument.expiration.date() for quote in quotes}),
            market_data_mode=MarketDataMode.SNAPSHOT,
            capabilities=ProviderCapabilities(
                snapshot_available=True,
                end_of_day_available=True,
            ),
        )
