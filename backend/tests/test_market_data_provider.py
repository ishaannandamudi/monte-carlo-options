import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

import pytest
from _pytest.monkeypatch import MonkeyPatch

import backend.app.market_data.massive as massive_module
from backend.app.core.exceptions import (
    ProviderAuthenticationError,
    ProviderMalformedResponseError,
    ProviderNetworkError,
    ProviderRateLimitError,
    ProviderServerError,
    ProviderSymbolNotFoundError,
)
from backend.app.market_data.fake import FakeMarketDataProvider
from backend.app.market_data.massive import MassiveMarketDataProvider
from backend.app.models.market_data import MarketExerciseStyle, OptionChainSnapshot
from backend.app.models.option import OptionType

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict[str, Any]:
    value = json.loads((FIXTURES / name).read_text())
    assert isinstance(value, dict)
    return value


def fixture_provider(monkeypatch: MonkeyPatch) -> MassiveMarketDataProvider:
    provider = MassiveMarketDataProvider(api_key="offline-fixture-key")
    underlying = load_fixture("massive_underlying.json")
    chain = load_fixture("massive_option_chain.json")
    contracts = load_fixture("massive_contracts.json")

    def request(path: str, params: dict[str, str] | None = None) -> dict[str, Any]:
        del params
        if "snapshot/options" in path:
            return chain
        if "reference/options/contracts" in path:
            return contracts
        if "snapshot/locale" in path:
            return underlying
        raise AssertionError(f"unexpected fixture path {path}")

    monkeypatch.setattr(provider, "_request_json", request)
    return provider


def test_massive_underlying_normalization(monkeypatch: MonkeyPatch) -> None:
    quote = fixture_provider(monkeypatch).get_underlying_quote(" xyz ")
    assert quote.symbol == "XYZ"
    assert quote.bid == 99.9
    assert quote.ask == 100.1
    assert quote.midpoint == 100.0
    assert quote.last == 100.0
    assert quote.previous_close == 99.5
    assert quote.volume == 1_250_000
    assert quote.timestamp == datetime(2026, 6, 1, 14, 30, tzinfo=UTC)


def test_massive_chain_normalizes_identity_quotes_and_metadata(
    monkeypatch: MonkeyPatch,
) -> None:
    snapshot = fixture_provider(monkeypatch).get_option_chain("xyz")
    assert snapshot.underlying_symbol == "XYZ"
    assert len(snapshot.quotes) == 4
    first = snapshot.quotes[0]
    assert first.instrument.option_type is OptionType.CALL
    assert first.instrument.strike == 100.0
    assert first.instrument.exercise_style is MarketExerciseStyle.AMERICAN
    assert first.instrument.multiplier == 100
    assert first.instrument.expiration == datetime(2026, 12, 18, 21, tzinfo=UTC)
    assert first.bid == 7.0
    assert first.ask == 7.2
    assert first.midpoint == 7.1
    assert first.bid_size == 11
    assert first.ask_size == 9
    assert first.volume == 850
    assert first.open_interest == 2400
    assert first.provider_implied_volatility == 0.25
    assert first.provider_greeks == {
        "delta": 0.55,
        "gamma": 0.03,
        "vega": 0.22,
        "theta": -0.04,
    }
    assert snapshot.quotes[2].bid is None
    assert snapshot.quotes[2].midpoint is None
    assert snapshot.quotes[3].bid == 2.0
    assert snapshot.quotes[3].ask == 1.0
    assert snapshot.quotes[3].midpoint is None
    assert snapshot.available_expirations == [date(2026, 12, 18), date(2027, 1, 15)]


def test_massive_expiration_normalization_is_sorted_unique(monkeypatch: MonkeyPatch) -> None:
    expirations = fixture_provider(monkeypatch).list_option_expirations("XYZ")
    assert expirations == [date(2026, 12, 18), date(2027, 1, 15)]


def test_massive_rejects_malformed_required_fields(monkeypatch: MonkeyPatch) -> None:
    provider = fixture_provider(monkeypatch)
    malformed = load_fixture("massive_option_chain.json")
    malformed["results"][0]["details"]["expiration_date"] = "not-a-date"
    underlying = load_fixture("massive_underlying.json")

    def request(path: str, params: dict[str, str] | None = None) -> dict[str, Any]:
        del params
        return underlying if "snapshot/locale" in path else malformed

    monkeypatch.setattr(provider, "_request_json", request)
    with pytest.raises(ProviderMalformedResponseError, match="expiration_date"):
        provider.get_option_chain("XYZ")


@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [
        (401, ProviderAuthenticationError),
        (404, ProviderSymbolNotFoundError),
        (429, ProviderRateLimitError),
        (503, ProviderServerError),
    ],
)
def test_massive_translates_http_failures(
    monkeypatch: MonkeyPatch,
    status_code: int,
    error_type: type[Exception],
) -> None:
    provider = MassiveMarketDataProvider(api_key="secret-never-logged")

    def fail(*_: object, **__: object) -> None:
        raise HTTPError("https://api.massive.com", status_code, "failure", None, None)

    monkeypatch.setattr(massive_module, "urlopen", fail)
    with pytest.raises(error_type):
        provider.get_underlying_quote("XYZ")


def test_massive_requires_credentials_and_translates_network_failure(
    monkeypatch: MonkeyPatch,
) -> None:
    with pytest.raises(ProviderAuthenticationError, match="MARKET_DATA_API_KEY"):
        MassiveMarketDataProvider(api_key=None).get_underlying_quote("XYZ")

    provider = MassiveMarketDataProvider(api_key="fixture-key")

    def fail(*_: object, **__: object) -> None:
        raise URLError("offline")

    monkeypatch.setattr(massive_module, "urlopen", fail)
    with pytest.raises(ProviderNetworkError):
        provider.get_underlying_quote("XYZ")


def test_fake_provider_supports_quote_chain_filter_and_expirations(
    monkeypatch: MonkeyPatch,
) -> None:
    snapshot = fixture_provider(monkeypatch).get_option_chain("XYZ")
    fake = FakeMarketDataProvider(
        underlying_quotes={"XYZ": snapshot.underlying_quote},
        option_chains={"XYZ": snapshot},
    )
    assert fake.get_underlying_quote("xyz") == snapshot.underlying_quote
    filtered = fake.get_option_chain("xyz", date(2027, 1, 15))
    assert len(filtered.quotes) == 1
    assert fake.list_option_expirations("xyz") == snapshot.available_expirations
    with pytest.raises(ProviderSymbolNotFoundError):
        fake.get_option_chain("MISSING")


def test_normalized_chain_does_not_retain_raw_provider_payload(
    monkeypatch: MonkeyPatch,
) -> None:
    snapshot = fixture_provider(monkeypatch).get_option_chain("XYZ")
    assert isinstance(snapshot, OptionChainSnapshot)
    serialized = snapshot.model_dump(mode="json")
    assert "request_id" not in serialized
    assert "break_even_price" not in str(serialized)
