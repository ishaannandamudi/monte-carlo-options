from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings, get_settings
from backend.app.core.exceptions import (
    ProviderAuthenticationError,
    ProviderEntitlementError,
    ProviderRateLimitError,
    ProviderSymbolNotFoundError,
)
from backend.app.main import app
from backend.app.market_data.fake import FakeMarketDataProvider
from backend.app.market_data.provider import get_market_data_provider
from backend.app.models.market import MarketEnvironment
from backend.app.models.market_data import (
    MarketDataSource,
    MarketExerciseStyle,
    OptionChainSnapshot,
    OptionInstrument,
    OptionQuote,
    UnderlyingQuote,
)
from backend.app.models.option import OptionContract, OptionType
from backend.app.pricing.black_scholes import black_scholes_price

VALUATION = datetime(2026, 1, 1, 21, tzinfo=UTC)
EXPIRATION = VALUATION + timedelta(days=365)


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def make_snapshot(count: int = 2) -> OptionChainSnapshot:
    underlying = UnderlyingQuote(
        symbol="XYZ",
        bid=99.9,
        ask=100.1,
        last=100.0,
        timestamp=VALUATION,
        source=MarketDataSource.FAKE,
    )
    quotes: list[OptionQuote] = []
    for index in range(count):
        option_type = OptionType.CALL if index % 2 == 0 else OptionType.PUT
        strike = 100.0 + 5.0 * (index // 2)
        contract = OptionContract(
            option_type=option_type,
            spot=100.0,
            strike=strike,
            maturity_years=1.0,
        )
        price = black_scholes_price(
            contract,
            MarketEnvironment(risk_free_rate=0.03, volatility=0.2, dividend_yield=0.01),
        )
        quotes.append(
            OptionQuote(
                instrument=OptionInstrument(
                    underlying_symbol="XYZ",
                    option_type=option_type,
                    strike=strike,
                    expiration=EXPIRATION,
                    provider_symbol=f"O:XYZ-{index}",
                    exercise_style=MarketExerciseStyle.AMERICAN,
                    multiplier=100,
                ),
                bid=price - 0.01,
                ask=price + 0.01,
                last=price,
                timestamp=VALUATION,
                source=MarketDataSource.FAKE,
            )
        )
    return OptionChainSnapshot(
        underlying_symbol="XYZ",
        underlying_quote=underlying,
        retrieved_at=VALUATION,
        source=MarketDataSource.FAKE,
        quotes=quotes,
        available_expirations=[EXPIRATION.date()],
    )


def install_fake(snapshot: OptionChainSnapshot) -> None:
    provider = FakeMarketDataProvider(
        underlying_quotes={"XYZ": snapshot.underlying_quote},
        option_chains={"XYZ": snapshot},
    )
    app.dependency_overrides[get_market_data_provider] = lambda: provider


def test_underlying_chain_and_expiration_endpoints(client: TestClient) -> None:
    snapshot = make_snapshot()
    install_fake(snapshot)
    underlying = client.get("/api/market-data/underlying/xyz")
    chain = client.get("/api/market-data/options/xyz")
    filtered = client.get(
        "/api/market-data/options/xyz",
        params={"expiration": EXPIRATION.date().isoformat()},
    )
    expirations = client.get("/api/market-data/options/xyz/expirations")
    assert underlying.status_code == 200
    assert underlying.json()["symbol"] == "XYZ"
    assert underlying.json()["midpoint"] == 100.0
    assert chain.status_code == 200
    assert len(chain.json()["quotes"]) == 2
    assert "raw_payload" not in chain.text
    assert filtered.status_code == 200
    assert len(filtered.json()["quotes"]) == 2
    assert expirations.status_code == 200
    assert expirations.json() == {
        "underlying_symbol": "XYZ",
        "expirations": [EXPIRATION.date().isoformat()],
    }


def test_enriched_chain_smile_and_parity_endpoints(client: TestClient) -> None:
    snapshot = make_snapshot()
    payload = {
        "chain": snapshot.model_dump(mode="json"),
        "valuation_time": VALUATION.isoformat(),
        "risk_free_rate": 0.03,
        "dividend_yield": 0.01,
    }
    chain = client.post("/api/analytics/options/chain", json=payload)
    smile = client.post(
        "/api/analytics/options/smile",
        json={
            **payload,
            "expiration": EXPIRATION.isoformat(),
            "option_type": "CALL",
        },
    )
    parity = client.post("/api/analytics/options/put-call-parity", json=payload)
    assert chain.status_code == 200
    assert chain.json()["summary"]["quotes_with_recovered_iv"] == 2
    assert chain.json()["quotes"][0]["european_bsm_approximation"] is True
    assert smile.status_code == 200
    assert len(smile.json()["points"]) == 1
    assert smile.json()["points"][0]["option_type"] == "CALL"
    assert parity.status_code == 200
    assert parity.json()["matched_pairs"] == 1
    assert parity.json()["violations"] == 0


def test_market_data_and_analytics_contract_limits(client: TestClient) -> None:
    snapshot = make_snapshot(3)
    install_fake(snapshot)
    app.dependency_overrides[get_settings] = lambda: Settings(max_option_chain_contracts=2)
    raw = client.get("/api/market-data/options/XYZ")
    enriched = client.post(
        "/api/analytics/options/chain",
        json={
            "chain": snapshot.model_dump(mode="json"),
            "risk_free_rate": 0.03,
            "dividend_yield": 0.01,
        },
    )
    assert raw.status_code == 422
    assert enriched.status_code == 422
    assert "API limit" in raw.json()["detail"]


class FailingProvider:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def get_underlying_quote(self, symbol: str) -> UnderlyingQuote:
        del symbol
        raise self.error

    def get_option_chain(
        self,
        symbol: str,
        expiration: date | None = None,
    ) -> OptionChainSnapshot:
        del symbol, expiration
        raise self.error

    def list_option_expirations(self, symbol: str) -> list[date]:
        del symbol
        raise self.error


@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (ProviderAuthenticationError("credentials rejected"), 401),
        (ProviderEntitlementError("endpoint unavailable for plan"), 403),
        (ProviderSymbolNotFoundError("symbol unavailable"), 404),
        (ProviderRateLimitError("provider throttled request"), 429),
    ],
)
def test_provider_failure_translation_without_credential_leak(
    client: TestClient,
    error: Exception,
    status_code: int,
) -> None:
    provider = FailingProvider(error)
    app.dependency_overrides[get_market_data_provider] = lambda: provider
    response = client.get("/api/market-data/underlying/XYZ")
    assert response.status_code == status_code
    assert response.json()["detail"] == str(error)
    assert "api_key" not in response.text.lower()
    assert "secret" not in response.text.lower()


def test_invalid_symbol_from_fake_provider_is_404(client: TestClient) -> None:
    install_fake(make_snapshot())
    response = client.get("/api/market-data/underlying/MISSING")
    assert response.status_code == 404


def test_naive_valuation_time_is_rejected_clearly(client: TestClient) -> None:
    snapshot = make_snapshot()
    response = client.post(
        "/api/analytics/options/chain",
        json={
            "chain": snapshot.model_dump(mode="json"),
            "valuation_time": "2026-01-01T21:00:00",
            "risk_free_rate": 0.03,
        },
    )
    assert response.status_code == 422
    assert "timezone-aware" in response.json()["detail"]
