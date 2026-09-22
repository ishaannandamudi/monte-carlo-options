"""Offline integration tests for the ticker-only American-options analyzer."""

import math
from collections.abc import Iterator
from datetime import UTC, date, datetime, time, timedelta

import pytest
from fastapi.testclient import TestClient

from backend.app.analytics.options_analyzer import (
    analyze_american_options,
    select_default_expiration,
)
from backend.app.core.config import Settings, get_settings
from backend.app.core.exceptions import (
    MarketDataValidationError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
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
from backend.app.models.options_analyzer import (
    AnalysisDataQuality,
    AnalysisOptionType,
    FairValueVolatilitySource,
    OptionsAnalysisRequest,
)
from backend.app.pricing.black_scholes import black_scholes_price


def _settings(**updates: object) -> Settings:
    return Settings(
        default_analysis_tree_steps=60,
        max_quote_age_seconds=900.0,
        **updates,
    )


def _snapshot(
    *,
    strikes: tuple[float, ...] = (90.0, 95.0, 100.0, 105.0, 110.0),
    days: int = 45,
    now: datetime | None = None,
    volatility_shift: dict[tuple[OptionType, float], float] | None = None,
    quote_age_seconds: float = 0.0,
    dividend_yield: float = 0.0,
) -> OptionChainSnapshot:
    valuation = (now or datetime.now(UTC)).replace(microsecond=0)
    expiration_date = valuation.date() + timedelta(days=days)
    expiration = datetime.combine(expiration_date, time(21), tzinfo=UTC)
    maturity = (expiration - valuation).total_seconds() / (365.0 * 86_400.0)
    underlying = UnderlyingQuote(
        symbol="XYZ",
        bid=99.9,
        ask=100.1,
        last=100.0,
        timestamp=valuation,
        source=MarketDataSource.FAKE,
    )
    quotes: list[OptionQuote] = []
    for side in (OptionType.CALL, OptionType.PUT):
        for strike in strikes:
            volatility = 0.20 + 0.0015 * abs(strike - 100.0)
            if volatility_shift is not None:
                volatility = volatility_shift.get((side, strike), volatility)
            price = black_scholes_price(
                OptionContract(
                    option_type=side,
                    spot=100.0,
                    strike=strike,
                    maturity_years=maturity,
                ),
                MarketEnvironment(
                    risk_free_rate=0.04,
                    volatility=volatility,
                    dividend_yield=dividend_yield,
                ),
            )
            half_spread = min(0.01, price * 0.2)
            quotes.append(
                OptionQuote(
                    instrument=OptionInstrument(
                        underlying_symbol="XYZ",
                        option_type=side,
                        strike=strike,
                        expiration=expiration,
                        provider_symbol=f"O:XYZ-{side.value}-{strike:g}",
                        exercise_style=MarketExerciseStyle.AMERICAN,
                        multiplier=100,
                    ),
                    bid=max(price - half_spread, 0.0),
                    ask=price + half_spread,
                    last=price,
                    bid_size=10,
                    ask_size=12,
                    volume=100.0 + strike,
                    open_interest=1_000 + int(strike),
                    timestamp=valuation - timedelta(seconds=quote_age_seconds),
                    source=MarketDataSource.FAKE,
                    provider_implied_volatility=volatility + 0.01,
                )
            )
    return OptionChainSnapshot(
        underlying_symbol="XYZ",
        underlying_quote=underlying,
        retrieved_at=valuation,
        source=MarketDataSource.FAKE,
        quotes=quotes,
        available_expirations=[expiration_date],
    )


def _provider(snapshot: OptionChainSnapshot) -> FakeMarketDataProvider:
    return FakeMarketDataProvider(
        underlying_quotes={"XYZ": snapshot.underlying_quote},
        option_chains={"XYZ": snapshot},
    )


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_request_normalizes_ticker_and_rejects_invalid_symbols() -> None:
    assert OptionsAnalysisRequest(symbol="  xyz ").symbol == "XYZ"
    for symbol in ("", "$AAPL", "TOO-LONG-SYMBOL-123"):
        with pytest.raises(ValueError):
            OptionsAnalysisRequest(symbol=symbol)


def test_default_expiration_is_future_nearest_45_days_with_earlier_tie_break() -> None:
    today = date(2026, 1, 1)
    expirations = [
        today,
        today + timedelta(days=60),
        today + timedelta(days=30),
        today + timedelta(days=44),
        today + timedelta(days=46),
    ]
    assert select_default_expiration(expirations, today, 45) == today + timedelta(days=44)


def test_analysis_filters_around_atm_balances_sides_and_is_deterministic() -> None:
    snapshot = _snapshot(strikes=(80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0))
    request = OptionsAnalysisRequest(symbol="xyz", max_contracts=6, strike_window_pct=0.11)
    first = analyze_american_options(request, provider=_provider(snapshot), settings=_settings())
    second = analyze_american_options(request, provider=_provider(snapshot), settings=_settings())

    assert first.symbol == "XYZ"
    assert first.summary.calls == first.summary.puts == 3
    assert first.summary.atm_strike == 100.0
    assert all(89.0 <= contract.strike <= 111.0 for contract in first.contracts)
    assert [contract.provider_symbol for contract in first.contracts] == [
        contract.provider_symbol for contract in second.contracts
    ]
    first_payload = first.model_dump(mode="json", exclude={"timings"})
    second_payload = second.model_dump(mode="json", exclude={"timings"})
    for contract in first_payload["contracts"]:
        for key in ("crr_pricing_runtime_ms", "greeks_runtime_ms", "model_runtime_ms"):
            contract.pop(key)
    for contract in second_payload["contracts"]:
        for key in ("crr_pricing_runtime_ms", "greeks_runtime_ms", "model_runtime_ms"):
            contract.pop(key)
    assert first_payload == second_payload


def test_midpoint_iv_independent_smile_fair_value_greeks_and_gap_math() -> None:
    snapshot = _snapshot()
    result = analyze_american_options(
        OptionsAnalysisRequest(symbol="XYZ", max_contracts=10),
        provider=_provider(snapshot),
        settings=_settings(),
    )
    target = next(
        contract
        for contract in result.contracts
        if contract.option_type is OptionType.CALL and contract.strike == 100.0
    )

    assert target.market_price_source is not None
    assert target.market_price_source.value == "MIDPOINT"
    assert target.implied_volatility == pytest.approx(0.20, abs=2e-4)
    assert target.fair_value_volatility_source is FairValueVolatilitySource.SMILE_INTERPOLATED
    assert target.fair_value_volatility_independent is True
    assert target.fair_value is not None
    assert target.delta is not None and target.gamma is not None and target.theta is not None
    assert target.vega is not None and target.rho is not None
    assert target.fair_value_gap == pytest.approx(target.fair_value - target.market_midpoint)  # type: ignore[operator]
    assert target.fair_value_gap_pct == pytest.approx(
        target.fair_value_gap / target.market_midpoint  # type: ignore[operator]
    )
    assert result.summary.contracts_with_independent_fair_vol > 0


def test_fair_value_volatility_does_not_reuse_target_own_iv() -> None:
    shifts = {(OptionType.CALL, 100.0): 0.65}
    result = analyze_american_options(
        OptionsAnalysisRequest(symbol="XYZ", option_type=AnalysisOptionType.CALL),
        provider=_provider(_snapshot(volatility_shift=shifts)),
        settings=_settings(),
    )
    target = next(contract for contract in result.contracts if contract.strike == 100.0)

    assert target.implied_volatility == pytest.approx(0.65, abs=2e-4)
    assert target.fair_value_volatility_source is FairValueVolatilitySource.SMILE_INTERPOLATED
    assert target.fair_value_volatility_independent is True
    assert target.fair_value_volatility != pytest.approx(target.implied_volatility, abs=0.1)


def test_fair_volatility_fallback_hierarchy_and_unavailable_state() -> None:
    atm_reference = analyze_american_options(
        OptionsAnalysisRequest(symbol="XYZ", option_type=AnalysisOptionType.CALL),
        provider=_provider(_snapshot(strikes=(95.0, 100.0))),
        settings=_settings(),
    )
    assert all(
        contract.fair_value_volatility_source is FairValueVolatilitySource.ATM_REFERENCE
        for contract in atm_reference.contracts
    )

    own_only = analyze_american_options(
        OptionsAnalysisRequest(symbol="XYZ", option_type=AnalysisOptionType.CALL),
        provider=_provider(_snapshot(strikes=(100.0,))),
        settings=_settings(),
    ).contracts[0]
    assert own_only.fair_value_volatility_source is FairValueVolatilitySource.CONTRACT_IV_FALLBACK
    assert "FAIR_VALUE_VOLATILITY_NOT_INDEPENDENT" in own_only.warning_flags

    snapshot = _snapshot(strikes=(100.0,))
    quote = snapshot.quotes[0].model_copy(update={"bid": None, "ask": None, "last": None})
    broken = snapshot.model_copy(update={"quotes": [quote]})
    unavailable = analyze_american_options(
        OptionsAnalysisRequest(symbol="XYZ", option_type=AnalysisOptionType.CALL),
        provider=_provider(broken),
        settings=_settings(),
    ).contracts[0]
    assert unavailable.fair_value_volatility_source is FairValueVolatilitySource.UNAVAILABLE
    assert unavailable.fair_value is None
    assert unavailable.analysis_quality is AnalysisDataQuality.LOW_DATA_QUALITY


def test_stale_wide_and_illiquid_quotes_are_visible_not_silent() -> None:
    snapshot = _snapshot(strikes=(95.0, 100.0, 105.0), quote_age_seconds=1_000.0)
    quote = snapshot.quotes[0].model_copy(
        update={"bid": 0.01, "ask": 2.0, "volume": 0.0, "open_interest": 0}
    )
    snapshot = snapshot.model_copy(update={"quotes": [quote, *snapshot.quotes[1:]]})
    result = analyze_american_options(
        OptionsAnalysisRequest(symbol="XYZ"),
        provider=_provider(snapshot),
        settings=_settings(),
    )
    flagged = next(
        contract
        for contract in result.contracts
        if contract.provider_symbol == quote.instrument.provider_symbol
    )
    assert {"STALE", "WIDE_SPREAD", "ILLIQUID"}.issubset(flagged.quality_flags)
    assert flagged.implied_volatility is None
    assert any("stale" in warning.lower() for warning in result.warnings)


def test_last_price_fallback_is_explicit() -> None:
    snapshot = _snapshot(strikes=(100.0,))
    call = snapshot.quotes[0]
    one_sided = call.model_copy(update={"bid": None, "ask": None, "last": call.midpoint})
    snapshot = snapshot.model_copy(update={"quotes": [one_sided]})
    contract = analyze_american_options(
        OptionsAnalysisRequest(
            symbol="XYZ",
            option_type=AnalysisOptionType.CALL,
            allow_last_price_fallback=True,
        ),
        provider=_provider(snapshot),
        settings=_settings(),
    ).contracts[0]
    assert contract.market_price_source is not None
    assert contract.market_price_source.value == "LAST"
    assert contract.market_midpoint is None
    assert contract.fair_value_gap is None


def test_american_workflow_has_meaningful_early_exercise_premium() -> None:
    snapshot = _snapshot(days=180, dividend_yield=0.20)
    result = analyze_american_options(
        OptionsAnalysisRequest(
            symbol="XYZ",
            option_type=AnalysisOptionType.CALL,
            dividend_yield=0.20,
            tree_steps=200,
        ),
        provider=_provider(snapshot),
        settings=_settings(),
    )
    contract = next(item for item in result.contracts if item.strike == 90.0)
    assert contract.early_exercise_premium is not None
    assert contract.early_exercise_premium > 1e-3
    assert contract.fair_value is not None and contract.european_reference_value is not None
    assert contract.fair_value > contract.european_reference_value


def test_no_expiration_empty_window_and_resource_limits_fail_clearly() -> None:
    snapshot = _snapshot()
    no_expirations = snapshot.model_copy(update={"available_expirations": []})
    provider = FakeMarketDataProvider(
        underlying_quotes={"XYZ": snapshot.underlying_quote},
        option_chains={"XYZ": no_expirations.model_copy(update={"quotes": []})},
    )
    with pytest.raises(MarketDataValidationError, match="no future option expiration"):
        analyze_american_options(
            OptionsAnalysisRequest(symbol="XYZ"), provider=provider, settings=_settings()
        )
    with pytest.raises(MarketDataValidationError, match="strike window"):
        analyze_american_options(
            OptionsAnalysisRequest(symbol="XYZ", strike_window_pct=0.001),
            provider=_provider(_snapshot(strikes=(90.0,))),
            settings=_settings(),
        )
    with pytest.raises(MarketDataValidationError, match="max_contracts"):
        analyze_american_options(
            OptionsAnalysisRequest(symbol="XYZ", max_contracts=3),
            provider=_provider(snapshot),
            settings=_settings(max_analysis_contracts=2),
        )
    with pytest.raises(MarketDataValidationError, match="option chain"):
        analyze_american_options(
            OptionsAnalysisRequest(symbol="XYZ"),
            provider=_provider(snapshot),
            settings=_settings(max_option_chain_contracts=2),
        )


class _FailingProvider:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def get_underlying_quote(self, symbol: str) -> UnderlyingQuote:
        del symbol
        raise self.error

    def list_option_expirations(self, symbol: str) -> list[date]:
        del symbol
        raise self.error

    def get_option_chain(self, symbol: str, expiration: date | None = None) -> OptionChainSnapshot:
        del symbol, expiration
        raise self.error


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (ProviderAuthenticationError("credentials rejected"), 401),
        (ProviderRateLimitError("rate limited"), 429),
    ],
)
def test_endpoint_translates_provider_errors(
    client: TestClient, error: Exception, status: int
) -> None:
    app.dependency_overrides[get_market_data_provider] = lambda: _FailingProvider(error)
    response = client.post("/api/options/analyze", json={"symbol": "XYZ"})
    assert response.status_code == status
    assert response.json()["detail"] == str(error)


def test_endpoint_serializes_complete_analysis_and_openapi_contract(client: TestClient) -> None:
    snapshot = _snapshot()
    app.dependency_overrides[get_market_data_provider] = lambda: _provider(snapshot)
    app.dependency_overrides[get_settings] = lambda: _settings()
    response = client.post(
        "/api/options/analyze",
        json={"symbol": " xyz ", "max_contracts": 4, "option_type": "BOTH"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "XYZ"
    assert payload["summary"]["contracts_analyzed"] == 4
    assert payload["assumptions"]["pricing_model"] == "CRR_AMERICAN"
    assert payload["contracts"][0]["implied_volatility_method"] == "BSM_EQUIVALENT"
    assert "/api/options/analyze" in app.openapi()["paths"]
    assert math.isfinite(payload["timings"]["total_ms"])


def test_analysis_settings_load_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_ANALYSIS_CONTRACTS", "75")
    monkeypatch.setenv("DEFAULT_ANALYSIS_TREE_STEPS", "350")
    monkeypatch.setenv("DEFAULT_ANALYSIS_RISK_FREE_RATE", "0.035")
    settings = Settings.from_environment()
    assert settings.max_analysis_contracts == 75
    assert settings.default_analysis_tree_steps == 350
    assert settings.default_analysis_risk_free_rate == pytest.approx(0.035)
