from datetime import UTC, datetime, timedelta

import pytest

from backend.app.analytics.market_data import enrich_option_chain
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
from backend.app.models.results import EnrichedOptionChain
from backend.app.models.volatility_surface import SmileConstructionConfig, SmileSelectionPolicy
from backend.app.pricing.black_scholes import black_scholes_price
from backend.app.volatility_surface.core import (
    build_market_surface,
    build_surface_from_option_chain,
    construct_market_smiles,
    query_surface,
)

VALUATION = datetime(2026, 1, 1, 21, tzinfo=UTC)


def quote(
    option_type: OptionType,
    strike: float,
    maturity: float,
    volatility: float,
    *,
    spread: float = 0.02,
    symbol_suffix: str = "",
    timestamp: datetime = VALUATION,
) -> OptionQuote:
    expiration = VALUATION + timedelta(days=365 * maturity)
    price = black_scholes_price(
        OptionContract(
            option_type=option_type,
            spot=100.0,
            strike=strike,
            maturity_years=maturity,
        ),
        MarketEnvironment(risk_free_rate=0.0, volatility=volatility, dividend_yield=0.0),
    )
    return OptionQuote(
        instrument=OptionInstrument(
            underlying_symbol="XYZ",
            option_type=option_type,
            strike=strike,
            expiration=expiration,
            provider_symbol=f"O:XYZ-{maturity}-{strike}-{option_type.value}-{symbol_suffix}",
            exercise_style=MarketExerciseStyle.EUROPEAN,
        ),
        bid=max(price - spread / 2.0, 0.0),
        ask=price + spread / 2.0,
        last=price,
        timestamp=timestamp,
        source=MarketDataSource.FAKE,
    )


def market_snapshot(*, include_second_expiration: bool = True) -> OptionChainSnapshot:
    maturities = [0.5, 1.0] if include_second_expiration else [0.5]
    quotes = [
        quote(option_type, strike, maturity, 0.2 + 0.05 * ((strike / 100.0 - 1.0) ** 2))
        for maturity in maturities
        for strike in [90.0, 100.0, 110.0]
        for option_type in [OptionType.CALL, OptionType.PUT]
    ]
    return OptionChainSnapshot(
        underlying_symbol="XYZ",
        underlying_quote=UnderlyingQuote(
            symbol="XYZ",
            bid=99.9,
            ask=100.1,
            last=100.0,
            timestamp=VALUATION,
            source=MarketDataSource.FAKE,
        ),
        retrieved_at=VALUATION,
        source=MarketDataSource.FAKE,
        quotes=quotes,
    )


def enriched_chain(*, include_second_expiration: bool = True) -> EnrichedOptionChain:
    return enrich_option_chain(
        market_snapshot(include_second_expiration=include_second_expiration),
        risk_free_rate=0.0,
        dividend_yield=0.0,
    )


def test_market_chain_builds_otm_smiles_and_queryable_surface() -> None:
    chain = enriched_chain()
    surface, diagnostics = build_market_surface(chain)
    assert diagnostics.total_quotes_considered == 12
    assert diagnostics.iv_points_accepted == 6
    assert diagnostics.smiles_created == 2
    for smile in surface.smiles:
        assert [point.option_type for point in smile.points] == [
            OptionType.PUT,
            OptionType.CALL,
            OptionType.CALL,
        ]
    result = query_surface(surface, maturity_years=0.75, log_moneyness=0.0)
    assert result.implied_volatility == pytest.approx(0.2, abs=2e-4)


def test_calls_only_and_puts_only_selection_policies() -> None:
    chain = enriched_chain(include_second_expiration=False)
    calls, _ = construct_market_smiles(
        chain,
        SmileConstructionConfig(selection_policy=SmileSelectionPolicy.CALLS_ONLY),
    )
    puts, _ = construct_market_smiles(
        chain,
        SmileConstructionConfig(selection_policy=SmileSelectionPolicy.PUTS_ONLY),
    )
    assert all(point.option_type is OptionType.CALL for point in calls[0].points)
    assert all(point.option_type is OptionType.PUT for point in puts[0].points)


def test_atm_policy_prefers_tighter_spread_then_call() -> None:
    chain = enriched_chain(include_second_expiration=False)
    chain_data = chain.model_dump()
    quotes = chain_data["quotes"]
    for item in quotes:
        if item["quote"]["instrument"]["strike"] == 100.0:
            if item["quote"]["instrument"]["option_type"] == OptionType.CALL:
                item["relative_spread"] = 0.01
            else:
                item["relative_spread"] = 0.02
    rebuilt = type(chain).model_validate(chain_data)
    smiles, _ = construct_market_smiles(rebuilt)
    atm = next(point for point in smiles[0].points if point.strike == 100.0)
    assert atm.option_type is OptionType.CALL


def test_duplicate_strike_resolution_is_deterministic() -> None:
    chain = enriched_chain(include_second_expiration=False)
    duplicate = chain.quotes[3].model_copy(
        update={
            "relative_spread": 0.0001,
            "quote": chain.quotes[3].quote.model_copy(
                update={
                    "instrument": chain.quotes[3].quote.instrument.model_copy(
                        update={"provider_symbol": "A-TIGHTER"}
                    )
                }
            ),
        }
    )
    rebuilt = chain.model_copy(update={"quotes": [*chain.quotes, duplicate]})
    smiles, diagnostics = construct_market_smiles(rebuilt)
    selected = next(
        point for point in smiles[0].points if point.strike == duplicate.quote.instrument.strike
    )
    assert selected.quality.provider_symbol == "A-TIGHTER"
    # One same-side duplicate plus the deterministic call/put ATM choice.
    assert diagnostics.exclusion_reasons["DUPLICATE_STRIKE"] == 2


def test_insufficient_expiration_is_reported_instead_of_fabricated() -> None:
    chain = enriched_chain(include_second_expiration=False)
    smiles, diagnostics = construct_market_smiles(
        chain,
        SmileConstructionConfig(minimum_points=4),
    )
    assert smiles == []
    assert diagnostics.expirations_rejected == 1
    assert len(diagnostics.rejected_expirations) == 1


def test_raw_normalized_chain_is_enriched_and_built_end_to_end() -> None:
    surface, diagnostics = build_surface_from_option_chain(
        market_snapshot(),
        risk_free_rate=0.0,
        dividend_yield=0.0,
    )
    assert len(surface.smiles) == 2
    assert diagnostics.iv_points_accepted == 6
    assert query_surface(surface, maturity_years=0.75, log_moneyness=0.0).implied_volatility > 0.0
