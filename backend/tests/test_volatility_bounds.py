import math

import pytest

from backend.app.core.exceptions import ArbitrageBoundsError, ImpliedVolatilityBracketError
from backend.app.models.option import OptionContract, OptionType
from backend.app.models.volatility import (
    ImpliedVolatilityMarket,
    ImpliedVolatilityObservation,
)
from backend.app.pricing.implied_volatility import (
    call_price_bounds,
    forward_price,
    put_price_bounds,
    solve_implied_volatility,
)


def test_call_and_put_bounds_with_dividends_and_negative_rate() -> None:
    spot, strike, maturity, rate, dividend = 105.0, 100.0, 2.0, -0.01, 0.03
    discounted_spot = spot * math.exp(-dividend * maturity)
    discounted_strike = strike * math.exp(-rate * maturity)

    call = call_price_bounds(spot, strike, maturity, rate, dividend)
    put = put_price_bounds(spot, strike, maturity, rate, dividend)

    assert call.lower == pytest.approx(max(discounted_spot - discounted_strike, 0.0))
    assert call.upper == pytest.approx(discounted_spot)
    assert put.lower == pytest.approx(max(discounted_strike - discounted_spot, 0.0))
    assert put.upper == pytest.approx(discounted_strike)


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_expiry_bounds_equal_intrinsic(option_type: OptionType) -> None:
    bounds_function = call_price_bounds if option_type is OptionType.CALL else put_price_bounds
    bounds = bounds_function(110.0, 100.0, 0.0, -0.5, 0.7)
    expected = 10.0 if option_type is OptionType.CALL else 0.0
    assert bounds.lower == expected
    assert bounds.upper == (110.0 if option_type is OptionType.CALL else 100.0)


def test_forward_price_uses_continuous_carry() -> None:
    assert forward_price(100.0, 1.5, 0.04, 0.01) == pytest.approx(100.0 * math.exp(0.03 * 1.5))


@pytest.mark.parametrize("observed_price", [4.9, 100.1])
def test_prices_outside_call_bounds_are_rejected(observed_price: float) -> None:
    contract = OptionContract(
        option_type=OptionType.CALL,
        spot=100.0,
        strike=95.0,
        maturity_years=0.0,
    )
    observation = ImpliedVolatilityObservation(
        contract=contract,
        market=ImpliedVolatilityMarket(risk_free_rate=0.0),
        observed_price=observed_price,
    )
    with pytest.raises(ArbitrageBoundsError, match="no-arbitrage bounds"):
        solve_implied_volatility(observation)


def test_exact_upper_bound_has_no_finite_implied_volatility() -> None:
    contract = OptionContract(
        option_type=OptionType.CALL,
        spot=100.0,
        strike=100.0,
        maturity_years=1.0,
    )
    observation = ImpliedVolatilityObservation(
        contract=contract,
        market=ImpliedVolatilityMarket(risk_free_rate=0.0),
        observed_price=100.0,
    )
    with pytest.raises(ImpliedVolatilityBracketError, match="tends to infinity"):
        solve_implied_volatility(observation)
