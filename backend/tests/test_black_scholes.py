import math

import pytest

from backend.app.core.exceptions import NumericalError
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.pricing.black_scholes import BlackScholesPricer, black_scholes_price


def contract(
    option_type: OptionType,
    *,
    spot: float = 100.0,
    strike: float = 100.0,
    maturity: float = 1.0,
) -> OptionContract:
    return OptionContract(
        option_type=option_type,
        spot=spot,
        strike=strike,
        maturity_years=maturity,
    )


def test_known_call_benchmark() -> None:
    value = black_scholes_price(
        contract(OptionType.CALL),
        MarketEnvironment(risk_free_rate=0.05, volatility=0.2),
    )
    assert value == pytest.approx(10.450583572185565, abs=1e-12)


def test_known_put_benchmark() -> None:
    value = black_scholes_price(
        contract(OptionType.PUT),
        MarketEnvironment(risk_free_rate=0.05, volatility=0.2),
    )
    assert value == pytest.approx(5.573526022256971, abs=1e-12)


def test_put_call_parity_with_dividend_and_negative_rate() -> None:
    market = MarketEnvironment(risk_free_rate=-0.01, volatility=0.3, dividend_yield=0.025)
    call = black_scholes_price(contract(OptionType.CALL, maturity=2.0), market)
    put = black_scholes_price(contract(OptionType.PUT, maturity=2.0), market)
    parity = 100.0 * math.exp(-0.025 * 2.0) - 100.0 * math.exp(0.01 * 2.0)
    assert call - put == pytest.approx(parity, abs=1e-12)


@pytest.mark.parametrize(
    ("option_type", "spot", "strike", "expected"),
    [
        (OptionType.CALL, 120.0, 100.0, 20.0),
        (OptionType.PUT, 80.0, 100.0, 20.0),
        (OptionType.CALL, 100.0, 100.0, 0.0),
        (OptionType.PUT, 100.0, 100.0, 0.0),
    ],
)
def test_zero_maturity_is_intrinsic(
    option_type: OptionType, spot: float, strike: float, expected: float
) -> None:
    value = black_scholes_price(
        contract(option_type, spot=spot, strike=strike, maturity=0.0),
        MarketEnvironment(risk_free_rate=0.03, volatility=0.2),
    )
    assert value == expected


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_zero_volatility_uses_discounted_deterministic_terminal_value(
    option_type: OptionType,
) -> None:
    market = MarketEnvironment(risk_free_rate=0.04, volatility=0.0, dividend_yield=0.01)
    option = contract(option_type, maturity=1.7)
    terminal = option.spot * math.exp((0.04 - 0.01) * 1.7)
    payoff = (
        max(terminal - option.strike, 0.0)
        if option_type is OptionType.CALL
        else max(option.strike - terminal, 0.0)
    )
    expected = math.exp(-0.04 * 1.7) * payoff
    assert black_scholes_price(option, market) == pytest.approx(expected, abs=1e-13)


def test_positive_dividend_yield_benchmark() -> None:
    market = MarketEnvironment(risk_free_rate=0.05, volatility=0.2, dividend_yield=0.02)
    assert black_scholes_price(contract(OptionType.CALL), market) == pytest.approx(
        9.227005508154036, abs=1e-12
    )
    assert black_scholes_price(contract(OptionType.PUT), market) == pytest.approx(
        6.330080627549918, abs=1e-12
    )


@pytest.mark.parametrize(
    ("option_type", "spot", "strike"),
    [
        (OptionType.CALL, 1_000.0, 1.0),
        (OptionType.CALL, 1.0, 1_000.0),
        (OptionType.PUT, 1_000.0, 1.0),
        (OptionType.PUT, 1.0, 1_000.0),
    ],
)
def test_deep_itm_and_otm_values_are_finite_non_negative(
    option_type: OptionType, spot: float, strike: float
) -> None:
    value = black_scholes_price(
        contract(option_type, spot=spot, strike=strike),
        MarketEnvironment(risk_free_rate=-0.02, volatility=0.4, dividend_yield=0.01),
    )
    assert math.isfinite(value)
    assert value >= 0.0


def test_near_zero_maturity_and_volatility_approach_deterministic_value() -> None:
    option = contract(OptionType.CALL, spot=101.0, strike=100.0, maturity=1e-14)
    near_zero = black_scholes_price(
        option, MarketEnvironment(risk_free_rate=-0.01, volatility=1e-14)
    )
    deterministic = black_scholes_price(
        option, MarketEnvironment(risk_free_rate=-0.01, volatility=0.0)
    )
    assert near_zero == pytest.approx(deterministic, abs=1e-12)


def test_typed_pricer_wraps_analytical_price() -> None:
    result = BlackScholesPricer().price(
        contract(OptionType.CALL), MarketEnvironment(risk_free_rate=0.05, volatility=0.2)
    )
    assert result.price == pytest.approx(10.450583572185565)


def test_numerical_overflow_raises_clear_domain_error() -> None:
    with pytest.raises(NumericalError, match="floating-point range"):
        black_scholes_price(
            contract(OptionType.PUT, maturity=2.0),
            MarketEnvironment(risk_free_rate=-1e308, volatility=0.0),
        )
