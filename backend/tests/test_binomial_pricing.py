import math

import pytest

from backend.app.models.binomial import BinomialTreeConfig, DeterministicBoundaryCase
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.models.volatility import ImpliedVolatilityMarket
from backend.app.pricing.binomial import binomial_price, crr_parameters
from backend.app.pricing.black_scholes import black_scholes_price
from backend.app.pricing.implied_volatility import european_option_price_bounds


def option(
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


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_one_step_tree_matches_manual_discounted_expectation(option_type: OptionType) -> None:
    contract = option(option_type)
    market = MarketEnvironment(risk_free_rate=0.05, volatility=0.2)
    config = BinomialTreeConfig(steps=1)
    parameters = crr_parameters(contract, market, config)
    stock_down = contract.spot * parameters.down_factor
    stock_up = contract.spot * parameters.up_factor
    if option_type is OptionType.CALL:
        payoff_down = max(stock_down - contract.strike, 0.0)
        payoff_up = max(stock_up - contract.strike, 0.0)
    else:
        payoff_down = max(contract.strike - stock_down, 0.0)
        payoff_up = max(contract.strike - stock_up, 0.0)
    expected = parameters.discount_factor * (
        (1.0 - parameters.risk_neutral_probability) * payoff_down
        + parameters.risk_neutral_probability * payoff_up
    )
    assert binomial_price(contract, market, config).price == pytest.approx(expected, abs=1e-12)


def test_two_step_tree_matches_manual_backward_induction_and_delta() -> None:
    contract = option(OptionType.CALL)
    market = MarketEnvironment(risk_free_rate=0.05, volatility=0.2)
    config = BinomialTreeConfig(steps=2)
    parameters = crr_parameters(contract, market, config)
    terminal_stocks = [
        contract.spot * parameters.down_factor**2,
        contract.spot,
        contract.spot * parameters.up_factor**2,
    ]
    terminal = [max(stock - contract.strike, 0.0) for stock in terminal_stocks]
    down_value = parameters.discount_factor * (
        (1.0 - parameters.risk_neutral_probability) * terminal[0]
        + parameters.risk_neutral_probability * terminal[1]
    )
    up_value = parameters.discount_factor * (
        (1.0 - parameters.risk_neutral_probability) * terminal[1]
        + parameters.risk_neutral_probability * terminal[2]
    )
    expected_price = parameters.discount_factor * (
        (1.0 - parameters.risk_neutral_probability) * down_value
        + parameters.risk_neutral_probability * up_value
    )
    expected_delta = (up_value - down_value) / (
        contract.spot * parameters.up_factor - contract.spot * parameters.down_factor
    )
    result = binomial_price(contract, market, config)
    assert result.price == pytest.approx(expected_price, abs=1e-12)
    assert result.delta == pytest.approx(expected_delta, abs=1e-12)
    assert result.gamma is not None
    assert result.theta is not None


@pytest.mark.parametrize(
    ("option_type", "spot", "strike", "expected"),
    [
        (OptionType.CALL, 110.0, 100.0, 10.0),
        (OptionType.CALL, 90.0, 100.0, 0.0),
        (OptionType.PUT, 90.0, 100.0, 10.0),
        (OptionType.PUT, 110.0, 100.0, 0.0),
    ],
)
def test_zero_maturity_returns_intrinsic_without_tree_parameters(
    option_type: OptionType,
    spot: float,
    strike: float,
    expected: float,
) -> None:
    result = binomial_price(
        option(option_type, spot=spot, strike=strike, maturity=0.0),
        MarketEnvironment(risk_free_rate=-0.02, volatility=0.3, dividend_yield=0.04),
        BinomialTreeConfig(steps=1),
    )
    assert result.price == expected
    assert result.deterministic_boundary_case is DeterministicBoundaryCase.ZERO_MATURITY
    assert result.up_factor is None
    assert result.risk_neutral_probability is None
    assert result.delta is result.gamma is result.theta is None


@pytest.mark.parametrize(
    ("option_type", "rate", "dividend"),
    [
        (OptionType.CALL, 0.05, 0.0),
        (OptionType.PUT, 0.05, 0.03),
        (OptionType.CALL, -0.02, 0.01),
        (OptionType.PUT, -0.02, -0.03),
    ],
)
def test_zero_volatility_matches_deterministic_discounted_payoff_and_black_scholes(
    option_type: OptionType,
    rate: float,
    dividend: float,
) -> None:
    contract = option(option_type, spot=105.0, strike=100.0, maturity=1.5)
    market = MarketEnvironment(
        risk_free_rate=rate,
        volatility=0.0,
        dividend_yield=dividend,
    )
    terminal_spot = contract.spot * math.exp((rate - dividend) * contract.maturity_years)
    terminal_payoff = (
        max(terminal_spot - contract.strike, 0.0)
        if option_type is OptionType.CALL
        else max(contract.strike - terminal_spot, 0.0)
    )
    expected = math.exp(-rate * contract.maturity_years) * terminal_payoff
    result = binomial_price(contract, market, BinomialTreeConfig(steps=17))
    assert result.price == pytest.approx(expected, abs=1e-13)
    assert result.price == pytest.approx(black_scholes_price(contract, market), abs=1e-13)
    assert result.deterministic_boundary_case is DeterministicBoundaryCase.ZERO_VOLATILITY
    assert result.risk_neutral_probability is None


@pytest.mark.parametrize(
    ("option_type", "rate", "dividend", "volatility", "maturity"),
    [
        (OptionType.CALL, 0.05, 0.0, 0.2, 1.0),
        (OptionType.PUT, 0.05, 0.02, 0.2, 1.0),
        (OptionType.CALL, -0.01, 0.01, 0.25, 2.0),
        (OptionType.PUT, 0.01, 0.04, 0.3, 0.5),
        (OptionType.CALL, 0.02, -0.01, 0.1, 0.1),
        (OptionType.PUT, 0.02, 0.0, 0.7, 5.0),
    ],
)
def test_european_call_put_dividend_and_negative_rate_convergence(
    option_type: OptionType,
    rate: float,
    dividend: float,
    volatility: float,
    maturity: float,
) -> None:
    contract = option(option_type, maturity=maturity)
    market = MarketEnvironment(
        risk_free_rate=rate,
        volatility=volatility,
        dividend_yield=dividend,
    )
    result = binomial_price(contract, market, BinomialTreeConfig(steps=1_000))
    assert result.price == pytest.approx(black_scholes_price(contract, market), abs=0.015)
    assert math.isfinite(result.price)


@pytest.mark.parametrize(
    ("option_type", "spot", "strike"),
    [
        (OptionType.CALL, 150.0, 100.0),
        (OptionType.CALL, 60.0, 100.0),
        (OptionType.PUT, 60.0, 100.0),
        (OptionType.PUT, 150.0, 100.0),
    ],
)
def test_deep_itm_and_otm_prices_converge(
    option_type: OptionType,
    spot: float,
    strike: float,
) -> None:
    contract = option(option_type, spot=spot, strike=strike, maturity=2.0)
    market = MarketEnvironment(risk_free_rate=0.02, volatility=0.35, dividend_yield=0.01)
    result = binomial_price(contract, market, BinomialTreeConfig(steps=1_000))
    assert result.price == pytest.approx(black_scholes_price(contract, market), abs=0.02)


def test_put_call_parity_arbitrage_bounds_and_economic_monotonicity() -> None:
    market = MarketEnvironment(risk_free_rate=0.03, volatility=0.25, dividend_yield=0.01)
    config = BinomialTreeConfig(steps=600)
    call = binomial_price(option(OptionType.CALL), market, config)
    put = binomial_price(option(OptionType.PUT), market, config)
    parity = 100.0 * math.exp(-0.01) - 100.0 * math.exp(-0.03)
    assert call.price - put.price == pytest.approx(parity, abs=1e-10)

    low_spot_call = binomial_price(option(OptionType.CALL, spot=90.0), market, config)
    high_spot_call = binomial_price(option(OptionType.CALL, spot=110.0), market, config)
    low_strike_call = binomial_price(option(OptionType.CALL, strike=90.0), market, config)
    high_strike_call = binomial_price(option(OptionType.CALL, strike=110.0), market, config)
    low_spot_put = binomial_price(option(OptionType.PUT, spot=90.0), market, config)
    high_spot_put = binomial_price(option(OptionType.PUT, spot=110.0), market, config)
    low_strike_put = binomial_price(option(OptionType.PUT, strike=90.0), market, config)
    high_strike_put = binomial_price(option(OptionType.PUT, strike=110.0), market, config)
    assert low_spot_call.price <= high_spot_call.price
    assert low_strike_call.price >= high_strike_call.price
    assert low_spot_put.price >= high_spot_put.price
    assert low_strike_put.price <= high_strike_put.price

    for contract, result in [
        (option(OptionType.CALL), call),
        (option(OptionType.PUT), put),
    ]:
        bounds = european_option_price_bounds(
            contract,
            ImpliedVolatilityMarket(risk_free_rate=0.03, dividend_yield=0.01),
        )
        assert bounds.lower - 1e-10 <= result.price <= bounds.upper + 1e-10


def test_repeated_tree_prices_are_exactly_deterministic_except_runtime() -> None:
    contract = option(OptionType.CALL)
    market = MarketEnvironment(risk_free_rate=0.05, volatility=0.2)
    config = BinomialTreeConfig(steps=257)
    first = binomial_price(contract, market, config)
    second = binomial_price(contract, market, config)
    assert first.model_dump(exclude={"runtime_ms"}) == second.model_dump(exclude={"runtime_ms"})
