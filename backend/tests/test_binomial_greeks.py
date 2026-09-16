import pytest

from backend.app.analytics.binomial import compare_binomial_to_black_scholes
from backend.app.models.binomial import BinomialTreeConfig
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.pricing.binomial import binomial_price
from backend.app.pricing.black_scholes_greeks import black_scholes_greeks


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_tree_native_greeks_converge_to_black_scholes(option_type: OptionType) -> None:
    contract = OptionContract(
        option_type=option_type,
        spot=100.0,
        strike=100.0,
        maturity_years=1.0,
    )
    market = MarketEnvironment(risk_free_rate=0.05, volatility=0.2, dividend_yield=0.01)
    tree = binomial_price(contract, market, BinomialTreeConfig(steps=1_000))
    analytical = black_scholes_greeks(contract, market)
    assert tree.delta == pytest.approx(analytical.delta, abs=5e-4)
    assert tree.gamma == pytest.approx(analytical.gamma, abs=5e-5)
    assert tree.theta == pytest.approx(analytical.theta, abs=0.02)


def test_one_step_delta_exists_but_gamma_and_theta_require_level_two() -> None:
    result = binomial_price(
        OptionContract(
            option_type=OptionType.CALL,
            spot=100.0,
            strike=100.0,
            maturity_years=1.0,
        ),
        MarketEnvironment(risk_free_rate=0.05, volatility=0.2),
        BinomialTreeConfig(steps=1),
    )
    assert result.delta is not None
    assert result.gamma is None
    assert result.theta is None


def test_typed_comparison_contains_price_and_greek_errors() -> None:
    result = compare_binomial_to_black_scholes(
        OptionContract(
            option_type=OptionType.PUT,
            spot=95.0,
            strike=100.0,
            maturity_years=1.3,
        ),
        MarketEnvironment(risk_free_rate=-0.01, volatility=0.3, dividend_yield=0.02),
        BinomialTreeConfig(steps=700),
    )
    assert result.absolute_error < 0.02
    assert result.delta.absolute_error is not None
    assert result.gamma.absolute_error is not None
    assert result.theta.absolute_error is not None
    assert result.delta.absolute_error < 0.002
