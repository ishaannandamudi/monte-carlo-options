import math

import numpy as np
import pytest
from numpy.random import Generator
from numpy.typing import NDArray
from pydantic import ValidationError

from backend.app.analytics.statistics import antithetic_standard_error
from backend.app.core.exceptions import NumericalError
from backend.app.models.digital import (
    DigitalOptionContract,
    DigitalSimulationConfig,
    DigitalType,
    DigitalVarianceReduction,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.models.simulation import SimulationConfig
from backend.app.pricing.black_scholes import black_scholes_price
from backend.app.pricing.digital import (
    DigitalMonteCarloPricer,
    digital_analytical_price,
    digital_monte_carlo_price,
)
from backend.app.simulation.base import StochasticProcess


def contract(
    option_type: OptionType = OptionType.CALL,
    digital_type: DigitalType = DigitalType.CASH_OR_NOTHING,
    *,
    spot: float = 100,
    strike: float = 100,
    maturity: float = 1,
    cash: float | None = 10,
) -> DigitalOptionContract:
    return DigitalOptionContract(
        option_type=option_type,
        spot=spot,
        strike=strike,
        maturity_years=maturity,
        digital_type=digital_type,
        cash_payout=cash if digital_type is DigitalType.CASH_OR_NOTHING else None,
    )


def test_cash_digital_put_call_identity_and_probability_link() -> None:
    market = MarketEnvironment(risk_free_rate=-0.02, volatility=0.25, dividend_yield=0.03)
    call = digital_analytical_price(contract(), market)
    put = digital_analytical_price(contract(OptionType.PUT), market)
    discounted_cash = 10 * math.exp(0.02)
    assert call.price + put.price == pytest.approx(discounted_cash, rel=2e-15)
    assert call.price / discounted_cash == pytest.approx(call.probability_itm)
    assert call.probability_itm + put.probability_itm == pytest.approx(1)


@pytest.mark.parametrize("option_type", list(OptionType))
def test_asset_cash_decomposition_matches_vanilla(option_type: OptionType) -> None:
    market = MarketEnvironment(risk_free_rate=0.04, volatility=0.3, dividend_yield=0.01)
    cash = digital_analytical_price(contract(option_type, cash=1), market).price
    asset = digital_analytical_price(
        contract(option_type, DigitalType.ASSET_OR_NOTHING, cash=None), market
    ).price
    vanilla = black_scholes_price(
        OptionContract(
            option_type=option_type,
            spot=100,
            strike=100,
            maturity_years=1,
        ),
        market,
    )
    decomposition = asset - 100 * cash if option_type is OptionType.CALL else 100 * cash - asset
    assert decomposition == pytest.approx(vanilla, rel=2e-14, abs=2e-14)


@pytest.mark.parametrize(
    "option_type,spot,expected",
    [
        (OptionType.CALL, 101, 10),
        (OptionType.CALL, 100, 0),
        (OptionType.PUT, 99, 10),
        (OptionType.PUT, 100, 0),
    ],
)
def test_zero_maturity_strict_equality(
    option_type: OptionType, spot: float, expected: float
) -> None:
    result = digital_analytical_price(
        contract(option_type, spot=spot, maturity=0),
        MarketEnvironment(risk_free_rate=1e308, volatility=1e308, dividend_yield=-1e308),
    )
    assert result.price == expected
    assert result.probability_itm == float(expected > 0)
    assert result.deterministic


@pytest.mark.parametrize("option_type", list(OptionType))
@pytest.mark.parametrize("digital_type", list(DigitalType))
def test_zero_volatility_analytical_and_monte_carlo_agree_exactly(
    option_type: OptionType, digital_type: DigitalType
) -> None:
    c = contract(option_type, digital_type, strike=95)
    market = MarketEnvironment(risk_free_rate=-0.01, volatility=0, dividend_yield=0.02)
    analytical = digital_analytical_price(c, market)
    mc = digital_monte_carlo_price(c, market, DigitalSimulationConfig(num_simulations=7))
    assert mc.price == analytical.price
    assert mc.standard_error == 0 and mc.deterministic


@pytest.mark.parametrize("option_type", list(OptionType))
@pytest.mark.parametrize("digital_type", list(DigitalType))
@pytest.mark.parametrize("mode", list(DigitalVarianceReduction))
def test_digital_monte_carlo_statistically_agrees_with_analytical(
    option_type: OptionType,
    digital_type: DigitalType,
    mode: DigitalVarianceReduction,
) -> None:
    c = contract(option_type, digital_type)
    market = MarketEnvironment(risk_free_rate=0.03, volatility=0.35, dividend_yield=0.01)
    result = digital_monte_carlo_price(
        c,
        market,
        DigitalSimulationConfig(num_simulations=40_000, seed=123, variance_reduction=mode),
    )
    assert abs(result.price - result.analytical_price) <= 5 * result.standard_error
    assert result.confidence_interval_low <= result.price <= result.confidence_interval_high
    assert 0 <= result.probability_itm <= 1
    assert math.isfinite(result.price) and math.isfinite(result.standard_error)


def test_same_seed_reproduces_and_near_strike_is_not_given_tiny_tolerance() -> None:
    c = contract(strike=100.001)
    market = MarketEnvironment(risk_free_rate=0, volatility=0.2)
    config = DigitalSimulationConfig(
        num_simulations=10_000,
        seed=88,
        variance_reduction=DigitalVarianceReduction.NONE,
    )
    first = digital_monte_carlo_price(c, market, config)
    same = digital_monte_carlo_price(c, market, config)
    different = digital_monte_carlo_price(c, market, config.model_copy(update={"seed": 89}))
    assert first.price == same.price and first.standard_error == same.standard_error
    assert first.price != different.price
    assert abs(first.price - first.analytical_price) <= 6 * first.standard_error


class FixedProcess(StochasticProcess):
    def __init__(self, values: NDArray[np.float64]) -> None:
        self.values = values

    def simulate_terminal(
        self,
        contract: OptionContract,
        market: MarketEnvironment,
        config: SimulationConfig,
        rng: Generator,
    ) -> NDArray[np.float64]:
        return self.values.copy()


def test_antithetic_standard_error_is_pair_level() -> None:
    terminal = np.array([110.0, 120.0, 80.0, 120.0, 110.0])
    c = contract(cash=2)
    result = DigitalMonteCarloPricer(FixedProcess(terminal)).price(
        c,
        MarketEnvironment(risk_free_rate=0, volatility=0.2),
        DigitalSimulationConfig(
            num_simulations=5,
            variance_reduction=DigitalVarianceReduction.ANTITHETIC,
        ),
    )
    payoffs = np.array([2.0, 2.0, 0.0, 2.0, 2.0])
    assert result.price == np.mean(payoffs)
    assert result.standard_error == pytest.approx(antithetic_standard_error(payoffs))


def test_zero_cash_payout_and_contract_validation() -> None:
    zero = contract(cash=0)
    result = digital_monte_carlo_price(
        zero,
        MarketEnvironment(risk_free_rate=0.03, volatility=0.2),
        DigitalSimulationConfig(num_simulations=10),
    )
    assert result.price == result.standard_error == 0
    with pytest.raises(ValidationError):
        DigitalOptionContract(
            option_type=OptionType.CALL,
            spot=100,
            strike=100,
            maturity_years=1,
            digital_type=DigitalType.ASSET_OR_NOTHING,
            cash_payout=1,
        )


@pytest.mark.parametrize("option_type", list(OptionType))
def test_zero_maturity_monte_carlo_uses_strict_equality(option_type: OptionType) -> None:
    c = contract(option_type, spot=100, strike=100, maturity=0)
    result = digital_monte_carlo_price(
        c,
        MarketEnvironment(risk_free_rate=1e308, volatility=1e308),
        DigitalSimulationConfig(num_simulations=5),
    )
    assert result.price == result.standard_error == result.probability_itm == 0
    assert result.deterministic


@pytest.mark.parametrize(
    "option_type,spot,expected_probability",
    [
        (OptionType.CALL, 1_000, 1.0),
        (OptionType.CALL, 1, 0.0),
        (OptionType.PUT, 1_000, 0.0),
        (OptionType.PUT, 1, 1.0),
    ],
)
def test_deep_itm_and_otm_analytical_prices_are_finite(
    option_type: OptionType, spot: float, expected_probability: float
) -> None:
    result = digital_analytical_price(
        contract(option_type, spot=spot),
        MarketEnvironment(risk_free_rate=0.03, volatility=0.1),
    )
    assert result.probability_itm == pytest.approx(expected_probability, abs=1e-12)
    assert math.isfinite(result.price)


def test_digital_rejects_volatility_variance_overflow() -> None:
    c = contract()
    market = MarketEnvironment(
        risk_free_rate=0.05,
        volatility=1e200,
        dividend_yield=0.0,
    )

    with pytest.raises(NumericalError, match="volatility variance"):
        digital_analytical_price(c, market)
    with pytest.raises(NumericalError, match="volatility variance"):
        digital_monte_carlo_price(
            c,
            market,
            DigitalSimulationConfig(num_simulations=100, seed=1),
        )
