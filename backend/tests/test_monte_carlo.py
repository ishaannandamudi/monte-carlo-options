import math

import numpy as np
import pytest

from backend.app.core.exceptions import StatisticalError
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.models.simulation import SimulationConfig, VarianceReduction
from backend.app.pricing.black_scholes import black_scholes_price
from backend.app.pricing.monte_carlo import MonteCarloPricer
from backend.app.simulation.base import StochasticProcess


def make_contract(
    option_type: OptionType = OptionType.CALL,
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


def test_same_seed_reproduces_all_quantitative_outputs() -> None:
    config = SimulationConfig(num_simulations=10_000, seed=81)
    market = MarketEnvironment(risk_free_rate=0.05, volatility=0.2)
    first = MonteCarloPricer().price(make_contract(), market, config)
    second = MonteCarloPricer().price(make_contract(), market, config)
    assert first.model_dump(exclude={"runtime_ms"}) == second.model_dump(exclude={"runtime_ms"})


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_call_and_put_results_are_finite(option_type: OptionType) -> None:
    result = MonteCarloPricer().price(
        make_contract(option_type),
        MarketEnvironment(risk_free_rate=0.03, volatility=0.25, dividend_yield=0.01),
        SimulationConfig(num_simulations=20_000, seed=14),
    )
    numeric_values = [
        value
        for value in result.model_dump().values()
        if isinstance(value, int | float) and not isinstance(value, bool)
    ]
    assert all(math.isfinite(float(value)) for value in numeric_values)


@pytest.mark.parametrize(
    ("option_type", "spot", "strike", "expected"),
    [
        (OptionType.CALL, 110.0, 100.0, 10.0),
        (OptionType.PUT, 90.0, 100.0, 10.0),
        (OptionType.CALL, 100.0, 100.0, 0.0),
    ],
)
def test_zero_maturity_equals_intrinsic(
    option_type: OptionType, spot: float, strike: float, expected: float
) -> None:
    result = MonteCarloPricer().price(
        make_contract(option_type, spot=spot, strike=strike, maturity=0.0),
        MarketEnvironment(risk_free_rate=-0.1, volatility=0.5),
        SimulationConfig(num_simulations=11),
    )
    assert result.price == expected
    assert result.standard_error == 0.0
    assert result.confidence_interval_low == expected
    assert result.confidence_interval_high == expected


def test_zero_volatility_matches_discounted_deterministic_payoff_and_itm() -> None:
    contract = make_contract(OptionType.CALL, strike=100.0, maturity=2.0)
    market = MarketEnvironment(risk_free_rate=0.05, volatility=0.0, dividend_yield=0.01)
    result = MonteCarloPricer().price(contract, market, SimulationConfig(num_simulations=9))
    terminal = 100.0 * math.exp((0.05 - 0.01) * 2.0)
    expected = math.exp(-0.05 * 2.0) * max(terminal - 100.0, 0.0)
    assert result.price == pytest.approx(expected)
    assert result.standard_error == 0.0
    assert result.probability_itm == 1.0


class FixedProcess(StochasticProcess):
    def simulate_terminal(self, contract, market, config, rng):  # type: ignore[no-untyped-def]
        return np.array([90.0, 110.0, 80.0, 120.0])


def test_monte_carlo_uses_pair_means_for_antithetic_se() -> None:
    result = MonteCarloPricer(FixedProcess()).price(
        make_contract(),
        MarketEnvironment(risk_free_rate=0.0, volatility=0.2),
        SimulationConfig(num_simulations=4, variance_reduction=VarianceReduction.ANTITHETIC),
    )
    # Discounted payoffs are [0, 10, 0, 20]; pair means [5, 10] have SE 2.5.
    assert result.price == 7.5
    assert result.standard_error == pytest.approx(2.5)
    assert result.standard_error != pytest.approx(result.discounted_payoff_std / 2.0)


def test_antithetic_reduces_estimated_variance_for_representative_vanilla_case() -> None:
    market = MarketEnvironment(risk_free_rate=0.05, volatility=0.2)
    plain = MonteCarloPricer().price(
        make_contract(),
        market,
        SimulationConfig(
            num_simulations=40_000, seed=42, variance_reduction=VarianceReduction.NONE
        ),
    )
    antithetic = MonteCarloPricer().price(
        make_contract(),
        market,
        SimulationConfig(
            num_simulations=40_000, seed=42, variance_reduction=VarianceReduction.ANTITHETIC
        ),
    )
    assert antithetic.standard_error < plain.standard_error


@pytest.mark.parametrize(
    "config",
    [
        SimulationConfig(num_simulations=1, variance_reduction=VarianceReduction.NONE),
        SimulationConfig(num_simulations=2, variance_reduction=VarianceReduction.ANTITHETIC),
    ],
)
def test_stochastic_pricing_requires_enough_independent_replications(
    config: SimulationConfig,
) -> None:
    with pytest.raises(StatisticalError):
        MonteCarloPricer().price(
            make_contract(),
            MarketEnvironment(risk_free_rate=0.05, volatility=0.2),
            config,
        )


def test_single_deterministic_observation_has_exact_zero_uncertainty() -> None:
    result = MonteCarloPricer().price(
        make_contract(maturity=0.0),
        MarketEnvironment(risk_free_rate=0.05, volatility=0.2),
        SimulationConfig(num_simulations=1),
    )
    assert result.standard_error == 0.0
    assert result.terminal_price_std == 0.0


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_equality_at_strike_is_not_itm(option_type: OptionType) -> None:
    result = MonteCarloPricer().price(
        make_contract(option_type, maturity=0.0),
        MarketEnvironment(risk_free_rate=0.05, volatility=0.2),
        SimulationConfig(num_simulations=3),
    )
    assert result.probability_itm == 0.0


@pytest.mark.parametrize("variance_reduction", list(VarianceReduction))
@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_monte_carlo_statistically_agrees_with_black_scholes(
    variance_reduction: VarianceReduction,
    option_type: OptionType,
) -> None:
    option = make_contract(option_type)
    market = MarketEnvironment(risk_free_rate=0.05, volatility=0.2, dividend_yield=0.01)
    result = MonteCarloPricer().price(
        option,
        market,
        SimulationConfig(
            num_simulations=100_000,
            seed=2026,
            variance_reduction=variance_reduction,
        ),
    )
    analytical = black_scholes_price(option, market)
    assert abs(result.price - analytical) <= 5.0 * result.standard_error
