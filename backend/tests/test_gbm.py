import math

import numpy as np
import pytest

from backend.app.core.exceptions import NumericalError
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.models.simulation import SimulationConfig, VarianceReduction
from backend.app.simulation.gbm import GBMProcess, theoretical_terminal_mean
from backend.app.simulation.random import create_rng


def contract(maturity: float = 1.0) -> OptionContract:
    return OptionContract(
        option_type=OptionType.CALL,
        spot=100.0,
        strike=100.0,
        maturity_years=maturity,
    )


@pytest.mark.parametrize(
    "method",
    [VarianceReduction.ANTITHETIC, VarianceReduction.ANTITHETIC_CONTROL_VARIATE],
)
def test_antithetic_modes_preserve_exact_odd_shape_and_positive_values(
    method: VarianceReduction,
) -> None:
    config = SimulationConfig(
        num_simulations=1_001,
        variance_reduction=method,
    )
    values = GBMProcess().simulate_terminal(
        contract(),
        MarketEnvironment(risk_free_rate=0.03, volatility=0.2),
        config,
        create_rng(config.seed),
    )
    assert values.shape == (1_001,)
    assert np.all(values > 0.0)


def test_antithetic_normals_are_paired_and_odd_policy_is_exact() -> None:
    normals = GBMProcess._antithetic_normals(5, create_rng(7))
    assert normals.shape == (5,)
    np.testing.assert_array_equal(normals[0:4:2], -normals[1:4:2])
    assert math.isfinite(normals[-1])


def test_zero_maturity_is_spot_and_consumes_no_stochastic_behavior() -> None:
    config = SimulationConfig(num_simulations=7, seed=9)
    values = GBMProcess().simulate_terminal(
        contract(0.0),
        MarketEnvironment(risk_free_rate=-0.02, volatility=0.8),
        config,
        create_rng(9),
    )
    np.testing.assert_array_equal(values, np.full(7, 100.0))


def test_zero_volatility_is_deterministic_with_negative_rate() -> None:
    config = SimulationConfig(num_simulations=5)
    market = MarketEnvironment(risk_free_rate=-0.01, volatility=0.0, dividend_yield=0.02)
    values = GBMProcess().simulate_terminal(contract(2.0), market, config, create_rng(42))
    np.testing.assert_allclose(values, 100.0 * math.exp((-0.01 - 0.02) * 2.0), rtol=1e-15)


def test_same_seed_reproduces_and_different_seed_changes_values() -> None:
    config = SimulationConfig(
        num_simulations=100, seed=123, variance_reduction=VarianceReduction.NONE
    )
    process = GBMProcess()
    market = MarketEnvironment(risk_free_rate=0.03, volatility=0.2)
    first = process.simulate_terminal(contract(), market, config, create_rng(123))
    second = process.simulate_terminal(contract(), market, config, create_rng(123))
    different = process.simulate_terminal(contract(), market, config, create_rng(124))
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)


def test_terminal_mean_is_statistically_consistent_with_theory() -> None:
    config = SimulationConfig(
        num_simulations=80_000, seed=314, variance_reduction=VarianceReduction.NONE
    )
    market = MarketEnvironment(risk_free_rate=0.04, volatility=0.25, dividend_yield=0.01)
    values = GBMProcess().simulate_terminal(contract(1.5), market, config, create_rng(config.seed))
    theoretical = theoretical_terminal_mean(contract(1.5), market)
    standard_error = float(np.std(values, ddof=1) / math.sqrt(values.size))
    assert abs(float(np.mean(values)) - theoretical) <= 5.0 * standard_error


def test_extremely_small_maturity_remains_finite() -> None:
    config = SimulationConfig(num_simulations=100, seed=12)
    values = GBMProcess().simulate_terminal(
        contract(1e-15),
        MarketEnvironment(risk_free_rate=-0.2, volatility=0.3),
        config,
        create_rng(12),
    )
    assert np.all(np.isfinite(values))
    assert np.all(values > 0.0)


def test_numerical_overflow_raises_clear_domain_error() -> None:
    config = SimulationConfig(num_simulations=10)
    with pytest.raises(NumericalError, match="floating-point range"):
        GBMProcess().simulate_terminal(
            contract(2.0),
            MarketEnvironment(risk_free_rate=1e308, volatility=0.0),
            config,
            create_rng(42),
        )
