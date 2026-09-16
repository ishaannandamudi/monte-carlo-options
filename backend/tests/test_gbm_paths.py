import math

import numpy as np
import pytest
from pydantic import ValidationError

from backend.app.core.exceptions import NumericalError
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.models.simulation import PathSimulationConfig, VarianceReduction
from backend.app.simulation.gbm import GBMProcess, path_normal_drivers, paths_from_normals
from backend.app.simulation.random import create_rng


def contract(maturity: float = 1.0) -> OptionContract:
    return OptionContract(option_type=OptionType.PUT, spot=100, strike=100, maturity_years=maturity)


@pytest.mark.parametrize("count", [1, 2, 5, 100])
@pytest.mark.parametrize("mode", [VarianceReduction.NONE, VarianceReduction.ANTITHETIC])
def test_path_shape_positivity_and_reproducibility(count: int, mode: VarianceReduction) -> None:
    config = PathSimulationConfig(num_simulations=count, num_time_steps=7, variance_reduction=mode)
    market = MarketEnvironment(risk_free_rate=-0.02, volatility=0.3, dividend_yield=0.01)
    a = GBMProcess().simulate_paths(contract(), market, config, create_rng(42))
    b = GBMProcess().simulate_paths(contract(), market, config, create_rng(42))
    c = GBMProcess().simulate_paths(contract(), market, config, create_rng(43))
    assert a.shape == (count, 8)
    assert np.all(a[:, 0] == 100.0)
    assert np.all(a > 0.0)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_exact_transitions_and_interleaved_antithetic_paths() -> None:
    normals = path_normal_drivers(5, 3, create_rng(42), antithetic=True)
    np.testing.assert_array_equal(normals[0], -normals[1])
    np.testing.assert_array_equal(normals[2], -normals[3])
    primary = create_rng(42).standard_normal((3, 3))
    np.testing.assert_array_equal(normals[-1], primary[-1])
    market = MarketEnvironment(risk_free_rate=0.05, volatility=0.2, dividend_yield=0.01)
    paths = paths_from_normals(contract(), market, normals)
    expected_increments = (0.05 - 0.01 - 0.5 * 0.2**2) / 3 + 0.2 * math.sqrt(1 / 3) * normals
    np.testing.assert_allclose(
        np.log(paths[:, 1:] / paths[:, :-1]), expected_increments, atol=2e-15
    )


@pytest.mark.parametrize("maturity,volatility", [(0.0, 0.3), (1.0, 0.0)])
def test_deterministic_paths_do_not_draw_random_numbers(maturity: float, volatility: float) -> None:
    rng = create_rng(42)
    before = rng.bit_generator.state
    market = MarketEnvironment(risk_free_rate=-0.02, volatility=volatility, dividend_yield=0.01)
    paths = GBMProcess().simulate_paths(
        contract(maturity), market, PathSimulationConfig(num_simulations=3, num_time_steps=4), rng
    )
    expected = 100 * np.exp(-0.03 * np.linspace(0, maturity, 5))
    np.testing.assert_allclose(paths, np.broadcast_to(expected, (3, 5)))
    assert before == rng.bit_generator.state


def test_terminal_distribution_mean_and_log_variance() -> None:
    config = PathSimulationConfig(
        num_simulations=30_000, num_time_steps=10, variance_reduction=VarianceReduction.NONE
    )
    market = MarketEnvironment(risk_free_rate=0.03, volatility=0.25, dividend_yield=0.01)
    terminal = GBMProcess().simulate_paths(contract(), market, config, create_rng(123))[:, -1]
    assert abs(np.mean(terminal) - 100 * math.exp(0.02)) <= 5 * np.std(
        terminal, ddof=1
    ) / math.sqrt(terminal.size)
    variance = np.var(np.log(terminal / 100), ddof=1)
    assert abs(variance - 0.25**2) < 5 * 0.25**2 * math.sqrt(2 / (terminal.size - 1))


@pytest.mark.parametrize(
    "updates",
    [
        {"num_simulations": 0},
        {"num_time_steps": 0},
        {"seed": -1},
        {"confidence_level": 0},
        {"confidence_level": 1},
        {"variance_reduction": "CONTROL_VARIATE"},
    ],
)
def test_path_config_rejects_invalid_inputs(updates: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        PathSimulationConfig.model_validate(updates)


def test_nonfinite_path_representation_fails_clearly() -> None:
    with pytest.raises(NumericalError):
        paths_from_normals(
            contract(), MarketEnvironment(risk_free_rate=1e308, volatility=0.2), np.ones((3, 2))
        )
