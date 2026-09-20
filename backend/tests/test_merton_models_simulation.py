import math

import numpy as np
import pytest
from pydantic import ValidationError

from backend.app.models.market import MarketEnvironment
from backend.app.models.merton import (
    MertonJumpParameters,
    MertonPathSimulationConfig,
    MertonSimulationConfig,
)
from backend.app.models.option import ExerciseStyle, OptionContract, OptionType
from backend.app.simulation.merton import (
    simulate_merton_paths,
    simulate_merton_terminal,
    terminal_prices_from_drivers,
    theoretical_terminal_mean,
)
from backend.app.simulation.random import create_rng


def contract(maturity: float = 1.0) -> OptionContract:
    return OptionContract(
        option_type=OptionType.CALL,
        exercise_style=ExerciseStyle.EUROPEAN,
        spot=100.0,
        strike=100.0,
        maturity_years=maturity,
    )


def market(volatility: float = 0.2) -> MarketEnvironment:
    return MarketEnvironment(risk_free_rate=0.03, dividend_yield=0.01, volatility=volatility)


def parameters() -> MertonJumpParameters:
    return MertonJumpParameters(jump_intensity=0.8, jump_mean=-0.12, jump_volatility=0.25)


@pytest.mark.parametrize(
    "values",
    [
        {"jump_intensity": -0.1, "jump_mean": 0.0, "jump_volatility": 0.2},
        {"jump_intensity": 0.1, "jump_mean": 0.0, "jump_volatility": -0.2},
        {"jump_intensity": math.inf, "jump_mean": 0.0, "jump_volatility": 0.2},
        {"jump_intensity": 0.1, "jump_mean": math.nan, "jump_volatility": 0.2},
    ],
)
def test_parameter_validation(values: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        MertonJumpParameters(**values)


@pytest.mark.parametrize("jump_mean", [-0.3, 0.2])
def test_jump_mean_accepts_both_signs(jump_mean: float) -> None:
    assert (
        MertonJumpParameters(jump_intensity=0.2, jump_mean=jump_mean, jump_volatility=0.1).jump_mean
        == jump_mean
    )


def test_jump_compensator() -> None:
    params = parameters()
    expected = math.exp(params.jump_mean + 0.5 * params.jump_volatility**2) - 1.0
    assert params.jump_compensator == pytest.approx(expected, abs=1e-15)


def test_fixed_driver_terminal_distribution() -> None:
    counts = np.asarray([0, 1, 4], dtype=np.int64)
    diffusion = np.asarray([0.5, -0.25, 1.0])
    jump_normal = np.asarray([7.0, 0.4, -0.8])
    result = terminal_prices_from_drivers(
        contract(), market(), parameters(), counts, diffusion, jump_normal
    )
    params = parameters()
    drift = 0.03 - 0.01 - params.jump_intensity * params.jump_compensator - 0.5 * 0.2**2
    expected = 100.0 * np.exp(
        drift
        + 0.2 * diffusion
        + counts * params.jump_mean
        + np.sqrt(counts) * params.jump_volatility * jump_normal
    )
    assert result == pytest.approx(expected)


def test_terminal_reproducibility_and_jump_moments() -> None:
    config = MertonSimulationConfig(num_simulations=200_000, seed=17)
    first = simulate_merton_terminal(contract(), market(), parameters(), config, create_rng(17))
    second = simulate_merton_terminal(contract(), market(), parameters(), config, create_rng(17))
    different = simulate_merton_terminal(contract(), market(), parameters(), config, create_rng(18))
    assert np.array_equal(first.terminal_spots, second.terminal_spots)
    assert np.array_equal(first.jump_counts, second.jump_counts)
    assert not np.array_equal(first.terminal_spots, different.terminal_spots)
    expected = parameters().jump_intensity
    assert float(np.mean(first.jump_counts)) == pytest.approx(expected, abs=0.006)
    assert float(np.var(first.jump_counts)) == pytest.approx(expected, abs=0.012)
    assert float(np.mean(first.jump_counts == 0)) == pytest.approx(math.exp(-expected), abs=0.004)
    expected_spot = theoretical_terminal_mean(contract(), market())
    standard_error = float(np.std(first.terminal_spots, ddof=1)) / math.sqrt(config.num_simulations)
    assert abs(float(np.mean(first.terminal_spots)) - expected_spot) <= 4.0 * standard_error


def test_zero_maturity_has_no_rng_dependent_state() -> None:
    result = simulate_merton_terminal(
        contract(0.0),
        market(),
        parameters(),
        MertonSimulationConfig(num_simulations=11),
        create_rng(1),
    )
    assert np.array_equal(result.terminal_spots, np.full(11, 100.0))
    assert np.array_equal(result.jump_counts, np.zeros(11, dtype=np.int64))


def test_zero_diffusion_remains_stochastic_and_zero_jump_vol_is_stable() -> None:
    params = MertonJumpParameters(jump_intensity=2.0, jump_mean=-0.1, jump_volatility=0.0)
    result = simulate_merton_terminal(
        contract(),
        market(0.0),
        params,
        MertonSimulationConfig(num_simulations=5_000, seed=19),
        create_rng(19),
    )
    assert np.unique(result.terminal_spots).size > 2
    assert np.all(np.isfinite(result.terminal_spots))


def test_full_paths_shape_reproducibility_and_multiple_jumps() -> None:
    config = MertonPathSimulationConfig(num_simulations=2_000, num_time_steps=4, seed=12)
    params = MertonJumpParameters(jump_intensity=12.0, jump_mean=-0.02, jump_volatility=0.1)
    first = simulate_merton_paths(contract(), market(), params, config, create_rng(config.seed))
    second = simulate_merton_paths(contract(), market(), params, config, create_rng(config.seed))
    assert first.spot_paths.shape == (2_000, 5)
    assert first.interval_jump_counts.shape == (2_000, 4)
    assert np.array_equal(first.spot_paths, second.spot_paths)
    assert int(np.max(first.interval_jump_counts)) > 1
    assert np.all(first.spot_paths > 0.0)
