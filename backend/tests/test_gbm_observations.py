import math

import numpy as np
import pytest

from backend.app.core.exceptions import NumericalError
from backend.app.models.market import MarketEnvironment
from backend.app.simulation.gbm import (
    observation_prices_from_normals,
    simulate_observation_prices,
)
from backend.app.simulation.random import create_rng


def test_irregular_time_transitions_are_exact() -> None:
    times = np.array([0.1, 0.4, 1.0])
    normals = np.array([[0.5, -1.0, 0.25], [-0.5, 1.0, -0.25]])
    market = MarketEnvironment(risk_free_rate=0.04, volatility=0.3, dividend_yield=0.01)
    paths = observation_prices_from_normals(100, market, times, normals)
    deltas = np.diff(np.concatenate(([0.0], times)))
    expected = (0.04 - 0.01 - 0.5 * 0.3**2) * deltas + 0.3 * np.sqrt(deltas) * normals
    realized = np.diff(
        np.column_stack((np.full(paths.shape[0], math.log(100)), np.log(paths))), axis=1
    )
    np.testing.assert_allclose(realized, expected, atol=2e-15)
    assert paths.shape == (2, 3) and np.all(paths > 0)


def test_observation_simulation_reproducibility_and_antithetic_pairing() -> None:
    times = np.array([0.2, 0.7, 1.0])
    market = MarketEnvironment(risk_free_rate=0.03, volatility=0.2)
    first = simulate_observation_prices(100, market, times, 5, create_rng(7), antithetic=True)
    same = simulate_observation_prices(100, market, times, 5, create_rng(7), antithetic=True)
    different = simulate_observation_prices(100, market, times, 5, create_rng(8), antithetic=True)
    np.testing.assert_array_equal(first, same)
    assert not np.array_equal(first, different)
    drift = (0.03 - 0.5 * 0.2**2) * times
    np.testing.assert_allclose(
        np.log(first[0] / 100) + np.log(first[1] / 100), 2 * drift, atol=2e-15
    )
    np.testing.assert_allclose(
        np.log(first[2] / 100) + np.log(first[3] / 100), 2 * drift, atol=2e-15
    )


def test_deterministic_observations_use_no_random_draws() -> None:
    rng = create_rng(42)
    before = rng.bit_generator.state
    times = np.array([0.2, 0.9])
    actual = simulate_observation_prices(
        100,
        MarketEnvironment(risk_free_rate=-0.02, volatility=0, dividend_yield=0.01),
        times,
        3,
        rng,
        antithetic=True,
    )
    expected = 100 * np.exp(-0.03 * times)
    np.testing.assert_allclose(actual, np.broadcast_to(expected, (3, 2)))
    assert before == rng.bit_generator.state
    empty = simulate_observation_prices(
        100,
        MarketEnvironment(risk_free_rate=0, volatility=0.2),
        np.array([]),
        3,
        rng,
        antithetic=False,
    )
    assert empty.shape == (3, 0)


@pytest.mark.parametrize(
    "times",
    [np.array([0.5, 0.4]), np.array([0.0, 0.5]), np.array([math.nan])],
)
def test_invalid_observation_times_fail(times: np.ndarray) -> None:
    with pytest.raises((ValueError, NumericalError)):
        observation_prices_from_normals(
            100, MarketEnvironment(volatility=0.2), times, np.ones((2, times.size))
        )


def test_nonfinite_observation_prices_raise_domain_error() -> None:
    with pytest.raises(NumericalError):
        observation_prices_from_normals(
            100,
            MarketEnvironment(risk_free_rate=1e308, volatility=0.2),
            np.array([1.0]),
            np.ones((2, 1)),
        )
