import math

import numpy as np
import pytest

from backend.app.models.multi_asset import MultiAssetMarketEnvironment
from backend.app.simulation.correlation import factorize_correlation
from backend.app.simulation.multi_asset_gbm import (
    multi_asset_normal_drivers,
    paths_from_normals,
    simulate_correlated_paths,
    simulate_correlated_terminal,
    terminal_prices_from_normals,
    theoretical_terminal_means,
)
from backend.app.simulation.random import create_rng


def market(rho: float = 0.35) -> MultiAssetMarketEnvironment:
    return MultiAssetMarketEnvironment(
        spots=(100.0, 80.0),
        risk_free_rate=-0.01,
        dividend_yields=(0.02, -0.01),
        volatilities=(0.2, 0.35),
        correlation_matrix=((1.0, rho), (rho, 1.0)),
        asset_ids=("ONE", "TWO"),
    )


@pytest.mark.parametrize("antithetic", [False, True])
def test_terminal_shape_positivity_seed_and_moments(antithetic: bool) -> None:
    sample = simulate_correlated_terminal(
        market(), 1.2, 40_000, create_rng(42), antithetic=antithetic
    )
    same = simulate_correlated_terminal(
        market(), 1.2, 40_000, create_rng(42), antithetic=antithetic
    )
    different = simulate_correlated_terminal(
        market(), 1.2, 40_000, create_rng(43), antithetic=antithetic
    )
    assert sample.shape == (40_000, 2) and np.all(sample > 0.0)
    np.testing.assert_array_equal(sample, same)
    assert not np.array_equal(sample, different)
    expected = theoretical_terminal_means(market(), 1.2)
    errors = 5 * np.std(sample, axis=0, ddof=1) / math.sqrt(sample.shape[0])
    assert np.all(np.abs(np.mean(sample, axis=0) - expected) <= errors)


def test_antithetic_vectors_are_paired_before_correlation() -> None:
    drivers = multi_asset_normal_drivers(5, 3, 2, create_rng(4), antithetic=True)
    np.testing.assert_array_equal(drivers[0], -drivers[1])
    np.testing.assert_array_equal(drivers[2], -drivers[3])
    terminal = terminal_prices_from_normals(market(), 1.0, drivers[:, 0, :])
    drift = np.asarray(
        [
            -0.01 - 0.02 - 0.5 * 0.2**2,
            -0.01 + 0.01 - 0.5 * 0.35**2,
        ]
    )
    np.testing.assert_allclose(
        np.log(terminal[0] / np.asarray(market().spots))
        + np.log(terminal[1] / np.asarray(market().spots)),
        2 * drift,
        atol=2e-15,
    )


def test_zero_maturity_and_all_zero_volatility_do_not_draw_rng() -> None:
    rng = create_rng(9)
    before = rng.bit_generator.state
    at_zero = simulate_correlated_terminal(market(), 0.0, 4, rng, antithetic=True)
    np.testing.assert_array_equal(at_zero, np.broadcast_to((100.0, 80.0), (4, 2)))
    assert before == rng.bit_generator.state

    deterministic_market = market().model_copy(update={"volatilities": (0.0, 0.0)})
    deterministic = simulate_correlated_terminal(deterministic_market, 2.0, 4, rng, antithetic=True)
    expected = np.asarray((100.0, 80.0)) * np.exp((-0.01 - np.asarray((0.02, -0.01))) * 2.0)
    np.testing.assert_allclose(deterministic, np.broadcast_to(expected, (4, 2)))
    assert before == rng.bit_generator.state


def test_one_zero_vol_asset_and_perfect_negative_correlation() -> None:
    mixed = market(-1.0).model_copy(update={"volatilities": (0.0, 0.3)})
    terminal = simulate_correlated_terminal(mixed, 1.0, 100, create_rng(7), antithetic=False)
    np.testing.assert_allclose(terminal[:, 0], 100 * math.exp(-0.03))
    assert np.std(terminal[:, 1]) > 0.0
    assert factorize_correlation(mixed.correlation_matrix).effective_rank == 1


def test_arbitrary_time_paths_use_exact_transitions_and_canonical_shape() -> None:
    times = np.array([0.1, 0.6, 1.0])
    drivers = np.array(
        [
            [[0.1, 0.2], [-0.3, 0.4], [0.5, -0.6]],
            [[-0.1, -0.2], [0.3, -0.4], [-0.5, 0.6]],
        ]
    )
    uncorrelated = market(0.0)
    paths = paths_from_normals(uncorrelated, times, drivers)
    assert paths.shape == (2, 4, 2)
    np.testing.assert_array_equal(paths[:, 0, :], np.broadcast_to((100.0, 80.0), (2, 2)))
    deltas = np.diff(np.concatenate(([0.0], times)))
    drift = np.asarray([-0.01 - 0.02 - 0.5 * 0.2**2, -0.01 + 0.01 - 0.5 * 0.35**2])
    expected = (
        drift[None, None, :] * deltas[None, :, None]
        + np.asarray((0.2, 0.35))[None, None, :] * np.sqrt(deltas)[None, :, None] * drivers
    )
    realized = np.diff(np.log(paths), axis=1)
    np.testing.assert_allclose(realized, expected, atol=2e-15)


def test_path_reproducibility_and_antithetic_pairing() -> None:
    times = np.array([0.2, 0.7, 1.1])
    first = simulate_correlated_paths(market(), times, 5, create_rng(5), antithetic=True)
    same = simulate_correlated_paths(market(), times, 5, create_rng(5), antithetic=True)
    np.testing.assert_array_equal(first, same)
    assert first.shape == (5, 4, 2)
    pair_sum = np.diff(np.log(first[0]), axis=0) + np.diff(np.log(first[1]), axis=0)
    deltas = np.diff(np.concatenate(([0.0], times)))
    drift = np.asarray([-0.01 - 0.02 - 0.5 * 0.2**2, -0.01 + 0.01 - 0.5 * 0.35**2])
    np.testing.assert_allclose(pair_sum, 2 * deltas[:, None] * drift, atol=3e-15)


@pytest.mark.parametrize("times", [np.array([0.0, 1.0]), np.array([0.5, 0.4])])
def test_invalid_path_times(times: np.ndarray) -> None:
    with pytest.raises(ValueError):
        paths_from_normals(market(), times, np.ones((2, len(times), 2)))
