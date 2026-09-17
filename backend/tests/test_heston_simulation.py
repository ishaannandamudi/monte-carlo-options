import math

import numpy as np
import pytest
from pydantic import ValidationError

from backend.app.analytics.heston import empirical_heston_driver_correlation
from backend.app.models.heston import (
    HestonParameters,
    HestonSimulationConfig,
    HestonSimulationScheme,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.simulation.heston import (
    correlated_heston_normals,
    full_truncation_euler_step,
    heston_step_drivers,
    qe_conditional_moments,
    quadratic_exponential_variance_step,
    simulate_heston_from_drivers,
    simulate_heston_paths,
    simulate_heston_terminal,
    theoretical_variance_mean,
)
from backend.app.simulation.random import create_rng


def parameters(**updates: float) -> HestonParameters:
    values = {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "sigma_v": 0.3, "rho": -0.7}
    values.update(updates)
    return HestonParameters.model_validate(values)


def contract(maturity: float = 1.0) -> OptionContract:
    return OptionContract(
        option_type=OptionType.CALL,
        spot=100,
        strike=100,
        maturity_years=maturity,
    )


def market() -> MarketEnvironment:
    return MarketEnvironment(risk_free_rate=0.03, dividend_yield=0.01, volatility=0.0)


@pytest.mark.parametrize(
    "update",
    [
        {"v0": -0.01},
        {"kappa": -1.0},
        {"theta": -0.01},
        {"sigma_v": -0.1},
        {"rho": -1.01},
        {"rho": 1.01},
        {"v0": math.nan},
        {"sigma_v": math.inf},
    ],
)
def test_parameter_validation(update: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        parameters(**update)


def test_feller_diagnostics_do_not_reject_violations() -> None:
    satisfied = parameters()
    violated = parameters(kappa=0.5, theta=0.02, sigma_v=0.8)
    deterministic = parameters(sigma_v=0.0)
    assert satisfied.feller_satisfied
    assert satisfied.feller_ratio == pytest.approx(1.777777777777778)
    assert not violated.feller_satisfied and violated.feller_ratio < 1.0
    assert deterministic.feller_satisfied and deterministic.feller_ratio is None


@pytest.mark.parametrize("rho", [-1.0, -0.8, 0.0, 0.65, 1.0])
def test_driver_correlation(rho: float) -> None:
    result = empirical_heston_driver_correlation(rho, 40_000, 17)
    assert result.absolute_error < 0.015


def test_antithetic_driver_pairing_and_odd_singleton() -> None:
    variance, perpendicular, uniforms = heston_step_drivers(
        5, create_rng(8), antithetic=True, include_uniform=True
    )
    np.testing.assert_array_equal(variance[:4:2], -variance[1:4:2])
    np.testing.assert_array_equal(perpendicular[:4:2], -perpendicular[1:4:2])
    assert uniforms is not None
    np.testing.assert_allclose(uniforms[:4:2] + uniforms[1:4:2], 1.0)
    spots = correlated_heston_normals(variance, perpendicular, -0.7)
    np.testing.assert_array_equal(spots[:4:2], -spots[1:4:2])


def test_fixed_fte_step_matches_formula() -> None:
    log_spot = np.log(np.array([100.0, 90.0]))
    variance = np.array([0.04, -0.01])
    z_variance = np.array([-0.5, 2.0])
    z_spot = np.array([0.25, -1.0])
    dt = 0.1
    next_log, next_variance, negatives = full_truncation_euler_step(
        log_spot,
        variance,
        dt,
        z_variance,
        z_spot,
        0.03,
        0.01,
        parameters(),
    )
    positive = np.maximum(variance, 0.0)
    expected_log = log_spot + (0.02 - 0.5 * positive) * dt + np.sqrt(positive * dt) * z_spot
    expected_variance = (
        variance + 2.0 * (0.04 - positive) * dt + 0.3 * np.sqrt(positive * dt) * z_variance
    )
    np.testing.assert_allclose(next_log, expected_log)
    np.testing.assert_allclose(next_variance, expected_variance)
    assert negatives == int(np.count_nonzero(expected_variance < 0.0))


def test_negative_fte_proposal_uses_zero_coefficients_on_next_step() -> None:
    params = parameters(kappa=0.0, sigma_v=1.0, rho=0.0)
    first_log, first_variance, negatives = full_truncation_euler_step(
        np.array([math.log(100.0)]),
        np.array([0.01]),
        1.0,
        np.array([-10.0]),
        np.array([0.0]),
        0.0,
        0.0,
        params,
    )
    assert first_variance[0] < 0.0 and negatives == 1
    second_log, second_variance, _ = full_truncation_euler_step(
        first_log,
        first_variance,
        0.5,
        np.array([100.0]),
        np.array([100.0]),
        0.0,
        0.0,
        params,
    )
    assert second_log[0] == first_log[0]
    assert second_variance[0] == first_variance[0]


@pytest.mark.parametrize(
    "current,params,dt,expected_branch",
    [
        (0.04, parameters(), 0.1, "low"),
        (0.001, parameters(kappa=0.2, theta=0.04, sigma_v=1.0), 1.0, "high"),
    ],
)
def test_qe_conditional_moments(
    current: float, params: HestonParameters, dt: float, expected_branch: str
) -> None:
    count = 150_000
    state = np.full(count, current)
    mean, variance = qe_conditional_moments(state[:1], dt, params)
    psi = variance[0] / mean[0] ** 2
    assert (psi <= 1.5) == (expected_branch == "low")
    rng = create_rng(123)
    samples = quadratic_exponential_variance_step(
        state,
        dt,
        rng.standard_normal(count),
        rng.random(count),
        params,
    )
    assert abs(float(np.mean(samples)) - mean[0]) < 5 * math.sqrt(variance[0] / count)
    # The high-psi branch is a point-mass/exponential mixture with a noisy
    # fourth moment; this tolerance corresponds to roughly four standard
    # errors for the parameterization above.
    variance_tolerance = 0.08 if expected_branch == "high" else 0.04
    assert float(np.var(samples, ddof=1)) == pytest.approx(variance[0], rel=variance_tolerance)
    assert np.all(samples >= 0.0)


@pytest.mark.parametrize("scheme", list(HestonSimulationScheme))
def test_terminal_reproducibility_shape_and_positivity(
    scheme: HestonSimulationScheme,
) -> None:
    config = HestonSimulationConfig(
        num_simulations=2_001,
        num_time_steps=20,
        seed=4,
        scheme=scheme,
        antithetic=True,
    )
    first = simulate_heston_terminal(contract(), market(), parameters(), config, create_rng(4))
    same = simulate_heston_terminal(contract(), market(), parameters(), config, create_rng(4))
    different = simulate_heston_terminal(contract(), market(), parameters(), config, create_rng(5))
    assert first.terminal_spots.shape == first.terminal_variances.shape == (2_001,)
    assert np.all(first.terminal_spots > 0.0) and np.all(first.terminal_variances >= 0.0)
    np.testing.assert_array_equal(first.terminal_spots, same.terminal_spots)
    assert not np.array_equal(first.terminal_spots, different.terminal_spots)


def test_fixed_driver_terminal_matches_manual_single_fte_step() -> None:
    zv = np.array([[0.5], [-0.5]])
    zp = np.array([[0.25], [-0.25]])
    result = simulate_heston_from_drivers(
        contract(),
        market(),
        parameters(),
        HestonSimulationScheme.FULL_TRUNCATION_EULER,
        zv,
        zp,
    )
    zs = correlated_heston_normals(zv[:, 0], zp[:, 0], -0.7)
    expected_log = math.log(100) + (0.02 - 0.5 * 0.04) + math.sqrt(0.04) * zs
    np.testing.assert_allclose(result.terminal_spots, np.exp(expected_log))


@pytest.mark.parametrize("scheme", list(HestonSimulationScheme))
def test_full_path_shape_initial_state_and_variance_mean(
    scheme: HestonSimulationScheme,
) -> None:
    config = HestonSimulationConfig(
        num_simulations=30_000,
        num_time_steps=20,
        seed=33,
        scheme=scheme,
        antithetic=False,
    )
    result = simulate_heston_paths(contract(), market(), parameters(), config, create_rng(33))
    assert result.spot_paths.shape == result.variance_paths.shape == (30_000, 21)
    np.testing.assert_array_equal(result.spot_paths[:, 0], np.full(30_000, 100.0))
    np.testing.assert_array_equal(result.variance_paths[:, 0], np.full(30_000, 0.04))
    expected = theoretical_variance_mean(parameters(), 1.0)
    assert abs(float(np.mean(result.variance_paths[:, -1])) - expected) < 0.0015


@pytest.mark.parametrize("scheme", list(HestonSimulationScheme))
def test_zero_vol_of_vol_has_exact_deterministic_variance_path(
    scheme: HestonSimulationScheme,
) -> None:
    params = parameters(v0=0.09, theta=0.04, kappa=1.5, sigma_v=0.0)
    config = HestonSimulationConfig(
        num_simulations=5,
        num_time_steps=8,
        scheme=scheme,
    )
    result = simulate_heston_paths(contract(), market(), params, config, create_rng(42))
    times = np.linspace(0, 1, 9)
    expected = 0.04 + (0.09 - 0.04) * np.exp(-1.5 * times)
    np.testing.assert_allclose(result.variance_paths, np.broadcast_to(expected, (5, 9)))


def test_terminal_spot_mean_matches_risk_neutral_theory() -> None:
    config = HestonSimulationConfig(
        num_simulations=50_000,
        num_time_steps=50,
        seed=71,
        scheme=HestonSimulationScheme.QUADRATIC_EXPONENTIAL,
        antithetic=False,
    )
    terminal = simulate_heston_terminal(contract(), market(), parameters(), config, create_rng(71))
    expected = 100 * math.exp(0.02)
    tolerance = 5 * float(np.std(terminal.terminal_spots, ddof=1)) / math.sqrt(50_000)
    assert abs(float(np.mean(terminal.terminal_spots)) - expected) <= tolerance


@pytest.mark.parametrize(
    "params",
    [
        parameters(v0=0.0),
        parameters(theta=0.0),
        parameters(kappa=0.0),
        parameters(rho=-1.0),
        parameters(rho=1.0),
    ],
)
def test_boundary_parameters_remain_finite_under_fte(params: HestonParameters) -> None:
    result = simulate_heston_terminal(
        contract(),
        market(),
        params,
        HestonSimulationConfig(
            num_simulations=2_000,
            num_time_steps=50,
            scheme=HestonSimulationScheme.FULL_TRUNCATION_EULER,
        ),
        create_rng(42),
    )
    assert np.all(np.isfinite(result.terminal_spots))
    assert np.all(np.isfinite(result.terminal_variances))
