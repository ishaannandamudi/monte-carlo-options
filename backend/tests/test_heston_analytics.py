import math

import pytest

from backend.app.analytics.heston import (
    compare_heston_schemes,
    heston_convergence_configs,
    heston_implied_volatility_smile,
    heston_parameter_sensitivity,
    heston_variance_diagnostics,
    run_heston_convergence,
)
from backend.app.core.exceptions import HestonPricingError
from backend.app.models.heston import (
    HestonConvergenceMode,
    HestonParameterName,
    HestonParameters,
    HestonSimulationConfig,
    HestonSimulationScheme,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.simulation.random import spawn_child_seeds


def contract() -> OptionContract:
    return OptionContract(
        option_type=OptionType.CALL,
        spot=100,
        strike=100,
        maturity_years=1,
    )


def market() -> MarketEnvironment:
    return MarketEnvironment(risk_free_rate=0.02, dividend_yield=0.01, volatility=0.0)


def parameters() -> HestonParameters:
    return HestonParameters(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7)


def simulation(**updates: object) -> HestonSimulationConfig:
    return HestonSimulationConfig.model_validate(
        {
            "num_simulations": 2_000,
            "num_time_steps": 20,
            "seed": 3,
            "scheme": HestonSimulationScheme.QUADRATIC_EXPONENTIAL,
            **updates,
        }
    )


def test_scheme_comparison_uses_one_analytical_reference() -> None:
    result = compare_heston_schemes(contract(), market(), parameters(), simulation())
    assert [point.scheme for point in result.points] == list(HestonSimulationScheme)
    assert all(point.analytical_price == result.analytical.price for point in result.points)
    assert all(math.isfinite(point.z_score or 0.0) for point in result.points)
    assert result.points[0].negative_variance_proposal_fraction is not None
    assert result.points[1].negative_variance_proposal_fraction is None


def test_path_count_convergence_uses_sorted_independent_child_seeds() -> None:
    config = simulation()
    configs = heston_convergence_configs(
        config, [2_000, 500, 1_000], HestonConvergenceMode.PATH_COUNT
    )
    assert [point.num_simulations for point in configs] == [500, 1_000, 2_000]
    assert [point.seed for point in configs] == spawn_child_seeds(config.seed, 3)

    result = run_heston_convergence(
        contract(), market(), parameters(), config, [1_000, 500], HestonConvergenceMode.PATH_COUNT
    )
    assert [point.num_simulations for point in result.points] == [500, 1_000]
    assert len({point.seed for point in result.points}) == 2
    assert all(math.isfinite(point.bias) for point in result.points)


def test_time_step_convergence_keeps_common_seed_and_path_count() -> None:
    config = simulation(num_simulations=3_000, seed=19)
    result = run_heston_convergence(
        contract(), market(), parameters(), config, [20, 5, 10], HestonConvergenceMode.TIME_STEP
    )
    assert [point.num_time_steps for point in result.points] == [5, 10, 20]
    assert {point.seed for point in result.points} == {19}
    assert {point.num_simulations for point in result.points} == {3_000}


@pytest.mark.parametrize("scheme", list(HestonSimulationScheme))
def test_variance_diagnostics_track_theoretical_mean(scheme: HestonSimulationScheme) -> None:
    result = heston_variance_diagnostics(
        contract(),
        market(),
        parameters(),
        simulation(num_simulations=8_000, num_time_steps=20, scheme=scheme),
    )
    assert len(result.points) == 21
    assert result.points[0].simulated_mean_variance == pytest.approx(parameters().v0)
    assert (
        abs(result.points[-1].simulated_mean_variance - result.points[-1].theoretical_mean_variance)
        < 0.003
    )
    assert set(result.terminal_variance_quantiles) == {"p01", "p05", "p50", "p95", "p99"}
    if scheme is HestonSimulationScheme.FULL_TRUNCATION_EULER:
        assert result.negative_variance_proposal_fraction is not None
    else:
        assert result.terminal_variance_min >= 0.0


def test_heston_smile_is_ordered_and_exhibits_representative_negative_skew() -> None:
    result = heston_implied_volatility_smile(contract(), market(), parameters(), [120, 80, 100])
    assert [point.strike for point in result.points] == [80, 100, 120]
    assert all(point.implied_volatility is not None for point in result.points)
    implied = [point.implied_volatility for point in result.points]
    assert implied[0] is not None and implied[-1] is not None
    assert implied[0] > implied[-1]
    assert all(point.heston_price >= 0.0 for point in result.points)


@pytest.mark.parametrize("parameter", list(HestonParameterName))
def test_parameter_sensitivity_supports_each_parameter(parameter: HestonParameterName) -> None:
    grids = {
        HestonParameterName.V0: [0.02, 0.05],
        HestonParameterName.KAPPA: [1.0, 3.0],
        HestonParameterName.THETA: [0.02, 0.06],
        HestonParameterName.SIGMA_V: [0.2, 0.5],
        HestonParameterName.RHO: [-0.8, -0.2],
    }
    result = heston_parameter_sensitivity(
        contract(), market(), parameters(), parameter, grids[parameter]
    )
    assert result.parameter is parameter
    assert len(result.points) == 2
    assert all(point.implied_volatility is not None for point in result.points)


def test_research_grid_validation() -> None:
    with pytest.raises(HestonPricingError, match="positive"):
        heston_convergence_configs(simulation(), [10, 0], HestonConvergenceMode.PATH_COUNT)
    with pytest.raises(HestonPricingError, match="unique"):
        heston_implied_volatility_smile(contract(), market(), parameters(), [100, 100])
    with pytest.raises(HestonPricingError, match="rho"):
        heston_parameter_sensitivity(
            contract(), market(), parameters(), HestonParameterName.RHO, [-2.0]
        )
