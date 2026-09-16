import pytest

from backend.app.analytics.barrier import (
    barrier_convergence_inputs,
    barrier_parity,
    compare_barrier_methods,
    run_barrier_convergence,
    run_barrier_level_sensitivity,
)
from backend.app.core.exceptions import BarrierPricingError
from backend.app.models.barrier import (
    BarrierConvergenceMode,
    BarrierDirection,
    BarrierMonitoringMode,
    BarrierOptionContract,
    BarrierSimulationConfig,
    BarrierStyle,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionType


def contract(mode: BarrierMonitoringMode = BarrierMonitoringMode.DISCRETE) -> BarrierOptionContract:
    return BarrierOptionContract(
        option_type=OptionType.CALL,
        barrier_direction=BarrierDirection.DOWN,
        barrier_style=BarrierStyle.OUT,
        spot=100,
        strike=100,
        barrier=90,
        maturity_years=1,
        monitoring_schedule={"monitoring_count": 4},
        monitoring_mode=mode,
    )


MARKET = MarketEnvironment(risk_free_rate=0.05, volatility=0.2, dividend_yield=0.01)


@pytest.mark.parametrize("mode", list(BarrierMonitoringMode))
def test_same_path_in_out_parity_is_machine_exact(mode: BarrierMonitoringMode) -> None:
    result = barrier_parity(
        contract(mode),
        MARKET,
        BarrierSimulationConfig(num_simulations=3_000, seed=17),
    )
    assert abs(result.parity_residual) < 1e-13
    assert result.knock_in_price + result.knock_out_price == pytest.approx(
        result.vanilla_price, abs=1e-13
    )


def test_parity_rejects_rebate() -> None:
    with pytest.raises(BarrierPricingError, match="zero rebate"):
        barrier_parity(contract().model_copy(update={"rebate": 1.0}), MARKET)


def test_initially_hit_parity_uses_active_in_value_and_zero_out_value() -> None:
    result = barrier_parity(
        contract().model_copy(update={"barrier": 100.0}),
        MARKET,
        BarrierSimulationConfig(num_simulations=2_000, seed=18),
    )
    assert result.knock_out_price == 0
    assert result.knock_in_price == result.vanilla_price
    assert result.parity_residual == 0


def test_comparison_reports_monitoring_bias_hit_difference_and_parity() -> None:
    result = compare_barrier_methods(
        contract(),
        MARKET,
        BarrierSimulationConfig(num_simulations=3_000, seed=19),
    )
    assert result.brownian_bridge.hit_probability >= result.discrete.hit_probability
    assert result.hit_probability_difference >= 0
    assert result.analytical_continuous is not None
    assert result.analytical_bridge_error is not None
    assert result.discrete_parity_residual is not None
    assert result.bridge_parity_residual is not None
    assert abs(result.discrete_parity_residual) < 1e-13
    assert abs(result.bridge_parity_residual) < 1e-13


@pytest.mark.parametrize(
    "mode,values",
    [
        (BarrierConvergenceMode.PATH_COUNT, [500, 1000, 2000]),
        (BarrierConvergenceMode.MONITORING_COUNT, [2, 4, 8]),
    ],
)
def test_barrier_convergence_uses_reproducible_independent_children(
    mode: BarrierConvergenceMode, values: list[int]
) -> None:
    config = BarrierSimulationConfig(num_simulations=2_000, seed=23)
    inputs = barrier_convergence_inputs(contract(), config, values, mode)
    assert inputs == barrier_convergence_inputs(contract(), config, list(reversed(values)), mode)
    assert len({point_config.seed for _, point_config in inputs}) == len(values)
    result = run_barrier_convergence(contract(), MARKET, config, values, mode)
    assert len(result.points) == len(values)
    assert all(point.analytical_continuous_price is not None for point in result.points)
    assert all(
        point.bridge_hit_probability >= point.discrete_hit_probability for point in result.points
    )


def test_barrier_convergence_validation() -> None:
    config = BarrierSimulationConfig(num_simulations=100)
    for values in ([], [0], [1, 1]):
        with pytest.raises(BarrierPricingError):
            barrier_convergence_inputs(
                contract(), config, values, BarrierConvergenceMode.PATH_COUNT
            )
    explicit = BarrierOptionContract.model_validate(
        {
            **contract().model_dump(),
            "monitoring_schedule": {"monitoring_times": [0.5, 1.0]},
        }
    )
    with pytest.raises(BarrierPricingError, match="regular"):
        barrier_convergence_inputs(
            explicit,
            config,
            [2, 4],
            BarrierConvergenceMode.MONITORING_COUNT,
        )


def test_barrier_level_sensitivity_uses_common_seed_and_reports_bridge_difference() -> None:
    result = run_barrier_level_sensitivity(
        contract(),
        MARKET,
        BarrierSimulationConfig(num_simulations=2_000, seed=31),
        [70, 80, 90, 95],
    )
    assert [point.barrier for point in result.levels] == [70, 80, 90, 95]
    assert all(
        point.bridge_hit_probability >= point.discrete_hit_probability for point in result.levels
    )
    assert result.levels[-1].bridge_hit_probability >= result.levels[0].bridge_hit_probability


def test_barrier_level_validation() -> None:
    for levels in ([], [0], [90, 90]):
        with pytest.raises(BarrierPricingError):
            run_barrier_level_sensitivity(
                contract(),
                MARKET,
                BarrierSimulationConfig(num_simulations=100),
                levels,
            )
