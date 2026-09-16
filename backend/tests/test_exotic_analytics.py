import pytest

from backend.app.analytics.asian import (
    asian_convergence_inputs,
    compare_asian_methods,
    run_asian_convergence,
)
from backend.app.analytics.digital import (
    compare_digital_prices,
    digital_convergence_configs,
    run_digital_convergence,
)
from backend.app.core.exceptions import ExoticPricingError
from backend.app.models.asian import (
    AsianConvergenceMode,
    AsianOptionContract,
    AsianSimulationConfig,
    AsianVarianceReduction,
    AveragingType,
)
from backend.app.models.digital import (
    DigitalOptionContract,
    DigitalSimulationConfig,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionType


def asian(averaging_type: AveragingType = AveragingType.ARITHMETIC) -> AsianOptionContract:
    return AsianOptionContract(
        option_type=OptionType.CALL,
        spot=100,
        strike=100,
        maturity_years=1,
        averaging_type=averaging_type,
        averaging_schedule={"monitoring_count": 4},
    )


def market() -> MarketEnvironment:
    return MarketEnvironment(risk_free_rate=0.05, volatility=0.25, dividend_yield=0.01)


def test_arithmetic_and_geometric_method_comparisons() -> None:
    config = AsianSimulationConfig(num_simulations=3_000, seed=11)
    arithmetic = compare_asian_methods(asian(), market(), config)
    assert [point.method for point in arithmetic.points] == list(AsianVarianceReduction)
    assert arithmetic.points[0].variance_reduction_ratio == pytest.approx(1)
    assert arithmetic.points[-1].variance_reduction_ratio is not None
    assert arithmetic.points[-1].variance_reduction_ratio > 1
    assert all(point.analytical_reference is None for point in arithmetic.points)

    geometric = compare_asian_methods(asian(AveragingType.GEOMETRIC), market(), config)
    assert [point.method for point in geometric.points] == [
        AsianVarianceReduction.NONE,
        AsianVarianceReduction.ANTITHETIC,
    ]
    assert all(point.analytical_reference is not None for point in geometric.points)
    assert all(
        point.absolute_error is not None and point.absolute_error <= 6 * point.standard_error
        for point in geometric.points
    )


@pytest.mark.parametrize(
    "mode,values",
    [
        (AsianConvergenceMode.PATH_COUNT, [500, 1000, 2000]),
        (AsianConvergenceMode.MONITORING_COUNT, [2, 4, 8]),
    ],
)
def test_asian_convergence_has_independent_reproducible_children(
    mode: AsianConvergenceMode, values: list[int]
) -> None:
    c = asian(AveragingType.GEOMETRIC)
    config = AsianSimulationConfig(num_simulations=2_000, seed=21)
    inputs = asian_convergence_inputs(c, config, values, mode)
    assert len({point_config.seed for _, point_config in inputs}) == len(values)
    assert inputs == asian_convergence_inputs(c, config, list(reversed(values)), mode)
    result = run_asian_convergence(c, market(), config, values, mode)
    assert len(result.points) == len(values)
    assert all(point.analytical_reference is not None for point in result.points)
    if mode is AsianConvergenceMode.PATH_COUNT:
        assert [point.num_simulations for point in result.points] == sorted(values)
    else:
        assert [point.monitoring_count for point in result.points] == sorted(values)


def test_asian_convergence_validation() -> None:
    config = AsianSimulationConfig(num_simulations=100)
    for values in ([], [0], [1, 1]):
        with pytest.raises(ExoticPricingError):
            asian_convergence_inputs(asian(), config, values, AsianConvergenceMode.PATH_COUNT)
    explicit = AsianOptionContract.model_validate(
        {
            **asian().model_dump(),
            "averaging_schedule": {"monitoring_times": [0.5, 1.0]},
        }
    )
    with pytest.raises(ExoticPricingError, match="regular"):
        asian_convergence_inputs(
            explicit,
            config,
            [2, 4],
            AsianConvergenceMode.MONITORING_COUNT,
        )


def digital() -> DigitalOptionContract:
    return DigitalOptionContract(
        option_type=OptionType.PUT,
        spot=100,
        strike=100,
        maturity_years=1,
        cash_payout=5,
    )


def test_digital_comparison_and_convergence() -> None:
    config = DigitalSimulationConfig(num_simulations=5_000, seed=31)
    comparison = compare_digital_prices(digital(), market(), config)
    assert comparison.absolute_error == abs(
        comparison.monte_carlo.price - comparison.analytical.price
    )
    assert comparison.z_score is not None
    assert comparison.relative_error is not None
    configs = digital_convergence_configs(config, [500, 1000, 2000])
    assert configs == digital_convergence_configs(config, [2000, 500, 1000])
    assert len({item.seed for item in configs}) == 3
    result = run_digital_convergence(digital(), market(), config, [500, 1000, 2000])
    assert [point.num_simulations for point in result.points] == [500, 1000, 2000]
    assert len({point.seed for point in result.points}) == 3
    assert len({point.analytical_price for point in result.points}) == 1


def test_digital_convergence_validation() -> None:
    config = DigitalSimulationConfig(num_simulations=100)
    for counts in ([], [0], [1, 1]):
        with pytest.raises(ExoticPricingError):
            digital_convergence_configs(config, counts)
