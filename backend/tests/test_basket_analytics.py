import math

import pytest

from backend.app.analytics.basket import (
    compare_basket_methods,
    run_basket_convergence,
    run_correlation_sensitivity,
    run_weight_sensitivity,
)
from backend.app.core.exceptions import MultiAssetPricingError
from backend.app.models.basket import (
    BasketOptionContract,
    BasketSimulationConfig,
    BasketType,
    BasketVarianceReduction,
)
from backend.app.models.multi_asset import MultiAssetMarketEnvironment
from backend.app.models.option import OptionType


@pytest.fixture
def market() -> MultiAssetMarketEnvironment:
    return MultiAssetMarketEnvironment(
        spots=(100.0, 90.0),
        risk_free_rate=0.02,
        dividend_yields=(0.01, 0.0),
        volatilities=(0.2, 0.3),
        correlation_matrix=((1.0, 0.3), (0.3, 1.0)),
    )


def contract(basket_type: BasketType = BasketType.ARITHMETIC) -> BasketOptionContract:
    return BasketOptionContract(
        option_type=OptionType.CALL,
        strike=95,
        maturity_years=1,
        weights=(0.5, 0.5),
        basket_type=basket_type,
    )


def test_method_comparisons(market: MultiAssetMarketEnvironment) -> None:
    config = BasketSimulationConfig(num_simulations=4_000, seed=11)
    arithmetic = compare_basket_methods(contract(), market, config)
    geometric = compare_basket_methods(contract(BasketType.GEOMETRIC), market, config)
    assert [point.method for point in arithmetic.points] == list(BasketVarianceReduction)
    assert len(geometric.points) == 2
    assert all(point.analytical_reference is not None for point in geometric.points)


def test_convergence_seeds_and_slope_are_reproducible(
    market: MultiAssetMarketEnvironment,
) -> None:
    config = BasketSimulationConfig(
        num_simulations=500,
        seed=19,
        variance_reduction=BasketVarianceReduction.NONE,
    )
    first = run_basket_convergence(
        contract(BasketType.GEOMETRIC), market, config, [4_000, 1_000, 2_000]
    )
    same = run_basket_convergence(
        contract(BasketType.GEOMETRIC), market, config, [4_000, 1_000, 2_000]
    )
    assert [point.model_copy(update={"runtime_ms": 0}) for point in first.points] == [
        point.model_copy(update={"runtime_ms": 0}) for point in same.points
    ]
    assert first.empirical_standard_error_slope == same.empirical_standard_error_slope
    assert [point.num_simulations for point in first.points] == [1_000, 2_000, 4_000]
    assert len({point.seed for point in first.points}) == 3
    assert first.slope_point_count == 3
    assert first.empirical_standard_error_slope is not None
    assert -0.8 < first.empirical_standard_error_slope < -0.2


def test_correlation_sensitivity_uses_valid_crn_scenarios(
    market: MultiAssetMarketEnvironment,
) -> None:
    config = BasketSimulationConfig(num_simulations=2_000, seed=2)
    first = run_correlation_sensitivity(contract(), market, config, [-0.9, -0.5, 0.0, 0.5, 0.9])
    same = run_correlation_sensitivity(contract(), market, config, [-0.9, -0.5, 0.0, 0.5, 0.9])
    assert [point.model_copy(update={"runtime_ms": 0}) for point in first.points] == [
        point.model_copy(update={"runtime_ms": 0}) for point in same.points
    ]
    assert [point.rho for point in first.points] == [-0.9, -0.5, 0.0, 0.5, 0.9]
    assert all(math.isfinite(point.price) for point in first.points)
    with pytest.raises(MultiAssetPricingError):
        run_correlation_sensitivity(contract(), market, config, [-1.1])


def test_weight_sensitivity_endpoints_and_validation(
    market: MultiAssetMarketEnvironment,
) -> None:
    result = run_weight_sensitivity(
        contract(),
        market,
        BasketSimulationConfig(num_simulations=3_000, seed=8),
        [0.0, 0.5, 1.0],
    )
    assert [point.weights for point in result.points] == [
        (0.0, 1.0),
        (0.5, 0.5),
        (1.0, 0.0),
    ]
    assert all(sum(point.weights) == pytest.approx(1.0) for point in result.points)
    assert all(math.isfinite(point.price) for point in result.points)
    with pytest.raises(MultiAssetPricingError):
        run_weight_sensitivity(contract(), market, BasketSimulationConfig(), [-0.1])


def test_weight_sensitivity_requires_two_assets() -> None:
    one = MultiAssetMarketEnvironment(
        spots=(100,),
        risk_free_rate=0.0,
        dividend_yields=(0.0,),
        volatilities=(0.2,),
        correlation_matrix=((1.0,),),
    )
    one_contract = contract().model_copy(update={"weights": (1.0,)})
    with pytest.raises(MultiAssetPricingError, match="exactly two"):
        run_weight_sensitivity(one_contract, one, BasketSimulationConfig(), [0.5])


@pytest.mark.parametrize("counts", [[], [0, 10], [10, 10]])
def test_invalid_convergence_counts(market: MultiAssetMarketEnvironment, counts: list[int]) -> None:
    with pytest.raises(MultiAssetPricingError):
        run_basket_convergence(contract(), market, BasketSimulationConfig(), counts)
