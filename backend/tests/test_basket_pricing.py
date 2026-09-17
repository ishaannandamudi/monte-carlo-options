import math

import numpy as np
import pytest

from backend.app.core.exceptions import MultiAssetPricingError
from backend.app.models.basket import (
    BasketOptionContract,
    BasketSimulationConfig,
    BasketType,
    BasketVarianceReduction,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.multi_asset import MultiAssetMarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.pricing.basket import (
    basket_monte_carlo_price,
    geometric_basket_analytical_price,
    geometric_basket_log_moments,
)
from backend.app.pricing.black_scholes import black_scholes_price
from backend.app.pricing.variance_reduction import apply_control_variate, cluster_observations
from backend.app.simulation.correlation import CorrelationFactorizationMethod


def market(
    rho: float = 0.4,
    *,
    volatilities: tuple[float, float] = (0.2, 0.3),
) -> MultiAssetMarketEnvironment:
    return MultiAssetMarketEnvironment(
        spots=(100.0, 95.0),
        risk_free_rate=-0.01,
        dividend_yields=(0.02, -0.005),
        volatilities=volatilities,
        correlation_matrix=((1.0, rho), (rho, 1.0)),
        asset_ids=("A", "B"),
    )


def contract(
    basket_type: BasketType = BasketType.ARITHMETIC,
    option_type: OptionType = OptionType.CALL,
    *,
    maturity: float = 1.0,
    strike: float = 100.0,
) -> BasketOptionContract:
    return BasketOptionContract(
        option_type=option_type,
        strike=strike,
        maturity_years=maturity,
        weights=(0.4, 0.6),
        basket_type=basket_type,
    )


@pytest.mark.parametrize("option_type", list(OptionType))
def test_single_asset_geometric_analytical_equals_black_scholes(option_type: OptionType) -> None:
    basket_contract = BasketOptionContract(
        option_type=option_type,
        strike=105,
        maturity_years=1.3,
        weights=(1.0,),
        basket_type=BasketType.GEOMETRIC,
    )
    multi_market = MultiAssetMarketEnvironment(
        spots=(100,),
        risk_free_rate=-0.02,
        dividend_yields=(0.03,),
        volatilities=(0.25,),
        correlation_matrix=((1.0,),),
    )
    vanilla = black_scholes_price(
        OptionContract(option_type=option_type, spot=100, strike=105, maturity_years=1.3),
        MarketEnvironment(risk_free_rate=-0.02, dividend_yield=0.03, volatility=0.25),
    )
    result = geometric_basket_analytical_price(basket_contract, multi_market)
    assert result.price == pytest.approx(vanilla, rel=2e-14, abs=2e-14)


def test_geometric_log_moments_match_explicit_formula() -> None:
    mean, variance = geometric_basket_log_moments(contract(BasketType.GEOMETRIC), market(-0.5))
    weights = np.array([0.4, 0.6])
    vol = np.array([0.2, 0.3])
    corr = np.array([[1.0, -0.5], [-0.5, 1.0]])
    expected_mean = weights @ (
        np.log([100.0, 95.0]) + (-0.01 - np.array([0.02, -0.005]) - 0.5 * vol**2)
    )
    assert mean == pytest.approx(expected_mean)
    assert variance == pytest.approx((weights * vol) @ corr @ (weights * vol))


@pytest.mark.parametrize("option_type", list(OptionType))
@pytest.mark.parametrize("strike", [1e-6, 1e6])
@pytest.mark.parametrize("volatilities", [(0.0, 0.3), (0.0, 0.0)])
def test_geometric_analytical_extremes_and_zero_volatility_components(
    option_type: OptionType,
    strike: float,
    volatilities: tuple[float, float],
) -> None:
    result = geometric_basket_analytical_price(
        contract(BasketType.GEOMETRIC, option_type, strike=strike),
        market(-0.5, volatilities=volatilities),
    )
    assert math.isfinite(result.price) and result.price >= 0.0
    if volatilities == (0.0, 0.0):
        assert result.deterministic and result.variance_log_basket == 0.0


@pytest.mark.parametrize("option_type", list(OptionType))
@pytest.mark.parametrize("rho", [-0.7, 0.0, 0.8, 1.0])
def test_geometric_mc_agrees_with_analytical(option_type: OptionType, rho: float) -> None:
    basket_contract = contract(BasketType.GEOMETRIC, option_type, strike=98)
    result = basket_monte_carlo_price(
        basket_contract,
        market(rho),
        BasketSimulationConfig(
            num_simulations=30_000,
            seed=123,
            variance_reduction=BasketVarianceReduction.ANTITHETIC,
        ),
    )
    analytical = geometric_basket_analytical_price(basket_contract, market(rho)).price
    assert abs(result.price - analytical) <= 5 * result.standard_error + 1e-12
    assert math.isfinite(result.price)
    if rho == 1.0:
        assert result.factorization.method is CorrelationFactorizationMethod.EIGEN


@pytest.mark.parametrize("option_type", list(OptionType))
@pytest.mark.parametrize("method", list(BasketVarianceReduction))
def test_arithmetic_mc_all_methods(
    option_type: OptionType, method: BasketVarianceReduction
) -> None:
    result = basket_monte_carlo_price(
        contract(BasketType.ARITHMETIC, option_type, strike=98),
        market(),
        BasketSimulationConfig(num_simulations=10_000, seed=9, variance_reduction=method),
    )
    assert result.price >= 0.0 and result.standard_error >= 0.0
    assert 0.0 <= result.probability_itm <= 1.0
    assert result.num_assets == 2 and math.isfinite(result.mean_basket_value)
    if method.uses_geometric_control:
        assert result.raw_price is not None
        assert result.adjusted_price == result.price
        assert result.analytical_geometric_price is not None
        assert result.control_variate_beta is not None


def test_control_variate_reduces_variance_in_representative_case() -> None:
    base = BasketSimulationConfig(
        num_simulations=20_000,
        seed=82,
        variance_reduction=BasketVarianceReduction.NONE,
    )
    raw = basket_monte_carlo_price(contract(), market(0.7), base)
    controlled = basket_monte_carlo_price(
        contract(),
        market(0.7),
        base.model_copy(update={"variance_reduction": BasketVarianceReduction.GEOMETRIC_CONTROL}),
    )
    assert controlled.standard_error < raw.standard_error
    assert controlled.variance_reduction_ratio is not None
    assert controlled.variance_reduction_ratio > 1.0


def test_pair_level_control_calculation_and_zero_variance_fallback() -> None:
    raw = np.array([10.0, 2.0, 8.0, 4.0, 7.0])
    control = np.array([8.0, 2.0, 6.0, 2.0, 5.0])
    adjustment = apply_control_variate(
        cluster_observations(raw, antithetic=True),
        cluster_observations(control, antithetic=True),
        4.5,
    )
    assert adjustment.beta == pytest.approx(3 / 14)
    np.testing.assert_allclose(
        adjustment.adjusted_observations,
        np.array([6.0, 6.0, 7.0]) - adjustment.beta * (np.array([5.0, 4.0, 5.0]) - 4.5),
    )
    fallback = apply_control_variate(
        cluster_observations(np.array([1.0, 2.0, 3.0]), antithetic=False),
        cluster_observations(np.ones(3), antithetic=False),
        1.0,
    )
    assert fallback.beta == 0.0 and fallback.adjusted_price == fallback.raw_price


@pytest.mark.parametrize("basket_type", list(BasketType))
@pytest.mark.parametrize("option_type", list(OptionType))
def test_zero_maturity_is_intrinsic_and_zero_se(
    basket_type: BasketType, option_type: OptionType
) -> None:
    basket_contract = contract(basket_type, option_type, maturity=0.0, strike=97)
    result = basket_monte_carlo_price(basket_contract, market())
    spots = np.array(market().spots)
    expected_basket = (
        float(np.array(basket_contract.weights) @ spots)
        if basket_type is BasketType.ARITHMETIC
        else float(np.exp(np.array(basket_contract.weights) @ np.log(spots)))
    )
    intrinsic = (
        max(expected_basket - 97, 0)
        if option_type is OptionType.CALL
        else max(97 - expected_basket, 0)
    )
    assert result.price == pytest.approx(intrinsic)
    assert result.standard_error == 0.0 and result.deterministic


@pytest.mark.parametrize("method", list(BasketVarianceReduction))
def test_all_zero_volatility_is_deterministic(method: BasketVarianceReduction) -> None:
    result = basket_monte_carlo_price(
        contract(),
        market(volatilities=(0.0, 0.0)),
        BasketSimulationConfig(num_simulations=101, variance_reduction=method),
    )
    assert result.standard_error == 0.0 and result.deterministic
    assert result.price >= 0.0


def test_identical_perfectly_correlated_assets_reduce_to_vanilla() -> None:
    identical_market = MultiAssetMarketEnvironment(
        spots=(100.0, 100.0),
        risk_free_rate=0.03,
        dividend_yields=(0.01, 0.01),
        volatilities=(0.2, 0.2),
        correlation_matrix=((1.0, 1.0), (1.0, 1.0)),
    )
    basket_contract = contract(BasketType.ARITHMETIC, strike=100)
    basket = basket_monte_carlo_price(
        basket_contract,
        identical_market,
        BasketSimulationConfig(num_simulations=20_000, seed=5),
    )
    vanilla = black_scholes_price(
        OptionContract(option_type=OptionType.CALL, spot=100, strike=100, maturity_years=1),
        MarketEnvironment(risk_free_rate=0.03, dividend_yield=0.01, volatility=0.2),
    )
    assert abs(basket.price - vanilla) <= 5 * basket.standard_error


@pytest.mark.parametrize("first_weight", [0.0, 1.0])
def test_arithmetic_weight_endpoints_reduce_to_single_asset(first_weight: float) -> None:
    basket_contract = contract().model_copy(update={"weights": (first_weight, 1.0 - first_weight)})
    result = basket_monte_carlo_price(
        basket_contract,
        market(),
        BasketSimulationConfig(num_simulations=30_000, seed=18),
    )
    index = 0 if first_weight == 1.0 else 1
    vanilla = black_scholes_price(
        OptionContract(
            option_type=OptionType.CALL,
            spot=market().spots[index],
            strike=basket_contract.strike,
            maturity_years=basket_contract.maturity_years,
        ),
        MarketEnvironment(
            risk_free_rate=market().risk_free_rate,
            dividend_yield=market().dividend_yields[index],
            volatility=market().volatilities[index],
        ),
    )
    assert abs(result.price - vanilla) <= 5 * result.standard_error


def test_reproducibility_and_incompatible_geometric_control() -> None:
    config = BasketSimulationConfig(num_simulations=2_000, seed=77)
    first = basket_monte_carlo_price(contract(), market(), config)
    same = basket_monte_carlo_price(contract(), market(), config)
    different = basket_monte_carlo_price(
        contract(), market(), config.model_copy(update={"seed": 78})
    )
    assert first.price == same.price and first.standard_error == same.standard_error
    assert first.price != different.price
    with pytest.raises(MultiAssetPricingError):
        basket_monte_carlo_price(
            contract(BasketType.GEOMETRIC),
            market(),
            config.model_copy(
                update={"variance_reduction": BasketVarianceReduction.GEOMETRIC_CONTROL}
            ),
        )


def test_weight_dimension_mismatch_is_clear() -> None:
    single_weight = contract().model_copy(update={"weights": (1.0,)})
    with pytest.raises(MultiAssetPricingError, match="weight count"):
        basket_monte_carlo_price(single_weight, market())
