import math

import numpy as np
import pytest

from backend.app.core.exceptions import QMCError
from backend.app.models.asian import AsianOptionContract, AveragingType
from backend.app.models.basket import BasketOptionContract, BasketType
from backend.app.models.market import MarketEnvironment
from backend.app.models.multi_asset import MultiAssetMarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.models.qmc import (
    QMCAsianRequest,
    QMCBasketRequest,
    QMCConfig,
    QMCEuropeanRequest,
    QMCPathConstruction,
    SamplingMethod,
)
from backend.app.pricing.qmc import price_asian_qmc, price_basket_qmc, price_european_qmc
from backend.app.simulation.multi_asset_gbm import terminal_prices_from_normals
from backend.app.simulation.qmc import generate_normal_samples


def market() -> MarketEnvironment:
    return MarketEnvironment(risk_free_rate=0.03, volatility=0.2, dividend_yield=0.01)


def contract() -> OptionContract:
    return OptionContract(
        option_type=OptionType.CALL,
        spot=100.0,
        strike=100.0,
        maturity_years=1.0,
    )


def scrambled(
    dimension: int, path: QMCPathConstruction = QMCPathConstruction.STANDARD
) -> QMCConfig:
    return QMCConfig(
        method=SamplingMethod.SCRAMBLED_SOBOL,
        dimension=dimension,
        num_samples=2_048,
        scramble=True,
        num_randomizations=8,
        seed=17,
        path_construction=path,
    )


def test_european_deterministic_and_randomized_qmc() -> None:
    deterministic = QMCConfig(
        method=SamplingMethod.SOBOL,
        dimension=1,
        num_samples=4_096,
        scramble=False,
        num_randomizations=1,
    )
    sobol = price_european_qmc(
        QMCEuropeanRequest(contract=contract(), market=market(), qmc=deterministic)
    )
    assert sobol.standard_error is None
    assert sobol.confidence_interval_low is None
    assert sobol.absolute_error is not None and sobol.absolute_error < 0.03

    request = QMCEuropeanRequest(contract=contract(), market=market(), qmc=scrambled(1))
    first = price_european_qmc(request)
    second = price_european_qmc(request)
    assert first.price == second.price
    assert [item.estimate for item in first.replicate_diagnostics] == [
        item.estimate for item in second.replicate_diagnostics
    ]
    assert first.metadata.reproducibility_signature == second.metadata.reproducibility_signature
    estimates = np.asarray([item.estimate for item in first.replicate_diagnostics])
    expected_se = float(np.std(estimates, ddof=1) / math.sqrt(estimates.size))
    assert first.standard_error == pytest.approx(expected_se)
    assert len(set(estimates)) > 1
    assert first.absolute_error is not None and first.absolute_error <= 5 * expected_se + 0.01


@pytest.mark.parametrize(
    "construction",
    [QMCPathConstruction.STANDARD, QMCPathConstruction.BROWNIAN_BRIDGE, QMCPathConstruction.PCA],
)
def test_geometric_asian_qmc_matches_analytical(construction: QMCPathConstruction) -> None:
    asian = AsianOptionContract(
        option_type=OptionType.CALL,
        spot=100.0,
        strike=100.0,
        maturity_years=1.0,
        averaging_type=AveragingType.GEOMETRIC,
        averaging_schedule={"monitoring_count": 12},
    )
    result = price_asian_qmc(
        QMCAsianRequest(contract=asian, market=market(), qmc=scrambled(12, construction))
    )
    assert result.reference_price is not None
    assert result.standard_error is not None
    assert abs(result.price - result.reference_price) < 0.03
    assert result.metadata.path_observation_times == list(asian.future_monitoring_times)
    assert result.metadata.path_coordinate_order is not None
    if construction is QMCPathConstruction.PCA:
        assert result.metadata.pca_cumulative_variance_explained is not None
        assert result.metadata.pca_cumulative_variance_explained[-1] == pytest.approx(1.0)


def test_arithmetic_asian_geometric_control_is_replicate_based() -> None:
    asian = AsianOptionContract(
        option_type=OptionType.PUT,
        spot=100.0,
        strike=102.0,
        maturity_years=1.0,
        averaging_type=AveragingType.ARITHMETIC,
        averaging_schedule={"monitoring_count": 12},
    )
    result = price_asian_qmc(
        QMCAsianRequest(
            contract=asian,
            market=market(),
            qmc=scrambled(12, QMCPathConstruction.BROWNIAN_BRIDGE),
            use_geometric_control=True,
        )
    )
    assert result.control_variate is True
    assert result.standard_error is not None and result.standard_error >= 0.0
    assert len(result.replicate_diagnostics) == 8


def basket_market() -> MultiAssetMarketEnvironment:
    return MultiAssetMarketEnvironment(
        asset_ids=("A", "B", "C"),
        spots=(100.0, 95.0, 105.0),
        volatilities=(0.2, 0.25, 0.18),
        risk_free_rate=0.03,
        dividend_yields=(0.01, 0.02, 0.0),
        correlation_matrix=((1.0, 0.4, 0.2), (0.4, 1.0, 0.3), (0.2, 0.3, 1.0)),
    )


def test_correlated_geometric_basket_qmc_matches_analytical() -> None:
    basket = BasketOptionContract(
        option_type=OptionType.CALL,
        strike=100.0,
        maturity_years=1.0,
        weights=(0.4, 0.3, 0.3),
        basket_type=BasketType.GEOMETRIC,
    )
    result = price_basket_qmc(
        QMCBasketRequest(contract=basket, market=basket_market(), qmc=scrambled(3))
    )
    assert result.reference_price is not None
    assert result.absolute_error is not None and result.absolute_error < 0.02
    assert result.standard_error is not None

    driver_config = scrambled(3).model_copy(update={"num_samples": 16_384})
    terminal = terminal_prices_from_normals(
        basket_market(), 1.0, generate_normal_samples(driver_config)
    )
    log_returns = np.log(terminal / np.asarray(basket_market().spots)[None, :])
    np.testing.assert_allclose(
        np.corrcoef(log_returns, rowvar=False),
        np.asarray(basket_market().correlation_matrix),
        atol=0.01,
    )


def test_arithmetic_basket_control_and_dimension_validation() -> None:
    basket = BasketOptionContract(
        option_type=OptionType.PUT,
        strike=100.0,
        maturity_years=1.0,
        weights=(0.4, 0.3, 0.3),
        basket_type=BasketType.ARITHMETIC,
    )
    result = price_basket_qmc(
        QMCBasketRequest(
            contract=basket,
            market=basket_market(),
            qmc=scrambled(3),
            use_geometric_control=True,
        )
    )
    assert result.control_variate
    with pytest.raises(QMCError):
        price_basket_qmc(
            QMCBasketRequest(contract=basket, market=basket_market(), qmc=scrambled(2))
        )
