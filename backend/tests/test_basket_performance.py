import math
from time import perf_counter

import numpy as np
import pytest

from backend.app.models.basket import (
    BasketOptionContract,
    BasketSimulationConfig,
    BasketType,
)
from backend.app.models.multi_asset import MultiAssetMarketEnvironment
from backend.app.models.option import OptionType
from backend.app.pricing.basket import basket_monte_carlo_price
from backend.app.simulation.correlation import build_equicorrelation_matrix, factorize_correlation
from backend.app.simulation.multi_asset_gbm import simulate_correlated_paths
from backend.app.simulation.random import create_rng


def case(assets: int) -> tuple[BasketOptionContract, MultiAssetMarketEnvironment]:
    return (
        BasketOptionContract(
            option_type=OptionType.CALL,
            strike=100,
            maturity_years=1,
            weights=(1 / assets,) * assets,
            basket_type=BasketType.ARITHMETIC,
        ),
        MultiAssetMarketEnvironment(
            spots=(100.0,) * assets,
            risk_free_rate=0.03,
            dividend_yields=(0.01,) * assets,
            volatilities=(0.2,) * assets,
            correlation_matrix=build_equicorrelation_matrix(assets, 0.25),
        ),
    )


@pytest.mark.performance
@pytest.mark.parametrize(
    "paths,assets", [(100_000, 2), (100_000, 10), (100_000, 50), (1_000_000, 2)]
)
def test_terminal_basket_scaling(paths: int, assets: int) -> None:
    contract, market = case(assets)
    started = perf_counter()
    result = basket_monte_carlo_price(
        contract, market, BasketSimulationConfig(num_simulations=paths, seed=42)
    )
    elapsed = perf_counter() - started
    assert math.isfinite(result.price) and elapsed < 30.0
    assert paths * assets * 8 <= 80_000_000


@pytest.mark.performance
def test_asset_count_factorization_scaling() -> None:
    for assets in (2, 5, 10, 25, 50, 100):
        _, market = case(assets)
        started = perf_counter()
        factor = factorize_correlation(market.correlation_matrix)
        assert factor.factor.shape == (assets, assets)
        assert perf_counter() - started < 5.0


@pytest.mark.performance
def test_full_path_research_allocation() -> None:
    _, market = case(5)
    paths = simulate_correlated_paths(
        market,
        np.linspace(0.02, 1.0, 50),
        10_000,
        create_rng(42),
        antithetic=True,
    )
    assert paths.shape == (10_000, 51, 5)
