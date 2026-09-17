import math
import os
from time import perf_counter

import pytest

from backend.app.models.heston import (
    HestonParameters,
    HestonSimulationConfig,
    HestonSimulationScheme,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.pricing.heston import heston_monte_carlo_price
from backend.app.pricing.heston_analytical import heston_analytical_price


def case() -> tuple[OptionContract, MarketEnvironment, HestonParameters]:
    return (
        OptionContract(
            option_type=OptionType.CALL,
            spot=100,
            strike=100,
            maturity_years=1,
        ),
        MarketEnvironment(risk_free_rate=0.02, dividend_yield=0.01, volatility=0.0),
        HestonParameters(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7),
    )


@pytest.mark.performance
@pytest.mark.parametrize("scheme", list(HestonSimulationScheme))
@pytest.mark.parametrize("steps", [50, 100, 250])
def test_heston_100k_scaling(scheme: HestonSimulationScheme, steps: int) -> None:
    contract, market, parameters = case()
    started = perf_counter()
    result = heston_monte_carlo_price(
        contract,
        market,
        parameters,
        HestonSimulationConfig(
            num_simulations=100_000,
            num_time_steps=steps,
            seed=42,
            scheme=scheme,
        ),
    )
    elapsed = perf_counter() - started
    assert math.isfinite(result.price) and elapsed < 60.0
    assert result.num_simulations == 100_000 and result.num_time_steps == steps


@pytest.mark.performance
@pytest.mark.skipif(
    os.getenv("RUN_LARGE_PERFORMANCE_TESTS") != "1",
    reason="set RUN_LARGE_PERFORMANCE_TESTS=1 for the 500k-path benchmark",
)
@pytest.mark.parametrize("scheme", list(HestonSimulationScheme))
def test_heston_500k_by_100_scaling(scheme: HestonSimulationScheme) -> None:
    contract, market, parameters = case()
    result = heston_monte_carlo_price(
        contract,
        market,
        parameters,
        HestonSimulationConfig(
            num_simulations=500_000,
            num_time_steps=100,
            seed=42,
            scheme=scheme,
        ),
    )
    assert math.isfinite(result.price)


@pytest.mark.performance
def test_heston_analytical_runtime() -> None:
    contract, market, parameters = case()
    started = perf_counter()
    result = heston_analytical_price(contract, market, parameters)
    assert result.price > 0.0 and perf_counter() - started < 5.0
