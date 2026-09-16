import os

import pytest

from backend.app.models.barrier import (
    BarrierDirection,
    BarrierMonitoringMode,
    BarrierOptionContract,
    BarrierSimulationConfig,
    BarrierStyle,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionType
from backend.app.pricing.barrier import barrier_monte_carlo_price


def benchmark_case(paths: int, observations: int) -> None:
    base = BarrierOptionContract(
        option_type=OptionType.CALL,
        barrier_direction=BarrierDirection.DOWN,
        barrier_style=BarrierStyle.OUT,
        spot=100,
        strike=100,
        barrier=90,
        maturity_years=1,
        monitoring_schedule={"monitoring_count": observations},
    )
    market = MarketEnvironment(risk_free_rate=0.05, volatility=0.2, dividend_yield=0.01)
    config = BarrierSimulationConfig(num_simulations=paths, seed=42)
    discrete = barrier_monte_carlo_price(base, market, config)
    bridge = barrier_monte_carlo_price(
        base.model_copy(update={"monitoring_mode": BarrierMonitoringMode.BROWNIAN_BRIDGE}),
        market,
        config,
    )
    assert bridge.hit_probability >= discrete.hit_probability
    assert bridge.price <= discrete.price
    matrix_mb = 8 * paths * (observations + 1) / 1e6
    print(
        f"Barrier {paths} x {observations}: discrete={discrete.runtime_ms:.2f}ms "
        f"bridge={bridge.runtime_ms:.2f}ms states={matrix_mb:.2f}MB"
    )


@pytest.mark.performance
@pytest.mark.parametrize(
    "paths,observations",
    [(10_000, 52), (100_000, 52), (100_000, 252)],
)
def test_barrier_performance(paths: int, observations: int) -> None:
    benchmark_case(paths, observations)


@pytest.mark.performance
@pytest.mark.skipif(
    os.getenv("RUN_LARGE_BARRIER_BENCHMARKS") != "1",
    reason="100k x 1000 requires about 0.8 GB for the retained state matrix alone",
)
def test_large_barrier_performance() -> None:
    benchmark_case(100_000, 1_000)
