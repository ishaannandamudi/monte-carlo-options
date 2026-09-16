import math

import numpy as np
import pytest

from backend.app.analytics.statistics import antithetic_standard_error
from backend.app.models.barrier import (
    BarrierDirection,
    BarrierMonitoringMode,
    BarrierOptionContract,
    BarrierSimulationConfig,
    BarrierStyle,
    BarrierVarianceReduction,
    RebateTiming,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionType
from backend.app.pricing import barrier as barrier_module
from backend.app.pricing.barrier import (
    barrier_monte_carlo_price,
    continuous_barrier_analytical_price,
)
from backend.app.pricing.black_scholes import black_scholes_price


def contract(
    option_type: OptionType = OptionType.CALL,
    direction: BarrierDirection = BarrierDirection.DOWN,
    style: BarrierStyle = BarrierStyle.OUT,
    *,
    barrier: float | None = None,
    mode: BarrierMonitoringMode = BarrierMonitoringMode.DISCRETE,
    maturity: float = 1,
    rebate: float = 0,
    timing: RebateTiming = RebateTiming.AT_EXPIRY,
    count: int = 12,
) -> BarrierOptionContract:
    return BarrierOptionContract(
        option_type=option_type,
        barrier_direction=direction,
        barrier_style=style,
        spot=100,
        strike=100,
        barrier=barrier
        if barrier is not None
        else (90 if direction is BarrierDirection.DOWN else 110),
        maturity_years=maturity,
        monitoring_schedule={"monitoring_count": count},
        monitoring_mode=mode,
        rebate=rebate,
        rebate_timing=timing,
    )


MARKET = MarketEnvironment(risk_free_rate=0.05, volatility=0.2, dividend_yield=0.0)


@pytest.mark.parametrize(
    "barrier_contract,expected",
    [
        (contract(style=BarrierStyle.IN), 1.785111913939902),
        (contract(style=BarrierStyle.OUT), 8.665471658245663),
        (
            contract(
                OptionType.PUT,
                BarrierDirection.UP,
                BarrierStyle.IN,
            ),
            1.3753322113314645,
        ),
        (
            contract(
                OptionType.PUT,
                BarrierDirection.UP,
                BarrierStyle.OUT,
            ),
            4.198193810925506,
        ),
    ],
)
def test_known_continuous_barrier_benchmarks(
    barrier_contract: BarrierOptionContract, expected: float
) -> None:
    assert continuous_barrier_analytical_price(barrier_contract, MARKET).price == pytest.approx(
        expected, rel=2e-13
    )


@pytest.mark.parametrize("option_type", list(OptionType))
@pytest.mark.parametrize("direction", list(BarrierDirection))
def test_all_eight_continuous_formulas_obey_in_out_parity(
    option_type: OptionType, direction: BarrierDirection
) -> None:
    knock_in = contract(option_type, direction, BarrierStyle.IN)
    knock_out = contract(option_type, direction, BarrierStyle.OUT)
    vanilla = black_scholes_price(barrier_module._vanilla_contract(knock_in), MARKET)
    total = (
        continuous_barrier_analytical_price(knock_in, MARKET).price
        + continuous_barrier_analytical_price(knock_out, MARKET).price
    )
    assert total == pytest.approx(vanilla, rel=2e-14, abs=1e-14)


@pytest.mark.parametrize(
    "option_type,direction,style",
    [
        (OptionType.CALL, BarrierDirection.DOWN, BarrierStyle.OUT),
        (OptionType.CALL, BarrierDirection.DOWN, BarrierStyle.IN),
        (OptionType.PUT, BarrierDirection.UP, BarrierStyle.OUT),
        (OptionType.PUT, BarrierDirection.UP, BarrierStyle.IN),
    ],
)
def test_bridge_monte_carlo_agrees_with_continuous_analytical(
    option_type: OptionType,
    direction: BarrierDirection,
    style: BarrierStyle,
) -> None:
    c = contract(option_type, direction, style, mode=BarrierMonitoringMode.BROWNIAN_BRIDGE)
    result = barrier_monte_carlo_price(
        c,
        MARKET,
        BarrierSimulationConfig(num_simulations=30_000, seed=42),
    )
    benchmark = continuous_barrier_analytical_price(c, MARKET).price
    assert abs(result.price - benchmark) <= 5 * result.standard_error
    assert result.survival_probability_weighting
    assert result.hit_probability + result.no_hit_probability == pytest.approx(1)


def test_bridge_hit_probability_exceeds_discrete_for_sparse_monitoring() -> None:
    base = contract(count=2)
    config = BarrierSimulationConfig(num_simulations=10_000, seed=7)
    discrete = barrier_monte_carlo_price(base, MARKET, config)
    bridge = barrier_monte_carlo_price(
        base.model_copy(update={"monitoring_mode": BarrierMonitoringMode.BROWNIAN_BRIDGE}),
        MARKET,
        config,
    )
    assert bridge.hit_probability >= discrete.hit_probability
    assert bridge.price <= discrete.price


def test_sigma_zero_uses_exact_discrete_and_continuous_crossing_times() -> None:
    market = MarketEnvironment(risk_free_rate=0.1, volatility=0.0, dividend_yield=0.0)
    discrete_contract = contract(
        direction=BarrierDirection.UP,
        barrier=105,
        rebate=4,
        timing=RebateTiming.AT_HIT,
        count=2,
    )
    discrete = barrier_monte_carlo_price(discrete_contract, market)
    assert discrete.deterministic and discrete.mean_hit_time == 0.5
    assert discrete.price == pytest.approx(4 * math.exp(-0.1 * 0.5))

    bridge_contract = contract(
        direction=BarrierDirection.UP,
        barrier=105,
        mode=BarrierMonitoringMode.BROWNIAN_BRIDGE,
        count=2,
    )
    bridge = barrier_monte_carlo_price(bridge_contract, market)
    expected_hit_time = math.log(1.05) / 0.1
    assert bridge.mean_hit_time == pytest.approx(expected_hit_time)
    assert bridge.price == 0


def test_unscheduled_maturity_is_payoff_only_in_discrete_mode() -> None:
    regular = contract(direction=BarrierDirection.UP, barrier=110, count=1)
    base = BarrierOptionContract.model_validate(
        {**regular.model_dump(), "monitoring_schedule": {"monitoring_times": [0.1]}}
    )
    market = MarketEnvironment(risk_free_rate=0.2, volatility=0.0, dividend_yield=0.0)
    discrete = barrier_monte_carlo_price(base, market)
    bridge = barrier_monte_carlo_price(
        base.model_copy(update={"monitoring_mode": BarrierMonitoringMode.BROWNIAN_BRIDGE}),
        market,
    )
    assert discrete.hit_probability == 0 and discrete.price > 0
    assert bridge.hit_probability == 1 and bridge.price == 0


@pytest.mark.parametrize("style,expected", [(BarrierStyle.OUT, 3.0), (BarrierStyle.IN, 0.0)])
def test_zero_maturity_applies_initial_hit_semantics(style: BarrierStyle, expected: float) -> None:
    c = contract(style=style, barrier=100, maturity=0, rebate=3)
    result = barrier_monte_carlo_price(
        c,
        MarketEnvironment(risk_free_rate=1e308, volatility=1e308),
    )
    assert result.price == expected
    assert result.standard_error == 0
    assert result.hit_probability == 1


def test_already_hit_knock_in_equals_vanilla_and_out_rebate_is_exact() -> None:
    knock_in = contract(style=BarrierStyle.IN, barrier=100)
    knock_out = contract(
        style=BarrierStyle.OUT,
        barrier=100,
        rebate=2,
        timing=RebateTiming.AT_HIT,
    )
    assert continuous_barrier_analytical_price(knock_in, MARKET).price == pytest.approx(
        black_scholes_price(barrier_module._vanilla_contract(knock_in), MARKET)
    )
    out = barrier_monte_carlo_price(
        knock_out,
        MARKET,
        BarrierSimulationConfig(num_simulations=2_000),
    )
    assert out.price == 2 and out.standard_error == 0
    assert out.mean_hit_time == 0


def test_antithetic_standard_error_uses_complete_path_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    future = np.array([[120.0], [80.0], [110.0], [95.0], [105.0]])

    def fixed(*args: object, **kwargs: object) -> np.ndarray:
        return future.copy()

    monkeypatch.setattr(barrier_module, "simulate_observation_prices", fixed)
    c = contract(count=1)
    result = barrier_monte_carlo_price(
        c,
        MARKET,
        BarrierSimulationConfig(
            num_simulations=5,
            variance_reduction=BarrierVarianceReduction.ANTITHETIC,
        ),
    )
    discounted = math.exp(-0.05) * np.array([20.0, 0.0, 10.0, 0.0, 5.0])
    assert result.price == pytest.approx(float(np.mean(discounted)))
    assert result.standard_error == pytest.approx(antithetic_standard_error(discounted))


def test_seed_reproducibility_and_different_streams() -> None:
    c = contract(mode=BarrierMonitoringMode.BROWNIAN_BRIDGE, count=4)
    config = BarrierSimulationConfig(num_simulations=2_000, seed=91)
    first = barrier_monte_carlo_price(c, MARKET, config)
    same = barrier_monte_carlo_price(c, MARKET, config)
    different = barrier_monte_carlo_price(c, MARKET, config.model_copy(update={"seed": 92}))
    assert first.price == same.price and first.standard_error == same.standard_error
    assert first.price != different.price


@pytest.mark.parametrize(
    "barrier_contract",
    [
        contract(
            OptionType.CALL,
            BarrierDirection.DOWN,
            BarrierStyle.OUT,
            barrier=1e-200,
        ),
        contract(
            OptionType.PUT,
            BarrierDirection.UP,
            BarrierStyle.OUT,
            barrier=1e200,
        ),
    ],
)
def test_far_barriers_with_negative_rates_remain_finite(
    barrier_contract: BarrierOptionContract,
) -> None:
    result = continuous_barrier_analytical_price(
        barrier_contract,
        MarketEnvironment(risk_free_rate=-0.02, volatility=0.2, dividend_yield=0.03),
    )
    assert math.isfinite(result.price) and result.price > 0


def test_tiny_volatility_analytical_limit_is_deterministic() -> None:
    result = continuous_barrier_analytical_price(
        contract(),
        MarketEnvironment(risk_free_rate=0.05, volatility=1e-300, dividend_yield=0.01),
    )
    assert result.deterministic and math.isfinite(result.price)
