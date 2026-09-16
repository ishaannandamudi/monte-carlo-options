import math

import numpy as np

from backend.app.models.barrier import (
    BarrierDirection,
    BarrierOptionContract,
    BarrierStyle,
    RebateTiming,
)
from backend.app.models.option import OptionType
from backend.app.payoffs.barrier import (
    discrete_discounted_barrier_payoffs,
    evaluate_discrete_barrier,
)

TIMES = np.array([0.0, 0.25, 0.5, 1.0])
INDICES = np.array([0, 1, 2, 3])


def contract(
    direction: BarrierDirection = BarrierDirection.DOWN,
    style: BarrierStyle = BarrierStyle.OUT,
    *,
    rebate: float = 0,
    timing: RebateTiming = RebateTiming.AT_EXPIRY,
) -> BarrierOptionContract:
    return BarrierOptionContract(
        option_type=OptionType.CALL,
        barrier_direction=direction,
        barrier_style=style,
        spot=100,
        strike=100,
        barrier=90 if direction is BarrierDirection.DOWN else 110,
        maturity_years=1,
        monitoring_schedule={"monitoring_count": 3},
        rebate=rebate,
        rebate_timing=timing,
    )


def test_down_barrier_first_hit_and_equality_convention() -> None:
    paths = np.array(
        [
            [100.0, 95.0, 90.0, 120.0],
            [100.0, 94.0, 91.0, 105.0],
            [89.0, 95.0, 100.0, 105.0],
        ]
    )
    event = evaluate_discrete_barrier(paths, TIMES, INDICES, BarrierDirection.DOWN, 90)
    np.testing.assert_array_equal(event.hit, [True, False, True])
    np.testing.assert_array_equal(event.first_hit_index, [2, -1, 0])
    assert event.first_hit_time[0] == 0.5
    assert math.isnan(event.first_hit_time[1])
    assert event.first_hit_time[2] == 0


def test_up_barrier_hit_and_unmonitored_crossing_is_ignored() -> None:
    paths = np.array([[100.0, 111.0, 105.0, 120.0], [100.0, 109.0, 110.0, 105.0]])
    event = evaluate_discrete_barrier(
        paths,
        TIMES,
        np.array([0, 2]),
        BarrierDirection.UP,
        110,
    )
    np.testing.assert_array_equal(event.hit, [False, True])
    np.testing.assert_array_equal(event.first_hit_index, [-1, 2])


def test_knock_out_zero_and_expiry_rebate_payoffs() -> None:
    terminal = np.array([120.0, 105.0])
    event = evaluate_discrete_barrier(
        np.array([[100.0, 90.0, 95.0, 120.0], [100.0, 95.0, 94.0, 105.0]]),
        TIMES,
        INDICES,
        BarrierDirection.DOWN,
        90,
    )
    zero = discrete_discounted_barrier_payoffs(terminal, event, contract(), 0.05)
    expiry = discrete_discounted_barrier_payoffs(
        terminal,
        event,
        contract(rebate=3, timing=RebateTiming.AT_EXPIRY),
        0.05,
    )
    np.testing.assert_allclose(zero, [0.0, 5 * math.exp(-0.05)])
    np.testing.assert_allclose(expiry, [3 * math.exp(-0.05), 5 * math.exp(-0.05)])


def test_at_hit_rebate_uses_actual_hit_time_with_negative_rate() -> None:
    observations = np.array([[100.0, 89.0, 95.0, 120.0], [100.0, 95.0, 90.0, 120.0]])
    event = evaluate_discrete_barrier(observations, TIMES, INDICES, BarrierDirection.DOWN, 90)
    result = discrete_discounted_barrier_payoffs(
        observations[:, -1],
        event,
        contract(rebate=4, timing=RebateTiming.AT_HIT),
        -0.02,
    )
    np.testing.assert_allclose(result, [4 * math.exp(0.02 * 0.25), 4 * math.exp(0.02 * 0.5)])


def test_knock_in_no_hit_rebate_and_pathwise_zero_rebate_parity() -> None:
    observations = np.array([[100.0, 89.0, 95.0, 120.0], [100.0, 95.0, 94.0, 105.0]])
    event = evaluate_discrete_barrier(observations, TIMES, INDICES, BarrierDirection.DOWN, 90)
    terminal = observations[:, -1]
    knock_in = discrete_discounted_barrier_payoffs(
        terminal, event, contract(style=BarrierStyle.IN), 0.0
    )
    knock_out = discrete_discounted_barrier_payoffs(terminal, event, contract(), 0.0)
    np.testing.assert_array_equal(knock_in + knock_out, [20.0, 5.0])
    rebated_in = discrete_discounted_barrier_payoffs(
        terminal,
        event,
        contract(style=BarrierStyle.IN, rebate=2),
        0.0,
    )
    np.testing.assert_array_equal(rebated_in, [20.0, 2.0])
