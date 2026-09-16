import math

import numpy as np
import pytest

from backend.app.models.barrier import BarrierDirection, BarrierOptionContract, BarrierStyle
from backend.app.models.option import OptionType
from backend.app.payoffs.barrier import (
    bridge_discounted_barrier_payoffs,
    brownian_bridge_crossing_probabilities,
    brownian_bridge_survival_probabilities,
)


def contract(style: BarrierStyle) -> BarrierOptionContract:
    return BarrierOptionContract(
        option_type=OptionType.CALL,
        barrier_direction=BarrierDirection.DOWN,
        barrier_style=style,
        spot=100,
        strike=100,
        barrier=90,
        maturity_years=1,
    )


def test_manual_down_crossing_probability_and_endpoint_breach() -> None:
    observations = np.array([[100.0, 95.0], [100.0, 90.0]])
    probabilities = brownian_bridge_crossing_probabilities(
        observations,
        np.array([0.0, 0.5]),
        BarrierDirection.DOWN,
        90,
        0.2,
    )
    expected = math.exp(-2 * math.log(100 / 90) * math.log(95 / 90) / (0.2**2 * 0.5))
    assert probabilities[0, 0] == pytest.approx(expected)
    assert probabilities[1, 0] == 1.0


def test_up_crossing_irregular_intervals_and_probability_bounds() -> None:
    observations = np.array([[100.0, 102.0, 105.0], [100.0, 110.0, 105.0]])
    probabilities = brownian_bridge_crossing_probabilities(
        observations,
        np.array([0.0, 0.1, 1.0]),
        BarrierDirection.UP,
        110,
        0.3,
    )
    assert probabilities.shape == (2, 2)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert probabilities[1, 0] == probabilities[1, 1] == 1


def test_near_barrier_higher_sigma_and_longer_dt_raise_crossing_probability() -> None:
    times_short = np.array([0.0, 0.1])
    times_long = np.array([0.0, 1.0])
    near = np.array([[100.0, 100.0]])
    far = np.array([[120.0, 120.0]])
    near_probability = brownian_bridge_crossing_probabilities(
        near, times_short, BarrierDirection.DOWN, 90, 0.2
    )[0, 0]
    far_probability = brownian_bridge_crossing_probabilities(
        far, times_short, BarrierDirection.DOWN, 90, 0.2
    )[0, 0]
    high_sigma = brownian_bridge_crossing_probabilities(
        near, times_short, BarrierDirection.DOWN, 90, 0.4
    )[0, 0]
    long_dt = brownian_bridge_crossing_probabilities(
        near, times_long, BarrierDirection.DOWN, 90, 0.2
    )[0, 0]
    assert near_probability > far_probability
    assert high_sigma > near_probability
    assert long_dt > near_probability


def test_bridge_survival_weighting_has_exact_pathwise_parity() -> None:
    observations = np.array([[100.0, 95.0, 105.0], [100.0, 110.0, 120.0]])
    times = np.array([0.0, 0.4, 1.0])
    survival = brownian_bridge_survival_probabilities(
        observations, times, BarrierDirection.DOWN, 90, 0.2
    )
    knock_in = bridge_discounted_barrier_payoffs(
        observations[:, -1], survival, contract(BarrierStyle.IN), 0.0
    )
    knock_out = bridge_discounted_barrier_payoffs(
        observations[:, -1], survival, contract(BarrierStyle.OUT), 0.0
    )
    np.testing.assert_allclose(knock_in + knock_out, [5.0, 20.0], atol=2e-15)


def test_expiry_rebates_are_conditionally_weighted() -> None:
    base = contract(BarrierStyle.OUT)
    out = base.model_copy(update={"rebate": 3.0})
    knock_in = base.model_copy(update={"barrier_style": BarrierStyle.IN, "rebate": 3.0})
    terminal = np.array([120.0])
    survival = np.array([0.25])
    assert bridge_discounted_barrier_payoffs(terminal, survival, out, 0.0)[0] == 7.25
    assert bridge_discounted_barrier_payoffs(terminal, survival, knock_in, 0.0)[0] == 15.75


def test_bridge_variance_underflow_has_stable_zero_crossing_for_safe_endpoints() -> None:
    probabilities = brownian_bridge_crossing_probabilities(
        np.array([[100.0, 101.0]]),
        np.array([0.0, 1e-300]),
        BarrierDirection.DOWN,
        90,
        1e-300,
    )
    assert probabilities[0, 0] == 0.0
