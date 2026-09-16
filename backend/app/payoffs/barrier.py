"""Pure vectorized barrier-event and realized-payoff helpers."""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from backend.app.core.exceptions import BarrierPricingError, NumericalError
from backend.app.core.numerics import finite_array_exp, finite_exp, require_finite
from backend.app.models.barrier import (
    BarrierDirection,
    BarrierOptionContract,
    BarrierStyle,
    RebateTiming,
)
from backend.app.models.option import OptionType
from backend.app.payoffs.vanilla import call_payoff, put_payoff


@dataclass(frozen=True)
class BarrierEventEvaluation:
    """Internal path-event state; `-1`/`nan` sentinels stay outside public models."""

    hit: NDArray[np.bool_]
    first_hit_index: NDArray[np.int64]
    first_hit_time: NDArray[np.float64]


def _path_inputs(
    observations: ArrayLike,
    observation_times: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    prices = np.asarray(observations, dtype=np.float64)
    times = np.asarray(observation_times, dtype=np.float64)
    if prices.ndim != 2 or prices.shape[0] == 0 or prices.shape[1] == 0:
        raise ValueError("barrier observations must be a non-empty two-dimensional matrix")
    if times.ndim != 1 or times.size != prices.shape[1]:
        raise ValueError("observation times must match barrier observation columns")
    require_finite(prices, context="barrier observations")
    require_finite(times, context="barrier observation times")
    if np.any(prices <= 0.0):
        raise NumericalError("barrier observations must be strictly positive")
    if times[0] != 0.0 or np.any(np.diff(times) <= 0.0):
        raise ValueError("barrier observation times must start at zero and strictly increase")
    return prices, times


def barrier_breaches(
    prices: ArrayLike,
    direction: BarrierDirection,
    barrier: float,
) -> NDArray[np.bool_]:
    """Apply the inclusive down/up hit convention to a price array."""

    values = np.asarray(prices, dtype=np.float64)
    require_finite(values, context="barrier event prices")
    if barrier <= 0.0 or not np.isfinite(barrier):
        raise ValueError("barrier must be positive and finite")
    if direction is BarrierDirection.DOWN:
        return np.asarray(values <= barrier, dtype=np.bool_)
    return np.asarray(values >= barrier, dtype=np.bool_)


def evaluate_discrete_barrier(
    observations: ArrayLike,
    observation_times: ArrayLike,
    monitored_indices: ArrayLike,
    direction: BarrierDirection,
    barrier: float,
) -> BarrierEventEvaluation:
    """Return first observed hits, always including the time-zero state."""

    prices, times = _path_inputs(observations, observation_times)
    indices = np.asarray(monitored_indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("at least the time-zero barrier index is required")
    if indices[0] != 0 or np.any(np.diff(indices) <= 0):
        raise ValueError("monitored indices must start at zero and strictly increase")
    if indices[-1] >= prices.shape[1]:
        raise ValueError("monitored index exceeds available barrier observations")

    monitored_hits = barrier_breaches(prices[:, indices], direction, barrier)
    hit = np.any(monitored_hits, axis=1)
    first_positions = np.argmax(monitored_hits, axis=1)
    first_indices = np.where(hit, indices[first_positions], -1).astype(np.int64)
    first_times = np.full(prices.shape[0], np.nan, dtype=np.float64)
    first_times[hit] = times[first_indices[hit]]
    return BarrierEventEvaluation(
        hit=np.asarray(hit, dtype=np.bool_),
        first_hit_index=first_indices,
        first_hit_time=first_times,
    )


def barrier_vanilla_payoff(
    terminal_prices: ArrayLike,
    contract: BarrierOptionContract,
) -> NDArray[np.float64]:
    if contract.option_type is OptionType.CALL:
        return call_payoff(terminal_prices, contract.strike)
    return put_payoff(terminal_prices, contract.strike)


def discrete_discounted_barrier_payoffs(
    terminal_prices: ArrayLike,
    event: BarrierEventEvaluation,
    contract: BarrierOptionContract,
    risk_free_rate: float,
) -> NDArray[np.float64]:
    """Apply discrete IN/OUT and rebate semantics with exactly one discounting step."""

    vanilla = barrier_vanilla_payoff(terminal_prices, contract)
    if vanilla.shape != event.hit.shape:
        raise ValueError("terminal prices and barrier events must have identical path counts")
    expiry_discount = finite_exp(
        -risk_free_rate * contract.maturity_years,
        context="barrier expiry discount factor",
    )
    discounted_vanilla = expiry_discount * vanilla

    if contract.barrier_style is BarrierStyle.OUT:
        payoff = np.where(event.hit, 0.0, discounted_vanilla)
        if contract.rebate > 0.0:
            if contract.rebate_timing is RebateTiming.AT_EXPIRY:
                rebate_values = np.full(vanilla.size, contract.rebate * expiry_discount)
            else:
                rebate_values = np.zeros(vanilla.size, dtype=np.float64)
                rebate_values[event.hit] = contract.rebate * finite_array_exp(
                    -risk_free_rate * event.first_hit_time[event.hit],
                    context="barrier hit-time rebate discount factors",
                )
            payoff = np.where(event.hit, rebate_values, payoff)
    else:
        payoff = np.where(event.hit, discounted_vanilla, contract.rebate * expiry_discount)

    result = np.asarray(payoff, dtype=np.float64)
    require_finite(result, context="discounted discrete barrier payoffs")
    return result


def brownian_bridge_crossing_probabilities(
    observations: ArrayLike,
    observation_times: ArrayLike,
    direction: BarrierDirection,
    barrier: float,
    volatility: float,
) -> NDArray[np.float64]:
    """Return exact conditional GBM barrier-crossing probabilities per interval."""

    prices, times = _path_inputs(observations, observation_times)
    if prices.shape[1] < 2:
        return np.empty((prices.shape[0], 0), dtype=np.float64)
    if volatility <= 0.0 or not np.isfinite(volatility):
        raise ValueError("Brownian-bridge volatility must be positive and finite")

    deltas = np.diff(times)
    logs = np.log(prices)
    barrier_log = np.log(barrier)
    left = logs[:, :-1]
    right = logs[:, 1:]
    left_breach = barrier_breaches(prices[:, :-1], direction, barrier)
    right_breach = barrier_breaches(prices[:, 1:], direction, barrier)
    breached = left_breach | right_breach

    if direction is BarrierDirection.DOWN:
        numerator = -2.0 * (left - barrier_log) * (right - barrier_log)
    else:
        numerator = -2.0 * (barrier_log - left) * (barrier_log - right)
    denominator = np.float64(volatility) * np.float64(volatility) * deltas
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        exponent = numerator / denominator
        safe_crossing = np.exp(exponent)
    safe_crossing = np.where(denominator == 0.0, 0.0, safe_crossing)
    probabilities = np.where(breached, 1.0, safe_crossing)
    if np.any(~np.isfinite(probabilities)):
        raise BarrierPricingError("Brownian-bridge crossing probability is non-finite")
    tolerance = 64.0 * np.finfo(np.float64).eps
    if np.any(probabilities < -tolerance) or np.any(probabilities > 1.0 + tolerance):
        raise BarrierPricingError("Brownian-bridge crossing probability is outside [0, 1]")
    return np.asarray(np.clip(probabilities, 0.0, 1.0), dtype=np.float64)


def brownian_bridge_survival_probabilities(
    observations: ArrayLike,
    observation_times: ArrayLike,
    direction: BarrierDirection,
    barrier: float,
    volatility: float,
) -> NDArray[np.float64]:
    """Return conditional survival while retaining only O(paths) bridge workspace."""

    prices, times = _path_inputs(observations, observation_times)
    if volatility <= 0.0 or not np.isfinite(volatility):
        raise ValueError("Brownian-bridge volatility must be positive and finite")
    if prices.shape[1] == 1:
        initial_hit = barrier_breaches(prices[:, 0], direction, barrier)
        return np.where(initial_hit, 0.0, 1.0).astype(np.float64)

    barrier_log = math.log(barrier)
    variance = np.float64(volatility) * np.float64(volatility)
    log_survival = np.zeros(prices.shape[0], dtype=np.float64)
    forced_hit = barrier_breaches(prices[:, 0], direction, barrier)
    for index, dt in enumerate(np.diff(times)):
        left_prices = prices[:, index]
        right_prices = prices[:, index + 1]
        breached = barrier_breaches(left_prices, direction, barrier) | barrier_breaches(
            right_prices, direction, barrier
        )
        left = np.log(left_prices)
        right = np.log(right_prices)
        if direction is BarrierDirection.DOWN:
            numerator = -2.0 * (left - barrier_log) * (right - barrier_log)
        else:
            numerator = -2.0 * (barrier_log - left) * (barrier_log - right)
        denominator = variance * dt
        with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
            crossing = np.exp(numerator / denominator)
        if denominator == 0.0:
            crossing.fill(0.0)
        crossing[breached] = 1.0
        if np.any(~np.isfinite(crossing)):
            raise BarrierPricingError("Brownian-bridge crossing probability is non-finite")
        tolerance = 64.0 * np.finfo(np.float64).eps
        if np.any(crossing < -tolerance) or np.any(crossing > 1.0 + tolerance):
            raise BarrierPricingError("Brownian-bridge crossing probability is outside [0, 1]")
        crossing = np.clip(crossing, 0.0, 1.0)
        newly_forced = crossing == 1.0
        forced_hit |= newly_forced
        safe = ~forced_hit
        with np.errstate(divide="ignore"):
            log_survival[safe] += np.log1p(-crossing[safe])
    survival = np.exp(log_survival)
    survival[forced_hit] = 0.0
    if np.any(~np.isfinite(survival)) or np.any((survival < 0.0) | (survival > 1.0)):
        raise BarrierPricingError("Brownian-bridge survival probability is invalid")
    return np.asarray(survival, dtype=np.float64)


def bridge_discounted_barrier_payoffs(
    terminal_prices: ArrayLike,
    survival_probability: ArrayLike,
    contract: BarrierOptionContract,
    risk_free_rate: float,
) -> NDArray[np.float64]:
    """Apply survival weighting for continuous IN/OUT value and expiry rebates."""

    survival = np.asarray(survival_probability, dtype=np.float64)
    if survival.ndim != 1:
        raise ValueError("survival probabilities must be one-dimensional")
    require_finite(survival, context="barrier survival probabilities")
    if np.any((survival < 0.0) | (survival > 1.0)):
        raise BarrierPricingError("barrier survival probabilities must lie in [0, 1]")
    if contract.rebate > 0.0 and contract.rebate_timing is RebateTiming.AT_HIT:
        raise BarrierPricingError(
            "survival weighting cannot value positive AT_HIT rebates without a hit-time law"
        )
    vanilla = barrier_vanilla_payoff(terminal_prices, contract)
    if vanilla.shape != survival.shape:
        raise ValueError("terminal prices and survival probabilities must have equal length")
    if contract.barrier_style is BarrierStyle.OUT:
        conditional = survival * vanilla + (1.0 - survival) * contract.rebate
    else:
        conditional = (1.0 - survival) * vanilla + survival * contract.rebate
    discount = finite_exp(
        -risk_free_rate * contract.maturity_years,
        context="barrier bridge expiry discount factor",
    )
    result = np.asarray(discount * conditional, dtype=np.float64)
    require_finite(result, context="discounted bridge barrier payoffs")
    return result
