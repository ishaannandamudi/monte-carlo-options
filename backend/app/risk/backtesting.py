"""Stable VaR exception tests with strict loss-greater-than-VaR semantics."""

from __future__ import annotations

import math

from scipy.special import xlogy
from scipy.stats import chi2

from backend.app.models.risk import (
    BacktestTrafficLight,
    VaRBacktestRequest,
    VaRBacktestResult,
)


def _binomial_log_likelihood(successes: int, total: int, probability: float) -> float:
    return float(xlogy(successes, probability) + xlogy(total - successes, 1.0 - probability))


def _likelihood_ratio(null_log_likelihood: float, alternative_log_likelihood: float) -> float:
    statistic = max(0.0, 2.0 * (alternative_log_likelihood - null_log_likelihood))
    return statistic if math.isfinite(statistic) else 0.0


def backtest_var(request: VaRBacktestRequest) -> VaRBacktestResult:
    """Run Kupiec POF, Christoffersen independence, and conditional-coverage tests."""

    exceptions = [item.loss > item.predicted_var for item in request.observations]
    total = len(exceptions)
    count = sum(exceptions)
    expected_probability = 1.0 - request.confidence_level
    observed_probability = count / total
    null = _binomial_log_likelihood(count, total, expected_probability)
    alternative = _binomial_log_likelihood(count, total, observed_probability)
    kupiec = _likelihood_ratio(null, alternative)

    n00 = n01 = n10 = n11 = 0
    for previous, current in zip(exceptions[:-1], exceptions[1:], strict=True):
        if not previous and not current:
            n00 += 1
        elif not previous and current:
            n01 += 1
        elif previous and not current:
            n10 += 1
        else:
            n11 += 1
    transitions = total - 1
    if transitions > 0:
        unconditional = (n01 + n11) / transitions
        p01 = n01 / (n00 + n01) if n00 + n01 else 0.0
        p11 = n11 / (n10 + n11) if n10 + n11 else 0.0
        independent_ll = _binomial_log_likelihood(n01 + n11, transitions, unconditional)
        markov_ll = _binomial_log_likelihood(n01, n00 + n01, p01) + _binomial_log_likelihood(
            n11, n10 + n11, p11
        )
        independence = _likelihood_ratio(independent_ll, markov_ll)
    else:
        independence = 0.0
    conditional = kupiec + independence
    expected_count = total * expected_probability
    standard_deviation = math.sqrt(total * expected_probability * (1.0 - expected_probability))
    lower = max(0.0, expected_count - 2.0 * standard_deviation)
    upper = expected_count + 2.0 * standard_deviation
    if count < lower:
        traffic = BacktestTrafficLight.BELOW_EXPECTED
    elif count > upper:
        traffic = BacktestTrafficLight.ABOVE_EXPECTED
    else:
        traffic = BacktestTrafficLight.WITHIN_EXPECTED_RANGE
    return VaRBacktestResult(
        observation_count=total,
        exception_count=count,
        exception_rate=observed_probability,
        expected_exception_rate=expected_probability,
        exception_dates=[
            item.date
            for item, exception in zip(request.observations, exceptions, strict=True)
            if exception
        ],
        exceptions=exceptions,
        kupiec_statistic=kupiec,
        kupiec_p_value=float(chi2.sf(kupiec, 1)),
        christoffersen_statistic=independence,
        christoffersen_p_value=float(chi2.sf(independence, 1)),
        conditional_coverage_statistic=conditional,
        conditional_coverage_p_value=float(chi2.sf(conditional, 2)),
        transitions_00=n00,
        transitions_01=n01,
        transitions_10=n10,
        transitions_11=n11,
        traffic_light=traffic,
    )
