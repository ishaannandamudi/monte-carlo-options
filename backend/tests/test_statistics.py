import math

import numpy as np
import pytest

from backend.app.analytics.statistics import (
    antithetic_standard_error,
    confidence_interval,
    sample_mean,
    sample_standard_deviation,
    sample_variance,
    standard_error,
)
from backend.app.core.exceptions import StatisticalError


def test_known_sample_statistics_use_bessel_correction() -> None:
    values = np.array([1.0, 2.0, 3.0])
    assert sample_mean(values) == 2.0
    assert sample_variance(values) == 1.0
    assert sample_standard_deviation(values) == 1.0
    assert standard_error(values) == pytest.approx(1.0 / math.sqrt(3.0))


def test_confidence_interval_uses_configured_normal_quantile() -> None:
    low, high = confidence_interval(2.0, 1.0, 0.95)
    assert low == pytest.approx(0.0400360154599459)
    assert high == pytest.approx(3.959963984540054)


def test_zero_variance_and_zero_se_are_finite() -> None:
    values = np.full(10, 4.0)
    assert sample_variance(values) == 0.0
    assert standard_error(values) == 0.0
    assert confidence_interval(4.0, 0.0, 0.99) == (4.0, 4.0)


def test_pair_level_antithetic_standard_error() -> None:
    # Pair means are [2, 6], whose sample variance is 8; SE = sqrt(8 / 2) = 2.
    payoffs = np.array([1.0, 3.0, 5.0, 7.0])
    assert antithetic_standard_error(payoffs) == pytest.approx(2.0)
    assert antithetic_standard_error(payoffs) != pytest.approx(standard_error(payoffs))


def test_odd_antithetic_standard_error_uses_weighted_singleton_cluster() -> None:
    payoffs = np.array([1.0, 3.0, 5.0, 7.0, 10.0])
    estimate = float(np.mean(payoffs))
    totals = np.array([4.0, 12.0, 10.0])
    sizes = np.array([2.0, 2.0, 1.0])
    expected_variance = 3.0 / 2.0 * float(np.sum((totals - sizes * estimate) ** 2)) / 25.0
    assert antithetic_standard_error(payoffs) == pytest.approx(math.sqrt(expected_variance))


def test_statistics_reject_empty_non_vector_and_non_finite_data() -> None:
    with pytest.raises(StatisticalError):
        sample_mean([])
    with pytest.raises(StatisticalError):
        sample_mean([[1.0]])
    with pytest.raises(StatisticalError):
        sample_mean([math.inf])
    with pytest.raises(StatisticalError):
        sample_variance([1.0])
    with pytest.raises(StatisticalError):
        antithetic_standard_error([1.0, 2.0])
