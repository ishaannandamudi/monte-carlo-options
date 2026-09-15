"""Reusable finite-sample statistics for Monte Carlo estimators."""

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

from backend.app.core.exceptions import StatisticalError


def _observations(values: ArrayLike) -> NDArray[np.float64]:
    observations = np.asarray(values, dtype=np.float64)
    if observations.ndim != 1:
        raise StatisticalError("observations must be a one-dimensional array")
    if observations.size == 0:
        raise StatisticalError("at least one observation is required")
    if not np.all(np.isfinite(observations)):
        raise StatisticalError("observations must all be finite")
    return observations


def sample_mean(values: ArrayLike) -> float:
    """Return the arithmetic sample mean."""

    return float(np.mean(_observations(values)))


def sample_variance(values: ArrayLike) -> float:
    """Return the Bessel-corrected sample variance.

    Two observations are required to estimate dispersion with one lost degree of
    freedom. Constant samples of size two or greater return zero.
    """

    observations = _observations(values)
    if observations.size == 1:
        raise StatisticalError("at least two observations are required for sample variance")
    if np.all(observations == observations[0]):
        return 0.0
    result = float(np.var(observations, ddof=1))
    if not math.isfinite(result) or result < 0.0:
        raise StatisticalError("sample variance is not a finite non-negative value")
    return result


def sample_standard_deviation(values: ArrayLike) -> float:
    """Return the Bessel-corrected sample standard deviation."""

    return math.sqrt(sample_variance(values))


def standard_error(values: ArrayLike) -> float:
    """Estimate the standard error of a mean of independent observations."""

    observations = _observations(values)
    return sample_standard_deviation(observations) / math.sqrt(observations.size)


def antithetic_standard_error(discounted_payoffs: ArrayLike) -> float:
    """Estimate mean uncertainty using independent antithetic clusters.

    Consecutive observations are ``Z, -Z`` payoffs. With an even path count this
    is exactly ``std(pair_means, ddof=1) / sqrt(number_of_pairs)``. For an odd
    count the final singleton forms its own independent cluster, and a
    finite-sample cluster-sandwich estimate preserves its half weight in the
    overall path mean. Fewer than two clusters have no observable between-cluster
    dispersion and raise a clear statistical error.
    """

    payoffs = _observations(discounted_payoffs)
    pair_count, remainder = divmod(payoffs.size, 2)
    cluster_count = pair_count + remainder
    if cluster_count < 2:
        raise StatisticalError(
            "at least two independent antithetic clusters are required for standard error"
        )
    if np.all(payoffs == payoffs[0]):
        return 0.0

    pair_totals = payoffs[: 2 * pair_count].reshape(pair_count, 2).sum(axis=1)
    if remainder:
        cluster_totals = np.concatenate((pair_totals, payoffs[-1:]))
        cluster_sizes = np.concatenate(
            (np.full(pair_count, 2.0, dtype=np.float64), np.ones(1, dtype=np.float64))
        )
    else:
        cluster_totals = pair_totals
        cluster_sizes = np.full(pair_count, 2.0, dtype=np.float64)

    estimate = float(np.mean(payoffs))
    cluster_scores = cluster_totals - cluster_sizes * estimate
    variance_of_mean = (
        cluster_count
        / (cluster_count - 1)
        * float(np.dot(cluster_scores, cluster_scores))
        / payoffs.size**2
    )
    if not math.isfinite(variance_of_mean) or variance_of_mean < 0.0:
        raise StatisticalError("antithetic estimator variance is invalid")
    return math.sqrt(variance_of_mean)


def confidence_interval(
    estimate: float,
    estimator_standard_error: float,
    confidence_level: float,
) -> tuple[float, float]:
    """Return a two-sided normal-approximation confidence interval."""

    if not math.isfinite(estimate):
        raise StatisticalError("estimate must be finite")
    if not math.isfinite(estimator_standard_error) or estimator_standard_error < 0.0:
        raise StatisticalError("standard error must be finite and non-negative")
    if not 0.0 < confidence_level < 1.0:
        raise StatisticalError("confidence level must lie strictly between zero and one")

    if estimator_standard_error == 0.0:
        return estimate, estimate
    quantile = float(norm.ppf(0.5 + confidence_level / 2.0))
    margin = quantile * estimator_standard_error
    low, high = estimate - margin, estimate + margin
    if not math.isfinite(low) or not math.isfinite(high):
        raise StatisticalError("confidence interval is non-finite")
    return low, high
