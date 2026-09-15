"""Reusable estimator-cluster and control-variate transformations."""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from backend.app.core.exceptions import NumericalError, StatisticalError
from backend.app.core.numerics import finite_exp, require_finite
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract


@dataclass(frozen=True)
class ClusteredObservations:
    """Independent cluster means and their numbers of represented paths."""

    values: NDArray[np.float64]
    sizes: NDArray[np.float64]

    @property
    def path_count(self) -> int:
        return int(np.sum(self.sizes))


@dataclass(frozen=True)
class ControlVariateAdjustment:
    """Internal control-variate transformation and complete diagnostics."""

    adjusted_observations: NDArray[np.float64]
    raw_price: float
    adjusted_price: float
    raw_standard_error: float
    adjusted_standard_error: float
    raw_estimator_variance: float
    adjusted_estimator_variance: float
    beta: float
    control_mean: float
    control_expected_mean: float
    variance_reduction_ratio: float | None


def cluster_observations(values: ArrayLike, *, antithetic: bool) -> ClusteredObservations:
    """Convert path values into independent path or antithetic-pair clusters."""

    path_values = np.asarray(values, dtype=np.float64)
    if path_values.ndim != 1 or path_values.size == 0:
        raise StatisticalError("path observations must be a non-empty one-dimensional array")
    require_finite(path_values, context="path observations")

    if not antithetic:
        return ClusteredObservations(
            values=path_values.copy(),
            sizes=np.ones(path_values.size, dtype=np.float64),
        )

    pair_count, remainder = divmod(path_values.size, 2)
    pair_means = path_values[: 2 * pair_count].reshape(pair_count, 2).mean(axis=1)
    sizes = np.full(pair_count, 2.0, dtype=np.float64)
    if remainder:
        odd_values = np.concatenate((pair_means, path_values[-1:]))
        odd_sizes = np.concatenate((sizes, np.ones(1, dtype=np.float64)))
        return ClusteredObservations(values=odd_values, sizes=odd_sizes)
    return ClusteredObservations(values=pair_means, sizes=sizes)


def weighted_estimate(observations: ClusteredObservations) -> float:
    """Return the path-weighted mean represented by independent clusters."""

    estimate = float(np.dot(observations.sizes, observations.values) / observations.path_count)
    if not math.isfinite(estimate):
        raise NumericalError("cluster-weighted estimate is non-finite")
    return estimate


def estimator_variance(observations: ClusteredObservations) -> float:
    """Estimate variance of the path-weighted mean using independent clusters."""

    cluster_count = observations.values.size
    if cluster_count < 2:
        raise StatisticalError("at least two independent clusters are required")
    estimate = weighted_estimate(observations)
    scores = observations.sizes * (observations.values - estimate)
    variance = (
        cluster_count
        / (cluster_count - 1)
        * float(np.dot(scores, scores))
        / observations.path_count**2
    )
    if not math.isfinite(variance) or variance < 0.0:
        raise NumericalError("cluster estimator variance is invalid")
    return variance


def estimator_covariance(
    left: ClusteredObservations,
    right: ClusteredObservations,
) -> float:
    """Estimate covariance of two path-weighted means over identical clusters."""

    if not np.array_equal(left.sizes, right.sizes):
        raise StatisticalError("covariance observations must use identical cluster sizes")
    cluster_count = left.values.size
    if cluster_count < 2:
        raise StatisticalError("at least two independent clusters are required")
    left_scores = left.sizes * (left.values - weighted_estimate(left))
    right_scores = right.sizes * (right.values - weighted_estimate(right))
    covariance = (
        cluster_count
        / (cluster_count - 1)
        * float(np.dot(left_scores, right_scores))
        / left.path_count**2
    )
    if not math.isfinite(covariance):
        raise NumericalError("cluster estimator covariance is non-finite")
    return covariance


def discounted_stock_control(
    terminal_prices: NDArray[np.float64],
    contract: OptionContract,
    market: MarketEnvironment,
) -> tuple[NDArray[np.float64], float]:
    """Return exp(-rT)S_T and its known expectation S_0 exp(-qT)."""

    discount = finite_exp(
        -market.risk_free_rate * contract.maturity_years,
        context="control discount factor",
    )
    control = np.asarray(discount * terminal_prices, dtype=np.float64)
    require_finite(control, context="discounted terminal stock control")
    expected_discount = finite_exp(
        -market.dividend_yield * contract.maturity_years,
        context="expected control dividend discount factor",
    )
    expected = contract.spot * expected_discount
    if not math.isfinite(expected) or expected <= 0.0:
        raise NumericalError("expected discounted terminal stock is outside positive finite range")
    return control, expected


def apply_control_variate(
    raw: ClusteredObservations,
    control: ClusteredObservations,
    control_expected_mean: float,
) -> ControlVariateAdjustment:
    """Estimate beta and adjust independent cluster observations."""

    if not np.array_equal(raw.sizes, control.sizes):
        raise StatisticalError("raw and control observations must use identical clusters")
    if not math.isfinite(control_expected_mean):
        raise NumericalError("control expected mean must be finite")

    raw_variance = estimator_variance(raw)
    control_variance = estimator_variance(control)
    covariance = estimator_covariance(raw, control)
    beta = 0.0 if control_variance == 0.0 else covariance / control_variance
    if not math.isfinite(beta):
        raise NumericalError("control-variate beta is non-finite")

    adjusted_values = np.asarray(
        raw.values - beta * (control.values - control_expected_mean),
        dtype=np.float64,
    )
    require_finite(adjusted_values, context="control-adjusted observations")
    adjusted = ClusteredObservations(values=adjusted_values, sizes=raw.sizes)
    adjusted_variance = estimator_variance(adjusted)
    raw_price = weighted_estimate(raw)
    adjusted_price = weighted_estimate(adjusted)
    ratio = raw_variance / adjusted_variance if adjusted_variance > 0.0 else None
    if ratio is not None and not math.isfinite(ratio):
        raise NumericalError("variance-reduction ratio is non-finite")

    return ControlVariateAdjustment(
        adjusted_observations=adjusted_values,
        raw_price=raw_price,
        adjusted_price=adjusted_price,
        raw_standard_error=math.sqrt(raw_variance),
        adjusted_standard_error=math.sqrt(adjusted_variance),
        raw_estimator_variance=raw_variance,
        adjusted_estimator_variance=adjusted_variance,
        beta=beta,
        control_mean=weighted_estimate(control),
        control_expected_mean=control_expected_mean,
        variance_reduction_ratio=ratio,
    )
