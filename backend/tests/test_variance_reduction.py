import math

import numpy as np
import pytest

from backend.app.pricing.variance_reduction import (
    apply_control_variate,
    cluster_observations,
    estimator_covariance,
    estimator_variance,
    weighted_estimate,
)


def test_known_plain_covariance_variance_and_beta() -> None:
    raw_values = np.array([1.0, 4.0, 8.0, 9.0])
    control_values = np.array([2.0, 3.0, 6.0, 8.0])
    raw = cluster_observations(raw_values, antithetic=False)
    control = cluster_observations(control_values, antithetic=False)
    expected_variance = float(np.var(control_values, ddof=1) / control_values.size)
    expected_covariance = float(
        np.cov(raw_values, control_values, ddof=1)[0, 1] / control_values.size
    )

    assert estimator_variance(control) == pytest.approx(expected_variance)
    assert estimator_covariance(raw, control) == pytest.approx(expected_covariance)
    adjustment = apply_control_variate(raw, control, control_expected_mean=5.0)
    assert adjustment.beta == pytest.approx(expected_covariance / expected_variance)


def test_known_control_adjustment_and_adjusted_mean() -> None:
    raw = cluster_observations([2.0, 5.0, 10.0, 13.0], antithetic=False)
    control = cluster_observations([1.0, 2.0, 4.0, 5.0], antithetic=False)
    adjustment = apply_control_variate(raw, control, control_expected_mean=3.5)
    expected = raw.values - adjustment.beta * (control.values - 3.5)
    np.testing.assert_allclose(adjustment.adjusted_observations, expected)
    assert adjustment.adjusted_price == pytest.approx(float(np.mean(expected)))
    assert adjustment.adjusted_standard_error**2 == pytest.approx(
        adjustment.adjusted_estimator_variance
    )


def test_zero_control_variance_falls_back_to_raw_estimator() -> None:
    raw = cluster_observations([1.0, 4.0, 2.0, 8.0], antithetic=False)
    control = cluster_observations([3.0, 3.0, 3.0, 3.0], antithetic=False)
    adjustment = apply_control_variate(raw, control, control_expected_mean=3.0)
    assert adjustment.beta == 0.0
    assert adjustment.adjusted_price == adjustment.raw_price
    assert adjustment.adjusted_estimator_variance == adjustment.raw_estimator_variance
    assert adjustment.variance_reduction_ratio == pytest.approx(1.0)
    assert all(
        math.isfinite(value)
        for value in (
            adjustment.beta,
            adjustment.adjusted_price,
            adjustment.adjusted_standard_error,
        )
    )


def test_variance_reduction_ratio_is_raw_over_adjusted() -> None:
    raw = cluster_observations([0.0, 2.0, 5.0, 11.0], antithetic=False)
    control = cluster_observations([1.0, 3.0, 4.0, 8.0], antithetic=False)
    adjustment = apply_control_variate(raw, control, control_expected_mean=4.0)
    assert adjustment.variance_reduction_ratio == pytest.approx(
        adjustment.raw_estimator_variance / adjustment.adjusted_estimator_variance
    )


def test_even_antithetic_clusters_are_pair_averages() -> None:
    clusters = cluster_observations([1.0, 3.0, 5.0, 7.0], antithetic=True)
    np.testing.assert_array_equal(clusters.values, [2.0, 6.0])
    np.testing.assert_array_equal(clusters.sizes, [2.0, 2.0])
    assert weighted_estimate(clusters) == 4.0
    assert estimator_variance(clusters) == pytest.approx(4.0)


def test_odd_antithetic_singleton_retains_one_path_weight() -> None:
    clusters = cluster_observations([1.0, 3.0, 5.0, 7.0, 10.0], antithetic=True)
    np.testing.assert_array_equal(clusters.values, [2.0, 6.0, 10.0])
    np.testing.assert_array_equal(clusters.sizes, [2.0, 2.0, 1.0])
    assert clusters.path_count == 5
    assert weighted_estimate(clusters) == pytest.approx(5.2)


def test_pair_level_beta_differs_from_correlated_path_level_beta() -> None:
    raw_paths = np.array([0.0, 8.0, 3.0, 9.0, 2.0, 14.0])
    control_paths = np.array([1.0, 9.0, 6.0, 8.0, 4.0, 10.0])
    raw_clusters = cluster_observations(raw_paths, antithetic=True)
    control_clusters = cluster_observations(control_paths, antithetic=True)
    adjustment = apply_control_variate(raw_clusters, control_clusters, 6.0)
    path_beta = float(
        np.cov(raw_paths, control_paths, ddof=1)[0, 1] / np.var(control_paths, ddof=1)
    )
    pair_beta = float(
        np.cov(raw_clusters.values, control_clusters.values, ddof=1)[0, 1]
        / np.var(control_clusters.values, ddof=1)
    )
    assert adjustment.beta == pytest.approx(pair_beta)
    assert adjustment.beta != pytest.approx(path_beta)


def test_odd_combined_adjustment_uses_weighted_cluster_scores_and_se() -> None:
    raw_paths = np.array([1.0, 5.0, 2.0, 10.0, 8.0])
    control_paths = np.array([2.0, 6.0, 3.0, 7.0, 9.0])
    raw = cluster_observations(raw_paths, antithetic=True)
    control = cluster_observations(control_paths, antithetic=True)
    adjustment = apply_control_variate(raw, control, control_expected_mean=5.5)

    raw_mean = float(np.dot(raw.sizes, raw.values) / 5.0)
    control_mean = float(np.dot(control.sizes, control.values) / 5.0)
    raw_scores = raw.sizes * (raw.values - raw_mean)
    control_scores = control.sizes * (control.values - control_mean)
    scale = 3.0 / 2.0 / 25.0
    expected_beta = float(
        scale
        * np.dot(raw_scores, control_scores)
        / (scale * np.dot(control_scores, control_scores))
    )
    adjusted = raw.values - expected_beta * (control.values - 5.5)
    adjusted_mean = float(np.dot(raw.sizes, adjusted) / 5.0)
    adjusted_scores = raw.sizes * (adjusted - adjusted_mean)
    expected_variance = scale * float(np.dot(adjusted_scores, adjusted_scores))

    assert raw.path_count == 5
    assert adjustment.beta == pytest.approx(expected_beta)
    assert adjustment.adjusted_price == pytest.approx(adjusted_mean)
    assert adjustment.adjusted_standard_error == pytest.approx(math.sqrt(expected_variance))
