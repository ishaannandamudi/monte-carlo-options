import numpy as np
import pytest

from backend.app.models.regime import (
    CanonicalOrdering,
    CovarianceType,
    GaussianHMMParameters,
    HMMFitConfig,
    HMMModelOrderRequest,
    HMMWalkForwardConfig,
)
from backend.app.regime.hmm import (
    fit_gaussian_hmm,
    hmm_parameter_count,
    model_order_comparison,
    walk_forward_hmm,
)
from backend.app.regime.synthetic import generate_synthetic_hmm, standardize_dataset


def true_parameters(multivariate: bool = False) -> GaussianHMMParameters:
    if multivariate:
        return GaussianHMMParameters(
            num_states=2,
            feature_dimension=2,
            covariance_type=CovarianceType.FULL,
            initial_probabilities=[0.5, 0.5],
            transition_matrix=[[0.96, 0.04], [0.07, 0.93]],
            means=[[-1.5, 0.0], [1.5, 1.0]],
            covariances=[[[0.3, 0.08], [0.08, 0.4]], [[0.4, -0.05], [-0.05, 0.3]]],
        )
    return GaussianHMMParameters(
        num_states=2,
        feature_dimension=1,
        covariance_type=CovarianceType.DIAGONAL,
        initial_probabilities=[0.5, 0.5],
        transition_matrix=[[0.96, 0.04], [0.07, 0.93]],
        means=[[-2.0], [2.0]],
        covariances=[[[0.25]], [[0.36]]],
    )


def test_synthetic_recovery_monotonicity_and_information_criteria() -> None:
    data = generate_synthetic_hmm(true_parameters(), 1_500, 17)
    config = HMMFitConfig(num_starts=3, max_iterations=100, seed=91)
    result = fit_gaussian_hmm(data.dataset(["return"]), config)
    np.testing.assert_allclose(np.ravel(result.parameters.means), [-2.0, 2.0], atol=0.15)
    np.testing.assert_allclose(
        result.parameters.transition_matrix, [[0.96, 0.04], [0.07, 0.93]], atol=0.04
    )
    assert all(
        b >= a - 1e-10
        for a, b in zip(result.likelihood_history, result.likelihood_history[1:], strict=False)
    )
    assert result.parameter_count == 7
    assert result.aic == pytest.approx(2 * 7 - 2 * result.log_likelihood)
    assert result.bic == pytest.approx(7 * np.log(1500) - 2 * result.log_likelihood)
    fitted_path = np.array(
        [
            np.argmin(abs(np.array(result.parameters.means)[:, 0] - value))
            for value in data.observations[:, 0]
        ]
    )
    assert np.mean(fitted_path == data.latent_states) > 0.98


def test_multistart_is_reproducible_and_canonical() -> None:
    dataset = generate_synthetic_hmm(true_parameters(), 500, 8).dataset()
    config = HMMFitConfig(
        num_starts=3,
        seed=22,
        canonical_ordering=CanonicalOrdering.FIRST_FEATURE_MEAN,
    )
    first = fit_gaussian_hmm(dataset, config)
    second = fit_gaussian_hmm(dataset, config)
    assert first.parameters == second.parameters
    assert first.start_summaries == second.start_summaries
    assert first.parameters.means[0][0] < first.parameters.means[1][0]


def test_multivariate_full_covariance_and_standardization() -> None:
    dataset = generate_synthetic_hmm(true_parameters(multivariate=True), 700, 44).dataset(
        ["r", "v"]
    )
    result = fit_gaussian_hmm(
        dataset,
        HMMFitConfig(covariance_type=CovarianceType.FULL, num_starts=2, seed=4, standardize=True),
    )
    assert np.isfinite(result.log_likelihood)
    assert result.parameters.standardization_mean is not None
    assert hmm_parameter_count(2, 2, CovarianceType.FULL) == 13
    transformed, mean, std = standardize_dataset(dataset)
    assert np.allclose(np.mean(transformed.observations, axis=0), 0.0)
    assert len(mean) == len(std) == 2


def test_model_order_is_bounded_and_finite() -> None:
    dataset = generate_synthetic_hmm(true_parameters(), 300, 3).dataset()
    result = model_order_comparison(
        HMMModelOrderRequest(
            dataset=dataset,
            state_counts=[2, 3],
            config=HMMFitConfig(num_starts=1, max_iterations=30, seed=4),
        )
    )
    assert [point.num_states for point in result.points] == [2, 3]
    assert all(np.isfinite(point.bic) for point in result.points)


def test_walk_forward_returns_only_out_of_sample_probabilities() -> None:
    dataset = generate_synthetic_hmm(true_parameters(), 120, 5).dataset()
    result = walk_forward_hmm(
        dataset,
        HMMWalkForwardConfig(
            initial_training_observations=80,
            refit_interval=20,
            fit=HMMFitConfig(num_starts=1, max_iterations=20, seed=2),
        ),
    )
    assert len(result.points) == 40
    assert result.points[0].observation_index == 80
    assert result.refit_count == 2
    assert all(sum(point.filtered_probabilities) == pytest.approx(1.0) for point in result.points)


def test_overlapping_states_and_covariance_floor_remain_finite() -> None:
    overlapping = true_parameters().model_copy(
        update={"means": [[-0.1], [0.1]], "covariances": [[[1.0]], [[1.0]]]}
    )
    dataset = generate_synthetic_hmm(overlapping, 300, 14).dataset()
    result = fit_gaussian_hmm(
        dataset,
        HMMFitConfig(
            initialization="RANDOM",
            num_starts=2,
            max_iterations=30,
            min_variance=0.02,
            seed=8,
        ),
    )
    assert np.isfinite(result.log_likelihood)
    assert all(covariance[0][0] >= 0.02 for covariance in result.parameters.covariances)
