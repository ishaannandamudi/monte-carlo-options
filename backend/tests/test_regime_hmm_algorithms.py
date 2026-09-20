import itertools

import numpy as np
import pytest
from scipy.stats import norm

from backend.app.models.regime import (
    CovarianceType,
    GaussianHMMParameters,
    HMMObservationDataset,
)
from backend.app.regime.hmm import (
    filter_step,
    forward_log,
    infer_hmm,
    viterbi_decode,
)


def parameters() -> GaussianHMMParameters:
    return GaussianHMMParameters(
        num_states=2,
        feature_dimension=1,
        covariance_type=CovarianceType.DIAGONAL,
        initial_probabilities=[0.6, 0.4],
        transition_matrix=[[0.75, 0.25], [0.2, 0.8]],
        means=[[-1.0], [1.2]],
        covariances=[[[0.49]], [[0.81]]],
    )


def brute_force(values: np.ndarray) -> tuple[float, np.ndarray, tuple[int, ...]]:
    params = parameters()
    weights: list[float] = []
    paths = list(itertools.product(range(2), repeat=values.size))
    for path in paths:
        weight = params.initial_probabilities[path[0]] * norm.pdf(
            values[0], params.means[path[0]][0], np.sqrt(params.covariances[path[0]][0][0])
        )
        for index in range(1, values.size):
            weight *= params.transition_matrix[path[index - 1]][path[index]]
            weight *= norm.pdf(
                values[index],
                params.means[path[index]][0],
                np.sqrt(params.covariances[path[index]][0][0]),
            )
        weights.append(float(weight))
    total = sum(weights)
    smoothed = np.zeros((values.size, 2))
    for path, weight in zip(paths, weights, strict=True):
        for index, state in enumerate(path):
            smoothed[index, state] += weight / total
    return total, smoothed, paths[int(np.argmax(weights))]


def test_forward_backward_and_viterbi_match_enumeration() -> None:
    values = np.array([[-0.8], [0.2], [1.0]])
    likelihood, expected_smoothed, expected_path = brute_force(values[:, 0])
    inference = infer_hmm(values, parameters())
    assert np.exp(inference.log_likelihood) == pytest.approx(likelihood)
    np.testing.assert_allclose(inference.smoothed, expected_smoothed, atol=1e-12)
    np.testing.assert_allclose(inference.filtered.sum(axis=1), 1.0, atol=1e-14)
    np.testing.assert_allclose(inference.smoothed.sum(axis=1), 1.0, atol=1e-14)
    path, _ = viterbi_decode(values, parameters())
    assert tuple(path) == expected_path


def test_filtering_has_no_look_ahead_and_filter_step_matches_batch() -> None:
    first = np.array([[-1.0], [-0.5], [0.2], [1.0]])
    changed = first.copy()
    changed[2:] = [[-9.0], [9.0]]
    _, filtered_a, _ = forward_log(first, parameters())
    _, filtered_b, _ = forward_log(changed, parameters())
    np.testing.assert_allclose(filtered_a[:2], filtered_b[:2], atol=0.0)
    assert not np.allclose(
        infer_hmm(first, parameters()).smoothed[:2], infer_hmm(changed, parameters()).smoothed[:2]
    )
    update = filter_step(parameters(), filtered_a[1].tolist(), [first[2, 0]])
    np.testing.assert_allclose(update.filtered_probabilities, filtered_a[2], atol=1e-14)


def test_dataset_rejects_missing_values() -> None:
    with pytest.raises(ValueError):
        HMMObservationDataset(observations=[[0.0], [float("nan")]])
