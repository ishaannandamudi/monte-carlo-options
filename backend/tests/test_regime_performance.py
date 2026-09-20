import pytest

from backend.app.models.regime import GaussianHMMParameters, HMMFitConfig
from backend.app.regime.hmm import fit_gaussian_hmm
from backend.app.regime.synthetic import generate_synthetic_hmm


@pytest.mark.performance
@pytest.mark.parametrize("length,states", [(1_000, 2), (10_000, 2), (10_000, 4)])
def test_hmm_fit_benchmark(length: int, states: int) -> None:
    means = [[float(index * 2)] for index in range(states)]
    transition = [[0.1 / (states - 1)] * states for _ in range(states)]
    for index in range(states):
        transition[index][index] = 0.9
    parameters = GaussianHMMParameters(
        num_states=states,
        feature_dimension=1,
        initial_probabilities=[1 / states] * states,
        transition_matrix=transition,
        means=means,
        covariances=[[[0.5]]] * states,
    )
    dataset = generate_synthetic_hmm(parameters, length, 5).dataset()
    result = fit_gaussian_hmm(
        dataset, HMMFitConfig(num_states=states, num_starts=1, max_iterations=10, seed=5)
    )
    assert result.runtime_ms >= 0.0
