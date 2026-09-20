import numpy as np
import pytest
from pydantic import ValidationError

from backend.app.models.regime import MarkovChain
from backend.app.regime.markov import (
    markov_diagnostics,
    n_step_transition,
    probability_forecasts,
    stationary_distribution,
)


def test_transition_validation() -> None:
    for matrix in (
        [[0.5, 0.5]],
        [[0.8, 0.2], [0.1]],
        [[1.1, -0.1], [0.2, 0.8]],
        [[0.8, 0.3], [0.2, 0.8]],
    ):
        with pytest.raises(ValidationError):
            MarkovChain(transition_matrix=matrix)


def test_stationary_duration_and_forecast() -> None:
    chain = MarkovChain(
        transition_matrix=[[0.9, 0.1], [0.2, 0.8]], initial_probabilities=[1.0, 0.0]
    )
    stationary, unique, irreducible, nullity = stationary_distribution(chain)
    np.testing.assert_allclose(stationary, [2.0 / 3.0, 1.0 / 3.0])
    assert unique and irreducible and nullity == 1
    diagnostics = markov_diagnostics(chain)
    assert diagnostics.states[0].expected_duration == pytest.approx(10.0)
    assert diagnostics.states[1].expected_duration == pytest.approx(5.0)
    np.testing.assert_allclose(
        n_step_transition(chain, 3), np.linalg.matrix_power(chain.transition_matrix, 3)
    )
    forecast = probability_forecasts(chain, [0.25, 0.75], [0, 1, 4])
    np.testing.assert_allclose(
        forecast[2].probabilities,
        np.array([0.25, 0.75]) @ np.linalg.matrix_power(chain.transition_matrix, 4),
    )


def test_reducible_and_absorbing_diagnostics() -> None:
    chain = MarkovChain(transition_matrix=[[1.0, 0.0], [0.0, 1.0]])
    diagnostics = markov_diagnostics(chain)
    assert diagnostics.stationary_distribution_unique is False
    assert diagnostics.irreducible is False
    assert diagnostics.stationary_nullity == 2
    assert all(state.absorbing and state.expected_duration is None for state in diagnostics.states)
