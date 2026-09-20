import numpy as np

from backend.app.models.regime import (
    GaussianHMMParameters,
    MarkovSwitchingGBMConfig,
    RegimeRiskRequest,
)
from backend.app.regime.risk import regime_conditioned_risk
from backend.app.regime.synthetic import simulate_markov_switching_gbm


def params() -> GaussianHMMParameters:
    return GaussianHMMParameters(
        num_states=2,
        feature_dimension=1,
        initial_probabilities=[0.5, 0.5],
        transition_matrix=[[0.95, 0.05], [0.1, 0.9]],
        means=[[0.0], [0.0]],
        covariances=[[[0.01]], [[0.09]]],
    )


def test_state_specific_and_mixture_risk_use_loss_convention() -> None:
    request = RegimeRiskRequest(
        parameters=params(),
        state_probabilities=[0.8, 0.2],
        factor_ids=["return"],
        delta_exposures=[100.0],
        confidence_level=0.99,
        num_simulations=100_000,
        seed=7,
    )
    result = regime_conditioned_risk(request)
    assert result.state_risks[1].value_at_risk > result.state_risks[0].value_at_risk
    assert result.mixture_value_at_risk > result.state_risks[0].value_at_risk
    assert result.mixture_value_at_risk != sum(
        item.probability * item.value_at_risk for item in result.state_risks
    )
    assert result.mixture_expected_shortfall >= result.mixture_value_at_risk
    assert regime_conditioned_risk(request) == result


def test_real_world_markov_switching_gbm_is_reproducible() -> None:
    config = MarkovSwitchingGBMConfig(
        initial_spot=100.0,
        drifts=[0.02, -0.01],
        volatilities=[0.1, 0.3],
        transition_matrix=[[0.95, 0.05], [0.1, 0.9]],
        initial_probabilities=[0.5, 0.5],
        num_steps=20,
        num_paths=100,
        time_step=1 / 252,
        seed=12,
    )
    first = simulate_markov_switching_gbm(config)
    second = simulate_markov_switching_gbm(config)
    np.testing.assert_array_equal(first.states, second.states)
    np.testing.assert_array_equal(first.spots, second.spots)
    assert np.all(first.spots > 0.0)
