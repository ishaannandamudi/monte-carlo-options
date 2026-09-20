"""Loss convention, empirical estimators, covariance, and typed-model tests."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from pydantic import ValidationError

from backend.app.core.exceptions import PortfolioRiskError
from backend.app.models.risk import (
    CovarianceInput,
    HistoricalRiskDataset,
    HistoricalRiskFactorObservation,
    HistoricalRiskScenario,
    PortfolioRiskConfig,
    RiskFactorMapping,
    RiskFactorType,
    RiskMethod,
    RiskValuationMode,
)
from backend.app.risk.statistics import (
    empirical_tail,
    estimate_sample_covariance,
    pnl_to_losses,
)


def test_loss_convention_is_negative_pnl() -> None:
    pnl = np.asarray([-3.0, 0.0, 2.0])
    np.testing.assert_array_equal(pnl_to_losses(pnl), [3.0, 0.0, -2.0])


def test_empirical_var_uses_explicit_linear_quantile() -> None:
    losses = np.asarray([0.0, 1.0, 2.0, 10.0])
    result = empirical_tail(losses, 0.75, quantile_method="linear")
    assert result.value_at_risk == pytest.approx(4.0)


def test_fractional_empirical_es_uses_exact_tail_mass() -> None:
    losses = np.asarray([1.0, 2.0, 3.0, 10.0])
    # Tail mass is 1.2 observations: 10 plus 20% of 3.
    result = empirical_tail(losses, 0.70)
    assert result.expected_shortfall == pytest.approx((10.0 + 0.2 * 3.0) / 1.2)
    np.testing.assert_allclose(result.weights, [1.0, 0.2])


def test_empirical_es_below_one_observation_is_worst_loss() -> None:
    result = empirical_tail(np.asarray([-1.0, 2.0, 7.0]), 0.99)
    assert result.expected_shortfall == 7.0


def test_zero_losses_produce_zero_var_and_es() -> None:
    result = empirical_tail(np.zeros(20), 0.99)
    assert result.value_at_risk == 0.0
    assert result.expected_shortfall == 0.0


def test_historical_dataset_requires_ordered_unique_timestamps() -> None:
    date = datetime(2025, 1, 1, tzinfo=UTC)
    observation = [HistoricalRiskFactorObservation(factor_id="x", value=0.0)]
    with pytest.raises(ValidationError):
        HistoricalRiskDataset(
            scenarios=[
                HistoricalRiskScenario(
                    timestamp=date + timedelta(days=1), observations=observation
                ),
                HistoricalRiskScenario(timestamp=date, observations=observation),
            ]
        )
    with pytest.raises(ValidationError):
        HistoricalRiskDataset(
            scenarios=[
                HistoricalRiskScenario(timestamp=date, observations=observation),
                HistoricalRiskScenario(timestamp=date, observations=observation),
            ]
        )


def test_historical_scenario_rejects_duplicate_factors() -> None:
    with pytest.raises(ValidationError):
        HistoricalRiskScenario(
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            observations=[
                HistoricalRiskFactorObservation(factor_id="x", value=1.0),
                HistoricalRiskFactorObservation(factor_id="x", value=2.0),
            ],
        )


def test_mapping_requires_target_and_custom_exposure() -> None:
    with pytest.raises(ValidationError):
        RiskFactorMapping(factor_id="spot", factor_type=RiskFactorType.SPOT_RETURN)
    with pytest.raises(ValidationError):
        RiskFactorMapping(factor_id="custom", factor_type=RiskFactorType.CUSTOM_GAUSSIAN)


def test_covariance_rejects_materially_indefinite_matrix() -> None:
    with pytest.raises(ValidationError):
        CovarianceInput(factor_ids=["a", "b"], covariance_matrix=((1.0, 2.0), (2.0, 1.0)))


def test_sample_covariance_matches_numpy_reference() -> None:
    values = np.asarray([[0.01, 0.02], [-0.01, -0.03], [0.02, 0.01], [0.0, -0.01]])
    result = estimate_sample_covariance(["a", "b"], values)
    np.testing.assert_allclose(result.covariance, np.cov(values, rowvar=False, ddof=1))
    np.testing.assert_allclose(result.means, np.mean(values, axis=0))


def test_invalid_loss_vectors_fail_clearly() -> None:
    with pytest.raises(PortfolioRiskError):
        empirical_tail(np.asarray([1.0, np.nan]), 0.99)
    with pytest.raises(PortfolioRiskError):
        pnl_to_losses(np.asarray([]))


def test_parametric_config_rejects_non_delta_mode() -> None:
    with pytest.raises(ValidationError):
        PortfolioRiskConfig(
            method=RiskMethod.PARAMETRIC_NORMAL,
            valuation_mode=RiskValuationMode.DELTA_GAMMA_DIAGONAL,
        )
