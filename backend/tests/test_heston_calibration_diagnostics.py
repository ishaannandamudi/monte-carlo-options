import math

import pytest

from backend.app.analytics.heston_calibration import (
    analyze_heston_calibration_stability,
    validate_heston_calibration,
)
from backend.app.calibration.heston import (
    evaluate_heston_calibration,
    generate_fitted_heston_surface,
)
from backend.app.calibration.synthetic import (
    FELLER_SATISFIED_PARAMETERS,
    FELLER_VIOLATING_PARAMETERS,
    generate_synthetic_heston_dataset,
)
from backend.app.core.exceptions import HestonCalibrationError
from backend.app.models.heston_calibration import (
    CalibrationHoldoutPolicy,
    HestonCalibrationConfig,
    HestonCalibrationHoldoutConfig,
    HestonCalibrationTarget,
)


def diagnostic_dataset():  # type: ignore[no-untyped-def]
    return generate_synthetic_heston_dataset(
        maturities=(0.25, 1.0, 2.0),
        log_moneyness=(-0.15, 0.0, 0.15),
    )


def test_exact_fit_diagnostics_are_grouped_and_locally_full_rank() -> None:
    result = evaluate_heston_calibration(
        diagnostic_dataset(),
        FELLER_SATISFIED_PARAMETERS,
        HestonCalibrationConfig(target=HestonCalibrationTarget.PRICE),
    )
    assert result.metrics.rmse < 2e-9
    assert len(result.per_expiry) == 3
    assert sum(point.objective_contribution for point in result.per_expiry) == pytest.approx(
        result.metrics.objective_contribution
    )
    assert result.identifiability is not None
    assert result.identifiability.jacobian_rank == 5
    assert result.identifiability.locally_full_rank is True
    assert result.identifiability.condition_number is not None
    assert result.identifiability.covariance_available is True


def test_sparse_dataset_reports_weak_identification_without_error() -> None:
    dataset = generate_synthetic_heston_dataset(maturities=(1.0,), log_moneyness=(-0.1, 0.1))
    result = evaluate_heston_calibration(
        dataset,
        FELLER_SATISFIED_PARAMETERS,
        HestonCalibrationConfig(target=HestonCalibrationTarget.PRICE),
    )
    assert result.identifiability is not None
    assert result.identifiability.jacobian_rank <= 2
    assert result.identifiability.locally_full_rank is False
    assert result.identifiability.covariance_available is False
    assert "weakly identified" in result.identifiability.note


def test_feller_violating_synthetic_dataset_fits_without_hard_rejection() -> None:
    dataset = generate_synthetic_heston_dataset(
        parameters=FELLER_VIOLATING_PARAMETERS,
        maturities=(0.5, 1.0),
        log_moneyness=(-0.1, 0.0, 0.1),
    )
    result = evaluate_heston_calibration(
        dataset,
        FELLER_VIOLATING_PARAMETERS,
        HestonCalibrationConfig(
            target=HestonCalibrationTarget.PRICE,
            identifiability_diagnostics=False,
        ),
    )
    assert result.metrics.rmse < 2e-8
    assert result.penalty_value == 0.0


def test_fitted_surface_reports_sorted_points_atm_term_and_skew() -> None:
    result = generate_fitted_heston_surface(
        diagnostic_dataset(),
        FELLER_SATISFIED_PARAMETERS,
        HestonCalibrationConfig(target=HestonCalibrationTarget.IMPLIED_VOLATILITY),
    )
    coordinates = [(point.maturity_years, point.strike) for point in result.points]
    assert coordinates == sorted(coordinates)
    assert len(result.points) == 9
    assert len(result.atm_term_structure) == 3
    assert len(result.skew_by_expiry) == 3
    assert all(point.observation_count == 3 for point in result.skew_by_expiry)
    assert result.metrics.average_iv_error == pytest.approx(0.0, abs=2e-7)


def test_fitted_surface_rejects_parameters_outside_selected_bounds() -> None:
    outside = FELLER_SATISFIED_PARAMETERS.model_copy(update={"v0": 3.0})
    with pytest.raises(HestonCalibrationError, match="fitted-surface"):
        generate_fitted_heston_surface(diagnostic_dataset(), outside, HestonCalibrationConfig())


@pytest.mark.parametrize(
    ("holdout", "expected_validation"),
    [
        (
            HestonCalibrationHoldoutConfig(
                policy=CalibrationHoldoutPolicy.EXPIRY,
                maturities=[2.0],
            ),
            3,
        ),
        (
            HestonCalibrationHoldoutConfig(
                policy=CalibrationHoldoutPolicy.EVERY_NTH_STRIKE,
                every_nth_strike=3,
            ),
            3,
        ),
        (
            HestonCalibrationHoldoutConfig(
                policy=CalibrationHoldoutPolicy.WINGS,
                wing_log_moneyness=0.14,
            ),
            6,
        ),
    ],
)
def test_structured_holdout_validation(
    holdout: HestonCalibrationHoldoutConfig,
    expected_validation: int,
) -> None:
    result = validate_heston_calibration(
        diagnostic_dataset(),
        HestonCalibrationConfig(
            target=HestonCalibrationTarget.PRICE,
            initial_parameters=FELLER_SATISFIED_PARAMETERS,
            max_iterations=3,
            identifiability_diagnostics=False,
        ),
        holdout,
    )
    assert result.validation_observations == expected_validation
    assert result.training_observations + result.validation_observations == 9
    assert result.validation_metrics.rmse < 2e-9
    assert len(result.validation_residuals) == expected_validation


def test_invalid_holdout_split_fails_clearly() -> None:
    dataset = generate_synthetic_heston_dataset(maturities=(1.0,), log_moneyness=(0.0,))
    with pytest.raises(HestonCalibrationError, match="nonempty"):
        validate_heston_calibration(
            dataset,
            HestonCalibrationConfig(
                initial_parameters=FELLER_SATISFIED_PARAMETERS,
                identifiability_diagnostics=False,
            ),
            HestonCalibrationHoldoutConfig(
                policy=CalibrationHoldoutPolicy.EXPIRY,
                maturities=[1.0],
            ),
        )


def test_stability_runs_use_reproducible_distinct_seeds() -> None:
    dataset = generate_synthetic_heston_dataset(maturities=(0.5,), log_moneyness=(-0.1, 0.0, 0.1))
    config = HestonCalibrationConfig(
        target=HestonCalibrationTarget.PRICE,
        initial_parameters=FELLER_SATISFIED_PARAMETERS,
        max_iterations=2,
        random_seed=22,
        identifiability_diagnostics=False,
    )
    first = analyze_heston_calibration_stability(dataset, config, 2)
    second = analyze_heston_calibration_stability(dataset, config, 2)
    assert [run.seed for run in first.runs] == [run.seed for run in second.runs]
    assert len(set(run.seed for run in first.runs)) == 2
    assert first.runs[0].objective_value < 1e-16
    assert first.runs[0].parameters.v0 == pytest.approx(FELLER_SATISFIED_PARAMETERS.v0)
    assert first.runs[0].converged is True
    assert first.objective_mean == second.objective_mean
    assert all(math.isfinite(value) for value in first.parameter_means.values())
