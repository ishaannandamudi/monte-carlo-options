import math

import pytest

from backend.app.calibration.heston import calibrate_heston, evaluate_heston_calibration
from backend.app.calibration.synthetic import generate_synthetic_heston_dataset
from backend.app.core.exceptions import HestonCalibrationError
from backend.app.models.heston import HestonParameters
from backend.app.models.heston_calibration import (
    CalibrationParameterRange,
    CalibrationRobustLoss,
    HestonCalibrationBounds,
    HestonCalibrationConfig,
    HestonCalibrationOptimizer,
    HestonCalibrationTarget,
)


def close_initial() -> HestonParameters:
    return HestonParameters(v0=0.045, kappa=1.8, theta=0.045, sigma_v=0.32, rho=-0.65)


@pytest.mark.parametrize(
    ("target", "tolerance"),
    [
        (HestonCalibrationTarget.PRICE, 2e-6),
        (HestonCalibrationTarget.IMPLIED_VOLATILITY, 2e-6),
        (HestonCalibrationTarget.TOTAL_VARIANCE, 2e-6),
    ],
)
def test_local_calibration_recovers_noise_free_surface(
    target: HestonCalibrationTarget,
    tolerance: float,
) -> None:
    dataset = generate_synthetic_heston_dataset(
        maturities=(0.25, 1.0, 2.0), log_moneyness=(-0.1, 0.0, 0.1)
    )
    result = calibrate_heston(
        dataset,
        HestonCalibrationConfig(
            target=target,
            initial_parameters=close_initial(),
            max_iterations=80,
            identifiability_diagnostics=False,
        ),
    )
    assert result.rmse < tolerance
    assert result.failed_evaluations == 0
    assert result.observations_used == 9
    assert len(result.residuals) == 9
    assert len(result.per_expiry) == 3
    assert math.isfinite(result.prices_per_second)
    if target is HestonCalibrationTarget.PRICE:
        assert result.calibrated_parameters.v0 == pytest.approx(0.04, abs=1e-7)
        assert result.calibrated_parameters.rho == pytest.approx(-0.7, abs=1e-6)
        assert result.cost >= 0.0
        assert result.optimality is not None


def test_same_seed_and_configuration_are_reproducible() -> None:
    dataset = generate_synthetic_heston_dataset(
        maturities=(0.5, 1.0), log_moneyness=(-0.1, 0.0, 0.1)
    )
    config = HestonCalibrationConfig(
        target=HestonCalibrationTarget.PRICE,
        initial_parameters=close_initial(),
        multi_start_count=2,
        random_seed=912,
        max_iterations=20,
        identifiability_diagnostics=False,
    )
    first = calibrate_heston(dataset, config)
    second = calibrate_heston(dataset, config)
    assert first.calibrated_parameters == second.calibrated_parameters
    assert first.objective_value == second.objective_value
    assert first.pricing_evaluations == second.pricing_evaluations
    winning_start = first.starts[first.best_start_index]
    assert first.initial_parameters == winning_start.initial_parameters


@pytest.mark.parametrize(
    "loss",
    [
        CalibrationRobustLoss.LINEAR,
        CalibrationRobustLoss.SOFT_L1,
        CalibrationRobustLoss.HUBER,
    ],
)
def test_noisy_iv_calibration_is_finite_and_bounded(loss: CalibrationRobustLoss) -> None:
    dataset = generate_synthetic_heston_dataset(
        maturities=(0.5, 1.0),
        log_moneyness=(-0.1, 0.0, 0.1),
        implied_volatility_noise_std=0.002,
        seed=11,
    )
    config = HestonCalibrationConfig(
        target=HestonCalibrationTarget.IMPLIED_VOLATILITY,
        initial_parameters=close_initial(),
        robust_loss=loss,
        max_iterations=12,
        identifiability_diagnostics=False,
    )
    result = calibrate_heston(dataset, config)
    assert math.isfinite(result.objective_value)
    assert math.isfinite(result.rmse)
    assert -0.999 <= result.calibrated_parameters.rho <= 0.999
    assert result.calibrated_parameters.sigma_v >= 0.0


def test_differential_evolution_is_deterministic_and_hybrid_refines() -> None:
    dataset = generate_synthetic_heston_dataset(maturities=(0.5,), log_moneyness=(-0.1, 0.0, 0.1))
    global_config = HestonCalibrationConfig(
        target=HestonCalibrationTarget.PRICE,
        optimizer=HestonCalibrationOptimizer.DIFFERENTIAL_EVOLUTION,
        max_iterations=1,
        differential_evolution_population=2,
        local_refinement=False,
        random_seed=7,
        identifiability_diagnostics=False,
    )
    first = calibrate_heston(dataset, global_config)
    second = calibrate_heston(dataset, global_config)
    assert first.calibrated_parameters == second.calibrated_parameters
    assert first.objective_value == second.objective_value
    assert first.starts[0].stage.value == "GLOBAL"
    assert 1e-6 <= first.calibrated_parameters.v0 <= 2.0
    assert 1e-4 <= first.calibrated_parameters.kappa <= 20.0
    assert 1e-6 <= first.calibrated_parameters.theta <= 2.0
    assert 1e-4 <= first.calibrated_parameters.sigma_v <= 5.0
    assert -0.999 <= first.calibrated_parameters.rho <= 0.999

    hybrid = calibrate_heston(
        dataset,
        global_config.model_copy(
            update={
                "optimizer": HestonCalibrationOptimizer.HYBRID,
                "max_iterations": 8,
                "initial_parameters": close_initial(),
            }
        ),
    )
    global_candidate = next(item for item in hybrid.starts if item.stage.value == "GLOBAL")
    assert hybrid.objective_value <= global_candidate.objective_value + 1e-15
    assert any(item.stage.value == "GLOBAL_REFINEMENT" for item in hybrid.starts)


def test_bad_but_bounded_local_initial_guess_stays_finite_and_bounded() -> None:
    dataset = generate_synthetic_heston_dataset(
        maturities=(0.5, 1.0), log_moneyness=(-0.1, 0.0, 0.1)
    )
    bad_initial = HestonParameters(
        v0=0.5,
        kappa=8.0,
        theta=0.4,
        sigma_v=1.5,
        rho=0.5,
    )
    result = calibrate_heston(
        dataset,
        HestonCalibrationConfig(
            target=HestonCalibrationTarget.PRICE,
            initial_parameters=bad_initial,
            max_iterations=8,
            identifiability_diagnostics=False,
        ),
    )
    assert math.isfinite(result.objective_value)
    assert 1e-6 <= result.calibrated_parameters.v0 <= 2.0
    assert 1e-4 <= result.calibrated_parameters.kappa <= 20.0
    assert 1e-6 <= result.calibrated_parameters.theta <= 2.0
    assert 1e-4 <= result.calibrated_parameters.sigma_v <= 5.0
    assert -0.999 <= result.calibrated_parameters.rho <= 0.999


def test_out_of_bounds_initial_and_diagnostic_parameters_fail() -> None:
    dataset = generate_synthetic_heston_dataset(maturities=(1.0,), log_moneyness=(0.0,))
    outside = HestonParameters(v0=3.0, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7)
    config = HestonCalibrationConfig(initial_parameters=outside)
    with pytest.raises(HestonCalibrationError, match="initial"):
        calibrate_heston(dataset, config)
    with pytest.raises(HestonCalibrationError, match="diagnostic"):
        evaluate_heston_calibration(dataset, outside, HestonCalibrationConfig())


def test_warm_start_on_perturbed_surface_is_accepted_and_reproducible() -> None:
    clean = generate_synthetic_heston_dataset(maturities=(0.5, 1.0), log_moneyness=(-0.1, 0.0, 0.1))
    initial = calibrate_heston(
        clean,
        HestonCalibrationConfig(
            target=HestonCalibrationTarget.PRICE,
            initial_parameters=close_initial(),
            max_iterations=30,
            identifiability_diagnostics=False,
        ),
    )
    perturbed = generate_synthetic_heston_dataset(
        maturities=(0.5, 1.0),
        log_moneyness=(-0.1, 0.0, 0.1),
        implied_volatility_noise_std=0.001,
        seed=3,
    )
    config = HestonCalibrationConfig(
        target=HestonCalibrationTarget.PRICE,
        initial_parameters=initial.calibrated_parameters,
        max_iterations=10,
        random_seed=55,
        identifiability_diagnostics=False,
    )
    first = calibrate_heston(perturbed, config)
    second = calibrate_heston(perturbed, config)
    assert first.initial_parameters == initial.calibrated_parameters
    assert first.calibrated_parameters == second.calibrated_parameters


def test_bound_constrained_fit_reports_active_limits() -> None:
    dataset = generate_synthetic_heston_dataset(
        maturities=(0.25, 1.0, 2.0), log_moneyness=(-0.1, 0.0, 0.1)
    )
    bounds = HestonCalibrationBounds(
        v0=CalibrationParameterRange(lower=0.039, upper=0.041),
        kappa=CalibrationParameterRange(lower=1.9, upper=2.1),
        theta=CalibrationParameterRange(lower=0.039, upper=0.041),
        sigma_v=CalibrationParameterRange(lower=0.29, upper=0.31),
        rho=CalibrationParameterRange(lower=-0.5, upper=0.2),
    )
    result = calibrate_heston(
        dataset,
        HestonCalibrationConfig(
            target=HestonCalibrationTarget.PRICE,
            bounds=bounds,
            initial_parameters=HestonParameters(
                v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.49
            ),
            max_iterations=30,
            identifiability_diagnostics=False,
        ),
    )
    assert result.calibrated_parameters.rho >= bounds.rho.lower
    assert "rho:LOWER" in result.active_bounds
