"""Heston calibration stability and structured out-of-sample validation."""

import math

import numpy as np

from backend.app.calibration.datasets import (
    clone_dataset_with_observations,
    prepare_calibration_data,
)
from backend.app.calibration.heston import (
    calibrate_heston,
    evaluate_heston_calibration,
    heuristic_initial_parameters,
    random_initial_parameters,
)
from backend.app.core.exceptions import HestonCalibrationError
from backend.app.models.heston_calibration import (
    CalibrationHoldoutPolicy,
    HestonCalibrationConfig,
    HestonCalibrationDataset,
    HestonCalibrationHoldoutConfig,
    HestonCalibrationOptimizer,
    HestonCalibrationStabilityResult,
    HestonCalibrationStabilityRun,
    HestonCalibrationValidationResult,
)
from backend.app.simulation.random import spawn_child_seeds


def analyze_heston_calibration_stability(
    dataset: HestonCalibrationDataset,
    config: HestonCalibrationConfig,
    run_count: int,
) -> HestonCalibrationStabilityResult:
    """Repeat deterministic local calibrations from distinct bounded starting points."""

    if run_count <= 0:
        raise HestonCalibrationError("calibration stability run_count must be positive")
    heuristic = heuristic_initial_parameters(prepare_calibration_data(dataset, config), config)
    seeds = spawn_child_seeds(config.random_seed, run_count)
    runs = []
    for index, seed in enumerate(seeds):
        if index == 0:
            initial = config.initial_parameters or heuristic
        else:
            initial = random_initial_parameters(seed, config)
        run_config = config.model_copy(
            update={
                "random_seed": seed,
                "initial_parameters": initial,
                "multi_start_count": 1,
                "optimizer": HestonCalibrationOptimizer.LEAST_SQUARES,
            }
        )
        result = calibrate_heston(dataset, run_config)
        runs.append(
            HestonCalibrationStabilityRun(
                run_index=index,
                seed=seed,
                parameters=result.calibrated_parameters,
                objective_value=result.objective_value,
                converged=result.converged,
                evaluations=result.evaluations,
                runtime_ms=result.runtime_ms,
            )
        )
    objectives = np.asarray([run.objective_value for run in runs])
    vectors = np.asarray(
        [
            [
                run.parameters.v0,
                run.parameters.kappa,
                run.parameters.theta,
                run.parameters.sigma_v,
                run.parameters.rho,
            ]
            for run in runs
        ]
    )
    names = ("v0", "kappa", "theta", "sigma_v", "rho")
    return HestonCalibrationStabilityResult(
        runs=runs,
        best_run_index=int(np.argmin(objectives)),
        worst_run_index=int(np.argmax(objectives)),
        objective_mean=float(np.mean(objectives)),
        objective_std=float(np.std(objectives, ddof=0)),
        parameter_means={
            name: float(value) for name, value in zip(names, np.mean(vectors, axis=0), strict=True)
        },
        parameter_standard_deviations={
            name: float(value) for name, value in zip(names, np.std(vectors, axis=0), strict=True)
        },
    )


def split_heston_calibration_holdout(
    dataset: HestonCalibrationDataset,
    config: HestonCalibrationConfig,
    holdout: HestonCalibrationHoldoutConfig,
) -> tuple[HestonCalibrationDataset, HestonCalibrationDataset]:
    prepared = prepare_calibration_data(dataset, config)
    observations = list(prepared.observations)
    held_indices: set[int] = set()
    if holdout.policy is CalibrationHoldoutPolicy.EXPIRY:
        for index, item in enumerate(observations):
            if any(
                math.isclose(item.observation.maturity_years, maturity, abs_tol=1e-12)
                for maturity in holdout.maturities
            ):
                held_indices.add(index)
    elif holdout.policy is CalibrationHoldoutPolicy.WINGS:
        held_indices = {
            index
            for index, item in enumerate(observations)
            if abs(item.log_moneyness) >= holdout.wing_log_moneyness
        }
    else:
        by_maturity: dict[float, list[tuple[int, float]]] = {}
        for index, item in enumerate(observations):
            by_maturity.setdefault(item.observation.maturity_years, []).append(
                (index, item.observation.strike)
            )
        for values in by_maturity.values():
            ordered = sorted(values, key=lambda value: value[1])
            for position, (index, _) in enumerate(ordered, start=1):
                if position % holdout.every_nth_strike == 0:
                    held_indices.add(index)
    training = [
        item.observation for index, item in enumerate(observations) if index not in held_indices
    ]
    validation = [
        item.observation for index, item in enumerate(observations) if index in held_indices
    ]
    if not training or not validation:
        raise HestonCalibrationError(
            "structured holdout must leave nonempty calibration and validation sets"
        )
    return (
        clone_dataset_with_observations(dataset, training),
        clone_dataset_with_observations(dataset, validation),
    )


def validate_heston_calibration(
    dataset: HestonCalibrationDataset,
    config: HestonCalibrationConfig,
    holdout: HestonCalibrationHoldoutConfig,
) -> HestonCalibrationValidationResult:
    """Calibrate on a structured subset and evaluate the untouched holdout."""

    training, validation = split_heston_calibration_holdout(dataset, config, holdout)
    calibration = calibrate_heston(training, config)
    validation_result = evaluate_heston_calibration(
        validation,
        calibration.calibrated_parameters,
        config.model_copy(update={"identifiability_diagnostics": False}),
    )
    return HestonCalibrationValidationResult(
        calibration=calibration,
        policy=holdout.policy,
        training_observations=calibration.observations_used,
        validation_observations=validation_result.observations_used,
        validation_metrics=validation_result.metrics,
        validation_residuals=validation_result.residuals,
    )
