import math
from time import perf_counter

import pytest

from backend.app.calibration.datasets import heston_parameters_to_vector, prepare_calibration_data
from backend.app.calibration.heston import (
    HestonCalibrationObjective,
    calibrate_heston,
    generate_fitted_heston_surface,
)
from backend.app.calibration.synthetic import (
    FELLER_SATISFIED_PARAMETERS,
    generate_synthetic_heston_dataset,
)
from backend.app.models.heston_calibration import (
    HestonCalibrationConfig,
    HestonCalibrationOptimizer,
    HestonCalibrationTarget,
)


@pytest.mark.performance
@pytest.mark.parametrize("strike_count", [10, 50, 200])
def test_objective_evaluation_scaling(strike_count: int) -> None:
    coordinates = [(-0.25 + 0.5 * index / (strike_count - 1)) for index in range(strike_count)]
    dataset = generate_synthetic_heston_dataset(
        maturities=(1.0,), log_moneyness=coordinates, bid_ask_half_spread=None
    )
    config = HestonCalibrationConfig(
        target=HestonCalibrationTarget.PRICE,
        identifiability_diagnostics=False,
    )
    objective = HestonCalibrationObjective(prepare_calibration_data(dataset, config), config)
    started = perf_counter()
    value = objective.scalar(heston_parameters_to_vector(FELLER_SATISFIED_PARAMETERS))
    elapsed = perf_counter() - started
    assert math.isfinite(value)
    assert objective.pricing_evaluations == strike_count
    assert elapsed < 10.0


@pytest.mark.performance
def test_fifty_quote_local_calibration_runtime() -> None:
    dataset = generate_synthetic_heston_dataset(
        maturities=(0.25, 0.5, 1.0, 2.0, 3.0),
        log_moneyness=(-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25),
    )
    started = perf_counter()
    result = calibrate_heston(
        dataset,
        HestonCalibrationConfig(
            target=HestonCalibrationTarget.PRICE,
            initial_parameters=FELLER_SATISFIED_PARAMETERS,
            max_iterations=3,
            identifiability_diagnostics=False,
        ),
    )
    elapsed = perf_counter() - started
    assert result.observations_used == 50
    assert result.rmse < 2e-8
    assert elapsed < 20.0


@pytest.mark.performance
def test_tiny_hybrid_calibration_runtime() -> None:
    dataset = generate_synthetic_heston_dataset(
        maturities=(0.5,),
        log_moneyness=(-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25),
    )
    started = perf_counter()
    result = calibrate_heston(
        dataset,
        HestonCalibrationConfig(
            target=HestonCalibrationTarget.PRICE,
            optimizer=HestonCalibrationOptimizer.HYBRID,
            max_iterations=2,
            differential_evolution_population=2,
            identifiability_diagnostics=False,
        ),
    )
    elapsed = perf_counter() - started
    assert math.isfinite(result.objective_value)
    assert len(result.starts) == 2
    assert elapsed < 20.0


@pytest.mark.performance
def test_two_hundred_point_fitted_surface_runtime() -> None:
    coordinates = [(-0.3 + 0.6 * index / 19) for index in range(20)]
    dataset = generate_synthetic_heston_dataset(
        maturities=tuple(0.25 * (index + 1) for index in range(10)),
        log_moneyness=coordinates,
        bid_ask_half_spread=None,
    )
    started = perf_counter()
    result = generate_fitted_heston_surface(dataset, FELLER_SATISFIED_PARAMETERS)
    elapsed = perf_counter() - started
    assert len(result.points) == 200
    assert result.metrics.rmse < 2e-7
    assert elapsed < 20.0
