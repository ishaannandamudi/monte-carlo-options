import math

import numpy as np
import pytest

from backend.app.calibration.datasets import heston_parameters_to_vector, prepare_calibration_data
from backend.app.calibration.heston import HestonCalibrationObjective
from backend.app.calibration.synthetic import (
    FELLER_SATISFIED_PARAMETERS,
    FELLER_VIOLATING_PARAMETERS,
    generate_synthetic_heston_dataset,
)
from backend.app.core.exceptions import HestonPricingError
from backend.app.models.heston_calibration import HestonCalibrationConfig, HestonCalibrationTarget
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.pricing.heston_analytical import HestonAnalyticalBatchPricer


@pytest.mark.parametrize("target", list(HestonCalibrationTarget))
def test_true_synthetic_parameters_have_near_zero_objective(
    target: HestonCalibrationTarget,
) -> None:
    dataset = generate_synthetic_heston_dataset(
        maturities=(0.5, 1.0), log_moneyness=(-0.1, 0.0, 0.1)
    )
    config = HestonCalibrationConfig(target=target)
    objective = HestonCalibrationObjective(prepare_calibration_data(dataset, config), config)
    residuals = objective.residual_vector(heston_parameters_to_vector(FELLER_SATISFIED_PARAMETERS))
    tolerance = 2e-7 if target is not HestonCalibrationTarget.PRICE else 2e-9
    assert np.max(np.abs(residuals)) < tolerance
    assert objective.scalar(heston_parameters_to_vector(FELLER_SATISFIED_PARAMETERS)) < tolerance**2


def test_objective_cache_avoids_duplicate_pricing() -> None:
    dataset = generate_synthetic_heston_dataset(maturities=(1.0,), log_moneyness=(-0.1, 0.1))
    config = HestonCalibrationConfig(target=HestonCalibrationTarget.PRICE)
    objective = HestonCalibrationObjective(prepare_calibration_data(dataset, config), config)
    vector = heston_parameters_to_vector(FELLER_SATISFIED_PARAMETERS)
    first = objective.evaluate(vector)
    priced = objective.pricing_evaluations
    second = objective.evaluate(vector.copy())
    assert first is second
    assert objective.pricing_evaluations == priced
    assert objective.cache_hits == 1


def test_maturity_batch_pricer_reuses_only_its_scoped_characteristic_cache() -> None:
    contract = OptionContract(
        option_type=OptionType.CALL,
        spot=100.0,
        strike=100.0,
        maturity_years=1.0,
    )
    pricer = HestonAnalyticalBatchPricer(
        spot=100.0,
        maturity_years=1.0,
        market=MarketEnvironment(risk_free_rate=0.02, dividend_yield=0.01, volatility=0.0),
        parameters=FELLER_SATISFIED_PARAMETERS,
    )
    first = pricer.price(contract)
    misses = pricer.cache_misses
    second = pricer.price(contract)
    assert second.price == first.price
    assert pricer.cache_misses == misses
    assert pricer.cache_hits > 0
    with pytest.raises(HestonPricingError, match="fixed spot and maturity"):
        pricer.price(contract.model_copy(update={"maturity_years": 2.0}))


def test_feller_violation_is_softly_penalized_without_rejection() -> None:
    dataset = generate_synthetic_heston_dataset(maturities=(1.0,), log_moneyness=(0.0,))
    config = HestonCalibrationConfig(
        target=HestonCalibrationTarget.PRICE,
        feller_penalty_weight=10.0,
    )
    objective = HestonCalibrationObjective(prepare_calibration_data(dataset, config), config)
    penalty = objective.feller_penalty(FELLER_VIOLATING_PARAMETERS)
    violation = max(
        FELLER_VIOLATING_PARAMETERS.sigma_v**2
        - 2.0 * FELLER_VIOLATING_PARAMETERS.kappa * FELLER_VIOLATING_PARAMETERS.theta,
        0.0,
    )
    assert penalty == pytest.approx(10.0 * violation**2)
    vector = heston_parameters_to_vector(FELLER_VIOLATING_PARAMETERS)
    penalized_value = objective.scalar(vector)
    unpenalized_config = config.model_copy(update={"feller_penalty_weight": 0.0})
    unpenalized = HestonCalibrationObjective(
        prepare_calibration_data(dataset, unpenalized_config), unpenalized_config
    )
    assert penalized_value - unpenalized.scalar(vector) == pytest.approx(penalty)
    assert math.isfinite(penalized_value)


def test_penalty_residual_dimension_is_constant_across_constraint_boundary() -> None:
    dataset = generate_synthetic_heston_dataset(maturities=(1.0,), log_moneyness=(0.0,))
    config = HestonCalibrationConfig(
        target=HestonCalibrationTarget.PRICE,
        feller_penalty_weight=5.0,
        regularization_weight=2.0,
        regularization_parameters=FELLER_SATISFIED_PARAMETERS,
    )
    objective = HestonCalibrationObjective(prepare_calibration_data(dataset, config), config)
    satisfied = objective.residual_vector(heston_parameters_to_vector(FELLER_SATISFIED_PARAMETERS))
    violated = objective.residual_vector(heston_parameters_to_vector(FELLER_VIOLATING_PARAMETERS))
    assert satisfied.shape == violated.shape == (3,)
    assert satisfied[-2:].tolist() == [0.0, 0.0]
    assert violated[-2] > 0.0


def test_pricing_failure_returns_finite_penalty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = generate_synthetic_heston_dataset(maturities=(1.0,), log_moneyness=(-0.1, 0.1))
    config = HestonCalibrationConfig(
        target=HestonCalibrationTarget.PRICE,
        failure_penalty=123.0,
    )
    objective = HestonCalibrationObjective(prepare_calibration_data(dataset, config), config)

    def fail(*_: object, **__: object) -> None:
        raise HestonPricingError("forced numerical failure")

    monkeypatch.setattr(HestonAnalyticalBatchPricer, "price", fail)
    evaluation = objective.evaluate(heston_parameters_to_vector(FELLER_SATISFIED_PARAMETERS))
    assert evaluation.raw_residuals.tolist() == [123.0, 123.0]
    assert all(message == "forced numerical failure" for message in evaluation.failure_messages)
    assert objective.failed_evaluations == 2
    assert math.isfinite(objective.scalar(heston_parameters_to_vector(FELLER_SATISFIED_PARAMETERS)))
