import math

import pytest
from pydantic import ValidationError

from backend.app.calibration.datasets import (
    heston_parameters_to_vector,
    parameters_within_bounds,
    prepare_calibration_data,
    vector_to_heston_parameters,
)
from backend.app.calibration.synthetic import (
    FELLER_SATISFIED_PARAMETERS,
    generate_synthetic_heston_dataset,
)
from backend.app.core.exceptions import HestonCalibrationError
from backend.app.models.heston import HestonParameters
from backend.app.models.heston_calibration import (
    CalibrationParameterRange,
    CalibrationSelectionPolicy,
    CalibrationWeighting,
    HestonCalibrationBounds,
    HestonCalibrationConfig,
    HestonCalibrationDataset,
    HestonCalibrationFilters,
    HestonCalibrationObservation,
    HestonCalibrationTarget,
)
from backend.app.models.option import OptionType


def test_parameter_vector_round_trip_and_bounds() -> None:
    vector = heston_parameters_to_vector(FELLER_SATISFIED_PARAMETERS)
    assert vector.tolist() == pytest.approx([0.04, 2.0, 0.04, 0.3, -0.7])
    assert vector_to_heston_parameters(vector) == FELLER_SATISFIED_PARAMETERS
    assert parameters_within_bounds(FELLER_SATISFIED_PARAMETERS, HestonCalibrationBounds())
    assert not parameters_within_bounds(
        HestonParameters(v0=3.0, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7),
        HestonCalibrationBounds(),
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"strike": 0, "maturity_years": 1, "option_type": "CALL", "observed_price": 1},
        {"strike": 100, "maturity_years": 0, "option_type": "CALL", "observed_price": 1},
        {"strike": 100, "maturity_years": 1, "option_type": "DIGITAL", "observed_price": 1},
        {
            "strike": 100,
            "maturity_years": 1,
            "option_type": "CALL",
            "observed_price": float("nan"),
        },
        {
            "strike": 100,
            "maturity_years": 1,
            "option_type": "CALL",
            "observed_price": 1,
            "bid": -0.01,
        },
    ],
)
def test_invalid_observations_are_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        HestonCalibrationObservation.model_validate(payload)


def test_invalid_bounds_and_filter_ranges_are_rejected() -> None:
    with pytest.raises(ValidationError, match="strictly below"):
        CalibrationParameterRange(lower=1.0, upper=1.0)
    with pytest.raises(ValidationError, match="rho"):
        HestonCalibrationBounds(rho=CalibrationParameterRange(lower=-1.1, upper=0.5))
    with pytest.raises(ValidationError, match="minimum_log_moneyness"):
        HestonCalibrationFilters(minimum_log_moneyness=0.2, maximum_log_moneyness=-0.2)


@pytest.mark.parametrize("target", list(HestonCalibrationTarget))
@pytest.mark.parametrize("weighting", list(CalibrationWeighting))
def test_all_targets_and_weighting_schemes_prepare_finite_normalized_data(
    target: HestonCalibrationTarget,
    weighting: CalibrationWeighting,
) -> None:
    dataset = generate_synthetic_heston_dataset(
        maturities=(0.5, 1.0), log_moneyness=(-0.1, 0.0, 0.1)
    )
    prepared = prepare_calibration_data(
        dataset,
        HestonCalibrationConfig(
            target=target,
            weighting=weighting,
            expiry_balancing=True,
        ),
    )
    weights = [item.weight for item in prepared.observations]
    assert len(weights) == 6
    assert all(math.isfinite(weight) and weight >= 0.0 for weight in weights)
    assert sum(weights) / len(weights) == pytest.approx(1.0)
    expiry_totals = {
        maturity: sum(
            item.weight
            for item in prepared.observations
            if item.observation.maturity_years == maturity
        )
        for maturity in (0.5, 1.0)
    }
    assert expiry_totals[0.5] == pytest.approx(expiry_totals[1.0])


def test_otm_policy_and_deterministic_duplicate_selection() -> None:
    base = generate_synthetic_heston_dataset(maturities=(1.0,), log_moneyness=(-0.1, 0.0, 0.1))
    put = base.observations[0]
    atm_call = base.observations[1]
    call = base.observations[2]
    itm_call = put.model_copy(update={"option_type": OptionType.CALL, "source": "ITM"})
    wide_duplicate = call.model_copy(
        update={
            "bid": max((call.mid or 0.0) - 0.5, 0.0),
            "ask": (call.mid or 0.0) + 0.5,
            "source": "WIDE",
        }
    )
    narrow_duplicate = call.model_copy(
        update={
            "bid": max((call.mid or 0.0) - 0.005, 0.0),
            "ask": (call.mid or 0.0) + 0.005,
            "source": "NARROW",
        }
    )
    dataset = base.model_copy(
        update={"observations": [put, atm_call, itm_call, wide_duplicate, narrow_duplicate]}
    )
    prepared = prepare_calibration_data(dataset, HestonCalibrationConfig())
    assert [item.observation.source for item in prepared.observations] == [
        "SYNTHETIC_HESTON",
        "SYNTHETIC_HESTON",
        "NARROW",
    ]
    assert prepared.exclusion_reasons == {
        "NOT_SELECTED_BY_OTM_POLICY": 1,
        "DUPLICATE_OBSERVATION": 1,
    }


def test_quality_filters_and_price_bounds_report_explicit_reasons() -> None:
    base = generate_synthetic_heston_dataset(maturities=(1.0,), log_moneyness=(0.0,))
    original = base.observations[0]
    observations = [
        original.model_copy(update={"quote_is_fresh": False}),
        original.model_copy(update={"bid": 2.0, "ask": 1.0, "source": "crossed"}),
        original.model_copy(
            update={
                "observed_price": 1_000.0,
                "mid": 1_000.0,
                "source": "bad-bound",
                "bid": None,
                "ask": None,
            }
        ),
    ]
    dataset = base.model_copy(update={"observations": observations})
    with pytest.raises(HestonCalibrationError, match="no calibration observations"):
        prepare_calibration_data(dataset, HestonCalibrationConfig())

    usable = original.model_copy(update={"source": "usable"})
    prepared = prepare_calibration_data(
        base.model_copy(update={"observations": [*observations, usable]}),
        HestonCalibrationConfig(selection_policy=CalibrationSelectionPolicy.ALL),
    )
    assert prepared.exclusion_reasons == {
        "STALE_OR_FUTURE_QUOTE": 1,
        "CROSSED_MARKET": 1,
        "PRICE_BOUND_VIOLATION": 1,
    }


def test_missing_market_values_and_custom_weights_fail_clearly() -> None:
    observation = HestonCalibrationObservation(
        strike=100.0,
        maturity_years=1.0,
        option_type=OptionType.CALL,
    )
    dataset = HestonCalibrationDataset(
        spot=100.0,
        risk_free_rate=0.02,
        observations=[observation],
    )
    with pytest.raises(HestonCalibrationError, match="no calibration observations"):
        prepare_calibration_data(dataset, HestonCalibrationConfig())

    dataset = generate_synthetic_heston_dataset(maturities=(1.0,), log_moneyness=(0.0,))
    dataset = dataset.model_copy(
        update={"observations": [dataset.observations[0].model_copy(update={"weight": None})]}
    )
    with pytest.raises(HestonCalibrationError, match="no calibration observations"):
        prepare_calibration_data(
            dataset,
            HestonCalibrationConfig(weighting=CalibrationWeighting.CUSTOM),
        )


def test_missing_mid_spread_and_extreme_moneyness_filters_are_auditable() -> None:
    base = generate_synthetic_heston_dataset(maturities=(1.0,), log_moneyness=(-0.3, 0.0, 0.1))
    usable = generate_synthetic_heston_dataset(maturities=(0.5,), log_moneyness=(0.0,))
    missing_mid = base.observations[1].model_copy(update={"mid": None, "source": "no-mid"})
    wide = base.observations[2].model_copy(
        update={"bid": 0.1, "ask": 2.1, "mid": 1.1, "source": "wide"}
    )
    dataset = base.model_copy(
        update={
            "observations": [
                base.observations[0],
                missing_mid,
                wide,
                usable.observations[0],
            ]
        }
    )
    prepared = prepare_calibration_data(
        dataset,
        HestonCalibrationConfig(
            selection_policy=CalibrationSelectionPolicy.ALL,
            filters=HestonCalibrationFilters(
                minimum_log_moneyness=-0.2,
                maximum_log_moneyness=0.2,
                maximum_relative_spread=0.5,
                exclude_missing_mid=True,
            ),
        ),
    )
    assert len(prepared.observations) == 1
    assert prepared.exclusion_reasons == {
        "BELOW_MINIMUM_LOG_MONEYNESS": 1,
        "MISSING_MIDPOINT": 1,
        "RELATIVE_SPREAD_TOO_WIDE": 1,
    }
