import pytest
from pydantic import ValidationError

from backend.app.models.asian import AsianOptionContract, AveragingType
from backend.app.models.monitoring import MonitoringSchedule
from backend.app.models.option import ExerciseStyle, OptionType


def test_regular_schedule_and_initial_spot_policy() -> None:
    schedule = MonitoringSchedule.regular(4)
    assert schedule.resolve(1.0, include_initial_spot=False) == (0.25, 0.5, 0.75, 1.0)
    assert schedule.resolve(1.0, include_initial_spot=True) == (0.0, 0.25, 0.5, 0.75, 1.0)


def test_explicit_schedule_normalizes_order_and_validates_zero_policy() -> None:
    schedule = MonitoringSchedule.explicit([1.0, 0.25, 0.5])
    assert schedule.monitoring_times == (0.25, 0.5, 1.0)
    assert schedule.resolve(1.0, include_initial_spot=False) == (0.25, 0.5, 1.0)
    with_zero = MonitoringSchedule.explicit([1.0, 0.0, 0.5])
    assert with_zero.resolve(1.0, include_initial_spot=True) == (0.0, 0.5, 1.0)
    with pytest.raises(ValueError, match="include_initial"):
        with_zero.resolve(1.0, include_initial_spot=False)
    assert schedule.resolve(1.0, include_initial_spot=True) == (0.0, 0.25, 0.5, 1.0)


@pytest.mark.parametrize(
    "data",
    [
        {"monitoring_count": 0},
        {"monitoring_times": []},
        {"monitoring_times": [-0.1, 1.0]},
        {"monitoring_times": [0.5, 0.5]},
        {"monitoring_count": 3, "monitoring_times": [1.0]},
    ],
)
def test_invalid_schedule_models(data: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        MonitoringSchedule.model_validate(data)


def test_contract_validates_maturity_and_reports_effective_observations() -> None:
    contract = AsianOptionContract(
        option_type=OptionType.CALL,
        spot=100,
        strike=100,
        maturity_years=1,
        averaging_type=AveragingType.ARITHMETIC,
        averaging_schedule={"monitoring_count": 2},
        include_initial_spot_in_average=True,
    )
    assert contract.observation_times == (0.0, 0.5, 1.0)
    assert contract.future_monitoring_times == (0.5, 1.0)
    assert contract.monitoring_count == 3
    with pytest.raises(ValidationError, match="exceed maturity"):
        AsianOptionContract.model_validate(
            {
                **contract.model_dump(),
                "include_initial_spot_in_average": False,
                "averaging_schedule": {"monitoring_times": [0.5, 1.1]},
            }
        )


def test_zero_maturity_has_one_defined_spot_observation() -> None:
    contract = AsianOptionContract(
        option_type=OptionType.PUT,
        spot=100,
        strike=110,
        maturity_years=0,
        averaging_type=AveragingType.GEOMETRIC,
    )
    assert contract.observation_times == (0.0,)
    explicit = contract.model_copy(
        update={"averaging_schedule": MonitoringSchedule.explicit([0.0])}
    )
    assert explicit.observation_times == (0.0,)


def test_regular_schedule_rejects_floating_point_interval_underflow() -> None:
    with pytest.raises(ValidationError, match="interval underflowed"):
        AsianOptionContract(
            option_type=OptionType.CALL,
            spot=100,
            strike=100,
            maturity_years=5e-324,
            averaging_type=AveragingType.ARITHMETIC,
            averaging_schedule=MonitoringSchedule.regular(2),
        )


def test_asian_contract_requires_european_exercise() -> None:
    with pytest.raises(ValidationError):
        AsianOptionContract(
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.AMERICAN,
            spot=100,
            strike=100,
            maturity_years=1,
            averaging_type=AveragingType.ARITHMETIC,
        )
