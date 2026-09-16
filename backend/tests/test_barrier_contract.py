import pytest
from pydantic import ValidationError

from backend.app.models.barrier import (
    BarrierDirection,
    BarrierMonitoringMode,
    BarrierOptionContract,
    BarrierStyle,
    RebateTiming,
)
from backend.app.models.option import ExerciseStyle, OptionType


def barrier_contract(**updates: object) -> BarrierOptionContract:
    values: dict[str, object] = {
        "option_type": OptionType.CALL,
        "barrier_direction": BarrierDirection.DOWN,
        "barrier_style": BarrierStyle.OUT,
        "spot": 100,
        "strike": 100,
        "barrier": 90,
        "maturity_years": 1,
        "monitoring_schedule": {"monitoring_count": 12},
    }
    values.update(updates)
    return BarrierOptionContract.model_validate(values)


@pytest.mark.parametrize("direction", list(BarrierDirection))
@pytest.mark.parametrize("style", list(BarrierStyle))
def test_valid_barrier_directions_and_styles(
    direction: BarrierDirection, style: BarrierStyle
) -> None:
    contract = barrier_contract(barrier_direction=direction, barrier_style=style)
    assert contract.monitoring_count == 12
    assert contract.simulation_state_count == 12
    assert contract.monitoring_times[0] == pytest.approx(1 / 12)
    assert contract.monitoring_times[-1] == 1


def test_irregular_schedule_reuses_monitoring_model_and_adds_terminal_state() -> None:
    contract = barrier_contract(
        monitoring_schedule={"monitoring_times": [0.0, 0.2, 0.7]},
    )
    assert contract.monitoring_times == (0.2, 0.7)
    assert contract.monitoring_count == 2
    assert contract.simulation_times == (0.2, 0.7, 1.0)
    assert contract.simulation_state_count == 3


@pytest.mark.parametrize(
    "updates",
    [
        {"barrier": 0},
        {"barrier": -1},
        {"rebate": -0.01},
        {"maturity_years": -1},
        {"monitoring_schedule": {"monitoring_times": [0.5, 1.1]}},
        {"exercise_style": ExerciseStyle.AMERICAN},
    ],
)
def test_invalid_barrier_contract_values(updates: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        barrier_contract(**updates)


def test_unsupported_rebate_combinations_are_rejected() -> None:
    with pytest.raises(ValidationError, match="knock-in rebates"):
        barrier_contract(
            barrier_style=BarrierStyle.IN,
            rebate=2,
            rebate_timing=RebateTiming.AT_HIT,
        )
    with pytest.raises(ValidationError, match="does not support"):
        barrier_contract(
            barrier_style=BarrierStyle.OUT,
            rebate=2,
            rebate_timing=RebateTiming.AT_HIT,
            monitoring_mode=BarrierMonitoringMode.BROWNIAN_BRIDGE,
        )


@pytest.mark.parametrize(
    "direction,spot,barrier,expected",
    [
        (BarrierDirection.DOWN, 90, 90, True),
        (BarrierDirection.DOWN, 91, 90, False),
        (BarrierDirection.UP, 110, 110, True),
        (BarrierDirection.UP, 109, 110, False),
    ],
)
def test_initial_hit_is_inclusive_and_schedule_independent(
    direction: BarrierDirection, spot: float, barrier: float, expected: bool
) -> None:
    contract = barrier_contract(
        barrier_direction=direction,
        spot=spot,
        barrier=barrier,
        monitoring_schedule={"monitoring_times": [1.0]},
    )
    assert contract.initially_hit is expected


def test_zero_maturity_has_initial_check_and_no_future_states() -> None:
    contract = barrier_contract(maturity_years=0)
    assert contract.monitoring_count == 0
    assert contract.monitoring_times == ()
    assert contract.simulation_times == ()
