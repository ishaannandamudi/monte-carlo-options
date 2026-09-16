import numpy as np
import pytest

from backend.app.models.asian import AsianOptionContract, AveragingType
from backend.app.models.digital import DigitalOptionContract, DigitalType
from backend.app.models.option import OptionType
from backend.app.payoffs.asian import asian_average, asian_payoff
from backend.app.payoffs.digital import digital_payoff


def asian_contract(option_type: OptionType, averaging_type: AveragingType) -> AsianOptionContract:
    return AsianOptionContract(
        option_type=option_type,
        spot=100,
        strike=100,
        maturity_years=1,
        averaging_type=averaging_type,
        averaging_schedule={"monitoring_count": 2},
    )


def test_fixed_arithmetic_and_geometric_averages() -> None:
    observations = np.array([[100.0, 121.0], [100.0, 81.0]])
    np.testing.assert_allclose(asian_average(observations, AveragingType.ARITHMETIC), [110.5, 90.5])
    np.testing.assert_allclose(asian_average(observations, AveragingType.GEOMETRIC), [110, 90])


@pytest.mark.parametrize(
    "option_type,averaging_type,expected",
    [
        (OptionType.CALL, AveragingType.ARITHMETIC, [10.5, 0]),
        (OptionType.PUT, AveragingType.ARITHMETIC, [0, 9.5]),
        (OptionType.CALL, AveragingType.GEOMETRIC, [10, 0]),
        (OptionType.PUT, AveragingType.GEOMETRIC, [0, 10]),
    ],
)
def test_fixed_asian_payoffs(
    option_type: OptionType, averaging_type: AveragingType, expected: list[float]
) -> None:
    observations = np.array([[100.0, 121.0], [100.0, 81.0]])
    np.testing.assert_allclose(
        asian_payoff(observations, asian_contract(option_type, averaging_type)), expected
    )


def test_include_initial_spot_changes_fixed_average_explicitly() -> None:
    future_only = np.array([[121.0]])
    with_initial = np.column_stack((np.full(1, 100.0), future_only))
    assert asian_average(future_only, AveragingType.ARITHMETIC)[0] == 121
    assert asian_average(with_initial, AveragingType.ARITHMETIC)[0] == 110.5
    assert asian_average(with_initial, AveragingType.GEOMETRIC)[0] == pytest.approx(110)


@pytest.mark.parametrize(
    "option_type,expected",
    [(OptionType.CALL, [0, 2, 0]), (OptionType.PUT, [2, 0, 0])],
)
def test_cash_digital_strict_equality(option_type: OptionType, expected: list[float]) -> None:
    contract = DigitalOptionContract(
        option_type=option_type,
        spot=100,
        strike=100,
        maturity_years=1,
        cash_payout=2,
    )
    np.testing.assert_array_equal(digital_payoff([90, 110, 100], contract), expected)


def test_asset_or_nothing_payoff() -> None:
    contract = DigitalOptionContract(
        option_type=OptionType.CALL,
        spot=100,
        strike=100,
        maturity_years=1,
        digital_type=DigitalType.ASSET_OR_NOTHING,
    )
    np.testing.assert_array_equal(digital_payoff([90, 110, 100], contract), [0, 110, 0])
