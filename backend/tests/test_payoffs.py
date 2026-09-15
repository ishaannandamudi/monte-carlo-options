import numpy as np

from backend.app.models.option import OptionContract, OptionType
from backend.app.payoffs.vanilla import call_payoff, intrinsic_value, put_payoff, vanilla_payoff


def test_known_call_payoff_vector_and_equality() -> None:
    result = call_payoff(np.array([80.0, 100.0, 125.0]), 100.0)
    np.testing.assert_array_equal(result, np.array([0.0, 0.0, 25.0]))


def test_known_put_payoff_vector_and_equality() -> None:
    result = put_payoff(np.array([80.0, 100.0, 125.0]), 100.0)
    np.testing.assert_array_equal(result, np.array([20.0, 0.0, 0.0]))


def test_dispatch_and_intrinsic_value() -> None:
    put = OptionContract(option_type=OptionType.PUT, spot=90.0, strike=100.0, maturity_years=1.0)
    np.testing.assert_array_equal(vanilla_payoff(np.array([90.0, 110.0]), put), [10.0, 0.0])
    assert intrinsic_value(OptionType.CALL, 110.0, 100.0) == 10.0
    assert intrinsic_value(OptionType.PUT, 90.0, 100.0) == 10.0
