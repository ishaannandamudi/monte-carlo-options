import numpy as np
import pytest
from pydantic import ValidationError

from backend.app.models.basket import BasketOptionContract, BasketType
from backend.app.models.option import OptionType
from backend.app.payoffs.basket import basket_payoff, basket_values


def contract(
    basket_type: BasketType, option_type: OptionType = OptionType.CALL
) -> BasketOptionContract:
    return BasketOptionContract(
        option_type=option_type,
        strike=100,
        maturity_years=1,
        weights=(0.25, 0.75),
        basket_type=basket_type,
    )


def test_arithmetic_and_geometric_values_and_payoffs() -> None:
    terminal = np.array([[80.0, 120.0], [100.0, 100.0], [140.0, 80.0]])
    arithmetic = basket_values(terminal, (0.25, 0.75), BasketType.ARITHMETIC)
    geometric = basket_values(terminal, (0.25, 0.75), BasketType.GEOMETRIC)
    np.testing.assert_allclose(arithmetic, [110.0, 100.0, 95.0])
    np.testing.assert_allclose(geometric, np.exp(np.log(terminal) @ np.array([0.25, 0.75])))
    np.testing.assert_allclose(basket_payoff(terminal, contract(BasketType.ARITHMETIC)), [10, 0, 0])
    np.testing.assert_allclose(
        basket_payoff(terminal, contract(BasketType.ARITHMETIC, OptionType.PUT)), [0, 0, 5]
    )


@pytest.mark.parametrize(
    "weights",
    [(0.2, 0.2), (-0.1, 1.1), (0.0, 0.0), (float("nan"), 0.0)],
)
def test_weight_policy_rejects_invalid_weights(weights: tuple[float, float]) -> None:
    with pytest.raises(ValidationError):
        BasketOptionContract(
            option_type=OptionType.CALL,
            strike=100,
            maturity_years=1,
            weights=weights,
            basket_type=BasketType.ARITHMETIC,
        )


def test_tiny_weight_rounding_is_normalized() -> None:
    value = BasketOptionContract(
        option_type=OptionType.CALL,
        strike=100,
        maturity_years=1,
        weights=(0.4, 0.600000000001),
        basket_type=BasketType.ARITHMETIC,
    )
    assert sum(value.weights) == pytest.approx(1.0, abs=1e-16)
