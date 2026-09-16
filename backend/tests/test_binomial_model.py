import math

import numpy as np
import pytest
from pydantic import ValidationError

from backend.app.core.exceptions import InvalidRiskNeutralProbabilityError
from backend.app.models.binomial import BinomialTreeConfig, TreeModel
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.pricing.binomial import crr_parameters, crr_terminal_stock_prices


def contract(*, maturity: float = 1.0) -> OptionContract:
    return OptionContract(
        option_type=OptionType.CALL,
        spot=100.0,
        strike=100.0,
        maturity_years=maturity,
    )


def test_tree_config_validation_and_model_enum() -> None:
    assert BinomialTreeConfig().steps == 500
    assert BinomialTreeConfig().model is TreeModel.CRR
    with pytest.raises(ValidationError):
        BinomialTreeConfig(steps=0)
    with pytest.raises(ValidationError):
        BinomialTreeConfig(model="JARROW_RUDD")
    with pytest.raises(ValidationError):
        BinomialTreeConfig(probability_tolerance=1e-4)


def test_crr_parameter_formulas_match_manual_calculation() -> None:
    option = contract(maturity=2.0)
    market = MarketEnvironment(risk_free_rate=0.03, volatility=0.25, dividend_yield=0.01)
    parameters = crr_parameters(option, market, BinomialTreeConfig(steps=8))
    expected_dt = 0.25
    expected_up = math.exp(0.25 * math.sqrt(expected_dt))
    expected_down = 1.0 / expected_up
    expected_growth = math.exp((0.03 - 0.01) * expected_dt)
    expected_probability = (expected_growth - expected_down) / (expected_up - expected_down)
    assert parameters.dt == expected_dt
    assert parameters.up_factor == pytest.approx(expected_up)
    assert parameters.down_factor == pytest.approx(expected_down)
    assert parameters.up_factor * parameters.down_factor == pytest.approx(1.0)
    assert parameters.risk_neutral_growth == pytest.approx(expected_growth)
    assert parameters.risk_neutral_probability == pytest.approx(expected_probability)
    assert parameters.discount_factor == pytest.approx(math.exp(-0.03 * expected_dt))


def test_terminal_prices_use_up_move_count_indexing() -> None:
    option = contract()
    parameters = crr_parameters(
        option,
        MarketEnvironment(risk_free_rate=0.05, volatility=0.2),
        BinomialTreeConfig(steps=2),
    )
    prices = crr_terminal_stock_prices(option, parameters)
    expected = np.array(
        [
            100.0 * parameters.down_factor**2,
            100.0 * parameters.down_factor * parameters.up_factor,
            100.0 * parameters.up_factor**2,
        ]
    )
    np.testing.assert_allclose(prices, expected, rtol=1e-14)


def test_invalid_coarse_tree_probability_is_rejected_without_clamping() -> None:
    with pytest.raises(InvalidRiskNeutralProbabilityError, match="outside"):
        crr_parameters(
            contract(),
            MarketEnvironment(risk_free_rate=0.50, volatility=0.01),
            BinomialTreeConfig(steps=1),
        )


@pytest.mark.parametrize("carry", [-0.2, 0.2])
def test_valid_boundary_probabilities_near_zero_or_one(carry: float) -> None:
    parameters = crr_parameters(
        contract(),
        MarketEnvironment(risk_free_rate=carry, volatility=0.2),
        BinomialTreeConfig(steps=1),
    )
    assert 0.0 <= parameters.risk_neutral_probability <= 1.0
    expected = 0.0 if carry < 0.0 else 1.0
    assert parameters.risk_neutral_probability == pytest.approx(expected, abs=1e-15)
