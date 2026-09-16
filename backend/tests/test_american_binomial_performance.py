import pytest

from backend.app.models.binomial import BinomialTreeConfig
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import ExerciseStyle, OptionContract, OptionType
from backend.app.pricing.binomial import american_exercise_boundary, binomial_price


@pytest.mark.performance
@pytest.mark.parametrize("steps", [100, 500, 1_000, 5_000])
def test_european_american_and_boundary_benchmark(steps: int) -> None:
    european = OptionContract(
        option_type=OptionType.PUT,
        exercise_style=ExerciseStyle.EUROPEAN,
        spot=100.0,
        strike=100.0,
        maturity_years=1.0,
    )
    american = european.model_copy(update={"exercise_style": ExerciseStyle.AMERICAN})
    market = MarketEnvironment(risk_free_rate=0.05, volatility=0.2)
    config = BinomialTreeConfig(steps=steps)

    european_result = binomial_price(european, market, config)
    american_result = binomial_price(american, market, config)
    boundary = american_exercise_boundary(american, market, config)

    assert european_result.price >= 0.0
    assert american_result.price >= european_result.price
    assert boundary.exercise_node_count > 0
