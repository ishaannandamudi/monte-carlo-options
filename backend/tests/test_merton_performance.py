import pytest

from backend.app.analytics.merton import merton_implied_volatility_smile
from backend.app.models.market import MarketEnvironment
from backend.app.models.merton import MertonJumpParameters, MertonSimulationConfig
from backend.app.models.option import ExerciseStyle, OptionContract, OptionType
from backend.app.pricing.merton import merton_analytical_price, merton_monte_carlo_price


@pytest.mark.performance
def test_merton_pricing_performance_smoke() -> None:
    contract = OptionContract(
        option_type=OptionType.CALL,
        exercise_style=ExerciseStyle.EUROPEAN,
        spot=100.0,
        strike=100.0,
        maturity_years=1.0,
    )
    market = MarketEnvironment(risk_free_rate=0.03, dividend_yield=0.01, volatility=0.2)
    parameters = MertonJumpParameters(jump_intensity=0.6, jump_mean=-0.1, jump_volatility=0.25)
    analytical = merton_analytical_price(contract, market, parameters)
    mc = merton_monte_carlo_price(
        contract,
        market,
        parameters,
        MertonSimulationConfig(num_simulations=1_000_000, seed=44),
    )
    smile = merton_implied_volatility_smile(
        contract, market, parameters, [50.0 + index for index in range(100)]
    )
    assert analytical.runtime_ms < 1_000.0
    assert mc.runtime_ms < 10_000.0
    assert len(smile.points) == 100
    assert abs(mc.price - analytical.price) <= 5.0 * mc.standard_error
