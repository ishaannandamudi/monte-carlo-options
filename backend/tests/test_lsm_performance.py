import pytest

from backend.app.models.longstaff_schwartz import LongstaffSchwartzConfig
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import ExerciseStyle, OptionContract, OptionType
from backend.app.pricing.longstaff_schwartz import longstaff_schwartz_price


@pytest.mark.performance
@pytest.mark.parametrize("paths,steps", [(10000, 50), (50000, 50), (100000, 50), (50000, 100)])
def test_lsm_performance(paths: int, steps: int) -> None:
    result = longstaff_schwartz_price(
        OptionContract(
            option_type=OptionType.PUT,
            exercise_style=ExerciseStyle.AMERICAN,
            spot=100,
            strike=100,
            maturity_years=1,
        ),
        MarketEnvironment(risk_free_rate=0.05, volatility=0.2),
        LongstaffSchwartzConfig(num_paths=paths, num_time_steps=steps),
    )
    assert abs(result.price - 6.09) < 5 * result.standard_error + 0.15
    assert result.approximate_path_array_bytes == 8 * paths * (steps + 1)
    print(
        f"{paths} x {steps}: simulation={result.simulation_runtime_ms:.2f}ms "
        f"regression={result.regression_runtime_ms:.2f}ms "
        f"evaluation={result.valuation_runtime_ms:.2f}ms total={result.runtime_ms:.2f}ms "
        f"path_array={result.approximate_path_array_bytes / 1e6:.2f}MB"
    )
