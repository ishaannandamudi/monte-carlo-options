import pytest

from backend.app.models.asian import (
    AsianOptionContract,
    AsianSimulationConfig,
    AsianVarianceReduction,
    AveragingType,
)
from backend.app.models.digital import (
    DigitalOptionContract,
    DigitalSimulationConfig,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionType
from backend.app.pricing.asian import asian_monte_carlo_price
from backend.app.pricing.digital import digital_monte_carlo_price


@pytest.mark.performance
@pytest.mark.parametrize(
    "paths,observations",
    [(10_000, 12), (100_000, 12), (100_000, 52), (100_000, 252)],
)
def test_asian_performance(paths: int, observations: int) -> None:
    contract = AsianOptionContract(
        option_type=OptionType.CALL,
        spot=100,
        strike=100,
        maturity_years=1,
        averaging_type=AveragingType.ARITHMETIC,
        averaging_schedule={"monitoring_count": observations},
    )
    market = MarketEnvironment(risk_free_rate=0.05, volatility=0.2, dividend_yield=0.01)
    results = []
    for method in (
        AsianVarianceReduction.NONE,
        AsianVarianceReduction.GEOMETRIC_CONTROL,
    ):
        results.append(
            asian_monte_carlo_price(
                contract,
                market,
                AsianSimulationConfig(
                    num_simulations=paths,
                    variance_reduction=method,
                ),
            )
        )
    raw, controlled = results
    assert raw.price > 0 and controlled.price > 0
    assert controlled.standard_error < raw.standard_error
    print(
        f"Asian {paths} x {observations}: raw={raw.runtime_ms:.2f}ms "
        f"control={controlled.runtime_ms:.2f}ms simulation={raw.simulation_runtime_ms:.2f}ms "
        f"observations={8 * paths * observations / 1e6:.2f}MB"
    )


@pytest.mark.performance
@pytest.mark.parametrize("paths", [100_000, 1_000_000])
def test_digital_performance(paths: int) -> None:
    result = digital_monte_carlo_price(
        DigitalOptionContract(
            option_type=OptionType.CALL,
            spot=100,
            strike=100,
            maturity_years=1,
            cash_payout=10,
        ),
        MarketEnvironment(risk_free_rate=0.05, volatility=0.2, dividend_yield=0.01),
        DigitalSimulationConfig(num_simulations=paths),
    )
    assert abs(result.price - result.analytical_price) <= 6 * result.standard_error
    print(f"Digital {paths}: total={result.runtime_ms:.2f}ms terminal={8 * paths / 1e6:.2f}MB")
