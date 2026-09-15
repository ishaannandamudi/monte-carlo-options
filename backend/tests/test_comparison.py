from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.models.simulation import SimulationConfig
from backend.app.pricing.comparison import compare_prices


def test_comparison_diagnostics_are_well_defined() -> None:
    comparison = compare_prices(
        OptionContract(
            option_type=OptionType.CALL,
            spot=100.0,
            strike=100.0,
            maturity_years=1.0,
        ),
        MarketEnvironment(risk_free_rate=0.05, volatility=0.2),
        SimulationConfig(num_simulations=100_000, seed=42),
    )
    assert comparison.relative_error is not None
    assert comparison.z_score is not None
    assert comparison.absolute_error == abs(
        comparison.monte_carlo.price - comparison.black_scholes_price
    )
    assert comparison.black_scholes_inside_confidence_interval


def test_zero_analytical_price_has_no_relative_error_or_z_score() -> None:
    comparison = compare_prices(
        OptionContract(
            option_type=OptionType.CALL,
            spot=90.0,
            strike=100.0,
            maturity_years=0.0,
        ),
        MarketEnvironment(risk_free_rate=-0.02, volatility=0.3),
        SimulationConfig(num_simulations=5),
    )
    assert comparison.black_scholes_price == 0.0
    assert comparison.relative_error is None
    assert comparison.z_score is None
    assert comparison.black_scholes_inside_confidence_interval
