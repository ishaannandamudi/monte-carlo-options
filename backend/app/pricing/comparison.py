"""Diagnostics comparing simulation with the analytical Phase 1 benchmark."""

from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract
from backend.app.models.results import PricingComparison
from backend.app.models.simulation import SimulationConfig
from backend.app.pricing.black_scholes import black_scholes_price
from backend.app.pricing.monte_carlo import MonteCarloPricer


def compare_prices(
    contract: OptionContract,
    market: MarketEnvironment,
    config: SimulationConfig,
    *,
    monte_carlo_pricer: MonteCarloPricer | None = None,
) -> PricingComparison:
    """Return absolute, relative, z-score, and interval-containment diagnostics."""

    monte_carlo = (monte_carlo_pricer or MonteCarloPricer()).price(contract, market, config)
    analytical = black_scholes_price(contract, market)
    absolute_error = abs(monte_carlo.price - analytical)
    relative_error = absolute_error / abs(analytical) if analytical != 0.0 else None
    z_score = (
        (monte_carlo.price - analytical) / monte_carlo.standard_error
        if monte_carlo.standard_error > 0.0
        else None
    )
    inside = (
        monte_carlo.confidence_interval_low <= analytical <= monte_carlo.confidence_interval_high
    )
    return PricingComparison(
        monte_carlo=monte_carlo,
        black_scholes_price=analytical,
        absolute_error=absolute_error,
        relative_error=relative_error,
        z_score=z_score,
        black_scholes_inside_confidence_interval=inside,
    )
