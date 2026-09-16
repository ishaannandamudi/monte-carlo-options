"""Analytical comparisons and independently seeded convergence for digital options."""

from collections.abc import Sequence

from backend.app.core.exceptions import ExoticPricingError
from backend.app.models.digital import (
    DigitalComparison,
    DigitalConvergencePoint,
    DigitalConvergenceResult,
    DigitalOptionContract,
    DigitalSimulationConfig,
)
from backend.app.models.market import MarketEnvironment
from backend.app.pricing.digital import digital_analytical_price, digital_monte_carlo_price
from backend.app.simulation.random import spawn_child_seeds


def compare_digital_prices(
    contract: DigitalOptionContract,
    market: MarketEnvironment,
    config: DigitalSimulationConfig | None = None,
) -> DigitalComparison:
    analytical = digital_analytical_price(contract, market)
    monte_carlo = digital_monte_carlo_price(contract, market, config)
    difference = monte_carlo.price - analytical.price
    return DigitalComparison(
        monte_carlo=monte_carlo,
        analytical=analytical,
        absolute_error=abs(difference),
        relative_error=(abs(difference) / analytical.price if analytical.price > 0.0 else None),
        z_score=(
            difference / monte_carlo.standard_error if monte_carlo.standard_error > 0.0 else None
        ),
        analytical_inside_confidence_interval=(
            monte_carlo.confidence_interval_low
            <= analytical.price
            <= monte_carlo.confidence_interval_high
        ),
    )


def digital_convergence_configs(
    config: DigitalSimulationConfig,
    simulation_counts: Sequence[int],
) -> list[DigitalSimulationConfig]:
    if (
        not simulation_counts
        or any(count <= 0 for count in simulation_counts)
        or len(set(simulation_counts)) != len(simulation_counts)
    ):
        raise ExoticPricingError("simulation_counts must be nonempty, positive, and unique")
    seeds = spawn_child_seeds(config.seed, len(simulation_counts))
    return [
        config.model_copy(update={"num_simulations": count, "seed": seed})
        for count, seed in zip(sorted(simulation_counts), seeds, strict=True)
    ]


def run_digital_convergence(
    contract: DigitalOptionContract,
    market: MarketEnvironment,
    config: DigitalSimulationConfig,
    simulation_counts: Sequence[int],
) -> DigitalConvergenceResult:
    analytical = digital_analytical_price(contract, market)
    points = []
    for point_config in digital_convergence_configs(config, simulation_counts):
        result = digital_monte_carlo_price(contract, market, point_config)
        absolute_error = abs(result.price - analytical.price)
        points.append(
            DigitalConvergencePoint(
                num_simulations=point_config.num_simulations,
                seed=point_config.seed,
                monte_carlo_price=result.price,
                analytical_price=analytical.price,
                absolute_error=absolute_error,
                relative_error=(
                    absolute_error / analytical.price if analytical.price > 0.0 else None
                ),
                standard_error=result.standard_error,
                confidence_interval_width=(
                    result.confidence_interval_high - result.confidence_interval_low
                ),
                runtime_ms=result.runtime_ms,
            )
        )
    return DigitalConvergenceResult(
        analytical_price=analytical.price,
        root_seed=config.seed,
        variance_reduction=config.variance_reduction,
        points=points,
    )
