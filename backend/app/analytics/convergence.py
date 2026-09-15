"""Reproducible Monte Carlo convergence experiments."""

import logging
import math
from collections.abc import Sequence

import numpy as np

from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract
from backend.app.models.results import ConvergencePoint, ConvergenceResult
from backend.app.models.simulation import SimulationConfig, VarianceReduction
from backend.app.pricing.black_scholes import black_scholes_price
from backend.app.pricing.monte_carlo import MonteCarloPricer
from backend.app.simulation.random import spawn_child_seeds

logger = logging.getLogger(__name__)


def estimate_standard_error_slope(
    simulation_counts: Sequence[int],
    standard_errors: Sequence[float],
) -> tuple[float | None, int]:
    """Fit log(SE) = intercept + slope*log(N) using valid positive points."""

    if len(simulation_counts) != len(standard_errors):
        raise ValueError("simulation counts and standard errors must have equal length")
    valid = [
        (count, error)
        for count, error in zip(simulation_counts, standard_errors, strict=True)
        if count > 0 and math.isfinite(error) and error > 0.0
    ]
    if len(valid) < 2:
        return None, len(valid)

    log_counts = np.log(np.asarray([count for count, _ in valid], dtype=np.float64))
    log_errors = np.log(np.asarray([error for _, error in valid], dtype=np.float64))
    slope = float(np.polyfit(log_counts, log_errors, deg=1)[0])
    return slope, len(valid)


def run_convergence(
    contract: OptionContract,
    market: MarketEnvironment,
    simulation_counts: Sequence[int],
    *,
    root_seed: int = 42,
    variance_reduction: VarianceReduction = VarianceReduction.ANTITHETIC,
    confidence_level: float = 0.95,
) -> ConvergenceResult:
    """Run each path count with an independent deterministic SeedSequence child."""

    counts = list(simulation_counts)
    if not counts:
        raise ValueError("at least one simulation count is required")
    if any(count <= 0 for count in counts):
        raise ValueError("all simulation counts must be positive")
    if root_seed < 0:
        raise ValueError("root_seed must be non-negative")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")

    logger.info(
        "convergence_start points=%d variance_reduction=%s",
        len(counts),
        variance_reduction.value,
    )
    analytical = black_scholes_price(contract, market)
    child_seeds = spawn_child_seeds(root_seed, len(counts))
    pricer = MonteCarloPricer()
    points: list[ConvergencePoint] = []

    for count, child_seed in zip(counts, child_seeds, strict=True):
        result = pricer.price(
            contract,
            market,
            SimulationConfig(
                num_simulations=count,
                seed=child_seed,
                variance_reduction=variance_reduction,
                confidence_level=confidence_level,
            ),
        )
        absolute_error = abs(result.price - analytical)
        points.append(
            ConvergencePoint(
                num_simulations=count,
                monte_carlo_price=result.price,
                black_scholes_price=analytical,
                absolute_error=absolute_error,
                relative_error=absolute_error / abs(analytical) if analytical != 0.0 else None,
                standard_error=result.standard_error,
                confidence_interval_width=(
                    result.confidence_interval_high - result.confidence_interval_low
                ),
                runtime_ms=result.runtime_ms,
                seed=child_seed,
            )
        )

    slope, used = estimate_standard_error_slope(
        [point.num_simulations for point in points],
        [point.standard_error for point in points],
    )
    logger.info("convergence_complete points=%d slope=%s", len(points), slope)
    return ConvergenceResult(
        black_scholes_price=analytical,
        points=points,
        estimated_standard_error_slope=slope,
        slope_points_used=used,
        root_seed=root_seed,
        variance_reduction=variance_reduction,
        confidence_level=confidence_level,
    )
