"""Reproducible Asian method comparisons and path/monitoring convergence experiments."""

import math
from collections.abc import Sequence

from backend.app.core.exceptions import ExoticPricingError
from backend.app.models.asian import (
    AsianComparisonPoint,
    AsianComparisonResult,
    AsianConvergenceMode,
    AsianConvergencePoint,
    AsianConvergenceResult,
    AsianOptionContract,
    AsianPricingResult,
    AsianSimulationConfig,
    AsianVarianceReduction,
    AveragingType,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.monitoring import MonitoringSchedule
from backend.app.pricing.asian import (
    asian_monte_carlo_price,
    geometric_asian_analytical_price,
)
from backend.app.simulation.random import spawn_child_seeds


def _comparison_point(
    result: AsianPricingResult,
    analytical_reference: float | None,
    baseline_variance: float,
) -> AsianComparisonPoint:
    relative_variance = (
        baseline_variance / result.estimator_variance if result.estimator_variance > 0.0 else None
    )
    return AsianComparisonPoint(
        method=result.variance_reduction,
        price=result.price,
        standard_error=result.standard_error,
        estimator_variance=result.estimator_variance,
        confidence_interval_width=(
            result.confidence_interval_high - result.confidence_interval_low
        ),
        runtime_ms=result.runtime_ms,
        analytical_reference=analytical_reference,
        absolute_error=(
            abs(result.price - analytical_reference) if analytical_reference is not None else None
        ),
        variance_reduction_ratio=relative_variance,
    )


def compare_asian_methods(
    contract: AsianOptionContract,
    market: MarketEnvironment,
    config: AsianSimulationConfig | None = None,
) -> AsianComparisonResult:
    """Compare compatible methods on shared seed-defined streams and one path budget."""

    base = config or AsianSimulationConfig()
    methods = (
        list(AsianVarianceReduction)
        if contract.averaging_type is AveragingType.ARITHMETIC
        else [AsianVarianceReduction.NONE, AsianVarianceReduction.ANTITHETIC]
    )
    results = [
        asian_monte_carlo_price(
            contract,
            market,
            base.model_copy(update={"variance_reduction": method}),
        )
        for method in methods
    ]
    baseline = results[0].estimator_variance
    analytical = (
        geometric_asian_analytical_price(contract, market).price
        if contract.averaging_type is AveragingType.GEOMETRIC
        else None
    )
    return AsianComparisonResult(
        averaging_type=contract.averaging_type,
        num_simulations=base.num_simulations,
        seed=base.seed,
        monitoring_count=contract.monitoring_count,
        points=[_comparison_point(result, analytical, baseline) for result in results],
    )


def asian_convergence_inputs(
    contract: AsianOptionContract,
    config: AsianSimulationConfig,
    values: Sequence[int],
    mode: AsianConvergenceMode,
) -> list[tuple[AsianOptionContract, AsianSimulationConfig]]:
    """Validate and derive independently seeded experiment inputs before pricing."""

    if not values or any(value <= 0 for value in values) or len(set(values)) != len(values):
        raise ExoticPricingError("convergence values must be nonempty, positive, and unique")
    if (
        mode is AsianConvergenceMode.MONITORING_COUNT
        and contract.averaging_schedule.monitoring_count is None
    ):
        raise ExoticPricingError(
            "monitoring-count convergence requires a regular monitoring schedule"
        )
    seeds = spawn_child_seeds(config.seed, len(values))
    result = []
    for value, seed in zip(sorted(values), seeds, strict=True):
        if mode is AsianConvergenceMode.PATH_COUNT:
            point_contract = contract
            point_config = config.model_copy(update={"num_simulations": value, "seed": seed})
        else:
            point_contract = contract.model_copy(
                update={"averaging_schedule": MonitoringSchedule.regular(value)}
            )
            point_config = config.model_copy(update={"seed": seed})
        result.append((point_contract, point_config))
    return result


def run_asian_convergence(
    contract: AsianOptionContract,
    market: MarketEnvironment,
    config: AsianSimulationConfig,
    values: Sequence[int],
    mode: AsianConvergenceMode,
) -> AsianConvergenceResult:
    inputs = asian_convergence_inputs(contract, config, values, mode)
    points = []
    for point_contract, point_config in inputs:
        result = asian_monte_carlo_price(point_contract, market, point_config)
        analytical = (
            geometric_asian_analytical_price(point_contract, market).price
            if point_contract.averaging_type is AveragingType.GEOMETRIC
            else None
        )
        points.append(
            AsianConvergencePoint(
                num_simulations=point_config.num_simulations,
                monitoring_count=point_contract.monitoring_count,
                seed=point_config.seed,
                price=result.price,
                standard_error=result.standard_error,
                confidence_interval_width=(
                    result.confidence_interval_high - result.confidence_interval_low
                ),
                runtime_ms=result.runtime_ms,
                analytical_reference=analytical,
                absolute_error=(abs(result.price - analytical) if analytical is not None else None),
            )
        )
    if any(not math.isfinite(point.price) for point in points):
        raise ExoticPricingError("Asian convergence produced a non-finite price")
    return AsianConvergenceResult(
        mode=mode,
        root_seed=config.seed,
        variance_reduction=config.variance_reduction,
        points=points,
    )
