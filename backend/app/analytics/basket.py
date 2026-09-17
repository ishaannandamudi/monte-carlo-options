"""Reproducible basket comparisons, convergence, and sensitivity experiments."""

import math
from collections.abc import Sequence

import numpy as np

from backend.app.core.exceptions import MultiAssetPricingError
from backend.app.models.basket import (
    BasketComparisonPoint,
    BasketComparisonResult,
    BasketConvergencePoint,
    BasketConvergenceResult,
    BasketOptionContract,
    BasketPricingResult,
    BasketSimulationConfig,
    BasketType,
    BasketVarianceReduction,
    CorrelationSensitivityPoint,
    CorrelationSensitivityResult,
    WeightSensitivityPoint,
    WeightSensitivityResult,
)
from backend.app.models.multi_asset import MultiAssetMarketEnvironment
from backend.app.pricing.basket import (
    basket_monte_carlo_price,
    geometric_basket_analytical_price,
)
from backend.app.simulation.correlation import build_equicorrelation_matrix
from backend.app.simulation.random import spawn_child_seeds


def _comparison_point(
    result: BasketPricingResult,
    analytical_reference: float | None,
    baseline_variance: float,
) -> BasketComparisonPoint:
    ratio = (
        baseline_variance / result.estimator_variance if result.estimator_variance > 0.0 else None
    )
    return BasketComparisonPoint(
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
        variance_reduction_ratio=ratio,
    )


def compare_basket_methods(
    contract: BasketOptionContract,
    market: MultiAssetMarketEnvironment,
    config: BasketSimulationConfig | None = None,
) -> BasketComparisonResult:
    """Compare compatible methods using the same configured root stream."""

    base = config or BasketSimulationConfig()
    methods = (
        list(BasketVarianceReduction)
        if contract.basket_type is BasketType.ARITHMETIC
        else [BasketVarianceReduction.NONE, BasketVarianceReduction.ANTITHETIC]
    )
    results = [
        basket_monte_carlo_price(
            contract, market, base.model_copy(update={"variance_reduction": method})
        )
        for method in methods
    ]
    analytical = (
        geometric_basket_analytical_price(contract, market).price
        if contract.basket_type is BasketType.GEOMETRIC
        else None
    )
    baseline_variance = results[0].estimator_variance
    return BasketComparisonResult(
        basket_type=contract.basket_type,
        num_simulations=base.num_simulations,
        seed=base.seed,
        num_assets=market.num_assets,
        points=[_comparison_point(result, analytical, baseline_variance) for result in results],
    )


def basket_convergence_inputs(
    config: BasketSimulationConfig, counts: Sequence[int]
) -> list[BasketSimulationConfig]:
    """Validate counts and create deterministic independent child streams."""

    if not counts or any(count <= 0 for count in counts) or len(set(counts)) != len(counts):
        raise MultiAssetPricingError(
            "basket convergence counts must be nonempty, positive, and unique"
        )
    sorted_counts = sorted(counts)
    seeds = spawn_child_seeds(config.seed, len(sorted_counts))
    return [
        config.model_copy(update={"num_simulations": count, "seed": seed})
        for count, seed in zip(sorted_counts, seeds, strict=True)
    ]


def run_basket_convergence(
    contract: BasketOptionContract,
    market: MultiAssetMarketEnvironment,
    config: BasketSimulationConfig,
    counts: Sequence[int],
) -> BasketConvergenceResult:
    """Run independently seeded path-count points and estimate the log-SE slope."""

    analytical = (
        geometric_basket_analytical_price(contract, market).price
        if contract.basket_type is BasketType.GEOMETRIC
        else None
    )
    points = []
    for point_config in basket_convergence_inputs(config, counts):
        result = basket_monte_carlo_price(contract, market, point_config)
        points.append(
            BasketConvergencePoint(
                num_simulations=point_config.num_simulations,
                seed=point_config.seed,
                price=result.price,
                standard_error=result.standard_error,
                confidence_interval_width=(
                    result.confidence_interval_high - result.confidence_interval_low
                ),
                runtime_ms=result.runtime_ms,
                analytical_reference=analytical,
                absolute_error=(abs(result.price - analytical) if analytical is not None else None),
                variance_reduction_ratio=result.variance_reduction_ratio,
            )
        )
    usable = [point for point in points if point.standard_error > 0.0]
    slope: float | None
    if len(usable) >= 2:
        slope = float(
            np.polyfit(
                np.log([point.num_simulations for point in usable]),
                np.log([point.standard_error for point in usable]),
                1,
            )[0]
        )
        if not math.isfinite(slope):
            slope = None
    else:
        slope = None
    return BasketConvergenceResult(
        root_seed=config.seed,
        variance_reduction=config.variance_reduction,
        basket_type=contract.basket_type,
        points=points,
        empirical_standard_error_slope=slope,
        slope_point_count=len(usable),
    )


def run_correlation_sensitivity(
    contract: BasketOptionContract,
    market: MultiAssetMarketEnvironment,
    config: BasketSimulationConfig,
    rho_values: Sequence[float],
) -> CorrelationSensitivityResult:
    """Price equicorrelation scenarios with common independent Gaussian draws."""

    if not rho_values or len(set(rho_values)) != len(rho_values):
        raise MultiAssetPricingError("correlation scenarios must be nonempty and unique")
    points = []
    for rho in rho_values:
        scenario = market.model_copy(
            update={
                "correlation_matrix": build_equicorrelation_matrix(market.num_assets, rho),
                "correlation_repaired": False,
                "correlation_adjustment_magnitude": 0.0,
                "original_min_eigenvalue": 0.0,
            }
        )
        result = basket_monte_carlo_price(contract, scenario, config)
        points.append(
            CorrelationSensitivityPoint(
                rho=rho,
                price=result.price,
                standard_error=result.standard_error,
                mean_basket_value=result.mean_basket_value,
                basket_variance=result.basket_std**2,
                runtime_ms=result.runtime_ms,
            )
        )
    return CorrelationSensitivityResult(
        seed=config.seed,
        num_assets=market.num_assets,
        points=points,
    )


def run_weight_sensitivity(
    contract: BasketOptionContract,
    market: MultiAssetMarketEnvironment,
    config: BasketSimulationConfig,
    first_asset_weights: Sequence[float],
) -> WeightSensitivityResult:
    """Sweep normalized two-asset weights with common Gaussian draws."""

    if market.num_assets != 2:
        raise MultiAssetPricingError("weight sensitivity currently requires exactly two assets")
    if (
        not first_asset_weights
        or len(set(first_asset_weights)) != len(first_asset_weights)
        or any(
            not math.isfinite(weight) or weight < 0.0 or weight > 1.0
            for weight in first_asset_weights
        )
    ):
        raise MultiAssetPricingError(
            "first-asset weights must be finite, unique, and within [0, 1]"
        )
    points = []
    for first_weight in first_asset_weights:
        weights = (first_weight, 1.0 - first_weight)
        scenario_contract = contract.model_copy(update={"weights": weights})
        result = basket_monte_carlo_price(scenario_contract, market, config)
        points.append(
            WeightSensitivityPoint(
                weights=weights,
                price=result.price,
                standard_error=result.standard_error,
                mean_basket_value=result.mean_basket_value,
                basket_variance=result.basket_std**2,
                runtime_ms=result.runtime_ms,
            )
        )
    return WeightSensitivityResult(seed=config.seed, points=points)
