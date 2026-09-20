"""QMC method comparisons and convergence-rate research for European GBM."""

from __future__ import annotations

import math

import numpy as np

from backend.app.models.qmc import (
    PowerOfTwoPolicy,
    QMCComparisonRequest,
    QMCComparisonResult,
    QMCConfig,
    QMCConvergenceRequest,
    QMCConvergenceResult,
    QMCEuropeanRequest,
    QMCMethodPoint,
    QMCMethodSlope,
    QMCPricingResult,
    SamplingMethod,
)
from backend.app.pricing.black_scholes import black_scholes_price
from backend.app.pricing.qmc import price_european_qmc
from backend.app.simulation.random import spawn_child_seeds


def _point(result: QMCPricingResult, reference: float) -> QMCMethodPoint:
    absolute_error = abs(result.price - reference)
    relative_error = 0.0 if reference == 0.0 else absolute_error / abs(reference)
    return QMCMethodPoint(
        method=result.sampling_method,
        num_samples=result.effective_samples,
        estimate=result.price,
        reference_price=reference,
        absolute_error=absolute_error,
        relative_error=relative_error,
        standard_error=result.standard_error,
        runtime_ms=result.runtime_ms,
        error_sqrt_runtime=absolute_error * math.sqrt(result.runtime_ms / 1_000.0),
        metadata=result.metadata,
    )


def _slope(points: list[QMCMethodPoint], attribute: str) -> tuple[float | None, int]:
    usable = [
        point
        for point in points
        if (value := getattr(point, attribute)) is not None and float(value) > 0.0
    ]
    if len(usable) < 2:
        return None, len(usable)
    x = np.log(np.asarray([point.num_samples for point in usable], dtype=np.float64))
    y = np.log(np.asarray([float(getattr(point, attribute)) for point in usable]))
    return float(np.polyfit(x, y, 1)[0]), len(usable)


def _configs(
    *,
    samples: int,
    seed: int,
    replicates: int,
    confidence_level: float,
    power_policy: PowerOfTwoPolicy,
) -> list[QMCConfig]:
    return [
        QMCConfig(
            method=SamplingMethod.PSEUDORANDOM,
            dimension=1,
            num_samples=samples,
            scramble=False,
            seed=seed,
            num_randomizations=1,
            power_of_two_policy=power_policy,
            confidence_level=confidence_level,
        ),
        QMCConfig(
            method=SamplingMethod.SOBOL,
            dimension=1,
            num_samples=samples,
            scramble=False,
            seed=seed,
            num_randomizations=1,
            power_of_two_policy=power_policy,
            confidence_level=confidence_level,
        ),
        QMCConfig(
            method=SamplingMethod.SCRAMBLED_SOBOL,
            dimension=1,
            num_samples=samples,
            scramble=True,
            seed=seed,
            num_randomizations=replicates,
            power_of_two_policy=power_policy,
            confidence_level=confidence_level,
        ),
    ]


def compare_qmc_methods(request: QMCComparisonRequest) -> QMCComparisonResult:
    reference = black_scholes_price(request.contract, request.market)
    seeds = spawn_child_seeds(request.root_seed, 3)
    points: list[QMCMethodPoint] = []
    for config, seed in zip(
        _configs(
            samples=request.num_samples,
            seed=request.root_seed,
            replicates=request.randomized_replicates,
            confidence_level=request.confidence_level,
            power_policy=request.power_of_two_policy,
        ),
        seeds,
        strict=True,
    ):
        configured = config.model_copy(update={"seed": seed})
        result = price_european_qmc(
            QMCEuropeanRequest(contract=request.contract, market=request.market, qmc=configured)
        )
        points.append(_point(result, reference))
    return QMCComparisonResult(
        reference_price=reference,
        reference_label="Black-Scholes-Merton analytical price",
        points=points,
    )


def qmc_convergence(request: QMCConvergenceRequest) -> QMCConvergenceResult:
    counts = [1 << power for power in range(request.minimum_power, request.maximum_power + 1)]
    seeds = spawn_child_seeds(request.root_seed, len(counts) * 3)
    reference = black_scholes_price(request.contract, request.market)
    points: list[QMCMethodPoint] = []
    for count_index, count in enumerate(counts):
        for method_index, config in enumerate(
            _configs(
                samples=count,
                seed=seeds[count_index * 3],
                replicates=request.randomized_replicates,
                confidence_level=request.confidence_level,
                power_policy=PowerOfTwoPolicy.REQUIRE_POWER_OF_TWO,
            )
        ):
            configured = config.model_copy(update={"seed": seeds[count_index * 3 + method_index]})
            result = price_european_qmc(
                QMCEuropeanRequest(contract=request.contract, market=request.market, qmc=configured)
            )
            points.append(_point(result, reference))
    slopes: list[QMCMethodSlope] = []
    for method in SamplingMethod:
        selected = [point for point in points if point.method is method]
        error_slope, error_count = _slope(selected, "absolute_error")
        se_slope, se_count = _slope(selected, "standard_error")
        slopes.append(
            QMCMethodSlope(
                method=method,
                error_slope=error_slope,
                error_points_used=error_count,
                standard_error_slope=se_slope,
                standard_error_points_used=se_count,
            )
        )
    return QMCConvergenceResult(
        reference_price=reference,
        reference_label="Black-Scholes-Merton analytical price",
        points=points,
        slopes=slopes,
    )


def qmc_compute_frontier(request: QMCConvergenceRequest) -> QMCConvergenceResult:
    """Expose Phase 15-style error/runtime points without registering a new financial model."""

    return qmc_convergence(request)
