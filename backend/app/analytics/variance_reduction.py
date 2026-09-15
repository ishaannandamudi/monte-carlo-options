"""Fixed-budget comparison of all supported variance-reduction methods."""

import logging
import math

from backend.app.core.exceptions import NumericalError
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract
from backend.app.models.results import (
    VarianceReductionComparisonPoint,
    VarianceReductionComparisonResult,
)
from backend.app.models.simulation import SimulationConfig, VarianceReduction
from backend.app.pricing.black_scholes import black_scholes_price
from backend.app.pricing.monte_carlo import MonteCarloPricer

logger = logging.getLogger(__name__)


def _finite_ratio(numerator: float, denominator: float, *, context: str) -> float | None:
    """Divide non-negative diagnostics without serializing infinity."""

    if denominator == 0.0:
        return None
    ratio = numerator / denominator
    if not math.isfinite(ratio):
        raise NumericalError(f"{context} is non-finite")
    return ratio


def compare_variance_reduction_methods(
    contract: OptionContract,
    market: MarketEnvironment,
    num_simulations: int,
    *,
    root_seed: int = 42,
    confidence_level: float = 0.95,
) -> VarianceReductionComparisonResult:
    """Compare methods using controlled common root randomness.

    NONE and CONTROL_VARIATE receive identical path normals. ANTITHETIC and
    ANTITHETIC_CONTROL_VARIATE receive identical primary normals and pairing.
    All four begin at the same PCG64 root seed.
    """

    if num_simulations <= 0:
        raise ValueError("num_simulations must be positive")
    if root_seed < 0:
        raise ValueError("root_seed must be non-negative")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")

    logger.info(
        "variance_reduction_comparison_start simulations=%d methods=%d",
        num_simulations,
        len(VarianceReduction),
    )
    analytical = black_scholes_price(contract, market)
    pricer = MonteCarloPricer()
    method_results = [
        (
            method,
            pricer.price(
                contract,
                market,
                SimulationConfig(
                    num_simulations=num_simulations,
                    seed=root_seed,
                    variance_reduction=method,
                    confidence_level=confidence_level,
                ),
            ),
        )
        for method in VarianceReduction
    ]
    baseline_result = next(
        result for method, result in method_results if method is VarianceReduction.NONE
    )
    baseline_variance = baseline_result.standard_error**2

    points: list[VarianceReductionComparisonPoint] = []
    for method, result in method_results:
        method_variance = result.standard_error**2
        relative_variance = _finite_ratio(
            method_variance,
            baseline_variance,
            context="relative variance",
        )
        relative_efficiency = _finite_ratio(
            baseline_variance,
            method_variance,
            context="relative efficiency",
        )
        points.append(
            VarianceReductionComparisonPoint(
                method=method,
                price=result.price,
                black_scholes_price=analytical,
                absolute_error=abs(result.price - analytical),
                standard_error=result.standard_error,
                estimator_variance=method_variance,
                raw_estimator_variance=result.raw_estimator_variance,
                adjusted_estimator_variance=result.adjusted_estimator_variance,
                variance_reduction_ratio=result.variance_reduction_ratio,
                relative_variance=relative_variance,
                relative_efficiency=relative_efficiency,
                confidence_interval_width=(
                    result.confidence_interval_high - result.confidence_interval_low
                ),
                runtime_ms=result.runtime_ms,
                control_variate_beta=result.control_variate_beta,
            )
        )

    if not all(math.isfinite(point.estimator_variance) for point in points):
        raise NumericalError("variance-reduction comparison produced non-finite variance")
    logger.info("variance_reduction_comparison_complete simulations=%d", num_simulations)
    return VarianceReductionComparisonResult(
        baseline_method=VarianceReduction.NONE,
        black_scholes_price=analytical,
        num_simulations=num_simulations,
        root_seed=root_seed,
        confidence_level=confidence_level,
        points=points,
    )
