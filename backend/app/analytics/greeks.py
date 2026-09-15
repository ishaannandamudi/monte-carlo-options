"""Comparison of stochastic Greek estimators with analytical Black-Scholes values."""

import math

from backend.app.models.greeks import Greek, GreekBumpConfig
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract
from backend.app.models.results import (
    FiniteDifferenceGreekResult,
    GreekComparisonEstimate,
    GreekComparisonResult,
    PathwiseGreekResult,
)
from backend.app.models.simulation import SimulationConfig, VarianceReduction
from backend.app.pricing.black_scholes_greeks import (
    analytical_greek_value,
    black_scholes_greeks,
)
from backend.app.pricing.monte_carlo_greeks import (
    finite_difference_greeks,
    pathwise_greeks,
)


def _comparison_estimate(
    result: FiniteDifferenceGreekResult | PathwiseGreekResult,
    analytical: float,
) -> GreekComparisonEstimate:
    error = abs(result.estimate - analytical)
    relative_error = error / abs(analytical) if analytical != 0.0 else None
    z_score = (
        (result.estimate - analytical) / result.standard_error
        if result.standard_error > 0.0
        else None
    )
    values = (error, relative_error, z_score)
    if not all(value is None or math.isfinite(value) for value in values):
        from backend.app.core.exceptions import NumericalError

        raise NumericalError("Greek comparison produced a non-finite diagnostic")
    return GreekComparisonEstimate(
        method=result.method,
        estimate=result.estimate,
        analytical_benchmark=analytical,
        absolute_error=error,
        relative_error=relative_error,
        standard_error=result.standard_error,
        z_score=z_score,
        confidence_interval_low=result.confidence_interval_low,
        confidence_interval_high=result.confidence_interval_high,
        runtime_ms=result.runtime_ms,
    )


def compare_greek_estimators(
    contract: OptionContract,
    market: MarketEnvironment,
    config: SimulationConfig,
    greek: Greek,
    bump_config: GreekBumpConfig | None = None,
) -> GreekComparisonResult:
    """Compare every supported stochastic estimator for one Greek."""

    analytical_result = black_scholes_greeks(contract, market)
    analytical = analytical_greek_value(analytical_result, greek)
    finite_difference = finite_difference_greeks(
        contract,
        market,
        config,
        [greek],
        bump_config,
    )[0]
    stochastic: list[FiniteDifferenceGreekResult | PathwiseGreekResult] = [finite_difference]
    if greek in {Greek.DELTA, Greek.VEGA} and config.variance_reduction in {
        VarianceReduction.NONE,
        VarianceReduction.ANTITHETIC,
    }:
        stochastic.append(pathwise_greeks(contract, market, config, [greek])[0])
    return GreekComparisonResult(
        greek=greek,
        analytical_benchmark=analytical,
        estimates=[_comparison_estimate(result, analytical) for result in stochastic],
    )
