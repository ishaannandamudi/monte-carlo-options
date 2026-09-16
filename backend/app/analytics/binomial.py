"""CRR comparison, Greek diagnostics, and step-convergence analytics."""

import logging
import math
from collections.abc import Sequence

from backend.app.models.binomial import (
    BinomialComparison,
    BinomialConvergencePoint,
    BinomialConvergenceResult,
    BinomialTreeConfig,
    OddEvenDiagnostic,
    StepParity,
    TreeGreekComparison,
    TreeModel,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract
from backend.app.pricing.binomial import binomial_price
from backend.app.pricing.black_scholes_greeks import black_scholes_greeks

logger = logging.getLogger(__name__)


def _relative_error(absolute_error: float, benchmark: float) -> float | None:
    return absolute_error / abs(benchmark) if benchmark != 0.0 else None


def _greek_comparison(
    tree_value: float | None,
    analytical_value: float,
) -> TreeGreekComparison:
    if tree_value is None:
        return TreeGreekComparison(
            tree_value=None,
            analytical_value=analytical_value,
            signed_error=None,
            absolute_error=None,
            relative_error=None,
        )
    signed = tree_value - analytical_value
    absolute = abs(signed)
    return TreeGreekComparison(
        tree_value=tree_value,
        analytical_value=analytical_value,
        signed_error=signed,
        absolute_error=absolute,
        relative_error=_relative_error(absolute, analytical_value),
    )


def compare_binomial_to_black_scholes(
    contract: OptionContract,
    market: MarketEnvironment,
    config: BinomialTreeConfig | None = None,
) -> BinomialComparison:
    """Compare one canonical CRR price and its native Greeks with analytical values."""

    result = binomial_price(contract, market, config)
    analytical = black_scholes_greeks(contract, market)
    return BinomialComparison(
        binomial=result,
        black_scholes_price=analytical.option_price,
        signed_error=result.signed_error,
        absolute_error=result.absolute_error,
        relative_error=result.relative_error,
        delta=_greek_comparison(result.delta, analytical.delta),
        gamma=_greek_comparison(result.gamma, analytical.gamma),
        theta=_greek_comparison(result.theta, analytical.theta),
    )


def _step_parity(steps: int) -> StepParity:
    return StepParity.EVEN if steps % 2 == 0 else StepParity.ODD


def run_binomial_convergence(
    contract: OptionContract,
    market: MarketEnvironment,
    step_counts: Sequence[int],
    *,
    model: TreeModel = TreeModel.CRR,
    probability_tolerance: float = 1e-12,
    arbitrage_tolerance: float = 1e-10,
) -> BinomialConvergenceResult:
    """Evaluate ordered unique step counts without assuming monotone convergence."""

    if not step_counts:
        raise ValueError("at least one binomial step count is required")
    if any(count <= 0 for count in step_counts):
        raise ValueError("binomial step counts must be positive")
    if len(set(step_counts)) != len(step_counts):
        raise ValueError("binomial step counts must be unique")
    ordered_counts = sorted(step_counts)
    analytical = black_scholes_greeks(contract, market)
    points: list[BinomialConvergencePoint] = []
    logger.info("binomial_convergence_start points=%d model=%s", len(ordered_counts), model.value)
    for steps in ordered_counts:
        result = binomial_price(
            contract,
            market,
            BinomialTreeConfig(
                steps=steps,
                model=model,
                probability_tolerance=probability_tolerance,
                arbitrage_tolerance=arbitrage_tolerance,
            ),
        )
        points.append(
            BinomialConvergencePoint(
                steps=steps,
                step_parity=_step_parity(steps),
                tree_price=result.price,
                black_scholes_price=analytical.option_price,
                signed_error=result.signed_error,
                absolute_error=result.absolute_error,
                relative_error=result.relative_error,
                runtime_ms=result.runtime_ms,
                delta=result.delta,
                delta_error=(None if result.delta is None else result.delta - analytical.delta),
                gamma=result.gamma,
                gamma_error=(None if result.gamma is None else result.gamma - analytical.gamma),
                theta=result.theta,
                theta_error=(None if result.theta is None else result.theta - analytical.theta),
            )
        )
    parity_diagnostics: list[OddEvenDiagnostic] = []
    for lower, upper in zip(points, points[1:], strict=False):
        if upper.steps != lower.steps + 1:
            continue
        average = 0.5 * (lower.tree_price + upper.tree_price)
        if not math.isfinite(average):
            continue
        parity_diagnostics.append(
            OddEvenDiagnostic(
                lower_steps=lower.steps,
                upper_steps=upper.steps,
                lower_parity=lower.step_parity,
                upper_parity=upper.step_parity,
                price_difference=upper.tree_price - lower.tree_price,
                adjacent_average=average,
                average_absolute_error=abs(average - analytical.option_price),
            )
        )
    logger.info(
        "binomial_convergence_complete points=%d adjacent_pairs=%d",
        len(points),
        len(parity_diagnostics),
    )
    return BinomialConvergenceResult(
        model=model,
        black_scholes_price=analytical.option_price,
        analytical_delta=analytical.delta,
        analytical_gamma=analytical.gamma,
        analytical_theta=analytical.theta,
        points=points,
        odd_even_diagnostics=parity_diagnostics,
    )
