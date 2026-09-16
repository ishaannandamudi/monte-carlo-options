"""CRR comparison, Greek diagnostics, and step-convergence analytics."""

import logging
import math
from collections.abc import Sequence

from backend.app.models.binomial import (
    AmericanBinomialComparison,
    AmericanBinomialConvergencePoint,
    AmericanBinomialConvergenceResult,
    AmericanBinomialPricingResult,
    AmericanTreeGreekDifference,
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
from backend.app.models.option import ExerciseStyle, OptionContract
from backend.app.pricing.binomial import american_binomial_price, binomial_price
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


def _american_greek_difference(
    american_value: float | None,
    european_value: float | None,
) -> AmericanTreeGreekDifference:
    difference = (
        None
        if american_value is None or european_value is None
        else american_value - european_value
    )
    return AmericanTreeGreekDifference(
        american_value=american_value,
        european_value=european_value,
        difference=difference,
    )


def compare_binomial_to_black_scholes(
    contract: OptionContract,
    market: MarketEnvironment,
    config: BinomialTreeConfig | None = None,
) -> BinomialComparison | AmericanBinomialComparison:
    """Compare European CRR to BSM or American CRR to its European comparators."""

    if contract.exercise_style is ExerciseStyle.AMERICAN:
        tree_config = config or BinomialTreeConfig()
        american = american_binomial_price(contract, market, tree_config)
        european_contract = contract.model_copy(update={"exercise_style": ExerciseStyle.EUROPEAN})
        european = binomial_price(european_contract, market, tree_config)
        if isinstance(european, AmericanBinomialPricingResult):
            raise RuntimeError("European comparator unexpectedly returned an American result")
        return AmericanBinomialComparison(
            american=american,
            european=european,
            black_scholes_european_price=american.black_scholes_european_price,
            early_exercise_premium=american.early_exercise_premium,
            delta=_american_greek_difference(american.delta, european.delta),
            gamma=_american_greek_difference(american.gamma, european.gamma),
            theta=_american_greek_difference(american.theta, european.theta),
        )

    result = binomial_price(contract, market, config)
    if isinstance(result, AmericanBinomialPricingResult):
        raise RuntimeError("European comparison unexpectedly returned an American result")
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
    exercise_tolerance: float = 1e-12,
) -> BinomialConvergenceResult | AmericanBinomialConvergenceResult:
    """Evaluate ordered unique step counts without assuming monotone convergence."""

    if not step_counts:
        raise ValueError("at least one binomial step count is required")
    if any(count <= 0 for count in step_counts):
        raise ValueError("binomial step counts must be positive")
    if len(set(step_counts)) != len(step_counts):
        raise ValueError("binomial step counts must be unique")
    ordered_counts = sorted(step_counts)
    if contract.exercise_style is ExerciseStyle.AMERICAN:
        european_contract = contract.model_copy(update={"exercise_style": ExerciseStyle.EUROPEAN})
        analytical = black_scholes_greeks(european_contract, market)
        american_points: list[AmericanBinomialConvergencePoint] = []
        logger.info(
            "american_binomial_convergence_start points=%d model=%s",
            len(ordered_counts),
            model.value,
        )
        for steps in ordered_counts:
            american_result = american_binomial_price(
                contract,
                market,
                BinomialTreeConfig(
                    steps=steps,
                    model=model,
                    probability_tolerance=probability_tolerance,
                    arbitrage_tolerance=arbitrage_tolerance,
                    exercise_tolerance=exercise_tolerance,
                ),
            )
            american_points.append(
                AmericanBinomialConvergencePoint(
                    steps=steps,
                    step_parity=_step_parity(steps),
                    american_price=american_result.price,
                    european_tree_price=american_result.european_tree_price,
                    early_exercise_premium=american_result.early_exercise_premium,
                    delta=american_result.delta,
                    gamma=american_result.gamma,
                    theta=american_result.theta,
                    runtime_ms=american_result.runtime_ms,
                    exercise_node_count=american_result.exercise_node_count,
                    earliest_exercise_time=american_result.earliest_exercise_time,
                )
            )
        logger.info("american_binomial_convergence_complete points=%d", len(american_points))
        return AmericanBinomialConvergenceResult(
            model=model,
            black_scholes_european_price=analytical.option_price,
            points=american_points,
        )

    analytical = black_scholes_greeks(contract, market)
    points: list[BinomialConvergencePoint] = []
    logger.info("binomial_convergence_start points=%d model=%s", len(ordered_counts), model.value)
    for steps in ordered_counts:
        european_result = binomial_price(
            contract,
            market,
            BinomialTreeConfig(
                steps=steps,
                model=model,
                probability_tolerance=probability_tolerance,
                arbitrage_tolerance=arbitrage_tolerance,
                exercise_tolerance=exercise_tolerance,
            ),
        )
        if isinstance(european_result, AmericanBinomialPricingResult):
            raise RuntimeError("European convergence unexpectedly returned an American result")
        points.append(
            BinomialConvergencePoint(
                steps=steps,
                step_parity=_step_parity(steps),
                tree_price=european_result.price,
                black_scholes_price=analytical.option_price,
                signed_error=european_result.signed_error,
                absolute_error=european_result.absolute_error,
                relative_error=european_result.relative_error,
                runtime_ms=european_result.runtime_ms,
                delta=european_result.delta,
                delta_error=(
                    None
                    if european_result.delta is None
                    else european_result.delta - analytical.delta
                ),
                gamma=european_result.gamma,
                gamma_error=(
                    None
                    if european_result.gamma is None
                    else european_result.gamma - analytical.gamma
                ),
                theta=european_result.theta,
                theta_error=(
                    None
                    if european_result.theta is None
                    else european_result.theta - analytical.theta
                ),
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
