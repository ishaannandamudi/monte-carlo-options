"""CRR comparisons and reproducible LSM path, time-grid, and basis experiments."""

from collections.abc import Sequence

from backend.app.core.exceptions import LongstaffSchwartzError
from backend.app.models.binomial import (
    AmericanBinomialPricingResult,
    BinomialTreeConfig,
    ExerciseBoundaryResult,
)
from backend.app.models.longstaff_schwartz import (
    BasisFamily,
    LongstaffSchwartzConfig,
    LongstaffSchwartzResult,
    LSMBasisComparisonResult,
    LSMBoundaryComparisonPoint,
    LSMBoundaryPoint,
    LSMComparison,
    LSMConvergenceMode,
    LSMConvergenceResult,
    LSMResearchPoint,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract
from backend.app.pricing.binomial import american_binomial_price, american_exercise_boundary
from backend.app.pricing.longstaff_schwartz import longstaff_schwartz_price
from backend.app.simulation.random import spawn_child_seeds


def compare_exercise_boundaries(
    points: Sequence[LSMBoundaryPoint],
    crr: ExerciseBoundaryResult,
    lsm_steps: int,
) -> tuple[LSMBoundaryComparisonPoint, ...]:
    """Match rational grid dates exactly, omitting unsupported or nonshared dates."""

    if lsm_steps <= 0:
        raise LongstaffSchwartzError("LSM boundary grid must have positive steps")
    tree_points = {point.step: point for point in crr.points}
    comparisons = []
    for point in points:
        step, remainder = divmod(point.time_index * crr.steps, lsm_steps)
        if remainder or step not in tree_points:
            continue
        benchmark = tree_points[step]
        difference = point.estimated_boundary_stock - benchmark.boundary_stock_price
        comparisons.append(
            LSMBoundaryComparisonPoint(
                time=point.time,
                lsm_time_index=point.time_index,
                crr_step=step,
                lsm_boundary_stock=point.estimated_boundary_stock,
                crr_boundary_stock=benchmark.boundary_stock_price,
                signed_difference=difference,
                absolute_difference=abs(difference),
                exercise_observations=point.exercise_observations,
                sample_support=point.sample_support,
            )
        )
    return tuple(comparisons)


def compare_longstaff_schwartz(
    contract: OptionContract,
    market: MarketEnvironment,
    config: LongstaffSchwartzConfig | None = None,
    tree: BinomialTreeConfig | None = None,
    *,
    include_boundary_comparison: bool = False,
) -> LSMComparison:
    config = config or LongstaffSchwartzConfig()
    tree = tree or BinomialTreeConfig(steps=1_000)
    if include_boundary_comparison:
        config = config.model_copy(update={"store_boundary_diagnostics": True})
    result = longstaff_schwartz_price(contract, market, config)
    crr = american_binomial_price(contract, market, tree)
    boundaries = (
        compare_exercise_boundaries(
            result.boundary_diagnostics,
            american_exercise_boundary(contract, market, tree),
            config.num_time_steps,
        )
        if include_boundary_comparison
        else None
    )
    signed = result.price - crr.price
    return LSMComparison(
        longstaff_schwartz=result,
        crr_american=crr,
        signed_error=signed,
        absolute_error=abs(signed),
        relative_error=abs(signed) / crr.price if crr.price else None,
        normalized_error=signed / result.standard_error if result.standard_error > 0.0 else None,
        european_reference_price=crr.black_scholes_european_price,
        lsm_early_exercise_premium=result.price - crr.black_scholes_european_price,
        crr_early_exercise_premium=crr.early_exercise_premium,
        european_reference_below_confidence_high=(
            crr.black_scholes_european_price <= result.confidence_interval_high
        ),
        boundary_comparison=boundaries,
    )


def convergence_configs(
    config: LongstaffSchwartzConfig,
    values: Sequence[int],
    mode: LSMConvergenceMode,
) -> list[LongstaffSchwartzConfig]:
    """Validate the entire experiment before any pricing or allocation."""

    if not values or any(value <= 0 for value in values) or len(set(values)) != len(values):
        raise LongstaffSchwartzError("convergence values must be nonempty, positive, and unique")
    if mode is LSMConvergenceMode.TIME_STEPS and config.exercise_steps is not None:
        raise LongstaffSchwartzError(
            "time-step convergence requires the default every-step schedule"
        )
    field = "num_paths" if mode is LSMConvergenceMode.PATH_COUNT else "num_time_steps"
    seeds = spawn_child_seeds(config.seed, len(values))
    return [
        LongstaffSchwartzConfig.model_validate({**config.model_dump(), field: value, "seed": seed})
        for value, seed in zip(sorted(values), seeds, strict=True)
    ]


def basis_configs(
    config: LongstaffSchwartzConfig,
    families: Sequence[BasisFamily],
    degrees: Sequence[int],
) -> list[LongstaffSchwartzConfig]:
    if (
        not families
        or not degrees
        or len(set(families)) != len(families)
        or len(set(degrees)) != len(degrees)
    ):
        raise LongstaffSchwartzError("basis families and degrees must be nonempty and unique")
    return [
        LongstaffSchwartzConfig.model_validate(
            {
                **config.model_dump(),
                "basis_family": family,
                "basis_degree": degree,
            }
        )
        for family in families
        for degree in degrees
    ]


def _research_point(
    result: LongstaffSchwartzResult,
    crr: AmericanBinomialPricingResult,
) -> LSMResearchPoint:
    return LSMResearchPoint(
        num_paths=result.num_paths,
        regression_paths=result.regression_paths,
        num_time_steps=result.num_time_steps,
        seed=result.seed,
        basis_family=result.basis_family,
        basis_degree=result.basis_degree,
        price=result.price,
        standard_error=result.standard_error,
        confidence_interval_low=result.confidence_interval_low,
        confidence_interval_high=result.confidence_interval_high,
        crr_price=crr.price,
        absolute_error=abs(result.price - crr.price),
        runtime_ms=result.runtime_ms,
        early_exercise_fraction=result.early_exercise_fraction,
        regression_count=result.regression_count,
        fallback_count=result.regression_fallback_count,
        condition_warning_count=result.condition_warning_count,
        max_condition_number=result.max_condition_number,
    )


def run_lsm_convergence(
    contract: OptionContract,
    market: MarketEnvironment,
    config: LongstaffSchwartzConfig,
    values: Sequence[int],
    mode: LSMConvergenceMode,
    tree: BinomialTreeConfig | None = None,
) -> LSMConvergenceResult:
    configs = convergence_configs(config, values, mode)
    benchmark = american_binomial_price(contract, market, tree or BinomialTreeConfig(steps=1_000))
    return LSMConvergenceResult(
        mode=mode,
        root_seed=config.seed,
        crr_steps=benchmark.steps,
        points=[
            _research_point(longstaff_schwartz_price(contract, market, item), benchmark)
            for item in configs
        ],
    )


def compare_lsm_bases(
    contract: OptionContract,
    market: MarketEnvironment,
    config: LongstaffSchwartzConfig,
    families: Sequence[BasisFamily] = (BasisFamily.POWER, BasisFamily.LAGUERRE),
    degrees: Sequence[int] = (1, 2, 3, 4),
    tree: BinomialTreeConfig | None = None,
) -> LSMBasisComparisonResult:
    """Use the same random streams for every candidate to isolate basis effects."""

    configs = basis_configs(config, families, degrees)
    benchmark = american_binomial_price(contract, market, tree or BinomialTreeConfig(steps=1_000))
    return LSMBasisComparisonResult(
        seed=config.seed,
        crr_steps=benchmark.steps,
        points=[
            _research_point(longstaff_schwartz_price(contract, market, item), benchmark)
            for item in configs
        ],
    )
