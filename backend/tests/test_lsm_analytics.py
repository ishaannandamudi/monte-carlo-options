import pytest

from backend.app.analytics.longstaff_schwartz import (
    basis_configs,
    compare_exercise_boundaries,
    compare_longstaff_schwartz,
    compare_lsm_bases,
    convergence_configs,
    run_lsm_convergence,
)
from backend.app.core.exceptions import LongstaffSchwartzError
from backend.app.models.binomial import BinomialTreeConfig
from backend.app.models.longstaff_schwartz import (
    BasisFamily,
    LongstaffSchwartzConfig,
    LSMBoundaryPoint,
    LSMConvergenceMode,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import ExerciseStyle, OptionContract, OptionType
from backend.app.pricing.binomial import american_exercise_boundary


def inputs() -> tuple[OptionContract, MarketEnvironment, LongstaffSchwartzConfig]:
    return (
        OptionContract(
            option_type=OptionType.PUT,
            exercise_style=ExerciseStyle.AMERICAN,
            spot=100,
            strike=100,
            maturity_years=1,
        ),
        MarketEnvironment(risk_free_rate=0.05, volatility=0.2),
        LongstaffSchwartzConfig(num_paths=2000, num_time_steps=10),
    )


def test_crr_comparison_errors_premiums_and_zero_se() -> None:
    c, m, config = inputs()
    result = compare_longstaff_schwartz(c, m, config, BinomialTreeConfig(steps=100))
    lsm, crr = result.longstaff_schwartz, result.crr_american
    assert result.signed_error == lsm.price - crr.price
    assert result.absolute_error == abs(result.signed_error)
    assert result.normalized_error == pytest.approx(result.signed_error / lsm.standard_error)
    assert result.lsm_early_exercise_premium == lsm.price - result.european_reference_price
    assert result.crr_early_exercise_premium == crr.price - crr.european_tree_price
    zero = compare_longstaff_schwartz(c.model_copy(update={"maturity_years": 0}), m, config)
    assert zero.normalized_error is None and zero.relative_error is None


@pytest.mark.parametrize(
    "mode,values",
    [
        (LSMConvergenceMode.PATH_COUNT, [400, 1000, 2000]),
        (LSMConvergenceMode.TIME_STEPS, [5, 10, 15]),
    ],
)
def test_convergence_modes_have_independent_reproducible_children(
    mode: LSMConvergenceMode, values: list[int]
) -> None:
    c, m, config = inputs()
    configs = convergence_configs(config, values, mode)
    assert len({item.seed for item in configs}) == len(values)
    assert configs == convergence_configs(config, list(reversed(values)), mode)
    result = run_lsm_convergence(c, m, config, values, mode, BinomialTreeConfig(steps=100))
    assert len(result.points) == 3
    assert all(point.standard_error >= 0 and point.regression_count > 0 for point in result.points)
    assert [point.seed for point in result.points] == [item.seed for item in configs]
    assert len({point.crr_price for point in result.points}) == 1


def test_basis_comparison_shares_streams_and_covers_families_and_degrees() -> None:
    c, m, config = inputs()
    families = list(BasisFamily)
    result = compare_lsm_bases(c, m, config, families, [1, 2, 3], BinomialTreeConfig(steps=100))
    assert len(result.points) == 6
    assert {point.seed for point in result.points} == {config.seed}
    assert [point.basis_degree for point in result.points] == [1, 2, 3, 1, 2, 3]
    assert {point.basis_family for point in result.points} == set(families)
    assert all(point.max_condition_number is not None for point in result.points)
    # Unweighted polynomial families span the same space at each degree.
    for power, laguerre in zip(result.points[:3], result.points[3:], strict=True):
        assert power.price == pytest.approx(laguerre.price, abs=1e-10)


def test_analytics_validate_before_computation() -> None:
    _, _, config = inputs()
    for values in ([], [0], [1, 1]):
        with pytest.raises(LongstaffSchwartzError):
            convergence_configs(config, values, LSMConvergenceMode.PATH_COUNT)
    with pytest.raises(LongstaffSchwartzError, match="schedule"):
        convergence_configs(
            config.model_copy(update={"exercise_steps": (5, 10)}),
            [5, 10],
            LSMConvergenceMode.TIME_STEPS,
        )
    with pytest.raises(LongstaffSchwartzError):
        basis_configs(config, [BasisFamily.POWER], [1, 1])


def test_boundary_comparison_matches_only_shared_supported_dates() -> None:
    c, m, _ = inputs()
    tree = american_exercise_boundary(c, m, BinomialTreeConfig(steps=10))
    benchmark = tree.points[-1]
    point = LSMBoundaryPoint(
        time_index=benchmark.step * 2,
        time=benchmark.time_from_valuation,
        time_to_maturity=benchmark.time_to_maturity,
        estimated_boundary_stock=benchmark.boundary_stock_price + 0.25,
        option_type=OptionType.PUT,
        exercise_region_direction="LOW_STOCK",
        exercise_observations=7,
        sample_support=0.07,
    )
    nonshared = point.model_copy(update={"time_index": benchmark.step * 2 + 1})
    unmatched = point.model_copy(update={"time_index": 0})
    result = compare_exercise_boundaries([point, nonshared, unmatched], tree, 20)
    assert len(result) == 1
    assert result[0].crr_step == benchmark.step
    assert result[0].signed_difference == result[0].absolute_difference == 0.25
    assert result[0].exercise_observations == 7 and result[0].sample_support == 0.07
    assert compare_exercise_boundaries([], tree, 20) == ()


def test_optional_boundary_comparison_and_output_cap() -> None:
    c, m, config = inputs()
    config = config.model_copy(
        update={"max_diagnostic_dates": 2, "store_boundary_diagnostics": False}
    )
    tree = BinomialTreeConfig(steps=100)
    plain = compare_longstaff_schwartz(c, m, config, tree)
    compared = compare_longstaff_schwartz(c, m, config, tree, include_boundary_comparison=True)
    assert plain.boundary_comparison is None
    assert plain.longstaff_schwartz.price == compared.longstaff_schwartz.price
    assert compared.boundary_comparison is not None
    assert 0 < len(compared.boundary_comparison) <= 2
    assert compared.longstaff_schwartz.boundary_diagnostics_truncated
    call = compare_longstaff_schwartz(
        c.model_copy(update={"option_type": OptionType.CALL}),
        m,
        config,
        tree,
        include_boundary_comparison=True,
    )
    assert call.boundary_comparison == ()
