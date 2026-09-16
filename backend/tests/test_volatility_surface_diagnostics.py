import pytest

from backend.app.analytics.volatility_surface import (
    atm_forward_term_structure,
    compare_smiles,
    diagnose_strike_slice,
    diagnose_surface,
)
from backend.app.models.option import OptionType
from backend.app.models.volatility_surface import ArbitrageIssueType, SurfaceDiagnosticConfig
from backend.app.volatility_surface.synthetic import generate_synthetic_surface


def test_increasing_total_variance_has_no_calendar_issue() -> None:
    surface = generate_synthetic_surface(
        maturities=[0.25, 0.5, 1.0],
        log_moneyness=[-0.2, -0.1, 0.0, 0.1, 0.2],
        base_volatility=0.2,
        term_slope=0.01,
    )
    result = diagnose_surface(surface)
    assert all(
        issue.issue_type is not ArbitrageIssueType.CALENDAR_TOTAL_VARIANCE
        for issue in result.issues
    )
    assert result.calendar_checks > 0


def test_calendar_total_variance_decrease_is_detected() -> None:
    surface = generate_synthetic_surface(
        maturities=[0.25, 1.0],
        log_moneyness=[-0.2, -0.1, 0.0, 0.1, 0.2],
        base_volatility=0.5,
        term_slope=-0.3,
    )
    result = diagnose_surface(surface, SurfaceDiagnosticConfig(default_grid_size=5))
    issues = [
        issue
        for issue in result.issues
        if issue.issue_type is ArbitrageIssueType.CALENDAR_TOTAL_VARIANCE
    ]
    assert issues
    assert result.quality_summary.calendar_issue_count == len(issues)


def test_irregular_strike_butterfly_violation_is_detected() -> None:
    issues, checks = diagnose_strike_slice(
        strikes=[80.0, 95.0, 110.0, 140.0],
        prices=[25.0, 15.0, 14.0, 1.0],
        option_type=OptionType.CALL,
        maturity_years=1.0,
    )
    assert checks == 5
    assert any(issue.issue_type is ArbitrageIssueType.BUTTERFLY_CONVEXITY for issue in issues)


def test_call_and_put_monotonicity_violations_are_detected() -> None:
    calls, _ = diagnose_strike_slice(
        strikes=[90.0, 100.0, 110.0],
        prices=[15.0, 16.0, 5.0],
        option_type=OptionType.CALL,
        maturity_years=1.0,
    )
    puts, _ = diagnose_strike_slice(
        strikes=[90.0, 100.0, 110.0],
        prices=[5.0, 4.0, 15.0],
        option_type=OptionType.PUT,
        maturity_years=1.0,
    )
    assert any(issue.issue_type is ArbitrageIssueType.STRIKE_MONOTONICITY for issue in calls)
    assert any(issue.issue_type is ArbitrageIssueType.STRIKE_MONOTONICITY for issue in puts)


def test_atm_term_structure_and_local_metrics() -> None:
    surface = generate_synthetic_surface(
        maturities=[0.5, 1.0, 2.0],
        log_moneyness=[-0.1, -0.01, 0.0, 0.01, 0.1],
        base_volatility=0.2,
        skew=-0.15,
        curvature=0.5,
        term_slope=0.01,
    )
    term = atm_forward_term_structure(surface)
    comparison = compare_smiles(surface, bump_size=0.01)
    assert len(term.points) == 3
    assert term.points[0].implied_volatility == pytest.approx(0.205)
    assert term.points[0].total_variance == pytest.approx(0.205**2 * 0.5)
    for point in comparison.points:
        assert point.skew_dvol_dlog_moneyness == pytest.approx(-0.15)
        assert point.curvature_d2vol_dlog_moneyness2 == pytest.approx(1.0)


def test_quality_summary_counts_points_ranges_and_coverage() -> None:
    surface = generate_synthetic_surface(
        maturities=[0.5, 1.0],
        log_moneyness=[-0.2, 0.0, 0.2],
    )
    result = diagnose_surface(surface)
    summary = result.quality_summary
    assert summary.smile_count == 2
    assert summary.total_iv_points == 6
    assert summary.average_points_per_smile == 3.0
    assert summary.minimum_maturity == 0.5
    assert summary.maximum_maturity == 1.0
    assert summary.interpolation_coverage == 1.0
