"""Volatility-surface quality, static-arbitrage, and local-shape analytics."""

import math
from collections.abc import Sequence

import numpy as np

from backend.app.core.exceptions import VolatilitySurfaceError
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.models.volatility_surface import (
    ArbitrageIssue,
    ArbitrageIssueType,
    ATMMaturityPoint,
    ATMTermStructureResult,
    SmileAnalyticsPoint,
    SmileComparisonResult,
    SmileInterpolationConfig,
    SmileInterpolationMethod,
    SurfaceDiagnosticConfig,
    SurfaceDiagnosticsResult,
    SurfaceQualitySummary,
    VolatilitySmile,
    VolatilitySurface,
)
from backend.app.pricing.black_scholes import black_scholes_price
from backend.app.volatility_surface.core import query_smile


def _coverage(surface: VolatilitySurface) -> float:
    global_lower = min(smile.points[0].log_moneyness for smile in surface.smiles)
    global_upper = max(smile.points[-1].log_moneyness for smile in surface.smiles)
    common_lower = max(smile.points[0].log_moneyness for smile in surface.smiles)
    common_upper = min(smile.points[-1].log_moneyness for smile in surface.smiles)
    width = global_upper - global_lower
    if width <= 0.0:
        return 1.0
    return min(max((common_upper - common_lower) / width, 0.0), 1.0)


def surface_quality_summary(
    surface: VolatilitySurface,
    issues: Sequence[ArbitrageIssue] = (),
) -> SurfaceQualitySummary:
    """Return descriptive point, range, construction, coverage, and issue metrics."""

    coordinates = [point.log_moneyness for smile in surface.smiles for point in smile.points]
    calendar_count = sum(
        issue.issue_type is ArbitrageIssueType.CALENDAR_TOTAL_VARIANCE for issue in issues
    )
    butterfly_count = sum(
        issue.issue_type is ArbitrageIssueType.BUTTERFLY_CONVEXITY for issue in issues
    )
    total_points = len(coordinates)
    return SurfaceQualitySummary(
        smile_count=len(surface.smiles),
        total_iv_points=total_points,
        minimum_maturity=min(surface.maturities),
        maximum_maturity=max(surface.maturities),
        minimum_log_moneyness=min(coordinates),
        maximum_log_moneyness=max(coordinates),
        average_points_per_smile=total_points / len(surface.smiles),
        excluded_point_count=surface.construction_diagnostics.quotes_excluded,
        arbitrage_issue_count=len(issues),
        calendar_issue_count=calendar_count,
        butterfly_issue_count=butterfly_count,
        interpolation_coverage=_coverage(surface),
    )


def diagnose_strike_slice(
    *,
    strikes: Sequence[float],
    prices: Sequence[float],
    option_type: OptionType,
    maturity_years: float,
    monotonicity_tolerance: float = 1e-8,
    convexity_tolerance: float = 1e-8,
) -> tuple[list[ArbitrageIssue], int]:
    """Check monotonicity and irregular-grid convexity through adjacent slopes."""

    if len(strikes) != len(prices):
        raise VolatilitySurfaceError("strike and price sequences must have equal lengths")
    if any(not math.isfinite(value) for value in (*strikes, *prices)):
        raise VolatilitySurfaceError("strike diagnostics require finite values")
    if any(strike <= 0.0 for strike in strikes):
        raise VolatilitySurfaceError("strike diagnostics require positive strikes")
    ordered = sorted(zip(strikes, prices, strict=True))
    if any(right[0] <= left[0] for left, right in zip(ordered, ordered[1:], strict=False)):
        raise VolatilitySurfaceError("strike diagnostics require unique positive strikes")
    issues: list[ArbitrageIssue] = []
    checks = 0
    for left, right in zip(ordered, ordered[1:], strict=False):
        checks += 1
        violation = (
            right[1] > left[1] + monotonicity_tolerance
            if option_type is OptionType.CALL
            else right[1] < left[1] - monotonicity_tolerance
        )
        if violation:
            issues.append(
                ArbitrageIssue(
                    issue_type=ArbitrageIssueType.STRIKE_MONOTONICITY,
                    maturity=maturity_years,
                    strike=right[0],
                    observed_values=[left[1], right[1]],
                    tolerance=monotonicity_tolerance,
                    message=f"{option_type.value.lower()} price has the wrong strike direction",
                )
            )
    slopes = [
        (right[1] - left[1]) / (right[0] - left[0])
        for left, right in zip(ordered, ordered[1:], strict=False)
    ]
    for index, (left_slope, right_slope) in enumerate(
        zip(slopes, slopes[1:], strict=False),
        start=1,
    ):
        checks += 1
        if right_slope < left_slope - convexity_tolerance:
            issues.append(
                ArbitrageIssue(
                    issue_type=ArbitrageIssueType.BUTTERFLY_CONVEXITY,
                    maturity=maturity_years,
                    strike=ordered[index][0],
                    observed_values=[left_slope, right_slope],
                    tolerance=convexity_tolerance,
                    message="adjacent option-price slopes decrease on the irregular strike grid",
                )
            )
    return issues, checks


def _reconstructed_call_prices(smile: VolatilitySmile) -> tuple[list[float], list[float]]:
    strikes: list[float] = []
    prices: list[float] = []
    for point in smile.points:
        strikes.append(point.strike)
        prices.append(
            black_scholes_price(
                OptionContract(
                    option_type=OptionType.CALL,
                    spot=smile.spot,
                    strike=point.strike,
                    maturity_years=smile.maturity_years,
                ),
                MarketEnvironment(
                    risk_free_rate=smile.risk_free_rate,
                    volatility=point.implied_volatility,
                    dividend_yield=smile.dividend_yield,
                ),
            )
        )
    return strikes, prices


def _diagnostic_grid(
    surface: VolatilitySurface,
    config: SurfaceDiagnosticConfig,
) -> list[float]:
    if config.log_moneyness_grid is not None:
        if not config.log_moneyness_grid:
            raise VolatilitySurfaceError("diagnostic log-moneyness grid must not be empty")
        if any(not math.isfinite(value) for value in config.log_moneyness_grid):
            raise VolatilitySurfaceError("diagnostic log-moneyness grid must be finite")
        return sorted(set(config.log_moneyness_grid))
    lower = max(smile.points[0].log_moneyness for smile in surface.smiles)
    upper = min(smile.points[-1].log_moneyness for smile in surface.smiles)
    if lower > upper:
        return []
    if math.isclose(lower, upper, abs_tol=1e-14):
        return [lower]
    return [float(value) for value in np.linspace(lower, upper, config.default_grid_size)]


def diagnose_surface(
    surface: VolatilitySurface,
    config: SurfaceDiagnosticConfig | None = None,
) -> SurfaceDiagnosticsResult:
    """Report strike and calendar issues without altering any surface value."""

    policy = config or SurfaceDiagnosticConfig()
    issues: list[ArbitrageIssue] = []
    strike_checks = 0
    for smile in surface.smiles:
        strikes, prices = _reconstructed_call_prices(smile)
        strike_issues, checks = diagnose_strike_slice(
            strikes=strikes,
            prices=prices,
            option_type=OptionType.CALL,
            maturity_years=smile.maturity_years,
            monotonicity_tolerance=policy.price_tolerance,
            convexity_tolerance=policy.convexity_tolerance,
        )
        issues.extend(strike_issues)
        strike_checks += checks

    grid = _diagnostic_grid(surface, policy)
    if not grid:
        issues.append(
            ArbitrageIssue(
                issue_type=ArbitrageIssueType.INSUFFICIENT_DATA,
                tolerance=0.0,
                message="smiles have no common log-moneyness domain for calendar checks",
            )
        )
    calendar_checks = 0
    for coordinate in grid:
        previous_total_variance: float | None = None
        previous_maturity: float | None = None
        for smile in surface.smiles:
            try:
                result = query_smile(
                    smile,
                    log_moneyness=coordinate,
                    config=surface.interpolation_config,
                )
            except VolatilitySurfaceError:
                issues.append(
                    ArbitrageIssue(
                        issue_type=ArbitrageIssueType.EXTRAPOLATION_REQUIRED,
                        maturity=smile.maturity_years,
                        log_moneyness=coordinate,
                        tolerance=0.0,
                        message="calendar grid point lies outside a smile domain",
                    )
                )
                previous_total_variance = None
                previous_maturity = None
                continue
            if result.total_variance < 0.0 or not math.isfinite(result.total_variance):
                issues.append(
                    ArbitrageIssue(
                        issue_type=ArbitrageIssueType.NEGATIVE_TOTAL_VARIANCE,
                        maturity=smile.maturity_years,
                        log_moneyness=coordinate,
                        observed_values=[result.total_variance],
                        tolerance=0.0,
                        message="surface produced invalid total variance",
                    )
                )
            if previous_total_variance is not None and previous_maturity is not None:
                calendar_checks += 1
                if result.total_variance < previous_total_variance - policy.calendar_tolerance:
                    issues.append(
                        ArbitrageIssue(
                            issue_type=ArbitrageIssueType.CALENDAR_TOTAL_VARIANCE,
                            maturity=smile.maturity_years,
                            log_moneyness=coordinate,
                            observed_values=[previous_total_variance, result.total_variance],
                            tolerance=policy.calendar_tolerance,
                            message=(
                                "total variance decreases from maturity "
                                f"{previous_maturity} to {smile.maturity_years}"
                            ),
                        )
                    )
            previous_total_variance = result.total_variance
            previous_maturity = smile.maturity_years
    summary = surface_quality_summary(surface, issues)
    return SurfaceDiagnosticsResult(
        issues=issues,
        quality_summary=summary,
        strike_checks=strike_checks,
        calendar_checks=calendar_checks,
    )


def atm_forward_term_structure(surface: VolatilitySurface) -> ATMTermStructureResult:
    """Return unsmoothed ``k=0`` IV and total variance at every usable maturity."""

    points: list[ATMMaturityPoint] = []
    excluded: list[float] = []
    for smile in surface.smiles:
        try:
            result = query_smile(
                smile,
                log_moneyness=0.0,
                config=surface.interpolation_config,
            )
        except VolatilitySurfaceError:
            excluded.append(smile.maturity_years)
            continue
        points.append(
            ATMMaturityPoint(
                maturity_years=smile.maturity_years,
                implied_volatility=result.implied_volatility,
                total_variance=result.total_variance,
                exact_observation=result.exact_observation,
                interpolated=result.interpolated,
            )
        )
    return ATMTermStructureResult(points=points, excluded_maturities=excluded)


def smile_local_metrics(
    smile: VolatilitySmile,
    *,
    bump_size: float = 0.01,
    config: SmileInterpolationConfig | None = None,
) -> SmileAnalyticsPoint:
    """Estimate ``d sigma/dk`` and ``d² sigma/dk²`` by centered differences at ATM."""

    if not math.isfinite(bump_size) or bump_size <= 0.0:
        raise VolatilitySurfaceError("smile metric bump_size must be finite and positive")
    policy = config or SmileInterpolationConfig(method=SmileInterpolationMethod.PCHIP)
    atm = query_smile(smile, log_moneyness=0.0, config=policy)
    skew: float | None = None
    curvature: float | None = None
    try:
        low = query_smile(smile, log_moneyness=-bump_size, config=policy)
        high = query_smile(smile, log_moneyness=bump_size, config=policy)
        skew = (high.implied_volatility - low.implied_volatility) / (2.0 * bump_size)
        curvature = (
            high.implied_volatility - 2.0 * atm.implied_volatility + low.implied_volatility
        ) / (bump_size * bump_size)
    except VolatilitySurfaceError:
        pass
    return SmileAnalyticsPoint(
        maturity_years=smile.maturity_years,
        atm_implied_volatility=atm.implied_volatility,
        atm_exact_observation=atm.exact_observation,
        skew_dvol_dlog_moneyness=skew,
        curvature_d2vol_dlog_moneyness2=curvature,
        bump_size=bump_size,
    )


def compare_smiles(
    surface: VolatilitySurface,
    *,
    bump_size: float = 0.01,
) -> SmileComparisonResult:
    """Compare ATM, local skew, and curvature across ordered surface maturities."""

    points: list[SmileAnalyticsPoint] = []
    for smile in surface.smiles:
        try:
            points.append(
                smile_local_metrics(
                    smile,
                    bump_size=bump_size,
                    config=surface.interpolation_config,
                )
            )
        except VolatilitySurfaceError:
            continue
    return SmileComparisonResult(points=points)
