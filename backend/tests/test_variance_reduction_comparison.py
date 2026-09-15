import pytest

from backend.app.analytics.variance_reduction import compare_variance_reduction_methods
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.models.simulation import VarianceReduction


def inputs() -> tuple[OptionContract, MarketEnvironment]:
    return (
        OptionContract(
            option_type=OptionType.CALL,
            spot=100.0,
            strike=100.0,
            maturity_years=1.0,
        ),
        MarketEnvironment(risk_free_rate=0.05, volatility=0.2, dividend_yield=0.01),
    )


def test_comparison_contains_every_method_and_none_baseline() -> None:
    contract, market = inputs()
    comparison = compare_variance_reduction_methods(contract, market, 20_000, root_seed=42)
    assert comparison.baseline_method is VarianceReduction.NONE
    assert [point.method for point in comparison.points] == list(VarianceReduction)
    baseline = comparison.points[0]
    assert baseline.relative_variance == pytest.approx(1.0)
    assert baseline.relative_efficiency == pytest.approx(1.0)


def test_representative_controls_reduce_fixed_budget_estimator_variance() -> None:
    contract, market = inputs()
    comparison = compare_variance_reduction_methods(contract, market, 80_000, root_seed=42)
    points = {point.method: point for point in comparison.points}
    assert (
        points[VarianceReduction.CONTROL_VARIATE].estimator_variance
        < points[VarianceReduction.NONE].estimator_variance
    )
    assert (
        points[VarianceReduction.ANTITHETIC_CONTROL_VARIATE].estimator_variance
        < points[VarianceReduction.ANTITHETIC].estimator_variance
    )
    assert points[VarianceReduction.CONTROL_VARIATE].variance_reduction_ratio > 1.0
    assert points[VarianceReduction.ANTITHETIC_CONTROL_VARIATE].variance_reduction_ratio > 1.0


def test_comparison_reproducible_except_runtime() -> None:
    contract, market = inputs()
    first = compare_variance_reduction_methods(contract, market, 10_000, root_seed=91)
    second = compare_variance_reduction_methods(contract, market, 10_000, root_seed=91)
    for left, right in zip(first.points, second.points, strict=True):
        assert left.model_dump(exclude={"runtime_ms"}) == right.model_dump(exclude={"runtime_ms"})


def test_deterministic_comparison_represents_ratios_safely() -> None:
    contract, _ = inputs()
    comparison = compare_variance_reduction_methods(
        contract,
        MarketEnvironment(risk_free_rate=-0.01, volatility=0.0, dividend_yield=0.02),
        10,
    )
    assert all(point.estimator_variance == 0.0 for point in comparison.points)
    assert all(point.relative_variance is None for point in comparison.points)
    assert all(point.relative_efficiency is None for point in comparison.points)


def test_comparison_rejects_invalid_library_inputs() -> None:
    contract, market = inputs()
    with pytest.raises(ValueError):
        compare_variance_reduction_methods(contract, market, 0)
    with pytest.raises(ValueError):
        compare_variance_reduction_methods(contract, market, 10, confidence_level=1.0)
