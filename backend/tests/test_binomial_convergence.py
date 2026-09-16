import pytest

from backend.app.analytics.binomial import run_binomial_convergence
from backend.app.models.binomial import StepParity
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract, OptionType


def inputs() -> tuple[OptionContract, MarketEnvironment]:
    return (
        OptionContract(
            option_type=OptionType.CALL,
            spot=100.0,
            strike=100.0,
            maturity_years=1.0,
        ),
        MarketEnvironment(risk_free_rate=0.05, volatility=0.2),
    )


def test_convergence_improves_in_aggregate_without_monotonic_assumption() -> None:
    contract, market = inputs()
    result = run_binomial_convergence(contract, market, [10, 25, 50, 100, 250, 500])
    assert [point.steps for point in result.points] == [10, 25, 50, 100, 250, 500]
    early_error = max(point.absolute_error for point in result.points[:2])
    late_error = max(point.absolute_error for point in result.points[-2:])
    assert late_error < early_error
    assert result.points[-1].delta_error == pytest.approx(
        result.points[-1].delta - result.analytical_delta  # type: ignore[operator]
    )


def test_odd_even_oscillation_and_adjacent_average_are_explicit() -> None:
    contract, market = inputs()
    result = run_binomial_convergence(contract, market, [9, 10, 11, 12])
    signed_errors = [point.signed_error for point in result.points]
    assert signed_errors[0] > 0.0
    assert signed_errors[1] < 0.0
    assert signed_errors[2] > 0.0
    assert signed_errors[3] < 0.0
    assert [point.step_parity for point in result.points] == [
        StepParity.ODD,
        StepParity.EVEN,
        StepParity.ODD,
        StepParity.EVEN,
    ]
    assert len(result.odd_even_diagnostics) == 3
    first = result.odd_even_diagnostics[0]
    assert first.adjacent_average == pytest.approx(
        0.5 * (result.points[0].tree_price + result.points[1].tree_price)
    )


def test_convergence_rejects_empty_nonpositive_and_duplicate_steps() -> None:
    contract, market = inputs()
    with pytest.raises(ValueError, match="at least one"):
        run_binomial_convergence(contract, market, [])
    with pytest.raises(ValueError, match="positive"):
        run_binomial_convergence(contract, market, [10, 0])
    with pytest.raises(ValueError, match="unique"):
        run_binomial_convergence(contract, market, [10, 10])
