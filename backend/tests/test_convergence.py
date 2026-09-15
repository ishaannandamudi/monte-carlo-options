import math

import pytest

from backend.app.analytics.convergence import estimate_standard_error_slope, run_convergence
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.models.simulation import VarianceReduction
from backend.app.simulation.random import spawn_child_seeds


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


def test_child_seed_policy_is_independent_and_reproducible() -> None:
    first = spawn_child_seeds(42, 5)
    second = spawn_child_seeds(42, 5)
    assert first == second
    assert len(set(first)) == 5
    assert first != spawn_child_seeds(43, 5)


def test_empirical_slope_exact_synthetic_case_and_too_few_points() -> None:
    counts = [100, 400, 1_600, 6_400]
    errors = [count**-0.5 for count in counts]
    slope, used = estimate_standard_error_slope(counts, errors)
    assert slope == pytest.approx(-0.5, abs=1e-12)
    assert used == 4
    assert estimate_standard_error_slope([100, 200], [0.0, math.nan]) == (None, 0)


def test_convergence_is_reproducible_except_wall_clock_runtime() -> None:
    contract, market = inputs()
    first = run_convergence(contract, market, [2_000, 8_000, 32_000], root_seed=177)
    second = run_convergence(contract, market, [2_000, 8_000, 32_000], root_seed=177)
    assert first.black_scholes_price == second.black_scholes_price
    assert first.estimated_standard_error_slope == second.estimated_standard_error_slope
    for left, right in zip(first.points, second.points, strict=True):
        assert left.model_dump(exclude={"runtime_ms"}) == right.model_dump(exclude={"runtime_ms"})


def test_empirical_monte_carlo_slope_is_near_theoretical_rate() -> None:
    contract, market = inputs()
    result = run_convergence(
        contract,
        market,
        [2_000, 8_000, 32_000, 128_000],
        root_seed=904,
        variance_reduction=VarianceReduction.ANTITHETIC,
    )
    assert result.estimated_standard_error_slope is not None
    assert -0.7 < result.estimated_standard_error_slope < -0.3
    assert result.slope_points_used == 4


def test_deterministic_convergence_has_no_slope() -> None:
    contract, _ = inputs()
    result = run_convergence(
        contract,
        MarketEnvironment(risk_free_rate=0.05, volatility=0.0),
        [10, 20, 40],
    )
    assert result.estimated_standard_error_slope is None
    assert result.slope_points_used == 0


def test_convergence_rejects_invalid_library_inputs() -> None:
    contract, market = inputs()
    with pytest.raises(ValueError):
        run_convergence(contract, market, [])
    with pytest.raises(ValueError):
        run_convergence(contract, market, [100, 0])


@pytest.mark.parametrize("method", list(VarianceReduction))
def test_convergence_supports_every_variance_reduction_method(
    method: VarianceReduction,
) -> None:
    contract, market = inputs()
    result = run_convergence(
        contract,
        market,
        [1_000, 4_000, 16_000],
        root_seed=202,
        variance_reduction=method,
    )
    assert result.variance_reduction is method
    assert len(result.points) == 3
    assert all(point.standard_error > 0.0 for point in result.points)
