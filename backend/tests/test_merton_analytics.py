import math

import pytest

from backend.app.analytics.merton import (
    compare_merton,
    merton_implied_volatility_smile,
    merton_jump_diagnostics,
    merton_parameter_sensitivity,
    merton_price_objective,
    run_merton_convergence,
    run_merton_series_convergence,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.merton import (
    MertonJumpParameters,
    MertonObjectiveObservation,
    MertonParameterName,
    MertonSimulationConfig,
)
from backend.app.models.option import ExerciseStyle, OptionContract, OptionType
from backend.app.pricing.merton import merton_analytical_price

CONTRACT = OptionContract(
    option_type=OptionType.CALL,
    exercise_style=ExerciseStyle.EUROPEAN,
    spot=100.0,
    strike=100.0,
    maturity_years=0.25,
)
MARKET = MarketEnvironment(risk_free_rate=0.02, dividend_yield=0.01, volatility=0.16)
PARAMETERS = MertonJumpParameters(jump_intensity=1.2, jump_mean=-0.12, jump_volatility=0.28)


def test_comparison_is_reproducible_and_statistically_consistent() -> None:
    config = MertonSimulationConfig(num_simulations=100_000, seed=91)
    first = compare_merton(CONTRACT, MARKET, PARAMETERS, config)
    second = compare_merton(CONTRACT, MARKET, PARAMETERS, config)
    assert first.monte_carlo.price == second.monte_carlo.price
    assert first.monte_carlo.mean_jump_count == second.monte_carlo.mean_jump_count
    assert abs(first.signed_error) <= 4.0 * first.monte_carlo.standard_error


def test_convergence_child_seeds_and_slope() -> None:
    first = run_merton_convergence(CONTRACT, MARKET, PARAMETERS, [2_000, 8_000, 32_000], 123)
    second = run_merton_convergence(CONTRACT, MARKET, PARAMETERS, [2_000, 8_000, 32_000], 123)
    assert [point.seed for point in first.points] == [point.seed for point in second.points]
    assert [point.price for point in first.points] == [point.price for point in second.points]
    assert len({point.seed for point in first.points}) == 3
    assert first.standard_error_slope == pytest.approx(-0.5, abs=0.15)


def test_series_convergence_terms_and_mass() -> None:
    result = run_merton_series_convergence(CONTRACT, MARKET, PARAMETERS, [1e-4, 1e-8, 1e-12])
    assert [point.terms_used for point in result.points] == sorted(
        point.terms_used for point in result.points
    )
    assert [point.omitted_poisson_mass for point in result.points] == sorted(
        (point.omitted_poisson_mass for point in result.points), reverse=True
    )


def test_short_dated_smile_is_finite_and_nonflat() -> None:
    result = merton_implied_volatility_smile(
        CONTRACT, MARKET, PARAMETERS, [75.0, 85.0, 100.0, 115.0, 125.0]
    )
    implied = [point.implied_volatility for point in result.points]
    assert all(value is not None and math.isfinite(value) for value in implied)
    finite = [float(value) for value in implied if value is not None]
    assert max(finite) - min(finite) > 0.02


@pytest.mark.parametrize("parameter", list(MertonParameterName))
def test_parameter_sensitivity_is_deterministic(parameter: MertonParameterName) -> None:
    values = {
        MertonParameterName.JUMP_INTENSITY: [0.0, 0.5, 1.5],
        MertonParameterName.JUMP_MEAN: [-0.2, -0.1, 0.1],
        MertonParameterName.JUMP_VOLATILITY: [0.0, 0.15, 0.3],
        MertonParameterName.DIFFUSION_VOLATILITY: [0.1, 0.2, 0.3],
    }[parameter]
    first = merton_parameter_sensitivity(CONTRACT, MARKET, PARAMETERS, parameter, values)
    second = merton_parameter_sensitivity(CONTRACT, MARKET, PARAMETERS, parameter, values)
    assert [point.price for point in first.points] == [point.price for point in second.points]
    assert all(math.isfinite(point.price) for point in first.points)


def test_jump_diagnostics_match_poisson_theory() -> None:
    result = merton_jump_diagnostics(
        CONTRACT,
        MARKET,
        PARAMETERS,
        MertonSimulationConfig(num_simulations=100_000, seed=24),
    )
    assert result.mean_simulated_jump_count == pytest.approx(result.expected_jump_count, abs=0.006)
    assert result.simulated_jump_count_variance == pytest.approx(
        result.theoretical_jump_count_variance, abs=0.01
    )
    assert result.zero_jump_fraction == pytest.approx(
        result.theoretical_zero_jump_probability, abs=0.004
    )


def test_calibration_objective_is_zero_for_synthetic_prices() -> None:
    observations = []
    for strike in (85.0, 100.0, 115.0):
        contract = CONTRACT.model_copy(update={"strike": strike})
        price = merton_analytical_price(contract, MARKET, PARAMETERS).price
        observations.append(
            MertonObjectiveObservation(
                strike=strike,
                maturity_years=CONTRACT.maturity_years,
                option_type="CALL",
                observed_price=price,
                weight=2.0,
            )
        )
    result = merton_price_objective(100.0, MARKET, PARAMETERS, observations)
    assert result.sum_squared_error == pytest.approx(0.0, abs=1e-24)
