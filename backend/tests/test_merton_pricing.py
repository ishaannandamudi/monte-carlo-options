import math

import pytest

from backend.app.models.market import MarketEnvironment
from backend.app.models.merton import (
    MertonJumpParameters,
    MertonSeriesConfig,
    MertonSimulationConfig,
)
from backend.app.models.option import ExerciseStyle, OptionContract, OptionType
from backend.app.models.simulation import SimulationConfig, VarianceReduction
from backend.app.pricing.black_scholes import black_scholes_price
from backend.app.pricing.merton import (
    merton_analytical_greeks,
    merton_analytical_price,
    merton_monte_carlo_price,
    price_lognormal_terminal_distribution,
)
from backend.app.pricing.monte_carlo import MonteCarloPricer


def option(option_type: OptionType = OptionType.CALL, maturity: float = 1.0) -> OptionContract:
    return OptionContract(
        option_type=option_type,
        exercise_style=ExerciseStyle.EUROPEAN,
        spot=100.0,
        strike=100.0,
        maturity_years=maturity,
    )


def market(
    rate: float = 0.05, volatility: float = 0.2, dividend: float = 0.01
) -> MarketEnvironment:
    return MarketEnvironment(risk_free_rate=rate, dividend_yield=dividend, volatility=volatility)


def params(intensity: float = 0.5) -> MertonJumpParameters:
    return MertonJumpParameters(jump_intensity=intensity, jump_mean=-0.1, jump_volatility=0.25)


def test_offline_reference_call_price() -> None:
    # Reference computed from the conditional Poisson/lognormal representation.
    result = merton_analytical_price(option(), market(), params())
    assert result.price == pytest.approx(12.097239475818775, abs=1e-11)
    assert result.converged
    assert result.omitted_poisson_mass <= 1e-12


def test_conditional_lognormal_helper_matches_bsm() -> None:
    contract, environment = option(), market()
    log_variance = environment.volatility**2 * contract.maturity_years
    log_mean = (
        math.log(contract.spot)
        + (
            environment.risk_free_rate
            - environment.dividend_yield
            - 0.5 * environment.volatility**2
        )
        * contract.maturity_years
    )
    direct = price_lognormal_terminal_distribution(
        log_mean,
        log_variance,
        contract.strike,
        environment.risk_free_rate,
        contract.maturity_years,
        contract.option_type,
    )
    assert direct == pytest.approx(black_scholes_price(contract, environment), abs=1e-12)


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_lambda_zero_is_exact_bsm(option_type: OptionType) -> None:
    contract = option(option_type)
    result = merton_analytical_price(contract, market(), params(0.0))
    assert result.price == black_scholes_price(contract, market())
    assert result.omitted_poisson_mass == 0.0


def test_lambda_zero_monte_carlo_is_exact_plain_gbm_for_same_seed() -> None:
    count, seed = 20_000, 77
    merton = merton_monte_carlo_price(
        option(), market(), params(0.0), MertonSimulationConfig(num_simulations=count, seed=seed)
    )
    gbm = MonteCarloPricer().price(
        option(),
        market(),
        SimulationConfig(
            num_simulations=count,
            seed=seed,
            variance_reduction=VarianceReduction.NONE,
        ),
    )
    assert merton.price == gbm.price
    assert merton.mean_terminal_spot == gbm.mean_terminal_price
    assert merton.standard_error == gbm.standard_error


def test_put_call_parity_with_dividend_and_negative_rate() -> None:
    environment = market(rate=-0.01, dividend=0.025)
    call = merton_analytical_price(option(OptionType.CALL), environment, params()).price
    put = merton_analytical_price(option(OptionType.PUT), environment, params()).price
    parity = 100.0 * math.exp(-0.025) - 100.0 * math.exp(0.01)
    assert call - put == pytest.approx(parity, abs=2e-10)


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_mc_agrees_with_analytical(option_type: OptionType) -> None:
    contract = option(option_type)
    analytical = merton_analytical_price(contract, market(), params()).price
    mc = merton_monte_carlo_price(
        contract,
        market(),
        params(),
        MertonSimulationConfig(num_simulations=200_000, seed=101),
    )
    assert abs(mc.price - analytical) <= 4.0 * mc.standard_error
    assert mc.confidence_interval == (mc.confidence_interval_low, mc.confidence_interval_high)
    assert math.isfinite(mc.price)


def test_zero_maturity_intrinsic_and_zero_uncertainty() -> None:
    contract = option(OptionType.PUT, 0.0).model_copy(update={"spot": 90.0})
    analytical = merton_analytical_price(contract, market(), params())
    mc = merton_monte_carlo_price(
        contract, market(), params(), MertonSimulationConfig(num_simulations=100, seed=2)
    )
    assert analytical.price == 10.0
    assert analytical.terms_used == 1
    assert mc.price == 10.0
    assert mc.standard_error == 0.0


def test_series_truncation_reports_nonconvergence_and_stabilizes() -> None:
    high_intensity = MertonJumpParameters(
        jump_intensity=20.0, jump_mean=-0.02, jump_volatility=0.12
    )
    short = merton_analytical_price(
        option(), market(), high_intensity, MertonSeriesConfig(max_terms=5)
    )
    converged = merton_analytical_price(
        option(),
        market(),
        high_intensity,
        MertonSeriesConfig(tail_probability_tolerance=1e-13, max_terms=1_000),
    )
    tighter = merton_analytical_price(
        option(),
        market(),
        high_intensity,
        MertonSeriesConfig(tail_probability_tolerance=1e-14, max_terms=1_000),
    )
    assert not short.converged
    assert short.omitted_poisson_mass > converged.omitted_poisson_mass
    assert converged.price == pytest.approx(tighter.price, abs=1e-10)


def test_zero_jump_volatility_and_jump_only_prices_are_finite() -> None:
    deterministic_jumps = MertonJumpParameters(
        jump_intensity=1.5, jump_mean=-0.08, jump_volatility=0.0
    )
    analytical = merton_analytical_price(option(), market(), deterministic_jumps)
    jump_only_market = market(volatility=0.0)
    mc = merton_monte_carlo_price(
        option(),
        jump_only_market,
        params(),
        MertonSimulationConfig(num_simulations=50_000, seed=5),
    )
    assert math.isfinite(analytical.price)
    assert math.isfinite(mc.price)
    assert mc.standard_error > 0.0


def test_greeks_are_finite_and_lambda_zero_tracks_bsm() -> None:
    greeks = merton_analytical_greeks(option(), market(), params(0.0))
    assert all(
        math.isfinite(value)
        for value in (
            greeks.delta,
            greeks.gamma,
            greeks.diffusion_vega,
            greeks.theta,
            greeks.rho,
            greeks.jump_intensity_sensitivity,
            greeks.jump_mean_sensitivity,
            greeks.jump_vol_sensitivity,
        )
    )
    assert greeks.delta == pytest.approx(0.611763, rel=2e-4)
    assert greeks.gamma == pytest.approx(0.01888, rel=2e-3)
