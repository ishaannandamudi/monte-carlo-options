import math

import numpy as np
import pytest

from backend.app.analytics.statistics import antithetic_standard_error
from backend.app.models.heston import (
    HestonParameters,
    HestonSimulationConfig,
    HestonSimulationScheme,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.pricing.heston import heston_monte_carlo_price
from backend.app.pricing.heston_analytical import heston_analytical_price
from backend.app.simulation.heston import simulate_heston_terminal
from backend.app.simulation.random import create_rng


def parameters(**updates: float) -> HestonParameters:
    data = {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "sigma_v": 0.3, "rho": -0.7}
    data.update(updates)
    return HestonParameters.model_validate(data)


def contract(option_type: OptionType = OptionType.CALL) -> OptionContract:
    return OptionContract(
        option_type=option_type,
        spot=100,
        strike=100,
        maturity_years=1,
    )


def market() -> MarketEnvironment:
    return MarketEnvironment(risk_free_rate=0.05, dividend_yield=0.0, volatility=0.0)


@pytest.mark.parametrize("option_type", list(OptionType))
@pytest.mark.parametrize("scheme", list(HestonSimulationScheme))
def test_mc_agrees_with_analytical(option_type: OptionType, scheme: HestonSimulationScheme) -> None:
    option = contract(option_type)
    result = heston_monte_carlo_price(
        option,
        market(),
        parameters(),
        HestonSimulationConfig(
            num_simulations=40_000,
            num_time_steps=100,
            seed=12,
            scheme=scheme,
            antithetic=True,
        ),
    )
    analytical = heston_analytical_price(option, market(), parameters()).price
    tolerance = 5 * result.standard_error + (
        0.08 if scheme is HestonSimulationScheme.FULL_TRUNCATION_EULER else 0.03
    )
    assert abs(result.price - analytical) <= tolerance
    assert result.feller_satisfied and result.feller_ratio is not None
    assert result.price >= 0.0 and 0.0 <= result.probability_itm <= 1.0


@pytest.mark.parametrize("scheme", list(HestonSimulationScheme))
def test_same_seed_reproducibility_and_different_seed(scheme: HestonSimulationScheme) -> None:
    config = HestonSimulationConfig(
        num_simulations=4_001,
        num_time_steps=25,
        seed=7,
        scheme=scheme,
        antithetic=True,
    )
    first = heston_monte_carlo_price(contract(), market(), parameters(), config)
    same = heston_monte_carlo_price(contract(), market(), parameters(), config)
    different = heston_monte_carlo_price(
        contract(), market(), parameters(), config.model_copy(update={"seed": 8})
    )
    assert first.price == same.price and first.standard_error == same.standard_error
    assert first.price != different.price


@pytest.mark.parametrize("scheme", list(HestonSimulationScheme))
def test_antithetic_se_uses_interleaved_pair_clusters(scheme: HestonSimulationScheme) -> None:
    config = HestonSimulationConfig(
        num_simulations=4_001,
        num_time_steps=25,
        seed=91,
        scheme=scheme,
        antithetic=True,
    )
    result = heston_monte_carlo_price(contract(), market(), parameters(), config)
    terminal = simulate_heston_terminal(
        contract(), market(), parameters(), config, create_rng(config.seed)
    )
    discounted = np.exp(-0.05) * np.maximum(terminal.terminal_spots - 100.0, 0.0)
    assert result.standard_error == antithetic_standard_error(discounted)
    assert result.price == float(np.mean(discounted))


@pytest.mark.parametrize("scheme", list(HestonSimulationScheme))
def test_constant_variance_mc_matches_black_scholes_analytical(
    scheme: HestonSimulationScheme,
) -> None:
    params = parameters(sigma_v=0.0)
    result = heston_monte_carlo_price(
        contract(),
        market(),
        params,
        HestonSimulationConfig(
            num_simulations=30_000,
            num_time_steps=10,
            seed=4,
            scheme=scheme,
        ),
    )
    analytical = heston_analytical_price(contract(), market(), params).price
    assert abs(result.price - analytical) <= 5 * result.standard_error
    assert result.std_terminal_variance == 0.0


def test_zero_maturity_and_zero_variance_are_deterministic() -> None:
    expiring = contract(OptionType.PUT).model_copy(update={"spot": 90.0, "maturity_years": 0.0})
    expiry = heston_monte_carlo_price(
        expiring,
        market(),
        parameters(),
        HestonSimulationConfig(num_simulations=1),
    )
    zero = heston_monte_carlo_price(
        contract(),
        market(),
        parameters(v0=0.0, theta=0.0),
        HestonSimulationConfig(num_simulations=10, num_time_steps=5),
    )
    assert expiry.price == 10.0 and expiry.standard_error == 0.0
    assert zero.standard_error == 0.0
    assert zero.mean_terminal_spot == pytest.approx(100 * math.exp(0.05))


def test_feller_violation_reports_diagnostic_and_finite_price() -> None:
    params = parameters(kappa=0.5, theta=0.02, sigma_v=0.8)
    result = heston_monte_carlo_price(
        contract(),
        market(),
        params,
        HestonSimulationConfig(
            num_simulations=10_000,
            num_time_steps=100,
            scheme=HestonSimulationScheme.QUADRATIC_EXPONENTIAL,
        ),
    )
    assert not result.feller_satisfied
    assert math.isfinite(result.price)


def test_fte_exposes_negative_proposal_fraction() -> None:
    result = heston_monte_carlo_price(
        contract(),
        market(),
        parameters(kappa=0.5, theta=0.02, sigma_v=1.0),
        HestonSimulationConfig(
            num_simulations=5_000,
            num_time_steps=20,
            scheme=HestonSimulationScheme.FULL_TRUNCATION_EULER,
        ),
    )
    assert result.negative_variance_proposal_fraction is not None
    assert result.negative_variance_proposal_fraction > 0.0


@pytest.mark.parametrize("rho", [-1.0, 0.0, 1.0])
def test_extreme_rho_fte_pricing_is_finite(rho: float) -> None:
    result = heston_monte_carlo_price(
        contract(),
        market(),
        parameters(rho=rho),
        HestonSimulationConfig(
            num_simulations=5_000,
            num_time_steps=50,
            scheme=HestonSimulationScheme.FULL_TRUNCATION_EULER,
        ),
    )
    assert math.isfinite(result.price)
