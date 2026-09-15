import math

import numpy as np
import pytest

from backend.app.core.exceptions import NumericalError
from backend.app.models.greeks import DifferenceScheme, Greek, GreekBumpConfig
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.models.simulation import SimulationConfig, VarianceReduction
from backend.app.payoffs.vanilla import vanilla_payoff
from backend.app.pricing.black_scholes_greeks import (
    analytical_greek_value,
    black_scholes_greeks,
)
from backend.app.pricing.monte_carlo import MonteCarloPricer
from backend.app.pricing.monte_carlo_greeks import finite_difference_greeks
from backend.app.pricing.variance_reduction import (
    cluster_observations,
    estimator_variance,
    weighted_estimate,
)
from backend.app.simulation.gbm import standard_normal_drivers, terminal_prices_from_normals
from backend.app.simulation.random import create_rng


def contract(
    option_type: OptionType = OptionType.CALL,
    *,
    spot: float = 100.0,
    strike: float = 100.0,
    maturity: float = 1.0,
) -> OptionContract:
    return OptionContract(
        option_type=option_type,
        spot=spot,
        strike=strike,
        maturity_years=maturity,
    )


def market() -> MarketEnvironment:
    return MarketEnvironment(risk_free_rate=0.05, volatility=0.2, dividend_yield=0.01)


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_all_finite_difference_greeks_agree_with_analytical_values(
    option_type: OptionType,
) -> None:
    option = contract(option_type)
    environment = market()
    analytical = black_scholes_greeks(option, environment)
    results = finite_difference_greeks(
        option,
        environment,
        SimulationConfig(
            num_simulations=150_000,
            seed=42,
            variance_reduction=VarianceReduction.ANTITHETIC,
        ),
        list(Greek),
    )
    for result in results:
        benchmark = analytical_greek_value(analytical, result.greek)
        assert abs(result.estimate - benchmark) <= 6.0 * result.standard_error + 0.003


@pytest.mark.parametrize(
    ("option", "environment"),
    [
        (
            contract(spot=120.0, strike=100.0, maturity=0.2),
            MarketEnvironment(risk_free_rate=0.03, volatility=0.25, dividend_yield=0.02),
        ),
        (
            contract(spot=80.0, strike=100.0, maturity=3.0),
            MarketEnvironment(risk_free_rate=-0.01, volatility=0.3, dividend_yield=0.01),
        ),
    ],
)
def test_finite_difference_delta_and_vega_cover_moneyness_and_maturity(
    option: OptionContract,
    environment: MarketEnvironment,
) -> None:
    analytical = black_scholes_greeks(option, environment)
    results = finite_difference_greeks(
        option,
        environment,
        SimulationConfig(
            num_simulations=100_000,
            seed=91,
            variance_reduction=VarianceReduction.ANTITHETIC_CONTROL_VARIATE,
        ),
        [Greek.DELTA, Greek.VEGA],
    )
    for result in results:
        benchmark = analytical_greek_value(analytical, result.greek)
        assert abs(result.estimate - benchmark) <= 7.0 * result.standard_error + 0.01


@pytest.mark.parametrize("method", list(VarianceReduction))
def test_finite_difference_supports_every_pricing_variance_method(
    method: VarianceReduction,
) -> None:
    results = finite_difference_greeks(
        contract(),
        market(),
        SimulationConfig(num_simulations=10_001, seed=7, variance_reduction=method),
        list(Greek),
    )
    assert [result.greek for result in results] == list(Greek)
    assert all(result.variance_reduction is method for result in results)
    assert all(result.num_simulations == 10_001 for result in results)
    assert all(math.isfinite(result.estimate) for result in results)
    assert all(math.isfinite(result.standard_error) for result in results)


@pytest.mark.parametrize("method", list(VarianceReduction))
def test_finite_difference_base_price_matches_existing_pricer(
    method: VarianceReduction,
) -> None:
    config = SimulationConfig(num_simulations=20_001, seed=19, variance_reduction=method)
    finite_difference = finite_difference_greeks(
        contract(),
        market(),
        config,
        [Greek.DELTA],
    )[0]
    price = MonteCarloPricer().price(contract(), market(), config)
    assert finite_difference.base_price == pytest.approx(price.price, abs=1e-14)


def test_crn_delta_standard_error_matches_direct_per_path_differences() -> None:
    option = contract()
    environment = market()
    config = SimulationConfig(
        num_simulations=2_000,
        seed=123,
        variance_reduction=VarianceReduction.NONE,
    )
    result = finite_difference_greeks(option, environment, config, [Greek.DELTA])[0]
    bump = option.spot * GreekBumpConfig().spot_relative_bump
    normals = standard_normal_drivers(
        config.num_simulations,
        create_rng(config.seed),
        antithetic=False,
    )
    up_contract = contract(spot=option.spot + bump)
    down_contract = contract(spot=option.spot - bump)
    discount = math.exp(-environment.risk_free_rate * option.maturity_years)
    up = discount * vanilla_payoff(
        terminal_prices_from_normals(up_contract, environment, normals),
        up_contract,
    )
    down = discount * vanilla_payoff(
        terminal_prices_from_normals(down_contract, environment, normals),
        down_contract,
    )
    contributions = (up - down) / (2.0 * bump)

    assert result.estimate == pytest.approx(float(np.mean(contributions)), abs=1e-14)
    assert result.standard_error == pytest.approx(
        float(np.std(contributions, ddof=1) / math.sqrt(contributions.size)),
        abs=1e-14,
    )


def test_odd_antithetic_gamma_uses_weighted_cluster_variance() -> None:
    option = contract()
    environment = market()
    config = SimulationConfig(
        num_simulations=1_001,
        seed=22,
        variance_reduction=VarianceReduction.ANTITHETIC,
    )
    result = finite_difference_greeks(option, environment, config, [Greek.GAMMA])[0]
    bump = result.bump_size
    normals = standard_normal_drivers(
        config.num_simulations,
        create_rng(config.seed),
        antithetic=True,
    )
    up_contract = contract(spot=option.spot + bump)
    down_contract = contract(spot=option.spot - bump)
    discount = math.exp(-environment.risk_free_rate * option.maturity_years)
    up = discount * vanilla_payoff(
        terminal_prices_from_normals(up_contract, environment, normals),
        up_contract,
    )
    base = discount * vanilla_payoff(
        terminal_prices_from_normals(option, environment, normals),
        option,
    )
    down = discount * vanilla_payoff(
        terminal_prices_from_normals(down_contract, environment, normals),
        down_contract,
    )
    path_gamma = (up - 2.0 * base + down) / bump**2
    clusters = cluster_observations(path_gamma, antithetic=True)

    assert clusters.path_count == 1_001
    assert result.estimate == pytest.approx(weighted_estimate(clusters), abs=1e-12)
    assert result.estimator_variance == pytest.approx(estimator_variance(clusters), rel=1e-12)


def test_crn_materially_reduces_delta_noise_against_independent_bumps() -> None:
    option = contract()
    environment = market()
    count = 30_000
    bump = option.spot * GreekBumpConfig().spot_relative_bump
    crn = finite_difference_greeks(
        option,
        environment,
        SimulationConfig(
            num_simulations=count,
            seed=45,
            variance_reduction=VarianceReduction.NONE,
        ),
        [Greek.DELTA],
    )[0]
    up = MonteCarloPricer().price(
        contract(spot=option.spot + bump),
        environment,
        SimulationConfig(
            num_simulations=count,
            seed=145,
            variance_reduction=VarianceReduction.NONE,
        ),
    )
    down = MonteCarloPricer().price(
        contract(spot=option.spot - bump),
        environment,
        SimulationConfig(
            num_simulations=count,
            seed=245,
            variance_reduction=VarianceReduction.NONE,
        ),
    )
    independent_se = math.sqrt(up.standard_error**2 + down.standard_error**2) / (2.0 * bump)
    assert crn.standard_error < independent_se / 20.0


@pytest.mark.parametrize("method", list(VarianceReduction))
def test_same_seed_reproduces_finite_difference_outputs(method: VarianceReduction) -> None:
    config = SimulationConfig(num_simulations=10_001, seed=55, variance_reduction=method)
    first = finite_difference_greeks(contract(), market(), config, list(Greek))
    second = finite_difference_greeks(contract(), market(), config, list(Greek))
    assert [item.model_dump(exclude={"runtime_ms"}) for item in first] == [
        item.model_dump(exclude={"runtime_ms"}) for item in second
    ]


def test_different_seed_changes_finite_difference_estimate() -> None:
    first = finite_difference_greeks(
        contract(),
        market(),
        SimulationConfig(num_simulations=10_000, seed=55),
        [Greek.DELTA],
    )[0]
    second = finite_difference_greeks(
        contract(),
        market(),
        SimulationConfig(num_simulations=10_000, seed=56),
        [Greek.DELTA],
    )[0]
    assert first.estimate != second.estimate


def test_one_sided_boundary_fallbacks_never_create_invalid_models() -> None:
    config = SimulationConfig(
        num_simulations=20_000,
        seed=8,
        variance_reduction=VarianceReduction.NONE,
    )
    vega = finite_difference_greeks(
        contract(),
        MarketEnvironment(risk_free_rate=0.0, volatility=0.0),
        config,
        [Greek.VEGA],
    )[0]
    theta = finite_difference_greeks(
        contract(maturity=0.0),
        market(),
        config,
        [Greek.THETA],
    )[0]
    assert vega.difference_scheme is DifferenceScheme.FORWARD
    assert theta.difference_scheme is DifferenceScheme.FORWARD
    assert vega.downward_price is None
    assert theta.downward_price is None
    assert math.isfinite(vega.estimate)
    assert math.isfinite(theta.estimate)


def test_reasonable_bump_sizes_produce_stable_delta_estimates_without_monotonic_claim() -> None:
    config = SimulationConfig(
        num_simulations=80_000,
        seed=39,
        variance_reduction=VarianceReduction.ANTITHETIC,
    )
    estimates = [
        finite_difference_greeks(
            contract(),
            market(),
            config,
            [Greek.DELTA],
            GreekBumpConfig(spot_relative_bump=bump),
        )[0].estimate
        for bump in (5e-4, 1e-3, 5e-3)
    ]
    analytical = black_scholes_greeks(contract(), market()).delta
    assert all(abs(estimate - analytical) < 0.01 for estimate in estimates)
    assert max(estimates) - min(estimates) < 0.005


def test_unrepresentably_small_bump_raises_clear_numerical_error() -> None:
    with pytest.raises(NumericalError, match="floating-point range"):
        finite_difference_greeks(
            contract(),
            market(),
            SimulationConfig(num_simulations=100, variance_reduction=VarianceReduction.NONE),
            [Greek.VEGA],
            GreekBumpConfig(volatility_absolute_bump=5e-324),
        )
