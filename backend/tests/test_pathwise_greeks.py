import math

import numpy as np
import pytest

from backend.app.core.exceptions import UnsupportedGreekMethodError
from backend.app.models.greeks import Greek
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.models.simulation import SimulationConfig, VarianceReduction
from backend.app.pricing.black_scholes_greeks import (
    analytical_greek_value,
    black_scholes_greeks,
)
from backend.app.pricing.monte_carlo_greeks import pathwise_greeks
from backend.app.pricing.variance_reduction import (
    cluster_observations,
    estimator_variance,
    weighted_estimate,
)
from backend.app.simulation.gbm import standard_normal_drivers, terminal_prices_from_normals
from backend.app.simulation.random import create_rng


def contract(option_type: OptionType = OptionType.CALL, *, maturity: float = 1.0) -> OptionContract:
    return OptionContract(
        option_type=option_type,
        spot=100.0,
        strike=100.0,
        maturity_years=maturity,
    )


def market(volatility: float = 0.2) -> MarketEnvironment:
    return MarketEnvironment(risk_free_rate=0.05, volatility=volatility, dividend_yield=0.01)


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
@pytest.mark.parametrize("method", [VarianceReduction.NONE, VarianceReduction.ANTITHETIC])
def test_pathwise_delta_and_vega_agree_with_analytical_values(
    option_type: OptionType,
    method: VarianceReduction,
) -> None:
    option = contract(option_type)
    environment = market()
    analytical = black_scholes_greeks(option, environment)
    results = pathwise_greeks(
        option,
        environment,
        SimulationConfig(num_simulations=150_000, seed=42, variance_reduction=method),
        [Greek.DELTA, Greek.VEGA],
    )
    for result in results:
        benchmark = analytical_greek_value(analytical, result.greek)
        assert abs(result.estimate - benchmark) <= 6.0 * result.standard_error + 1e-4


@pytest.mark.parametrize("greek", [Greek.DELTA, Greek.VEGA])
def test_pathwise_odd_antithetic_se_uses_independent_weighted_clusters(greek: Greek) -> None:
    option = contract()
    environment = market()
    config = SimulationConfig(
        num_simulations=1_001,
        seed=73,
        variance_reduction=VarianceReduction.ANTITHETIC,
    )
    result = pathwise_greeks(option, environment, config, [greek])[0]
    normals = standard_normal_drivers(
        config.num_simulations,
        create_rng(config.seed),
        antithetic=True,
    )
    terminal = terminal_prices_from_normals(option, environment, normals)
    discount = math.exp(-environment.risk_free_rate * option.maturity_years)
    indicator = np.asarray(terminal > option.strike, dtype=np.float64)
    if greek is Greek.DELTA:
        contributions = discount * indicator * terminal / option.spot
    else:
        contributions = (
            discount
            * indicator
            * terminal
            * (
                -environment.volatility * option.maturity_years
                + math.sqrt(option.maturity_years) * normals
            )
        )
    clusters = cluster_observations(contributions, antithetic=True)

    assert clusters.path_count == 1_001
    assert result.estimate == pytest.approx(weighted_estimate(clusters), abs=1e-14)
    assert result.estimator_variance == pytest.approx(estimator_variance(clusters), rel=1e-13)


@pytest.mark.parametrize("method", [VarianceReduction.NONE, VarianceReduction.ANTITHETIC])
def test_pathwise_same_seed_reproduces_and_different_seed_changes_results(
    method: VarianceReduction,
) -> None:
    config = SimulationConfig(num_simulations=20_001, seed=20, variance_reduction=method)
    first = pathwise_greeks(contract(), market(), config, [Greek.DELTA, Greek.VEGA])
    second = pathwise_greeks(contract(), market(), config, [Greek.DELTA, Greek.VEGA])
    different = pathwise_greeks(
        contract(),
        market(),
        SimulationConfig(num_simulations=20_001, seed=21, variance_reduction=method),
        [Greek.DELTA, Greek.VEGA],
    )
    assert [item.model_dump(exclude={"runtime_ms"}) for item in first] == [
        item.model_dump(exclude={"runtime_ms"}) for item in second
    ]
    assert [item.estimate for item in first] != [item.estimate for item in different]


@pytest.mark.parametrize(
    "method",
    [VarianceReduction.CONTROL_VARIATE, VarianceReduction.ANTITHETIC_CONTROL_VARIATE],
)
def test_pathwise_rejects_control_variate_modes(method: VarianceReduction) -> None:
    with pytest.raises(UnsupportedGreekMethodError, match="NONE or ANTITHETIC"):
        pathwise_greeks(
            contract(),
            market(),
            SimulationConfig(num_simulations=100, variance_reduction=method),
            [Greek.DELTA],
        )


@pytest.mark.parametrize(
    "greek",
    [Greek.GAMMA, Greek.THETA, Greek.RHO, Greek.DIVIDEND_RHO],
)
def test_pathwise_rejects_unsupported_greeks(greek: Greek) -> None:
    with pytest.raises(UnsupportedGreekMethodError, match="does not support"):
        pathwise_greeks(
            contract(),
            market(),
            SimulationConfig(num_simulations=100, variance_reduction=VarianceReduction.NONE),
            [greek],
        )


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_pathwise_expiry_equality_uses_strict_itm_indicator(option_type: OptionType) -> None:
    results = pathwise_greeks(
        contract(option_type, maturity=0.0),
        market(),
        SimulationConfig(num_simulations=1, variance_reduction=VarianceReduction.NONE),
        [Greek.DELTA, Greek.VEGA],
    )
    assert all(result.estimate == 0.0 for result in results)
    assert all(result.standard_error == 0.0 for result in results)


def test_pathwise_zero_volatility_delta_is_deterministic_and_finite() -> None:
    result = pathwise_greeks(
        contract(),
        market(volatility=0.0),
        SimulationConfig(num_simulations=5, seed=3, variance_reduction=VarianceReduction.NONE),
        [Greek.DELTA],
    )[0]
    assert math.isfinite(result.estimate)
    assert result.standard_error == 0.0
