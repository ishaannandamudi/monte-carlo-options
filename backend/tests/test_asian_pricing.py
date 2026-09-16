import math

import numpy as np
import pytest

from backend.app.analytics.statistics import antithetic_standard_error
from backend.app.core.exceptions import ExoticPricingError, NumericalError
from backend.app.models.asian import (
    AsianOptionContract,
    AsianSimulationConfig,
    AsianVarianceReduction,
    AveragingType,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.payoffs.asian import asian_average, asian_payoff_from_average
from backend.app.pricing import asian as asian_module
from backend.app.pricing.asian import (
    AsianMonteCarloPricer,
    asian_monte_carlo_price,
    geometric_asian_analytical_price,
    geometric_asian_log_moments,
)
from backend.app.pricing.black_scholes import black_scholes_price
from backend.app.pricing.variance_reduction import apply_control_variate, cluster_observations


def contract(
    option_type: OptionType = OptionType.CALL,
    averaging_type: AveragingType = AveragingType.ARITHMETIC,
    *,
    times: list[float] | None = None,
    count: int = 12,
    include_initial: bool = False,
    maturity: float = 1.0,
) -> AsianOptionContract:
    schedule = {"monitoring_times": times} if times is not None else {"monitoring_count": count}
    return AsianOptionContract(
        option_type=option_type,
        spot=100,
        strike=100,
        maturity_years=maturity,
        averaging_type=averaging_type,
        averaging_schedule=schedule,
        include_initial_spot_in_average=include_initial,
    )


def test_geometric_moments_for_irregular_schedule_and_initial_spot() -> None:
    c = contract(averaging_type=AveragingType.GEOMETRIC, times=[0.2, 0.7, 1.0])
    market = MarketEnvironment(risk_free_rate=0.03, volatility=0.4, dividend_yield=0.01)
    mean_log, variance_log = geometric_asian_log_moments(c, market)
    times = np.array([0.2, 0.7, 1.0])
    expected_mean = math.log(100) + (0.03 - 0.01 - 0.5 * 0.4**2) * np.mean(times)
    expected_variance = 0.4**2 * np.minimum.outer(times, times).sum() / times.size**2
    assert mean_log == pytest.approx(expected_mean)
    assert variance_log == pytest.approx(expected_variance)

    included = contract(
        averaging_type=AveragingType.GEOMETRIC,
        times=[0.2, 0.7, 1.0],
        include_initial=True,
    )
    included_mean, included_variance = geometric_asian_log_moments(included, market)
    all_times = np.array([0.0, 0.2, 0.7, 1.0])
    assert included_mean == pytest.approx(
        math.log(100) + (0.03 - 0.01 - 0.5 * 0.4**2) * np.mean(all_times)
    )
    assert included_variance == pytest.approx(
        0.4**2 * np.minimum.outer(all_times, all_times).sum() / all_times.size**2
    )


@pytest.mark.parametrize("option_type", list(OptionType))
def test_one_maturity_monitor_reduces_to_vanilla(option_type: OptionType) -> None:
    c = contract(option_type, AveragingType.GEOMETRIC, times=[1.0])
    market = MarketEnvironment(risk_free_rate=-0.01, volatility=0.25, dividend_yield=0.03)
    vanilla = OptionContract(
        option_type=option_type,
        spot=c.spot,
        strike=c.strike,
        maturity_years=c.maturity_years,
    )
    assert geometric_asian_analytical_price(c, market).price == pytest.approx(
        black_scholes_price(vanilla, market), rel=2e-14, abs=1e-14
    )


def test_include_initial_one_future_monitor_has_sqrt_distribution() -> None:
    c = contract(
        averaging_type=AveragingType.GEOMETRIC,
        times=[1.0],
        include_initial=True,
    )
    market = MarketEnvironment(risk_free_rate=0.05, volatility=0.2, dividend_yield=0.01)
    mean_log, variance_log = geometric_asian_log_moments(c, market)
    assert mean_log == pytest.approx(math.log(100) + (0.05 - 0.01 - 0.5 * 0.2**2) / 2)
    assert variance_log == pytest.approx(0.2**2 / 4)


@pytest.mark.parametrize("option_type", list(OptionType))
@pytest.mark.parametrize("include_initial", [False, True])
def test_geometric_monte_carlo_agrees_with_discrete_analytical(
    option_type: OptionType, include_initial: bool
) -> None:
    times = [0.1, 0.35, 0.8, 1.0]
    c = contract(
        option_type,
        AveragingType.GEOMETRIC,
        times=times,
        include_initial=include_initial,
    )
    market = MarketEnvironment(risk_free_rate=0.04, volatility=0.3, dividend_yield=0.02)
    result = asian_monte_carlo_price(
        c,
        market,
        AsianSimulationConfig(
            num_simulations=30_000,
            seed=123,
            variance_reduction=AsianVarianceReduction.ANTITHETIC,
        ),
    )
    benchmark = geometric_asian_analytical_price(c, market).price
    assert abs(result.price - benchmark) <= 5 * result.standard_error
    assert result.monitoring_count == len(times) + int(include_initial)
    assert 0 <= result.probability_itm <= 1


@pytest.mark.parametrize("averaging_type", list(AveragingType))
def test_zero_maturity_is_spot_intrinsic_without_simulation(
    averaging_type: AveragingType,
) -> None:
    c = contract(
        OptionType.PUT,
        averaging_type,
        maturity=0,
    ).model_copy(update={"strike": 110.0})
    result = asian_monte_carlo_price(
        c,
        MarketEnvironment(risk_free_rate=1e308, volatility=1e308, dividend_yield=-1e308),
    )
    assert result.price == 10 and result.standard_error == 0
    assert result.mean_average == 100 and result.monitoring_count == 1
    assert result.deterministic and result.simulation_runtime_ms >= 0


@pytest.mark.parametrize("averaging_type", list(AveragingType))
def test_zero_volatility_uses_exact_deterministic_observations(
    averaging_type: AveragingType,
) -> None:
    c = contract(OptionType.CALL, averaging_type, times=[0.2, 0.7, 1.0])
    market = MarketEnvironment(risk_free_rate=0.05, volatility=0, dividend_yield=0.01)
    result = asian_monte_carlo_price(c, market)
    stocks = 100 * np.exp(0.04 * np.array([0.2, 0.7, 1.0]))
    expected_average = (
        float(np.mean(stocks))
        if averaging_type is AveragingType.ARITHMETIC
        else float(np.exp(np.mean(np.log(stocks))))
    )
    expected = math.exp(-0.05) * max(expected_average - 100, 0)
    assert result.price == pytest.approx(expected)
    assert result.mean_average == pytest.approx(expected_average)
    assert result.standard_error == result.average_std == 0


def test_geometric_control_materially_reduces_representative_arithmetic_variance() -> None:
    c = contract(OptionType.CALL, AveragingType.ARITHMETIC, count=12)
    market = MarketEnvironment(risk_free_rate=0.05, volatility=0.3, dividend_yield=0.01)
    result = asian_monte_carlo_price(
        c,
        market,
        AsianSimulationConfig(
            num_simulations=20_000,
            seed=77,
            variance_reduction=AsianVarianceReduction.ANTITHETIC_GEOMETRIC_CONTROL,
        ),
    )
    assert result.raw_price is not None and result.adjusted_price == result.price
    assert result.control_variate_beta is not None and math.isfinite(result.control_variate_beta)
    assert result.raw_standard_error is not None and result.adjusted_standard_error is not None
    assert result.adjusted_standard_error < result.raw_standard_error / 5
    assert result.variance_reduction_ratio is not None and result.variance_reduction_ratio > 25


@pytest.mark.parametrize("option_type", list(OptionType))
@pytest.mark.parametrize("mode", list(AsianVarianceReduction))
def test_arithmetic_calls_and_puts_support_every_variance_mode(
    option_type: OptionType, mode: AsianVarianceReduction
) -> None:
    c = contract(option_type, AveragingType.ARITHMETIC, count=6)
    market = MarketEnvironment(risk_free_rate=0.03, volatility=0.25, dividend_yield=0.02)
    config = AsianSimulationConfig(num_simulations=4_000, seed=19, variance_reduction=mode)
    first = asian_monte_carlo_price(c, market, config)
    same = asian_monte_carlo_price(c, market, config)
    assert first.price == same.price and first.standard_error == same.standard_error
    assert math.isfinite(first.price) and first.standard_error >= 0
    assert first.confidence_interval_low <= first.price <= first.confidence_interval_high
    if mode.uses_geometric_control:
        assert first.adjusted_price == first.price
        assert first.analytical_geometric_control_price is not None


def test_different_asian_seed_changes_stochastic_estimate() -> None:
    c = contract(OptionType.PUT, AveragingType.ARITHMETIC, count=4)
    market = MarketEnvironment(risk_free_rate=0.03, volatility=0.25)
    base = AsianSimulationConfig(
        num_simulations=2_000,
        seed=9,
        variance_reduction=AsianVarianceReduction.NONE,
    )
    assert (
        asian_monte_carlo_price(c, market, base).price
        != asian_monte_carlo_price(c, market, base.model_copy(update={"seed": 10})).price
    )


def test_deterministic_geometric_control_has_explicit_zero_variance_fallback() -> None:
    result = asian_monte_carlo_price(
        contract(count=4),
        MarketEnvironment(risk_free_rate=0.05, volatility=0),
        AsianSimulationConfig(
            num_simulations=10,
            variance_reduction=AsianVarianceReduction.GEOMETRIC_CONTROL,
        ),
    )
    assert result.control_variate_beta == 0
    assert result.raw_standard_error == result.adjusted_standard_error == 0
    assert result.raw_estimator_variance == result.adjusted_estimator_variance == 0
    assert result.variance_reduction_ratio is None


def test_antithetic_geometric_control_uses_pair_clusters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations = np.array(
        [
            [80.0, 120.0],
            [125.0, 85.0],
            [90.0, 130.0],
            [115.0, 75.0],
            [105.0, 110.0],
        ]
    )

    def fixed(*args, **kwargs):
        return observations.copy()

    monkeypatch.setattr(asian_module, "simulate_observation_prices", fixed)
    c = contract(OptionType.CALL, AveragingType.ARITHMETIC, times=[0.5, 1.0])
    market = MarketEnvironment(risk_free_rate=0, volatility=0.2)
    config = AsianSimulationConfig(
        num_simulations=5,
        variance_reduction=AsianVarianceReduction.ANTITHETIC_GEOMETRIC_CONTROL,
    )
    result = AsianMonteCarloPricer().price(c, market, config)
    raw = asian_payoff_from_average(asian_average(observations, AveragingType.ARITHMETIC), c)
    geometric_contract = c.model_copy(update={"averaging_type": AveragingType.GEOMETRIC})
    control_values = asian_payoff_from_average(
        asian_average(observations, AveragingType.GEOMETRIC), geometric_contract
    )
    known = geometric_asian_analytical_price(c, market).price
    expected = apply_control_variate(
        cluster_observations(raw, antithetic=True),
        cluster_observations(control_values, antithetic=True),
        known,
    )
    assert result.price == pytest.approx(expected.adjusted_price)
    assert result.standard_error == pytest.approx(expected.adjusted_standard_error)
    assert result.control_variate_beta == pytest.approx(expected.beta)
    assert result.raw_standard_error == pytest.approx(antithetic_standard_error(raw))


def test_geometric_control_rejected_for_geometric_target() -> None:
    with pytest.raises(ExoticPricingError, match="ARITHMETIC"):
        asian_monte_carlo_price(
            contract(averaging_type=AveragingType.GEOMETRIC),
            MarketEnvironment(risk_free_rate=0.05, volatility=0.2),
            AsianSimulationConfig(variance_reduction=AsianVarianceReduction.GEOMETRIC_CONTROL),
        )


def test_extreme_geometric_inputs_raise_instead_of_leaking_nonfinite_values() -> None:
    with pytest.raises(NumericalError, match="finite numerical range"):
        geometric_asian_analytical_price(
            contract(averaging_type=AveragingType.GEOMETRIC),
            MarketEnvironment(risk_free_rate=0, volatility=1e308),
        )
