"""Analytical discrete-geometric and Monte Carlo fixed-strike Asian pricing."""

import logging
import math
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from backend.app.analytics.statistics import (
    antithetic_standard_error,
    confidence_interval,
    sample_mean,
    sample_standard_deviation,
    standard_error,
)
from backend.app.core.exceptions import ExoticPricingError, NumericalError
from backend.app.core.numerics import finite_exp, require_finite
from backend.app.models.asian import (
    AsianOptionContract,
    AsianPricingResult,
    AsianSimulationConfig,
    AsianVarianceReduction,
    AveragingType,
    GeometricAsianAnalyticalResult,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionType
from backend.app.payoffs.asian import asian_average, asian_payoff_from_average
from backend.app.pricing.variance_reduction import (
    apply_control_variate,
    cluster_observations,
)
from backend.app.simulation.gbm import simulate_observation_prices
from backend.app.simulation.random import create_rng

logger = logging.getLogger(__name__)


def geometric_asian_log_moments(
    contract: AsianOptionContract,
    market: MarketEnvironment,
) -> tuple[float, float]:
    """Return mean and variance of the discrete geometric average's logarithm."""

    times = contract.observation_times
    count = len(times)
    if all(time == 0.0 for time in times):
        return math.log(contract.spot), 0.0
    mean_time = math.fsum(times) / count
    volatility = np.float64(market.volatility)
    with np.errstate(over="ignore", invalid="ignore"):
        variance_rate = volatility * volatility
        drift_rate = np.float64(market.risk_free_rate - market.dividend_yield) - (
            np.float64(0.5) * variance_rate
        )
        mean_log = np.float64(math.log(contract.spot)) + drift_rate * mean_time
        covariance_time_sum = math.fsum(
            (2 * (count - index) - 1) * time for index, time in enumerate(times)
        )
        variance_log = variance_rate * covariance_time_sum / (count * count)
    mean_value = float(mean_log)
    variance_value = float(variance_log)
    if not math.isfinite(mean_value) or not math.isfinite(variance_value) or variance_value < 0.0:
        raise NumericalError("geometric Asian log moments are outside finite numerical range")
    return mean_value, variance_value


def geometric_asian_analytical_price(
    contract: AsianOptionContract,
    market: MarketEnvironment,
) -> GeometricAsianAnalyticalResult:
    """Price a discrete geometric-average payoff from its exact lognormal law."""

    mean_log, variance_log = geometric_asian_log_moments(contract, market)
    only_initial = all(time == 0.0 for time in contract.observation_times)
    expected_average = (
        contract.spot
        if only_initial
        else finite_exp(
            mean_log + 0.5 * variance_log,
            context="expected geometric Asian average",
        )
    )
    discount = finite_exp(
        -market.risk_free_rate * contract.maturity_years,
        context="geometric Asian discount factor",
    )
    deterministic = variance_log == 0.0
    if deterministic:
        deterministic_average = (
            contract.spot
            if only_initial
            else finite_exp(mean_log, context="deterministic geometric Asian average")
        )
        payoff = (
            max(deterministic_average - contract.strike, 0.0)
            if contract.option_type is OptionType.CALL
            else max(contract.strike - deterministic_average, 0.0)
        )
        price = discount * payoff
    else:
        standard_deviation = math.sqrt(variance_log)
        d2 = (mean_log - math.log(contract.strike)) / standard_deviation
        d1 = d2 + standard_deviation
        if contract.option_type is OptionType.CALL:
            expectation = expected_average * float(norm.cdf(d1)) - contract.strike * float(
                norm.cdf(d2)
            )
        else:
            expectation = contract.strike * float(norm.cdf(-d2)) - expected_average * float(
                norm.cdf(-d1)
            )
        price = discount * max(expectation, 0.0)
    if not math.isfinite(price) or price < 0.0:
        raise NumericalError("geometric Asian analytical price is invalid")
    return GeometricAsianAnalyticalResult(
        price=price,
        mean_log_average=mean_log,
        variance_log_average=variance_log,
        expected_geometric_average=expected_average,
        monitoring_count=contract.monitoring_count,
        include_initial_spot_in_average=contract.include_initial_spot_in_average,
        deterministic=deterministic,
    )


def _observation_matrix(
    contract: AsianOptionContract,
    market: MarketEnvironment,
    config: AsianSimulationConfig,
) -> tuple[NDArray[np.float64], float, bool]:
    """Return only payoff observations and the measured simulation time."""

    future_times = np.asarray(contract.future_monitoring_times, dtype=np.float64)
    deterministic = market.volatility == 0.0 or future_times.size == 0
    count = 1 if deterministic else config.num_simulations
    started = perf_counter()
    future = simulate_observation_prices(
        contract.spot,
        market,
        future_times,
        count,
        create_rng(config.seed, config.random_engine),
        antithetic=config.variance_reduction.uses_antithetic,
    )
    simulation_ms = (perf_counter() - started) * 1_000.0
    if contract.observation_times[0] == 0.0:
        observations = np.column_stack((np.full(count, contract.spot), future))
    else:
        observations = future
    require_finite(observations, context="Asian simulated observations")
    if observations.shape != (count, contract.monitoring_count) or np.any(observations <= 0.0):
        raise NumericalError("Asian simulator returned invalid observations")
    return observations, simulation_ms, deterministic


class AsianMonteCarloPricer:
    """Price arithmetic or geometric fixed-strike Asians on exact GBM observations."""

    def price(
        self,
        contract: AsianOptionContract,
        market: MarketEnvironment,
        config: AsianSimulationConfig,
    ) -> AsianPricingResult:
        if (
            contract.averaging_type is AveragingType.GEOMETRIC
            and config.variance_reduction.uses_geometric_control
        ):
            raise ExoticPricingError(
                "geometric control variates apply only to ARITHMETIC Asian options"
            )
        started = perf_counter()
        logger.debug("asian_config=%s", config.model_dump(mode="json"))
        observations, simulation_ms, deterministic = _observation_matrix(contract, market, config)
        averages = asian_average(observations, contract.averaging_type)
        raw_payoffs = asian_payoff_from_average(averages, contract)
        discount = finite_exp(
            -market.risk_free_rate * contract.maturity_years,
            context="Asian payoff discount factor",
        )
        discounted = np.asarray(discount * raw_payoffs, dtype=np.float64)
        require_finite(discounted, context="discounted Asian payoffs")
        raw_estimate = sample_mean(discounted)

        raw_price = None
        adjusted_price = None
        raw_se = None
        adjusted_se = None
        raw_variance = None
        adjusted_variance = None
        beta = None
        control_mean = None
        control_expected = None
        variance_ratio = None

        if deterministic:
            estimate = raw_estimate
            estimator_se = 0.0
            if config.variance_reduction.uses_geometric_control:
                control_expected = geometric_asian_analytical_price(contract, market).price
                raw_price = adjusted_price = raw_estimate
                raw_se = adjusted_se = 0.0
                raw_variance = adjusted_variance = 0.0
                beta = 0.0
                control_mean = control_expected
        elif config.variance_reduction.uses_geometric_control:
            geometric_averages = asian_average(observations, AveragingType.GEOMETRIC)
            geometric_contract = contract.model_copy(
                update={"averaging_type": AveragingType.GEOMETRIC}
            )
            control_values = np.asarray(
                discount * asian_payoff_from_average(geometric_averages, geometric_contract),
                dtype=np.float64,
            )
            control_expected = geometric_asian_analytical_price(contract, market).price
            adjustment = apply_control_variate(
                cluster_observations(
                    discounted, antithetic=config.variance_reduction.uses_antithetic
                ),
                cluster_observations(
                    control_values, antithetic=config.variance_reduction.uses_antithetic
                ),
                control_expected,
            )
            estimate = adjustment.adjusted_price
            estimator_se = adjustment.adjusted_standard_error
            raw_price = adjustment.raw_price
            adjusted_price = adjustment.adjusted_price
            raw_se = adjustment.raw_standard_error
            adjusted_se = adjustment.adjusted_standard_error
            raw_variance = adjustment.raw_estimator_variance
            adjusted_variance = adjustment.adjusted_estimator_variance
            beta = adjustment.beta
            control_mean = adjustment.control_mean
            variance_ratio = adjustment.variance_reduction_ratio
        else:
            estimate = raw_estimate
            estimator_se = (
                antithetic_standard_error(discounted)
                if config.variance_reduction is AsianVarianceReduction.ANTITHETIC
                else standard_error(discounted)
            )
        if not math.isfinite(estimate):
            raise NumericalError("Asian Monte Carlo estimate is non-finite")
        low, high = confidence_interval(estimate, estimator_se, config.confidence_level)
        average_std = 0.0 if deterministic else sample_standard_deviation(averages)
        if contract.option_type is OptionType.CALL:
            itm = averages > contract.strike
        else:
            itm = averages < contract.strike
        result = AsianPricingResult(
            price=estimate,
            standard_error=estimator_se,
            estimator_variance=estimator_se * estimator_se,
            confidence_interval_low=low,
            confidence_interval_high=high,
            confidence_level=config.confidence_level,
            num_simulations=config.num_simulations,
            seed=config.seed,
            variance_reduction=config.variance_reduction,
            averaging_type=contract.averaging_type,
            monitoring_count=contract.monitoring_count,
            future_monitoring_count=len(contract.future_monitoring_times),
            include_initial_spot_in_average=contract.include_initial_spot_in_average,
            runtime_ms=(perf_counter() - started) * 1_000.0,
            simulation_runtime_ms=simulation_ms,
            mean_average=sample_mean(averages),
            average_std=average_std,
            probability_itm=float(np.mean(itm)),
            deterministic=deterministic,
            raw_price=raw_price,
            adjusted_price=adjusted_price,
            raw_standard_error=raw_se,
            adjusted_standard_error=adjusted_se,
            raw_estimator_variance=raw_variance,
            adjusted_estimator_variance=adjusted_variance,
            control_variate_beta=beta,
            control_sample_mean=control_mean,
            analytical_geometric_control_price=control_expected,
            variance_reduction_ratio=variance_ratio,
        )
        logger.info(
            "asian_complete averaging=%s method=%s price=%g se=%g simulations=%d runtime_ms=%.3f",
            contract.averaging_type,
            config.variance_reduction,
            result.price,
            result.standard_error,
            config.num_simulations,
            result.runtime_ms,
        )
        return result


def asian_monte_carlo_price(
    contract: AsianOptionContract,
    market: MarketEnvironment,
    config: AsianSimulationConfig | None = None,
) -> AsianPricingResult:
    return AsianMonteCarloPricer().price(contract, market, config or AsianSimulationConfig())
