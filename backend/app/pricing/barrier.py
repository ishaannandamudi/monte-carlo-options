"""Monte Carlo and continuous analytical pricing for single-barrier options."""

import logging
import math
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from scipy.special import log_ndtr
from scipy.stats import norm

from backend.app.analytics.statistics import (
    antithetic_standard_error,
    confidence_interval,
    sample_mean,
    standard_error,
)
from backend.app.core.exceptions import BarrierPricingError, NumericalError
from backend.app.core.numerics import finite_exp, require_finite
from backend.app.models.barrier import (
    BarrierAnalyticalResult,
    BarrierDirection,
    BarrierMonitoringMode,
    BarrierOptionContract,
    BarrierPricingResult,
    BarrierSimulationConfig,
    BarrierStyle,
    BarrierVarianceReduction,
    RebateTiming,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.payoffs.barrier import (
    BarrierEventEvaluation,
    barrier_breaches,
    barrier_vanilla_payoff,
    bridge_discounted_barrier_payoffs,
    brownian_bridge_survival_probabilities,
    discrete_discounted_barrier_payoffs,
    evaluate_discrete_barrier,
)
from backend.app.pricing.black_scholes import black_scholes_price
from backend.app.simulation.gbm import simulate_observation_prices
from backend.app.simulation.random import create_rng

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BarrierSampleEvaluation:
    """Internal shared-path values used by pricing and parity analytics."""

    discounted_payoffs: NDArray[np.float64]
    discounted_vanilla: NDArray[np.float64]
    hit_probability: float
    no_hit_probability: float
    mean_hit_time: float | None
    median_hit_time: float | None
    observations: NDArray[np.float64]
    observation_times: NDArray[np.float64]
    simulation_runtime_ms: float
    deterministic: bool


def _vanilla_contract(contract: BarrierOptionContract) -> OptionContract:
    return OptionContract(
        option_type=contract.option_type,
        exercise_style=contract.exercise_style,
        spot=contract.spot,
        strike=contract.strike,
        maturity_years=contract.maturity_years,
    )


def _monitored_indices(
    contract: BarrierOptionContract,
    all_times: NDArray[np.float64],
) -> NDArray[np.int64]:
    positions = {float(time): index for index, time in enumerate(all_times)}
    try:
        indices = [0, *(positions[time] for time in contract.monitoring_times)]
    except KeyError as exc:  # pragma: no cover - guarded by simulation-time construction
        raise BarrierPricingError(
            "barrier monitoring time is absent from simulated states"
        ) from exc
    return np.asarray(sorted(set(indices)), dtype=np.int64)


def _deterministic_continuous_event(
    contract: BarrierOptionContract,
    market: MarketEnvironment,
) -> BarrierEventEvaluation:
    """Return exact continuous hit status/time for monotone deterministic GBM."""

    if contract.initially_hit:
        hit = True
        hit_time = 0.0
    else:
        carry = market.risk_free_rate - market.dividend_yield
        terminal_log_ratio = carry * contract.maturity_years
        terminal = contract.spot * finite_exp(
            terminal_log_ratio,
            context="deterministic continuous barrier terminal growth",
        )
        terminal_hit = bool(
            barrier_breaches(
                np.asarray([terminal]),
                contract.barrier_direction,
                contract.barrier,
            )[0]
        )
        hit = terminal_hit
        if terminal_hit:
            if carry == 0.0:
                raise BarrierPricingError("constant deterministic path cannot newly hit a barrier")
            hit_time = math.log(contract.barrier / contract.spot) / carry
            hit_time = min(max(hit_time, 0.0), contract.maturity_years)
        else:
            hit_time = math.nan
    return BarrierEventEvaluation(
        hit=np.asarray([hit], dtype=np.bool_),
        first_hit_index=np.asarray([0 if hit_time == 0.0 else (-1)], dtype=np.int64),
        first_hit_time=np.asarray([hit_time], dtype=np.float64),
    )


def _observation_matrix(
    contract: BarrierOptionContract,
    market: MarketEnvironment,
    config: BarrierSimulationConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, bool]:
    future_times = np.asarray(contract.simulation_times, dtype=np.float64)
    deterministic = contract.maturity_years == 0.0 or market.volatility == 0.0
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
    observations = np.column_stack((np.full(count, contract.spot), future))
    times = np.concatenate((np.zeros(1, dtype=np.float64), future_times))
    require_finite(observations, context="barrier simulated observations")
    if np.any(observations <= 0.0):
        raise NumericalError("barrier simulated observations must be strictly positive")
    return observations, times, simulation_ms, deterministic


def evaluate_barrier_samples(
    contract: BarrierOptionContract,
    market: MarketEnvironment,
    config: BarrierSimulationConfig,
) -> BarrierSampleEvaluation:
    """Simulate and evaluate pathwise barrier values for one contract."""

    if contract.initially_hit and contract.barrier_style is BarrierStyle.OUT:
        discount_time = (
            0.0 if contract.rebate_timing is RebateTiming.AT_HIT else contract.maturity_years
        )
        rebate_value = (
            0.0
            if contract.rebate == 0.0
            else contract.rebate
            * finite_exp(
                -market.risk_free_rate * discount_time,
                context="initially hit barrier rebate discount factor",
            )
        )
        return BarrierSampleEvaluation(
            discounted_payoffs=np.asarray([rebate_value], dtype=np.float64),
            discounted_vanilla=np.zeros(1, dtype=np.float64),
            hit_probability=1.0,
            no_hit_probability=0.0,
            mean_hit_time=0.0,
            median_hit_time=0.0,
            observations=np.asarray([[contract.spot]], dtype=np.float64),
            observation_times=np.zeros(1, dtype=np.float64),
            simulation_runtime_ms=0.0,
            deterministic=True,
        )

    observations, times, simulation_ms, deterministic = _observation_matrix(
        contract, market, config
    )
    terminal = observations[:, -1]
    discount = finite_exp(
        -market.risk_free_rate * contract.maturity_years,
        context="barrier vanilla discount factor",
    )
    discounted_vanilla = np.asarray(
        discount * barrier_vanilla_payoff(terminal, contract), dtype=np.float64
    )

    if contract.monitoring_mode is BarrierMonitoringMode.DISCRETE:
        event = evaluate_discrete_barrier(
            observations,
            times,
            _monitored_indices(contract, times),
            contract.barrier_direction,
            contract.barrier,
        )
        discounted = discrete_discounted_barrier_payoffs(
            terminal,
            event,
            contract,
            market.risk_free_rate,
        )
        hit_probability = float(np.mean(event.hit))
        hit_times = event.first_hit_time[event.hit]
        mean_hit_time = float(np.mean(hit_times)) if hit_times.size else None
        median_hit_time = float(np.median(hit_times)) if hit_times.size else None
    elif deterministic:
        event = _deterministic_continuous_event(contract, market)
        discounted = discrete_discounted_barrier_payoffs(
            terminal,
            event,
            contract,
            market.risk_free_rate,
        )
        hit_probability = float(event.hit[0])
        mean_hit_time = float(event.first_hit_time[0]) if event.hit[0] else None
        median_hit_time = mean_hit_time
    else:
        survival = brownian_bridge_survival_probabilities(
            observations,
            times,
            contract.barrier_direction,
            contract.barrier,
            market.volatility,
        )
        discounted = bridge_discounted_barrier_payoffs(
            terminal,
            survival,
            contract,
            market.risk_free_rate,
        )
        hit_probability = 1.0 - float(np.mean(survival))
        mean_hit_time = None
        median_hit_time = None

    hit_probability = min(max(hit_probability, 0.0), 1.0)
    require_finite(discounted, context="discounted barrier estimator observations")
    return BarrierSampleEvaluation(
        discounted_payoffs=discounted,
        discounted_vanilla=discounted_vanilla,
        hit_probability=hit_probability,
        no_hit_probability=1.0 - hit_probability,
        mean_hit_time=mean_hit_time,
        median_hit_time=median_hit_time,
        observations=observations,
        observation_times=times,
        simulation_runtime_ms=simulation_ms,
        deterministic=deterministic,
    )


def _continuous_knock_in_formula(
    contract: BarrierOptionContract,
    market: MarketEnvironment,
) -> float:
    """Return the Reiner-Rubinstein zero-rebate continuously monitored IN value."""

    spot = contract.spot
    strike = contract.strike
    barrier = contract.barrier
    maturity = contract.maturity_years
    volatility = market.volatility
    rate = market.risk_free_rate
    dividend = market.dividend_yield
    root_t = math.sqrt(maturity)
    variance = volatility * volatility
    if variance == 0.0 or not math.isfinite(variance):
        raise BarrierPricingError("continuous barrier volatility variance is not representable")
    mu = (rate - dividend - 0.5 * variance) / variance
    vol_t = volatility * root_t
    if vol_t == 0.0 or not math.isfinite(vol_t):
        raise BarrierPricingError("continuous barrier volatility-time scale is not representable")
    log_sk = math.log(spot / strike)
    log_sh = math.log(spot / barrier)
    log_hs = -log_sh
    x1 = log_sk / vol_t + (1.0 + mu) * vol_t
    x2 = log_sh / vol_t + (1.0 + mu) * vol_t
    y1 = (2.0 * math.log(barrier) - math.log(spot) - math.log(strike)) / vol_t + (1.0 + mu) * vol_t
    y2 = log_hs / vol_t + (1.0 + mu) * vol_t
    phi = 1.0 if contract.option_type is OptionType.CALL else -1.0
    eta = 1.0 if contract.barrier_direction is BarrierDirection.DOWN else -1.0
    spot_discount = finite_exp(-dividend * maturity, context="barrier dividend discount")
    strike_discount = finite_exp(-rate * maturity, context="barrier strike discount")

    def reflected_term(log_scale: float, normal_argument: float) -> float:
        """Evaluate exp(log_scale) N(argument) without a huge reflection factor."""

        log_value = log_scale + float(log_ndtr(normal_argument))
        if log_value == -math.inf:
            return 0.0
        if not math.isfinite(log_value):
            raise BarrierPricingError("barrier reflected term is outside finite range")
        try:
            value = math.exp(log_value)
        except OverflowError as exc:
            raise BarrierPricingError("barrier reflected term overflowed") from exc
        if not math.isfinite(value):
            raise BarrierPricingError("barrier reflected term is non-finite")
        return value

    def a() -> float:
        return phi * spot * spot_discount * float(norm.cdf(phi * x1)) - (
            phi * strike * strike_discount * float(norm.cdf(phi * (x1 - vol_t)))
        )

    def b() -> float:
        return phi * spot * spot_discount * float(norm.cdf(phi * x2)) - (
            phi * strike * strike_discount * float(norm.cdf(phi * (x2 - vol_t)))
        )

    def c() -> float:
        asset_term = reflected_term(
            math.log(spot) - dividend * maturity + 2.0 * (mu + 1.0) * log_hs,
            eta * y1,
        )
        strike_term = reflected_term(
            math.log(strike) - rate * maturity + 2.0 * mu * log_hs,
            eta * (y1 - vol_t),
        )
        return phi * (asset_term - strike_term)

    def d() -> float:
        asset_term = reflected_term(
            math.log(spot) - dividend * maturity + 2.0 * (mu + 1.0) * log_hs,
            eta * y2,
        )
        strike_term = reflected_term(
            math.log(strike) - rate * maturity + 2.0 * mu * log_hs,
            eta * (y2 - vol_t),
        )
        return phi * (asset_term - strike_term)

    strike_above_barrier = strike >= barrier
    if contract.option_type is OptionType.CALL:
        if contract.barrier_direction is BarrierDirection.DOWN:
            value = c() if strike_above_barrier else a() - b() + d()
        else:
            value = a() if strike_above_barrier else b() - c() + d()
    elif contract.barrier_direction is BarrierDirection.DOWN:
        value = b() - c() + d() if strike_above_barrier else a()
    else:
        value = a() - b() + d() if strike_above_barrier else c()

    vanilla = black_scholes_price(_vanilla_contract(contract), market)
    tolerance = 1e-10 * max(1.0, vanilla)
    if not math.isfinite(value) or value < -tolerance or value > vanilla + tolerance:
        raise BarrierPricingError("continuous barrier formula produced an invalid knock-in value")
    return min(max(value, 0.0), vanilla)


def continuous_barrier_analytical_price(
    contract: BarrierOptionContract,
    market: MarketEnvironment,
) -> BarrierAnalyticalResult:
    """Price any zero-rebate continuously monitored single barrier under GBM."""

    started = perf_counter()
    if contract.rebate != 0.0:
        raise BarrierPricingError("continuous analytical barrier pricing supports zero rebate only")
    volatility_scale = market.volatility * math.sqrt(contract.maturity_years)
    deterministic_inputs = (
        contract.maturity_years == 0.0
        or market.volatility == 0.0
        or volatility_scale == 0.0
        or market.volatility * market.volatility == 0.0
    )
    if contract.initially_hit and contract.barrier_style is BarrierStyle.OUT:
        price = 0.0
        deterministic = True
    else:
        vanilla = black_scholes_price(_vanilla_contract(contract), market)
        deterministic = deterministic_inputs
        if contract.initially_hit:
            price = vanilla
        elif deterministic:
            event = _deterministic_continuous_event(contract, market)
            activated = bool(event.hit[0])
            if contract.barrier_style is BarrierStyle.IN:
                price = vanilla if activated else 0.0
            else:
                price = 0.0 if activated else vanilla
        else:
            knock_in = _continuous_knock_in_formula(contract, market)
            price = knock_in if contract.barrier_style is BarrierStyle.IN else vanilla - knock_in
    if not math.isfinite(price) or price < 0.0:
        raise BarrierPricingError("continuous analytical barrier price is invalid")
    return BarrierAnalyticalResult(
        price=0.0 if price == 0.0 else max(price, 0.0),
        barrier_direction=contract.barrier_direction,
        barrier_style=contract.barrier_style,
        barrier=contract.barrier,
        rebate=contract.rebate,
        initially_hit=contract.initially_hit,
        deterministic=deterministic,
        runtime_ms=(perf_counter() - started) * 1_000.0,
    )


class BarrierMonteCarloPricer:
    """Price discrete or bridge-weighted single barriers on exact GBM observations."""

    def price(
        self,
        contract: BarrierOptionContract,
        market: MarketEnvironment,
        config: BarrierSimulationConfig,
    ) -> BarrierPricingResult:
        started = perf_counter()
        sample = evaluate_barrier_samples(contract, market, config)
        price = sample_mean(sample.discounted_payoffs)
        if sample.deterministic or np.all(
            sample.discounted_payoffs == sample.discounted_payoffs[0]
        ):
            estimator_se = 0.0
        elif config.variance_reduction is BarrierVarianceReduction.ANTITHETIC:
            estimator_se = antithetic_standard_error(sample.discounted_payoffs)
        else:
            estimator_se = standard_error(sample.discounted_payoffs)
        low, high = confidence_interval(price, estimator_se, config.confidence_level)
        analytical = (
            continuous_barrier_analytical_price(contract, market).price
            if contract.rebate == 0.0
            else None
        )
        activation_probability = (
            sample.hit_probability
            if contract.barrier_style is BarrierStyle.IN
            else sample.no_hit_probability
        )
        result = BarrierPricingResult(
            price=price,
            standard_error=estimator_se,
            estimator_variance=estimator_se * estimator_se,
            confidence_interval_low=low,
            confidence_interval_high=high,
            confidence_level=config.confidence_level,
            num_simulations=config.num_simulations,
            seed=config.seed,
            variance_reduction=config.variance_reduction,
            barrier_direction=contract.barrier_direction,
            barrier_style=contract.barrier_style,
            barrier=contract.barrier,
            monitoring_mode=contract.monitoring_mode,
            monitoring_count=contract.monitoring_count,
            simulation_state_count=contract.simulation_state_count,
            rebate=contract.rebate,
            rebate_timing=contract.rebate_timing,
            hit_probability=sample.hit_probability,
            no_hit_probability=sample.no_hit_probability,
            activation_probability=activation_probability,
            mean_hit_time=sample.mean_hit_time,
            median_hit_time=sample.median_hit_time,
            initially_hit=contract.initially_hit,
            deterministic=sample.deterministic,
            survival_probability_weighting=(
                contract.monitoring_mode is BarrierMonitoringMode.BROWNIAN_BRIDGE
                and not sample.deterministic
            ),
            runtime_ms=(perf_counter() - started) * 1_000.0,
            simulation_runtime_ms=sample.simulation_runtime_ms,
            analytical_continuous_price=analytical,
        )
        logger.info(
            "barrier_complete direction=%s style=%s monitoring=%s price=%g se=%g "
            "simulations=%d runtime_ms=%.3f",
            contract.barrier_direction,
            contract.barrier_style,
            contract.monitoring_mode,
            result.price,
            result.standard_error,
            config.num_simulations,
            result.runtime_ms,
        )
        return result


def barrier_monte_carlo_price(
    contract: BarrierOptionContract,
    market: MarketEnvironment,
    config: BarrierSimulationConfig | None = None,
) -> BarrierPricingResult:
    return BarrierMonteCarloPricer().price(
        contract,
        market,
        config or BarrierSimulationConfig(),
    )
