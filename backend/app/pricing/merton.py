"""Semi-analytical and Monte Carlo pricing under Merton jump diffusion."""

import logging
import math
from collections.abc import Callable
from time import perf_counter

import numpy as np
from scipy.stats import norm, poisson

from backend.app.analytics.statistics import (
    confidence_interval,
    sample_mean,
    sample_standard_deviation,
    sample_variance,
    standard_error,
)
from backend.app.core.exceptions import MertonPricingError
from backend.app.core.numerics import finite_exp, require_finite
from backend.app.models.market import MarketEnvironment
from backend.app.models.merton import (
    MertonAnalyticalMethod,
    MertonAnalyticalResult,
    MertonGreekBumps,
    MertonGreekResult,
    MertonJumpParameters,
    MertonMonteCarloResult,
    MertonSeriesConfig,
    MertonSimulationConfig,
    MertonSimulationScheme,
)
from backend.app.models.option import OptionContract, OptionType
from backend.app.payoffs.vanilla import intrinsic_value, vanilla_payoff
from backend.app.pricing.base import require_european_exercise
from backend.app.pricing.black_scholes import black_scholes_price
from backend.app.simulation.merton import simulate_merton_terminal
from backend.app.simulation.random import create_rng

logger = logging.getLogger(__name__)


def price_lognormal_terminal_distribution(
    log_mean: float,
    log_variance: float,
    strike: float,
    rate: float,
    maturity: float,
    option_type: OptionType,
) -> float:
    """Discount a vanilla payoff when terminal log spot is normally distributed."""

    values = (log_mean, log_variance, strike, rate, maturity)
    if not all(math.isfinite(value) for value in values):
        raise MertonPricingError("conditional lognormal inputs must be finite")
    if log_variance < 0.0 or strike <= 0.0 or maturity < 0.0:
        raise MertonPricingError("conditional lognormal inputs are outside their domains")
    discount = finite_exp(-rate * maturity, context="Merton discount factor")
    if log_variance == 0.0:
        terminal = finite_exp(log_mean, context="conditional deterministic terminal spot")
        return discount * intrinsic_value(option_type, terminal, strike)
    root_variance = math.sqrt(log_variance)
    d2 = (log_mean - math.log(strike)) / root_variance
    d1 = d2 + root_variance
    expected_spot = finite_exp(
        log_mean + 0.5 * log_variance,
        context="conditional lognormal terminal mean",
    )
    if option_type is OptionType.CALL:
        undiscounted = expected_spot * float(norm.cdf(d1)) - strike * float(norm.cdf(d2))
    else:
        undiscounted = strike * float(norm.cdf(-d2)) - expected_spot * float(norm.cdf(-d1))
    value = discount * undiscounted
    if not math.isfinite(value):
        raise MertonPricingError("conditional lognormal price is non-finite")
    return max(value, 0.0)


def _series_term_count(mean_count: float, config: MertonSeriesConfig) -> int:
    if mean_count == 0.0:
        return 1
    quantile = float(poisson.ppf(1.0 - config.tail_probability_tolerance, mean_count))
    if not math.isfinite(quantile):
        return config.max_terms
    return min(config.max_terms, max(1, int(quantile) + 1))


def merton_analytical_price(
    contract: OptionContract,
    market: MarketEnvironment,
    parameters: MertonJumpParameters,
    config: MertonSeriesConfig | None = None,
) -> MertonAnalyticalResult:
    """Price a European vanilla option as a truncated conditional Poisson mixture."""

    require_european_exercise(contract, engine="Merton analytical pricing")
    series = config or MertonSeriesConfig()
    started = perf_counter()
    maturity = contract.maturity_years
    try:
        compensator = parameters.jump_compensator
    except ValueError as exc:
        raise MertonPricingError(str(exc)) from exc
    expected_count = parameters.jump_intensity * maturity
    if not math.isfinite(expected_count):
        raise MertonPricingError("expected Merton jump count must be finite")
    if maturity == 0.0:
        return MertonAnalyticalResult(
            price=intrinsic_value(contract.option_type, contract.spot, contract.strike),
            terms_used=1,
            omitted_poisson_mass=0.0,
            runtime_ms=(perf_counter() - started) * 1_000.0,
            converged=True,
            jump_compensator=compensator,
            expected_jump_count=0.0,
            method=MertonAnalyticalMethod.INTRINSIC,
        )
    if parameters.jump_intensity == 0.0:
        return MertonAnalyticalResult(
            price=black_scholes_price(contract, market),
            terms_used=1,
            omitted_poisson_mass=0.0,
            runtime_ms=(perf_counter() - started) * 1_000.0,
            converged=True,
            jump_compensator=compensator,
            expected_jump_count=0.0,
            method=MertonAnalyticalMethod.BLACK_SCHOLES_LIMIT,
        )

    terms = _series_term_count(expected_count, series)
    indices = np.arange(terms, dtype=np.int64)
    weights = np.asarray(poisson.pmf(indices, expected_count), dtype=np.float64)
    require_finite(weights, context="Merton Poisson mixture weights")
    base_log_mean = (
        math.log(contract.spot)
        + (
            market.risk_free_rate
            - market.dividend_yield
            - parameters.jump_intensity * compensator
            - 0.5 * market.volatility * market.volatility
        )
        * maturity
    )
    diffusion_variance = market.volatility * market.volatility * maturity
    prices = np.asarray(
        [
            price_lognormal_terminal_distribution(
                base_log_mean + int(index) * parameters.jump_mean,
                diffusion_variance
                + int(index) * parameters.jump_volatility * parameters.jump_volatility,
                contract.strike,
                market.risk_free_rate,
                maturity,
                contract.option_type,
            )
            for index in indices
        ],
        dtype=np.float64,
    )
    value = float(np.dot(weights, prices))
    omitted = float(poisson.sf(terms - 1, expected_count))
    if not math.isfinite(value) or not math.isfinite(omitted):
        raise MertonPricingError("Merton analytical mixture produced a non-finite result")
    result = MertonAnalyticalResult(
        price=max(value, 0.0),
        terms_used=terms,
        omitted_poisson_mass=min(max(omitted, 0.0), 1.0),
        runtime_ms=(perf_counter() - started) * 1_000.0,
        converged=omitted <= series.tail_probability_tolerance,
        jump_compensator=compensator,
        expected_jump_count=expected_count,
        method=MertonAnalyticalMethod.POISSON_MIXTURE,
    )
    logger.info(
        "merton_analytical_complete terms=%d omitted_mass=%g price=%g runtime_ms=%.3f",
        result.terms_used,
        result.omitted_poisson_mass,
        result.price,
        result.runtime_ms,
    )
    return result


def merton_monte_carlo_price(
    contract: OptionContract,
    market: MarketEnvironment,
    parameters: MertonJumpParameters,
    config: MertonSimulationConfig | None = None,
) -> MertonMonteCarloResult:
    """Price a European payoff from exact Merton terminal draws."""

    require_european_exercise(contract, engine="Merton Monte Carlo pricing")
    simulation_config = config or MertonSimulationConfig()
    started = perf_counter()
    simulation_started = perf_counter()
    simulation = simulate_merton_terminal(
        contract,
        market,
        parameters,
        simulation_config,
        create_rng(simulation_config.seed, simulation_config.random_engine),
    )
    simulation_ms = (perf_counter() - simulation_started) * 1_000.0
    spots, counts = simulation.terminal_spots, simulation.jump_counts
    discount = finite_exp(
        -market.risk_free_rate * contract.maturity_years,
        context="Merton payoff discount factor",
    )
    discounted = np.asarray(discount * vanilla_payoff(spots, contract), dtype=np.float64)
    require_finite(discounted, context="discounted Merton payoffs")
    price = sample_mean(discounted)
    one_path = simulation_config.num_simulations == 1
    se = 0.0 if one_path or np.all(discounted == discounted[0]) else standard_error(discounted)
    spot_std = 0.0 if one_path or np.all(spots == spots[0]) else sample_standard_deviation(spots)
    count_variance = 0.0 if one_path or np.all(counts == counts[0]) else sample_variance(counts)
    low, high = confidence_interval(price, se, simulation_config.confidence_level)
    itm = (
        spots > contract.strike
        if contract.option_type is OptionType.CALL
        else spots < contract.strike
    )
    zero_fraction = float(np.mean(counts == 0))
    result = MertonMonteCarloResult(
        price=price,
        standard_error=se,
        estimator_variance=se * se,
        confidence_interval=(low, high),
        confidence_interval_low=low,
        confidence_interval_high=high,
        confidence_level=simulation_config.confidence_level,
        num_simulations=simulation_config.num_simulations,
        seed=simulation_config.seed,
        runtime_ms=(perf_counter() - started) * 1_000.0,
        simulation_runtime_ms=simulation_ms,
        mean_terminal_spot=sample_mean(spots),
        std_terminal_spot=spot_std,
        probability_itm=float(np.mean(itm)),
        expected_jump_count=parameters.jump_intensity * contract.maturity_years,
        mean_jump_count=float(np.mean(counts)),
        jump_count_variance=count_variance,
        zero_jump_fraction=zero_fraction,
        at_least_one_jump_fraction=1.0 - zero_fraction,
        max_observed_jumps=int(np.max(counts)),
        scheme=MertonSimulationScheme.COMPOUND_POISSON_EXACT,
    )
    logger.info(
        "merton_mc_complete simulations=%d price=%g se=%g runtime_ms=%.3f",
        result.num_simulations,
        result.price,
        result.standard_error,
        result.runtime_ms,
    )
    return result


def _central_or_forward(
    function: Callable[[float], float], value: float, bump: float, *, lower: float | None = None
) -> float:
    if lower is not None and value - bump < lower:
        return (function(value + bump) - function(value)) / bump
    return (function(value + bump) - function(value - bump)) / (2.0 * bump)


def merton_analytical_greeks(
    contract: OptionContract,
    market: MarketEnvironment,
    parameters: MertonJumpParameters,
    series: MertonSeriesConfig | None = None,
    bumps: MertonGreekBumps | None = None,
) -> MertonGreekResult:
    """Finite-difference standard Greeks and explicitly named jump sensitivities."""

    cfg, bump = series or MertonSeriesConfig(), bumps or MertonGreekBumps()

    def price(
        price_contract: OptionContract,
        price_market: MarketEnvironment,
        price_parameters: MertonJumpParameters,
    ) -> float:
        return merton_analytical_price(price_contract, price_market, price_parameters, cfg).price

    base = price(contract, market, parameters)
    spot_bump = contract.spot * bump.relative_spot_bump
    up = price(contract.model_copy(update={"spot": contract.spot + spot_bump}), market, parameters)
    down = price(
        contract.model_copy(update={"spot": contract.spot - spot_bump}), market, parameters
    )
    delta = (up - down) / (2.0 * spot_bump)
    gamma = (up - 2.0 * base + down) / (spot_bump * spot_bump)
    diffusion_vega = _central_or_forward(
        lambda value: price(contract, market.model_copy(update={"volatility": value}), parameters),
        market.volatility,
        bump.volatility_bump,
        lower=0.0,
    )
    if contract.maturity_years <= bump.maturity_bump:
        theta = (
            -(
                price(
                    contract.model_copy(
                        update={"maturity_years": contract.maturity_years + bump.maturity_bump}
                    ),
                    market,
                    parameters,
                )
                - base
            )
            / bump.maturity_bump
        )
    else:
        theta = -_central_or_forward(
            lambda value: price(
                contract.model_copy(update={"maturity_years": value}), market, parameters
            ),
            contract.maturity_years,
            bump.maturity_bump,
        )
    rho = _central_or_forward(
        lambda value: price(
            contract, market.model_copy(update={"risk_free_rate": value}), parameters
        ),
        market.risk_free_rate,
        bump.rate_bump,
    )

    def parameter_price(name: str, value: float) -> float:
        return price(contract, market, parameters.model_copy(update={name: value}))

    intensity = _central_or_forward(
        lambda value: parameter_price("jump_intensity", value),
        parameters.jump_intensity,
        bump.jump_intensity_bump,
        lower=0.0,
    )
    jump_mean = _central_or_forward(
        lambda value: parameter_price("jump_mean", value),
        parameters.jump_mean,
        bump.jump_mean_bump,
    )
    jump_vol = _central_or_forward(
        lambda value: parameter_price("jump_volatility", value),
        parameters.jump_volatility,
        bump.jump_volatility_bump,
        lower=0.0,
    )
    values = (delta, gamma, diffusion_vega, theta, rho, intensity, jump_mean, jump_vol)
    if not all(math.isfinite(value) for value in values):
        raise MertonPricingError("Merton finite-difference Greeks are non-finite")
    return MertonGreekResult(
        price=base,
        delta=delta,
        gamma=gamma,
        diffusion_vega=diffusion_vega,
        theta=theta,
        rho=rho,
        jump_intensity_sensitivity=intensity,
        jump_mean_sensitivity=jump_mean,
        jump_vol_sensitivity=jump_vol,
    )
