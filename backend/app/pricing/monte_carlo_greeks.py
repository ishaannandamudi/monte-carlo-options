"""CRN finite-difference and pathwise Greeks for terminal risk-neutral GBM."""

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from backend.app.analytics.statistics import confidence_interval
from backend.app.core.exceptions import NumericalError, UnsupportedGreekMethodError
from backend.app.core.numerics import finite_exp, require_finite
from backend.app.models.greeks import DifferenceScheme, Greek, GreekBumpConfig, GreekMethod
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.models.results import (
    FiniteDifferenceGreekResult,
    MonteCarloGreeksResult,
    PathwiseGreekResult,
)
from backend.app.models.simulation import SimulationConfig, VarianceReduction
from backend.app.payoffs.vanilla import vanilla_payoff
from backend.app.pricing.base import require_european_exercise
from backend.app.pricing.variance_reduction import (
    ClusteredObservations,
    apply_control_variate,
    cluster_observations,
    discounted_stock_control,
    estimator_variance,
    weighted_estimate,
)
from backend.app.simulation.gbm import standard_normal_drivers, terminal_prices_from_normals
from backend.app.simulation.random import create_rng

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ValuationSample:
    observations: ClusteredObservations
    price: float
    deterministic: bool


def _contract_with(contract: OptionContract, **updates: float) -> OptionContract:
    values = contract.model_dump()
    values.update(updates)
    return OptionContract.model_validate(values)


def _market_with(market: MarketEnvironment, **updates: float) -> MarketEnvironment:
    values = market.model_dump()
    values.update(updates)
    return MarketEnvironment.model_validate(values)


def _shift(value: float, bump: float, direction: float, *, context: str) -> float:
    shifted = value + direction * bump
    if not math.isfinite(shifted) or shifted == value:
        raise NumericalError(f"{context} bump is outside representable floating-point range")
    return shifted


def _inverse(value: float, *, context: str) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise NumericalError(f"{context} denominator is outside positive floating-point range")
    inverse = 1.0 / value
    if not math.isfinite(inverse):
        raise NumericalError(f"{context} reciprocal overflowed floating-point range")
    return inverse


def _valuation_sample(
    contract: OptionContract,
    market: MarketEnvironment,
    config: SimulationConfig,
    normals: NDArray[np.float64],
) -> _ValuationSample:
    terminal_prices = terminal_prices_from_normals(contract, market, normals)
    payoffs = vanilla_payoff(terminal_prices, contract)
    discount = finite_exp(
        -market.risk_free_rate * contract.maturity_years,
        context="Greek payoff discount factor",
    )
    discounted_payoffs = np.asarray(discount * payoffs, dtype=np.float64)
    require_finite(discounted_payoffs, context="Greek discounted payoffs")
    raw = cluster_observations(
        discounted_payoffs,
        antithetic=config.variance_reduction.uses_antithetic,
    )
    deterministic = contract.maturity_years == 0.0 or market.volatility == 0.0
    if config.variance_reduction.uses_control_variate and not deterministic:
        control_values, expected_control = discounted_stock_control(
            terminal_prices,
            contract,
            market,
        )
        control = cluster_observations(
            control_values,
            antithetic=config.variance_reduction.uses_antithetic,
        )
        adjustment = apply_control_variate(raw, control, expected_control)
        observations = ClusteredObservations(
            values=adjustment.adjusted_observations,
            sizes=raw.sizes,
        )
        return _ValuationSample(
            observations=observations,
            price=adjustment.adjusted_price,
            deterministic=False,
        )
    return _ValuationSample(
        observations=raw,
        price=weighted_estimate(raw),
        deterministic=deterministic,
    )


def _combine_samples(
    samples: Sequence[_ValuationSample],
    coefficients: Sequence[float],
) -> tuple[float, float]:
    if len(samples) != len(coefficients) or not samples:
        raise ValueError("samples and coefficients must have the same positive length")
    sizes = samples[0].observations.sizes
    if not all(np.array_equal(sample.observations.sizes, sizes) for sample in samples[1:]):
        raise NumericalError("Greek valuation samples do not share identical clusters")

    values = np.zeros_like(samples[0].observations.values)
    for sample, coefficient in zip(samples, coefficients, strict=True):
        values += coefficient * sample.observations.values
    require_finite(values, context="finite-difference Greek observations")
    observations = ClusteredObservations(values=values, sizes=sizes)
    estimate = weighted_estimate(observations)
    if observations.values.size < 2 and all(sample.deterministic for sample in samples):
        return estimate, 0.0
    variance = estimator_variance(observations)
    return estimate, variance


def _finite_difference_result(
    *,
    greek: Greek,
    scheme: DifferenceScheme,
    bump_size: float,
    base: _ValuationSample,
    upward: _ValuationSample,
    downward: _ValuationSample | None,
    coefficients: Sequence[float],
    config: SimulationConfig,
    runtime_ms: float,
) -> FiniteDifferenceGreekResult:
    samples = [upward, base] if downward is None else [upward, base, downward]
    estimate, variance = _combine_samples(samples, coefficients)
    standard_error = math.sqrt(variance)
    low, high = confidence_interval(estimate, standard_error, config.confidence_level)
    return FiniteDifferenceGreekResult(
        greek=greek,
        estimate=estimate,
        difference_scheme=scheme,
        bump_size=bump_size,
        base_price=base.price,
        upward_price=upward.price,
        downward_price=None if downward is None else downward.price,
        standard_error=standard_error,
        estimator_variance=variance,
        confidence_interval_low=low,
        confidence_interval_high=high,
        confidence_level=config.confidence_level,
        num_simulations=config.num_simulations,
        seed=config.seed,
        variance_reduction=config.variance_reduction,
        runtime_ms=runtime_ms,
    )


def finite_difference_greeks(
    contract: OptionContract,
    market: MarketEnvironment,
    config: SimulationConfig,
    greeks: Sequence[Greek],
    bump_config: GreekBumpConfig | None = None,
) -> list[FiniteDifferenceGreekResult]:
    """Estimate requested Greeks from direct CRN cluster differences."""

    require_european_exercise(contract, engine="Monte Carlo Greeks")
    requested = list(greeks)
    if not requested:
        raise ValueError("at least one Greek is required")
    bumps = bump_config or GreekBumpConfig()
    logger.debug("finite_difference_bumps=%s", bumps.model_dump(mode="json"))
    setup_started = perf_counter()
    rng = create_rng(config.seed, config.random_engine)
    normals = standard_normal_drivers(
        config.num_simulations,
        rng,
        antithetic=config.variance_reduction.uses_antithetic,
    )
    base = _valuation_sample(contract, market, config, normals)
    setup_ms = (perf_counter() - setup_started) * 1_000.0
    results: list[FiniteDifferenceGreekResult] = []

    for greek in requested:
        started = perf_counter()
        downward: _ValuationSample | None
        coefficients: Sequence[float]
        if greek in {Greek.DELTA, Greek.GAMMA}:
            bump = contract.spot * bumps.spot_relative_bump
            upward_spot = _shift(contract.spot, bump, 1.0, context="spot")
            downward_spot = _shift(contract.spot, bump, -1.0, context="spot")
            if downward_spot <= 0.0:
                raise NumericalError("downward spot bump must remain positive")
            upward = _valuation_sample(
                _contract_with(contract, spot=upward_spot),
                market,
                config,
                normals,
            )
            downward = _valuation_sample(
                _contract_with(contract, spot=downward_spot),
                market,
                config,
                normals,
            )
            if greek is Greek.DELTA:
                inverse = _inverse(bump, context="Delta bump")
                coefficients = (0.5 * inverse, 0.0, -0.5 * inverse)
            else:
                inverse_square = _inverse(bump * bump, context="Gamma bump squared")
                coefficients = (inverse_square, -2.0 * inverse_square, inverse_square)
            scheme = DifferenceScheme.CENTRAL
        elif greek is Greek.VEGA:
            bump = bumps.volatility_absolute_bump
            upward_volatility = _shift(market.volatility, bump, 1.0, context="volatility")
            upward = _valuation_sample(
                contract,
                _market_with(market, volatility=upward_volatility),
                config,
                normals,
            )
            inverse = _inverse(bump, context="Vega bump")
            if market.volatility >= bump:
                downward_volatility = _shift(
                    market.volatility,
                    bump,
                    -1.0,
                    context="volatility",
                )
                downward = _valuation_sample(
                    contract,
                    _market_with(market, volatility=downward_volatility),
                    config,
                    normals,
                )
                coefficients = (0.5 * inverse, 0.0, -0.5 * inverse)
                scheme = DifferenceScheme.CENTRAL
            else:
                downward = None
                coefficients = (inverse, -inverse)
                scheme = DifferenceScheme.FORWARD
        elif greek is Greek.THETA:
            bump = bumps.maturity_absolute_bump
            upward_maturity = _shift(
                contract.maturity_years,
                bump,
                1.0,
                context="maturity",
            )
            upward = _valuation_sample(
                _contract_with(contract, maturity_years=upward_maturity),
                market,
                config,
                normals,
            )
            inverse = _inverse(bump, context="Theta bump")
            if contract.maturity_years >= bump:
                downward_maturity = _shift(
                    contract.maturity_years,
                    bump,
                    -1.0,
                    context="maturity",
                )
                downward = _valuation_sample(
                    _contract_with(contract, maturity_years=downward_maturity),
                    market,
                    config,
                    normals,
                )
                coefficients = (-0.5 * inverse, 0.0, 0.5 * inverse)
                scheme = DifferenceScheme.CENTRAL
            else:
                downward = None
                coefficients = (-inverse, inverse)
                scheme = DifferenceScheme.FORWARD
        elif greek is Greek.RHO:
            bump = bumps.rate_absolute_bump
            upward_rate = _shift(market.risk_free_rate, bump, 1.0, context="risk-free rate")
            downward_rate = _shift(
                market.risk_free_rate,
                bump,
                -1.0,
                context="risk-free rate",
            )
            upward = _valuation_sample(
                contract,
                _market_with(market, risk_free_rate=upward_rate),
                config,
                normals,
            )
            downward = _valuation_sample(
                contract,
                _market_with(market, risk_free_rate=downward_rate),
                config,
                normals,
            )
            inverse = _inverse(bump, context="Rho bump")
            coefficients = (0.5 * inverse, 0.0, -0.5 * inverse)
            scheme = DifferenceScheme.CENTRAL
        else:
            bump = bumps.dividend_yield_absolute_bump
            upward_yield = _shift(
                market.dividend_yield,
                bump,
                1.0,
                context="dividend yield",
            )
            downward_yield = _shift(
                market.dividend_yield,
                bump,
                -1.0,
                context="dividend yield",
            )
            upward = _valuation_sample(
                contract,
                _market_with(market, dividend_yield=upward_yield),
                config,
                normals,
            )
            downward = _valuation_sample(
                contract,
                _market_with(market, dividend_yield=downward_yield),
                config,
                normals,
            )
            inverse = _inverse(bump, context="dividend Rho bump")
            coefficients = (0.5 * inverse, 0.0, -0.5 * inverse)
            scheme = DifferenceScheme.CENTRAL

        runtime_ms = setup_ms + (perf_counter() - started) * 1_000.0
        result = _finite_difference_result(
            greek=greek,
            scheme=scheme,
            bump_size=bump,
            base=base,
            upward=upward,
            downward=downward,
            coefficients=coefficients,
            config=config,
            runtime_ms=runtime_ms,
        )
        logger.info(
            "greek_complete method=%s greek=%s simulations=%d runtime_ms=%.3f",
            GreekMethod.FINITE_DIFFERENCE.value,
            greek.value,
            config.num_simulations,
            runtime_ms,
        )
        results.append(result)
    return results


def _pathwise_result(
    greek: Greek,
    path_values: NDArray[np.float64],
    config: SimulationConfig,
    *,
    deterministic: bool,
    runtime_ms: float,
) -> PathwiseGreekResult:
    clusters = cluster_observations(
        path_values,
        antithetic=config.variance_reduction.uses_antithetic,
    )
    estimate = weighted_estimate(clusters)
    variance = 0.0 if clusters.values.size < 2 and deterministic else estimator_variance(clusters)
    standard_error = math.sqrt(variance)
    low, high = confidence_interval(estimate, standard_error, config.confidence_level)
    return PathwiseGreekResult(
        greek=greek,
        estimate=estimate,
        standard_error=standard_error,
        estimator_variance=variance,
        confidence_interval_low=low,
        confidence_interval_high=high,
        confidence_level=config.confidence_level,
        num_simulations=config.num_simulations,
        seed=config.seed,
        variance_reduction=config.variance_reduction,
        runtime_ms=runtime_ms,
    )


def pathwise_greeks(
    contract: OptionContract,
    market: MarketEnvironment,
    config: SimulationConfig,
    greeks: Sequence[Greek],
) -> list[PathwiseGreekResult]:
    """Estimate pathwise Delta/Vega with exact antithetic cluster uncertainty."""

    require_european_exercise(contract, engine="Monte Carlo Greeks")
    requested = list(greeks)
    if not requested:
        raise ValueError("at least one Greek is required")
    unsupported = [greek for greek in requested if greek not in {Greek.DELTA, Greek.VEGA}]
    if unsupported:
        names = ", ".join(greek.value for greek in unsupported)
        raise UnsupportedGreekMethodError(f"PATHWISE does not support: {names}")
    if config.variance_reduction not in {
        VarianceReduction.NONE,
        VarianceReduction.ANTITHETIC,
    }:
        raise UnsupportedGreekMethodError("PATHWISE supports variance_reduction NONE or ANTITHETIC")

    started = perf_counter()
    rng = create_rng(config.seed, config.random_engine)
    normals = standard_normal_drivers(
        config.num_simulations,
        rng,
        antithetic=config.variance_reduction.uses_antithetic,
    )
    terminal_prices = terminal_prices_from_normals(contract, market, normals)
    maturity = contract.maturity_years
    discount = finite_exp(-market.risk_free_rate * maturity, context="pathwise discount factor")
    if contract.option_type is OptionType.CALL:
        payoff_derivative = np.asarray(terminal_prices > contract.strike, dtype=np.float64)
    else:
        payoff_derivative = -np.asarray(terminal_prices < contract.strike, dtype=np.float64)

    elapsed_setup_ms = (perf_counter() - started) * 1_000.0
    results: list[PathwiseGreekResult] = []
    for greek in requested:
        greek_started = perf_counter()
        if greek is Greek.DELTA:
            values = np.asarray(
                discount * payoff_derivative * terminal_prices / contract.spot,
                dtype=np.float64,
            )
            deterministic = maturity == 0.0 or market.volatility == 0.0
        else:
            terminal_vega = terminal_prices * (
                -market.volatility * maturity + math.sqrt(maturity) * normals
            )
            values = np.asarray(
                discount * payoff_derivative * terminal_vega,
                dtype=np.float64,
            )
            deterministic = maturity == 0.0
        require_finite(values, context=f"pathwise {greek.value} observations")
        runtime_ms = elapsed_setup_ms + (perf_counter() - greek_started) * 1_000.0
        result = _pathwise_result(
            greek,
            values,
            config,
            deterministic=deterministic,
            runtime_ms=runtime_ms,
        )
        logger.info(
            "greek_complete method=%s greek=%s simulations=%d runtime_ms=%.3f",
            GreekMethod.PATHWISE.value,
            greek.value,
            config.num_simulations,
            runtime_ms,
        )
        results.append(result)
    return results


def monte_carlo_greeks(
    contract: OptionContract,
    market: MarketEnvironment,
    config: SimulationConfig,
    greeks: Sequence[Greek],
    method: GreekMethod,
    bump_config: GreekBumpConfig | None = None,
) -> MonteCarloGreeksResult:
    """Dispatch one supported Monte Carlo Greek estimator method."""

    require_european_exercise(contract, engine="Monte Carlo Greeks")
    if method is GreekMethod.FINITE_DIFFERENCE:
        results: list[FiniteDifferenceGreekResult | PathwiseGreekResult] = list(
            finite_difference_greeks(contract, market, config, greeks, bump_config)
        )
    elif method is GreekMethod.PATHWISE:
        results = list(pathwise_greeks(contract, market, config, greeks))
    else:
        raise UnsupportedGreekMethodError(
            "Monte Carlo Greeks support FINITE_DIFFERENCE or PATHWISE"
        )
    return MonteCarloGreeksResult(method=method, results=results)
