"""Black-Scholes implied-volatility inversion for European vanilla options."""

import math
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import brentq

from backend.app.core.exceptions import (
    ArbitrageBoundsError,
    ImpliedVolatilityBracketError,
    ImpliedVolatilityConvergenceError,
    NumericalError,
    QuantitativeError,
)
from backend.app.core.numerics import finite_exp
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.models.results import ImpliedVolatilityResult
from backend.app.models.volatility import (
    ImpliedVolatilityConfig,
    ImpliedVolatilityMarket,
    ImpliedVolatilityMethod,
    ImpliedVolatilityObservation,
    ImpliedVolatilityStatus,
)
from backend.app.pricing.black_scholes import black_scholes_price
from backend.app.pricing.black_scholes_greeks import black_scholes_greeks


@dataclass(frozen=True)
class EuropeanPriceBounds:
    """Closed no-arbitrage interval for a European vanilla option price."""

    lower: float
    upper: float


def _discounted_values(
    spot: float,
    strike: float,
    maturity_years: float,
    risk_free_rate: float,
    dividend_yield: float,
) -> tuple[float, float]:
    if not all(
        math.isfinite(value)
        for value in (spot, strike, maturity_years, risk_free_rate, dividend_yield)
    ):
        raise NumericalError("price-bound inputs must be finite")
    if spot <= 0.0 or strike <= 0.0 or maturity_years < 0.0:
        raise NumericalError("price bounds require positive spot/strike and non-negative maturity")
    discounted_spot = spot * finite_exp(
        -dividend_yield * maturity_years,
        context="dividend discount factor for price bounds",
    )
    discounted_strike = strike * finite_exp(
        -risk_free_rate * maturity_years,
        context="risk-free discount factor for price bounds",
    )
    if not math.isfinite(discounted_spot) or not math.isfinite(discounted_strike):
        raise NumericalError("European price bounds overflowed floating-point range")
    return discounted_spot, discounted_strike


def call_price_bounds(
    spot: float,
    strike: float,
    maturity_years: float,
    risk_free_rate: float,
    dividend_yield: float,
) -> EuropeanPriceBounds:
    """Return call bounds ``max(S e^-qT - K e^-rT, 0)`` and ``S e^-qT``."""

    discounted_spot, discounted_strike = _discounted_values(
        spot,
        strike,
        maturity_years,
        risk_free_rate,
        dividend_yield,
    )
    return EuropeanPriceBounds(
        lower=max(discounted_spot - discounted_strike, 0.0),
        upper=discounted_spot,
    )


def put_price_bounds(
    spot: float,
    strike: float,
    maturity_years: float,
    risk_free_rate: float,
    dividend_yield: float,
) -> EuropeanPriceBounds:
    """Return put bounds ``max(K e^-rT - S e^-qT, 0)`` and ``K e^-rT``."""

    discounted_spot, discounted_strike = _discounted_values(
        spot,
        strike,
        maturity_years,
        risk_free_rate,
        dividend_yield,
    )
    return EuropeanPriceBounds(
        lower=max(discounted_strike - discounted_spot, 0.0),
        upper=discounted_strike,
    )


def european_option_price_bounds(
    contract: OptionContract,
    market: ImpliedVolatilityMarket,
) -> EuropeanPriceBounds:
    """Return the appropriate reusable call or put no-arbitrage bounds."""

    arguments = (
        contract.spot,
        contract.strike,
        contract.maturity_years,
        market.risk_free_rate,
        market.dividend_yield,
    )
    if contract.option_type is OptionType.CALL:
        return call_price_bounds(*arguments)
    return put_price_bounds(*arguments)


def forward_price(
    spot: float,
    maturity_years: float,
    risk_free_rate: float,
    dividend_yield: float,
) -> float:
    """Return the finite continuous-carry forward ``S exp((r-q)T)``."""

    if not all(
        math.isfinite(value) for value in (spot, maturity_years, risk_free_rate, dividend_yield)
    ):
        raise NumericalError("forward-price inputs must be finite")
    if spot <= 0.0 or maturity_years < 0.0:
        raise NumericalError("forward price requires positive spot and non-negative maturity")
    forward = spot * finite_exp(
        (risk_free_rate - dividend_yield) * maturity_years,
        context="forward price",
    )
    if not math.isfinite(forward) or forward <= 0.0:
        raise NumericalError("forward price is outside finite positive floating-point range")
    return forward


class _PriceObjective:
    """Count every Black-Scholes evaluation made by one inversion request."""

    def __init__(self, observation: ImpliedVolatilityObservation) -> None:
        self.observation = observation
        self.function_evaluations = 0

    def price(self, volatility: float) -> float:
        self.function_evaluations += 1
        market = MarketEnvironment(
            risk_free_rate=self.observation.market.risk_free_rate,
            volatility=volatility,
            dividend_yield=self.observation.market.dividend_yield,
        )
        return black_scholes_price(self.observation.contract, market)

    def value(self, volatility: float) -> float:
        return self.price(volatility) - self.observation.observed_price


def _validate_target(
    observation: ImpliedVolatilityObservation,
    config: ImpliedVolatilityConfig,
    bounds: EuropeanPriceBounds,
) -> None:
    target = observation.observed_price
    tolerance = config.price_tolerance
    if target < bounds.lower - tolerance or target > bounds.upper + tolerance:
        raise ArbitrageBoundsError(
            "observed price "
            f"{target:.16g} is outside European no-arbitrage bounds "
            f"[{bounds.lower:.16g}, {bounds.upper:.16g}]"
        )


def _validated_bracket(
    objective: _PriceObjective,
    config: ImpliedVolatilityConfig,
) -> tuple[float, float, float, float]:
    lower = config.min_volatility
    upper = config.max_volatility
    f_lower = objective.value(lower)
    f_upper = objective.value(upper)
    if not math.isfinite(f_lower) or not math.isfinite(f_upper):
        raise NumericalError("implied-volatility bracket produced a non-finite price residual")
    if abs(f_lower) <= config.price_tolerance:
        return lower, lower, f_lower, f_lower
    if abs(f_upper) <= config.price_tolerance:
        return upper, upper, f_upper, f_upper
    if f_lower > 0.0 or f_upper < 0.0:
        raise ImpliedVolatilityBracketError(
            "observed price has no finite implied volatility in configured range "
            f"[{lower:.16g}, {upper:.16g}]"
        )
    return lower, upper, f_lower, f_upper


def _solve_brent(
    objective: _PriceObjective,
    config: ImpliedVolatilityConfig,
    bracket: tuple[float, float, float, float],
) -> tuple[float, int]:
    lower, upper, _, _ = bracket
    if lower == upper:
        return lower, 0
    relative_tolerance = 4.0 * np.finfo(float).eps
    contract = objective.observation.contract
    discounted_spot = contract.spot * finite_exp(
        -objective.observation.market.dividend_yield * contract.maturity_years,
        context="Brent Vega scale",
    )
    vega_upper_bound = (
        discounted_spot * math.sqrt(contract.maturity_years) / math.sqrt(2.0 * math.pi)
    )
    price_driven_tolerance = config.price_tolerance / max(vega_upper_bound, 1.0)
    absolute_tolerance = max(
        float(np.nextafter(0.0, 1.0)),
        min(config.volatility_tolerance, price_driven_tolerance),
    )
    try:
        root, details = brentq(
            objective.value,
            lower,
            upper,
            xtol=absolute_tolerance,
            rtol=relative_tolerance,
            maxiter=config.max_iterations,
            full_output=True,
            disp=False,
        )
    except (RuntimeError, ValueError) as exc:
        raise ImpliedVolatilityConvergenceError(f"Brent solver failed: {exc}") from exc
    if not details.converged or not math.isfinite(root):
        raise ImpliedVolatilityConvergenceError("Brent solver did not converge to a finite root")
    return float(root), int(details.iterations)


def _solve_bisection(
    objective: _PriceObjective,
    config: ImpliedVolatilityConfig,
    bracket: tuple[float, float, float, float],
) -> tuple[float, int]:
    lower, upper, f_lower, _ = bracket
    if lower == upper:
        return lower, 0
    for iteration in range(1, config.max_iterations + 1):
        midpoint = lower + 0.5 * (upper - lower)
        if midpoint in (lower, upper):
            raise ImpliedVolatilityConvergenceError(
                "bisection reached floating-point resolution before price tolerance"
            )
        f_midpoint = objective.value(midpoint)
        if not math.isfinite(f_midpoint):
            raise NumericalError("bisection produced a non-finite price residual")
        half_width = 0.5 * (upper - lower)
        if abs(f_midpoint) <= config.price_tolerance and half_width <= config.volatility_tolerance:
            return midpoint, iteration
        if f_lower * f_midpoint <= 0.0:
            upper = midpoint
        else:
            lower = midpoint
            f_lower = f_midpoint
    raise ImpliedVolatilityConvergenceError(
        "bisection did not reach price and volatility tolerances in "
        f"{config.max_iterations} iterations"
    )


def _vega(
    observation: ImpliedVolatilityObservation,
    volatility: float,
) -> float:
    market = MarketEnvironment(
        risk_free_rate=observation.market.risk_free_rate,
        volatility=volatility,
        dividend_yield=observation.market.dividend_yield,
    )
    return black_scholes_greeks(observation.contract, market).vega


def _solve_newton(
    objective: _PriceObjective,
    config: ImpliedVolatilityConfig,
) -> tuple[float, int]:
    volatility = config.initial_volatility
    residual = objective.value(volatility)
    if abs(residual) <= config.price_tolerance:
        return volatility, 0
    for iteration in range(1, config.max_iterations + 1):
        vega = _vega(objective.observation, volatility)
        if not math.isfinite(vega) or vega <= config.vega_tolerance:
            raise ImpliedVolatilityConvergenceError(
                "Newton solver stopped because analytical Vega is too small"
            )
        candidate = volatility - residual / vega
        if (
            not math.isfinite(candidate)
            or candidate < config.min_volatility
            or candidate > config.max_volatility
        ):
            raise ImpliedVolatilityConvergenceError(
                "Newton solver proposed a volatility outside the configured bounds"
            )
        if candidate == volatility:
            raise ImpliedVolatilityConvergenceError(
                "Newton solver stagnated before reaching price tolerance"
            )
        volatility = candidate
        residual = objective.value(volatility)
        if not math.isfinite(residual):
            raise NumericalError("Newton solver produced a non-finite price residual")
        if abs(residual) <= config.price_tolerance:
            return volatility, iteration
    raise ImpliedVolatilityConvergenceError(
        f"Newton solver did not reach price tolerance in {config.max_iterations} iterations"
    )


def _solve_method(
    method: ImpliedVolatilityMethod,
    objective: _PriceObjective,
    config: ImpliedVolatilityConfig,
    bracket: tuple[float, float, float, float],
) -> tuple[float, int]:
    if method is ImpliedVolatilityMethod.BRENT:
        return _solve_brent(objective, config, bracket)
    if method is ImpliedVolatilityMethod.BISECTION:
        return _solve_bisection(objective, config, bracket)
    if method is ImpliedVolatilityMethod.NEWTON:
        return _solve_newton(objective, config)
    raise ImpliedVolatilityConvergenceError("AUTO must select a concrete solver")


def implied_volatility_failure_status(exc: QuantitativeError) -> ImpliedVolatilityStatus:
    """Map a typed quantitative failure to a stable partial-result status."""

    if isinstance(exc, ArbitrageBoundsError):
        return ImpliedVolatilityStatus.ARBITRAGE_VIOLATION
    if isinstance(exc, ImpliedVolatilityBracketError):
        return ImpliedVolatilityStatus.NO_FINITE_ROOT
    if isinstance(exc, ImpliedVolatilityConvergenceError):
        return ImpliedVolatilityStatus.NON_CONVERGENCE
    return ImpliedVolatilityStatus.NUMERICAL_FAILURE


def solve_implied_volatility(
    observation: ImpliedVolatilityObservation,
    config: ImpliedVolatilityConfig | None = None,
) -> ImpliedVolatilityResult:
    """Invert one European Black-Scholes price under an explicit solver policy."""

    solver_config = config or ImpliedVolatilityConfig()
    started = perf_counter()
    bounds = european_option_price_bounds(observation.contract, observation.market)
    _validate_target(observation, solver_config, bounds)
    target = observation.observed_price

    if observation.contract.maturity_years == 0.0:
        final_price = bounds.lower
        error = abs(final_price - target)
        if error > solver_config.price_tolerance:
            raise ArbitrageBoundsError(
                "at exact expiry the observed price must equal intrinsic value "
                f"{final_price:.16g} within price tolerance"
            )
        return ImpliedVolatilityResult(
            implied_volatility=None,
            method_requested=solver_config.method,
            method_used=None,
            converged=False,
            iterations=0,
            function_evaluations=0,
            final_price=final_price,
            target_price=target,
            absolute_price_error=error,
            price_lower_bound=bounds.lower,
            price_upper_bound=final_price,
            volatility_lower_bound=solver_config.min_volatility,
            volatility_upper_bound=solver_config.max_volatility,
            runtime_ms=(perf_counter() - started) * 1000.0,
            initial_guess=solver_config.initial_volatility,
            fallback_used=False,
            final_vega=None,
            price_to_vol_sensitivity=None,
            status=ImpliedVolatilityStatus.NON_IDENTIFIABLE,
            message="implied volatility is not identifiable at exact expiry",
        )

    if abs(target - bounds.upper) <= solver_config.price_tolerance:
        raise ImpliedVolatilityBracketError(
            "the European upper-bound price is reached only as volatility tends to infinity"
        )

    objective = _PriceObjective(observation)
    bracket = _validated_bracket(objective, solver_config)
    requested = solver_config.method
    used: ImpliedVolatilityMethod | None = requested
    fallback_used = False

    if bracket[0] == bracket[1]:
        volatility, iterations = bracket[0], 0
        used = None
    elif requested is ImpliedVolatilityMethod.AUTO:
        try:
            volatility, iterations = _solve_newton(objective, solver_config)
            used = ImpliedVolatilityMethod.NEWTON
        except ImpliedVolatilityConvergenceError:
            fallback_used = True
            try:
                volatility, iterations = _solve_brent(objective, solver_config, bracket)
                used = ImpliedVolatilityMethod.BRENT
            except ImpliedVolatilityConvergenceError:
                volatility, iterations = _solve_bisection(objective, solver_config, bracket)
                used = ImpliedVolatilityMethod.BISECTION
    else:
        volatility, iterations = _solve_method(requested, objective, solver_config, bracket)

    final_price = objective.price(volatility)
    absolute_error = abs(final_price - target)
    if absolute_error > solver_config.price_tolerance:
        raise ImpliedVolatilityConvergenceError(
            "solver root failed final price reconstruction: "
            f"error {absolute_error:.16g} exceeds tolerance "
            f"{solver_config.price_tolerance:.16g}"
        )
    final_vega = _vega(observation, volatility)
    sensitivity = None
    if final_vega > solver_config.vega_tolerance:
        inverse_vega = 1.0 / final_vega
        if math.isfinite(inverse_vega):
            sensitivity = inverse_vega

    return ImpliedVolatilityResult(
        implied_volatility=volatility,
        method_requested=requested,
        method_used=used,
        converged=True,
        iterations=iterations,
        function_evaluations=objective.function_evaluations,
        final_price=final_price,
        target_price=target,
        absolute_price_error=absolute_error,
        price_lower_bound=bounds.lower,
        price_upper_bound=bounds.upper,
        volatility_lower_bound=solver_config.min_volatility,
        volatility_upper_bound=solver_config.max_volatility,
        runtime_ms=(perf_counter() - started) * 1000.0,
        initial_guess=solver_config.initial_volatility,
        fallback_used=fallback_used,
        final_vega=final_vega,
        price_to_vol_sensitivity=sensitivity,
        status=ImpliedVolatilityStatus.CONVERGED,
        message=(
            f"converged with {used.value}"
            if used is not None
            else "target price matched a configured volatility boundary"
        ),
    )
