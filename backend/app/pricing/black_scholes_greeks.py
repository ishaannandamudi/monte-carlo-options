"""Analytical Black-Scholes-Merton Greeks in canonical raw model units."""

import math

from scipy.stats import norm

from backend.app.core.exceptions import NumericalError
from backend.app.models.greeks import Greek
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.models.results import AnalyticalGreeksResult
from backend.app.pricing.base import require_european_exercise
from backend.app.pricing.black_scholes import (
    black_scholes_d1_d2,
    black_scholes_price,
    discounted_spot_and_strike,
)


def _require_finite_greeks(values: tuple[float, ...]) -> None:
    if not all(math.isfinite(value) for value in values):
        raise NumericalError("Black-Scholes Greeks produced a non-finite value")


def _expiry_greeks(contract: OptionContract, market: MarketEnvironment) -> AnalyticalGreeksResult:
    """Apply the explicit symmetric-kink convention at exact expiry."""

    if contract.spot > contract.strike:
        call_delta = 1.0
    elif contract.spot < contract.strike:
        call_delta = 0.0
    else:
        call_delta = 0.5
    delta = call_delta if contract.option_type is OptionType.CALL else call_delta - 1.0
    return AnalyticalGreeksResult(
        option_price=black_scholes_price(contract, market),
        delta=delta,
        gamma=0.0,
        vega=0.0,
        theta=0.0,
        rho=0.0,
        dividend_rho=0.0,
        d1=None,
        d2=None,
    )


def _deterministic_greeks(
    contract: OptionContract,
    market: MarketEnvironment,
) -> AnalyticalGreeksResult:
    """Return the documented zero-volatility deterministic-limit convention."""

    maturity = contract.maturity_years
    discounted_spot, discounted_strike = discounted_spot_and_strike(contract, market)
    gap = discounted_spot - discounted_strike
    if gap > 0.0:
        call_weight = 1.0
    elif gap < 0.0:
        call_weight = 0.0
    else:
        call_weight = 0.5

    spot_discount = discounted_spot / contract.spot
    if contract.option_type is OptionType.CALL:
        delta = call_weight * spot_discount
        rho = call_weight * maturity * discounted_strike
        dividend_rho = -call_weight * maturity * discounted_spot
        theta = call_weight * (
            market.dividend_yield * discounted_spot - market.risk_free_rate * discounted_strike
        )
    else:
        put_weight = 1.0 - call_weight
        delta = -put_weight * spot_discount
        rho = -put_weight * maturity * discounted_strike
        dividend_rho = put_weight * maturity * discounted_spot
        theta = put_weight * (
            market.risk_free_rate * discounted_strike - market.dividend_yield * discounted_spot
        )

    vega = discounted_spot * math.sqrt(maturity) * float(norm.pdf(0.0)) if gap == 0.0 else 0.0
    values = (delta, vega, theta, rho, dividend_rho)
    _require_finite_greeks(values)
    return AnalyticalGreeksResult(
        option_price=black_scholes_price(contract, market),
        delta=delta,
        gamma=0.0,
        vega=vega,
        theta=theta,
        rho=rho,
        dividend_rho=dividend_rho,
        d1=None,
        d2=None,
    )


def black_scholes_greeks(
    contract: OptionContract,
    market: MarketEnvironment,
) -> AnalyticalGreeksResult:
    """Return price and six analytical Greeks with calendar-time Theta.

    Vega, Rho, dividend Rho, and Theta use absolute volatility, absolute rate,
    absolute dividend-yield, and year units respectively.
    """

    require_european_exercise(contract, engine="Black-Scholes Greeks")
    maturity = contract.maturity_years
    if maturity == 0.0:
        return _expiry_greeks(contract, market)
    terms = black_scholes_d1_d2(contract, market)
    if terms is None:
        return _deterministic_greeks(contract, market)

    d1, d2 = terms
    sqrt_maturity = math.sqrt(maturity)
    discounted_spot, discounted_strike = discounted_spot_and_strike(contract, market)
    spot_discount = discounted_spot / contract.spot
    density = float(norm.pdf(d1))
    cdf_d1 = float(norm.cdf(d1))
    cdf_d2 = float(norm.cdf(d2))

    gamma = spot_discount * density / (contract.spot * market.volatility * sqrt_maturity)
    vega = discounted_spot * density * sqrt_maturity
    diffusion_theta = -discounted_spot * density * market.volatility / (2.0 * sqrt_maturity)

    if contract.option_type is OptionType.CALL:
        delta = spot_discount * cdf_d1
        theta = (
            diffusion_theta
            + market.dividend_yield * discounted_spot * cdf_d1
            - market.risk_free_rate * discounted_strike * cdf_d2
        )
        rho = maturity * discounted_strike * cdf_d2
        dividend_rho = -maturity * discounted_spot * cdf_d1
    else:
        cdf_minus_d1 = float(norm.cdf(-d1))
        cdf_minus_d2 = float(norm.cdf(-d2))
        delta = spot_discount * (cdf_d1 - 1.0)
        theta = (
            diffusion_theta
            - market.dividend_yield * discounted_spot * cdf_minus_d1
            + market.risk_free_rate * discounted_strike * cdf_minus_d2
        )
        rho = -maturity * discounted_strike * cdf_minus_d2
        dividend_rho = maturity * discounted_spot * cdf_minus_d1

    values = (delta, gamma, vega, theta, rho, dividend_rho)
    _require_finite_greeks(values)
    return AnalyticalGreeksResult(
        option_price=black_scholes_price(contract, market),
        delta=delta,
        gamma=max(gamma, 0.0),
        vega=max(vega, 0.0),
        theta=theta,
        rho=rho,
        dividend_rho=dividend_rho,
        d1=d1,
        d2=d2,
    )


def analytical_greek_value(result: AnalyticalGreeksResult, greek: Greek) -> float:
    """Return one Greek value without duplicating field-name mapping in analytics."""

    return float(getattr(result, greek.value.lower()))
