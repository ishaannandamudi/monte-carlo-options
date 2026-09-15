"""Black-Scholes-Merton pricing with a continuous dividend yield."""

import math

from scipy.stats import norm

from backend.app.core.exceptions import NumericalError
from backend.app.core.numerics import finite_exp
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.models.results import BlackScholesResult
from backend.app.payoffs.vanilla import intrinsic_value
from backend.app.pricing.base import Pricer


def _deterministic_price(contract: OptionContract, market: MarketEnvironment) -> float:
    """Return the sigma=0 discounted deterministic terminal payoff."""

    maturity = contract.maturity_years
    spot_discount = finite_exp(
        -market.dividend_yield * maturity, context="dividend discount factor"
    )
    strike_discount = finite_exp(
        -market.risk_free_rate * maturity,
        context="risk-free discount factor",
    )
    discounted_spot = contract.spot * spot_discount
    discounted_strike = contract.strike * strike_discount
    if not math.isfinite(discounted_spot) or not math.isfinite(discounted_strike):
        raise NumericalError("deterministic option value overflowed floating-point range")
    if contract.option_type is OptionType.CALL:
        return max(discounted_spot - discounted_strike, 0.0)
    return max(discounted_strike - discounted_spot, 0.0)


def black_scholes_price(contract: OptionContract, market: MarketEnvironment) -> float:
    """Return the European vanilla Black-Scholes-Merton value.

    Exact zero maturity returns intrinsic value. Exact zero volatility, including
    a positive volatility-time product that underflows to zero, uses the
    deterministic discounted-forward limit and never divides by zero.
    """

    maturity = contract.maturity_years
    if maturity == 0.0:
        return intrinsic_value(contract.option_type, contract.spot, contract.strike)

    volatility_time = market.volatility * math.sqrt(maturity)
    if market.volatility == 0.0 or volatility_time == 0.0:
        return _deterministic_price(contract, market)

    log_moneyness = math.log(contract.spot) - math.log(contract.strike)
    numerator = (
        log_moneyness
        + (market.risk_free_rate - market.dividend_yield + 0.5 * market.volatility**2) * maturity
    )
    d1 = numerator / volatility_time
    d2 = d1 - volatility_time
    if not math.isfinite(d1) or not math.isfinite(d2):
        # Infinite d-values are the exact deterministic limit in floating point.
        return _deterministic_price(contract, market)

    spot_present = contract.spot * finite_exp(
        -market.dividend_yield * maturity,
        context="dividend discount factor",
    )
    strike_present = contract.strike * finite_exp(
        -market.risk_free_rate * maturity,
        context="risk-free discount factor",
    )
    if not math.isfinite(spot_present) or not math.isfinite(strike_present):
        raise NumericalError("Black-Scholes present values overflowed floating-point range")

    if contract.option_type is OptionType.CALL:
        value = spot_present * float(norm.cdf(d1)) - strike_present * float(norm.cdf(d2))
    else:
        value = strike_present * float(norm.cdf(-d2)) - spot_present * float(norm.cdf(-d1))
    if not math.isfinite(value):
        raise NumericalError("Black-Scholes formula produced a non-finite value")
    return max(value, 0.0)


class BlackScholesPricer(Pricer[None, BlackScholesResult]):
    """Typed analytical pricing engine."""

    def price(
        self,
        contract: OptionContract,
        market: MarketEnvironment,
        config: None = None,
    ) -> BlackScholesResult:
        return BlackScholesResult(price=black_scholes_price(contract, market))
