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


def discounted_spot_and_strike(
    contract: OptionContract,
    market: MarketEnvironment,
) -> tuple[float, float]:
    """Return finite dividend-discounted spot and risk-free-discounted strike."""

    maturity = contract.maturity_years
    discounted_spot = contract.spot * finite_exp(
        -market.dividend_yield * maturity,
        context="dividend discount factor",
    )
    discounted_strike = contract.strike * finite_exp(
        -market.risk_free_rate * maturity,
        context="risk-free discount factor",
    )
    if not math.isfinite(discounted_spot) or not math.isfinite(discounted_strike):
        raise NumericalError("Black-Scholes present values overflowed floating-point range")
    return discounted_spot, discounted_strike


def black_scholes_d1_d2(
    contract: OptionContract,
    market: MarketEnvironment,
) -> tuple[float, float] | None:
    """Return finite d1/d2, or ``None`` at a deterministic numerical boundary."""

    maturity = contract.maturity_years
    volatility_time = market.volatility * math.sqrt(maturity)
    if maturity == 0.0 or market.volatility == 0.0 or volatility_time == 0.0:
        return None
    log_moneyness = math.log(contract.spot) - math.log(contract.strike)
    numerator = (
        log_moneyness
        + (market.risk_free_rate - market.dividend_yield + 0.5 * market.volatility**2) * maturity
    )
    d1 = numerator / volatility_time
    d2 = d1 - volatility_time
    if not math.isfinite(d1) or not math.isfinite(d2):
        return None
    return d1, d2


def _deterministic_price(contract: OptionContract, market: MarketEnvironment) -> float:
    """Return the sigma=0 discounted deterministic terminal payoff."""

    discounted_spot, discounted_strike = discounted_spot_and_strike(contract, market)
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

    terms = black_scholes_d1_d2(contract, market)
    if terms is None:
        # Infinite d-values are the exact deterministic limit in floating point.
        return _deterministic_price(contract, market)
    d1, d2 = terms

    spot_present, strike_present = discounted_spot_and_strike(contract, market)

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
