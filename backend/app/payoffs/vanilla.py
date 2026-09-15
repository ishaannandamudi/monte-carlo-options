"""Vectorized European vanilla payoff functions with no discounting."""

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from backend.app.core.exceptions import NumericalError
from backend.app.core.numerics import require_finite
from backend.app.models.option import OptionContract, OptionType


def call_payoff(terminal_prices: ArrayLike, strike: float) -> NDArray[np.float64]:
    """Return max(S_T - K, 0) elementwise."""

    prices = np.asarray(terminal_prices, dtype=np.float64)
    require_finite(prices, context="terminal prices")
    result = np.maximum(prices - strike, 0.0)
    require_finite(result, context="call payoffs")
    return np.asarray(result, dtype=np.float64)


def put_payoff(terminal_prices: ArrayLike, strike: float) -> NDArray[np.float64]:
    """Return max(K - S_T, 0) elementwise."""

    prices = np.asarray(terminal_prices, dtype=np.float64)
    require_finite(prices, context="terminal prices")
    result = np.maximum(strike - prices, 0.0)
    require_finite(result, context="put payoffs")
    return np.asarray(result, dtype=np.float64)


def vanilla_payoff(
    terminal_prices: ArrayLike,
    contract: OptionContract,
) -> NDArray[np.float64]:
    """Dispatch to the payoff selected by the contract."""

    if contract.option_type is OptionType.CALL:
        return call_payoff(terminal_prices, contract.strike)
    return put_payoff(terminal_prices, contract.strike)


def intrinsic_value(option_type: OptionType, spot: float, strike: float) -> float:
    """Return the undiscounted vanilla payoff at immediate maturity."""

    value = max(spot - strike, 0.0) if option_type is OptionType.CALL else max(strike - spot, 0.0)
    if not math.isfinite(value):
        raise NumericalError("intrinsic value is non-finite")
    return value
