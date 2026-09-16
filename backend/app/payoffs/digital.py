"""Strict-ITM cash-or-nothing and asset-or-nothing terminal payoffs."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from backend.app.core.numerics import require_finite
from backend.app.models.digital import DigitalOptionContract, DigitalType
from backend.app.models.option import OptionType


def digital_itm(terminal_prices: ArrayLike, contract: DigitalOptionContract) -> NDArray[np.bool_]:
    prices = np.asarray(terminal_prices, dtype=np.float64)
    if prices.ndim != 1 or prices.size == 0:
        raise ValueError("terminal prices must be a non-empty one-dimensional array")
    require_finite(prices, context="digital terminal prices")
    if contract.option_type is OptionType.CALL:
        return np.asarray(prices > contract.strike, dtype=np.bool_)
    return np.asarray(prices < contract.strike, dtype=np.bool_)


def digital_payoff(
    terminal_prices: ArrayLike, contract: DigitalOptionContract
) -> NDArray[np.float64]:
    """Return undiscounted digital payoffs; equality at strike always pays zero."""

    prices = np.asarray(terminal_prices, dtype=np.float64)
    itm = digital_itm(prices, contract)
    if contract.digital_type is DigitalType.CASH_OR_NOTHING:
        assert contract.cash_payout is not None
        values = contract.cash_payout * itm
    else:
        values = prices * itm
    result = np.asarray(values, dtype=np.float64)
    require_finite(result, context="digital payoffs")
    return result
