"""Vectorized weighted arithmetic and geometric basket payoffs."""

import numpy as np
from numpy.typing import NDArray

from backend.app.core.exceptions import MultiAssetPricingError
from backend.app.core.numerics import finite_array_exp, require_finite
from backend.app.models.basket import BasketOptionContract, BasketType
from backend.app.models.option import OptionType


def basket_values(
    terminal_prices: NDArray[np.float64],
    weights: tuple[float, ...],
    basket_type: BasketType,
) -> NDArray[np.float64]:
    """Return one weighted basket value per terminal-price row."""

    terminal = np.asarray(terminal_prices, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    if terminal.ndim != 2 or terminal.shape[0] == 0:
        raise ValueError("terminal prices must be a non-empty (paths, assets) matrix")
    if terminal.shape[1] != weight_array.size:
        raise ValueError("weight count must match the terminal asset dimension")
    require_finite(terminal, context="basket terminal prices")
    if np.any(terminal <= 0.0):
        raise MultiAssetPricingError("basket terminal prices must be strictly positive")
    with np.errstate(all="ignore"):
        if basket_type is BasketType.ARITHMETIC:
            values = terminal @ weight_array
        else:
            log_values = np.log(terminal) @ weight_array
            values = finite_array_exp(log_values, context="geometric basket values")
    values = np.asarray(values, dtype=np.float64)
    require_finite(values, context="basket values")
    if np.any(values <= 0.0):
        raise MultiAssetPricingError("basket values must be positive")
    return values


def basket_payoff_from_values(
    values: NDArray[np.float64], contract: BasketOptionContract
) -> NDArray[np.float64]:
    """Apply a European call or put payoff to basket values without discounting."""

    basket = np.asarray(values, dtype=np.float64)
    if basket.ndim != 1 or basket.size == 0:
        raise ValueError("basket values must be a non-empty one-dimensional array")
    require_finite(basket, context="basket payoff inputs")
    if contract.option_type is OptionType.CALL:
        payoff = np.maximum(basket - contract.strike, 0.0)
    else:
        payoff = np.maximum(contract.strike - basket, 0.0)
    return np.asarray(payoff, dtype=np.float64)


def basket_payoff(
    terminal_prices: NDArray[np.float64], contract: BasketOptionContract
) -> NDArray[np.float64]:
    """Compute a weighted basket and apply the contract payoff."""

    return basket_payoff_from_values(
        basket_values(terminal_prices, contract.weights, contract.basket_type), contract
    )
