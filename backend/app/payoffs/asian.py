"""Vectorized fixed-strike Asian averages and payoffs over supplied observations."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from backend.app.core.exceptions import NumericalError
from backend.app.core.numerics import finite_array_exp, require_finite
from backend.app.models.asian import AsianOptionContract, AveragingType
from backend.app.models.option import OptionType


def asian_average(observations: ArrayLike, averaging_type: AveragingType) -> NDArray[np.float64]:
    """Return one arithmetic or stable log-geometric average per path."""

    prices = np.asarray(observations, dtype=np.float64)
    if prices.ndim != 2 or prices.shape[0] == 0 or prices.shape[1] == 0:
        raise ValueError("Asian observations must be a non-empty two-dimensional matrix")
    require_finite(prices, context="Asian observations")
    if np.any(prices <= 0.0):
        raise NumericalError("Asian observations must be strictly positive")
    if averaging_type is AveragingType.ARITHMETIC:
        with np.errstate(over="ignore", invalid="ignore"):
            averages = np.mean(prices, axis=1)
        require_finite(averages, context="arithmetic Asian averages")
        return np.asarray(averages, dtype=np.float64)
    if averaging_type is AveragingType.GEOMETRIC:
        if prices.shape[1] == 1:
            return np.asarray(prices[:, 0].copy(), dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_logs = np.mean(np.log(prices), axis=1)
        require_finite(mean_logs, context="geometric Asian mean logs")
        return finite_array_exp(mean_logs, context="geometric Asian averages")
    raise ValueError("unsupported Asian averaging type")


def asian_payoff_from_average(
    averages: ArrayLike, contract: AsianOptionContract
) -> NDArray[np.float64]:
    """Return an undiscounted fixed-strike payoff from caller-supplied averages."""

    values = np.asarray(averages, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Asian averages must be a non-empty one-dimensional array")
    require_finite(values, context="Asian averages")
    if np.any(values <= 0.0):
        raise NumericalError("Asian averages must be strictly positive")
    if contract.option_type is OptionType.CALL:
        payoff = np.maximum(values - contract.strike, 0.0)
    else:
        payoff = np.maximum(contract.strike - values, 0.0)
    require_finite(payoff, context="Asian payoffs")
    return np.asarray(payoff, dtype=np.float64)


def asian_payoff(observations: ArrayLike, contract: AsianOptionContract) -> NDArray[np.float64]:
    """Return the contract payoff for a fixed path-observation matrix."""

    return asian_payoff_from_average(asian_average(observations, contract.averaging_type), contract)
