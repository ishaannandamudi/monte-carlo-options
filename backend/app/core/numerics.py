"""Central finite-number checks for quantitative calculations."""

import math

import numpy as np
from numpy.typing import NDArray

from backend.app.core.exceptions import NumericalError


def finite_exp(exponent: float, *, context: str) -> float:
    """Evaluate a scalar exponential or raise a clear domain error."""

    try:
        result = math.exp(exponent)
    except OverflowError as exc:
        raise NumericalError(f"{context} overflowed floating-point range") from exc
    if not math.isfinite(result) or result == 0.0:
        raise NumericalError(f"{context} is outside positive finite floating-point range")
    return result


def finite_array_exp(exponents: NDArray[np.float64], *, context: str) -> NDArray[np.float64]:
    """Evaluate vectorized exponentials and reject overflow or underflow."""

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = np.exp(exponents)
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise NumericalError(f"{context} produced non-positive or non-finite values")
    return np.asarray(result, dtype=np.float64)


def require_finite(values: NDArray[np.float64], *, context: str) -> None:
    """Reject NaN and infinity before they can enter a typed result."""

    if not np.all(np.isfinite(values)):
        raise NumericalError(f"{context} produced non-finite values")
