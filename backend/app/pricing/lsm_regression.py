"""Normalized polynomial bases and rank-aware continuation regressions."""

import math

import numpy as np
from numpy.typing import NDArray

from backend.app.core.exceptions import LongstaffSchwartzError, NumericalError
from backend.app.core.numerics import require_finite
from backend.app.models.longstaff_schwartz import (
    BasisFamily,
    LongstaffSchwartzConfig,
    LSMRegressionDiagnostic,
    RegressionFallback,
)


def basis_matrix(
    stock: NDArray[np.float64], strike: float, family: BasisFamily, degree: int
) -> NDArray[np.float64]:
    """Return [L0,...,Ld] or [1,x,...,x**d], where x=S/K; Laguerre is unweighted."""

    if stock.ndim != 1 or not math.isfinite(strike) or strike <= 0.0:
        raise LongstaffSchwartzError("basis requires a stock vector and positive finite strike")
    if not 0 <= degree <= 5:
        raise LongstaffSchwartzError("basis degree must be between zero and five")
    require_finite(stock, context="LSM basis stock")
    if np.any(stock <= 0.0):
        raise LongstaffSchwartzError("basis stock must be positive")
    with np.errstate(over="ignore", invalid="ignore"):
        normalized = stock / strike
        if family is BasisFamily.POWER:
            raw_matrix = np.polynomial.polynomial.polyvander(normalized, degree)
        elif family is BasisFamily.LAGUERRE:
            raw_matrix = np.polynomial.laguerre.lagvander(normalized, degree)
        else:
            raise LongstaffSchwartzError("unsupported LSM basis family")
    matrix = np.asarray(raw_matrix, dtype=np.float64)
    require_finite(matrix, context="LSM basis matrix")
    return np.asarray(matrix, dtype=np.float64)


def fit_continuation(
    stock: NDArray[np.float64],
    targets: NDArray[np.float64],
    strike: float,
    config: LongstaffSchwartzConfig,
    *,
    time_index: int,
    time: float,
) -> LSMRegressionDiagnostic:
    """Fit by SVD least squares; any fallback keeps every candidate's future cash flow."""

    if stock.ndim != 1 or targets.shape != stock.shape:
        raise LongstaffSchwartzError("regression states and targets must be matching vectors")
    require_finite(stock, context="LSM regression states")
    if np.any(stock <= 0.0) or not math.isfinite(strike) or strike <= 0.0:
        raise LongstaffSchwartzError("regression states and strike must be positive and finite")
    require_finite(targets, context="LSM regression targets")
    count = stock.size
    terms = config.basis_degree + 1
    reason = None
    rank = None
    condition = None
    residual = None
    coefficients = None
    condition_warning = False
    if count == 0:
        reason = RegressionFallback.NO_ITM_PATHS
    elif count < max(config.min_regression_paths, terms):
        reason = RegressionFallback.INSUFFICIENT_PATHS
    else:
        matrix = basis_matrix(stock, strike, config.basis_family, config.basis_degree)
        try:
            raw_fitted, _, raw_rank, singular = np.linalg.lstsq(matrix, targets, rcond=None)
        except np.linalg.LinAlgError:
            reason = RegressionFallback.SOLVER_FAILURE
        else:
            rank = int(raw_rank)
            if singular.size and singular[-1] > 0.0:
                ratio = float(singular[0] / singular[-1])
                condition = ratio if math.isfinite(ratio) else None
            condition_warning = condition is None or condition > config.condition_warning_threshold
            if rank < terms:
                reason = RegressionFallback.RANK_DEFICIENT
            else:
                fitted = np.asarray(raw_fitted, dtype=np.float64)
                require_finite(fitted, context="LSM regression coefficients")
                errors = targets - np.einsum("ij,j->i", matrix, fitted)
                residual = float(np.dot(errors, errors))
                if not math.isfinite(residual):
                    raise NumericalError("LSM regression residual is non-finite")
                coefficients = tuple(float(value) for value in fitted)
    return LSMRegressionDiagnostic(
        time_index=time_index,
        time=time,
        basis_family=config.basis_family,
        basis_degree=config.basis_degree,
        itm_path_count=count,
        regression_path_count=count,
        basis_term_count=terms,
        matrix_rank=rank,
        condition_number=condition,
        condition_warning=condition_warning,
        residual_sum_squares=residual,
        fallback_used=reason is not None,
        fallback_reason=reason,
        coefficients=coefficients,
    )


def predict_continuation(
    stock: NDArray[np.float64], strike: float, date: LSMRegressionDiagnostic
) -> NDArray[np.float64]:
    """Evaluate a fitted current-state policy; nonnegative cash flows imply a zero floor."""

    if date.fallback_used or date.coefficients is None:
        raise LongstaffSchwartzError("fallback date has no continuation regression")
    matrix = basis_matrix(stock, strike, date.basis_family, date.basis_degree)
    if len(date.coefficients) != matrix.shape[1]:
        raise LongstaffSchwartzError("policy coefficients do not match basis degree")
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.einsum("ij,j->i", matrix, np.asarray(date.coefficients, dtype=np.float64))
    require_finite(values, context="LSM continuation predictions")
    return np.asarray(np.maximum(values, 0.0), dtype=np.float64)
