"""Correlation validation, tiny PSD repair, and deterministic factorization."""

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from backend.app.core.exceptions import MultiAssetPricingError

SYMMETRY_TOLERANCE = 1e-12
DIAGONAL_TOLERANCE = 1e-12
BOUND_TOLERANCE = 1e-12
PSD_REPAIR_TOLERANCE = 1e-10
RANK_TOLERANCE = 1e-12


class CorrelationFactorizationMethod(StrEnum):
    CHOLESKY = "CHOLESKY"
    EIGEN = "EIGEN"


@dataclass(frozen=True)
class PreparedCorrelation:
    matrix: NDArray[np.float64]
    original_min_eigenvalue: float
    effective_min_eigenvalue: float
    repaired: bool
    adjustment_magnitude: float


@dataclass(frozen=True)
class CorrelationFactorization:
    factor: NDArray[np.float64]
    method: CorrelationFactorizationMethod
    min_eigenvalue: float
    max_eigenvalue: float
    effective_rank: int
    condition_number: float | None


def prepare_correlation_matrix(values: ArrayLike) -> PreparedCorrelation:
    """Validate a correlation matrix and repair only tiny negative eigenvalues."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise MultiAssetPricingError("correlation matrix must be non-empty and square")
    if not np.all(np.isfinite(matrix)):
        raise MultiAssetPricingError("correlation matrix entries must be finite")
    asymmetry = float(np.max(np.abs(matrix - matrix.T)))
    if asymmetry > SYMMETRY_TOLERANCE:
        raise MultiAssetPricingError("correlation matrix must be symmetric")
    diagonal_error = float(np.max(np.abs(np.diag(matrix) - 1.0)))
    if diagonal_error > DIAGONAL_TOLERANCE:
        raise MultiAssetPricingError("correlation matrix diagonal must equal one")
    if np.any(matrix < -1.0 - BOUND_TOLERANCE) or np.any(matrix > 1.0 + BOUND_TOLERANCE):
        raise MultiAssetPricingError("correlation entries must lie in [-1, 1]")

    candidate = np.clip(0.5 * (matrix + matrix.T), -1.0, 1.0)
    np.fill_diagonal(candidate, 1.0)
    eigenvalues, eigenvectors = np.linalg.eigh(candidate)
    original_min = float(eigenvalues[0])
    if original_min < -PSD_REPAIR_TOLERANCE:
        raise MultiAssetPricingError("correlation matrix is materially non-positive-semidefinite")

    repaired = original_min < 0.0
    if repaired:
        clipped = np.maximum(eigenvalues, 0.0)
        reconstructed = (eigenvectors * clipped) @ eigenvectors.T
        scales = np.sqrt(np.diag(reconstructed))
        if np.any(scales <= 0.0) or not np.all(np.isfinite(scales)):
            raise MultiAssetPricingError("correlation repair produced an invalid diagonal")
        effective = reconstructed / np.outer(scales, scales)
        effective = 0.5 * (effective + effective.T)
        np.fill_diagonal(effective, 1.0)
    else:
        effective = candidate
    adjustment = float(np.linalg.norm(effective - matrix, ord="fro"))
    effective_min = float(np.linalg.eigvalsh(effective)[0])
    if effective_min < -PSD_REPAIR_TOLERANCE:
        raise MultiAssetPricingError("effective correlation matrix remains non-PSD")
    return PreparedCorrelation(
        matrix=np.asarray(effective, dtype=np.float64),
        original_min_eigenvalue=original_min,
        effective_min_eigenvalue=effective_min,
        repaired=repaired,
        adjustment_magnitude=adjustment,
    )


def factorize_correlation(matrix: ArrayLike) -> CorrelationFactorization:
    """Use Cholesky for PD input and an eigen factor for singular PSD input."""

    prepared = prepare_correlation_matrix(matrix)
    effective = prepared.matrix
    eigenvalues = np.linalg.eigvalsh(effective)
    minimum = float(eigenvalues[0])
    maximum = float(eigenvalues[-1])
    rank_threshold = RANK_TOLERANCE * max(1.0, maximum)
    rank = int(np.sum(eigenvalues > rank_threshold))
    try:
        factor = np.linalg.cholesky(effective)
        method = CorrelationFactorizationMethod.CHOLESKY
    except np.linalg.LinAlgError:
        values, vectors = np.linalg.eigh(effective)
        if float(values[0]) < -PSD_REPAIR_TOLERANCE:
            raise MultiAssetPricingError(
                "correlation eigen factor received non-PSD input"
            ) from None
        factor = vectors @ np.diag(np.sqrt(np.maximum(values, 0.0)))
        method = CorrelationFactorizationMethod.EIGEN
    positive = eigenvalues[eigenvalues > rank_threshold]
    condition = float(maximum / positive[0]) if positive.size and rank == len(eigenvalues) else None
    if not np.all(np.isfinite(factor)):
        raise MultiAssetPricingError("correlation factorization produced non-finite values")
    return CorrelationFactorization(
        factor=np.asarray(factor, dtype=np.float64),
        method=method,
        min_eigenvalue=minimum,
        max_eigenvalue=maximum,
        effective_rank=rank,
        condition_number=condition if condition is None or math.isfinite(condition) else None,
    )


def build_equicorrelation_matrix(dimension: int, rho: float) -> tuple[tuple[float, ...], ...]:
    """Return an N-dimensional equicorrelation matrix after exact PSD-bound validation."""

    if dimension <= 0:
        raise MultiAssetPricingError("equicorrelation dimension must be positive")
    if not math.isfinite(rho):
        raise MultiAssetPricingError("equicorrelation rho must be finite")
    lower = -1.0 / (dimension - 1) if dimension > 1 else -1.0
    if rho < lower - BOUND_TOLERANCE or rho > 1.0 + BOUND_TOLERANCE:
        raise MultiAssetPricingError(
            f"equicorrelation rho must lie in [{lower}, 1] for dimension {dimension}"
        )
    effective_rho = min(max(rho, lower), 1.0)
    matrix = np.full((dimension, dimension), effective_rho, dtype=np.float64)
    np.fill_diagonal(matrix, 1.0)
    return tuple(tuple(float(value) for value in row) for row in matrix)
