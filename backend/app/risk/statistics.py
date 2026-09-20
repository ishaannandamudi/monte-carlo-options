"""Loss-distribution estimators and covariance diagnostics for portfolio risk."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import jarque_bera, kurtosis, skew

from backend.app.core.exceptions import PortfolioRiskError
from backend.app.models.risk import (
    CovarianceDiagnostics,
    CovarianceSource,
    LossDistributionSummary,
)

FloatArray = NDArray[np.float64]
QuantileMethod = Literal["linear", "lower", "higher", "midpoint", "nearest"]


@dataclass(frozen=True)
class EmpiricalTail:
    value_at_risk: float
    expected_shortfall: float
    ordered_indices: NDArray[np.int64]
    weights: FloatArray


@dataclass(frozen=True)
class PreparedCovariance:
    factor_ids: tuple[str, ...]
    covariance: FloatArray
    means: FloatArray
    diagnostics: CovarianceDiagnostics


def pnl_to_losses(pnl: FloatArray) -> FloatArray:
    """Apply the single Phase 17 sign convention: loss equals negative P&L."""

    values = np.asarray(pnl, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise PortfolioRiskError("P&L samples must be a non-empty finite vector")
    return np.asarray(-values, dtype=np.float64)


def empirical_tail(
    losses: FloatArray,
    confidence_level: float,
    *,
    quantile_method: QuantileMethod = "linear",
) -> EmpiricalTail:
    """Return explicit-quantile VaR and an exact-probability-mass empirical ES.

    ES averages the worst ``n * (1-alpha)`` observations. If this mass is not an
    integer, the next observation receives the fractional remaining weight. When
    the tail mass is below one observation, ES is the single worst loss.
    """

    values = np.asarray(losses, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise PortfolioRiskError("loss samples must be a non-empty finite vector")
    if not 0.0 < confidence_level < 1.0:
        raise PortfolioRiskError("confidence_level must lie strictly between zero and one")
    value_at_risk = float(np.quantile(values, confidence_level, method=quantile_method))
    order = np.argsort(-values, kind="stable").astype(np.int64)
    mass = values.size * (1.0 - confidence_level)
    full = int(math.floor(mass))
    fractional = mass - full
    if full == 0:
        indices = order[:1]
        weights = np.ones(1, dtype=np.float64)
    else:
        count = full + (1 if fractional > 1e-14 else 0)
        indices = order[:count]
        weights = np.ones(count, dtype=np.float64)
        if count > full:
            weights[-1] = fractional
    expected_shortfall = float(np.dot(values[indices], weights) / np.sum(weights))
    if value_at_risk == 0.0:
        value_at_risk = 0.0
    if expected_shortfall == 0.0:
        expected_shortfall = 0.0
    return EmpiricalTail(value_at_risk, expected_shortfall, indices, weights)


def distribution_summary(
    pnl: FloatArray,
    *,
    confidence_level: float,
    tail_count: int,
    quantile_method: QuantileMethod,
) -> LossDistributionSummary:
    values = np.asarray(pnl, dtype=np.float64)
    losses = pnl_to_losses(values)
    count = values.size
    selected = sorted({0.01, 0.05, 0.5, 0.95, 0.975, 0.99, 0.995, confidence_level})
    quantiles = {
        f"{level:.6g}": float(np.quantile(losses, level, method=quantile_method))
        for level in selected
    }
    if count >= 3 and float(np.std(losses)) > 0.0:
        skewness = float(skew(losses, bias=False))
        excess = float(kurtosis(losses, fisher=True, bias=False)) if count >= 4 else 0.0
        jb = jarque_bera(losses) if count >= 8 else None
    else:
        skewness = 0.0
        excess = 0.0
        jb = None
    return LossDistributionSummary(
        pnl_mean=float(np.mean(values)),
        pnl_standard_deviation=float(np.std(values, ddof=1)) if count > 1 else 0.0,
        minimum_pnl=float(np.min(values)),
        maximum_pnl=float(np.max(values)),
        loss_mean=float(np.mean(losses)),
        loss_standard_deviation=float(np.std(losses, ddof=1)) if count > 1 else 0.0,
        median_loss=float(np.median(losses)),
        worst_loss=float(np.max(losses)),
        best_pnl=float(np.max(values)),
        skewness=skewness,
        excess_kurtosis=excess,
        loss_quantiles=quantiles,
        tail_count=tail_count,
        jarque_bera_statistic=None if jb is None else float(jb.statistic),
        jarque_bera_p_value=None if jb is None else float(jb.pvalue),
    )


def validate_covariance(
    factor_ids: list[str],
    covariance: FloatArray,
    means: FloatArray,
    *,
    source: CovarianceSource,
    observation_count: int | None = None,
) -> PreparedCovariance:
    matrix = np.asarray(covariance, dtype=np.float64)
    mean_values = np.asarray(means, dtype=np.float64)
    dimension = len(factor_ids)
    if matrix.shape != (dimension, dimension) or mean_values.shape != (dimension,):
        raise PortfolioRiskError("covariance dimensions must match factor identifiers")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(mean_values)):
        raise PortfolioRiskError("covariance and mean changes must be finite")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12):
        raise PortfolioRiskError("covariance matrix must be symmetric")
    eigenvalues = np.linalg.eigvalsh(matrix)
    scale = max(1.0, float(np.max(np.abs(matrix))))
    if float(eigenvalues[0]) < -1e-10 * scale:
        raise PortfolioRiskError("covariance matrix is materially non-positive-semidefinite")
    effective = 0.5 * (matrix + matrix.T)
    if eigenvalues[0] < 0.0:
        values, vectors = np.linalg.eigh(effective)
        effective = (vectors * np.maximum(values, 0.0)) @ vectors.T
        effective = 0.5 * (effective + effective.T)
        eigenvalues = np.linalg.eigvalsh(effective)
    threshold = 1e-12 * max(1.0, float(eigenvalues[-1]))
    positive = eigenvalues[eigenvalues > threshold]
    rank = int(positive.size)
    condition = (
        float(eigenvalues[-1] / positive[0]) if rank == dimension and positive.size else None
    )
    return PreparedCovariance(
        factor_ids=tuple(factor_ids),
        covariance=np.asarray(effective, dtype=np.float64),
        means=mean_values,
        diagnostics=CovarianceDiagnostics(
            source=source,
            factor_ids=factor_ids,
            observation_count=observation_count,
            minimum_eigenvalue=float(eigenvalues[0]),
            effective_rank=rank,
            condition_number=condition,
            daily_means=mean_values.tolist(),
        ),
    )


def estimate_sample_covariance(
    factor_ids: list[str], observations: FloatArray
) -> PreparedCovariance:
    values = np.asarray(observations, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != len(factor_ids):
        raise PortfolioRiskError(
            "at least two complete historical factor observations are required"
        )
    means = np.mean(values, axis=0)
    covariance = np.atleast_2d(np.cov(values, rowvar=False, ddof=1))
    return validate_covariance(
        factor_ids,
        covariance,
        means,
        source=CovarianceSource.HISTORICAL_SAMPLE,
        observation_count=values.shape[0],
    )
