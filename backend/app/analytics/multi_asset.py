"""Diagnostics for Gaussian multi-asset correlation generation."""

import numpy as np
from numpy.typing import NDArray

from backend.app.models.multi_asset import (
    CorrelationDiagnosticConfig,
    CorrelationFactorizationDiagnostics,
    EmpiricalCorrelationResult,
    MultiAssetMarketEnvironment,
)
from backend.app.simulation.correlation import factorize_correlation
from backend.app.simulation.multi_asset_gbm import (
    correlate_normal_drivers,
    multi_asset_normal_drivers,
)
from backend.app.simulation.random import create_rng


def empirical_correlation_diagnostics(
    market: MultiAssetMarketEnvironment,
    config: CorrelationDiagnosticConfig | None = None,
) -> EmpiricalCorrelationResult:
    """Simulate correlated standard normals and compare sample and target matrices."""

    effective_config = config or CorrelationDiagnosticConfig()
    factorization = factorize_correlation(market.correlation_matrix)
    independent = multi_asset_normal_drivers(
        effective_config.num_simulations,
        1,
        market.num_assets,
        create_rng(effective_config.seed, effective_config.random_engine),
        antithetic=effective_config.antithetic,
    )[:, 0, :]
    correlated = correlate_normal_drivers(independent, factorization)
    empirical: NDArray[np.float64]
    if market.num_assets == 1:
        empirical = np.ones((1, 1), dtype=np.float64)
    else:
        empirical = np.asarray(np.corrcoef(correlated, rowvar=False), dtype=np.float64)
    target = np.asarray(market.correlation_matrix, dtype=np.float64)
    absolute = np.abs(empirical - target)
    return EmpiricalCorrelationResult(
        num_simulations=effective_config.num_simulations,
        seed=effective_config.seed,
        target_matrix=target.tolist(),
        empirical_matrix=empirical.tolist(),
        absolute_error_matrix=absolute.tolist(),
        maximum_absolute_error=float(np.max(absolute)),
        frobenius_error=float(np.linalg.norm(empirical - target, ord="fro")),
        factorization=CorrelationFactorizationDiagnostics(
            dimension=market.num_assets,
            method=factorization.method,
            min_eigenvalue=factorization.min_eigenvalue,
            max_eigenvalue=factorization.max_eigenvalue,
            effective_rank=factorization.effective_rank,
            condition_number=factorization.condition_number,
            repaired=market.correlation_repaired,
            adjustment_magnitude=market.correlation_adjustment_magnitude,
        ),
    )
