import numpy as np
import pytest

from backend.app.models.qmc import QMCPathConstruction
from backend.app.simulation.qmc import (
    brownian_bridge_order,
    brownian_bridge_values,
    brownian_increments_from_normals,
    pca_brownian_values,
    pca_path_diagnostics,
)
from backend.app.simulation.random import create_rng


def test_fixed_brownian_bridge_mapping_and_reconstruction() -> None:
    times = np.array([0.25, 0.5, 1.0])
    normals = np.array([[1.0, 0.0, 0.0]])
    assert brownian_bridge_order(times) == [2, 1, 0]
    values = brownian_bridge_values(normals, times)
    np.testing.assert_allclose(values, [[0.25, 0.5, 1.0]], atol=1e-14)
    increments = brownian_increments_from_normals(
        normals, times, QMCPathConstruction.BROWNIAN_BRIDGE
    )
    np.testing.assert_allclose(np.cumsum(increments, axis=1), values, atol=1e-14)


def test_bridge_covariance_and_increment_variance() -> None:
    times = np.array([0.1, 0.35, 0.7, 1.0])
    normals = create_rng(12).standard_normal((100_000, times.size))
    values = brownian_bridge_values(normals, times)
    empirical = np.cov(values, rowvar=False, ddof=0)
    np.testing.assert_allclose(empirical, np.minimum(times[:, None], times[None, :]), atol=0.012)
    increments = np.diff(np.column_stack((np.zeros(values.shape[0]), values)), axis=1)
    np.testing.assert_allclose(
        np.var(increments, axis=0), np.diff(np.concatenate(([0.0], times))), atol=0.008
    )


def test_pca_reconstructs_brownian_covariance_and_reports_concentration() -> None:
    times = np.linspace(0.1, 1.0, 10)
    diagnostics = pca_path_diagnostics(times)
    with np.errstate(all="ignore"):
        covariance = diagnostics.factor @ diagnostics.factor.T
    np.testing.assert_allclose(covariance, np.minimum(times[:, None], times[None, :]), atol=1e-12)
    assert np.all(np.diff(diagnostics.cumulative_variance_explained) >= 0.0)
    assert diagnostics.cumulative_variance_explained[-1] == pytest.approx(1.0)
    normals = create_rng(2).standard_normal((50_000, times.size))
    values = pca_brownian_values(normals, times)
    np.testing.assert_allclose(np.cov(values, rowvar=False), covariance, atol=0.025)
