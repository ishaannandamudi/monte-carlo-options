import numpy as np
import pytest
from pydantic import ValidationError

from backend.app.analytics.multi_asset import empirical_correlation_diagnostics
from backend.app.core.exceptions import MultiAssetPricingError
from backend.app.models.multi_asset import (
    CorrelationDiagnosticConfig,
    MultiAssetMarketEnvironment,
)
from backend.app.simulation.correlation import (
    CorrelationFactorizationMethod,
    build_equicorrelation_matrix,
    factorize_correlation,
    prepare_correlation_matrix,
)


def market(correlation: tuple[tuple[float, ...], ...]) -> MultiAssetMarketEnvironment:
    count = len(correlation)
    return MultiAssetMarketEnvironment(
        spots=(100.0,) * count,
        risk_free_rate=0.02,
        dividend_yields=(0.01,) * count,
        volatilities=(0.2,) * count,
        correlation_matrix=correlation,
    )


@pytest.mark.parametrize(
    "update",
    [
        {"spots": (100.0, -1.0)},
        {"volatilities": (0.2, -0.1)},
        {"spots": (100.0, float("nan"))},
        {"risk_free_rate": float("inf")},
        {"dividend_yields": (0.0,)},
        {"correlation_matrix": ((1.0, 0.2),)},
        {"correlation_matrix": ((1.0, 0.2), (0.1, 1.0))},
        {"correlation_matrix": ((0.9, 0.2), (0.2, 1.0))},
        {"correlation_matrix": ((1.0, 1.1), (1.1, 1.0))},
        {"asset_ids": ("A", "A")},
    ],
)
def test_market_rejects_invalid_inputs(update: dict[str, object]) -> None:
    data: dict[str, object] = {
        "spots": (100.0, 90.0),
        "risk_free_rate": -0.01,
        "dividend_yields": (0.0, 0.02),
        "volatilities": (0.2, 0.3),
        "correlation_matrix": ((1.0, 0.3), (0.3, 1.0)),
        "asset_ids": ("A", "B"),
    }
    data.update(update)
    with pytest.raises((ValidationError, MultiAssetPricingError)):
        MultiAssetMarketEnvironment.model_validate(data)


def test_materially_non_psd_matrix_is_rejected() -> None:
    invalid = ((1.0, 0.9, 0.9), (0.9, 1.0, -0.9), (0.9, -0.9, 1.0))
    with pytest.raises(MultiAssetPricingError, match="materially"):
        prepare_correlation_matrix(invalid)


def test_tiny_negative_eigenvalue_is_repaired_and_exposed() -> None:
    rho = -0.5 - 1e-12
    repaired = market(build_equicorrelation_matrix(3, -0.5)).model_copy(
        update={"correlation_matrix": ((1.0, rho, rho), (rho, 1.0, rho), (rho, rho, 1.0))}
    )
    # model_copy deliberately bypasses validation, so exercise the public constructor.
    repaired = MultiAssetMarketEnvironment.model_validate(repaired.model_dump())
    assert repaired.correlation_repaired
    assert repaired.correlation_adjustment_magnitude > 0.0
    assert np.linalg.eigvalsh(repaired.correlation_matrix)[0] >= -1e-12


def test_factorization_uses_cholesky_and_singular_eigen_fallback() -> None:
    identity = factorize_correlation(((1.0, 0.0), (0.0, 1.0)))
    regular = factorize_correlation(((1.0, 0.4), (0.4, 1.0)))
    singular = factorize_correlation(((1.0, 1.0), (1.0, 1.0)))
    assert identity.method is CorrelationFactorizationMethod.CHOLESKY
    assert regular.method is CorrelationFactorizationMethod.CHOLESKY
    assert singular.method is CorrelationFactorizationMethod.EIGEN
    assert singular.effective_rank == 1 and singular.condition_number is None
    np.testing.assert_allclose(singular.factor @ singular.factor.T, np.ones((2, 2)))


def test_equicorrelation_bounds() -> None:
    np.testing.assert_allclose(
        build_equicorrelation_matrix(3, -0.5),
        np.eye(3) - 0.5 * (1 - np.eye(3)),
    )
    with pytest.raises(MultiAssetPricingError):
        build_equicorrelation_matrix(3, -0.5001)
    with pytest.raises(MultiAssetPricingError):
        build_equicorrelation_matrix(2, 1.01)


@pytest.mark.parametrize("rho", [0.0, 0.65, -0.6])
def test_empirical_correlation_matches_target(rho: float) -> None:
    result = empirical_correlation_diagnostics(
        market(((1.0, rho), (rho, 1.0))),
        CorrelationDiagnosticConfig(num_simulations=40_000, seed=71),
    )
    assert result.maximum_absolute_error < 0.02
    assert result.frobenius_error < 0.03


def test_single_asset_empirical_correlation() -> None:
    result = empirical_correlation_diagnostics(
        market(((1.0,),)), CorrelationDiagnosticConfig(num_simulations=10, seed=1)
    )
    assert result.empirical_matrix == [[1.0]]


def test_identity_empirical_off_diagonal_is_near_zero() -> None:
    result = empirical_correlation_diagnostics(
        market(((1.0, 0.0), (0.0, 1.0))),
        CorrelationDiagnosticConfig(num_simulations=30_000, seed=22),
    )
    assert abs(result.empirical_matrix[0][1]) < 0.02
    assert result.factorization.symmetric and result.factorization.positive_semidefinite
