import numpy as np
import pytest

from backend.app.core.exceptions import NumericalError
from backend.app.models.longstaff_schwartz import (
    BasisFamily,
    LongstaffSchwartzConfig,
    RegressionFallback,
)
from backend.app.pricing.lsm_regression import basis_matrix, fit_continuation, predict_continuation


def test_power_basis_uses_strike_normalization() -> None:
    x = np.array([0.5, 1.0, 2.0])
    actual = basis_matrix(x * 100, 100, BasisFamily.POWER, 3)
    np.testing.assert_allclose(actual, np.column_stack([np.ones(3), x, x * x, x * x * x]))
    np.testing.assert_array_equal(basis_matrix(x * 200, 200, BasisFamily.POWER, 3), actual)


def test_standard_unweighted_laguerre_polynomials() -> None:
    x = np.array([0.5, 1.0, 2.0])
    expected = np.column_stack(
        [np.ones(3), 1 - x, 1 - 2 * x + x * x / 2, 1 - 3 * x + 1.5 * x * x - x * x * x / 6]
    )
    np.testing.assert_allclose(
        basis_matrix(x * 100, 100, BasisFamily.LAGUERRE, 3), expected, atol=1e-15
    )


@pytest.mark.parametrize("family", list(BasisFamily))
def test_regression_recovers_known_polynomial_and_diagnostics(family: BasisFamily) -> None:
    stock = np.linspace(60, 99, 30)
    matrix = basis_matrix(stock, 100, family, 2)
    targets = np.einsum("ij,j->i", matrix, [3.0, 2.0, 1.0])
    date = fit_continuation(
        stock, targets, 100, LongstaffSchwartzConfig(basis_family=family), time_index=1, time=0.1
    )
    np.testing.assert_allclose(date.coefficients, [3, 2, 1], atol=1e-10)
    np.testing.assert_allclose(predict_continuation(stock, 100, date), targets)
    assert date.matrix_rank == 3
    assert date.regression_path_count == 30
    assert date.residual_sum_squares is not None and date.residual_sum_squares < 1e-20
    assert date.condition_number is not None
    assert not date.fallback_used


@pytest.mark.parametrize(
    "stock,reason",
    [
        (np.array([]), RegressionFallback.NO_ITM_PATHS),
        (np.array([80.0, 90.0]), RegressionFallback.INSUFFICIENT_PATHS),
        (np.full(30, 80.0), RegressionFallback.RANK_DEFICIENT),
    ],
)
def test_explicit_continue_fallbacks(stock: np.ndarray, reason: RegressionFallback) -> None:
    date = fit_continuation(
        stock, np.ones(stock.size), 100, LongstaffSchwartzConfig(), time_index=1, time=0.2
    )
    assert date.fallback_used and date.fallback_reason is reason
    assert date.coefficients is None
    if reason is RegressionFallback.RANK_DEFICIENT:
        assert date.matrix_rank == 1


def test_condition_warning_does_not_reject_full_rank_fit() -> None:
    stock = np.linspace(90, 99, 30)
    date = fit_continuation(
        stock,
        stock / 100,
        100,
        LongstaffSchwartzConfig(condition_warning_threshold=2),
        time_index=1,
        time=0.1,
    )
    assert date.condition_warning
    assert not date.fallback_used


def test_nonnegative_continuation_floor() -> None:
    stock = np.linspace(60, 99, 30)
    date = fit_continuation(
        stock,
        np.full(30, -2.0),
        100,
        LongstaffSchwartzConfig(basis_degree=0),
        time_index=1,
        time=0.1,
    )
    np.testing.assert_array_equal(predict_continuation(stock, 100, date), np.zeros(30))


def test_nonfinite_state_is_not_hidden_by_small_sample_fallback() -> None:
    with pytest.raises(NumericalError, match="regression states"):
        fit_continuation(
            np.array([np.nan]), np.ones(1), 100, LongstaffSchwartzConfig(), time_index=1, time=0.1
        )
