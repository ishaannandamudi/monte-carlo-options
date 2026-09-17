"""Exact correlated multi-asset risk-neutral GBM simulation."""

import math

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from backend.app.core.exceptions import MultiAssetPricingError, NumericalError
from backend.app.core.numerics import finite_array_exp, require_finite
from backend.app.models.multi_asset import MultiAssetMarketEnvironment
from backend.app.simulation.correlation import CorrelationFactorization, factorize_correlation


def multi_asset_normal_drivers(
    count: int,
    steps: int,
    assets: int,
    rng: Generator,
    *,
    antithetic: bool,
) -> NDArray[np.float64]:
    """Generate independent Gaussian vectors with interleaved antithetic paths."""

    if count <= 0 or steps <= 0 or assets <= 0:
        raise ValueError("path count, step count, and asset count must be positive")
    if not antithetic:
        return np.asarray(rng.standard_normal((count, steps, assets)), dtype=np.float64)
    pair_count, remainder = divmod(count, 2)
    primary = np.asarray(
        rng.standard_normal((pair_count + remainder, steps, assets)), dtype=np.float64
    )
    drivers = np.empty((count, steps, assets), dtype=np.float64)
    drivers[: 2 * pair_count : 2] = primary[:pair_count]
    drivers[1 : 2 * pair_count : 2] = -primary[:pair_count]
    if remainder:
        drivers[-1] = primary[-1]
    return drivers


def correlate_normal_drivers(
    independent: NDArray[np.float64],
    factorization: CorrelationFactorization,
) -> NDArray[np.float64]:
    """Apply the correlation factor along the final asset axis."""

    drivers = np.asarray(independent, dtype=np.float64)
    if drivers.ndim not in {2, 3} or drivers.shape[-1] != factorization.factor.shape[0]:
        raise ValueError("normal-driver asset dimension must match correlation factor")
    require_finite(drivers, context="multi-asset independent normal drivers")
    with np.errstate(all="ignore"):
        correlated = np.asarray(drivers @ factorization.factor.T, dtype=np.float64)
    require_finite(correlated, context="multi-asset correlated normal drivers")
    return correlated


def terminal_prices_from_normals(
    market: MultiAssetMarketEnvironment,
    maturity_years: float,
    independent_normals: NDArray[np.float64],
    factorization: CorrelationFactorization | None = None,
) -> NDArray[np.float64]:
    """Evaluate terminal prices from caller-owned independent Gaussian vectors."""

    drivers = np.asarray(independent_normals, dtype=np.float64)
    if drivers.ndim != 2 or drivers.shape[0] == 0 or drivers.shape[1] != market.num_assets:
        raise ValueError("terminal drivers must have shape (paths, assets)")
    if not math.isfinite(maturity_years) or maturity_years < 0.0:
        raise ValueError("maturity must be finite and non-negative")
    require_finite(drivers, context="multi-asset terminal normal drivers")
    count = drivers.shape[0]
    spots = np.asarray(market.spots, dtype=np.float64)
    if maturity_years == 0.0:
        return np.broadcast_to(spots, (count, market.num_assets)).copy()

    volatilities = np.asarray(market.volatilities, dtype=np.float64)
    dividends = np.asarray(market.dividend_yields, dtype=np.float64)
    if np.all(volatilities == 0.0):
        deterministic = spots * finite_array_exp(
            (market.risk_free_rate - dividends) * maturity_years,
            context="deterministic multi-asset terminal prices",
        )
        return np.broadcast_to(deterministic, (count, market.num_assets)).copy()

    decomposition = factorization or factorize_correlation(market.correlation_matrix)
    correlated = correlate_normal_drivers(drivers, decomposition)
    with np.errstate(over="ignore", invalid="ignore"):
        variance_rates = volatilities * volatilities
        drift = (market.risk_free_rate - dividends - 0.5 * variance_rates) * maturity_years
        log_returns = drift + volatilities * math.sqrt(maturity_years) * correlated
    require_finite(log_returns, context="multi-asset terminal log returns")
    terminal = spots * finite_array_exp(
        np.asarray(log_returns, dtype=np.float64), context="multi-asset terminal prices"
    )
    require_finite(terminal, context="multi-asset terminal prices")
    return terminal


def simulate_correlated_terminal(
    market: MultiAssetMarketEnvironment,
    maturity_years: float,
    count: int,
    rng: Generator,
    *,
    antithetic: bool,
    factorization: CorrelationFactorization | None = None,
) -> NDArray[np.float64]:
    """Simulate a ``(paths, assets)`` terminal matrix without full-path allocation."""

    if count <= 0:
        raise ValueError("path count must be positive")
    if maturity_years == 0.0 or all(value == 0.0 for value in market.volatilities):
        empty_drivers = np.zeros((count, market.num_assets), dtype=np.float64)
        return terminal_prices_from_normals(
            market, maturity_years, empty_drivers, factorization=factorization
        )
    drivers = multi_asset_normal_drivers(count, 1, market.num_assets, rng, antithetic=antithetic)[
        :, 0, :
    ]
    return terminal_prices_from_normals(
        market, maturity_years, drivers, factorization=factorization
    )


def paths_from_normals(
    market: MultiAssetMarketEnvironment,
    observation_times: NDArray[np.float64],
    independent_normals: NDArray[np.float64],
    factorization: CorrelationFactorization | None = None,
) -> NDArray[np.float64]:
    """Evaluate exact arbitrary-time transitions in ``(paths, times, assets)`` order.

    ``observation_times`` contains strictly positive future times. The returned
    matrix prepends time zero, so its middle dimension has one additional state.
    """

    times = np.asarray(observation_times, dtype=np.float64)
    drivers = np.asarray(independent_normals, dtype=np.float64)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("observation times must be a non-empty one-dimensional array")
    if (
        drivers.ndim != 3
        or drivers.shape[0] == 0
        or drivers.shape[1] != times.size
        or drivers.shape[2] != market.num_assets
    ):
        raise ValueError("path drivers must have shape (paths, observation_times, assets)")
    require_finite(times, context="multi-asset observation times")
    require_finite(drivers, context="multi-asset path normal drivers")
    deltas = np.diff(np.concatenate((np.zeros(1, dtype=np.float64), times)))
    if np.any(deltas <= 0.0):
        raise ValueError("observation times must be strictly increasing and positive")
    if np.any(deltas == 0.0):
        raise NumericalError("multi-asset observation time increment underflowed")

    factor = factorization or factorize_correlation(market.correlation_matrix)
    correlated = correlate_normal_drivers(drivers, factor)
    volatilities = np.asarray(market.volatilities, dtype=np.float64)
    dividends = np.asarray(market.dividend_yields, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        drift_rates = market.risk_free_rate - dividends - 0.5 * volatilities * volatilities
        increments = (
            drift_rates[None, None, :] * deltas[None, :, None]
            + volatilities[None, None, :] * np.sqrt(deltas)[None, :, None] * correlated
        )
        np.cumsum(increments, axis=1, out=increments)
        increments += np.log(np.asarray(market.spots, dtype=np.float64))[None, None, :]
    require_finite(increments, context="multi-asset path log prices")
    future = finite_array_exp(increments, context="multi-asset path prices")
    paths = np.empty((drivers.shape[0], times.size + 1, market.num_assets), dtype=np.float64)
    paths[:, 0, :] = np.asarray(market.spots, dtype=np.float64)
    paths[:, 1:, :] = future
    return paths


def simulate_correlated_paths(
    market: MultiAssetMarketEnvironment,
    observation_times: NDArray[np.float64],
    count: int,
    rng: Generator,
    *,
    antithetic: bool,
    factorization: CorrelationFactorization | None = None,
) -> NDArray[np.float64]:
    """Simulate exact correlated paths at caller-selected deterministic times."""

    times = np.asarray(observation_times, dtype=np.float64)
    if count <= 0:
        raise ValueError("path count must be positive")
    if times.ndim != 1 or times.size == 0:
        raise ValueError("observation times must be a non-empty one-dimensional array")
    if np.any(np.asarray(market.volatilities) != 0.0):
        drivers = multi_asset_normal_drivers(
            count, times.size, market.num_assets, rng, antithetic=antithetic
        )
    else:
        drivers = np.zeros((count, times.size, market.num_assets), dtype=np.float64)
    return paths_from_normals(market, times, drivers, factorization=factorization)


def theoretical_terminal_means(
    market: MultiAssetMarketEnvironment, maturity_years: float
) -> NDArray[np.float64]:
    """Return each risk-neutral theoretical terminal mean."""

    if not math.isfinite(maturity_years) or maturity_years < 0.0:
        raise MultiAssetPricingError("maturity must be finite and non-negative")
    spots = np.asarray(market.spots, dtype=np.float64)
    dividends = np.asarray(market.dividend_yields, dtype=np.float64)
    means = spots * finite_array_exp(
        (market.risk_free_rate - dividends) * maturity_years,
        context="multi-asset theoretical terminal means",
    )
    require_finite(means, context="multi-asset theoretical terminal means")
    return means
