"""Risk-neutral geometric Brownian motion terminal simulation."""

import math

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from backend.app.core.exceptions import NumericalError
from backend.app.core.numerics import finite_array_exp, finite_exp, require_finite
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract
from backend.app.models.simulation import PathSimulationConfig, SimulationConfig
from backend.app.simulation.base import PathStochasticProcess, StochasticProcess


class GBMProcess(StochasticProcess, PathStochasticProcess):
    """Exact risk-neutral GBM terminal and full-path simulation."""

    def simulate_paths(
        self,
        contract: OptionContract,
        market: MarketEnvironment,
        config: PathSimulationConfig,
        rng: Generator,
    ) -> NDArray[np.float64]:
        """Simulate exact transitions with interleaved antithetic path rows."""

        count, steps = config.num_simulations, config.num_time_steps
        if contract.maturity_years == 0.0:
            return np.full((count, steps + 1), contract.spot, dtype=np.float64)
        if market.volatility == 0.0:
            times = np.linspace(0.0, contract.maturity_years, steps + 1, dtype=np.float64)
            path = contract.spot * finite_array_exp(
                (market.risk_free_rate - market.dividend_yield) * times,
                context="deterministic GBM path",
            )
            require_finite(path, context="deterministic GBM path")
            if np.any(path <= 0.0):
                raise NumericalError("deterministic GBM path is outside positive finite range")
            return np.broadcast_to(path, (count, steps + 1)).copy()
        normals = path_normal_drivers(
            count, steps, rng, antithetic=config.variance_reduction.uses_antithetic
        )
        return paths_from_normals(contract, market, normals)

    def simulate_terminal(
        self,
        contract: OptionContract,
        market: MarketEnvironment,
        config: SimulationConfig,
        rng: Generator,
    ) -> NDArray[np.float64]:
        """Simulate terminal values, preserving exact requested path count.

        Antithetic output is interleaved as ``Z, -Z``. For odd counts, the last
        independently drawn ``Z`` is retained as a documented unpaired outcome.
        """

        count = config.num_simulations
        maturity = contract.maturity_years
        volatility = market.volatility

        if maturity == 0.0:
            return np.full(count, contract.spot, dtype=np.float64)

        if volatility == 0.0:
            drift = (market.risk_free_rate - market.dividend_yield) * maturity
            growth = finite_exp(drift, context="deterministic GBM growth factor")
            deterministic_terminal = contract.spot * growth
            if not math.isfinite(deterministic_terminal) or deterministic_terminal <= 0.0:
                raise NumericalError(
                    "deterministic GBM terminal price is outside positive finite range"
                )
            return np.full(count, deterministic_terminal, dtype=np.float64)

        volatility_variance = volatility * volatility
        if not math.isfinite(volatility_variance):
            raise NumericalError("GBM volatility variance overflowed")
        drift = (
            market.risk_free_rate - market.dividend_yield - 0.5 * volatility_variance
        ) * maturity
        if not math.isfinite(drift):
            raise NumericalError("GBM terminal drift overflowed")

        normals = standard_normal_drivers(
            count,
            rng,
            antithetic=config.variance_reduction.uses_antithetic,
        )
        return terminal_prices_from_normals(contract, market, normals)

    @staticmethod
    def _antithetic_normals(count: int, rng: Generator) -> NDArray[np.float64]:
        return standard_normal_drivers(count, rng, antithetic=True)


def standard_normal_drivers(
    count: int,
    rng: Generator,
    *,
    antithetic: bool,
) -> NDArray[np.float64]:
    """Generate explicit reusable normal drivers with the Phase 1 pairing policy."""

    if count <= 0:
        raise ValueError("normal driver count must be positive")
    if not antithetic:
        return np.asarray(rng.standard_normal(count), dtype=np.float64)

    pair_count, remainder = divmod(count, 2)
    primary = np.asarray(rng.standard_normal(pair_count + remainder), dtype=np.float64)
    normals = np.empty(count, dtype=np.float64)
    normals[0 : 2 * pair_count : 2] = primary[:pair_count]
    normals[1 : 2 * pair_count : 2] = -primary[:pair_count]
    if remainder:
        normals[-1] = primary[-1]
    return normals


def terminal_prices_from_normals(
    contract: OptionContract,
    market: MarketEnvironment,
    normals: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate terminal GBM values from caller-owned standard-normal drivers."""

    drivers = np.asarray(normals, dtype=np.float64)
    if drivers.ndim != 1 or drivers.size == 0:
        raise ValueError("normal drivers must be a non-empty one-dimensional array")
    require_finite(drivers, context="GBM normal drivers")

    maturity = contract.maturity_years
    if maturity == 0.0:
        return np.full(drivers.size, contract.spot, dtype=np.float64)

    volatility = market.volatility
    if volatility == 0.0:
        drift = (market.risk_free_rate - market.dividend_yield) * maturity
        growth = finite_exp(drift, context="deterministic GBM growth factor")
        terminal = contract.spot * growth
        if not math.isfinite(terminal) or terminal <= 0.0:
            raise NumericalError(
                "deterministic GBM terminal price is outside positive finite range"
            )
        return np.full(drivers.size, terminal, dtype=np.float64)

    volatility_variance = volatility * volatility
    if not math.isfinite(volatility_variance):
        raise NumericalError("GBM volatility variance overflowed")
    drift = (market.risk_free_rate - market.dividend_yield - 0.5 * volatility_variance) * maturity
    if not math.isfinite(drift):
        raise NumericalError("GBM terminal drift overflowed")

    log_returns = drift + volatility * math.sqrt(maturity) * drivers
    terminal_values = contract.spot * finite_array_exp(
        log_returns,
        context="GBM terminal prices",
    )
    require_finite(terminal_values, context="GBM terminal prices")
    return terminal_values


def theoretical_terminal_mean(
    contract: OptionContract,
    market: MarketEnvironment,
) -> float:
    """Return E[S_T] under the risk-neutral GBM measure."""

    growth = finite_exp(
        (market.risk_free_rate - market.dividend_yield) * contract.maturity_years,
        context="GBM theoretical mean growth factor",
    )
    result = contract.spot * growth
    if not math.isfinite(result) or result <= 0.0:
        raise NumericalError("GBM theoretical mean overflowed floating-point range")
    return result


def path_normal_drivers(
    count: int, steps: int, rng: Generator, *, antithetic: bool
) -> NDArray[np.float64]:
    """Generate row-wise normal vectors; an odd final row is an independent singleton."""

    if count <= 0 or steps <= 0:
        raise ValueError("path count and time steps must be positive")
    if not antithetic:
        return np.asarray(rng.standard_normal((count, steps)), dtype=np.float64)
    pairs, remainder = divmod(count, 2)
    primary = rng.standard_normal((pairs + remainder, steps))
    normals = np.empty((count, steps), dtype=np.float64)
    normals[: 2 * pairs : 2] = primary[:pairs]
    normals[1 : 2 * pairs : 2] = -primary[:pairs]
    if remainder:
        normals[-1] = primary[-1]
    return normals


def paths_from_normals(
    contract: OptionContract, market: MarketEnvironment, normals: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Evaluate exact log-GBM increments from a caller-owned (paths, steps) matrix."""

    if normals.ndim != 2 or min(normals.shape) == 0:
        raise ValueError("path drivers must be a nonempty two-dimensional matrix")
    require_finite(normals, context="GBM path normal drivers")
    count, steps = normals.shape
    if contract.maturity_years == 0.0:
        return np.full((count, steps + 1), contract.spot, dtype=np.float64)
    dt = contract.maturity_years / steps
    if contract.maturity_years > 0.0 and dt == 0.0:
        raise NumericalError("GBM path time step underflowed")
    paths = np.empty((count, steps + 1), dtype=np.float64)
    paths[:, 0] = contract.spot
    with np.errstate(over="ignore", invalid="ignore"):
        increments = market.volatility * math.sqrt(dt) * normals
        increments += (
            market.risk_free_rate
            - market.dividend_yield
            - 0.5 * market.volatility * market.volatility
        ) * dt
        np.cumsum(increments, axis=1, out=increments)
        increments += math.log(contract.spot)
    paths[:, 1:] = finite_array_exp(increments, context="GBM path prices")
    return paths


def observation_prices_from_normals(
    spot: float,
    market: MarketEnvironment,
    observation_times: NDArray[np.float64],
    normals: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate exact GBM states at arbitrary strictly positive observation times."""

    times = np.asarray(observation_times, dtype=np.float64)
    drivers = np.asarray(normals, dtype=np.float64)
    if times.ndim != 1 or drivers.ndim != 2 or drivers.shape[1] != times.size:
        raise ValueError("normal columns must match the one-dimensional observation-time array")
    if drivers.shape[0] == 0:
        raise ValueError("at least one path is required")
    require_finite(times, context="GBM observation times")
    require_finite(drivers, context="GBM observation normal drivers")
    if not math.isfinite(spot) or spot <= 0.0:
        raise ValueError("spot must be positive and finite")
    if times.size == 0:
        return np.empty((drivers.shape[0], 0), dtype=np.float64)
    deltas = np.diff(np.concatenate((np.zeros(1, dtype=np.float64), times)))
    if np.any(deltas <= 0.0):
        raise ValueError("observation times must be strictly increasing and positive")

    volatility = np.float64(market.volatility)
    with np.errstate(over="ignore", invalid="ignore"):
        drift_rate = np.float64(market.risk_free_rate - market.dividend_yield) - (
            np.float64(0.5) * volatility * volatility
        )
        increments = volatility * np.sqrt(deltas) * drivers + drift_rate * deltas
        np.cumsum(increments, axis=1, out=increments)
        increments += math.log(spot)
    require_finite(increments, context="GBM observation log prices")
    return finite_array_exp(increments, context="GBM observation prices")


def simulate_observation_prices(
    spot: float,
    market: MarketEnvironment,
    observation_times: NDArray[np.float64],
    count: int,
    rng: Generator,
    *,
    antithetic: bool,
) -> NDArray[np.float64]:
    """Simulate only requested GBM observations, preserving path-pair correspondence."""

    times = np.asarray(observation_times, dtype=np.float64)
    if count <= 0:
        raise ValueError("path count must be positive")
    if times.ndim != 1:
        raise ValueError("observation times must be one-dimensional")
    if times.size == 0:
        return np.empty((count, 0), dtype=np.float64)
    if market.volatility == 0.0:
        deterministic = spot * finite_array_exp(
            (market.risk_free_rate - market.dividend_yield) * times,
            context="deterministic GBM observations",
        )
        return np.broadcast_to(deterministic, (count, times.size)).copy()
    normals = path_normal_drivers(count, times.size, rng, antithetic=antithetic)
    return observation_prices_from_normals(spot, market, times, normals)
