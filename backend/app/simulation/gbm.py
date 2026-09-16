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

        drift = (market.risk_free_rate - market.dividend_yield - 0.5 * volatility**2) * maturity
        if volatility == 0.0:
            growth = finite_exp(drift, context="deterministic GBM growth factor")
            deterministic_terminal = contract.spot * growth
            if not math.isfinite(deterministic_terminal) or deterministic_terminal <= 0.0:
                from backend.app.core.exceptions import NumericalError

                raise NumericalError(
                    "deterministic GBM terminal price is outside positive finite range"
                )
            return np.full(count, deterministic_terminal, dtype=np.float64)

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
    drift = (market.risk_free_rate - market.dividend_yield - 0.5 * volatility**2) * maturity
    if volatility == 0.0:
        growth = finite_exp(drift, context="deterministic GBM growth factor")
        terminal = contract.spot * growth
        if not math.isfinite(terminal) or terminal <= 0.0:
            from backend.app.core.exceptions import NumericalError

            raise NumericalError(
                "deterministic GBM terminal price is outside positive finite range"
            )
        return np.full(drivers.size, terminal, dtype=np.float64)

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
        from backend.app.core.exceptions import NumericalError

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
