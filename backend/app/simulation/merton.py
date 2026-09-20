"""Exact terminal and interval simulation for Merton compound-Poisson dynamics."""

import math
from dataclasses import dataclass

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from backend.app.core.exceptions import MertonPricingError
from backend.app.core.numerics import finite_array_exp, finite_exp, require_finite
from backend.app.models.market import MarketEnvironment
from backend.app.models.merton import (
    MertonJumpParameters,
    MertonPathSimulationConfig,
    MertonSimulationConfig,
)
from backend.app.models.option import OptionContract


@dataclass(frozen=True)
class MertonTerminalSimulation:
    terminal_spots: NDArray[np.float64]
    jump_counts: NDArray[np.int64]


@dataclass(frozen=True)
class MertonPathSimulation:
    spot_paths: NDArray[np.float64]
    interval_jump_counts: NDArray[np.int64]


def _jump_compensator(parameters: MertonJumpParameters) -> float:
    try:
        return parameters.jump_compensator
    except ValueError as exc:
        raise MertonPricingError(str(exc)) from exc


def terminal_prices_from_drivers(
    contract: OptionContract,
    market: MarketEnvironment,
    parameters: MertonJumpParameters,
    jump_counts: NDArray[np.integer],
    diffusion_normals: NDArray[np.float64],
    aggregate_jump_normals: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate terminal spots from fixed counts and independent standard normals."""

    counts = np.asarray(jump_counts, dtype=np.int64)
    diffusion = np.asarray(diffusion_normals, dtype=np.float64)
    jump_normals = np.asarray(aggregate_jump_normals, dtype=np.float64)
    if counts.ndim != 1 or counts.size == 0:
        raise ValueError("jump counts must be a non-empty one-dimensional array")
    if diffusion.shape != counts.shape or jump_normals.shape != counts.shape:
        raise ValueError("fixed Merton drivers must have matching one-dimensional shapes")
    if np.any(counts < 0):
        raise ValueError("jump counts must be non-negative")
    require_finite(diffusion, context="Merton diffusion drivers")
    require_finite(jump_normals, context="Merton aggregate-jump drivers")
    maturity = contract.maturity_years
    if maturity == 0.0:
        return np.full(counts.size, contract.spot, dtype=np.float64)
    sigma = market.volatility
    drift = (
        market.risk_free_rate
        - market.dividend_yield
        - parameters.jump_intensity * _jump_compensator(parameters)
        - 0.5 * sigma * sigma
    ) * maturity
    aggregate_jump = (
        counts.astype(np.float64) * parameters.jump_mean
        + np.sqrt(counts.astype(np.float64)) * parameters.jump_volatility * jump_normals
    )
    log_returns = drift + sigma * math.sqrt(maturity) * diffusion + aggregate_jump
    terminal = contract.spot * finite_array_exp(log_returns, context="Merton terminal prices")
    require_finite(terminal, context="Merton terminal prices")
    if np.any(terminal <= 0.0):
        raise MertonPricingError("Merton terminal prices must remain positive")
    return np.asarray(terminal, dtype=np.float64)


def simulate_merton_terminal(
    contract: OptionContract,
    market: MarketEnvironment,
    parameters: MertonJumpParameters,
    config: MertonSimulationConfig,
    rng: Generator,
) -> MertonTerminalSimulation:
    """Simulate exact terminal values using one aggregate normal per path."""

    count = config.num_simulations
    maturity = contract.maturity_years
    if maturity == 0.0:
        return MertonTerminalSimulation(
            np.full(count, contract.spot, dtype=np.float64), np.zeros(count, dtype=np.int64)
        )
    mean_count = parameters.jump_intensity * maturity
    if not math.isfinite(mean_count):
        raise MertonPricingError("expected Merton jump count must be finite")
    try:
        counts = (
            np.zeros(count, dtype=np.int64)
            if mean_count == 0.0
            else np.asarray(rng.poisson(mean_count, count), dtype=np.int64)
        )
    except ValueError as exc:
        raise MertonPricingError("expected Merton jump count exceeds Poisson RNG range") from exc
    diffusion = (
        np.zeros(count, dtype=np.float64)
        if market.volatility == 0.0
        else np.asarray(rng.standard_normal(count), dtype=np.float64)
    )
    aggregate = (
        np.zeros(count, dtype=np.float64)
        if parameters.jump_volatility == 0.0 or not np.any(counts)
        else np.asarray(rng.standard_normal(count), dtype=np.float64)
    )
    terminal = terminal_prices_from_drivers(
        contract, market, parameters, counts, diffusion, aggregate
    )
    return MertonTerminalSimulation(terminal, counts)


def simulate_merton_paths(
    contract: OptionContract,
    market: MarketEnvironment,
    parameters: MertonJumpParameters,
    config: MertonPathSimulationConfig,
    rng: Generator,
) -> MertonPathSimulation:
    """Simulate exact independent Merton increments on a regular time grid."""

    count, steps = config.num_simulations, config.num_time_steps
    paths = np.empty((count, steps + 1), dtype=np.float64)
    paths[:, 0] = contract.spot
    jump_counts = np.zeros((count, steps), dtype=np.int64)
    if contract.maturity_years == 0.0:
        paths[:, 1:] = contract.spot
        return MertonPathSimulation(paths, jump_counts)
    dt = contract.maturity_years / steps
    if dt == 0.0:
        raise MertonPricingError("Merton path time step underflowed")
    sigma = market.volatility
    drift = (
        market.risk_free_rate
        - market.dividend_yield
        - parameters.jump_intensity * _jump_compensator(parameters)
        - 0.5 * sigma * sigma
    ) * dt
    log_spot = np.full(count, math.log(contract.spot), dtype=np.float64)
    for step in range(steps):
        mean_count = parameters.jump_intensity * dt
        if not math.isfinite(mean_count):
            raise MertonPricingError("Merton interval jump count must be finite")
        try:
            counts = (
                np.zeros(count, dtype=np.int64)
                if mean_count == 0.0
                else np.asarray(rng.poisson(mean_count, count), dtype=np.int64)
            )
        except ValueError as exc:
            raise MertonPricingError(
                "Merton interval jump count exceeds Poisson RNG range"
            ) from exc
        diffusion = (
            np.zeros(count, dtype=np.float64)
            if sigma == 0.0
            else np.asarray(rng.standard_normal(count), dtype=np.float64)
        )
        jump_normal = (
            np.zeros(count, dtype=np.float64)
            if parameters.jump_volatility == 0.0 or not np.any(counts)
            else np.asarray(rng.standard_normal(count), dtype=np.float64)
        )
        log_spot += (
            drift
            + sigma * math.sqrt(dt) * diffusion
            + counts.astype(np.float64) * parameters.jump_mean
            + np.sqrt(counts.astype(np.float64)) * parameters.jump_volatility * jump_normal
        )
        paths[:, step + 1] = finite_array_exp(log_spot, context="Merton spot paths")
        jump_counts[:, step] = counts
    require_finite(paths, context="Merton spot paths")
    return MertonPathSimulation(paths, jump_counts)


def theoretical_terminal_mean(contract: OptionContract, market: MarketEnvironment) -> float:
    growth = finite_exp(
        (market.risk_free_rate - market.dividend_yield) * contract.maturity_years,
        context="Merton theoretical mean growth factor",
    )
    result = contract.spot * growth
    if not math.isfinite(result) or result <= 0.0:
        raise MertonPricingError("Merton theoretical terminal mean is outside finite range")
    return result
