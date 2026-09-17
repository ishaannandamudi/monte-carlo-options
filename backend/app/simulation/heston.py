"""Risk-neutral Heston simulation with FTE and Andersen QE-M schemes."""

import math
from dataclasses import dataclass

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from backend.app.core.exceptions import HestonPricingError, NumericalError
from backend.app.core.numerics import finite_array_exp, require_finite
from backend.app.models.heston import (
    HestonParameters,
    HestonSimulationConfig,
    HestonSimulationScheme,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract

QE_PSI_THRESHOLD = 1.5
QE_GAMMA_1 = 0.5
QE_GAMMA_2 = 0.5


@dataclass(frozen=True)
class HestonTerminalSimulation:
    terminal_spots: NDArray[np.float64]
    terminal_variances: NDArray[np.float64]
    raw_terminal_variances: NDArray[np.float64]
    negative_variance_proposal_fraction: float | None


@dataclass(frozen=True)
class HestonPathSimulation:
    spot_paths: NDArray[np.float64]
    variance_paths: NDArray[np.float64]
    negative_variance_proposal_fraction: float | None


def heston_step_drivers(
    count: int,
    rng: Generator,
    *,
    antithetic: bool,
    include_uniform: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
    """Generate interleaved variance/perpendicular normals and optional QE uniforms."""

    if count <= 0:
        raise ValueError("Heston driver count must be positive")
    if not antithetic:
        variance = np.asarray(rng.standard_normal(count), dtype=np.float64)
        perpendicular = np.asarray(rng.standard_normal(count), dtype=np.float64)
        uniforms = np.asarray(rng.random(count), dtype=np.float64) if include_uniform else None
        return variance, perpendicular, uniforms

    pairs, remainder = divmod(count, 2)
    primary_count = pairs + remainder
    primary_variance = np.asarray(rng.standard_normal(primary_count), dtype=np.float64)
    primary_perpendicular = np.asarray(rng.standard_normal(primary_count), dtype=np.float64)
    variance = np.empty(count, dtype=np.float64)
    perpendicular = np.empty(count, dtype=np.float64)
    variance[: 2 * pairs : 2] = primary_variance[:pairs]
    variance[1 : 2 * pairs : 2] = -primary_variance[:pairs]
    perpendicular[: 2 * pairs : 2] = primary_perpendicular[:pairs]
    perpendicular[1 : 2 * pairs : 2] = -primary_perpendicular[:pairs]
    if remainder:
        variance[-1] = primary_variance[-1]
        perpendicular[-1] = primary_perpendicular[-1]

    if not include_uniform:
        return variance, perpendicular, None
    primary_uniforms = np.asarray(rng.random(primary_count), dtype=np.float64)
    uniforms = np.empty(count, dtype=np.float64)
    uniforms[: 2 * pairs : 2] = primary_uniforms[:pairs]
    uniforms[1 : 2 * pairs : 2] = 1.0 - primary_uniforms[:pairs]
    if remainder:
        uniforms[-1] = primary_uniforms[-1]
    return variance, perpendicular, uniforms


def correlated_heston_normals(
    variance_normals: NDArray[np.float64],
    perpendicular_normals: NDArray[np.float64],
    rho: float,
) -> NDArray[np.float64]:
    """Return spot normals with ``corr(Z_s, Z_v)=rho``."""

    z_variance = np.asarray(variance_normals, dtype=np.float64)
    z_perpendicular = np.asarray(perpendicular_normals, dtype=np.float64)
    if z_variance.shape != z_perpendicular.shape or z_variance.ndim != 1:
        raise ValueError("Heston normal-driver vectors must have matching one-dimensional shape")
    require_finite(z_variance, context="Heston variance normal drivers")
    require_finite(z_perpendicular, context="Heston perpendicular normal drivers")
    residual = math.sqrt(max(0.0, 1.0 - rho * rho))
    spot = np.asarray(rho * z_variance + residual * z_perpendicular, dtype=np.float64)
    require_finite(spot, context="Heston spot normal drivers")
    return spot


def full_truncation_euler_step(
    log_spot: NDArray[np.float64],
    variance: NDArray[np.float64],
    dt: float,
    variance_normals: NDArray[np.float64],
    spot_normals: NDArray[np.float64],
    risk_free_rate: float,
    dividend_yield: float,
    parameters: HestonParameters,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Apply one canonical FTE step while retaining the raw variance proposal."""

    if dt <= 0.0 or not math.isfinite(dt):
        raise ValueError("Heston time step must be positive and finite")
    values = [log_spot, variance, variance_normals, spot_normals]
    if any(np.asarray(value).ndim != 1 for value in values):
        raise ValueError("FTE state and driver inputs must be one-dimensional")
    if len({np.asarray(value).size for value in values}) != 1:
        raise ValueError("FTE state and driver vectors must have matching lengths")
    for value, context in zip(
        values,
        ("FTE log spots", "FTE variances", "FTE variance normals", "FTE spot normals"),
        strict=True,
    ):
        require_finite(np.asarray(value, dtype=np.float64), context=context)

    positive = np.maximum(np.asarray(variance, dtype=np.float64), 0.0)
    root_variance_dt = np.sqrt(positive * dt)
    next_log_spot = np.asarray(
        log_spot
        + (risk_free_rate - dividend_yield - 0.5 * positive) * dt
        + root_variance_dt * spot_normals,
        dtype=np.float64,
    )
    next_variance = np.asarray(
        variance
        + parameters.kappa * (parameters.theta - positive) * dt
        + parameters.sigma_v * root_variance_dt * variance_normals,
        dtype=np.float64,
    )
    require_finite(next_log_spot, context="FTE updated log spots")
    require_finite(next_variance, context="FTE updated variance proposals")
    return next_log_spot, next_variance, int(np.count_nonzero(next_variance < 0.0))


def qe_conditional_moments(
    variance: NDArray[np.float64],
    dt: float,
    parameters: HestonParameters,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return exact conditional CIR mean and variance over one uniform time step."""

    current = np.asarray(variance, dtype=np.float64)
    if current.ndim != 1 or current.size == 0 or np.any(current < 0.0):
        raise ValueError("QE variance state must be a non-empty nonnegative vector")
    if dt <= 0.0 or not math.isfinite(dt):
        raise ValueError("QE time step must be positive and finite")
    require_finite(current, context="QE variance state")
    sigma_squared = parameters.sigma_v * parameters.sigma_v
    if parameters.kappa == 0.0:
        mean = current.copy()
        conditional_variance = sigma_squared * current * dt
    else:
        decay = math.exp(-parameters.kappa * dt)
        one_minus_decay = -math.expm1(-parameters.kappa * dt)
        mean = parameters.theta + (current - parameters.theta) * decay
        conditional_variance = (
            current * sigma_squared * decay * one_minus_decay / parameters.kappa
            + parameters.theta
            * sigma_squared
            * one_minus_decay
            * one_minus_decay
            / (2.0 * parameters.kappa)
        )
    mean = np.asarray(mean, dtype=np.float64)
    conditional_variance = np.asarray(np.maximum(conditional_variance, 0.0), dtype=np.float64)
    require_finite(mean, context="QE conditional mean")
    require_finite(conditional_variance, context="QE conditional variance")
    return mean, conditional_variance


def _qe_variance_and_log_mgf(
    variance: NDArray[np.float64],
    dt: float,
    variance_normals: NDArray[np.float64],
    uniforms: NDArray[np.float64],
    parameters: HestonParameters,
    mgf_argument: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    mean, conditional_variance = qe_conditional_moments(variance, dt, parameters)
    z = np.asarray(variance_normals, dtype=np.float64)
    uniform = np.asarray(uniforms, dtype=np.float64)
    if z.shape != mean.shape or uniform.shape != mean.shape:
        raise ValueError("QE drivers must match the variance-state shape")
    require_finite(z, context="QE variance normals")
    require_finite(uniform, context="QE uniforms")
    if np.any(uniform < 0.0) or np.any(uniform > 1.0):
        raise ValueError("QE uniforms must lie in [0, 1]")

    next_variance = np.empty_like(mean)
    log_mgf = np.empty_like(mean)
    deterministic = conditional_variance == 0.0
    next_variance[deterministic] = mean[deterministic]
    log_mgf[deterministic] = mgf_argument * mean[deterministic]

    stochastic = ~deterministic
    psi = np.zeros_like(mean)
    positive_mean = stochastic & (mean > 0.0)
    psi[positive_mean] = conditional_variance[positive_mean] / (mean[positive_mean] ** 2)
    impossible = stochastic & ~positive_mean
    if np.any(impossible):
        raise HestonPricingError("QE conditional moments have variance with zero mean")

    low = stochastic & (psi <= QE_PSI_THRESHOLD)
    if np.any(low):
        inverse_psi = 2.0 / psi[low]
        b_squared = inverse_psi - 1.0 + np.sqrt(inverse_psi * (inverse_psi - 1.0))
        a = mean[low] / (1.0 + b_squared)
        next_variance[low] = a * (np.sqrt(b_squared) + z[low]) ** 2
        denominator = 1.0 - 2.0 * mgf_argument * a
        if np.any(denominator <= 0.0):
            raise HestonPricingError(
                "QE-M low-psi moment condition failed; increase the time-step count"
            )
        log_mgf[low] = mgf_argument * a * b_squared / denominator - 0.5 * np.log(denominator)

    high = stochastic & ~low
    if np.any(high):
        p = (psi[high] - 1.0) / (psi[high] + 1.0)
        beta = (1.0 - p) / mean[high]
        upper = np.nextafter(1.0, 0.0)
        draws = np.clip(uniform[high], np.nextafter(0.0, 1.0), upper)
        positive_draw = draws > p
        values = np.zeros_like(draws)
        values[positive_draw] = (
            np.log((1.0 - p[positive_draw]) / (1.0 - draws[positive_draw])) / beta[positive_draw]
        )
        next_variance[high] = values
        if np.any(beta <= mgf_argument):
            raise HestonPricingError(
                "QE-M high-psi moment condition failed; increase the time-step count"
            )
        mgf = p + (1.0 - p) * beta / (beta - mgf_argument)
        if np.any(mgf <= 0.0) or not np.all(np.isfinite(mgf)):
            raise HestonPricingError("QE-M high-psi moment is invalid")
        log_mgf[high] = np.log(mgf)

    require_finite(next_variance, context="QE updated variances")
    require_finite(log_mgf, context="QE conditional log-MGF")
    if np.any(next_variance < 0.0):
        raise HestonPricingError("QE produced a negative variance")
    return next_variance, log_mgf


def quadratic_exponential_variance_step(
    variance: NDArray[np.float64],
    dt: float,
    variance_normals: NDArray[np.float64],
    uniforms: NDArray[np.float64],
    parameters: HestonParameters,
) -> NDArray[np.float64]:
    """Apply only the QE CIR transition for fixed-driver moment tests."""

    next_variance, _ = _qe_variance_and_log_mgf(
        variance,
        dt,
        variance_normals,
        uniforms,
        parameters,
        0.0,
    )
    return next_variance


def quadratic_exponential_martingale_step(
    log_spot: NDArray[np.float64],
    variance: NDArray[np.float64],
    dt: float,
    variance_normals: NDArray[np.float64],
    perpendicular_normals: NDArray[np.float64],
    uniforms: NDArray[np.float64],
    risk_free_rate: float,
    dividend_yield: float,
    parameters: HestonParameters,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply Andersen's QE variance transition and martingale-corrected spot update."""

    if parameters.sigma_v == 0.0:
        raise HestonPricingError("QE-M stochastic step requires positive sigma_v")
    current = np.asarray(variance, dtype=np.float64)
    z_perpendicular = np.asarray(perpendicular_normals, dtype=np.float64)
    if current.shape != np.asarray(log_spot).shape or current.shape != z_perpendicular.shape:
        raise ValueError("QE-M state and perpendicular drivers must have matching shapes")

    rho_over_sigma = parameters.rho / parameters.sigma_v
    common = parameters.kappa * rho_over_sigma - 0.5
    k1 = QE_GAMMA_1 * dt * common - rho_over_sigma
    k2 = QE_GAMMA_2 * dt * common + rho_over_sigma
    k3 = QE_GAMMA_1 * dt * (1.0 - parameters.rho * parameters.rho)
    k4 = QE_GAMMA_2 * dt * (1.0 - parameters.rho * parameters.rho)
    mgf_argument = k2 + 0.5 * k4
    next_variance, log_mgf = _qe_variance_and_log_mgf(
        current,
        dt,
        variance_normals,
        uniforms,
        parameters,
        mgf_argument,
    )
    martingale_constant = -(k1 + 0.5 * k3) * current - log_mgf
    innovation_variance = np.maximum(k3 * current + k4 * next_variance, 0.0)
    next_log_spot = np.asarray(
        log_spot
        + (risk_free_rate - dividend_yield) * dt
        + martingale_constant
        + k1 * current
        + k2 * next_variance
        + np.sqrt(innovation_variance) * z_perpendicular,
        dtype=np.float64,
    )
    require_finite(next_log_spot, context="QE-M updated log spots")
    return next_log_spot, next_variance


def _deterministic_variance_transition(
    variance: NDArray[np.float64], dt: float, parameters: HestonParameters
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if parameters.kappa == 0.0:
        next_variance = variance.copy()
        integrated_variance = variance * dt
    else:
        decay = math.exp(-parameters.kappa * dt)
        next_variance = parameters.theta + (variance - parameters.theta) * decay
        integrated_variance = (
            parameters.theta * dt
            + (variance - parameters.theta)
            * (-math.expm1(-parameters.kappa * dt))
            / parameters.kappa
        )
    return (
        np.asarray(np.maximum(next_variance, 0.0), dtype=np.float64),
        np.asarray(np.maximum(integrated_variance, 0.0), dtype=np.float64),
    )


def _advance_step(
    log_spot: NDArray[np.float64],
    variance: NDArray[np.float64],
    dt: float,
    variance_normals: NDArray[np.float64],
    perpendicular_normals: NDArray[np.float64],
    uniforms: NDArray[np.float64] | None,
    market: MarketEnvironment,
    parameters: HestonParameters,
    scheme: HestonSimulationScheme,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    if parameters.sigma_v == 0.0:
        next_variance, integrated = _deterministic_variance_transition(
            np.maximum(variance, 0.0), dt, parameters
        )
        spot_normals = correlated_heston_normals(
            variance_normals, perpendicular_normals, parameters.rho
        )
        next_log_spot = np.asarray(
            log_spot
            + (market.risk_free_rate - market.dividend_yield) * dt
            - 0.5 * integrated
            + np.sqrt(integrated) * spot_normals,
            dtype=np.float64,
        )
        require_finite(next_log_spot, context="deterministic-variance Heston log spots")
        return next_log_spot, next_variance, 0
    if scheme is HestonSimulationScheme.FULL_TRUNCATION_EULER:
        spot_normals = correlated_heston_normals(
            variance_normals, perpendicular_normals, parameters.rho
        )
        return full_truncation_euler_step(
            log_spot,
            variance,
            dt,
            variance_normals,
            spot_normals,
            market.risk_free_rate,
            market.dividend_yield,
            parameters,
        )
    if uniforms is None:
        raise ValueError("QE simulation requires uniform drivers")
    next_log, next_variance = quadratic_exponential_martingale_step(
        log_spot,
        np.maximum(variance, 0.0),
        dt,
        variance_normals,
        perpendicular_normals,
        uniforms,
        market.risk_free_rate,
        market.dividend_yield,
        parameters,
    )
    return next_log, next_variance, 0


def simulate_heston_from_drivers(
    contract: OptionContract,
    market: MarketEnvironment,
    parameters: HestonParameters,
    scheme: HestonSimulationScheme,
    variance_normals: NDArray[np.float64],
    perpendicular_normals: NDArray[np.float64],
    uniforms: NDArray[np.float64] | None = None,
) -> HestonTerminalSimulation:
    """Evaluate terminal Heston states from caller-owned ``(paths, steps)`` drivers."""

    z_variance = np.asarray(variance_normals, dtype=np.float64)
    z_perpendicular = np.asarray(perpendicular_normals, dtype=np.float64)
    if (
        z_variance.ndim != 2
        or min(z_variance.shape) == 0
        or z_variance.shape != z_perpendicular.shape
    ):
        raise ValueError("Heston fixed normal drivers must have matching (paths, steps) shape")
    if scheme is HestonSimulationScheme.QUADRATIC_EXPONENTIAL:
        if uniforms is None or np.asarray(uniforms).shape != z_variance.shape:
            raise ValueError("QE fixed uniforms must match the normal-driver shape")
        uniform_array = np.asarray(uniforms, dtype=np.float64)
    else:
        uniform_array = None
    require_finite(z_variance, context="fixed Heston variance drivers")
    require_finite(z_perpendicular, context="fixed Heston perpendicular drivers")

    count, steps = z_variance.shape
    if contract.maturity_years == 0.0:
        return HestonTerminalSimulation(
            terminal_spots=np.full(count, contract.spot, dtype=np.float64),
            terminal_variances=np.full(count, parameters.v0, dtype=np.float64),
            raw_terminal_variances=np.full(count, parameters.v0, dtype=np.float64),
            negative_variance_proposal_fraction=(
                0.0 if scheme is HestonSimulationScheme.FULL_TRUNCATION_EULER else None
            ),
        )
    dt = contract.maturity_years / steps
    if dt == 0.0:
        raise NumericalError("Heston time step underflowed")
    log_spot: NDArray[np.float64] = np.full(count, math.log(contract.spot), dtype=np.float64)
    variance: NDArray[np.float64] = np.full(count, parameters.v0, dtype=np.float64)
    negative_count = 0
    for step in range(steps):
        step_uniforms = uniform_array[:, step] if uniform_array is not None else None
        log_spot, variance, negatives = _advance_step(
            log_spot,
            variance,
            dt,
            z_variance[:, step],
            z_perpendicular[:, step],
            step_uniforms,
            market,
            parameters,
            scheme,
        )
        negative_count += negatives
    terminal_spots = finite_array_exp(log_spot, context="Heston terminal spots")
    effective_variance = np.asarray(np.maximum(variance, 0.0), dtype=np.float64)
    return HestonTerminalSimulation(
        terminal_spots=terminal_spots,
        terminal_variances=effective_variance,
        raw_terminal_variances=np.asarray(variance, dtype=np.float64),
        negative_variance_proposal_fraction=(
            negative_count / (count * steps)
            if scheme is HestonSimulationScheme.FULL_TRUNCATION_EULER
            else None
        ),
    )


def simulate_heston_terminal(
    contract: OptionContract,
    market: MarketEnvironment,
    parameters: HestonParameters,
    config: HestonSimulationConfig,
    rng: Generator,
) -> HestonTerminalSimulation:
    """Simulate terminal states with O(paths) memory."""

    count = config.num_simulations
    if contract.maturity_years == 0.0:
        return HestonTerminalSimulation(
            terminal_spots=np.full(count, contract.spot, dtype=np.float64),
            terminal_variances=np.full(count, parameters.v0, dtype=np.float64),
            raw_terminal_variances=np.full(count, parameters.v0, dtype=np.float64),
            negative_variance_proposal_fraction=(
                0.0 if config.scheme is HestonSimulationScheme.FULL_TRUNCATION_EULER else None
            ),
        )
    dt = contract.maturity_years / config.num_time_steps
    if dt == 0.0:
        raise NumericalError("Heston time step underflowed")
    log_spot: NDArray[np.float64] = np.full(count, math.log(contract.spot), dtype=np.float64)
    variance: NDArray[np.float64] = np.full(count, parameters.v0, dtype=np.float64)
    negative_count = 0
    include_uniform = config.scheme is HestonSimulationScheme.QUADRATIC_EXPONENTIAL
    for _ in range(config.num_time_steps):
        z_variance, z_perpendicular, uniforms = heston_step_drivers(
            count,
            rng,
            antithetic=config.antithetic,
            include_uniform=include_uniform,
        )
        log_spot, variance, negatives = _advance_step(
            log_spot,
            variance,
            dt,
            z_variance,
            z_perpendicular,
            uniforms,
            market,
            parameters,
            config.scheme,
        )
        negative_count += negatives
    terminal_spots = finite_array_exp(log_spot, context="Heston terminal spots")
    return HestonTerminalSimulation(
        terminal_spots=terminal_spots,
        terminal_variances=np.asarray(np.maximum(variance, 0.0), dtype=np.float64),
        raw_terminal_variances=np.asarray(variance, dtype=np.float64),
        negative_variance_proposal_fraction=(
            negative_count / (count * config.num_time_steps)
            if config.scheme is HestonSimulationScheme.FULL_TRUNCATION_EULER
            else None
        ),
    )


def simulate_heston_paths(
    contract: OptionContract,
    market: MarketEnvironment,
    parameters: HestonParameters,
    config: HestonSimulationConfig,
    rng: Generator,
) -> HestonPathSimulation:
    """Simulate full spot and variance paths for diagnostics only."""

    count, steps = config.num_simulations, config.num_time_steps
    spot_paths = np.empty((count, steps + 1), dtype=np.float64)
    variance_paths = np.empty((count, steps + 1), dtype=np.float64)
    spot_paths[:, 0] = contract.spot
    variance_paths[:, 0] = parameters.v0
    if contract.maturity_years == 0.0:
        spot_paths[:, 1:] = contract.spot
        variance_paths[:, 1:] = parameters.v0
        return HestonPathSimulation(
            spot_paths=spot_paths,
            variance_paths=variance_paths,
            negative_variance_proposal_fraction=(
                0.0 if config.scheme is HestonSimulationScheme.FULL_TRUNCATION_EULER else None
            ),
        )
    dt = contract.maturity_years / steps
    if dt == 0.0:
        raise NumericalError("Heston time step underflowed")
    log_spot: NDArray[np.float64] = np.full(count, math.log(contract.spot), dtype=np.float64)
    variance: NDArray[np.float64] = np.full(count, parameters.v0, dtype=np.float64)
    negative_count = 0
    include_uniform = config.scheme is HestonSimulationScheme.QUADRATIC_EXPONENTIAL
    for step in range(steps):
        z_variance, z_perpendicular, uniforms = heston_step_drivers(
            count,
            rng,
            antithetic=config.antithetic,
            include_uniform=include_uniform,
        )
        log_spot, variance, negatives = _advance_step(
            log_spot,
            variance,
            dt,
            z_variance,
            z_perpendicular,
            uniforms,
            market,
            parameters,
            config.scheme,
        )
        spot_paths[:, step + 1] = finite_array_exp(log_spot, context="Heston spot paths")
        variance_paths[:, step + 1] = variance
        negative_count += negatives
    require_finite(spot_paths, context="Heston spot paths")
    require_finite(variance_paths, context="Heston variance paths")
    return HestonPathSimulation(
        spot_paths=spot_paths,
        variance_paths=variance_paths,
        negative_variance_proposal_fraction=(
            negative_count / (count * steps)
            if config.scheme is HestonSimulationScheme.FULL_TRUNCATION_EULER
            else None
        ),
    )


def theoretical_variance_mean(parameters: HestonParameters, time_years: float) -> float:
    """Return the exact CIR variance-process first moment."""

    if not math.isfinite(time_years) or time_years < 0.0:
        raise ValueError("variance-mean time must be finite and non-negative")
    if parameters.kappa == 0.0:
        return parameters.v0
    value = parameters.theta + (parameters.v0 - parameters.theta) * math.exp(
        -parameters.kappa * time_years
    )
    if not math.isfinite(value) or value < 0.0:
        raise NumericalError("theoretical Heston variance mean is invalid")
    return value
