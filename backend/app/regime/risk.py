"""Regime-conditioned factor risk using Phase 17 loss conventions."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from backend.app.core.exceptions import RegimeModelError
from backend.app.models.regime import (
    RegimeRiskRequest,
    RegimeRiskResult,
    RegimeRiskValuationMode,
    StateConditionalRisk,
)
from backend.app.risk.statistics import empirical_tail, pnl_to_losses
from backend.app.simulation.random import create_rng, spawn_child_seeds

FloatArray = NDArray[np.float64]


def _physical_emissions(request: RegimeRiskRequest) -> tuple[FloatArray, FloatArray]:
    parameters = request.parameters
    means = np.asarray(parameters.means, dtype=np.float64)
    covariances = np.asarray(parameters.covariances, dtype=np.float64)
    if parameters.standardization_mean is not None:
        center = np.asarray(parameters.standardization_mean, dtype=np.float64)
        scale = np.asarray(parameters.standardization_std, dtype=np.float64)
        means = center[None, :] + means * scale[None, :]
        covariances = np.einsum("i,kij,j->kij", scale, covariances, scale)
    return means, covariances


def _covariance_factor(covariance: FloatArray) -> FloatArray:
    values, vectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    scale = max(1.0, float(np.max(np.abs(covariance))))
    if values[0] < -1e-10 * scale:
        raise RegimeModelError("regime covariance must be positive semidefinite")
    return np.asarray(vectors * np.sqrt(np.maximum(values, 0.0)), dtype=np.float64)


def _pnl(
    samples: FloatArray,
    delta: FloatArray,
    gamma: FloatArray | None,
) -> FloatArray:
    result = samples @ delta
    if gamma is not None:
        result += 0.5 * np.sum(gamma[None, :] * samples * samples, axis=1)
    return np.asarray(result, dtype=np.float64)


def regime_conditioned_risk(request: RegimeRiskRequest) -> RegimeRiskResult:
    """Compute conditional state risk and nonlinear Monte Carlo mixture VaR/ES.

    The mixture samples regimes and factors jointly. It never averages conditional
    state VaRs, which would not be a quantile of the mixture loss distribution.
    """

    means, covariances = _physical_emissions(request)
    probabilities = np.asarray(request.state_probabilities, dtype=np.float64)
    delta = np.asarray(request.delta_exposures, dtype=np.float64)
    gamma = (
        None
        if request.valuation_mode is RegimeRiskValuationMode.DELTA
        else np.asarray(request.diagonal_gamma_exposures, dtype=np.float64)
    )
    state_risks: list[StateConditionalRisk] = []
    z_score = float(norm.ppf(request.confidence_level))
    seeds = spawn_child_seeds(request.seed, request.parameters.num_states + 1)
    state_sample_count = request.num_simulations
    for state in range(request.parameters.num_states):
        pnl_mean = float(delta @ means[state])
        pnl_variance = max(0.0, float(delta @ covariances[state] @ delta))
        pnl_std = math.sqrt(pnl_variance)
        if gamma is None:
            value_at_risk = -pnl_mean + z_score * pnl_std
            expected_shortfall = -pnl_mean + pnl_std * float(norm.pdf(z_score)) / (
                1.0 - request.confidence_level
            )
        else:
            rng = create_rng(seeds[state])
            draws = rng.standard_normal((state_sample_count, means.shape[1]))
            samples = means[state] + draws @ _covariance_factor(covariances[state]).T
            pnl_values = _pnl(samples, delta, gamma)
            tail = empirical_tail(pnl_to_losses(pnl_values), request.confidence_level)
            value_at_risk = tail.value_at_risk
            expected_shortfall = tail.expected_shortfall
            pnl_mean = float(np.mean(pnl_values))
            pnl_std = float(np.std(pnl_values, ddof=1))
        state_risks.append(
            StateConditionalRisk(
                state=state,
                probability=float(probabilities[state]),
                value_at_risk=value_at_risk,
                expected_shortfall=expected_shortfall,
                pnl_mean=pnl_mean,
                pnl_standard_deviation=pnl_std,
                mean=means[state].tolist(),
                covariance=covariances[state].tolist(),
            )
        )

    rng = create_rng(seeds[-1])
    sampled_states = rng.choice(
        request.parameters.num_states,
        size=request.num_simulations,
        p=probabilities,
    )
    samples = np.empty((request.num_simulations, means.shape[1]), dtype=np.float64)
    for state in range(request.parameters.num_states):
        indices = np.flatnonzero(sampled_states == state)
        if indices.size:
            draws = rng.standard_normal((indices.size, means.shape[1]))
            samples[indices] = means[state] + draws @ _covariance_factor(covariances[state]).T
    pnl_values = _pnl(samples, delta, gamma)
    tail = empirical_tail(pnl_to_losses(pnl_values), request.confidence_level)
    return RegimeRiskResult(
        confidence_level=request.confidence_level,
        valuation_mode=request.valuation_mode,
        state_probabilities=request.state_probabilities,
        state_risks=state_risks,
        mixture_value_at_risk=tail.value_at_risk,
        mixture_expected_shortfall=tail.expected_shortfall,
        mixture_pnl_mean=float(np.mean(pnl_values)),
        mixture_pnl_standard_deviation=(
            float(np.std(pnl_values, ddof=1)) if pnl_values.size > 1 else 0.0
        ),
        num_simulations=request.num_simulations,
        seed=request.seed,
    )
