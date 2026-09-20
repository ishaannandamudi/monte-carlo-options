"""Numerically stable finite-state Markov-chain utilities."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from backend.app.core.exceptions import RegimeModelError
from backend.app.models.regime import (
    MarkovChain,
    MarkovDiagnostics,
    MarkovDiagnosticsResult,
    MarkovStateDiagnostic,
    ProbabilityForecast,
)

FloatArray = NDArray[np.float64]


def _matrix(chain: MarkovChain) -> FloatArray:
    return np.asarray(chain.transition_matrix, dtype=np.float64)


def stationary_distribution(
    chain: MarkovChain,
) -> tuple[FloatArray, bool, bool, int]:
    """Return one stationary distribution and uniqueness/reducibility diagnostics.

    A reducible chain can still have a unique stationary distribution when it has
    one closed recurrent class. Uniqueness is therefore based on the nullity of
    ``P.T - I`` rather than reducibility alone.
    """

    matrix = _matrix(chain)
    k = chain.num_states
    system = np.vstack((matrix.T - np.eye(k), np.ones((1, k))))
    target = np.concatenate((np.zeros(k), np.ones(1)))
    solution, *_ = np.linalg.lstsq(system, target, rcond=None)
    solution = np.maximum(solution, 0.0)
    total = float(np.sum(solution))
    if not math.isfinite(total) or total <= 0.0:
        raise RegimeModelError("a finite stationary distribution could not be computed")
    solution /= total
    residual = float(np.max(np.abs(solution @ matrix - solution)))
    if residual > 1e-8:
        raise RegimeModelError("stationary-distribution residual exceeds numerical tolerance")
    rank = int(np.linalg.matrix_rank(matrix.T - np.eye(k), tol=1e-10))
    nullity = k - rank
    irreducible = _is_irreducible(matrix)
    return solution.astype(np.float64), nullity == 1, irreducible, nullity


def _is_irreducible(matrix: FloatArray) -> bool:
    adjacency = matrix > 0.0
    reachability = adjacency.copy()
    np.fill_diagonal(reachability, True)
    for pivot in range(matrix.shape[0]):
        reachability |= reachability[:, pivot, None] & reachability[None, pivot, :]
    return bool(np.all(reachability))


def n_step_transition(chain: MarkovChain, steps: int) -> FloatArray:
    if steps < 0:
        raise RegimeModelError("transition steps must be nonnegative")
    result = np.linalg.matrix_power(_matrix(chain), steps)
    result = np.maximum(result, 0.0)
    result /= np.sum(result, axis=1, keepdims=True)
    return np.asarray(result, dtype=np.float64)


def probability_forecasts(
    chain: MarkovChain,
    current_probabilities: list[float] | FloatArray,
    horizons: list[int],
) -> list[ProbabilityForecast]:
    current = np.asarray(current_probabilities, dtype=np.float64)
    if current.shape != (chain.num_states,) or np.any(current < 0.0):
        raise RegimeModelError("current state probabilities have invalid shape or values")
    if not np.isclose(float(np.sum(current)), 1.0, rtol=0.0, atol=1e-10):
        raise RegimeModelError("current state probabilities must sum to one")
    return [
        ProbabilityForecast(
            horizon=horizon,
            probabilities=(current @ n_step_transition(chain, horizon)).tolist(),
        )
        for horizon in horizons
    ]


def markov_diagnostics(chain: MarkovChain) -> MarkovDiagnostics:
    matrix = _matrix(chain)
    stationary, unique, irreducible, nullity = stationary_distribution(chain)
    states: list[MarkovStateDiagnostic] = []
    for state, persistence in enumerate(np.diag(matrix)):
        absorbing = bool(np.isclose(persistence, 1.0, rtol=0.0, atol=1e-12))
        duration = None if absorbing else 1.0 / (1.0 - float(persistence))
        states.append(
            MarkovStateDiagnostic(
                state=state,
                persistence=float(persistence),
                expected_duration=duration,
                absorbing=absorbing,
                stationary_probability=float(stationary[state]),
            )
        )
    safe = np.where(matrix > 0.0, matrix, 1.0)
    entropies = -np.sum(np.where(matrix > 0.0, matrix * np.log(safe), 0.0), axis=1)
    return MarkovDiagnostics(
        transition_matrix=matrix.tolist(),
        stationary_distribution=stationary.tolist(),
        stationary_distribution_unique=unique,
        irreducible=irreducible,
        stationary_nullity=nullity,
        states=states,
        row_entropies=entropies.tolist(),
    )


def diagnostics_with_forecasts(
    chain: MarkovChain,
    horizons: list[int],
    current_probabilities: list[float] | None,
) -> MarkovDiagnosticsResult:
    current = current_probabilities or chain.initial_probabilities
    if current is None:
        current = [1.0 / chain.num_states] * chain.num_states
    return MarkovDiagnosticsResult(
        diagnostics=markov_diagnostics(chain),
        forecasts=probability_forecasts(chain, current, horizons),
    )
