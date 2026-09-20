"""Deterministic synthetic Gaussian-HMM data and real-world switching GBM research paths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from backend.app.models.regime import (
    GaussianHMMParameters,
    HMMObservationDataset,
    MarkovSwitchingGBMConfig,
)
from backend.app.simulation.random import create_rng

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class SyntheticHMMData:
    latent_states: IntArray
    observations: FloatArray

    def dataset(self, feature_names: list[str] | None = None) -> HMMObservationDataset:
        return HMMObservationDataset(
            observations=self.observations.tolist(), feature_names=feature_names
        )


@dataclass(frozen=True)
class MarkovSwitchingGBMPaths:
    spots: FloatArray
    states: IntArray


def generate_synthetic_hmm(
    parameters: GaussianHMMParameters,
    length: int,
    seed: int,
) -> SyntheticHMMData:
    if length <= 0:
        raise ValueError("length must be positive")
    rng = create_rng(seed)
    transition = np.asarray(parameters.transition_matrix, dtype=np.float64)
    means = np.asarray(parameters.means, dtype=np.float64)
    covariances = np.asarray(parameters.covariances, dtype=np.float64)
    states = np.empty(length, dtype=np.int64)
    observations = np.empty((length, parameters.feature_dimension), dtype=np.float64)
    states[0] = rng.choice(parameters.num_states, p=parameters.initial_probabilities)
    observations[0] = rng.multivariate_normal(means[states[0]], covariances[states[0]])
    for index in range(1, length):
        states[index] = rng.choice(parameters.num_states, p=transition[states[index - 1]])
        observations[index] = rng.multivariate_normal(
            means[states[index]], covariances[states[index]]
        )
    return SyntheticHMMData(states, observations)


def simulate_markov_switching_gbm(config: MarkovSwitchingGBMConfig) -> MarkovSwitchingGBMPaths:
    """Simulate real-world research paths; this is not a derivative-pricing process."""

    rng = create_rng(config.seed)
    transition = np.asarray(config.transition_matrix, dtype=np.float64)
    drifts = np.asarray(config.drifts, dtype=np.float64)
    volatilities = np.asarray(config.volatilities, dtype=np.float64)
    states = np.empty((config.num_paths, config.num_steps + 1), dtype=np.int64)
    spots = np.empty((config.num_paths, config.num_steps + 1), dtype=np.float64)
    states[:, 0] = rng.choice(
        len(config.drifts), size=config.num_paths, p=config.initial_probabilities
    )
    spots[:, 0] = config.initial_spot
    root_dt = np.sqrt(config.time_step)
    for step in range(1, config.num_steps + 1):
        uniforms = rng.random(config.num_paths)
        cumulative = np.cumsum(transition[states[:, step - 1]], axis=1)
        states[:, step] = np.sum(uniforms[:, None] > cumulative, axis=1)
        current = states[:, step]
        shocks = rng.standard_normal(config.num_paths)
        log_returns = (
            drifts[current] - 0.5 * volatilities[current] ** 2
        ) * config.time_step + volatilities[current] * root_dt * shocks
        spots[:, step] = spots[:, step - 1] * np.exp(log_returns)
    return MarkovSwitchingGBMPaths(spots=spots, states=states)


def standardize_dataset(
    dataset: HMMObservationDataset,
) -> tuple[HMMObservationDataset, list[float], list[float]]:
    """Explicitly standardize features and return the reusable transform."""

    values = np.asarray(dataset.observations, dtype=np.float64)
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0, ddof=0)
    std = np.where(std > 0.0, std, 1.0)
    transformed = HMMObservationDataset(
        observations=((values - mean) / std).tolist(),
        timestamps=dataset.timestamps,
        feature_names=dataset.feature_names,
    )
    return transformed, mean.tolist(), std.tolist()
