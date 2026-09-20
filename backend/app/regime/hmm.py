"""Log-space Gaussian HMM inference, fitting, decoding, and forecasts."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
from scipy.special import logsumexp

from backend.app.core.exceptions import RegimeModelError
from backend.app.models.regime import (
    CanonicalOrdering,
    CovarianceType,
    GaussianHMMParameters,
    HMMDecodeRequest,
    HMMDecodeResult,
    HMMFilterResult,
    HMMFitConfig,
    HMMFitResult,
    HMMForecastResult,
    HMMInitialization,
    HMMModelOrderPoint,
    HMMModelOrderRequest,
    HMMModelOrderResult,
    HMMMomentForecast,
    HMMObservationDataset,
    HMMProbabilityPoint,
    HMMStartSummary,
    HMMStateSummary,
    HMMWalkForwardConfig,
    HMMWalkForwardPoint,
    HMMWalkForwardResult,
    MarkovChain,
    ProbabilityOutputMode,
    WalkForwardWindow,
)
from backend.app.regime.markov import markov_diagnostics, n_step_transition
from backend.app.simulation.random import create_rng, spawn_child_seeds

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
LOG_TWO_PI = math.log(2.0 * math.pi)


@dataclass(frozen=True)
class HMMInferenceArrays:
    log_likelihood: float
    filtered: FloatArray
    smoothed: FloatArray
    log_alpha: FloatArray
    log_beta: FloatArray
    expected_transitions: FloatArray


@dataclass(frozen=True)
class _StartFit:
    parameters: GaussianHMMParameters
    inference: HMMInferenceArrays
    converged: bool
    iterations: int
    likelihood_history: list[float]
    message: str | None


def _observations(
    dataset: HMMObservationDataset, parameters: GaussianHMMParameters | None = None
) -> FloatArray:
    values = np.asarray(dataset.observations, dtype=np.float64)
    if parameters is not None:
        if values.shape[1] != parameters.feature_dimension:
            raise RegimeModelError("observation dimension does not match HMM parameters")
        if parameters.standardization_mean is not None:
            mean = np.asarray(parameters.standardization_mean, dtype=np.float64)
            std = np.asarray(parameters.standardization_std, dtype=np.float64)
            values = (values - mean) / std
    return np.asarray(values, dtype=np.float64)


def gaussian_log_emissions(
    observations: FloatArray, parameters: GaussianHMMParameters
) -> FloatArray:
    """Evaluate log Gaussian emission densities for all observations and states."""

    values = np.asarray(observations, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != parameters.feature_dimension:
        raise RegimeModelError("observations have the wrong feature dimension")
    means = np.asarray(parameters.means, dtype=np.float64)
    covariances = np.asarray(parameters.covariances, dtype=np.float64)
    result = np.empty((values.shape[0], parameters.num_states), dtype=np.float64)
    for state in range(parameters.num_states):
        difference = values - means[state]
        if parameters.covariance_type is CovarianceType.DIAGONAL:
            variances = np.diag(covariances[state])
            quadratic = np.sum(difference * difference / variances, axis=1)
            log_determinant = float(np.sum(np.log(variances)))
        else:
            covariance = _floor_covariance(covariances[state], 1e-14)
            sign, log_determinant = np.linalg.slogdet(covariance)
            if sign <= 0.0:
                raise RegimeModelError("a covariance matrix is not positive definite")
            try:
                solved = np.linalg.solve(covariance, difference.T).T
            except np.linalg.LinAlgError as exc:
                raise RegimeModelError("a covariance matrix could not be solved") from exc
            quadratic = np.sum(difference * solved, axis=1)
        result[:, state] = -0.5 * (
            parameters.feature_dimension * LOG_TWO_PI + log_determinant + quadratic
        )
    if not np.all(np.isfinite(result)):
        raise RegimeModelError("Gaussian emission evaluation produced non-finite values")
    return result


def forward_log(
    observations: FloatArray, parameters: GaussianHMMParameters
) -> tuple[float, FloatArray, FloatArray]:
    """Run the forward recursion in log space and return normalized filtering."""

    emissions = gaussian_log_emissions(observations, parameters)
    transition = np.asarray(parameters.transition_matrix, dtype=np.float64)
    initial = np.asarray(parameters.initial_probabilities, dtype=np.float64)
    with np.errstate(divide="ignore"):
        log_transition = np.log(transition)
        log_initial = np.log(initial)
    log_alpha = np.empty_like(emissions)
    log_alpha[0] = log_initial + emissions[0]
    for index in range(1, emissions.shape[0]):
        log_alpha[index] = emissions[index] + logsumexp(
            log_alpha[index - 1, :, None] + log_transition, axis=0
        )
    normalizers = logsumexp(log_alpha, axis=1, keepdims=True)
    filtered = np.exp(log_alpha - normalizers)
    log_likelihood = float(logsumexp(log_alpha[-1]))
    if not math.isfinite(log_likelihood) or not np.all(np.isfinite(filtered)):
        raise RegimeModelError("forward inference produced non-finite output")
    return log_likelihood, filtered, log_alpha


def backward_log(observations: FloatArray, parameters: GaussianHMMParameters) -> FloatArray:
    """Run the backward recursion in log space."""

    emissions = gaussian_log_emissions(observations, parameters)
    with np.errstate(divide="ignore"):
        log_transition = np.log(np.asarray(parameters.transition_matrix, dtype=np.float64))
    log_beta = np.zeros_like(emissions)
    for index in range(emissions.shape[0] - 2, -1, -1):
        log_beta[index] = logsumexp(
            log_transition + emissions[index + 1][None, :] + log_beta[index + 1][None, :],
            axis=1,
        )
    return log_beta


def smooth_probabilities(log_alpha: FloatArray, log_beta: FloatArray) -> FloatArray:
    """Normalize alpha-times-beta into full-sample posterior probabilities."""

    combined = log_alpha + log_beta
    result = np.exp(combined - logsumexp(combined, axis=1, keepdims=True))
    if not np.all(np.isfinite(result)):
        raise RegimeModelError("smoothed probabilities are non-finite")
    return np.asarray(result, dtype=np.float64)


def infer_hmm(observations: FloatArray, parameters: GaussianHMMParameters) -> HMMInferenceArrays:
    """Return filtering, smoothing, and accumulated transition sufficient statistics."""

    log_likelihood, filtered, log_alpha = forward_log(observations, parameters)
    log_beta = backward_log(observations, parameters)
    smoothed = smooth_probabilities(log_alpha, log_beta)
    emissions = gaussian_log_emissions(observations, parameters)
    with np.errstate(divide="ignore"):
        log_transition = np.log(np.asarray(parameters.transition_matrix, dtype=np.float64))
    counts = np.zeros((parameters.num_states, parameters.num_states), dtype=np.float64)
    for index in range(observations.shape[0] - 1):
        log_xi = (
            log_alpha[index, :, None]
            + log_transition
            + emissions[index + 1][None, :]
            + log_beta[index + 1][None, :]
        )
        log_xi -= logsumexp(log_xi)
        counts += np.exp(log_xi)
    return HMMInferenceArrays(
        log_likelihood=log_likelihood,
        filtered=filtered,
        smoothed=smoothed,
        log_alpha=log_alpha,
        log_beta=log_beta,
        expected_transitions=counts,
    )


def viterbi_decode(
    observations: FloatArray, parameters: GaussianHMMParameters
) -> tuple[IntArray, float]:
    """Decode the maximum-probability path with lowest-index deterministic ties."""

    emissions = gaussian_log_emissions(observations, parameters)
    with np.errstate(divide="ignore"):
        log_transition = np.log(np.asarray(parameters.transition_matrix, dtype=np.float64))
    delta = np.empty_like(emissions)
    predecessors = np.zeros(emissions.shape, dtype=np.int64)
    delta[0] = np.log(np.asarray(parameters.initial_probabilities)) + emissions[0]
    for index in range(1, observations.shape[0]):
        candidates = delta[index - 1, :, None] + log_transition
        predecessors[index] = np.argmax(candidates, axis=0)
        delta[index] = candidates[predecessors[index], np.arange(parameters.num_states)]
        delta[index] += emissions[index]
    path = np.empty(observations.shape[0], dtype=np.int64)
    path[-1] = int(np.argmax(delta[-1]))
    for index in range(observations.shape[0] - 2, -1, -1):
        path[index] = predecessors[index + 1, path[index + 1]]
    return path, float(delta[-1, path[-1]])


def fit_gaussian_hmm(
    dataset: HMMObservationDataset,
    config: HMMFitConfig,
    *,
    max_probability_points: int | None = None,
) -> HMMFitResult:
    """Fit a Gaussian HMM with deterministic independent starts and Baum-Welch EM."""

    started = time.perf_counter()
    raw = _observations(dataset)
    if raw.shape[0] < config.num_states:
        raise RegimeModelError("the observation count must be at least the number of states")
    standardization_mean: list[float] | None = None
    standardization_std: list[float] | None = None
    values = raw
    if config.standardize:
        center = np.mean(raw, axis=0)
        scale = np.std(raw, axis=0, ddof=0)
        scale = np.where(scale > 0.0, scale, 1.0)
        values = (raw - center) / scale
        standardization_mean = center.tolist()
        standardization_std = scale.tolist()

    seeds = spawn_child_seeds(config.seed, config.num_starts)
    fits: list[_StartFit] = []
    summaries: list[HMMStartSummary] = []
    for start_index, seed in enumerate(seeds):
        method = config.initialization if start_index == 0 else HMMInitialization.RANDOM
        initial = _initialize_parameters(
            values,
            config,
            method,
            create_rng(seed),
            standardization_mean,
            standardization_std,
        )
        fit = _fit_start(values, initial, config)
        fits.append(fit)
        summaries.append(
            HMMStartSummary(
                start_index=start_index,
                seed=seed,
                initialization=method,
                converged=fit.converged,
                iterations=fit.iterations,
                log_likelihood=fit.inference.log_likelihood,
                message=fit.message,
            )
        )
    winning = max(fits, key=lambda item: item.inference.log_likelihood)
    parameters, order = _canonicalize(winning.parameters, config.canonical_ordering)
    inference = infer_hmm(values, parameters)
    path, _ = viterbi_decode(values, parameters)
    state_summaries = _state_summaries(raw, parameters, inference.smoothed, path)
    count = hmm_parameter_count(
        parameters.num_states, parameters.feature_dimension, parameters.covariance_type
    )
    log_likelihood = inference.log_likelihood
    probability_series, truncated = _probability_series(
        dataset,
        inference.filtered,
        inference.smoothed,
        config.probability_output_mode,
        config.probability_stride,
        max_probability_points,
    )
    runtime_ms = (time.perf_counter() - started) * 1000.0
    winning_history = winning.likelihood_history
    if not np.array_equal(order, np.arange(config.num_states)):
        winning_history = list(winning_history)
    return HMMFitResult(
        parameters=parameters,
        converged=winning.converged,
        iterations=winning.iterations,
        log_likelihood=log_likelihood,
        likelihood_history=winning_history,
        parameter_count=count,
        aic=2.0 * count - 2.0 * log_likelihood,
        bic=count * math.log(raw.shape[0]) - 2.0 * log_likelihood,
        start_summaries=summaries,
        state_summaries=state_summaries,
        transition_diagnostics=markov_diagnostics(
            MarkovChain(
                transition_matrix=parameters.transition_matrix,
                initial_probabilities=parameters.initial_probabilities,
            )
        ),
        filtered_final_probabilities=inference.filtered[-1].tolist(),
        smoothed_final_probabilities=inference.smoothed[-1].tolist(),
        probability_series=probability_series,
        probability_series_truncated=truncated,
        runtime_ms=runtime_ms,
    )


def _fit_start(
    observations: FloatArray,
    initial: GaussianHMMParameters,
    config: HMMFitConfig,
) -> _StartFit:
    parameters = initial
    inference = infer_hmm(observations, parameters)
    history = [inference.log_likelihood]
    converged = False
    message: str | None = None
    iterations = 0
    for iteration in range(1, config.max_iterations + 1):
        candidate = _maximization(observations, parameters, inference, config)
        candidate_inference = infer_hmm(observations, candidate)
        delta = candidate_inference.log_likelihood - inference.log_likelihood
        allowed_drop = 1e-8 * (1.0 + abs(inference.log_likelihood))
        if delta < -allowed_drop:
            message = "EM update materially decreased log likelihood; previous iterate retained"
            break
        if delta < 0.0:
            converged = True
            message = "EM stopped at numerical likelihood tolerance"
            break
        parameters = candidate
        inference = candidate_inference
        history.append(inference.log_likelihood)
        iterations = iteration
        if delta <= config.tolerance * (1.0 + abs(history[-2])):
            converged = True
            break
    return _StartFit(parameters, inference, converged, iterations, history, message)


def _maximization(
    observations: FloatArray,
    previous: GaussianHMMParameters,
    inference: HMMInferenceArrays,
    config: HMMFitConfig,
) -> GaussianHMMParameters:
    gamma = inference.smoothed
    mass = np.sum(gamma, axis=0)
    pi = np.maximum(gamma[0], config.transition_floor)
    pi /= np.sum(pi)
    transition = np.maximum(inference.expected_transitions, config.transition_floor)
    transition /= np.sum(transition, axis=1, keepdims=True)
    old_means = np.asarray(previous.means, dtype=np.float64)
    old_covariances = np.asarray(previous.covariances, dtype=np.float64)
    means = np.empty_like(old_means)
    covariances = np.empty_like(old_covariances)
    minimum_mass = config.occupancy_floor * observations.shape[0]
    for state in range(previous.num_states):
        if mass[state] <= minimum_mass:
            means[state] = old_means[state]
            covariances[state] = old_covariances[state]
            continue
        weights = gamma[:, state]
        means[state] = weights @ observations / mass[state]
        differences = observations - means[state]
        covariance = np.einsum("t,ti,tj->ij", weights, differences, differences) / mass[state]
        if previous.covariance_type is CovarianceType.DIAGONAL:
            diagonal = np.maximum(np.diag(covariance), config.min_variance)
            covariances[state] = np.diag(diagonal)
        else:
            covariances[state] = _floor_covariance(covariance, config.min_variance)
    return GaussianHMMParameters(
        num_states=previous.num_states,
        feature_dimension=previous.feature_dimension,
        covariance_type=previous.covariance_type,
        initial_probabilities=pi.tolist(),
        transition_matrix=transition.tolist(),
        means=means.tolist(),
        covariances=covariances.tolist(),
        standardization_mean=previous.standardization_mean,
        standardization_std=previous.standardization_std,
    )


def _initialize_parameters(
    observations: FloatArray,
    config: HMMFitConfig,
    method: HMMInitialization,
    rng: Generator,
    standardization_mean: list[float] | None,
    standardization_std: list[float] | None,
) -> GaussianHMMParameters:
    count, dimension = observations.shape
    k = config.num_states
    if method is HMMInitialization.QUANTILE:
        order = np.argsort(observations[:, 0], kind="stable")
        assignments = np.empty(count, dtype=np.int64)
        for state, indices in enumerate(np.array_split(order, k)):
            assignments[indices] = state
    else:
        selected = rng.choice(count, size=k, replace=False)
        centers = observations[selected]
        squared = np.sum((observations[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        assignments = np.argmin(squared, axis=1).astype(np.int64)
    global_covariance = np.atleast_2d(np.cov(observations, rowvar=False, ddof=0))
    global_covariance = _floor_covariance(global_covariance, config.min_variance)
    means = np.empty((k, dimension), dtype=np.float64)
    covariances = np.empty((k, dimension, dimension), dtype=np.float64)
    for state in range(k):
        selected_values = observations[assignments == state]
        if selected_values.shape[0] == 0:
            means[state] = observations[state % count]
            covariance = global_covariance
        else:
            means[state] = np.mean(selected_values, axis=0)
            covariance = (
                np.atleast_2d(np.cov(selected_values, rowvar=False, ddof=0))
                if selected_values.shape[0] > 1
                else global_covariance
            )
        if config.covariance_type is CovarianceType.DIAGONAL:
            covariances[state] = np.diag(np.maximum(np.diag(covariance), config.min_variance))
        else:
            covariances[state] = _floor_covariance(covariance, config.min_variance)
    transition = np.full((k, k), 0.1 / max(1, k - 1), dtype=np.float64)
    np.fill_diagonal(transition, 0.9)
    if k == 2:
        transition[:, :] = [[0.9, 0.1], [0.1, 0.9]]
    transition /= np.sum(transition, axis=1, keepdims=True)
    initial = np.bincount(assignments, minlength=k).astype(np.float64) + 1.0
    initial /= np.sum(initial)
    return GaussianHMMParameters(
        num_states=k,
        feature_dimension=dimension,
        covariance_type=config.covariance_type,
        initial_probabilities=initial.tolist(),
        transition_matrix=transition.tolist(),
        means=means.tolist(),
        covariances=covariances.tolist(),
        standardization_mean=standardization_mean,
        standardization_std=standardization_std,
    )


def _floor_covariance(covariance: FloatArray, floor: float) -> FloatArray:
    symmetric = 0.5 * (covariance + covariance.T)
    values, vectors = np.linalg.eigh(symmetric)
    scale = max(1.0, float(np.max(np.abs(symmetric))))
    if values[0] < -1e-10 * scale:
        raise RegimeModelError("covariance matrix is materially non-positive-semidefinite")
    floored = (vectors * np.maximum(values, floor)) @ vectors.T
    return np.asarray(0.5 * (floored + floored.T), dtype=np.float64)


def _canonicalize(
    parameters: GaussianHMMParameters, ordering: CanonicalOrdering
) -> tuple[GaussianHMMParameters, IntArray]:
    means = np.asarray(parameters.means)
    covariances = np.asarray(parameters.covariances)
    key = (
        means[:, 0]
        if ordering is CanonicalOrdering.FIRST_FEATURE_MEAN
        else np.sqrt(covariances[:, 0, 0])
    )
    order = np.argsort(key, kind="stable").astype(np.int64)
    transition = np.asarray(parameters.transition_matrix)[np.ix_(order, order)]
    initial = np.asarray(parameters.initial_probabilities)[order]
    canonical = GaussianHMMParameters(
        num_states=parameters.num_states,
        feature_dimension=parameters.feature_dimension,
        covariance_type=parameters.covariance_type,
        initial_probabilities=initial.tolist(),
        transition_matrix=transition.tolist(),
        means=means[order].tolist(),
        covariances=covariances[order].tolist(),
        standardization_mean=parameters.standardization_mean,
        standardization_std=parameters.standardization_std,
    )
    return canonical, order


def _state_summaries(
    raw_observations: FloatArray,
    parameters: GaussianHMMParameters,
    gamma: FloatArray,
    path: IntArray,
) -> list[HMMStateSummary]:
    diagnostics = markov_diagnostics(
        MarkovChain(
            transition_matrix=parameters.transition_matrix,
            initial_probabilities=parameters.initial_probabilities,
        )
    )
    mass = np.sum(gamma, axis=0)
    parameter_means = np.asarray(parameters.means, dtype=np.float64)
    parameter_covariances = np.asarray(parameters.covariances, dtype=np.float64)
    if parameters.standardization_mean is not None:
        center = np.asarray(parameters.standardization_mean, dtype=np.float64)
        scale = np.asarray(parameters.standardization_std, dtype=np.float64)
        parameter_means = center[None, :] + parameter_means * scale[None, :]
        parameter_covariances = np.einsum("i,kij,j->kij", scale, parameter_covariances, scale)
    result: list[HMMStateSummary] = []
    for state in range(parameters.num_states):
        weights = gamma[:, state]
        normalized = weights / max(float(mass[state]), np.finfo(float).tiny)
        first = raw_observations[:, 0]
        weighted_mean = float(np.sum(normalized * first))
        centered = first - weighted_mean
        variance = float(np.sum(normalized * centered**2))
        if variance > 0.0:
            skewness = float(np.sum(normalized * centered**3) / variance**1.5)
            excess_kurtosis = float(np.sum(normalized * centered**4) / variance**2 - 3.0)
        else:
            skewness = 0.0
            excess_kurtosis = 0.0
        state_diag = diagnostics.states[state]
        covariance = parameter_covariances[state]
        assigned = first[path == state]
        result.append(
            HMMStateSummary(
                state=state,
                occupancy_probability=float(mass[state] / raw_observations.shape[0]),
                mean=parameter_means[state].tolist(),
                standard_deviation=np.sqrt(np.diag(covariance)).tolist(),
                covariance=covariance.tolist(),
                persistence=state_diag.persistence,
                expected_duration=state_diag.expected_duration,
                stationary_probability=state_diag.stationary_probability,
                posterior_mass=float(mass[state]),
                empirical_observation_count=int(np.count_nonzero(path == state)),
                weighted_skewness=skewness,
                weighted_excess_kurtosis=excess_kurtosis,
                minimum_first_feature=(None if assigned.size == 0 else float(np.min(assigned))),
                maximum_first_feature=(None if assigned.size == 0 else float(np.max(assigned))),
            )
        )
    return result


def hmm_parameter_count(k: int, d: int, covariance_type: CovarianceType) -> int:
    covariance_count = k * d if covariance_type is CovarianceType.DIAGONAL else k * d * (d + 1) // 2
    return (k - 1) + k * (k - 1) + k * d + covariance_count


def _probability_series(
    dataset: HMMObservationDataset,
    filtered: FloatArray,
    smoothed: FloatArray,
    mode: ProbabilityOutputMode,
    stride: int,
    max_points: int | None,
) -> tuple[list[HMMProbabilityPoint], bool]:
    if mode is ProbabilityOutputMode.FINAL_ONLY:
        return [], False
    effective_stride = stride if mode is ProbabilityOutputMode.THINNED else 1
    indices = list(range(0, filtered.shape[0], effective_stride))
    if indices[-1] != filtered.shape[0] - 1:
        indices.append(filtered.shape[0] - 1)
    truncated = max_points is not None and len(indices) > max_points
    if truncated:
        assert max_points is not None
        selected = np.linspace(0, len(indices) - 1, max_points, dtype=np.int64)
        indices = [indices[index] for index in selected]
    return (
        [
            HMMProbabilityPoint(
                observation_index=index,
                timestamp=None if dataset.timestamps is None else dataset.timestamps[index],
                filtered_probabilities=filtered[index].tolist(),
                smoothed_probabilities=smoothed[index].tolist(),
            )
            for index in indices
        ],
        truncated,
    )


def decode_hmm(
    request: HMMDecodeRequest, *, max_probability_points: int | None = None
) -> HMMDecodeResult:
    values = _observations(request.dataset, request.parameters)
    inference = infer_hmm(values, request.parameters)
    path, path_log_probability = viterbi_decode(values, request.parameters)
    mode = (
        ProbabilityOutputMode.THINNED
        if request.include_probabilities
        else ProbabilityOutputMode.FINAL_ONLY
    )
    series, truncated = _probability_series(
        request.dataset,
        inference.filtered,
        inference.smoothed,
        mode,
        request.probability_stride,
        max_probability_points,
    )
    return HMMDecodeResult(
        state_sequence=path.tolist(),
        path_log_probability=path_log_probability,
        log_likelihood=inference.log_likelihood,
        probability_series=series,
        probability_series_truncated=truncated,
    )


def filter_step(
    parameters: GaussianHMMParameters,
    prior_probabilities: list[float],
    observation: list[float],
) -> HMMFilterResult:
    """Predict one transition then assimilate one new observation."""

    prior = np.asarray(prior_probabilities, dtype=np.float64)
    transition = np.asarray(parameters.transition_matrix, dtype=np.float64)
    predicted = prior @ transition
    raw = np.asarray(observation, dtype=np.float64)[None, :]
    if parameters.standardization_mean is not None:
        raw = (raw - np.asarray(parameters.standardization_mean)) / np.asarray(
            parameters.standardization_std
        )
    log_emission = gaussian_log_emissions(raw, parameters)[0]
    with np.errstate(divide="ignore"):
        log_unnormalized = np.log(predicted) + log_emission
    log_normalizer = float(logsumexp(log_unnormalized))
    filtered = np.exp(log_unnormalized - log_normalizer)
    return HMMFilterResult(
        predicted_probabilities=predicted.tolist(),
        filtered_probabilities=filtered.tolist(),
        observation_log_likelihood=log_normalizer,
    )


def forecast_hmm(
    parameters: GaussianHMMParameters,
    current_probabilities: list[float],
    horizons: list[int],
) -> HMMForecastResult:
    chain = MarkovChain(transition_matrix=parameters.transition_matrix)
    current = np.asarray(current_probabilities, dtype=np.float64)
    means = np.asarray(parameters.means, dtype=np.float64)
    covariances = np.asarray(parameters.covariances, dtype=np.float64)
    if parameters.standardization_mean is not None:
        center = np.asarray(parameters.standardization_mean, dtype=np.float64)
        scale = np.asarray(parameters.standardization_std, dtype=np.float64)
        means = center[None, :] + means * scale[None, :]
        covariances = np.einsum("i,kij,j->kij", scale, covariances, scale)
    forecasts: list[HMMMomentForecast] = []
    for horizon in horizons:
        probabilities = current @ n_step_transition(chain, horizon)
        expected = probabilities @ means
        second = np.einsum("k,kij->ij", probabilities, covariances)
        second += np.einsum("k,ki,kj->ij", probabilities, means, means)
        covariance = second - np.outer(expected, expected)
        forecasts.append(
            HMMMomentForecast(
                horizon=horizon,
                state_probabilities=probabilities.tolist(),
                expected_mean=expected.tolist(),
                covariance=(0.5 * (covariance + covariance.T)).tolist(),
            )
        )
    return HMMForecastResult(forecasts=forecasts)


def model_order_comparison(
    request: HMMModelOrderRequest,
) -> HMMModelOrderResult:
    points: list[HMMModelOrderPoint] = []
    child_seeds = spawn_child_seeds(request.config.seed, len(request.state_counts))
    for states, seed in zip(request.state_counts, child_seeds, strict=True):
        config = request.config.model_copy(update={"num_states": states, "seed": seed})
        result = fit_gaussian_hmm(request.dataset, config)
        points.append(
            HMMModelOrderPoint(
                num_states=states,
                log_likelihood=result.log_likelihood,
                aic=result.aic,
                bic=result.bic,
                converged=result.converged,
                runtime_ms=result.runtime_ms,
            )
        )
    return HMMModelOrderResult(points=points)


def walk_forward_hmm(
    dataset: HMMObservationDataset,
    config: HMMWalkForwardConfig,
) -> HMMWalkForwardResult:
    """Run bounded out-of-sample filtering with expanding or rolling refits."""

    total = dataset.observation_count
    start = config.initial_training_observations
    if start >= total:
        raise RegimeModelError("walk-forward training window must leave observations to score")
    parameters: GaussianHMMParameters | None = None
    prior: list[float] | None = None
    points: list[HMMWalkForwardPoint] = []
    refit_count = 0
    total_log_likelihood = 0.0
    for index in range(start, total):
        refitted = parameters is None or (index - start) % config.refit_interval == 0
        if refitted:
            window_start = 0
            if config.window is WalkForwardWindow.ROLLING:
                assert config.rolling_window_observations is not None
                window_start = max(0, index - config.rolling_window_observations)
            training = HMMObservationDataset(
                observations=dataset.observations[window_start:index],
                timestamps=(
                    None if dataset.timestamps is None else dataset.timestamps[window_start:index]
                ),
                feature_names=dataset.feature_names,
            )
            parameters = fit_gaussian_hmm(training, config.fit).parameters
            training_values = _observations(training, parameters)
            prior = forward_log(training_values, parameters)[1][-1].tolist()
            refit_count += 1
        assert parameters is not None and prior is not None
        updated = filter_step(parameters, prior, dataset.observations[index])
        total_log_likelihood += updated.observation_log_likelihood
        prior = updated.filtered_probabilities
        points.append(
            HMMWalkForwardPoint(
                observation_index=index,
                timestamp=None if dataset.timestamps is None else dataset.timestamps[index],
                filtered_probabilities=prior,
                observation_log_likelihood=updated.observation_log_likelihood,
                refitted=refitted,
            )
        )
    return HMMWalkForwardResult(
        points=points,
        total_out_of_sample_log_likelihood=total_log_likelihood,
        refit_count=refit_count,
    )
