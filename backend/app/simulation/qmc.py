"""Sobol sampling, inverse-normal transforms, and Brownian path constructions."""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtri
from scipy.stats import qmc as scipy_qmc

from backend.app.core.exceptions import QMCError
from backend.app.models.qmc import (
    NumericalMethodMetadata,
    QMCConfig,
    QMCPathConstruction,
    SamplingDiagnosticsRequest,
    SamplingDiagnosticsResult,
    SamplingMethod,
)
from backend.app.simulation.random import create_rng, spawn_child_seeds

FloatArray = NDArray[np.float64]
NORMAL_CLIP_EPSILON = np.finfo(np.float64).eps
MAX_DISCREPANCY_SAMPLES = 4_096
MAX_DISCREPANCY_DIMENSION = 32
MAX_DISCREPANCY_COORDINATES = 65_536


@dataclass(frozen=True)
class PCAPathDiagnostics:
    eigenvalues: FloatArray
    cumulative_variance_explained: FloatArray
    factor: FloatArray


def qmc_replicate_seeds(config: QMCConfig) -> list[int]:
    """Return deterministic independent scramble seeds or the single configured seed."""

    if config.method is SamplingMethod.SCRAMBLED_SOBOL:
        return spawn_child_seeds(config.seed, config.num_randomizations)
    return [config.seed]


def generate_uniform_samples(config: QMCConfig, replicate_index: int = 0) -> FloatArray:
    """Generate one complete pseudorandom or Sobol sample matrix."""

    seeds = qmc_replicate_seeds(config)
    if replicate_index < 0 or replicate_index >= len(seeds):
        raise QMCError("replicate index is outside the configured range")
    count = config.effective_num_samples
    if config.method is SamplingMethod.PSEUDORANDOM:
        return np.asarray(
            create_rng(seeds[replicate_index]).random((count, config.dimension)),
            dtype=np.float64,
        )
    try:
        engine = scipy_qmc.Sobol(
            d=config.dimension,
            scramble=config.method is SamplingMethod.SCRAMBLED_SOBOL,
            seed=(
                seeds[replicate_index] if config.method is SamplingMethod.SCRAMBLED_SOBOL else None
            ),
        )
        if config.skip:
            engine.fast_forward(config.skip)
        if config.is_power_of_two and config.skip == 0:
            exponent = count.bit_length() - 1
            points = engine.random_base2(exponent)
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="The balance properties of Sobol")
                points = engine.random(count)
    except ValueError as exc:
        raise QMCError(f"Sobol generation failed: {exc}") from exc
    result = np.asarray(points, dtype=np.float64)
    if result.shape != (count, config.dimension):
        raise QMCError("Sobol generator returned an unexpected sample shape")
    if not np.all(np.isfinite(result)) or np.any(result < 0.0) or np.any(result >= 1.0):
        raise QMCError("uniform sample contains values outside [0, 1)")
    return result


def uniforms_to_normals(uniforms: FloatArray) -> FloatArray:
    """Map uniform coordinates through ndtri after finite endpoint clipping."""

    values = np.asarray(uniforms, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise QMCError("uniform samples must be a non-empty matrix")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise QMCError("uniform samples must be finite and lie in [0, 1]")
    clipped = np.clip(values, NORMAL_CLIP_EPSILON, 1.0 - NORMAL_CLIP_EPSILON)
    normals = np.asarray(ndtri(clipped), dtype=np.float64)
    if not np.all(np.isfinite(normals)):
        raise QMCError("inverse-normal transformation produced non-finite values")
    return normals


def generate_normal_samples(config: QMCConfig, replicate_index: int = 0) -> FloatArray:
    return uniforms_to_normals(generate_uniform_samples(config, replicate_index))


def _validated_times(times: FloatArray) -> FloatArray:
    values = np.asarray(times, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise QMCError("Brownian observation times must be a non-empty finite vector")
    if np.any(values <= 0.0) or np.any(np.diff(values) <= 0.0):
        raise QMCError("Brownian observation times must be strictly increasing and positive")
    return values


def brownian_bridge_order(times: FloatArray) -> list[int]:
    """Return the deterministic terminal-first recursive coordinate assignment."""

    values = _validated_times(times)
    last = values.size - 1
    order = [last]
    intervals: list[tuple[int, int]] = [(-1, last)]
    while intervals:
        left, right = intervals.pop(0)
        candidates = list(range(left + 1, right))
        if not candidates:
            continue
        left_time = 0.0 if left < 0 else float(values[left])
        midpoint = 0.5 * (left_time + float(values[right]))
        selected = min(candidates, key=lambda index: (abs(float(values[index]) - midpoint), index))
        order.append(selected)
        intervals.append((left, selected))
        intervals.append((selected, right))
    if len(order) != values.size:
        raise QMCError("Brownian bridge ordering did not cover every time")
    return order


def brownian_bridge_values(normals: FloatArray, times: FloatArray) -> FloatArray:
    """Construct Brownian values with terminal value assigned to coordinate zero."""

    values = _validated_times(times)
    drivers = np.asarray(normals, dtype=np.float64)
    if drivers.ndim != 2 or drivers.shape[1] != values.size or drivers.shape[0] == 0:
        raise QMCError("bridge normals must have shape (samples, observation_times)")
    if not np.all(np.isfinite(drivers)):
        raise QMCError("bridge normals must be finite")
    order = brownian_bridge_order(values)
    paths = np.empty_like(drivers)
    last = order[0]
    paths[:, last] = np.sqrt(values[last]) * drivers[:, 0]
    assigned = {last}
    for coordinate, target in enumerate(order[1:], start=1):
        left_candidates = [index for index in assigned if index < target]
        right_candidates = [index for index in assigned if index > target]
        left = max(left_candidates) if left_candidates else -1
        right = min(right_candidates)
        left_time = 0.0 if left < 0 else float(values[left])
        right_time = float(values[right])
        target_time = float(values[target])
        left_value: FloatArray | float = 0.0 if left < 0 else paths[:, left]
        mean = (target_time - left_time) / (right_time - left_time) * paths[:, right] + (
            right_time - target_time
        ) / (right_time - left_time) * left_value
        variance = (target_time - left_time) * (right_time - target_time) / (right_time - left_time)
        paths[:, target] = mean + np.sqrt(variance) * drivers[:, coordinate]
        assigned.add(target)
    return np.asarray(paths, dtype=np.float64)


def pca_path_diagnostics(times: FloatArray) -> PCAPathDiagnostics:
    values = _validated_times(times)
    covariance = np.minimum(values[:, None], values[None, :])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    signs = np.where(eigenvectors[0] < 0.0, -1.0, 1.0)
    eigenvectors *= signs[None, :]
    total = float(np.sum(eigenvalues))
    cumulative = np.cumsum(eigenvalues) / total
    factor = eigenvectors * np.sqrt(eigenvalues)[None, :]
    return PCAPathDiagnostics(eigenvalues, cumulative, np.asarray(factor, dtype=np.float64))


def pca_brownian_values(normals: FloatArray, times: FloatArray) -> FloatArray:
    drivers = np.asarray(normals, dtype=np.float64)
    diagnostics = pca_path_diagnostics(times)
    if drivers.ndim != 2 or drivers.shape[1] != diagnostics.factor.shape[1]:
        raise QMCError("PCA normals must match the Brownian time dimension")
    with np.errstate(all="ignore"):
        result = np.asarray(drivers @ diagnostics.factor.T, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise QMCError("PCA Brownian construction produced non-finite values")
    return result


def brownian_increments_from_normals(
    normals: FloatArray,
    times: FloatArray,
    construction: QMCPathConstruction,
) -> FloatArray:
    """Convert independent normals into standard Brownian increments."""

    values = _validated_times(times)
    drivers = np.asarray(normals, dtype=np.float64)
    if drivers.ndim != 2 or drivers.shape[1] != values.size:
        raise QMCError("normal columns must match Brownian observation times")
    deltas = np.diff(np.concatenate((np.zeros(1), values)))
    if construction is QMCPathConstruction.STANDARD:
        return np.asarray(drivers * np.sqrt(deltas)[None, :], dtype=np.float64)
    brownian = (
        brownian_bridge_values(drivers, values)
        if construction is QMCPathConstruction.BROWNIAN_BRIDGE
        else pca_brownian_values(drivers, values)
    )
    return np.diff(np.column_stack((np.zeros(drivers.shape[0]), brownian)), axis=1)


def normalized_increment_drivers(
    normals: FloatArray,
    times: FloatArray,
    construction: QMCPathConstruction,
) -> FloatArray:
    values = _validated_times(times)
    deltas = np.diff(np.concatenate((np.zeros(1), values)))
    increments = brownian_increments_from_normals(normals, values, construction)
    return np.asarray(increments / np.sqrt(deltas)[None, :], dtype=np.float64)


def numerical_method_metadata(
    config: QMCConfig,
    financial_model: str,
    signature_input: object | None = None,
    observation_times: FloatArray | None = None,
) -> NumericalMethodMetadata:
    if hasattr(signature_input, "model_dump"):
        signature_input = signature_input.model_dump(mode="json")
    payload = {
        "financial_model": financial_model,
        "qmc": config.model_dump(mode="json"),
        "input": signature_input,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    signature = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    path_times: list[float] | None = None
    path_order: list[int] | None = None
    pca_concentration: list[float] | None = None
    if observation_times is not None:
        times = _validated_times(observation_times)
        if times.size != config.dimension:
            raise QMCError("path observation times must match the configured dimension")
        path_times = times.tolist()
        path_order = (
            brownian_bridge_order(times)
            if config.path_construction is QMCPathConstruction.BROWNIAN_BRIDGE
            else list(range(config.dimension))
        )
        if config.path_construction is QMCPathConstruction.PCA:
            pca_concentration = pca_path_diagnostics(times).cumulative_variance_explained.tolist()
    return NumericalMethodMetadata(
        financial_model=financial_model,
        sampling_method=config.method,
        scramble=config.scramble,
        seed=config.seed,
        dimension=config.dimension,
        requested_samples=config.num_samples,
        effective_samples=config.effective_num_samples,
        replicate_count=config.num_randomizations,
        path_construction=config.path_construction,
        path_observation_times=path_times,
        path_coordinate_order=path_order,
        pca_cumulative_variance_explained=pca_concentration,
        skip=config.skip,
        reproducibility_signature=signature,
    )


def sampling_diagnostics(request: SamplingDiagnosticsRequest) -> SamplingDiagnosticsResult:
    started = perf_counter()
    points = generate_uniform_samples(request.config)
    normals = uniforms_to_normals(points)
    discrepancy = None
    if (
        request.calculate_discrepancy
        and points.shape[0] <= MAX_DISCREPANCY_SAMPLES
        and points.shape[1] <= MAX_DISCREPANCY_DIMENSION
        and points.size <= MAX_DISCREPANCY_COORDINATES
    ):
        discrepancy = float(scipy_qmc.discrepancy(points))
    output_count = min(request.max_output_points, points.shape[0]) if request.include_points else 0
    times = (
        np.asarray(request.observation_times, dtype=np.float64)
        if request.observation_times is not None
        else np.linspace(
            1.0 / request.config.dimension,
            1.0,
            request.config.dimension,
            dtype=np.float64,
        )
    )
    order = (
        brownian_bridge_order(times)
        if request.config.path_construction is QMCPathConstruction.BROWNIAN_BRIDGE
        else list(range(request.config.dimension))
    )
    pca_concentration = (
        pca_path_diagnostics(times).cumulative_variance_explained.tolist()
        if request.config.path_construction is QMCPathConstruction.PCA
        else None
    )
    return SamplingDiagnosticsResult(
        method=request.config.method,
        dimension=request.config.dimension,
        requested_samples=request.config.num_samples,
        effective_samples=request.config.effective_num_samples,
        power_of_two=request.config.is_power_of_two,
        scramble=request.config.scramble,
        seed=request.config.seed,
        replicate_count=request.config.num_randomizations,
        replicate_seeds=qmc_replicate_seeds(request.config),
        skip=request.config.skip,
        path_construction=request.config.path_construction,
        path_observation_times=times.tolist(),
        path_coordinate_order=order,
        pca_cumulative_variance_explained=pca_concentration,
        minimum_uniform=float(np.min(points)),
        maximum_uniform=float(np.max(points)),
        mean_uniform=float(np.mean(points)),
        discrepancy=discrepancy,
        first_points=points[:output_count].tolist(),
        normal_mean=float(np.mean(normals)),
        normal_variance=float(np.var(normals)),
        runtime_ms=(perf_counter() - started) * 1_000.0,
    )
