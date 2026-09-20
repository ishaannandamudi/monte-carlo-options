import numpy as np
import pytest
from pydantic import ValidationError

from backend.app.models.qmc import (
    PowerOfTwoPolicy,
    QMCConfig,
    SamplingDiagnosticsRequest,
    SamplingMethod,
)
from backend.app.simulation.qmc import (
    generate_normal_samples,
    generate_uniform_samples,
    qmc_replicate_seeds,
    sampling_diagnostics,
    uniforms_to_normals,
)


def config(**updates: object) -> QMCConfig:
    values: dict[str, object] = {
        "method": SamplingMethod.SCRAMBLED_SOBOL,
        "dimension": 3,
        "num_samples": 1_024,
        "scramble": True,
        "seed": 9,
        "num_randomizations": 4,
    }
    values.update(updates)
    return QMCConfig.model_validate(values)


def test_default_sampling_method_remains_pseudorandom() -> None:
    default = QMCConfig()
    assert default.method is SamplingMethod.PSEUDORANDOM
    assert default.scramble is False
    assert default.num_randomizations == 1


def test_sobol_reproducibility_shape_bounds_and_scramble_independence() -> None:
    qmc = config()
    first = generate_uniform_samples(qmc, 0)
    second = generate_uniform_samples(qmc, 0)
    other = generate_uniform_samples(qmc, 1)
    assert first.shape == (1_024, 3)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)
    assert np.all((first >= 0.0) & (first < 1.0))
    seeds = qmc_replicate_seeds(qmc)
    assert len(seeds) == len(set(seeds)) == 4


def test_power_of_two_policies_are_explicit() -> None:
    with pytest.raises(ValidationError):
        config(num_samples=1_000)
    rounded = config(
        num_samples=1_000,
        power_of_two_policy=PowerOfTwoPolicy.ROUND_UP,
    )
    assert rounded.num_samples == 1_000
    assert rounded.effective_num_samples == 1_024
    assert generate_uniform_samples(rounded).shape[0] == 1_024
    arbitrary = config(
        num_samples=1_000,
        power_of_two_policy=PowerOfTwoPolicy.ALLOW_ARBITRARY,
    )
    assert arbitrary.effective_num_samples == 1_000
    assert generate_uniform_samples(arbitrary).shape[0] == 1_000


def test_deterministic_sobol_nesting_and_skip_reproducibility() -> None:
    short = config(
        method=SamplingMethod.SOBOL,
        scramble=False,
        num_randomizations=1,
        num_samples=256,
    )
    long = short.model_copy(update={"num_samples": 512})
    np.testing.assert_array_equal(
        generate_uniform_samples(short), generate_uniform_samples(long)[:256]
    )
    skipped = short.model_copy(update={"skip": 17})
    np.testing.assert_array_equal(
        generate_uniform_samples(skipped), generate_uniform_samples(skipped)
    )


def test_inverse_normal_is_finite_and_has_reasonable_moments() -> None:
    endpoints = np.array([[0.0, 1.0, 0.5]])
    assert np.all(np.isfinite(uniforms_to_normals(endpoints)))
    normals = generate_normal_samples(config(dimension=4, num_samples=16_384))
    np.testing.assert_allclose(np.mean(normals, axis=0), 0.0, atol=0.01)
    np.testing.assert_allclose(np.var(normals, axis=0), 1.0, atol=0.02)
    correlation = np.corrcoef(normals, rowvar=False)
    assert np.max(np.abs(correlation - np.eye(4))) < 0.02


def test_sampling_diagnostics_are_bounded_and_reproducible() -> None:
    request = SamplingDiagnosticsRequest(config=config(), include_points=True, max_output_points=5)
    first = sampling_diagnostics(request)
    second = sampling_diagnostics(request)
    assert first.first_points == second.first_points
    assert len(first.first_points) == 5
    assert first.discrepancy is not None
    assert first.normal_variance == pytest.approx(1.0, abs=0.03)

    bridge = sampling_diagnostics(
        SamplingDiagnosticsRequest(
            config=config(
                dimension=4,
                path_construction="BROWNIAN_BRIDGE",
            ),
            observation_times=[0.1, 0.3, 0.65, 1.0],
        )
    )
    assert bridge.path_coordinate_order == [3, 2, 1, 0]
    assert bridge.path_observation_times == [0.1, 0.3, 0.65, 1.0]

    pca = sampling_diagnostics(
        SamplingDiagnosticsRequest(
            config=config(dimension=4, path_construction="PCA"),
            observation_times=[0.1, 0.3, 0.65, 1.0],
        )
    )
    assert pca.pca_cumulative_variance_explained is not None
    assert pca.pca_cumulative_variance_explained[-1] == pytest.approx(1.0)


def test_method_consistency_validation() -> None:
    with pytest.raises(ValidationError):
        QMCConfig(method="SOBOL", scramble=True, num_randomizations=1)
    with pytest.raises(ValidationError):
        QMCConfig(method="SCRAMBLED_SOBOL", scramble=True, num_randomizations=1)
    with pytest.raises(ValidationError):
        QMCConfig(method="PSEUDORANDOM", scramble=False, num_randomizations=2)
    with pytest.raises(ValidationError):
        QMCConfig(method="PSEUDORANDOM", scramble=False, skip=1)
