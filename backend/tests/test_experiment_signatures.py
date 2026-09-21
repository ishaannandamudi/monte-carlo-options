"""Canonical-signature and secret-redaction tests for Phase 21."""

from enum import StrEnum
from pathlib import Path

import pytest

from backend.app.core.config import Settings
from backend.app.core.exceptions import ExperimentError
from backend.app.models.experiments import ExperimentDefinition, ExperimentType
from backend.app.research.environment import capture_environment
from backend.app.research.signatures import (
    REDACTED,
    canonical_json,
    experiment_signature,
    normalize_json,
    redact_secrets,
)
from backend.tests.experiment_test_helpers import bsm_definition, mc_definition


class SampleEnum(StrEnum):
    VALUE = "VALUE"


def test_canonical_serialization_normalizes_mapping_order_models_and_enums() -> None:
    left = {"z": SampleEnum.VALUE, "a": {"y": 2.0, "x": [1, 2]}}
    right = {"a": {"x": [1, 2], "y": 2.0}, "z": "VALUE"}

    assert canonical_json(left) == canonical_json(right)
    assert normalize_json(bsm_definition())["experiment_type"] == "PRICING"


def test_equivalent_definitions_have_identical_signatures() -> None:
    left = bsm_definition()
    payload = left.model_dump(mode="python")
    payload["input_payload"] = {
        "market": payload["input_payload"]["market"],
        "contract": payload["input_payload"]["contract"],
    }
    right = ExperimentDefinition.model_validate(payload)

    assert experiment_signature(left) == experiment_signature(right)


@pytest.mark.parametrize(
    "changed",
    [
        mc_definition(seed=72),
        bsm_definition(strike=101.0),
        ExperimentDefinition(
            experiment_type=ExperimentType.QMC,
            operation="EUROPEAN",
            name="qmc",
            input_payload={"qmc": {"sampling_method": "SOBOL"}},
        ),
    ],
)
def test_material_config_changes_change_signature(changed: ExperimentDefinition) -> None:
    assert experiment_signature(changed) != experiment_signature(mc_definition())


def test_display_metadata_notes_and_description_do_not_change_signature() -> None:
    original = bsm_definition()
    changed = original.model_copy(
        update={
            "name": "different display name",
            "description": "different description",
            "notes": "different notes",
            "tags": ["other"],
            "metadata": {"display": "different"},
        }
    )

    assert experiment_signature(original) == experiment_signature(changed)


def test_secret_fields_are_recursively_redacted_and_do_not_affect_signature() -> None:
    first = bsm_definition().model_copy(
        update={"config_payload": {"API_KEY": "first", "nested": {"token": "secret"}}}
    )
    second = first.model_copy(
        update={"config_payload": {"API_KEY": "second", "nested": {"token": "changed"}}}
    )

    redacted = redact_secrets(first)
    assert redacted["config_payload"]["API_KEY"] == REDACTED
    assert redacted["config_payload"]["nested"]["token"] == REDACTED
    assert experiment_signature(first) == experiment_signature(second)


def test_environment_snapshot_uses_an_allowlist_and_redacts_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MARKET_DATA_API_KEY", "must-not-be-stored")
    monkeypatch.setenv("PYTHONHASHSEED", "7")

    snapshot = capture_environment(
        repository_root=tmp_path,
        additional_metadata={"provider_token": "also-secret", "researcher": "quant"},
    )

    assert "MARKET_DATA_API_KEY" not in snapshot.variables
    assert snapshot.variables["PYTHONHASHSEED"] == "7"
    assert snapshot.variables["metadata.provider_token"] == REDACTED
    assert snapshot.variables["metadata.researcher"] == "quant"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_values_cannot_enter_canonical_records(value: float) -> None:
    with pytest.raises(ExperimentError, match="finite"):
        canonical_json({"value": value})


def test_experiment_settings_load_primary_and_compatibility_sweep_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAX_EXPERIMENT_SWEEP_RUNS", "7")
    monkeypatch.setenv("MAX_EXPERIMENT_REPETITIONS", "4")
    compatibility = Settings.from_environment()
    monkeypatch.setenv("MAX_EXPERIMENT_RUNS_PER_SWEEP", "9")
    primary = Settings.from_environment()

    assert compatibility.max_experiment_runs_per_sweep == 7
    assert compatibility.max_experiment_repetitions == 4
    assert primary.max_experiment_runs_per_sweep == 9
