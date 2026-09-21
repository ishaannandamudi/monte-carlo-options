"""Canonical JSON signatures and secret-safe metadata normalization."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, is_dataclass
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from backend.app.core.exceptions import ExperimentError
from backend.app.models.experiments import ExperimentDefinition

SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
REDACTED = "[REDACTED]"


def _normalized(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _normalized(value.model_dump(mode="python"))
    if isinstance(value, Enum):
        return _normalized(value.value)
    if isinstance(value, datetime):
        stamp = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return stamp.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _normalized(asdict(value))
    if isinstance(value, dict):
        return {
            str(key): _normalized(item)
            for key, item in sorted(value.items(), key=lambda x: str(x[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalized(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_normalized(item) for item in value]
        return sorted(normalized, key=lambda item: canonical_json(item))
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExperimentError("experiment signatures require finite floating-point values")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise ExperimentError(f"unsupported canonical value type: {type(value).__name__}")


def normalize_json(value: Any) -> Any:
    """Return a JSON-compatible structure with deterministic mapping and enum handling."""

    return _normalized(value)


def canonical_json(value: Any) -> str:
    """Serialize without float rounding; Python's shortest round-trip representation is retained."""

    return json.dumps(
        normalize_json(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_signature(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def experiment_signature(definition: ExperimentDefinition) -> str:
    """Hash only execution-relevant fields, excluding display metadata and timestamps."""

    payload = {
        "experiment_type": definition.experiment_type,
        "operation": definition.operation,
        "input_payload": definition.input_payload,
        "config_payload": definition.config_payload,
        "root_seed": definition.root_seed,
        "repetitions": definition.repetitions,
    }
    return sha256_signature(redact_secrets(payload))


def contains_secret_marker(key: str) -> bool:
    upper = key.upper()
    return any(marker in upper for marker in SECRET_MARKERS)


def redact_secrets(value: Any) -> Any:
    """Recursively redact values whose field name contains a denylisted secret marker."""

    normalized = normalize_json(value)

    def redact(item: Any) -> Any:
        if isinstance(item, dict):
            return {
                key: REDACTED if contains_secret_marker(key) else redact(child)
                for key, child in item.items()
            }
        if isinstance(item, list):
            return [redact(child) for child in item]
        return item

    return redact(normalized)
