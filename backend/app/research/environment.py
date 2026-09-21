"""Research-relevant environment and repository metadata without credentials."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from backend.app import __version__
from backend.app.models.experiments import EnvironmentSnapshot
from backend.app.research.signatures import redact_secrets

PACKAGE_NAMES = ("numpy", "scipy", "pydantic", "fastapi")
SAFE_ENVIRONMENT_NAMES = (
    "PYTHONHASHSEED",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
)


def _git_value(arguments: list[str], repository_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def capture_environment(
    repository_root: Path | None = None,
    *,
    additional_metadata: dict[str, Any] | None = None,
) -> EnvironmentSnapshot:
    root = repository_root or Path.cwd()
    packages: dict[str, str] = {}
    for name in PACKAGE_NAMES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            continue
    variables = {name: os.environ[name] for name in SAFE_ENVIRONMENT_NAMES if name in os.environ}
    if additional_metadata:
        redacted = redact_secrets(additional_metadata)
        variables.update({f"metadata.{key}": str(value) for key, value in redacted.items()})
    commit = _git_value(["rev-parse", "HEAD"], root)
    dirty_output = _git_value(["status", "--porcelain"], root)
    return EnvironmentSnapshot(
        application_version=__version__,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        packages=packages,
        git_commit=commit,
        git_dirty=None if dirty_output is None else bool(dirty_output),
        variables=variables,
    )
