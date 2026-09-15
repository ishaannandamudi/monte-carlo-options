"""Typed application settings sourced from environment variables."""

import os
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    """HTTP and logging limits; these never alter mathematical library results."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    max_api_simulations: int = Field(default=5_000_000, gt=0)
    max_convergence_points: int = Field(default=20, gt=0)
    max_sensitivity_points: int = Field(default=100, gt=0)
    log_level: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] = "INFO"

    @classmethod
    def from_environment(cls) -> "Settings":
        """Load recognized values from conventional uppercase environment variables."""

        values: dict[str, object] = {}
        if value := os.getenv("MAX_API_SIMULATIONS"):
            values["max_api_simulations"] = value
        if value := os.getenv("MAX_CONVERGENCE_POINTS"):
            values["max_convergence_points"] = value
        if value := os.getenv("MAX_SENSITIVITY_POINTS"):
            values["max_sensitivity_points"] = value
        if value := os.getenv("LOG_LEVEL"):
            values["log_level"] = value.upper()
        return cls.model_validate(values)


@lru_cache
def get_settings() -> Settings:
    """Return one immutable settings snapshot per application process."""

    return Settings.from_environment()
