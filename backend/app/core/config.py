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
    max_iv_batch_size: int = Field(default=500, gt=0)
    max_smile_points: int = Field(default=500, gt=0)
    max_surface_smiles: int = Field(default=100, gt=0)
    max_surface_query_points: int = Field(default=1_000, gt=0)
    max_diagnostic_grid_points: int = Field(default=10_000, gt=0)
    max_binomial_steps: int = Field(default=10_000, gt=0)
    max_binomial_convergence_points: int = Field(default=20, gt=0)
    max_lsm_paths: int = Field(default=500_000, gt=0)
    max_lsm_time_steps: int = Field(default=500, gt=0)
    max_lsm_total_path_nodes: int = Field(default=25_000_000, gt=0)
    max_lsm_convergence_points: int = Field(default=20, gt=0)
    max_lsm_diagnostic_dates: int = Field(default=100, gt=0)
    max_asian_monitoring_points: int = Field(default=1_000, gt=0)
    max_asian_total_path_nodes: int = Field(default=25_000_000, gt=0)
    max_exotic_convergence_points: int = Field(default=20, gt=0)
    max_barrier_monitoring_points: int = Field(default=5_000, gt=0)
    max_barrier_total_path_nodes: int = Field(default=25_000_000, gt=0)
    max_barrier_convergence_points: int = Field(default=20, gt=0)
    max_option_chain_contracts: int = Field(default=5_000, gt=0)
    max_quote_age_seconds: float = Field(default=900.0, gt=0.0)
    market_data_provider: Literal["MASSIVE"] = "MASSIVE"
    market_data_api_key: str | None = None
    market_data_base_url: str = "https://api.massive.com"
    market_data_timeout_seconds: float = Field(default=10.0, gt=0.0)
    log_level: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] = "INFO"

    @classmethod
    def from_environment(cls) -> "Settings":
        """Load recognized values from conventional uppercase environment variables."""

        values: dict[str, object] = {}
        for field in (
            "max_lsm_paths",
            "max_lsm_time_steps",
            "max_lsm_total_path_nodes",
            "max_lsm_convergence_points",
            "max_lsm_diagnostic_dates",
            "max_asian_monitoring_points",
            "max_asian_total_path_nodes",
            "max_exotic_convergence_points",
            "max_barrier_monitoring_points",
            "max_barrier_total_path_nodes",
            "max_barrier_convergence_points",
        ):
            if value := os.getenv(field.upper()):
                values[field] = value
        if value := os.getenv("MAX_API_SIMULATIONS"):
            values["max_api_simulations"] = value
        if value := os.getenv("MAX_CONVERGENCE_POINTS"):
            values["max_convergence_points"] = value
        if value := os.getenv("MAX_SENSITIVITY_POINTS"):
            values["max_sensitivity_points"] = value
        if value := os.getenv("MAX_IV_BATCH_SIZE"):
            values["max_iv_batch_size"] = value
        if value := os.getenv("MAX_SMILE_POINTS"):
            values["max_smile_points"] = value
        if value := os.getenv("MAX_SURFACE_SMILES"):
            values["max_surface_smiles"] = value
        if value := os.getenv("MAX_SURFACE_QUERY_POINTS"):
            values["max_surface_query_points"] = value
        if value := os.getenv("MAX_DIAGNOSTIC_GRID_POINTS"):
            values["max_diagnostic_grid_points"] = value
        if value := os.getenv("MAX_BINOMIAL_STEPS"):
            values["max_binomial_steps"] = value
        if value := os.getenv("MAX_BINOMIAL_CONVERGENCE_POINTS"):
            values["max_binomial_convergence_points"] = value
        if value := os.getenv("MAX_OPTION_CHAIN_CONTRACTS"):
            values["max_option_chain_contracts"] = value
        if value := os.getenv("MAX_QUOTE_AGE_SECONDS"):
            values["max_quote_age_seconds"] = value
        if value := os.getenv("MARKET_DATA_PROVIDER"):
            values["market_data_provider"] = value.upper()
        if value := os.getenv("MARKET_DATA_API_KEY"):
            values["market_data_api_key"] = value
        if value := os.getenv("MARKET_DATA_BASE_URL"):
            values["market_data_base_url"] = value
        if value := os.getenv("MARKET_DATA_TIMEOUT_SECONDS"):
            values["market_data_timeout_seconds"] = value
        if value := os.getenv("LOG_LEVEL"):
            values["log_level"] = value.upper()
        return cls.model_validate(values)


@lru_cache
def get_settings() -> Settings:
    """Return one immutable settings snapshot per application process."""

    return Settings.from_environment()
