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
    max_multi_asset_count: int = Field(default=100, gt=0)
    max_multi_asset_path_nodes: int = Field(default=25_000_000, gt=0)
    max_basket_convergence_points: int = Field(default=20, gt=0)
    max_basket_sensitivity_points: int = Field(default=50, gt=0)
    max_heston_paths: int = Field(default=500_000, gt=0)
    max_heston_time_steps: int = Field(default=2_000, gt=0)
    max_heston_total_path_nodes: int = Field(default=25_000_000, gt=0)
    max_heston_convergence_points: int = Field(default=20, gt=0)
    max_heston_smile_points: int = Field(default=200, gt=0)
    max_heston_sensitivity_points: int = Field(default=100, gt=0)
    max_merton_paths: int = Field(default=5_000_000, gt=0)
    max_merton_series_terms: int = Field(default=10_000, gt=0)
    max_merton_convergence_points: int = Field(default=20, gt=0)
    max_merton_smile_points: int = Field(default=500, gt=0)
    max_merton_sensitivity_points: int = Field(default=100, gt=0)
    max_merton_path_nodes: int = Field(default=25_000_000, gt=0)
    max_heston_calibration_observations: int = Field(default=250, gt=0)
    max_heston_calibration_starts: int = Field(default=10, gt=0)
    max_heston_calibration_iterations: int = Field(default=500, gt=0)
    max_heston_calibration_surface_points: int = Field(default=500, gt=0)
    max_heston_calibration_diagnostic_points: int = Field(default=500, gt=0)
    max_model_comparison_models: int = Field(default=9, gt=0)
    max_model_comparison_observations: int = Field(default=500, gt=0)
    max_model_comparison_surface_points: int = Field(default=1_000, gt=0)
    max_model_comparison_experiment_points: int = Field(default=20, gt=0)
    max_model_comparison_runtime_repetitions: int = Field(default=10, gt=0)
    max_portfolio_positions: int = Field(default=1_000, gt=0)
    max_portfolio_scenario_points: int = Field(default=1_000, gt=0)
    max_portfolio_grid_points: int = Field(default=2_500, gt=0)
    max_portfolio_stress_scenarios: int = Field(default=100, gt=0)
    max_portfolio_model_risk_models: int = Field(default=9, gt=0)
    max_risk_historical_scenarios: int = Field(default=10_000, gt=0)
    max_risk_monte_carlo_simulations: int = Field(default=1_000_000, gt=0)
    max_risk_positions: int = Field(default=1_000, gt=0)
    max_risk_scenario_position_evaluations: int = Field(default=250_000, gt=0)
    max_risk_contribution_positions: int = Field(default=250, gt=0)
    max_risk_backtest_observations: int = Field(default=10_000, gt=0)
    max_risk_tail_output: int = Field(default=100, gt=0)
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
            "max_multi_asset_count",
            "max_multi_asset_path_nodes",
            "max_basket_convergence_points",
            "max_basket_sensitivity_points",
            "max_heston_paths",
            "max_heston_time_steps",
            "max_heston_total_path_nodes",
            "max_heston_convergence_points",
            "max_heston_smile_points",
            "max_heston_sensitivity_points",
            "max_merton_paths",
            "max_merton_series_terms",
            "max_merton_convergence_points",
            "max_merton_smile_points",
            "max_merton_sensitivity_points",
            "max_merton_path_nodes",
            "max_heston_calibration_observations",
            "max_heston_calibration_starts",
            "max_heston_calibration_iterations",
            "max_heston_calibration_surface_points",
            "max_heston_calibration_diagnostic_points",
            "max_model_comparison_models",
            "max_model_comparison_observations",
            "max_model_comparison_surface_points",
            "max_model_comparison_experiment_points",
            "max_model_comparison_runtime_repetitions",
            "max_portfolio_positions",
            "max_portfolio_scenario_points",
            "max_portfolio_grid_points",
            "max_portfolio_stress_scenarios",
            "max_portfolio_model_risk_models",
            "max_risk_historical_scenarios",
            "max_risk_monte_carlo_simulations",
            "max_risk_positions",
            "max_risk_scenario_position_evaluations",
            "max_risk_contribution_positions",
            "max_risk_backtest_observations",
            "max_risk_tail_output",
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
