"""Small deterministic fixtures shared by Phase 21 experiment tests."""

from pathlib import Path

from backend.app.core.config import Settings
from backend.app.models.experiments import ExperimentDefinition, ExperimentType
from backend.app.research.runner import ExperimentRunner
from backend.app.research.store import ExperimentStore


def experiment_settings(tmp_path: Path, **updates: object) -> Settings:
    values: dict[str, object] = {
        "experiment_db_path": str(tmp_path / "experiments.sqlite3"),
        "experiment_artifact_dir": str(tmp_path / "artifacts"),
        "max_api_simulations": 50_000,
        "max_experiment_runs_per_sweep": 20,
        "max_experiment_repetitions": 8,
        "max_benchmark_repetitions": 20,
    }
    values.update(updates)
    return Settings.model_validate(values)


def experiment_runner(tmp_path: Path, **updates: object) -> ExperimentRunner:
    settings = experiment_settings(tmp_path, **updates)
    store = ExperimentStore(
        settings.experiment_db_path,
        settings.experiment_artifact_dir,
        max_artifacts_per_run=settings.max_experiment_artifacts,
        max_artifact_bytes=settings.max_experiment_artifact_bytes,
        max_notes_length=settings.max_experiment_notes_length,
        max_tags_per_run=settings.max_experiment_tags,
    )
    return ExperimentRunner(store, settings)


def bsm_definition(*, strike: float = 100.0, tags: list[str] | None = None) -> ExperimentDefinition:
    return ExperimentDefinition(
        experiment_type=ExperimentType.PRICING,
        operation="BLACK_SCHOLES",
        name="deterministic BSM",
        tags=tags or ["bsm", "phase21"],
        input_payload={
            "contract": {
                "option_type": "CALL",
                "exercise_style": "EUROPEAN",
                "spot": 100.0,
                "strike": strike,
                "maturity_years": 1.0,
            },
            "market": {
                "risk_free_rate": 0.03,
                "volatility": 0.2,
                "dividend_yield": 0.01,
            },
        },
    )


def mc_definition(
    *, seed: int = 71, repetitions: int = 1, simulations: int = 2_048
) -> ExperimentDefinition:
    definition = bsm_definition().model_dump(mode="python")
    definition.update(
        {
            "operation": "GBM_MONTE_CARLO",
            "name": "seeded GBM Monte Carlo",
            "root_seed": seed,
            "repetitions": repetitions,
        }
    )
    definition["input_payload"]["simulation"] = {
        "num_simulations": simulations,
        "seed": seed,
        "variance_reduction": "ANTITHETIC",
        "confidence_level": 0.95,
    }
    return ExperimentDefinition.model_validate(definition)


def noop_definition(*, value: float = 1.0) -> ExperimentDefinition:
    return ExperimentDefinition(
        experiment_type=ExperimentType.BENCHMARK,
        operation="NOOP",
        name="tracking no-op",
        tags=["benchmark"],
        input_payload={"value": value},
    )
