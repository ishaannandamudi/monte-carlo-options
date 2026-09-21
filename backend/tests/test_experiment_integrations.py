"""Offline Phase 1-20 engine integration coverage for experiment dispatch."""

from pathlib import Path

import numpy as np

from backend.app.calibration.synthetic import (
    FELLER_SATISFIED_PARAMETERS,
    generate_synthetic_heston_dataset,
)
from backend.app.models.experiments import (
    ExperimentDefinition,
    ExperimentRunStatus,
    ExperimentType,
)
from backend.app.models.portfolio import MarketScenario, Portfolio, PortfolioScenarioRequest
from backend.app.models.qmc import QMCConvergenceRequest
from backend.app.models.regime import HMMFitConfig, HMMFitRequest, HMMObservationDataset
from backend.app.models.risk import (
    PortfolioRiskConfig,
    PortfolioRiskRequest,
    RiskMethod,
    RiskValuationMode,
)
from backend.tests.experiment_test_helpers import bsm_definition, experiment_runner
from backend.tests.portfolio_test_helpers import vanilla
from backend.tests.risk_test_helpers import single_portfolio, spot_covariance, spot_mapping


def definition(
    experiment_type: ExperimentType, operation: str, payload: object
) -> ExperimentDefinition:
    return ExperimentDefinition(
        experiment_type=experiment_type,
        operation=operation,
        name=f"{experiment_type.value} integration",
        tags=["integration", "phase21"],
        input_payload=payload.model_dump(mode="json"),  # type: ignore[attr-defined]
    )


def test_heston_analytical_pricing_integration(tmp_path: Path) -> None:
    runner = experiment_runner(tmp_path)
    payload = {
        **bsm_definition().input_payload,
        "parameters": {
            "v0": 0.04,
            "kappa": 2.0,
            "theta": 0.04,
            "sigma_v": 0.3,
            "rho": -0.7,
        },
    }

    run = runner.run(
        ExperimentDefinition(
            experiment_type=ExperimentType.PRICING,
            operation="HESTON_ANALYTICAL",
            name="Heston analytical",
            input_payload=payload,
        )
    )

    assert run.status is ExperimentRunStatus.COMPLETED
    assert {item.name for item in run.metrics} >= {"price", "integration_error"}


def test_heston_calibration_integration(tmp_path: Path) -> None:
    runner = experiment_runner(tmp_path)
    dataset = generate_synthetic_heston_dataset(
        maturities=(0.5, 1.0), log_moneyness=(-0.1, 0.0, 0.1)
    )
    initial = {
        name: getattr(FELLER_SATISFIED_PARAMETERS, name)
        for name in ("v0", "kappa", "theta", "sigma_v", "rho")
    }
    payload = {
        "dataset": dataset.model_dump(mode="json"),
        "config": {
            "target": "PRICE",
            "initial_parameters": initial,
            "max_iterations": 3,
            "max_returned_residuals": 20,
            "identifiability_diagnostics": False,
        },
    }

    run = runner.run(
        ExperimentDefinition(
            experiment_type=ExperimentType.CALIBRATION,
            operation="HESTON",
            name="Heston calibration",
            input_payload=payload,
        )
    )

    assert run.status is ExperimentRunStatus.COMPLETED
    assert next(item.value for item in run.metrics if item.name == "rmse") < 1e-8
    assert run.result_summary is not None
    assert run.result_summary["observations_used"] == 6


def test_model_comparison_and_portfolio_scenario_integrations(tmp_path: Path) -> None:
    runner = experiment_runner(tmp_path)
    model_payload = {
        **bsm_definition().input_payload,
        "config": {
            "selected_models": ["BLACK_SCHOLES", "CRR_BINOMIAL"],
            "reference_model": "BLACK_SCHOLES",
            "include_greeks": False,
        },
        "tree": {"steps": 100},
    }
    comparison = runner.run(
        ExperimentDefinition(
            experiment_type=ExperimentType.MODEL_COMPARISON,
            operation="COMPARE_MODELS",
            name="BSM versus CRR",
            input_payload=model_payload,
        )
    )
    portfolio = Portfolio(portfolio_id="phase21", positions=[vanilla()])
    scenario_request = PortfolioScenarioRequest(
        portfolio=portfolio,
        scenario=MarketScenario(spot_relative_shift=-0.05),
    )
    scenario = runner.run(
        definition(ExperimentType.PORTFOLIO_SCENARIO, "SCENARIO", scenario_request)
    )

    assert comparison.status is ExperimentRunStatus.COMPLETED
    assert next(item.value for item in comparison.metrics if item.name == "successful_models") == 2
    assert scenario.status is ExperimentRunStatus.COMPLETED
    assert any(item.name == "scenario_pnl" for item in scenario.metrics)


def test_var_es_and_hmm_fit_integrations(tmp_path: Path) -> None:
    runner = experiment_runner(tmp_path)
    risk_request = PortfolioRiskRequest(
        portfolio=single_portfolio(),
        factor_mappings=spot_mapping(),
        covariance=spot_covariance(),
        config=PortfolioRiskConfig(
            method=RiskMethod.PARAMETRIC_NORMAL,
            valuation_mode=RiskValuationMode.DELTA,
        ),
    )
    risk_run = runner.run(definition(ExperimentType.VAR_ES, "PORTFOLIO_RISK", risk_request))
    observations = np.concatenate([np.linspace(-1.2, -0.8, 30), np.linspace(0.8, 1.2, 30)])
    hmm_request = HMMFitRequest(
        dataset=HMMObservationDataset(
            observations=[[float(value)] for value in observations],
            feature_names=["return"],
        ),
        config=HMMFitConfig(num_states=2, num_starts=1, max_iterations=20, seed=9),
    )
    hmm_run = runner.run(definition(ExperimentType.REGIME, "HMM_FIT", hmm_request))

    assert risk_run.status is ExperimentRunStatus.COMPLETED
    assert {item.name for item in risk_run.metrics} >= {"var", "expected_shortfall"}
    assert hmm_run.status is ExperimentRunStatus.COMPLETED
    assert {item.name for item in hmm_run.metrics} >= {"log_likelihood", "aic", "bic"}


def test_qmc_convergence_integration_is_reproducible(tmp_path: Path) -> None:
    runner = experiment_runner(tmp_path)
    base = bsm_definition().input_payload
    request = QMCConvergenceRequest(
        contract=base["contract"],
        market=base["market"],
        minimum_power=6,
        maximum_power=7,
        randomized_replicates=2,
        root_seed=31,
    )
    experiment = definition(ExperimentType.QMC, "CONVERGENCE", request).model_copy(
        update={"root_seed": 31}
    )

    first = runner.run(experiment)
    second = runner.run(experiment)

    assert first.status is second.status is ExperimentRunStatus.COMPLETED
    assert first.input_signature == second.input_signature
    first_metrics = {
        item.name: item.value for item in first.metrics if item.category != "performance"
    }
    second_metrics = {
        item.name: item.value for item in second.metrics if item.category != "performance"
    }
    assert first_metrics == second_metrics


def test_existing_mc_crr_heston_and_merton_convergence_wrappers(tmp_path: Path) -> None:
    runner = experiment_runner(tmp_path)
    bsm = bsm_definition().input_payload
    heston_parameters = {
        "v0": 0.04,
        "kappa": 2.0,
        "theta": 0.04,
        "sigma_v": 0.3,
        "rho": -0.7,
    }
    cases = [
        (
            "GBM_MONTE_CARLO",
            {
                **bsm,
                "simulation_counts": [256, 512],
                "root_seed": 5,
                "variance_reduction": "ANTITHETIC",
            },
        ),
        ("CRR", {**bsm, "step_counts": [25, 50]}),
        (
            "HESTON",
            {
                **bsm,
                "parameters": heston_parameters,
                "simulation": {
                    "num_simulations": 128,
                    "num_time_steps": 4,
                    "seed": 5,
                    "scheme": "FULL_TRUNCATION_EULER",
                },
                "values": [64, 128],
                "mode": "PATH_COUNT",
            },
        ),
        (
            "MERTON",
            {
                **bsm,
                "parameters": {
                    "jump_intensity": 0.5,
                    "jump_mean": -0.1,
                    "jump_volatility": 0.2,
                },
                "path_counts": [256, 512],
                "root_seed": 5,
            },
        ),
    ]

    runs = [
        runner.run(
            ExperimentDefinition(
                experiment_type=ExperimentType.CONVERGENCE,
                operation=operation,
                name=f"{operation} convergence",
                input_payload=payload,
            )
        )
        for operation, payload in cases
    ]

    assert all(run.status is ExperimentRunStatus.COMPLETED for run in runs)
    assert all(
        next(item.value for item in run.metrics if item.name == "point_count") == 2 for run in runs
    )
