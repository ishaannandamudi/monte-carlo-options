"""Synchronous bounded experiment dispatch over existing Phase 1-20 services."""

from __future__ import annotations

import itertools
import math
import statistics
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from pydantic import BaseModel, ValidationError

from backend.app.analytics.binomial import run_binomial_convergence
from backend.app.analytics.convergence import run_convergence
from backend.app.analytics.heston import run_heston_convergence
from backend.app.analytics.merton import run_merton_convergence
from backend.app.analytics.model_comparison import compare_models
from backend.app.analytics.qmc import compare_qmc_methods, qmc_convergence
from backend.app.api.analytics import ConvergenceRequest
from backend.app.api.binomial import BinomialConvergenceRequest
from backend.app.api.heston import HestonAnalyticalRequest, HestonConvergenceRequest
from backend.app.api.heston_calibration import HestonCalibrationRequest
from backend.app.api.merton import MertonConvergenceRequest
from backend.app.calibration.heston import calibrate_heston
from backend.app.core.config import Settings
from backend.app.core.exceptions import ExperimentError
from backend.app.models.experiments import (
    ExperimentBenchmarkRequest,
    ExperimentDefinition,
    ExperimentFailure,
    ExperimentMetric,
    ExperimentRun,
    ExperimentRunStatus,
    ExperimentSweepRequest,
    ExperimentSweepResult,
    ExperimentType,
    MetricAggregate,
    ReproduceRequest,
    ReproducibilityResult,
    ResearchFlag,
    SweepAxis,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.model_comparison import ModelComparisonRequest
from backend.app.models.option import OptionContract
from backend.app.models.portfolio import PortfolioScenarioRequest
from backend.app.models.qmc import (
    QMCAsianRequest,
    QMCBasketRequest,
    QMCComparisonRequest,
    QMCConfig,
    QMCConvergenceRequest,
    QMCEuropeanRequest,
)
from backend.app.models.regime import HMMFitRequest
from backend.app.models.risk import PortfolioRiskRequest
from backend.app.models.simulation import SimulationConfig
from backend.app.portfolio.engine import run_scenario
from backend.app.pricing.black_scholes import black_scholes_price
from backend.app.pricing.heston_analytical import heston_analytical_price
from backend.app.pricing.monte_carlo import MonteCarloPricer
from backend.app.pricing.qmc import price_asian_qmc, price_basket_qmc, price_european_qmc
from backend.app.regime.hmm import fit_gaussian_hmm
from backend.app.research.compare import metric_differences, reproducible_metrics
from backend.app.research.environment import capture_environment
from backend.app.research.signatures import canonical_json, experiment_signature, normalize_json
from backend.app.research.store import ExperimentStore
from backend.app.risk.engine import calculate_portfolio_risk
from backend.app.simulation.random import spawn_child_seeds


@dataclass(frozen=True)
class DispatchOutput:
    result: Any
    metrics: list[ExperimentMetric]


def _metric(
    name: str,
    value: float | int | None,
    *,
    unit: str | None = None,
    category: str | None = None,
) -> ExperimentMetric | None:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ExperimentError(f"non-finite experiment metric: {name}")
    return ExperimentMetric(name=name, value=numeric, unit=unit, category=category)


def _metrics(*values: ExperimentMetric | None) -> list[ExperimentMetric]:
    return [value for value in values if value is not None]


def _json_result(result: Any) -> Any:
    if isinstance(result, BaseModel):
        return result.model_dump(mode="json")
    return normalize_json(result)


def _inject_seed(value: Any, seed: int) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                seed
                if key.lower() in {"seed", "root_seed", "deterministic_seed"}
                else _inject_seed(child, seed)
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_inject_seed(child, seed) for child in value]
    return value


def _with_seed(definition: ExperimentDefinition, seed: int | None) -> ExperimentDefinition:
    if seed is None:
        return definition
    return definition.model_copy(
        update={
            "root_seed": seed,
            "input_payload": _inject_seed(definition.input_payload, seed),
            "config_payload": _inject_seed(definition.config_payload, seed),
        }
    )


def _operation(definition: ExperimentDefinition, expected_type: ExperimentType) -> str:
    if definition.experiment_type is not expected_type:
        raise ExperimentError(
            f"operation {definition.operation} is incompatible with {definition.experiment_type}"
        )
    return definition.operation


class ExperimentDispatcher:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def dispatch(self, definition: ExperimentDefinition) -> DispatchOutput:
        if definition.operation == "FAIL":
            raise ExperimentError("controlled experiment failure")
        handlers = {
            ExperimentType.PRICING: self._pricing,
            ExperimentType.CONVERGENCE: self._convergence,
            ExperimentType.MODEL_COMPARISON: self._model_comparison,
            ExperimentType.CALIBRATION: self._calibration,
            ExperimentType.PORTFOLIO_SCENARIO: self._portfolio,
            ExperimentType.VAR_ES: self._risk,
            ExperimentType.REGIME: self._regime,
            ExperimentType.QMC: self._qmc,
            ExperimentType.BENCHMARK: self._noop_benchmark,
        }
        return handlers[definition.experiment_type](definition)

    def _pricing(self, definition: ExperimentDefinition) -> DispatchOutput:
        operation = _operation(definition, ExperimentType.PRICING)
        payload = definition.input_payload
        if operation == "BLACK_SCHOLES":
            contract = OptionContract.model_validate(payload["contract"])
            market = MarketEnvironment.model_validate(payload["market"])
            price = black_scholes_price(contract, market)
            return DispatchOutput(
                {"price": price, "model": "BLACK_SCHOLES"},
                _metrics(_metric("price", price, category="pricing")),
            )
        if operation == "GBM_MONTE_CARLO":
            contract = OptionContract.model_validate(payload["contract"])
            market = MarketEnvironment.model_validate(payload["market"])
            simulation = SimulationConfig.model_validate(payload.get("simulation", {}))
            if simulation.num_simulations > self.settings.max_api_simulations:
                raise ExperimentError("GBM experiment simulation count exceeds configured limit")
            mc_result = MonteCarloPricer().price(contract, market, simulation)
            return DispatchOutput(
                mc_result,
                _metrics(
                    _metric("price", mc_result.price, category="pricing"),
                    _metric("standard_error", mc_result.standard_error, category="uncertainty"),
                    _metric(
                        "model_runtime_ms",
                        mc_result.runtime_ms,
                        unit="ms",
                        category="performance",
                    ),
                ),
            )
        if operation == "HESTON_ANALYTICAL":
            heston_request = HestonAnalyticalRequest.model_validate(payload)
            heston_result = heston_analytical_price(
                heston_request.contract,
                heston_request.market,
                heston_request.parameters,
                heston_request.integration,
            )
            return DispatchOutput(
                heston_result,
                _metrics(
                    _metric("price", heston_result.price, category="pricing"),
                    _metric("integration_error", heston_result.integration_error, category="error"),
                    _metric(
                        "model_runtime_ms",
                        heston_result.runtime_ms,
                        unit="ms",
                        category="performance",
                    ),
                ),
            )
        raise ExperimentError(f"unsupported pricing experiment operation: {operation}")

    def _convergence(self, definition: ExperimentDefinition) -> DispatchOutput:
        operation = _operation(definition, ExperimentType.CONVERGENCE)
        payload = definition.input_payload
        if operation == "GBM_MONTE_CARLO":
            mc_request = ConvergenceRequest.model_validate(payload)
            if len(mc_request.simulation_counts) > self.settings.max_convergence_points:
                raise ExperimentError("convergence point count exceeds configured limit")
            if max(mc_request.simulation_counts) > self.settings.max_api_simulations:
                raise ExperimentError("convergence simulation count exceeds configured limit")
            mc_result = run_convergence(
                mc_request.contract,
                mc_request.market,
                mc_request.simulation_counts,
                root_seed=mc_request.root_seed,
                variance_reduction=mc_request.variance_reduction,
                confidence_level=mc_request.confidence_level,
            )
            return DispatchOutput(
                mc_result,
                _metrics(
                    _metric(
                        "convergence_slope",
                        mc_result.estimated_standard_error_slope,
                        category="convergence",
                    ),
                    _metric("point_count", len(mc_result.points), category="workload"),
                ),
            )
        if operation == "CRR":
            crr_request = BinomialConvergenceRequest.model_validate(payload)
            if len(crr_request.step_counts) > self.settings.max_binomial_convergence_points:
                raise ExperimentError("CRR convergence point count exceeds configured limit")
            if max(crr_request.step_counts) > self.settings.max_binomial_steps:
                raise ExperimentError("CRR step count exceeds configured limit")
            crr_result = run_binomial_convergence(
                crr_request.contract,
                crr_request.market,
                crr_request.step_counts,
                model=crr_request.model,
                probability_tolerance=crr_request.probability_tolerance,
                arbitrage_tolerance=crr_request.arbitrage_tolerance,
                exercise_tolerance=crr_request.exercise_tolerance,
            )
            return DispatchOutput(
                crr_result,
                _metrics(_metric("point_count", len(crr_result.points), category="workload")),
            )
        if operation == "HESTON":
            heston_request = HestonConvergenceRequest.model_validate(payload)
            if len(heston_request.values) > self.settings.max_heston_convergence_points:
                raise ExperimentError("Heston convergence point count exceeds configured limit")
            self._validate_heston_config(heston_request.simulation)
            heston_result = run_heston_convergence(
                heston_request.contract,
                heston_request.market,
                heston_request.parameters,
                heston_request.simulation,
                heston_request.values,
                heston_request.mode,
                heston_request.integration,
            )
            return DispatchOutput(
                heston_result,
                _metrics(_metric("point_count", len(heston_result.points), category="workload")),
            )
        if operation == "MERTON":
            merton_request = MertonConvergenceRequest.model_validate(payload)
            if len(merton_request.path_counts) > self.settings.max_merton_convergence_points:
                raise ExperimentError("Merton convergence point count exceeds configured limit")
            if max(merton_request.path_counts) > self.settings.max_merton_paths:
                raise ExperimentError("Merton path count exceeds configured limit")
            merton_result = run_merton_convergence(
                merton_request.contract,
                merton_request.market,
                merton_request.parameters,
                merton_request.path_counts,
                merton_request.root_seed,
                merton_request.confidence_level,
                merton_request.series,
            )
            return DispatchOutput(
                merton_result,
                _metrics(_metric("point_count", len(merton_result.points), category="workload")),
            )
        if operation == "QMC":
            return self._qmc_convergence(payload)
        raise ExperimentError(f"unsupported convergence experiment operation: {operation}")

    def _model_comparison(self, definition: ExperimentDefinition) -> DispatchOutput:
        _operation(definition, ExperimentType.MODEL_COMPARISON)
        if definition.operation != "COMPARE_MODELS":
            raise ExperimentError(f"unsupported model-comparison operation: {definition.operation}")
        request = ModelComparisonRequest.model_validate(definition.input_payload)
        if len(request.config.selected_models) > self.settings.max_model_comparison_models:
            raise ExperimentError("model comparison count exceeds configured limit")
        result = compare_models(request)
        return DispatchOutput(
            result,
            _metrics(
                _metric("successful_models", len(result.successful_models), category="models"),
                _metric("failed_models", len(result.failed_models), category="models"),
                _metric("price_range", result.price_dispersion.range, category="dispersion"),
                _metric(
                    "relative_price_range",
                    result.price_dispersion.relative_range,
                    category="dispersion",
                ),
                _metric(
                    "model_runtime_ms",
                    result.experiment.total_runtime_ms,
                    unit="ms",
                    category="performance",
                ),
            ),
        )

    def _calibration(self, definition: ExperimentDefinition) -> DispatchOutput:
        _operation(definition, ExperimentType.CALIBRATION)
        if definition.operation != "HESTON":
            raise ExperimentError(f"unsupported calibration operation: {definition.operation}")
        request = HestonCalibrationRequest.model_validate(definition.input_payload)
        if len(request.dataset.observations) > self.settings.max_heston_calibration_observations:
            raise ExperimentError("calibration observation count exceeds configured limit")
        if request.config.multi_start_count > self.settings.max_heston_calibration_starts:
            raise ExperimentError("calibration start count exceeds configured limit")
        if request.config.max_iterations > self.settings.max_heston_calibration_iterations:
            raise ExperimentError("calibration iteration count exceeds configured limit")
        result = calibrate_heston(request.dataset, request.config)
        return DispatchOutput(
            result,
            _metrics(
                _metric("objective", result.objective_value, category="calibration"),
                _metric("rmse", result.rmse, category="error"),
                _metric("weighted_rmse", result.weighted_rmse, category="error"),
                _metric("model_runtime_ms", result.runtime_ms, unit="ms", category="performance"),
            ),
        )

    def _portfolio(self, definition: ExperimentDefinition) -> DispatchOutput:
        _operation(definition, ExperimentType.PORTFOLIO_SCENARIO)
        if definition.operation != "SCENARIO":
            raise ExperimentError(f"unsupported portfolio operation: {definition.operation}")
        request = PortfolioScenarioRequest.model_validate(definition.input_payload)
        if len(request.portfolio.positions) > self.settings.max_portfolio_positions:
            raise ExperimentError("portfolio position count exceeds configured limit")
        result = run_scenario(request)
        return DispatchOutput(
            result,
            _metrics(
                _metric("portfolio_value", result.scenario_value, category="portfolio"),
                _metric("scenario_pnl", result.full_repricing_pnl, category="portfolio"),
                _metric("approximation_error", result.approximation_residual, category="error"),
                _metric("model_runtime_ms", result.runtime_ms, unit="ms", category="performance"),
            ),
        )

    def _risk(self, definition: ExperimentDefinition) -> DispatchOutput:
        _operation(definition, ExperimentType.VAR_ES)
        if definition.operation != "PORTFOLIO_RISK":
            raise ExperimentError(f"unsupported risk operation: {definition.operation}")
        request = PortfolioRiskRequest.model_validate(definition.input_payload)
        if len(request.portfolio.positions) > self.settings.max_risk_positions:
            raise ExperimentError("risk position count exceeds configured limit")
        if request.config.simulations > self.settings.max_risk_monte_carlo_simulations:
            raise ExperimentError("risk simulation count exceeds configured limit")
        result = calculate_portfolio_risk(request)
        return DispatchOutput(
            result,
            _metrics(
                _metric("var", result.value_at_risk, category="risk"),
                _metric("expected_shortfall", result.expected_shortfall, category="risk"),
                _metric("confidence_level", result.confidence_level, category="risk"),
                _metric("scenario_count", result.scenario_count, category="workload"),
                _metric("model_runtime_ms", result.runtime_ms, unit="ms", category="performance"),
            ),
        )

    def _regime(self, definition: ExperimentDefinition) -> DispatchOutput:
        _operation(definition, ExperimentType.REGIME)
        if definition.operation != "HMM_FIT":
            raise ExperimentError(f"unsupported regime operation: {definition.operation}")
        request = HMMFitRequest.model_validate(definition.input_payload)
        if request.dataset.observation_count > self.settings.max_hmm_observations:
            raise ExperimentError("HMM observation count exceeds configured limit")
        if request.dataset.feature_dimension > self.settings.max_hmm_features:
            raise ExperimentError("HMM feature count exceeds configured limit")
        if request.config.num_states > self.settings.max_hmm_states:
            raise ExperimentError("HMM state count exceeds configured limit")
        if request.config.max_iterations > self.settings.max_hmm_iterations:
            raise ExperimentError("HMM iteration count exceeds configured limit")
        if request.config.num_starts > self.settings.max_hmm_starts:
            raise ExperimentError("HMM start count exceeds configured limit")
        result = fit_gaussian_hmm(
            request.dataset,
            request.config,
            max_probability_points=self.settings.max_hmm_probability_output_points,
        )
        return DispatchOutput(
            result,
            _metrics(
                _metric("log_likelihood", result.log_likelihood, category="fit"),
                _metric("aic", result.aic, category="fit"),
                _metric("bic", result.bic, category="fit"),
                _metric("state_count", result.parameters.num_states, category="model"),
                _metric("model_runtime_ms", result.runtime_ms, unit="ms", category="performance"),
            ),
        )

    def _qmc(self, definition: ExperimentDefinition) -> DispatchOutput:
        _operation(definition, ExperimentType.QMC)
        operation = definition.operation
        if operation == "EUROPEAN":
            european_request = QMCEuropeanRequest.model_validate(definition.input_payload)
            self._validate_qmc(european_request.qmc)
            european_result = price_european_qmc(european_request)
            return self._qmc_price_output(european_result)
        if operation == "ASIAN":
            asian_request = QMCAsianRequest.model_validate(definition.input_payload)
            self._validate_qmc(asian_request.qmc)
            asian_result = price_asian_qmc(asian_request)
            return self._qmc_price_output(asian_result)
        if operation == "BASKET":
            basket_request = QMCBasketRequest.model_validate(definition.input_payload)
            self._validate_qmc(basket_request.qmc)
            basket_result = price_basket_qmc(basket_request)
            return self._qmc_price_output(basket_result)
        if operation == "CONVERGENCE":
            return self._qmc_convergence(definition.input_payload)
        if operation == "COMPARE":
            comparison_request = QMCComparisonRequest.model_validate(definition.input_payload)
            if comparison_request.randomized_replicates > self.settings.max_qmc_replicates:
                raise ExperimentError("QMC replicate count exceeds configured limit")
            comparison_result = compare_qmc_methods(comparison_request)
            return DispatchOutput(
                comparison_result,
                _metrics(
                    _metric("point_count", len(comparison_result.points), category="workload")
                ),
            )
        raise ExperimentError(f"unsupported QMC experiment operation: {operation}")

    def _qmc_convergence(self, payload: dict[str, Any]) -> DispatchOutput:
        request = QMCConvergenceRequest.model_validate(payload)
        count = request.maximum_power - request.minimum_power + 1
        if count > self.settings.max_qmc_convergence_points:
            raise ExperimentError("QMC convergence point count exceeds configured limit")
        if 1 << request.maximum_power > self.settings.max_qmc_samples:
            raise ExperimentError("QMC convergence sample count exceeds configured limit")
        if request.randomized_replicates > self.settings.max_qmc_replicates:
            raise ExperimentError("QMC convergence replicate count exceeds configured limit")
        result = qmc_convergence(request)
        metrics = [_metric("point_count", len(result.points), category="workload")]
        metrics.extend(
            _metric(
                f"error_slope.{slope.method.value.lower()}",
                slope.error_slope,
                category="convergence",
            )
            for slope in result.slopes
        )
        return DispatchOutput(result, _metrics(*metrics))

    @staticmethod
    def _qmc_price_output(result: Any) -> DispatchOutput:
        return DispatchOutput(
            result,
            _metrics(
                _metric("price", result.price, category="pricing"),
                _metric("standard_error", result.standard_error, category="uncertainty"),
                _metric("absolute_error", result.absolute_error, category="error"),
                _metric("model_runtime_ms", result.runtime_ms, unit="ms", category="performance"),
            ),
        )

    @staticmethod
    def _noop_benchmark(definition: ExperimentDefinition) -> DispatchOutput:
        if definition.operation != "NOOP":
            raise ExperimentError(f"unsupported benchmark operation: {definition.operation}")
        value = float(definition.input_payload.get("value", 0.0))
        return DispatchOutput({"value": value}, _metrics(_metric("value", value)))

    def _validate_heston_config(self, config: Any) -> None:
        if config.num_simulations > self.settings.max_heston_paths:
            raise ExperimentError("Heston path count exceeds configured limit")
        if config.num_time_steps > self.settings.max_heston_time_steps:
            raise ExperimentError("Heston time-step count exceeds configured limit")
        if (
            config.num_simulations * config.num_time_steps
            > self.settings.max_heston_total_path_nodes
        ):
            raise ExperimentError("Heston path-time work exceeds configured limit")

    def _validate_qmc(self, config: QMCConfig) -> None:
        if config.effective_num_samples > self.settings.max_qmc_samples:
            raise ExperimentError("QMC sample count exceeds configured limit")
        if config.dimension > self.settings.max_qmc_dimension:
            raise ExperimentError("QMC dimension exceeds configured limit")
        if config.num_randomizations > self.settings.max_qmc_replicates:
            raise ExperimentError("QMC replicate count exceeds configured limit")
        if (
            config.effective_num_samples * config.dimension * config.num_randomizations
            > self.settings.max_qmc_total_coordinates
        ):
            raise ExperimentError("QMC coordinate count exceeds configured limit")


def aggregate_metrics(runs: list[ExperimentRun]) -> list[MetricAggregate]:
    values: dict[str, list[float]] = {}
    for run in runs:
        if run.status is not ExperimentRunStatus.COMPLETED:
            continue
        for metric in run.metrics:
            if metric.category != "performance":
                values.setdefault(metric.name, []).append(metric.value)
    aggregates = []
    for name in sorted(values):
        sample = values[name]
        std = statistics.stdev(sample) if len(sample) > 1 else 0.0
        aggregates.append(
            MetricAggregate(
                name=name,
                count=len(sample),
                mean=statistics.fmean(sample),
                standard_deviation=std,
                minimum=min(sample),
                maximum=max(sample),
                median=statistics.median(sample),
                standard_error=std / math.sqrt(len(sample)),
            )
        )
    return aggregates


def _aggregate_metric_records(aggregates: list[MetricAggregate]) -> list[ExperimentMetric]:
    result: list[ExperimentMetric] = []
    for aggregate in aggregates:
        for suffix, value in (
            ("mean", aggregate.mean),
            ("std", aggregate.standard_deviation),
            ("min", aggregate.minimum),
            ("max", aggregate.maximum),
            ("median", aggregate.median),
            ("standard_error", aggregate.standard_error),
        ):
            result.append(
                ExperimentMetric(
                    name=f"{aggregate.name}.{suffix}", value=value, category="aggregation"
                )
            )
    return result


class ExperimentRunner:
    def __init__(self, store: ExperimentStore, settings: Settings) -> None:
        self.store = store
        self.settings = settings
        self.dispatcher = ExperimentDispatcher(settings)

    def run(
        self,
        definition: ExperimentDefinition,
        *,
        parent_run_id: str | None = None,
        baseline_run_id: str | None = None,
    ) -> ExperimentRun:
        self._validate_definition(definition)
        if definition.repetitions > 1:
            return self._run_repetitions(definition, parent_run_id, baseline_run_id)
        return self._run_single(definition, parent_run_id, baseline_run_id)

    def _run_single(
        self,
        definition: ExperimentDefinition,
        parent_run_id: str | None,
        baseline_run_id: str | None,
    ) -> ExperimentRun:
        definition = _with_seed(definition, definition.root_seed)
        started = perf_counter()
        signature = experiment_signature(definition)
        environment = capture_environment(additional_metadata=definition.metadata)
        run = self.store.create_run(
            definition,
            signature,
            environment,
            parent_run_id=parent_run_id,
            baseline_run_id=baseline_run_id,
            seed_lineage=[] if definition.root_seed is None else [definition.root_seed],
        )
        self.store.start_run(run.run_id)
        engine_started = perf_counter()
        try:
            output = self.dispatcher.dispatch(definition)
            engine_runtime_ms = (perf_counter() - engine_started) * 1_000.0
            metrics = list(output.metrics)
            metrics.append(
                ExperimentMetric(
                    name="engine_runtime_ms",
                    value=engine_runtime_ms,
                    unit="ms",
                    category="performance",
                )
            )
            if len(metrics) > self.settings.max_experiment_metrics:
                raise ExperimentError("experiment metric count exceeds configured limit")
            summary = self._bounded_summary(output.result)
            duration_ms = (perf_counter() - started) * 1_000.0
            tracking_ms = max(duration_ms - engine_runtime_ms, 0.0)
            return self.store.complete_run(
                run.run_id,
                result_summary=summary,
                metrics=metrics,
                duration_ms=duration_ms,
                engine_runtime_ms=engine_runtime_ms,
                tracking_overhead_ms=tracking_ms,
            )
        except Exception as exc:
            duration_ms = (perf_counter() - started) * 1_000.0
            engine_runtime_ms = (perf_counter() - engine_started) * 1_000.0
            failure = ExperimentFailure(
                failure_type=type(exc).__name__, message=str(exc)[:2_000] or type(exc).__name__
            )
            return self.store.fail_run(
                run.run_id,
                failure,
                duration_ms=duration_ms,
                engine_runtime_ms=engine_runtime_ms,
                tracking_overhead_ms=max(duration_ms - engine_runtime_ms, 0.0),
            )

    def _run_repetitions(
        self,
        definition: ExperimentDefinition,
        parent_run_id: str | None,
        baseline_run_id: str | None,
    ) -> ExperimentRun:
        assert definition.root_seed is not None
        started = perf_counter()
        seeds = spawn_child_seeds(definition.root_seed, definition.repetitions)
        signature = experiment_signature(definition)
        parent = self.store.create_run(
            definition,
            signature,
            capture_environment(additional_metadata=definition.metadata),
            parent_run_id=parent_run_id,
            baseline_run_id=baseline_run_id,
            seed_lineage=seeds,
        )
        self.store.start_run(parent.run_id)
        children = [
            self._run_single(
                _with_seed(
                    definition.model_copy(
                        update={
                            "name": f"{definition.name} repetition {index + 1}",
                            "repetitions": 1,
                        }
                    ),
                    seed,
                ),
                parent.run_id,
                baseline_run_id,
            )
            for index, seed in enumerate(seeds)
        ]
        if any(child.status is ExperimentRunStatus.FAILED for child in children):
            duration = (perf_counter() - started) * 1_000.0
            return self.store.fail_run(
                parent.run_id,
                ExperimentFailure(
                    failure_type="ChildRunFailure", message="one or more repeated runs failed"
                ),
                duration_ms=duration,
                engine_runtime_ms=sum(child.engine_runtime_ms or 0.0 for child in children),
                tracking_overhead_ms=max(
                    duration - sum(child.engine_runtime_ms or 0.0 for child in children), 0.0
                ),
            )
        aggregates = aggregate_metrics(children)
        metrics = _aggregate_metric_records(aggregates)
        if len(metrics) > self.settings.max_experiment_metrics:
            duration = (perf_counter() - started) * 1_000.0
            engine = sum(child.engine_runtime_ms or 0.0 for child in children)
            return self.store.fail_run(
                parent.run_id,
                ExperimentFailure(
                    failure_type="ExperimentError",
                    message="repeat aggregation metric count exceeds configured limit",
                ),
                duration_ms=duration,
                engine_runtime_ms=engine,
                tracking_overhead_ms=max(duration - engine, 0.0),
            )
        duration = (perf_counter() - started) * 1_000.0
        engine = sum(child.engine_runtime_ms or 0.0 for child in children)
        return self.store.complete_run(
            parent.run_id,
            result_summary=self._bounded_summary(
                {
                    "child_run_ids": [child.run_id for child in children],
                    "seed_lineage": seeds,
                    "metric_aggregates": [item.model_dump(mode="json") for item in aggregates],
                }
            ),
            metrics=metrics,
            duration_ms=duration,
            engine_runtime_ms=engine,
            tracking_overhead_ms=max(duration - engine, 0.0),
        )

    def sweep(
        self, request: ExperimentSweepRequest, *, parent_run_id: str | None = None
    ) -> ExperimentSweepResult:
        self._validate_definition(request.definition)
        combinations = math.prod(len(axis.values) for axis in request.axes)
        total_runs = combinations * request.definition.repetitions
        if total_runs > self.settings.max_experiment_runs_per_sweep:
            raise ExperimentError("experiment sweep exceeds configured run limit")
        root_seed_swept = any(axis.path == "root_seed" for axis in request.axes)
        child_seeds = (
            spawn_child_seeds(request.definition.root_seed, combinations)
            if request.definition.root_seed is not None and not root_seed_swept
            else [None] * combinations
        )
        child_definitions: list[ExperimentDefinition] = []
        products = itertools.product(*(axis.values for axis in request.axes))
        for index, (values, seed) in enumerate(zip(products, child_seeds, strict=True)):
            child = request.definition.model_dump(mode="python")
            for axis, value in zip(request.axes, values, strict=True):
                self._set_sweep_value(child, axis.path, value)
            child["name"] = f"{request.definition.name} sweep {index + 1}"
            if seed is not None:
                child["root_seed"] = seed
            try:
                child_definition = ExperimentDefinition.model_validate(child)
            except ValidationError as exc:
                raise ExperimentError(
                    f"invalid sweep child at combination {index + 1}: {exc.errors()[0]['msg']}"
                ) from exc
            self._validate_definition(child_definition)
            child_definitions.append(child_definition)
        parent_definition = request.definition.model_copy(
            update={
                "name": f"{request.definition.name} sweep",
                "operation": f"{request.definition.operation}_SWEEP",
                "repetitions": 1,
                "config_payload": {
                    "_experiment_sweep": {
                        "axes": [axis.model_dump(mode="json") for axis in request.axes],
                        "base_operation": request.definition.operation,
                        "base_repetitions": request.definition.repetitions,
                        "base_config_payload": request.definition.config_payload,
                    }
                },
            }
        )
        started = perf_counter()
        parent = self.store.create_run(
            parent_definition,
            experiment_signature(parent_definition),
            capture_environment(additional_metadata=parent_definition.metadata),
            parent_run_id=parent_run_id,
            seed_lineage=[
                child.root_seed for child in child_definitions if child.root_seed is not None
            ],
        )
        self.store.start_run(parent.run_id)
        children = [
            self.run(child_definition, parent_run_id=parent.run_id)
            for child_definition in child_definitions
        ]
        if any(child.status is ExperimentRunStatus.FAILED for child in children):
            duration = (perf_counter() - started) * 1_000.0
            failed_parent = self.store.fail_run(
                parent.run_id,
                ExperimentFailure(
                    failure_type="ChildRunFailure", message="one or more sweep children failed"
                ),
                duration_ms=duration,
                engine_runtime_ms=sum(child.engine_runtime_ms or 0.0 for child in children),
                tracking_overhead_ms=max(
                    duration - sum(child.engine_runtime_ms or 0.0 for child in children), 0.0
                ),
            )
            return ExperimentSweepResult(
                parent=failed_parent, children=children, metric_aggregates=[]
            )
        aggregates = aggregate_metrics(children)
        duration = (perf_counter() - started) * 1_000.0
        engine = sum(child.engine_runtime_ms or 0.0 for child in children)
        metrics = _aggregate_metric_records(aggregates)
        if len(metrics) > self.settings.max_experiment_metrics:
            failed = self.store.fail_run(
                parent.run_id,
                ExperimentFailure(
                    failure_type="ExperimentError",
                    message="sweep aggregation metric count exceeds configured limit",
                ),
                duration_ms=duration,
                engine_runtime_ms=engine,
                tracking_overhead_ms=max(duration - engine, 0.0),
            )
            return ExperimentSweepResult(parent=failed, children=children, metric_aggregates=[])
        completed = self.store.complete_run(
            parent.run_id,
            result_summary=self._bounded_summary(
                {
                    "child_run_ids": [child.run_id for child in children],
                    "combinations": combinations,
                    "metric_aggregates": [item.model_dump(mode="json") for item in aggregates],
                }
            ),
            metrics=metrics,
            duration_ms=duration,
            engine_runtime_ms=engine,
            tracking_overhead_ms=max(duration - engine, 0.0),
        )
        return ExperimentSweepResult(
            parent=completed, children=children, metric_aggregates=aggregates
        )

    def benchmark(
        self, request: ExperimentBenchmarkRequest, *, parent_run_id: str | None = None
    ) -> ExperimentRun:
        self._validate_definition(request.target)
        if request.target.repetitions != 1:
            raise ExperimentError("benchmark targets must define one execution per timing sample")
        if request.repetition_count > self.settings.max_benchmark_repetitions:
            raise ExperimentError("benchmark repetition count exceeds configured limit")
        benchmark_definition = ExperimentDefinition(
            experiment_type=ExperimentType.BENCHMARK,
            operation="BENCHMARK",
            name=request.operation,
            tags=sorted(set([*request.target.tags, "benchmark"])),
            input_payload={
                "target": request.target.model_dump(mode="json"),
                "benchmark_operation": request.operation,
                "workload_size": request.workload_size,
            },
            config_payload={
                "warmup_count": request.warmup_count,
                "repetition_count": request.repetition_count,
                "thresholds": request.thresholds.model_dump(mode="json"),
            },
            root_seed=request.target.root_seed,
        )
        started = perf_counter()
        run = self.store.create_run(
            benchmark_definition,
            experiment_signature(benchmark_definition),
            capture_environment(additional_metadata=request.target.metadata),
            parent_run_id=parent_run_id,
            baseline_run_id=request.baseline_run_id,
        )
        self.store.start_run(run.run_id)
        warmup_runtime = 0.0
        try:
            for _ in range(request.warmup_count):
                warmup_started = perf_counter()
                self.dispatcher.dispatch(_with_seed(request.target, request.target.root_seed))
                warmup_runtime += (perf_counter() - warmup_started) * 1_000.0
            timings: list[float] = []
            last_output: DispatchOutput | None = None
            for _ in range(request.repetition_count):
                point_started = perf_counter()
                last_output = self.dispatcher.dispatch(
                    _with_seed(request.target, request.target.root_seed)
                )
                timings.append((perf_counter() - point_started) * 1_000.0)
            assert last_output is not None
            mean = statistics.fmean(timings)
            median = statistics.median(timings)
            p95 = float(np.quantile(np.asarray(timings), 0.95))
            metrics = _metrics(
                _metric("benchmark_mean_ms", mean, unit="ms", category="performance"),
                _metric("benchmark_median_ms", median, unit="ms", category="performance"),
                _metric("benchmark_p95_ms", p95, unit="ms", category="performance"),
                _metric("benchmark_min_ms", min(timings), unit="ms", category="performance"),
                _metric("benchmark_max_ms", max(timings), unit="ms", category="performance"),
                *[
                    ExperimentMetric(
                        name=f"target.{metric.name}",
                        value=metric.value,
                        unit=metric.unit,
                        category=metric.category,
                    )
                    for metric in last_output.metrics
                    if metric.category != "performance"
                ],
            )
            flags = self._benchmark_flags(request, median, metrics)
            baseline_comparison = self._benchmark_baseline_comparison(request, median, metrics)
            duration = (perf_counter() - started) * 1_000.0
            measured_engine = sum(timings)
            tracking = max(duration - measured_engine - warmup_runtime, 0.0)
            return self.store.complete_run(
                run.run_id,
                result_summary=self._bounded_summary(
                    {
                        "operation": request.operation,
                        "workload_size": request.workload_size,
                        "warmup_count": request.warmup_count,
                        "repetition_count": request.repetition_count,
                        "timing_samples_ms": timings,
                        "warmup_runtime_ms": warmup_runtime,
                        "mean_ms": mean,
                        "median_ms": median,
                        "p95_ms": p95,
                        "minimum_ms": min(timings),
                        "maximum_ms": max(timings),
                        "baseline_comparison": baseline_comparison,
                    }
                ),
                metrics=metrics,
                duration_ms=duration,
                engine_runtime_ms=measured_engine,
                tracking_overhead_ms=tracking,
                flags=flags,
            )
        except Exception as exc:
            duration = (perf_counter() - started) * 1_000.0
            return self.store.fail_run(
                run.run_id,
                ExperimentFailure(failure_type=type(exc).__name__, message=str(exc)),
                duration_ms=duration,
                engine_runtime_ms=None,
                tracking_overhead_ms=duration,
            )

    def reproduce(self, run_id: str, request: ReproduceRequest) -> ReproducibilityResult:
        original = self.store.get_run(run_id)
        if original.status is not ExperimentRunStatus.COMPLETED:
            raise ExperimentError("only completed runs can be reproduced")
        definition = request.override_definition or original.definition
        if (
            request.override_definition is None
            and definition.experiment_type is ExperimentType.BENCHMARK
            and definition.operation == "BENCHMARK"
        ):
            replay = self.benchmark(
                ExperimentBenchmarkRequest(
                    target=ExperimentDefinition.model_validate(definition.input_payload["target"]),
                    operation=str(
                        definition.input_payload.get("benchmark_operation", definition.name)
                    ),
                    workload_size=definition.input_payload["workload_size"],
                    warmup_count=int(definition.config_payload["warmup_count"]),
                    repetition_count=int(definition.config_payload["repetition_count"]),
                    baseline_run_id=original.baseline_run_id,
                    thresholds=definition.config_payload["thresholds"],
                ),
                parent_run_id=original.run_id,
            )
        elif (
            request.override_definition is None
            and definition.operation.endswith("_SWEEP")
            and "_experiment_sweep" in definition.config_payload
        ):
            sweep_metadata = definition.config_payload["_experiment_sweep"]
            if not isinstance(sweep_metadata, dict):
                raise ExperimentError("stored sweep metadata is invalid")
            base = definition.model_copy(
                update={
                    "operation": sweep_metadata["base_operation"],
                    "repetitions": sweep_metadata["base_repetitions"],
                    "config_payload": sweep_metadata["base_config_payload"],
                }
            )
            replay = self.sweep(
                ExperimentSweepRequest(
                    definition=base,
                    axes=[SweepAxis.model_validate(axis) for axis in sweep_metadata["axes"]],
                ),
                parent_run_id=original.run_id,
            ).parent
        else:
            replay = self.run(definition, parent_run_id=original.run_id)
        signature_match = replay.input_signature == original.input_signature
        metric_match, differences, diagnostics = reproducible_metrics(
            original, replay, request.tolerance
        )
        if replay.status is not ExperimentRunStatus.COMPLETED:
            metric_match = False
            diagnostics.append("replay execution failed")
        if not signature_match:
            diagnostics.append("input signature differs")
        result_match = signature_match and metric_match
        flags = [] if result_match else [ResearchFlag.REPRODUCIBILITY_MISMATCH]
        runtime_difference = (
            None
            if original.engine_runtime_ms is None or replay.engine_runtime_ms is None
            else replay.engine_runtime_ms - original.engine_runtime_ms
        )
        return ReproducibilityResult(
            original_run_id=original.run_id,
            replay_run_id=replay.run_id,
            signature_match=signature_match,
            result_match=result_match,
            metric_differences=differences,
            runtime_difference_ms=runtime_difference,
            diagnostics=diagnostics,
            flags=flags,
        )

    def _bounded_summary(self, result: Any) -> dict[str, Any]:
        normalized = _json_result(result)
        if not isinstance(normalized, dict):
            normalized = {"value": normalized}
        encoded = canonical_json(normalized).encode("utf-8")
        if len(encoded) <= self.settings.max_experiment_result_bytes:
            return normalized
        scalars = {
            key: value
            for key, value in normalized.items()
            if value is None or isinstance(value, (str, int, float, bool))
        }
        summary = {
            "summary_truncated": True,
            "original_size_bytes": len(encoded),
            "result_type": type(result).__name__,
            "scalar_fields": scalars,
        }
        if len(canonical_json(summary).encode("utf-8")) > self.settings.max_experiment_result_bytes:
            raise ExperimentError("bounded result summary exceeds configured size limit")
        return summary

    def _validate_definition(self, definition: ExperimentDefinition) -> None:
        if definition.repetitions > self.settings.max_experiment_repetitions:
            raise ExperimentError("experiment repetitions exceed configured limit")
        if len(definition.tags) > self.settings.max_experiment_tags:
            raise ExperimentError("experiment tag count exceeds configured limit")
        if (
            definition.notes is not None
            and len(definition.notes) > self.settings.max_experiment_notes_length
        ):
            raise ExperimentError("experiment notes exceed configured limit")

    @staticmethod
    def _set_sweep_value(definition: dict[str, Any], path: str, value: Any) -> None:
        parts = path.split(".")
        if parts[0] not in {"input_payload", "config_payload", "root_seed"}:
            parts.insert(0, "input_payload")
        target: dict[str, Any] = definition
        for component in parts[:-1]:
            child = target.get(component)
            if not isinstance(child, dict):
                child = {}
                target[component] = child
            target = child
        target[parts[-1]] = deepcopy(value)

    def _benchmark_flags(
        self,
        request: ExperimentBenchmarkRequest,
        median_ms: float,
        metrics: list[ExperimentMetric],
    ) -> list[ResearchFlag]:
        if request.baseline_run_id is None:
            return []
        baseline = self.store.get_run(request.baseline_run_id)
        flags: list[ResearchFlag] = []
        baseline_metrics = {metric.name: metric.value for metric in baseline.metrics}
        baseline_median = baseline_metrics.get("benchmark_median_ms")
        threshold = request.thresholds.max_runtime_regression_pct
        if baseline_median is not None and baseline_median > 0.0 and threshold is not None:
            increase = (median_ms - baseline_median) / baseline_median * 100.0
            if increase > threshold:
                flags.append(ResearchFlag.RUNTIME_REGRESSION)
        metric_threshold = request.thresholds.max_metric_change
        if metric_threshold is not None:
            for metric in metrics:
                if (
                    metric.category != "performance"
                    and metric.name in baseline_metrics
                    and abs(metric.value - baseline_metrics[metric.name]) > metric_threshold
                ):
                    flags.append(ResearchFlag.RESULT_CHANGED)
                    break
        error_threshold = request.thresholds.max_error_increase
        if error_threshold is not None:
            for metric in metrics:
                if (
                    metric.category != "performance"
                    and "error" in metric.name.lower()
                    and metric.name in baseline_metrics
                    and metric.value - baseline_metrics[metric.name] > error_threshold
                ):
                    flags.append(ResearchFlag.ERROR_INCREASED)
                    break
        return flags

    def _benchmark_baseline_comparison(
        self,
        request: ExperimentBenchmarkRequest,
        median_ms: float,
        metrics: list[ExperimentMetric],
    ) -> dict[str, Any] | None:
        if request.baseline_run_id is None:
            return None
        baseline = self.store.get_run(request.baseline_run_id)
        baseline_metrics = {metric.name: metric.value for metric in baseline.metrics}
        baseline_median = baseline_metrics.get("benchmark_median_ms")
        runtime_difference = None if baseline_median is None else median_ms - baseline_median
        runtime_difference_pct = (
            None
            if runtime_difference is None or baseline_median is None or baseline_median == 0.0
            else runtime_difference / baseline_median * 100.0
        )
        return {
            "baseline_run_id": baseline.run_id,
            "median_runtime_difference_ms": runtime_difference,
            "median_runtime_difference_pct": runtime_difference_pct,
            "metric_differences": [
                item.model_dump(mode="json")
                for item in metric_differences(baseline.metrics, metrics)
            ],
        }
