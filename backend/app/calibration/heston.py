"""Deterministic bounded Heston calibration and fit diagnostics."""

import math
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import differential_evolution, least_squares

from backend.app.calibration.datasets import (
    HESTON_PARAMETER_ORDER,
    PreparedCalibrationData,
    PreparedCalibrationObservation,
    heston_parameters_to_vector,
    parameters_within_bounds,
    prepare_calibration_data,
    scipy_parameter_bounds,
    vector_to_heston_parameters,
)
from backend.app.core.exceptions import HestonCalibrationError, QuantitativeError
from backend.app.models.heston import HestonIntegrationConfig, HestonParameters
from backend.app.models.heston_calibration import (
    CalibrationAccuracy,
    CalibrationMoneynessBucket,
    CalibrationRobustLoss,
    CalibrationStage,
    HestonATMFitPoint,
    HestonCalibrationConfig,
    HestonCalibrationDataset,
    HestonCalibrationDiagnosticsResult,
    HestonCalibrationMetrics,
    HestonCalibrationOptimizer,
    HestonCalibrationResidual,
    HestonCalibrationResult,
    HestonCalibrationStartResult,
    HestonCalibrationTarget,
    HestonExpiryCalibrationDiagnostics,
    HestonFittedSurfacePoint,
    HestonFittedSurfaceResult,
    HestonIdentifiabilityDiagnostics,
    HestonMoneynessCalibrationDiagnostics,
    HestonSkewFitPoint,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.volatility import ImpliedVolatilityConfig, ImpliedVolatilityMarket
from backend.app.models.volatility import ImpliedVolatilityObservation as IVObservation
from backend.app.pricing.heston_analytical import HestonAnalyticalBatchPricer
from backend.app.pricing.implied_volatility import solve_implied_volatility
from backend.app.simulation.random import create_rng, spawn_child_seeds


@dataclass(frozen=True)
class _ModelEvaluation:
    parameters: HestonParameters
    raw_residuals: NDArray[np.float64]
    weighted_residuals: NDArray[np.float64]
    model_prices: tuple[float | None, ...]
    model_implied_volatilities: tuple[float | None, ...]
    failure_messages: tuple[str | None, ...]


@dataclass(frozen=True)
class _OptimizationOutcome:
    vector: NDArray[np.float64]
    objective: float
    cost: float
    optimality: float | None
    converged: bool
    status: int
    message: str
    iterations: int
    evaluations: int
    jacobian_evaluations: int | None
    runtime_ms: float
    stage: CalibrationStage
    initial: NDArray[np.float64]


def calibration_integration_config(config: HestonCalibrationConfig) -> HestonIntegrationConfig:
    if config.integration is not None:
        return config.integration
    if config.accuracy is CalibrationAccuracy.FAST:
        return HestonIntegrationConfig(
            upper_bound=200.0,
            segment_width=25.0,
            absolute_tolerance=1e-6,
            relative_tolerance=1e-6,
            tail_tolerance=1e-6,
            quadrature_limit=100,
        )
    if config.accuracy is CalibrationAccuracy.HIGH_ACCURACY:
        return HestonIntegrationConfig(
            upper_bound=600.0,
            segment_width=20.0,
            absolute_tolerance=1e-10,
            relative_tolerance=1e-9,
            tail_tolerance=1e-10,
            quadrature_limit=300,
        )
    return HestonIntegrationConfig(
        upper_bound=400.0,
        segment_width=20.0,
        absolute_tolerance=1e-8,
        relative_tolerance=1e-7,
        tail_tolerance=1e-7,
        quadrature_limit=200,
    )


def _iv_config(config: HestonCalibrationConfig) -> ImpliedVolatilityConfig:
    if config.accuracy is CalibrationAccuracy.FAST:
        return ImpliedVolatilityConfig(price_tolerance=1e-7, volatility_tolerance=1e-7)
    if config.accuracy is CalibrationAccuracy.HIGH_ACCURACY:
        return ImpliedVolatilityConfig(price_tolerance=1e-11, volatility_tolerance=1e-11)
    return ImpliedVolatilityConfig(price_tolerance=1e-9, volatility_tolerance=1e-9)


class HestonCalibrationObjective:
    """One lifetime-scoped finite objective with exact parameter-vector caching."""

    def __init__(self, data: PreparedCalibrationData, config: HestonCalibrationConfig) -> None:
        self.data = data
        self.config = config
        self.integration = calibration_integration_config(config)
        self.iv_config = _iv_config(config)
        self.iv_market = ImpliedVolatilityMarket(
            risk_free_rate=data.dataset.risk_free_rate,
            dividend_yield=data.dataset.dividend_yield,
        )
        self.market = MarketEnvironment(
            risk_free_rate=data.dataset.risk_free_rate,
            dividend_yield=data.dataset.dividend_yield,
            volatility=0.0,
        )
        self.cache: dict[tuple[float, ...], _ModelEvaluation] = {}
        self.objective_evaluations = 0
        self.pricing_evaluations = 0
        self.failed_evaluations = 0
        self.cache_hits = 0

    def evaluate(self, vector: NDArray[np.float64]) -> _ModelEvaluation:
        self.objective_evaluations += 1
        key = tuple(float(value) for value in vector)
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        parameters = vector_to_heston_parameters(vector)
        batch_pricers: dict[tuple[float, float], HestonAnalyticalBatchPricer] = {}
        for item in self.data.observations:
            batch_key = (item.contract.spot, item.contract.maturity_years)
            if batch_key not in batch_pricers:
                batch_pricers[batch_key] = HestonAnalyticalBatchPricer(
                    spot=item.contract.spot,
                    maturity_years=item.contract.maturity_years,
                    market=self.market,
                    parameters=parameters,
                    config=self.integration,
                )
        raw: list[float] = []
        model_prices: list[float | None] = []
        model_ivs: list[float | None] = []
        failures: list[str | None] = []
        for item in self.data.observations:
            try:
                self.pricing_evaluations += 1
                batch_key = (item.contract.spot, item.contract.maturity_years)
                model_price = batch_pricers[batch_key].price(item.contract).price
                model_iv: float | None = None
                if self.config.target is not HestonCalibrationTarget.PRICE:
                    model_iv = solve_implied_volatility(
                        IVObservation(
                            contract=item.contract,
                            market=self.iv_market,
                            observed_price=model_price,
                        ),
                        self.iv_config,
                    ).implied_volatility
                    if model_iv is None:
                        raise HestonCalibrationError(
                            "model price has no identifiable implied volatility"
                        )
                if self.config.target is HestonCalibrationTarget.PRICE:
                    model_target = model_price
                elif self.config.target is HestonCalibrationTarget.IMPLIED_VOLATILITY:
                    assert model_iv is not None
                    model_target = model_iv
                else:
                    assert model_iv is not None
                    model_target = model_iv * model_iv * item.observation.maturity_years
                residual = model_target - item.target_value
                if not math.isfinite(residual):
                    raise HestonCalibrationError("model target residual is non-finite")
                raw.append(residual)
                model_prices.append(model_price)
                model_ivs.append(model_iv)
                failures.append(None)
            except (QuantitativeError, ValueError, OverflowError, FloatingPointError) as exc:
                self.failed_evaluations += 1
                raw.append(self.config.failure_penalty)
                model_prices.append(None)
                model_ivs.append(None)
                failures.append(str(exc))
        raw_array = np.asarray(raw, dtype=np.float64)
        weights = np.asarray([item.weight for item in self.data.observations], dtype=np.float64)
        weighted = np.sqrt(weights) * raw_array
        if not np.all(np.isfinite(weighted)):
            weighted = np.full_like(weighted, self.config.failure_penalty)
        result = _ModelEvaluation(
            parameters=parameters,
            raw_residuals=raw_array,
            weighted_residuals=weighted,
            model_prices=tuple(model_prices),
            model_implied_volatilities=tuple(model_ivs),
            failure_messages=tuple(failures),
        )
        self.cache[key] = result
        return result

    def feller_penalty(self, parameters: HestonParameters) -> float:
        violation = max(
            parameters.sigma_v * parameters.sigma_v - 2.0 * parameters.kappa * parameters.theta,
            0.0,
        )
        return self.config.feller_penalty_weight * violation * violation

    def regularization(self, vector: NDArray[np.float64]) -> float:
        prior = self.config.regularization_parameters
        if self.config.regularization_weight == 0.0 or prior is None:
            return 0.0
        lower, upper = scipy_parameter_bounds(self.config.bounds)
        normalized = (vector - heston_parameters_to_vector(prior)) / (upper - lower)
        return self.config.regularization_weight * float(np.dot(normalized, normalized))

    def residual_vector(self, vector: NDArray[np.float64]) -> NDArray[np.float64]:
        evaluation = self.evaluate(vector)
        count = len(self.data.observations)
        extras: list[float] = []
        feller = self.feller_penalty(evaluation.parameters)
        if self.config.feller_penalty_weight > 0.0:
            extras.append(math.sqrt(count * feller))
        regularization = self.regularization(vector)
        if self.config.regularization_weight > 0.0:
            extras.append(math.sqrt(count * regularization))
        if not extras:
            return evaluation.weighted_residuals
        return np.concatenate((evaluation.weighted_residuals, np.asarray(extras)))

    def scalar(self, vector: NDArray[np.float64]) -> float:
        residuals = self.residual_vector(vector)
        scaled_squared = (residuals / self.config.f_scale) ** 2
        if self.config.robust_loss is CalibrationRobustLoss.LINEAR:
            transformed = scaled_squared
        elif self.config.robust_loss is CalibrationRobustLoss.SOFT_L1:
            transformed = 2.0 * (np.sqrt(1.0 + scaled_squared) - 1.0)
        else:
            transformed = np.where(
                scaled_squared <= 1.0,
                scaled_squared,
                2.0 * np.sqrt(scaled_squared) - 1.0,
            )
        value = self.config.f_scale**2 * float(np.sum(transformed)) / len(self.data.observations)
        return value if math.isfinite(value) else self.config.failure_penalty**2


def _clipped(value: float, lower: float, upper: float) -> float:
    margin = 1e-8 * (upper - lower)
    return min(max(value, lower + margin), upper - margin)


def heuristic_initial_parameters(
    data: PreparedCalibrationData,
    config: HestonCalibrationConfig,
) -> HestonParameters:
    """Use short/long ATM variance and a simple observed skew orientation."""

    by_maturity: dict[float, list[PreparedCalibrationObservation]] = {}
    for item in data.observations:
        by_maturity.setdefault(item.observation.maturity_years, []).append(item)
    maturities = sorted(by_maturity)
    short = min(by_maturity[maturities[0]], key=lambda item: abs(item.log_moneyness))
    long = min(by_maturity[maturities[-1]], key=lambda item: abs(item.log_moneyness))
    all_items = list(data.observations)
    coordinates = np.asarray([item.log_moneyness for item in all_items])
    volatilities = np.asarray([item.market_implied_volatility for item in all_items])
    rho = -0.5
    if len(all_items) >= 2 and float(np.ptp(coordinates)) > 1e-12:
        slope = float(np.polyfit(coordinates, volatilities, 1)[0])
        rho = min(max(4.0 * slope, -0.8), 0.8)
    ranges = config.bounds
    values = {
        "v0": _clipped(short.market_implied_volatility**2, ranges.v0.lower, ranges.v0.upper),
        "kappa": _clipped(1.5, ranges.kappa.lower, ranges.kappa.upper),
        "theta": _clipped(
            long.market_implied_volatility**2, ranges.theta.lower, ranges.theta.upper
        ),
        "sigma_v": _clipped(0.4, ranges.sigma_v.lower, ranges.sigma_v.upper),
        "rho": _clipped(rho, ranges.rho.lower, ranges.rho.upper),
    }
    return HestonParameters(**values)


def _random_start(
    seed: int,
    config: HestonCalibrationConfig,
) -> HestonParameters:
    rng = create_rng(seed)
    values: list[float] = []
    for name in HESTON_PARAMETER_ORDER:
        interval = getattr(config.bounds, name)
        if name != "rho" and interval.lower > 0.0:
            value = math.exp(rng.uniform(math.log(interval.lower), math.log(interval.upper)))
        else:
            value = rng.uniform(interval.lower, interval.upper)
        values.append(float(value))
    return vector_to_heston_parameters(values)


def random_initial_parameters(
    seed: int,
    config: HestonCalibrationConfig,
) -> HestonParameters:
    """Return one deterministic bounded random start for stability research."""

    return _random_start(seed, config)


def calibration_initial_starts(
    data: PreparedCalibrationData,
    config: HestonCalibrationConfig,
) -> list[HestonParameters]:
    starts: list[HestonParameters] = []
    if config.initial_parameters is not None:
        if not parameters_within_bounds(config.initial_parameters, config.bounds):
            raise HestonCalibrationError("initial Heston parameters lie outside calibration bounds")
        starts.append(config.initial_parameters)
    heuristic = heuristic_initial_parameters(data, config)
    if not starts or not np.array_equal(
        heston_parameters_to_vector(starts[0]), heston_parameters_to_vector(heuristic)
    ):
        starts.append(heuristic)
    seeds = spawn_child_seeds(config.random_seed, config.multi_start_count)
    index = 0
    while len(starts) < config.multi_start_count:
        starts.append(_random_start(seeds[index], config))
        index += 1
    return starts[: config.multi_start_count]


def _run_least_squares(
    objective: HestonCalibrationObjective,
    start: NDArray[np.float64],
    config: HestonCalibrationConfig,
    stage: CalibrationStage = CalibrationStage.LOCAL,
) -> _OptimizationOutcome:
    lower, upper = scipy_parameter_bounds(config.bounds)
    started = perf_counter()
    result = least_squares(
        objective.residual_vector,
        start,
        bounds=(lower, upper),
        ftol=config.tolerance,
        xtol=config.tolerance,
        gtol=config.tolerance,
        max_nfev=config.max_iterations,
        loss=config.robust_loss.value.lower(),
        f_scale=config.f_scale,
    )
    vector = np.asarray(result.x, dtype=np.float64)
    return _OptimizationOutcome(
        vector=vector,
        objective=objective.scalar(vector),
        cost=max(float(result.cost), 0.0),
        optimality=max(float(result.optimality), 0.0),
        converged=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        iterations=int(result.nfev),
        evaluations=int(result.nfev),
        jacobian_evaluations=(int(result.njev) if result.njev is not None else None),
        runtime_ms=(perf_counter() - started) * 1_000.0,
        stage=stage,
        initial=start.copy(),
    )


def _run_differential_evolution(
    objective: HestonCalibrationObjective,
    config: HestonCalibrationConfig,
) -> _OptimizationOutcome:
    lower, upper = scipy_parameter_bounds(config.bounds)
    bounds = list(zip(lower, upper, strict=True))
    started = perf_counter()
    result = differential_evolution(
        objective.scalar,
        bounds,
        seed=config.random_seed,
        maxiter=config.max_iterations,
        popsize=config.differential_evolution_population,
        tol=config.tolerance,
        polish=(
            config.optimizer is HestonCalibrationOptimizer.DIFFERENTIAL_EVOLUTION
            and config.local_refinement
        ),
        workers=1,
        updating="immediate",
    )
    vector = np.asarray(result.x, dtype=np.float64)
    objective_value = objective.scalar(vector)
    return _OptimizationOutcome(
        vector=vector,
        objective=objective_value,
        cost=max(0.5 * objective_value * len(objective.data.observations), 0.0),
        optimality=None,
        converged=bool(result.success),
        status=0 if result.success else 1,
        message=str(result.message),
        iterations=int(result.nit),
        evaluations=int(result.nfev),
        jacobian_evaluations=None,
        runtime_ms=(perf_counter() - started) * 1_000.0,
        stage=CalibrationStage.GLOBAL,
        initial=vector.copy(),
    )


def _outcome_summary(outcome: _OptimizationOutcome, index: int) -> HestonCalibrationStartResult:
    return HestonCalibrationStartResult(
        start_index=index,
        stage=outcome.stage,
        initial_parameters=vector_to_heston_parameters(outcome.initial),
        final_parameters=vector_to_heston_parameters(outcome.vector),
        objective_value=max(outcome.objective, 0.0),
        cost=outcome.cost,
        optimality=outcome.optimality,
        converged=outcome.converged,
        status=outcome.status,
        message=outcome.message,
        iterations=outcome.iterations,
        evaluations=outcome.evaluations,
        jacobian_evaluations=outcome.jacobian_evaluations,
        runtime_ms=outcome.runtime_ms,
    )


def _run_optimizers(
    objective: HestonCalibrationObjective,
    starts: list[HestonParameters],
    config: HestonCalibrationConfig,
) -> tuple[list[_OptimizationOutcome], list[HestonCalibrationStartResult]]:
    outcomes: list[_OptimizationOutcome] = []
    summaries: list[HestonCalibrationStartResult] = []
    if config.optimizer is HestonCalibrationOptimizer.LEAST_SQUARES:
        for index, parameters in enumerate(starts):
            outcome = _run_least_squares(objective, heston_parameters_to_vector(parameters), config)
            outcomes.append(outcome)
            summaries.append(_outcome_summary(outcome, index))
        return outcomes, summaries

    global_outcome = _run_differential_evolution(objective, config)
    outcomes.append(global_outcome)
    summaries.append(_outcome_summary(global_outcome, 0))
    if config.optimizer is HestonCalibrationOptimizer.HYBRID:
        local_starts = [global_outcome.vector]
        local_starts.extend(
            heston_parameters_to_vector(value)
            for value in starts[: max(config.multi_start_count - 1, 0)]
        )
        for vector in local_starts:
            outcome = _run_least_squares(
                objective,
                vector,
                config,
                stage=CalibrationStage.GLOBAL_REFINEMENT,
            )
            outcomes.append(outcome)
            summaries.append(_outcome_summary(outcome, len(summaries)))
    return outcomes, summaries


def _residual_models(
    data: PreparedCalibrationData,
    evaluation: _ModelEvaluation,
    target: HestonCalibrationTarget,
) -> list[HestonCalibrationResidual]:
    residuals = []
    for item, raw, weighted, model_price, model_iv, failure in zip(
        data.observations,
        evaluation.raw_residuals,
        evaluation.weighted_residuals,
        evaluation.model_prices,
        evaluation.model_implied_volatilities,
        evaluation.failure_messages,
        strict=True,
    ):
        if model_iv is None and model_price is not None:
            try:
                model_iv = solve_implied_volatility(
                    IVObservation(
                        contract=item.contract,
                        market=ImpliedVolatilityMarket(
                            risk_free_rate=data.dataset.risk_free_rate,
                            dividend_yield=data.dataset.dividend_yield,
                        ),
                        observed_price=model_price,
                    )
                ).implied_volatility
            except QuantitativeError:
                model_iv = None
        residuals.append(
            HestonCalibrationResidual(
                observation_index=item.original_index,
                strike=item.observation.strike,
                maturity_years=item.observation.maturity_years,
                option_type=item.observation.option_type,
                market_price=item.market_price,
                model_price=model_price,
                market_implied_volatility=item.market_implied_volatility,
                model_implied_volatility=model_iv,
                raw_residual=float(raw),
                weighted_residual=float(weighted),
                weight=item.weight,
                moneyness=item.moneyness,
                log_moneyness=item.log_moneyness,
                target=target,
                evaluation_failed=failure is not None,
                failure_message=failure,
            )
        )
    return residuals


def _metrics(
    residuals: list[HestonCalibrationResidual],
    *,
    total_observations: int | None = None,
) -> HestonCalibrationMetrics:
    if not residuals:
        return HestonCalibrationMetrics(
            observation_count=0,
            rmse=0.0,
            mae=0.0,
            weighted_rmse=0.0,
            max_abs_error=0.0,
            average_iv_error=None,
            objective_contribution=0.0,
        )
    raw = np.asarray([item.raw_residual for item in residuals])
    weighted = np.asarray([item.weighted_residual for item in residuals])
    iv_errors = [
        item.model_implied_volatility - item.market_implied_volatility
        for item in residuals
        if item.model_implied_volatility is not None and item.market_implied_volatility is not None
    ]
    denominator = total_observations or len(residuals)
    return HestonCalibrationMetrics(
        observation_count=len(residuals),
        rmse=math.sqrt(float(np.mean(raw**2))),
        mae=float(np.mean(np.abs(raw))),
        weighted_rmse=math.sqrt(float(np.mean(weighted**2))),
        max_abs_error=float(np.max(np.abs(raw))),
        average_iv_error=(float(np.mean(iv_errors)) if iv_errors else None),
        objective_contribution=float(np.dot(weighted, weighted)) / denominator,
    )


def _expiry_diagnostics(
    residuals: list[HestonCalibrationResidual],
) -> list[HestonExpiryCalibrationDiagnostics]:
    points = []
    total = len(residuals)
    for maturity in sorted({item.maturity_years for item in residuals}):
        selected = [item for item in residuals if item.maturity_years == maturity]
        metrics = _metrics(selected, total_observations=total)
        points.append(
            HestonExpiryCalibrationDiagnostics(
                maturity_years=maturity,
                **metrics.model_dump(),
            )
        )
    return points


def _bucket(residual: HestonCalibrationResidual) -> CalibrationMoneynessBucket:
    signed = (
        residual.log_moneyness if residual.option_type.value == "CALL" else -residual.log_moneyness
    )
    if signed > 0.20:
        return CalibrationMoneynessBucket.DEEP_OTM
    if signed > 0.05:
        return CalibrationMoneynessBucket.OTM
    if signed >= -0.05:
        return CalibrationMoneynessBucket.ATM
    return CalibrationMoneynessBucket.ITM


def _moneyness_diagnostics(
    residuals: list[HestonCalibrationResidual],
) -> list[HestonMoneynessCalibrationDiagnostics]:
    bounds = {
        CalibrationMoneynessBucket.DEEP_OTM: (0.20, None),
        CalibrationMoneynessBucket.OTM: (0.05, 0.20),
        CalibrationMoneynessBucket.ATM: (-0.05, 0.05),
        CalibrationMoneynessBucket.ITM: (None, -0.05),
    }
    points = []
    total = len(residuals)
    for bucket in CalibrationMoneynessBucket:
        selected = [item for item in residuals if _bucket(item) is bucket]
        if not selected:
            continue
        metrics = _metrics(selected, total_observations=total)
        lower, upper = bounds[bucket]
        points.append(
            HestonMoneynessCalibrationDiagnostics(
                bucket=bucket,
                lower_signed_moneyness=lower,
                upper_signed_moneyness=upper,
                **metrics.model_dump(),
            )
        )
    return points


def _finite_difference_jacobian(
    objective: HestonCalibrationObjective,
    vector: NDArray[np.float64],
) -> NDArray[np.float64]:
    lower, upper = scipy_parameter_bounds(objective.config.bounds)
    columns = []
    for index in range(len(vector)):
        step = 1e-4 * max(abs(vector[index]), 0.01 * (upper[index] - lower[index]), 1e-3)
        low = vector.copy()
        high = vector.copy()
        low[index] = max(lower[index], vector[index] - step)
        high[index] = min(upper[index], vector[index] + step)
        width = high[index] - low[index]
        if width <= 0.0:
            columns.append(np.zeros(len(objective.data.observations)))
            continue
        low_residuals = objective.evaluate(low).weighted_residuals
        high_residuals = objective.evaluate(high).weighted_residuals
        columns.append((high_residuals - low_residuals) / width)
    jacobian = np.column_stack(columns)
    if not np.all(np.isfinite(jacobian)):
        raise HestonCalibrationError("calibration Jacobian is non-finite")
    return np.asarray(jacobian, dtype=np.float64)


def _identifiability(
    objective: HestonCalibrationObjective,
    vector: NDArray[np.float64],
) -> HestonIdentifiabilityDiagnostics:
    jacobian = _finite_difference_jacobian(objective, vector)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    rank = int(np.linalg.matrix_rank(jacobian))
    full_rank = rank == len(HESTON_PARAMETER_ORDER)
    condition = None
    if full_rank and singular[-1] > 0.0:
        condition = float(singular[0] / singular[-1])
    covariance: list[list[float]] | None = None
    correlation: list[list[float]] | None = None
    count = jacobian.shape[0]
    if full_rank and count > len(HESTON_PARAMETER_ORDER):
        try:
            gram_inverse = np.linalg.inv(jacobian.T @ jacobian)
            residuals = objective.evaluate(vector).weighted_residuals
            residual_variance = float(np.dot(residuals, residuals)) / (
                count - len(HESTON_PARAMETER_ORDER)
            )
            covariance_array = gram_inverse * residual_variance
            standard = np.sqrt(np.maximum(np.diag(covariance_array), 0.0))
            denominator = np.outer(standard, standard)
            correlation_array = np.divide(
                covariance_array,
                denominator,
                out=np.zeros_like(covariance_array),
                where=denominator > 0.0,
            )
            if np.all(np.isfinite(covariance_array)) and np.all(np.isfinite(correlation_array)):
                covariance = covariance_array.tolist()
                correlation = correlation_array.tolist()
        except np.linalg.LinAlgError:
            pass
    note = (
        "Local finite-difference Jacobian is full rank; covariance is a local linear approximation."
        if full_rank
        else "Local finite-difference Jacobian is rank deficient; parameters are weakly identified."
    )
    return HestonIdentifiabilityDiagnostics(
        jacobian_rank=rank,
        singular_values=[max(float(value), 0.0) for value in singular],
        condition_number=condition,
        locally_full_rank=full_rank,
        covariance_available=covariance is not None,
        local_covariance=covariance,
        local_correlation=correlation,
        note=note,
    )


def _active_bounds(vector: NDArray[np.float64], config: HestonCalibrationConfig) -> list[str]:
    lower, upper = scipy_parameter_bounds(config.bounds)
    active = []
    for index, name in enumerate(HESTON_PARAMETER_ORDER):
        tolerance = 1e-5 * (upper[index] - lower[index])
        if vector[index] - lower[index] <= tolerance:
            active.append(f"{name}:LOWER")
        elif upper[index] - vector[index] <= tolerance:
            active.append(f"{name}:UPPER")
    return active


def _diagnostics(
    data: PreparedCalibrationData,
    objective: HestonCalibrationObjective,
    vector: NDArray[np.float64],
    config: HestonCalibrationConfig,
) -> tuple[
    _ModelEvaluation,
    list[HestonCalibrationResidual],
    HestonCalibrationMetrics,
    list[HestonExpiryCalibrationDiagnostics],
    list[HestonMoneynessCalibrationDiagnostics],
    HestonIdentifiabilityDiagnostics | None,
]:
    evaluation = objective.evaluate(vector)
    residuals = _residual_models(data, evaluation, config.target)
    metrics = _metrics(residuals)
    expiry = _expiry_diagnostics(residuals)
    moneyness = _moneyness_diagnostics(residuals)
    identifiability = (
        _identifiability(objective, vector) if config.identifiability_diagnostics else None
    )
    return evaluation, residuals, metrics, expiry, moneyness, identifiability


def calibrate_heston(
    dataset: HestonCalibrationDataset,
    config: HestonCalibrationConfig | None = None,
) -> HestonCalibrationResult:
    """Calibrate native bounded risk-neutral Heston parameters deterministically."""

    policy = config or HestonCalibrationConfig()
    started = perf_counter()
    data = prepare_calibration_data(dataset, policy)
    starts = calibration_initial_starts(data, policy)
    objective = HestonCalibrationObjective(data, policy)
    outcomes, summaries = _run_optimizers(objective, starts, policy)
    if not outcomes:
        raise HestonCalibrationError("calibration optimizer produced no candidate solution")
    best_index = min(range(len(outcomes)), key=lambda index: outcomes[index].objective)
    best = outcomes[best_index]
    evaluation, residuals, metrics, expiry, moneyness, identifiability = _diagnostics(
        data, objective, best.vector, policy
    )
    elapsed_ms = (perf_counter() - started) * 1_000.0
    parameter_result = evaluation.parameters
    returned_residuals = (
        residuals[: policy.max_returned_residuals] if policy.return_residuals else []
    )
    returned_starts = (
        summaries[: policy.max_returned_starts] if policy.return_start_diagnostics else []
    )
    final_evaluation_failed = any(message is not None for message in evaluation.failure_messages)
    result_message = best.message
    if final_evaluation_failed:
        result_message = f"{best.message}; selected solution contains failed model evaluations"
    return HestonCalibrationResult(
        calibrated_parameters=parameter_result,
        objective_value=max(objective.scalar(best.vector), 0.0),
        cost=best.cost,
        optimality=best.optimality,
        weighted_rmse=metrics.weighted_rmse,
        rmse=metrics.rmse,
        mae=metrics.mae,
        max_abs_error=metrics.max_abs_error,
        target=policy.target,
        weighting=policy.weighting,
        optimizer=policy.optimizer,
        converged=best.converged and not final_evaluation_failed,
        status=best.status,
        message=result_message,
        iterations=best.iterations,
        evaluations=objective.objective_evaluations,
        jacobian_evaluations=best.jacobian_evaluations,
        pricing_evaluations=objective.pricing_evaluations,
        failed_evaluations=objective.failed_evaluations,
        cache_hits=objective.cache_hits,
        runtime_ms=elapsed_ms,
        prices_per_second=(
            objective.pricing_evaluations / (elapsed_ms / 1_000.0) if elapsed_ms > 0.0 else 0.0
        ),
        feller_satisfied=parameter_result.feller_satisfied,
        feller_ratio=parameter_result.feller_ratio,
        penalty_value=objective.feller_penalty(parameter_result),
        regularization_value=objective.regularization(best.vector),
        observations_used=len(data.observations),
        observations_excluded=len(data.exclusions),
        exclusion_reasons=data.exclusion_reasons,
        initial_parameters=vector_to_heston_parameters(best.initial),
        best_start_index=best_index,
        active_bounds=_active_bounds(best.vector, policy),
        residual_summary=metrics,
        residuals=returned_residuals,
        exclusions=list(data.exclusions[: policy.max_returned_residuals]),
        per_expiry=expiry,
        per_moneyness=moneyness,
        identifiability=identifiability,
        starts=returned_starts,
    )


def evaluate_heston_calibration(
    dataset: HestonCalibrationDataset,
    parameters: HestonParameters,
    config: HestonCalibrationConfig | None = None,
) -> HestonCalibrationDiagnosticsResult:
    """Evaluate a fixed parameter set with the same target, weights, and filters as calibration."""

    policy = config or HestonCalibrationConfig()
    if not parameters_within_bounds(parameters, policy.bounds):
        raise HestonCalibrationError("diagnostic Heston parameters lie outside calibration bounds")
    data = prepare_calibration_data(dataset, policy)
    objective = HestonCalibrationObjective(data, policy)
    vector = heston_parameters_to_vector(parameters)
    _, residuals, metrics, expiry, moneyness, identifiability = _diagnostics(
        data, objective, vector, policy
    )
    return HestonCalibrationDiagnosticsResult(
        parameters=parameters,
        target=policy.target,
        weighting=policy.weighting,
        metrics=metrics,
        penalty_value=objective.feller_penalty(parameters),
        residuals=residuals[: policy.max_returned_residuals],
        per_expiry=expiry,
        per_moneyness=moneyness,
        identifiability=identifiability,
        observations_used=len(data.observations),
        observations_excluded=len(data.exclusions),
        exclusions=list(data.exclusions[: policy.max_returned_residuals]),
        failed_evaluations=objective.failed_evaluations,
    )


def _fitted_points(
    data: PreparedCalibrationData,
    evaluation: _ModelEvaluation,
) -> list[HestonFittedSurfacePoint]:
    points = []
    for item, model_price, model_iv in zip(
        data.observations,
        evaluation.model_prices,
        evaluation.model_implied_volatilities,
        strict=True,
    ):
        if model_price is None:
            continue
        if model_iv is None:
            try:
                model_iv = solve_implied_volatility(
                    IVObservation(
                        contract=item.contract,
                        market=ImpliedVolatilityMarket(
                            risk_free_rate=data.dataset.risk_free_rate,
                            dividend_yield=data.dataset.dividend_yield,
                        ),
                        observed_price=model_price,
                    )
                ).implied_volatility
            except QuantitativeError:
                model_iv = None
        points.append(
            HestonFittedSurfacePoint(
                strike=item.observation.strike,
                maturity_years=item.observation.maturity_years,
                option_type=item.observation.option_type,
                market_price=item.market_price,
                heston_price=model_price,
                market_implied_volatility=item.market_implied_volatility,
                heston_implied_volatility=model_iv,
                price_residual=model_price - item.market_price,
                implied_volatility_residual=(
                    model_iv - item.market_implied_volatility if model_iv is not None else None
                ),
                moneyness=item.moneyness,
                log_moneyness=item.log_moneyness,
            )
        )
    return sorted(points, key=lambda point: (point.maturity_years, point.strike, point.option_type))


def _atm_fit(points: list[HestonFittedSurfacePoint]) -> list[HestonATMFitPoint]:
    result = []
    for maturity in sorted({point.maturity_years for point in points}):
        selected = min(
            (point for point in points if point.maturity_years == maturity),
            key=lambda point: abs(point.log_moneyness),
        )
        result.append(
            HestonATMFitPoint(
                maturity_years=maturity,
                strike=selected.strike,
                market_implied_volatility=selected.market_implied_volatility,
                heston_implied_volatility=selected.heston_implied_volatility,
                residual=selected.implied_volatility_residual,
            )
        )
    return result


def _skew_fit(points: list[HestonFittedSurfacePoint]) -> list[HestonSkewFitPoint]:
    result = []
    for maturity in sorted({point.maturity_years for point in points}):
        selected = [
            point
            for point in points
            if point.maturity_years == maturity
            and point.market_implied_volatility is not None
            and point.heston_implied_volatility is not None
        ]
        market_skew: float | None = None
        heston_skew: float | None = None
        if len(selected) >= 2 and len({point.log_moneyness for point in selected}) >= 2:
            coordinates = np.asarray([point.log_moneyness for point in selected])
            market_vols = np.asarray(
                [point.market_implied_volatility for point in selected], dtype=np.float64
            )
            model_vols = np.asarray(
                [point.heston_implied_volatility for point in selected], dtype=np.float64
            )
            market_skew = float(np.polyfit(coordinates, market_vols, 1)[0])
            heston_skew = float(np.polyfit(coordinates, model_vols, 1)[0])
        result.append(
            HestonSkewFitPoint(
                maturity_years=maturity,
                market_skew=market_skew,
                heston_skew=heston_skew,
                residual=(
                    heston_skew - market_skew
                    if heston_skew is not None and market_skew is not None
                    else None
                ),
                observation_count=len(selected),
            )
        )
    return result


def generate_fitted_heston_surface(
    dataset: HestonCalibrationDataset,
    parameters: HestonParameters,
    config: HestonCalibrationConfig | None = None,
) -> HestonFittedSurfaceResult:
    """Return market-versus-model points without replacing the Phase 6 surface."""

    policy = config or HestonCalibrationConfig()
    if not parameters_within_bounds(parameters, policy.bounds):
        raise HestonCalibrationError(
            "fitted-surface Heston parameters lie outside calibration bounds"
        )
    data = prepare_calibration_data(dataset, policy)
    objective = HestonCalibrationObjective(data, policy)
    evaluation = objective.evaluate(heston_parameters_to_vector(parameters))
    points = _fitted_points(data, evaluation)
    residuals = _residual_models(data, evaluation, policy.target)
    return HestonFittedSurfaceResult(
        parameters=parameters,
        points=points,
        atm_term_structure=_atm_fit(points),
        skew_by_expiry=_skew_fit(points),
        metrics=_metrics(residuals),
        observations_excluded=len(data.exclusions),
        exclusions=list(data.exclusions[: policy.max_returned_residuals]),
    )
