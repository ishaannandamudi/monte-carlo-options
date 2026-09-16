"""Longstaff-Schwartz fitting and independent evaluation of American vanilla policies."""

import logging
import math
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from backend.app.analytics.statistics import (
    antithetic_standard_error,
    confidence_interval,
    sample_mean,
    standard_error,
)
from backend.app.core.exceptions import LongstaffSchwartzError, NumericalError
from backend.app.core.numerics import finite_array_exp, require_finite
from backend.app.models.longstaff_schwartz import (
    LongstaffSchwartzConfig,
    LongstaffSchwartzResult,
    LSMBoundaryPoint,
    LSMEvaluationMode,
    LSMPolicy,
    LSMRegressionDiagnostic,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import ExerciseStyle, OptionContract, OptionType
from backend.app.models.simulation import PathSimulationConfig, VarianceReduction
from backend.app.payoffs.vanilla import intrinsic_value, vanilla_payoff
from backend.app.pricing.base import Pricer
from backend.app.pricing.lsm_regression import fit_continuation, predict_continuation
from backend.app.simulation.base import PathStochasticProcess
from backend.app.simulation.gbm import GBMProcess
from backend.app.simulation.random import create_rng, spawn_child_seeds

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LSMCashflows:
    """Cash flows in exercise-date dollars and final selected stopping indices."""

    cashflows: NDArray[np.float64]
    exercise_indices: NDArray[np.int64]


@dataclass(frozen=True)
class LSMTrainingResult:
    policy: LSMPolicy
    sample: LSMCashflows


def _require_american(contract: OptionContract) -> None:
    if contract.exercise_style is not ExerciseStyle.AMERICAN:
        raise LongstaffSchwartzError("Longstaff-Schwartz requires AMERICAN exercise style")


def _validate_paths(paths: NDArray[np.float64], contract: OptionContract, steps: int) -> None:
    if paths.ndim != 2 or paths.shape[0] == 0 or paths.shape[1] != steps + 1:
        raise LongstaffSchwartzError("path matrix must have shape (paths, num_time_steps + 1)")
    require_finite(paths, context="LSM paths")
    if np.any(paths <= 0.0) or not np.all(paths[:, 0] == contract.spot):
        raise LongstaffSchwartzError("LSM paths must be positive and start at the contract spot")


def discounted_cashflows(
    sample: LSMCashflows, *, rate: float, dt: float, current_index: int = 0
) -> NDArray[np.float64]:
    """Discount once from each selected exercise date to the requested earlier date."""

    if (
        sample.cashflows.ndim != 1
        or sample.exercise_indices.shape != sample.cashflows.shape
        or np.any(sample.exercise_indices < current_index)
    ):
        raise LongstaffSchwartzError("cash-flow times must match and follow the valuation date")
    discounts = finite_array_exp(
        np.asarray(-rate * (sample.exercise_indices - current_index) * dt, dtype=np.float64),
        context="LSM cash-flow discount",
    )
    with np.errstate(over="ignore", invalid="ignore"):
        values = sample.cashflows * discounts
    require_finite(values, context="LSM discounted realized cash flows")
    return values


def _no_call_exercise(contract: OptionContract, market: MarketEnvironment) -> bool:
    return (
        contract.option_type is OptionType.CALL
        and market.dividend_yield == 0.0
        and market.risk_free_rate >= 0.0
    )


def fit_exercise_policy(
    paths: NDArray[np.float64],
    contract: OptionContract,
    market: MarketEnvironment,
    config: LongstaffSchwartzConfig,
) -> LSMTrainingResult:
    """Fit backward on fixed paths; a later chosen stop remains eligible for replacement."""

    _require_american(contract)
    _validate_paths(paths, contract, config.num_time_steps)
    if contract.maturity_years == 0.0 or market.volatility == 0.0:
        raise LongstaffSchwartzError("fixed-path fitting requires positive maturity and volatility")
    dt = contract.maturity_years / config.num_time_steps
    if dt == 0.0:
        raise NumericalError("LSM time step underflowed")
    cashflows = vanilla_payoff(paths[:, -1], contract)
    stops = np.full(paths.shape[0], config.num_time_steps, dtype=np.int64)
    dates: list[LSMRegressionDiagnostic] = []
    theorem = _no_call_exercise(contract, market)
    if not theorem:
        for step in reversed(config.schedule[:-1]):
            intrinsic = vanilla_payoff(paths[:, step], contract)
            eligible = np.flatnonzero((intrinsic > 0.0) & (stops > step))
            selected = LSMCashflows(cashflows[eligible], stops[eligible])
            targets = discounted_cashflows(
                selected, rate=market.risk_free_rate, dt=dt, current_index=step
            )
            date = fit_continuation(
                paths[eligible, step],
                targets,
                contract.strike,
                config,
                time_index=step,
                time=step * dt,
            )
            dates.append(date)
            if date.fallback_used:
                continue
            continuation = predict_continuation(paths[eligible, step], contract.strike, date)
            exercise = eligible[intrinsic[eligible] > continuation + config.exercise_tolerance]
            cashflows[exercise] = intrinsic[exercise]
            stops[exercise] = step
    values = discounted_cashflows(LSMCashflows(cashflows, stops), rate=market.risk_free_rate, dt=dt)
    continuation_price = sample_mean(values)
    if not math.isfinite(continuation_price):
        raise NumericalError("LSM training continuation mean is non-finite")
    root_intrinsic = intrinsic_value(contract.option_type, contract.spot, contract.strike)
    immediate = not theorem and root_intrinsic > continuation_price
    if immediate:
        cashflows.fill(root_intrinsic)
        stops.fill(0)
    policy = LSMPolicy(
        contract=contract,
        market=market,
        num_time_steps=config.num_time_steps,
        exercise_steps=config.schedule,
        exercise_tolerance=config.exercise_tolerance,
        dates=tuple(reversed(dates)),
        immediate_exercise=immediate,
        training_continuation_price=continuation_price,
        no_early_exercise_theorem=theorem,
    )
    return LSMTrainingResult(policy, LSMCashflows(cashflows, stops))


def apply_exercise_policy(paths: NDArray[np.float64], policy: LSMPolicy) -> LSMCashflows:
    """Apply frozen dates chronologically, including the root decision fixed in training."""

    contract = policy.contract
    _require_american(contract)
    _validate_paths(paths, contract, policy.num_time_steps)
    cashflows = vanilla_payoff(paths[:, -1], contract)
    stops = np.full(paths.shape[0], policy.num_time_steps, dtype=np.int64)
    if policy.immediate_exercise:
        cashflows.fill(intrinsic_value(contract.option_type, contract.spot, contract.strike))
        stops.fill(0)
        return LSMCashflows(cashflows, stops)
    alive = np.ones(paths.shape[0], dtype=np.bool_)
    for date in policy.dates:
        step = date.time_index
        if date.fallback_used:
            continue
        intrinsic = vanilla_payoff(paths[:, step], contract)
        eligible = np.flatnonzero(alive & (intrinsic > 0.0))
        if eligible.size == 0:
            continue
        continuation = predict_continuation(paths[eligible, step], contract.strike, date)
        exercise = eligible[intrinsic[eligible] > continuation + policy.exercise_tolerance]
        cashflows[exercise] = intrinsic[exercise]
        stops[exercise] = step
        alive[exercise] = False
    return LSMCashflows(cashflows, stops)


def empirical_exercise_boundary(
    paths: NDArray[np.float64],
    sample: LSMCashflows,
    contract: OptionContract,
    *,
    min_observations: int,
) -> tuple[LSMBoundaryPoint, ...]:
    """Summarize final realized early stops, excluding zero payoffs and maturity."""

    if paths.ndim != 2 or paths.shape[1] < 2:
        raise LongstaffSchwartzError("boundary paths require a nonempty exercise grid")
    steps = paths.shape[1] - 1
    _validate_paths(paths, contract, steps)
    if (
        min_observations <= 0
        or sample.cashflows.shape != (paths.shape[0],)
        or sample.exercise_indices.shape != sample.cashflows.shape
        or np.any(sample.exercise_indices < 0)
        or np.any(sample.exercise_indices > steps)
    ):
        raise LongstaffSchwartzError("invalid boundary support or cash-flow sample")
    require_finite(sample.cashflows, context="LSM boundary cash flows")
    points = []
    early = (sample.exercise_indices < steps) & (sample.cashflows > 0.0)
    for step in np.unique(sample.exercise_indices[early]):
        mask = early & (sample.exercise_indices == step)
        count = int(np.count_nonzero(mask))
        if count < min_observations or contract.maturity_years == 0.0:
            continue
        stocks = paths[mask, int(step)]
        is_put = contract.option_type is OptionType.PUT
        points.append(
            LSMBoundaryPoint(
                time_index=int(step),
                time=float(step) * contract.maturity_years / steps,
                time_to_maturity=(steps - int(step)) * (contract.maturity_years / steps),
                estimated_boundary_stock=float(np.max(stocks) if is_put else np.min(stocks)),
                option_type=contract.option_type,
                exercise_region_direction="LOW_STOCK" if is_put else "HIGH_STOCK",
                exercise_observations=count,
                sample_support=count / paths.shape[0],
            )
        )
    return tuple(points)


def _deterministic_sample(
    contract: OptionContract, market: MarketEnvironment, config: LongstaffSchwartzConfig
) -> tuple[LSMCashflows, LSMPolicy]:
    if contract.maturity_years == 0.0:
        intrinsic = intrinsic_value(contract.option_type, contract.spot, contract.strike)
        # Expiry has no rate/dividend arithmetic or path-array allocation.
        return LSMCashflows(
            np.array([intrinsic]), np.array([config.num_time_steps], dtype=np.int64)
        ), LSMPolicy(
            contract=contract,
            market=market,
            num_time_steps=config.num_time_steps,
            exercise_steps=config.schedule,
            exercise_tolerance=config.exercise_tolerance,
            dates=(),
            immediate_exercise=False,
            training_continuation_price=intrinsic,
            no_early_exercise_theorem=_no_call_exercise(contract, market),
        )
    indices = np.asarray((0, *config.schedule), dtype=np.int64)
    times = np.asarray(
        indices * (contract.maturity_years / config.num_time_steps), dtype=np.float64
    )
    stocks = contract.spot * finite_array_exp(
        (market.risk_free_rate - market.dividend_yield) * times,
        context="deterministic LSM path",
    )
    if np.any(stocks <= 0.0):
        raise NumericalError("deterministic LSM stocks underflowed")
    cashflows = vanilla_payoff(stocks, contract)
    present_values = discounted_cashflows(
        LSMCashflows(cashflows, indices),
        rate=market.risk_free_rate,
        dt=contract.maturity_years / config.num_time_steps,
    )
    best = int(np.argmax(present_values)) if np.any(present_values > 0.0) else len(indices) - 1
    sample = LSMCashflows(cashflows[best : best + 1], indices[best : best + 1])
    policy = LSMPolicy(
        contract=contract,
        market=market,
        num_time_steps=config.num_time_steps,
        exercise_steps=config.schedule,
        exercise_tolerance=config.exercise_tolerance,
        dates=(),
        immediate_exercise=best == 0,
        training_continuation_price=float(np.max(present_values[1:])),
        no_early_exercise_theorem=_no_call_exercise(contract, market),
    )
    return sample, policy


def _result(
    sample: LSMCashflows,
    policy: LSMPolicy,
    config: LongstaffSchwartzConfig,
    *,
    started: float,
    simulation_ms: float,
    regression_ms: float,
    valuation_ms: float,
    boundary: tuple[LSMBoundaryPoint, ...],
    deterministic: bool,
    training_seed: int | None,
    valuation_seed: int | None,
) -> LongstaffSchwartzResult:
    contract, market = policy.contract, policy.market
    values = (
        sample.cashflows
        if contract.maturity_years == 0.0
        else discounted_cashflows(
            sample, rate=market.risk_free_rate, dt=contract.maturity_years / config.num_time_steps
        )
    )
    price = sample_mean(values)
    if not math.isfinite(price):
        raise NumericalError("LSM estimate is non-finite")
    if deterministic or policy.immediate_exercise:
        se = 0.0
    elif config.variance_reduction is VarianceReduction.ANTITHETIC:
        se = antithetic_standard_error(values)
    else:
        se = standard_error(values)
    low, high = confidence_interval(price, se, config.confidence_level)
    positive = sample.cashflows > 0.0
    early = positive & (sample.exercise_indices < config.num_time_steps)
    times = sample.exercise_indices[positive] * (contract.maturity_years / config.num_time_steps)
    conditions = [
        date.condition_number for date in policy.dates if date.condition_number is not None
    ]
    root_intrinsic = intrinsic_value(contract.option_type, contract.spot, contract.strike)
    cap = config.max_diagnostic_dates
    diagnostics = policy.dates[:cap] if config.store_regression_diagnostics else ()
    returned_boundary = boundary[:cap] if config.store_boundary_diagnostics else ()
    note = (
        "Conditional sampling SE for the frozen policy, including its training-selected root. "
        "Excludes training, basis, exercise-grid and model bias; it is not an optimal-value CI."
        if config.evaluation_mode is LSMEvaluationMode.OUT_OF_SAMPLE
        else "Descriptive cash-flow SE: shared in-sample regression and root selection invalidate "
        "independent-policy inference; regression bias and exercise-grid error are excluded."
    )
    return LongstaffSchwartzResult(
        price=price,
        standard_error=se,
        estimator_variance=se**2,
        confidence_interval_low=low,
        confidence_interval_high=high,
        confidence_level=config.confidence_level,
        num_paths=config.num_paths,
        regression_paths=config.training_path_count,
        num_time_steps=config.num_time_steps,
        seed=config.seed,
        training_seed=training_seed,
        valuation_seed=valuation_seed,
        option_type=contract.option_type,
        variance_reduction=config.variance_reduction,
        evaluation_mode=config.evaluation_mode,
        basis_family=config.basis_family,
        basis_degree=config.basis_degree,
        runtime_ms=(perf_counter() - started) * 1_000.0,
        simulation_runtime_ms=simulation_ms,
        regression_runtime_ms=regression_ms,
        valuation_runtime_ms=valuation_ms,
        approximate_path_array_bytes=(
            0
            if deterministic
            else 8 * max(config.training_path_count, config.num_paths) * (config.num_time_steps + 1)
        ),
        early_exercise_occurred=bool(np.any(early)) and contract.maturity_years > 0.0,
        immediate_exercise=policy.immediate_exercise,
        immediate_exercise_fraction=float(np.mean(positive & (sample.exercise_indices == 0))),
        early_exercise_fraction=float(np.mean(early)),
        maturity_exercise_fraction=float(
            np.mean(positive & (sample.exercise_indices == config.num_time_steps))
        ),
        zero_cashflow_fraction=float(np.mean(~positive)),
        mean_exercise_time=float(np.mean(times)) if times.size else None,
        median_exercise_time=float(np.median(times)) if times.size else None,
        regression_count=sum(not date.fallback_used for date in policy.dates),
        regression_fallback_count=sum(date.fallback_used for date in policy.dates),
        condition_warning_count=sum(date.condition_warning for date in policy.dates),
        max_condition_number=max(conditions) if conditions else None,
        training_continuation_price=policy.training_continuation_price,
        intrinsic_value=root_intrinsic,
        below_intrinsic=price < root_intrinsic,
        deterministic=deterministic,
        no_early_exercise_theorem=policy.no_early_exercise_theorem,
        regression_diagnostics=diagnostics,
        boundary_diagnostics=returned_boundary,
        regression_diagnostics_truncated=config.store_regression_diagnostics
        and len(policy.dates) > cap,
        boundary_diagnostics_truncated=config.store_boundary_diagnostics and len(boundary) > cap,
        uncertainty_note=note,
    )


class LongstaffSchwartzPricer(Pricer[LongstaffSchwartzConfig, LongstaffSchwartzResult]):
    """American GBM Monte Carlo with optional independent policy valuation."""

    def __init__(self, process: PathStochasticProcess | None = None) -> None:
        self._process = process or GBMProcess()

    def price(
        self,
        contract: OptionContract,
        market: MarketEnvironment,
        config: LongstaffSchwartzConfig,
    ) -> LongstaffSchwartzResult:
        _require_american(contract)
        started = perf_counter()
        logger.info(
            "lsm_start paths=%d steps=%d mode=%s",
            config.num_paths,
            config.num_time_steps,
            config.evaluation_mode,
        )
        deterministic = contract.maturity_years == 0.0 or market.volatility == 0.0
        if contract.maturity_years > 0.0 and contract.maturity_years / config.num_time_steps == 0.0:
            raise NumericalError("LSM time step underflowed")
        if deterministic:
            sample, policy = _deterministic_sample(contract, market, config)
            return _result(
                sample,
                policy,
                config,
                started=started,
                simulation_ms=0.0,
                regression_ms=0.0,
                valuation_ms=0.0,
                boundary=(),
                deterministic=True,
                training_seed=None,
                valuation_seed=None,
            )
        training_seed, valuation_seed = spawn_child_seeds(config.seed, 2)

        def simulate(count: int, seed: int) -> NDArray[np.float64]:
            path_config = PathSimulationConfig(
                num_simulations=count,
                num_time_steps=config.num_time_steps,
                seed=seed,
                variance_reduction=config.variance_reduction,
                confidence_level=config.confidence_level,
            )
            paths = self._process.simulate_paths(contract, market, path_config, create_rng(seed))
            _validate_paths(paths, contract, config.num_time_steps)
            if paths.shape[0] != count:
                raise LongstaffSchwartzError("process returned the wrong number of paths")
            return paths

        clock = perf_counter()
        paths = simulate(config.training_path_count, training_seed)
        simulation_ms = (perf_counter() - clock) * 1_000.0
        clock = perf_counter()
        training = fit_exercise_policy(paths, contract, market, config)
        regression_ms = (perf_counter() - clock) * 1_000.0
        policy = training.policy
        sample = training.sample
        valuation_ms = 0.0
        if config.evaluation_mode is LSMEvaluationMode.OUT_OF_SAMPLE:
            del paths
            clock = perf_counter()
            paths = simulate(config.num_paths, valuation_seed)
            simulation_ms += (perf_counter() - clock) * 1_000.0
            clock = perf_counter()
            sample = apply_exercise_policy(paths, policy)
            valuation_ms = (perf_counter() - clock) * 1_000.0
        else:
            valuation_seed = training_seed
        boundary = (
            empirical_exercise_boundary(
                paths, sample, contract, min_observations=config.min_boundary_observations
            )
            if config.store_boundary_diagnostics
            else ()
        )
        result = _result(
            sample,
            policy,
            config,
            started=started,
            simulation_ms=simulation_ms,
            regression_ms=regression_ms,
            valuation_ms=valuation_ms,
            boundary=boundary,
            deterministic=False,
            training_seed=training_seed,
            valuation_seed=valuation_seed,
        )
        logger.info(
            "lsm_complete price=%g se=%g regressions=%d fallbacks=%d runtime_ms=%.3f",
            result.price,
            result.standard_error,
            result.regression_count,
            result.regression_fallback_count,
            result.runtime_ms,
        )
        return result


def longstaff_schwartz_price(
    contract: OptionContract,
    market: MarketEnvironment,
    config: LongstaffSchwartzConfig | None = None,
) -> LongstaffSchwartzResult:
    return LongstaffSchwartzPricer().price(contract, market, config or LongstaffSchwartzConfig())
