"""Heston comparison, convergence, variance, smile, and sensitivity research."""

import math
from collections.abc import Sequence
from time import perf_counter

import numpy as np

from backend.app.core.exceptions import HestonPricingError, QuantitativeError
from backend.app.models.heston import (
    HestonComparisonPoint,
    HestonComparisonResult,
    HestonConvergenceMode,
    HestonConvergencePoint,
    HestonConvergenceResult,
    HestonDriverCorrelationResult,
    HestonIntegrationConfig,
    HestonParameterName,
    HestonParameters,
    HestonSensitivityPoint,
    HestonSensitivityResult,
    HestonSimulationConfig,
    HestonSimulationScheme,
    HestonSmilePoint,
    HestonSmileResult,
    HestonVarianceDiagnosticsResult,
    HestonVarianceTimePoint,
    require_finite_parameter_value,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import ExerciseStyle, OptionContract, OptionType
from backend.app.models.volatility import (
    ImpliedVolatilityConfig,
    ImpliedVolatilityMarket,
    ImpliedVolatilityObservation,
    ImpliedVolatilityStatus,
)
from backend.app.pricing.heston import heston_monte_carlo_price
from backend.app.pricing.heston_analytical import heston_analytical_price
from backend.app.pricing.implied_volatility import (
    forward_price,
    implied_volatility_failure_status,
    solve_implied_volatility,
)
from backend.app.simulation.heston import (
    correlated_heston_normals,
    heston_step_drivers,
    simulate_heston_paths,
    theoretical_variance_mean,
)
from backend.app.simulation.random import create_rng, spawn_child_seeds


def empirical_heston_driver_correlation(
    rho: float, num_simulations: int, seed: int
) -> HestonDriverCorrelationResult:
    """Measure the sample correlation of constructed spot and variance drivers."""

    if not math.isfinite(rho) or not -1.0 <= rho <= 1.0:
        raise HestonPricingError("Heston driver rho must lie in [-1, 1]")
    if num_simulations < 2 or seed < 0:
        raise HestonPricingError("driver diagnostics require at least two paths and a valid seed")
    variance, perpendicular, _ = heston_step_drivers(
        num_simulations, create_rng(seed), antithetic=False, include_uniform=False
    )
    spot = correlated_heston_normals(variance, perpendicular, rho)
    empirical = float(np.corrcoef(spot, variance)[0, 1])
    return HestonDriverCorrelationResult(
        target_correlation=rho,
        empirical_correlation=empirical,
        absolute_error=abs(empirical - rho),
        num_simulations=num_simulations,
        seed=seed,
    )


def compare_heston_schemes(
    contract: OptionContract,
    market: MarketEnvironment,
    parameters: HestonParameters,
    config: HestonSimulationConfig,
    integration_config: HestonIntegrationConfig | None = None,
) -> HestonComparisonResult:
    """Compare FTE and QE-M against one semi-analytical reference."""

    analytical = heston_analytical_price(contract, market, parameters, integration_config)
    points = []
    for scheme in HestonSimulationScheme:
        result = heston_monte_carlo_price(
            contract,
            market,
            parameters,
            config.model_copy(update={"scheme": scheme}),
        )
        error = result.price - analytical.price
        points.append(
            HestonComparisonPoint(
                scheme=scheme,
                price=result.price,
                standard_error=result.standard_error,
                analytical_price=analytical.price,
                absolute_error=abs(error),
                relative_error=(abs(error) / analytical.price if analytical.price > 0.0 else None),
                z_score=(error / result.standard_error if result.standard_error > 0.0 else None),
                runtime_ms=result.runtime_ms,
                mean_terminal_variance=result.mean_terminal_variance,
                negative_variance_proposal_fraction=(result.negative_variance_proposal_fraction),
            )
        )
    return HestonComparisonResult(
        analytical=analytical,
        points=points,
        feller_satisfied=parameters.feller_satisfied,
        feller_ratio=parameters.feller_ratio,
    )


def heston_convergence_configs(
    config: HestonSimulationConfig,
    values: Sequence[int],
    mode: HestonConvergenceMode,
) -> list[HestonSimulationConfig]:
    if not values or any(value <= 0 for value in values) or len(set(values)) != len(values):
        raise HestonPricingError("Heston convergence values must be nonempty, positive, and unique")
    sorted_values = sorted(values)
    if mode is HestonConvergenceMode.PATH_COUNT:
        seeds = spawn_child_seeds(config.seed, len(sorted_values))
        return [
            config.model_copy(update={"num_simulations": value, "seed": seed})
            for value, seed in zip(sorted_values, seeds, strict=True)
        ]
    return [config.model_copy(update={"num_time_steps": value}) for value in sorted_values]


def run_heston_convergence(
    contract: OptionContract,
    market: MarketEnvironment,
    parameters: HestonParameters,
    config: HestonSimulationConfig,
    values: Sequence[int],
    mode: HestonConvergenceMode,
    integration_config: HestonIntegrationConfig | None = None,
) -> HestonConvergenceResult:
    analytical = heston_analytical_price(contract, market, parameters, integration_config)
    points = []
    for point_config in heston_convergence_configs(config, values, mode):
        result = heston_monte_carlo_price(contract, market, parameters, point_config)
        bias = result.price - analytical.price
        points.append(
            HestonConvergencePoint(
                num_simulations=point_config.num_simulations,
                num_time_steps=point_config.num_time_steps,
                seed=point_config.seed,
                scheme=point_config.scheme,
                price=result.price,
                standard_error=result.standard_error,
                analytical_price=analytical.price,
                bias=bias,
                absolute_error=abs(bias),
                runtime_ms=result.runtime_ms,
            )
        )
    return HestonConvergenceResult(
        mode=mode,
        root_seed=config.seed,
        analytical_price=analytical.price,
        points=points,
    )


def heston_variance_diagnostics(
    contract: OptionContract,
    market: MarketEnvironment,
    parameters: HestonParameters,
    config: HestonSimulationConfig,
    *,
    near_zero_tolerance: float = 1e-8,
) -> HestonVarianceDiagnosticsResult:
    """Summarize full variance paths without returning path matrices."""

    if near_zero_tolerance < 0.0 or not math.isfinite(near_zero_tolerance):
        raise HestonPricingError("near-zero tolerance must be finite and non-negative")
    started = perf_counter()
    paths = simulate_heston_paths(
        contract,
        market,
        parameters,
        config,
        create_rng(config.seed, config.random_engine),
    )
    times = np.linspace(0.0, contract.maturity_years, config.num_time_steps + 1, dtype=np.float64)
    points = [
        HestonVarianceTimePoint(
            time_years=float(time),
            simulated_mean_variance=float(np.mean(paths.variance_paths[:, index])),
            theoretical_mean_variance=theoretical_variance_mean(parameters, float(time)),
            fraction_near_zero=float(
                np.mean(np.maximum(paths.variance_paths[:, index], 0.0) <= near_zero_tolerance)
            ),
        )
        for index, time in enumerate(times)
    ]
    terminal = paths.variance_paths[:, -1]
    quantiles = np.quantile(terminal, [0.01, 0.05, 0.5, 0.95, 0.99])
    return HestonVarianceDiagnosticsResult(
        scheme=config.scheme,
        num_simulations=config.num_simulations,
        num_time_steps=config.num_time_steps,
        seed=config.seed,
        points=points,
        terminal_variance_mean=float(np.mean(terminal)),
        # This is a descriptive distribution summary rather than an estimator
        # uncertainty, so the population convention remains defined for one path.
        terminal_variance_std=float(np.std(terminal, ddof=0)),
        terminal_variance_min=float(np.min(terminal)),
        terminal_variance_max=float(np.max(terminal)),
        terminal_variance_quantiles={
            label: float(value)
            for label, value in zip(("p01", "p05", "p50", "p95", "p99"), quantiles, strict=True)
        },
        negative_variance_proposal_fraction=(paths.negative_variance_proposal_fraction),
        feller_satisfied=parameters.feller_satisfied,
        feller_ratio=parameters.feller_ratio,
        runtime_ms=(perf_counter() - started) * 1_000.0,
    )


def _implied_volatility_for_price(
    contract: OptionContract,
    market: MarketEnvironment,
    price: float,
    solver_config: ImpliedVolatilityConfig,
) -> tuple[float | None, ImpliedVolatilityStatus]:
    observation = ImpliedVolatilityObservation(
        contract=contract,
        market=ImpliedVolatilityMarket(
            risk_free_rate=market.risk_free_rate,
            dividend_yield=market.dividend_yield,
        ),
        observed_price=price,
    )
    try:
        result = solve_implied_volatility(observation, solver_config)
    except QuantitativeError as exc:
        return None, implied_volatility_failure_status(exc)
    return result.implied_volatility, result.status


def heston_implied_volatility_smile(
    contract: OptionContract,
    market: MarketEnvironment,
    parameters: HestonParameters,
    strikes: Sequence[float],
    integration_config: HestonIntegrationConfig | None = None,
    solver_config: ImpliedVolatilityConfig | None = None,
) -> HestonSmileResult:
    """Generate a call-price and Black-Scholes implied-volatility strike slice."""

    if (
        not strikes
        or len(set(strikes)) != len(strikes)
        or any(not math.isfinite(strike) or strike <= 0.0 for strike in strikes)
    ):
        raise HestonPricingError("Heston smile strikes must be positive, finite, and unique")
    solver = solver_config or ImpliedVolatilityConfig()
    forward = forward_price(
        contract.spot,
        contract.maturity_years,
        market.risk_free_rate,
        market.dividend_yield,
    )
    points = []
    for strike in sorted(strikes):
        point_contract = contract.model_copy(
            update={
                "option_type": OptionType.CALL,
                "exercise_style": ExerciseStyle.EUROPEAN,
                "strike": strike,
            }
        )
        price = heston_analytical_price(
            point_contract, market, parameters, integration_config
        ).price
        implied, status = _implied_volatility_for_price(point_contract, market, price, solver)
        points.append(
            HestonSmilePoint(
                strike=strike,
                moneyness=strike / contract.spot,
                log_moneyness=math.log(strike / forward),
                heston_price=price,
                implied_volatility=implied,
                inversion_status=status,
            )
        )
    return HestonSmileResult(
        maturity_years=contract.maturity_years,
        points=points,
        feller_satisfied=parameters.feller_satisfied,
        feller_ratio=parameters.feller_ratio,
    )


def _updated_parameters(
    parameters: HestonParameters, name: HestonParameterName, value: float
) -> HestonParameters:
    require_finite_parameter_value(name, value)
    if name is HestonParameterName.RHO:
        if not -1.0 <= value <= 1.0:
            raise HestonPricingError("rho sensitivity values must lie in [-1, 1]")
    elif value < 0.0:
        raise HestonPricingError(f"{name.value} sensitivity values must be non-negative")
    data = parameters.model_dump(exclude={"feller_satisfied", "feller_ratio"})
    data[name.value] = value
    return HestonParameters.model_validate(data)


def heston_parameter_sensitivity(
    contract: OptionContract,
    market: MarketEnvironment,
    parameters: HestonParameters,
    parameter: HestonParameterName,
    values: Sequence[float],
    integration_config: HestonIntegrationConfig | None = None,
    solver_config: ImpliedVolatilityConfig | None = None,
) -> HestonSensitivityResult:
    """Sweep one risk-neutral Heston parameter without fitting it to observations."""

    if not values or len(set(values)) != len(values):
        raise HestonPricingError("Heston sensitivity values must be nonempty and unique")
    solver = solver_config or ImpliedVolatilityConfig()
    points = []
    for value in sorted(values):
        scenario = _updated_parameters(parameters, parameter, value)
        started = perf_counter()
        price = heston_analytical_price(contract, market, scenario, integration_config).price
        implied, status = _implied_volatility_for_price(contract, market, price, solver)
        points.append(
            HestonSensitivityPoint(
                parameter_value=value,
                price=price,
                implied_volatility=implied,
                inversion_status=status,
                runtime_ms=(perf_counter() - started) * 1_000.0,
            )
        )
    return HestonSensitivityResult(
        parameter=parameter,
        base_value=float(getattr(parameters, parameter.value)),
        points=points,
    )
