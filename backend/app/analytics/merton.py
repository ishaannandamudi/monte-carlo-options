"""Merton comparison, convergence, smile, sensitivity, and jump diagnostics."""

import math
from collections.abc import Sequence
from time import perf_counter

import numpy as np

from backend.app.core.exceptions import MertonPricingError, QuantitativeError
from backend.app.models.heston import HestonIntegrationConfig, HestonParameters
from backend.app.models.market import MarketEnvironment
from backend.app.models.merton import (
    MertonComparisonResult,
    MertonConvergencePoint,
    MertonConvergenceResult,
    MertonJumpDiagnosticsResult,
    MertonJumpParameters,
    MertonObjectiveObservation,
    MertonObjectiveResult,
    MertonParameterName,
    MertonSensitivityPoint,
    MertonSensitivityResult,
    MertonSeriesConfig,
    MertonSeriesConvergencePoint,
    MertonSeriesConvergenceResult,
    MertonSimulationConfig,
    MertonSmilePoint,
    MertonSmileResult,
)
from backend.app.models.option import ExerciseStyle, OptionContract, OptionType
from backend.app.models.volatility import (
    ImpliedVolatilityConfig,
    ImpliedVolatilityMarket,
    ImpliedVolatilityObservation,
    ImpliedVolatilityStatus,
)
from backend.app.pricing.black_scholes import black_scholes_price
from backend.app.pricing.heston_analytical import heston_analytical_price
from backend.app.pricing.implied_volatility import (
    forward_price,
    implied_volatility_failure_status,
    solve_implied_volatility,
)
from backend.app.pricing.merton import merton_analytical_price, merton_monte_carlo_price
from backend.app.simulation.merton import simulate_merton_terminal
from backend.app.simulation.random import create_rng, spawn_child_seeds


def _implied_volatility(
    contract: OptionContract,
    market: MarketEnvironment,
    price: float,
    config: ImpliedVolatilityConfig,
) -> tuple[float | None, ImpliedVolatilityStatus]:
    observation = ImpliedVolatilityObservation(
        contract=contract,
        market=ImpliedVolatilityMarket(
            risk_free_rate=market.risk_free_rate, dividend_yield=market.dividend_yield
        ),
        observed_price=price,
    )
    try:
        result = solve_implied_volatility(observation, config)
    except QuantitativeError as exc:
        return None, implied_volatility_failure_status(exc)
    return result.implied_volatility, result.status


def compare_merton(
    contract: OptionContract,
    market: MarketEnvironment,
    parameters: MertonJumpParameters,
    simulation: MertonSimulationConfig,
    series: MertonSeriesConfig | None = None,
    heston_parameters: HestonParameters | None = None,
    heston_integration: HestonIntegrationConfig | None = None,
) -> MertonComparisonResult:
    analytical = merton_analytical_price(contract, market, parameters, series)
    monte_carlo = merton_monte_carlo_price(contract, market, parameters, simulation)
    bsm = black_scholes_price(contract, market)
    heston = (
        heston_analytical_price(contract, market, heston_parameters, heston_integration).price
        if heston_parameters is not None
        else None
    )
    error = monte_carlo.price - analytical.price
    return MertonComparisonResult(
        analytical=analytical,
        monte_carlo=monte_carlo,
        black_scholes_price=bsm,
        heston_price=heston,
        signed_error=error,
        absolute_error=abs(error),
        relative_error=abs(error) / analytical.price if analytical.price > 0.0 else None,
        z_score=error / monte_carlo.standard_error if monte_carlo.standard_error > 0.0 else None,
        black_scholes_difference=analytical.price - bsm,
    )


def run_merton_convergence(
    contract: OptionContract,
    market: MarketEnvironment,
    parameters: MertonJumpParameters,
    path_counts: Sequence[int],
    root_seed: int,
    confidence_level: float = 0.95,
    series: MertonSeriesConfig | None = None,
) -> MertonConvergenceResult:
    if (
        not path_counts
        or len(set(path_counts)) != len(path_counts)
        or any(n <= 0 for n in path_counts)
    ):
        raise MertonPricingError("Merton path counts must be nonempty, positive, and unique")
    analytical = merton_analytical_price(contract, market, parameters, series)
    seeds = spawn_child_seeds(root_seed, len(path_counts))
    points = []
    for count, seed in zip(sorted(path_counts), seeds, strict=True):
        result = merton_monte_carlo_price(
            contract,
            market,
            parameters,
            MertonSimulationConfig(
                num_simulations=count, seed=seed, confidence_level=confidence_level
            ),
        )
        error = result.price - analytical.price
        points.append(
            MertonConvergencePoint(
                num_simulations=count,
                seed=seed,
                price=result.price,
                standard_error=result.standard_error,
                analytical_price=analytical.price,
                signed_error=error,
                absolute_error=abs(error),
                runtime_ms=result.runtime_ms,
            )
        )
    usable = [point for point in points if point.standard_error > 0.0]
    slope = (
        float(
            np.polyfit(
                np.log([point.num_simulations for point in usable]),
                np.log([point.standard_error for point in usable]),
                1,
            )[0]
        )
        if len(usable) >= 2
        else None
    )
    return MertonConvergenceResult(
        root_seed=root_seed,
        analytical_price=analytical.price,
        points=points,
        standard_error_slope=slope,
        slope_points_used=len(usable),
    )


def run_merton_series_convergence(
    contract: OptionContract,
    market: MarketEnvironment,
    parameters: MertonJumpParameters,
    tolerances: Sequence[float],
    max_terms: int = 10_000,
) -> MertonSeriesConvergenceResult:
    if (
        not tolerances
        or len(set(tolerances)) != len(tolerances)
        or any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in tolerances)
    ):
        raise MertonPricingError("series tolerances must be finite, unique, and in (0, 1)")
    points = []
    for tolerance in sorted(tolerances, reverse=True):
        result = merton_analytical_price(
            contract,
            market,
            parameters,
            MertonSeriesConfig(tail_probability_tolerance=tolerance, max_terms=max_terms),
        )
        points.append(
            MertonSeriesConvergencePoint(
                tolerance=tolerance,
                terms_used=result.terms_used,
                price=result.price,
                omitted_poisson_mass=result.omitted_poisson_mass,
                converged=result.converged,
                runtime_ms=result.runtime_ms,
            )
        )
    return MertonSeriesConvergenceResult(points=points)


def merton_implied_volatility_smile(
    contract: OptionContract,
    market: MarketEnvironment,
    parameters: MertonJumpParameters,
    strikes: Sequence[float],
    series: MertonSeriesConfig | None = None,
    solver: ImpliedVolatilityConfig | None = None,
) -> MertonSmileResult:
    if (
        not strikes
        or len(set(strikes)) != len(strikes)
        or any(not math.isfinite(strike) or strike <= 0.0 for strike in strikes)
    ):
        raise MertonPricingError("Merton smile strikes must be positive, finite, and unique")
    iv_config = solver or ImpliedVolatilityConfig()
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
        price = merton_analytical_price(point_contract, market, parameters, series).price
        implied, status = _implied_volatility(point_contract, market, price, iv_config)
        points.append(
            MertonSmilePoint(
                strike=strike,
                moneyness=strike / contract.spot,
                log_moneyness=math.log(strike / forward),
                merton_price=price,
                implied_volatility=implied,
                inversion_status=status,
            )
        )
    return MertonSmileResult(maturity_years=contract.maturity_years, points=points)


def merton_parameter_sensitivity(
    contract: OptionContract,
    market: MarketEnvironment,
    parameters: MertonJumpParameters,
    parameter: MertonParameterName,
    values: Sequence[float],
    series: MertonSeriesConfig | None = None,
    solver: ImpliedVolatilityConfig | None = None,
) -> MertonSensitivityResult:
    if not values or len(set(values)) != len(values) or any(not math.isfinite(v) for v in values):
        raise MertonPricingError("Merton sensitivity values must be finite, nonempty, and unique")
    if parameter is not MertonParameterName.JUMP_MEAN and any(value < 0.0 for value in values):
        raise MertonPricingError(f"{parameter.value} sensitivity values must be non-negative")
    iv_config = solver or ImpliedVolatilityConfig()
    points = []
    for value in sorted(values):
        scenario_market = market
        scenario_parameters = parameters
        if parameter is MertonParameterName.DIFFUSION_VOLATILITY:
            scenario_market = market.model_copy(update={"volatility": value})
        else:
            scenario_parameters = parameters.model_copy(update={parameter.value: value})
        started = perf_counter()
        price = merton_analytical_price(
            contract, scenario_market, scenario_parameters, series
        ).price
        implied, status = _implied_volatility(contract, scenario_market, price, iv_config)
        points.append(
            MertonSensitivityPoint(
                parameter_value=value,
                price=price,
                implied_volatility=implied,
                inversion_status=status,
                runtime_ms=(perf_counter() - started) * 1_000.0,
            )
        )
    base = (
        market.volatility
        if parameter is MertonParameterName.DIFFUSION_VOLATILITY
        else float(getattr(parameters, parameter.value))
    )
    return MertonSensitivityResult(parameter=parameter, base_value=base, points=points)


def merton_jump_diagnostics(
    contract: OptionContract,
    market: MarketEnvironment,
    parameters: MertonJumpParameters,
    config: MertonSimulationConfig,
) -> MertonJumpDiagnosticsResult:
    started = perf_counter()
    simulation = simulate_merton_terminal(
        contract, market, parameters, config, create_rng(config.seed, config.random_engine)
    )
    counts = simulation.jump_counts
    expected = parameters.jump_intensity * contract.maturity_years
    zero = math.exp(-expected)
    variance = float(np.var(counts, ddof=1)) if counts.size > 1 else 0.0
    zero_fraction = float(np.mean(counts == 0))
    return MertonJumpDiagnosticsResult(
        expected_jump_count=expected,
        theoretical_jump_count_variance=expected,
        theoretical_zero_jump_probability=zero,
        theoretical_at_least_one_jump_probability=1.0 - zero,
        mean_simulated_jump_count=float(np.mean(counts)),
        simulated_jump_count_variance=variance,
        zero_jump_fraction=zero_fraction,
        at_least_one_jump_fraction=1.0 - zero_fraction,
        max_observed_jumps=int(np.max(counts)),
        num_simulations=config.num_simulations,
        seed=config.seed,
        runtime_ms=(perf_counter() - started) * 1_000.0,
    )


def merton_price_objective(
    spot: float,
    market: MarketEnvironment,
    parameters: MertonJumpParameters,
    observations: Sequence[MertonObjectiveObservation],
    series: MertonSeriesConfig | None = None,
) -> MertonObjectiveResult:
    """Return finite weighted price residuals for a future bounded calibrator."""

    residuals, weighted = [], []
    for observation in observations:
        contract = OptionContract(
            option_type=observation.option_type,
            exercise_style=ExerciseStyle.EUROPEAN,
            spot=spot,
            strike=observation.strike,
            maturity_years=observation.maturity_years,
        )
        residual = (
            merton_analytical_price(contract, market, parameters, series).price
            - observation.observed_price
        )
        residuals.append(residual)
        weighted.append(math.sqrt(observation.weight) * residual)
    return MertonObjectiveResult(
        residuals=residuals,
        weighted_residuals=weighted,
        sum_squared_error=float(np.dot(weighted, weighted)),
    )
