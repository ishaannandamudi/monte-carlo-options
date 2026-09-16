"""Barrier comparisons, parity, convergence, and level-sensitivity research."""

import math
from collections.abc import Sequence

from backend.app.analytics.statistics import sample_mean
from backend.app.core.exceptions import BarrierPricingError
from backend.app.models.barrier import (
    BarrierComparisonResult,
    BarrierConvergenceMode,
    BarrierConvergencePoint,
    BarrierConvergenceResult,
    BarrierLevelPoint,
    BarrierLevelSensitivityResult,
    BarrierMonitoringMode,
    BarrierOptionContract,
    BarrierParityResult,
    BarrierSimulationConfig,
    BarrierStyle,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.monitoring import MonitoringSchedule
from backend.app.pricing.barrier import (
    barrier_monte_carlo_price,
    continuous_barrier_analytical_price,
    evaluate_barrier_samples,
)
from backend.app.simulation.random import spawn_child_seeds


def _contract_with(
    contract: BarrierOptionContract,
    **updates: object,
) -> BarrierOptionContract:
    try:
        return BarrierOptionContract.model_validate({**contract.model_dump(), **updates})
    except ValueError as exc:
        raise BarrierPricingError(f"unsupported barrier experiment: {exc}") from exc


def barrier_parity(
    contract: BarrierOptionContract,
    market: MarketEnvironment,
    config: BarrierSimulationConfig | None = None,
) -> BarrierParityResult:
    """Verify zero-rebate IN/OUT parity on identical seed-defined paths."""

    if contract.rebate != 0.0:
        raise BarrierPricingError("knock-in/out parity requires zero rebate")
    base = config or BarrierSimulationConfig()
    knock_in = _contract_with(contract, barrier_style=BarrierStyle.IN)
    knock_out = _contract_with(contract, barrier_style=BarrierStyle.OUT)
    in_sample = evaluate_barrier_samples(knock_in, market, base)
    in_price = sample_mean(in_sample.discounted_payoffs)
    vanilla_price = sample_mean(in_sample.discounted_vanilla)
    if contract.initially_hit:
        out_price = 0.0
    else:
        out_sample = evaluate_barrier_samples(knock_out, market, base)
        if not (
            in_sample.observations.shape == out_sample.observations.shape
            and (in_sample.observations == out_sample.observations).all()
        ):
            raise BarrierPricingError("barrier parity paths were not reproduced identically")
        out_price = sample_mean(out_sample.discounted_payoffs)
    residual = in_price + out_price - vanilla_price
    if not math.isfinite(residual):
        raise BarrierPricingError("barrier parity residual is non-finite")
    return BarrierParityResult(
        knock_in_price=in_price,
        knock_out_price=out_price,
        vanilla_price=vanilla_price,
        parity_residual=residual,
        monitoring_mode=contract.monitoring_mode,
        num_simulations=base.num_simulations,
        seed=base.seed,
    )


def compare_barrier_methods(
    contract: BarrierOptionContract,
    market: MarketEnvironment,
    config: BarrierSimulationConfig | None = None,
) -> BarrierComparisonResult:
    """Compare observed discrete monitoring with continuous bridge weighting."""

    base = config or BarrierSimulationConfig()
    discrete_contract = _contract_with(contract, monitoring_mode=BarrierMonitoringMode.DISCRETE)
    bridge_contract = _contract_with(
        contract,
        monitoring_mode=BarrierMonitoringMode.BROWNIAN_BRIDGE,
    )
    discrete = barrier_monte_carlo_price(discrete_contract, market, base)
    bridge = barrier_monte_carlo_price(bridge_contract, market, base)
    analytical = (
        continuous_barrier_analytical_price(bridge_contract, market)
        if contract.rebate == 0.0
        else None
    )
    discrete_parity = (
        barrier_parity(discrete_contract, market, base).parity_residual
        if contract.rebate == 0.0
        else None
    )
    bridge_parity = (
        barrier_parity(bridge_contract, market, base).parity_residual
        if contract.rebate == 0.0
        else None
    )
    return BarrierComparisonResult(
        discrete=discrete,
        brownian_bridge=bridge,
        analytical_continuous=analytical,
        bridge_minus_discrete=bridge.price - discrete.price,
        hit_probability_difference=bridge.hit_probability - discrete.hit_probability,
        analytical_bridge_error=(
            abs(bridge.price - analytical.price) if analytical is not None else None
        ),
        discrete_parity_residual=discrete_parity,
        bridge_parity_residual=bridge_parity,
    )


def barrier_convergence_inputs(
    contract: BarrierOptionContract,
    config: BarrierSimulationConfig,
    values: Sequence[int],
    mode: BarrierConvergenceMode,
) -> list[tuple[BarrierOptionContract, BarrierSimulationConfig]]:
    if not values or any(value <= 0 for value in values) or len(set(values)) != len(values):
        raise BarrierPricingError("convergence values must be nonempty, positive, and unique")
    if (
        mode is BarrierConvergenceMode.MONITORING_COUNT
        and contract.monitoring_schedule.monitoring_count is None
    ):
        raise BarrierPricingError(
            "monitoring-count convergence requires a regular monitoring schedule"
        )
    seeds = spawn_child_seeds(config.seed, len(values))
    inputs = []
    for value, seed in zip(sorted(values), seeds, strict=True):
        if mode is BarrierConvergenceMode.PATH_COUNT:
            point_contract = contract
            point_config = config.model_copy(update={"num_simulations": value, "seed": seed})
        else:
            point_contract = _contract_with(
                contract,
                monitoring_schedule=MonitoringSchedule.regular(value),
            )
            point_config = config.model_copy(update={"seed": seed})
        inputs.append((point_contract, point_config))
    return inputs


def run_barrier_convergence(
    contract: BarrierOptionContract,
    market: MarketEnvironment,
    config: BarrierSimulationConfig,
    values: Sequence[int],
    mode: BarrierConvergenceMode,
) -> BarrierConvergenceResult:
    points = []
    for point_contract, point_config in barrier_convergence_inputs(contract, config, values, mode):
        discrete_contract = _contract_with(
            point_contract,
            monitoring_mode=BarrierMonitoringMode.DISCRETE,
        )
        bridge_contract = _contract_with(
            point_contract,
            monitoring_mode=BarrierMonitoringMode.BROWNIAN_BRIDGE,
        )
        discrete = barrier_monte_carlo_price(discrete_contract, market, point_config)
        bridge = barrier_monte_carlo_price(bridge_contract, market, point_config)
        analytical = (
            continuous_barrier_analytical_price(bridge_contract, market).price
            if point_contract.rebate == 0.0
            else None
        )
        points.append(
            BarrierConvergencePoint(
                num_simulations=point_config.num_simulations,
                monitoring_count=point_contract.monitoring_count,
                seed=point_config.seed,
                discrete_price=discrete.price,
                bridge_price=bridge.price,
                discrete_standard_error=discrete.standard_error,
                bridge_standard_error=bridge.standard_error,
                discrete_hit_probability=discrete.hit_probability,
                bridge_hit_probability=bridge.hit_probability,
                analytical_continuous_price=analytical,
                bridge_absolute_error=(
                    abs(bridge.price - analytical) if analytical is not None else None
                ),
                runtime_ms=discrete.runtime_ms + bridge.runtime_ms,
            )
        )
    return BarrierConvergenceResult(mode=mode, root_seed=config.seed, points=points)


def run_barrier_level_sensitivity(
    contract: BarrierOptionContract,
    market: MarketEnvironment,
    config: BarrierSimulationConfig,
    barrier_levels: Sequence[float],
) -> BarrierLevelSensitivityResult:
    """Evaluate barrier levels with common random streams for cleaner comparisons."""

    if (
        not barrier_levels
        or any(not math.isfinite(level) or level <= 0.0 for level in barrier_levels)
        or len(set(barrier_levels)) != len(barrier_levels)
    ):
        raise BarrierPricingError("barrier levels must be finite, positive, and unique")
    points = []
    for level in sorted(barrier_levels):
        point_contract = _contract_with(contract, barrier=level)
        discrete = barrier_monte_carlo_price(
            _contract_with(point_contract, monitoring_mode=BarrierMonitoringMode.DISCRETE),
            market,
            config,
        )
        bridge = barrier_monte_carlo_price(
            _contract_with(
                point_contract,
                monitoring_mode=BarrierMonitoringMode.BROWNIAN_BRIDGE,
            ),
            market,
            config,
        )
        points.append(
            BarrierLevelPoint(
                barrier=level,
                discrete_price=discrete.price,
                bridge_price=bridge.price,
                discrete_standard_error=discrete.standard_error,
                bridge_standard_error=bridge.standard_error,
                discrete_hit_probability=discrete.hit_probability,
                bridge_hit_probability=bridge.hit_probability,
                bridge_minus_discrete=bridge.price - discrete.price,
            )
        )
    return BarrierLevelSensitivityResult(
        levels=points,
        num_simulations=config.num_simulations,
        root_seed=config.seed,
    )
