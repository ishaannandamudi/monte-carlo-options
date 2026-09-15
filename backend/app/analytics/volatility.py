"""Batch, synthetic-smile, and solver-comparison implied-volatility analytics."""

import math
from collections.abc import Sequence

from backend.app.core.exceptions import QuantitativeError
from backend.app.models.option import OptionContract, OptionType
from backend.app.models.results import (
    ImpliedVolatilityBatchItemResult,
    ImpliedVolatilityBatchResult,
    ImpliedVolatilitySolverComparisonResult,
    SolverComparisonPoint,
    VolatilitySmilePoint,
    VolatilitySmileResult,
)
from backend.app.models.volatility import (
    ImpliedVolatilityConfig,
    ImpliedVolatilityMarket,
    ImpliedVolatilityMethod,
    ImpliedVolatilityObservation,
)
from backend.app.pricing.implied_volatility import (
    forward_price,
    implied_volatility_failure_status,
    solve_implied_volatility,
)


def solve_implied_volatility_batch(
    observations: Sequence[ImpliedVolatilityObservation],
    config: ImpliedVolatilityConfig | None = None,
) -> ImpliedVolatilityBatchResult:
    """Solve observations in order and retain every per-item failure."""

    solver_config = config or ImpliedVolatilityConfig()
    items: list[ImpliedVolatilityBatchItemResult] = []
    success_count = 0
    for index, observation in enumerate(observations):
        try:
            result = solve_implied_volatility(observation, solver_config)
            success = result.converged and result.implied_volatility is not None
            success_count += int(success)
            items.append(
                ImpliedVolatilityBatchItemResult(
                    index=index,
                    success=success,
                    result=result,
                    status=result.status,
                    message=result.message,
                )
            )
        except QuantitativeError as exc:
            status = implied_volatility_failure_status(exc)
            items.append(
                ImpliedVolatilityBatchItemResult(
                    index=index,
                    success=False,
                    status=status,
                    message=str(exc),
                )
            )
    return ImpliedVolatilityBatchResult(
        items=items,
        total_count=len(items),
        success_count=success_count,
        failure_count=len(items) - success_count,
    )


def run_volatility_smile(
    *,
    spot: float,
    maturity_years: float,
    risk_free_rate: float,
    dividend_yield: float,
    option_type: OptionType,
    strikes: Sequence[float],
    option_prices: Sequence[float],
    config: ImpliedVolatilityConfig | None = None,
) -> VolatilitySmileResult:
    """Recover one ordered synthetic smile with ``log(K/F)`` coordinates."""

    if len(strikes) != len(option_prices):
        raise ValueError("strikes and option_prices must have matching lengths")
    if not strikes:
        raise ValueError("at least one smile point is required")
    market = ImpliedVolatilityMarket(
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
    )
    forward = forward_price(spot, maturity_years, risk_free_rate, dividend_yield)
    observations = [
        ImpliedVolatilityObservation(
            contract=OptionContract(
                option_type=option_type,
                spot=spot,
                strike=strike,
                maturity_years=maturity_years,
            ),
            market=market,
            observed_price=price,
        )
        for strike, price in zip(strikes, option_prices, strict=True)
    ]
    batch = solve_implied_volatility_batch(observations, config)
    points: list[VolatilitySmilePoint] = []
    for observation, item in zip(observations, batch.items, strict=True):
        strike = observation.contract.strike
        implied_volatility = None
        if item.result is not None:
            implied_volatility = item.result.implied_volatility
        log_moneyness = math.log(strike / forward)
        points.append(
            VolatilitySmilePoint(
                strike=strike,
                moneyness=strike / spot,
                log_moneyness=log_moneyness,
                option_price=observation.observed_price,
                implied_volatility=implied_volatility,
                status=item.status,
                message=item.message,
            )
        )
    return VolatilitySmileResult(
        spot=spot,
        maturity_years=maturity_years,
        forward_price=forward,
        option_type=option_type,
        points=points,
    )


def compare_implied_volatility_solvers(
    observation: ImpliedVolatilityObservation,
    config: ImpliedVolatilityConfig | None = None,
) -> ImpliedVolatilitySolverComparisonResult:
    """Run Brent, bisection, and Newton independently on one observation."""

    base_config = config or ImpliedVolatilityConfig()
    points: list[SolverComparisonPoint] = []
    for method in (
        ImpliedVolatilityMethod.BRENT,
        ImpliedVolatilityMethod.BISECTION,
        ImpliedVolatilityMethod.NEWTON,
    ):
        method_config = ImpliedVolatilityConfig.model_validate(
            {**base_config.model_dump(), "method": method}
        )
        try:
            result = solve_implied_volatility(observation, method_config)
            success = result.converged and result.implied_volatility is not None
            points.append(
                SolverComparisonPoint(
                    method=method,
                    success=success,
                    result=result,
                    status=result.status,
                    message=result.message,
                )
            )
        except QuantitativeError as exc:
            points.append(
                SolverComparisonPoint(
                    method=method,
                    success=False,
                    status=implied_volatility_failure_status(exc),
                    message=str(exc),
                )
            )
    return ImpliedVolatilitySolverComparisonResult(
        target_price=observation.observed_price,
        points=points,
    )
