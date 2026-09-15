"""One-dimensional analytical price and Greek sensitivity experiments."""

import math
from collections.abc import Sequence

from backend.app.core.exceptions import SensitivityGridError
from backend.app.models.greeks import SensitivityParameter
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract
from backend.app.models.results import SensitivityPoint, SensitivityResult
from backend.app.pricing.black_scholes_greeks import black_scholes_greeks


def _base_value(
    contract: OptionContract,
    market: MarketEnvironment,
    parameter: SensitivityParameter,
) -> float:
    if parameter is SensitivityParameter.SPOT:
        return contract.spot
    if parameter is SensitivityParameter.VOLATILITY:
        return market.volatility
    if parameter is SensitivityParameter.MATURITY:
        return contract.maturity_years
    if parameter is SensitivityParameter.RISK_FREE_RATE:
        return market.risk_free_rate
    return market.dividend_yield


def _validate_values(parameter: SensitivityParameter, values: Sequence[float]) -> list[float]:
    grid = [float(value) for value in values]
    if not grid:
        raise SensitivityGridError("at least one sensitivity value is required")
    if not all(math.isfinite(value) for value in grid):
        raise SensitivityGridError("sensitivity values must be finite")
    if parameter is SensitivityParameter.SPOT and any(value <= 0.0 for value in grid):
        raise SensitivityGridError("spot sensitivity values must be positive")
    if parameter in {SensitivityParameter.VOLATILITY, SensitivityParameter.MATURITY} and any(
        value < 0.0 for value in grid
    ):
        raise SensitivityGridError(
            f"{parameter.value.lower()} sensitivity values must be non-negative"
        )
    return grid


def _state_at(
    contract: OptionContract,
    market: MarketEnvironment,
    parameter: SensitivityParameter,
    value: float,
) -> tuple[OptionContract, MarketEnvironment]:
    contract_values = contract.model_dump()
    market_values = market.model_dump()
    if parameter is SensitivityParameter.SPOT:
        contract_values["spot"] = value
    elif parameter is SensitivityParameter.VOLATILITY:
        market_values["volatility"] = value
    elif parameter is SensitivityParameter.MATURITY:
        contract_values["maturity_years"] = value
    elif parameter is SensitivityParameter.RISK_FREE_RATE:
        market_values["risk_free_rate"] = value
    else:
        market_values["dividend_yield"] = value
    return (
        OptionContract.model_validate(contract_values),
        MarketEnvironment.model_validate(market_values),
    )


def run_sensitivity(
    contract: OptionContract,
    market: MarketEnvironment,
    parameter: SensitivityParameter,
    values: Sequence[float],
) -> SensitivityResult:
    """Evaluate Black-Scholes price and all analytical Greeks across a validated grid."""

    grid = _validate_values(parameter, values)
    points: list[SensitivityPoint] = []
    for value in grid:
        point_contract, point_market = _state_at(contract, market, parameter, value)
        result = black_scholes_greeks(point_contract, point_market)
        points.append(
            SensitivityPoint(
                parameter_value=value,
                black_scholes_price=result.option_price,
                delta=result.delta,
                gamma=result.gamma,
                vega=result.vega,
                theta=result.theta,
                rho=result.rho,
                dividend_rho=result.dividend_rho,
            )
        )
    return SensitivityResult(
        varied_parameter=parameter,
        base_value=_base_value(contract, market, parameter),
        points=points,
    )
