import math

import pytest

from backend.app.analytics.sensitivity import run_sensitivity
from backend.app.core.exceptions import SensitivityGridError
from backend.app.models.greeks import SensitivityParameter
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.pricing.black_scholes_greeks import black_scholes_greeks


def inputs() -> tuple[OptionContract, MarketEnvironment]:
    return (
        OptionContract(
            option_type=OptionType.CALL,
            spot=100.0,
            strike=100.0,
            maturity_years=1.0,
        ),
        MarketEnvironment(risk_free_rate=0.05, volatility=0.2, dividend_yield=0.01),
    )


@pytest.mark.parametrize(
    ("parameter", "values", "expected_base"),
    [
        (SensitivityParameter.SPOT, [80.0, 100.0, 120.0], 100.0),
        (SensitivityParameter.VOLATILITY, [0.0, 0.2, 0.4], 0.2),
        (SensitivityParameter.MATURITY, [0.0, 1.0, 2.0], 1.0),
        (SensitivityParameter.RISK_FREE_RATE, [-0.02, 0.0, 0.08], 0.05),
        (SensitivityParameter.DIVIDEND_YIELD, [-0.01, 0.0, 0.04], 0.01),
    ],
)
def test_sensitivity_supports_every_parameter(
    parameter: SensitivityParameter,
    values: list[float],
    expected_base: float,
) -> None:
    contract, market = inputs()
    result = run_sensitivity(contract, market, parameter, values)
    assert result.varied_parameter is parameter
    assert result.base_value == expected_base
    assert [point.parameter_value for point in result.points] == values
    assert all(math.isfinite(point.delta) for point in result.points)
    assert all(math.isfinite(point.theta) for point in result.points)


def test_sensitivity_point_matches_direct_analytical_calculation() -> None:
    contract, market = inputs()
    result = run_sensitivity(contract, market, SensitivityParameter.SPOT, [110.0])
    direct = black_scholes_greeks(contract.model_copy(update={"spot": 110.0}), market)
    point = result.points[0]
    assert point.black_scholes_price == direct.option_price
    assert point.delta == direct.delta
    assert point.gamma == direct.gamma
    assert point.vega == direct.vega
    assert point.theta == direct.theta
    assert point.rho == direct.rho
    assert point.dividend_rho == direct.dividend_rho


def test_call_price_increases_across_spot_grid() -> None:
    contract, market = inputs()
    result = run_sensitivity(
        contract,
        market,
        SensitivityParameter.SPOT,
        [80.0, 100.0, 120.0],
    )
    prices = [point.black_scholes_price for point in result.points]
    assert prices == sorted(prices)
    assert len(set(prices)) == len(prices)


@pytest.mark.parametrize(
    ("parameter", "values", "message"),
    [
        (SensitivityParameter.SPOT, [0.0], "positive"),
        (SensitivityParameter.VOLATILITY, [-0.1], "non-negative"),
        (SensitivityParameter.MATURITY, [-1.0], "non-negative"),
        (SensitivityParameter.SPOT, [math.inf], "finite"),
        (SensitivityParameter.SPOT, [], "at least one"),
    ],
)
def test_sensitivity_rejects_invalid_grids(
    parameter: SensitivityParameter,
    values: list[float],
    message: str,
) -> None:
    contract, market = inputs()
    with pytest.raises(SensitivityGridError, match=message):
        run_sensitivity(contract, market, parameter, values)
