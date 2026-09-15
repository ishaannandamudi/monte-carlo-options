import math

import pytest

from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.pricing.black_scholes import black_scholes_price
from backend.app.pricing.black_scholes_greeks import black_scholes_greeks


def contract(
    option_type: OptionType,
    *,
    spot: float = 100.0,
    strike: float = 100.0,
    maturity: float = 1.0,
) -> OptionContract:
    return OptionContract(
        option_type=option_type,
        spot=spot,
        strike=strike,
        maturity_years=maturity,
    )


@pytest.mark.parametrize(
    ("option_type", "expected"),
    [
        (
            OptionType.CALL,
            {
                "delta": 0.6368306511756191,
                "gamma": 0.018762017345846895,
                "vega": 37.52403469169379,
                "theta": -6.414027546438197,
                "rho": 53.232481545376345,
                "dividend_rho": -63.68306511756191,
            },
        ),
        (
            OptionType.PUT,
            {
                "delta": -0.3631693488243809,
                "gamma": 0.018762017345846895,
                "vega": 37.52403469169379,
                "theta": -1.657880423934626,
                "rho": -41.89046090469506,
                "dividend_rho": 36.31693488243809,
            },
        ),
    ],
)
def test_known_black_scholes_greek_benchmarks(
    option_type: OptionType,
    expected: dict[str, float],
) -> None:
    result = black_scholes_greeks(
        contract(option_type),
        MarketEnvironment(risk_free_rate=0.05, volatility=0.2),
    )
    for field, value in expected.items():
        assert getattr(result, field) == pytest.approx(value, abs=1e-12)
    assert result.option_price == black_scholes_price(
        contract(option_type),
        MarketEnvironment(risk_free_rate=0.05, volatility=0.2),
    )
    assert result.d1 == pytest.approx(0.35)
    assert result.d2 == pytest.approx(0.15)


def test_call_put_greek_relationships_with_dividends_and_negative_rate() -> None:
    maturity = 1.7
    market = MarketEnvironment(risk_free_rate=-0.01, volatility=0.3, dividend_yield=0.025)
    call = black_scholes_greeks(contract(OptionType.CALL, maturity=maturity), market)
    put = black_scholes_greeks(contract(OptionType.PUT, maturity=maturity), market)
    spot_present = 100.0 * math.exp(-0.025 * maturity)
    strike_present = 100.0 * math.exp(0.01 * maturity)

    assert call.delta - put.delta == pytest.approx(math.exp(-0.025 * maturity), abs=1e-13)
    assert call.gamma == pytest.approx(put.gamma, abs=1e-15)
    assert call.vega == pytest.approx(put.vega, abs=1e-13)
    assert call.rho - put.rho == pytest.approx(maturity * strike_present, abs=1e-12)
    assert call.dividend_rho - put.dividend_rho == pytest.approx(
        -maturity * spot_present,
        abs=1e-12,
    )


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_theta_is_negative_maturity_derivative(option_type: OptionType) -> None:
    option = contract(option_type, maturity=1.3)
    market = MarketEnvironment(risk_free_rate=0.03, volatility=0.27, dividend_yield=0.01)
    bump = 1e-5
    upward = contract(option_type, maturity=option.maturity_years + bump)
    downward = contract(option_type, maturity=option.maturity_years - bump)
    maturity_derivative = (
        black_scholes_price(upward, market) - black_scholes_price(downward, market)
    ) / (2.0 * bump)
    assert black_scholes_greeks(option, market).theta == pytest.approx(
        -maturity_derivative,
        rel=1e-9,
        abs=1e-9,
    )


@pytest.mark.parametrize(
    ("option_type", "spot", "expected_delta"),
    [
        (OptionType.CALL, 110.0, 1.0),
        (OptionType.CALL, 90.0, 0.0),
        (OptionType.CALL, 100.0, 0.5),
        (OptionType.PUT, 110.0, 0.0),
        (OptionType.PUT, 90.0, -1.0),
        (OptionType.PUT, 100.0, -0.5),
    ],
)
def test_exact_expiry_boundary_convention(
    option_type: OptionType,
    spot: float,
    expected_delta: float,
) -> None:
    result = black_scholes_greeks(
        contract(option_type, spot=spot, maturity=0.0),
        MarketEnvironment(risk_free_rate=-0.02, volatility=0.4, dividend_yield=0.03),
    )
    assert result.delta == expected_delta
    assert result.gamma == 0.0
    assert result.vega == 0.0
    assert result.theta == 0.0
    assert result.rho == 0.0
    assert result.dividend_rho == 0.0
    assert result.d1 is None
    assert result.d2 is None


def test_zero_volatility_deterministic_limit_and_forward_kink_conventions() -> None:
    market = MarketEnvironment(risk_free_rate=0.0, volatility=0.0, dividend_yield=0.0)
    call_itm = black_scholes_greeks(
        contract(OptionType.CALL, spot=120.0, strike=100.0),
        market,
    )
    put_itm = black_scholes_greeks(
        contract(OptionType.PUT, spot=80.0, strike=100.0),
        market,
    )
    call_forward_atm = black_scholes_greeks(contract(OptionType.CALL), market)
    put_forward_atm = black_scholes_greeks(contract(OptionType.PUT), market)

    assert call_itm.delta == 1.0
    assert put_itm.delta == -1.0
    assert call_itm.gamma == put_itm.gamma == 0.0
    assert call_itm.vega == put_itm.vega == 0.0
    assert call_itm.theta == put_itm.theta == 0.0
    assert call_itm.rho == 100.0
    assert put_itm.rho == -100.0
    assert call_itm.dividend_rho == -120.0
    assert put_itm.dividend_rho == 80.0
    expected_kink_vega = 100.0 / math.sqrt(2.0 * math.pi)
    assert call_forward_atm.delta == 0.5
    assert put_forward_atm.delta == -0.5
    assert call_forward_atm.gamma == put_forward_atm.gamma == 0.0
    assert call_forward_atm.vega == pytest.approx(expected_kink_vega)
    assert put_forward_atm.vega == pytest.approx(expected_kink_vega)
    assert call_forward_atm.rho == 50.0
    assert put_forward_atm.rho == -50.0
    assert call_forward_atm.dividend_rho == -50.0
    assert put_forward_atm.dividend_rho == 50.0


@pytest.mark.parametrize(
    ("spot", "strike", "maturity", "volatility"),
    [
        (100.0, 100.0, 1e-12, 0.2),
        (100.0, 100.0, 1.0, 1e-12),
        (1_000.0, 1.0, 3.0, 0.4),
        (1.0, 1_000.0, 3.0, 0.4),
    ],
)
def test_analytical_greeks_remain_finite_at_numerical_edges(
    spot: float,
    strike: float,
    maturity: float,
    volatility: float,
) -> None:
    result = black_scholes_greeks(
        contract(OptionType.CALL, spot=spot, strike=strike, maturity=maturity),
        MarketEnvironment(risk_free_rate=-0.03, volatility=volatility, dividend_yield=0.02),
    )
    numeric = [
        value
        for value in result.model_dump().values()
        if isinstance(value, int | float) and value is not None
    ]
    assert all(math.isfinite(float(value)) for value in numeric)
