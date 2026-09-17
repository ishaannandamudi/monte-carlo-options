import math

import pytest

from backend.app.models.heston import HestonAnalyticalMethod, HestonParameters
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.pricing.black_scholes import black_scholes_price
from backend.app.pricing.heston_analytical import (
    heston_analytical_price,
    heston_characteristic_function,
)


def parameters(**updates: float) -> HestonParameters:
    data = {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "sigma_v": 0.3, "rho": -0.7}
    data.update(updates)
    return HestonParameters.model_validate(data)


def contract(option_type: OptionType = OptionType.CALL, strike: float = 100) -> OptionContract:
    return OptionContract(
        option_type=option_type,
        spot=100,
        strike=strike,
        maturity_years=1,
    )


def market(rate: float = 0.05, dividend: float = 0.0) -> MarketEnvironment:
    return MarketEnvironment(risk_free_rate=rate, dividend_yield=dividend, volatility=0.0)


def test_known_heston_benchmark_call() -> None:
    result = heston_analytical_price(contract(), market(), parameters())
    # Published/common reference set: S=K=100, T=1, r=5%, v0=theta=.04,
    # kappa=2, sigma_v=.3, rho=-.7.
    assert result.price == pytest.approx(10.394218565, abs=2e-8)
    assert result.method is HestonAnalyticalMethod.LITTLE_TRAP_FOURIER
    assert result.integration_evaluations > 0


def test_put_call_parity() -> None:
    call = heston_analytical_price(contract(OptionType.CALL), market(), parameters()).price
    put = heston_analytical_price(contract(OptionType.PUT), market(), parameters()).price
    parity = 100 - 100 * math.exp(-0.05)
    assert call - put == pytest.approx(parity, abs=2e-12)


@pytest.mark.parametrize("option_type", list(OptionType))
def test_zero_vol_of_vol_black_scholes_limit(option_type: OptionType) -> None:
    params = parameters(sigma_v=0.0)
    heston = heston_analytical_price(contract(option_type), market(), params)
    black_scholes = black_scholes_price(
        contract(option_type),
        MarketEnvironment(risk_free_rate=0.05, dividend_yield=0.0, volatility=0.2),
    )
    assert heston.price == pytest.approx(black_scholes, abs=2e-14)
    assert heston.method is HestonAnalyticalMethod.DETERMINISTIC_VARIANCE_LIMIT


def test_nearly_zero_vol_of_vol_uses_stable_limit() -> None:
    heston = heston_analytical_price(contract(), market(), parameters(sigma_v=1e-5)).price
    black_scholes = black_scholes_price(
        contract(), MarketEnvironment(risk_free_rate=0.05, volatility=0.2)
    )
    assert heston == pytest.approx(black_scholes, abs=1e-10)


@pytest.mark.parametrize("rho", [-1.0, -0.9, 0.0, 0.9, 1.0])
def test_characteristic_function_is_finite_and_normalized(rho: float) -> None:
    params = parameters(rho=rho)
    assert heston_characteristic_function(0j, contract(), market(), params) == 1.0 + 0j
    for frequency in (0.1, 1.0, 10.0, 50.0):
        value = heston_characteristic_function(complex(frequency), contract(), market(), params)
        assert math.isfinite(value.real) and math.isfinite(value.imag)
        assert abs(value) <= 1.0 + 1e-12


@pytest.mark.parametrize(
    "params",
    [
        parameters(kappa=0.0),
        parameters(theta=0.0),
        parameters(v0=0.0),
        parameters(kappa=0.5, theta=0.02, sigma_v=0.8),
    ],
)
def test_boundary_and_feller_violating_analytical_prices(params: HestonParameters) -> None:
    result = heston_analytical_price(contract(), market(-0.01, 0.02), params)
    assert math.isfinite(result.price) and result.price >= 0.0


@pytest.mark.parametrize("option_type", list(OptionType))
def test_zero_maturity_is_intrinsic(option_type: OptionType) -> None:
    expiring = contract(option_type).model_copy(update={"maturity_years": 0.0, "spot": 90.0})
    result = heston_analytical_price(expiring, market(), parameters())
    expected = max(90 - 100, 0) if option_type is OptionType.CALL else max(100 - 90, 0)
    assert result.price == expected
    assert result.method is HestonAnalyticalMethod.INTRINSIC
