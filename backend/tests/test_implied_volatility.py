import math

import pytest
from pydantic import ValidationError

from backend.app.core.exceptions import (
    ArbitrageBoundsError,
    ImpliedVolatilityBracketError,
    ImpliedVolatilityConvergenceError,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.models.volatility import (
    ImpliedVolatilityConfig,
    ImpliedVolatilityMarket,
    ImpliedVolatilityMethod,
    ImpliedVolatilityObservation,
    ImpliedVolatilityStatus,
)
from backend.app.pricing.black_scholes import black_scholes_price
from backend.app.pricing.implied_volatility import solve_implied_volatility


def observation_from_volatility(
    option_type: OptionType,
    *,
    spot: float = 100.0,
    strike: float = 100.0,
    maturity: float = 1.0,
    rate: float = 0.03,
    dividend: float = 0.01,
    volatility: float = 0.2,
) -> ImpliedVolatilityObservation:
    contract = OptionContract(
        option_type=option_type,
        spot=spot,
        strike=strike,
        maturity_years=maturity,
    )
    price = black_scholes_price(
        contract,
        MarketEnvironment(
            risk_free_rate=rate,
            volatility=volatility,
            dividend_yield=dividend,
        ),
    )
    return ImpliedVolatilityObservation(
        contract=contract,
        market=ImpliedVolatilityMarket(
            risk_free_rate=rate,
            dividend_yield=dividend,
        ),
        observed_price=price,
    )


@pytest.mark.parametrize("method", list(ImpliedVolatilityMethod))
@pytest.mark.parametrize(
    ("option_type", "strike", "maturity", "rate", "dividend", "volatility"),
    [
        (OptionType.CALL, 80.0, 0.25, 0.03, 0.00, 0.10),
        (OptionType.CALL, 100.0, 1.0, 0.05, 0.02, 0.25),
        (OptionType.CALL, 120.0, 3.0, -0.01, 0.03, 0.60),
        (OptionType.PUT, 80.0, 2.0, -0.02, 0.00, 0.35),
        (OptionType.PUT, 100.0, 0.10, 0.01, 0.01, 0.15),
        (OptionType.PUT, 120.0, 5.0, 0.06, 0.02, 1.00),
        (OptionType.CALL, 100.0, 1.5, 0.01, 0.00, 2.00),
    ],
)
def test_implied_volatility_round_trip(
    method: ImpliedVolatilityMethod,
    option_type: OptionType,
    strike: float,
    maturity: float,
    rate: float,
    dividend: float,
    volatility: float,
) -> None:
    observation = observation_from_volatility(
        option_type,
        strike=strike,
        maturity=maturity,
        rate=rate,
        dividend=dividend,
        volatility=volatility,
    )
    result = solve_implied_volatility(
        observation,
        ImpliedVolatilityConfig(method=method, initial_volatility=0.3),
    )
    assert result.converged is True
    assert result.implied_volatility == pytest.approx(volatility, abs=2e-8)
    assert result.absolute_price_error is not None
    assert result.absolute_price_error <= 1e-10
    assert result.final_vega is not None and result.final_vega >= 0.0
    assert math.isfinite(result.runtime_ms)


def test_exact_expiry_is_explicitly_non_identifiable() -> None:
    contract = OptionContract(
        option_type=OptionType.PUT,
        spot=90.0,
        strike=100.0,
        maturity_years=0.0,
    )
    result = solve_implied_volatility(
        ImpliedVolatilityObservation(
            contract=contract,
            market=ImpliedVolatilityMarket(risk_free_rate=-0.02, dividend_yield=0.03),
            observed_price=10.0,
        )
    )
    assert result.implied_volatility is None
    assert result.converged is False
    assert result.status is ImpliedVolatilityStatus.NON_IDENTIFIABLE
    assert result.method_used is None


def test_exact_expiry_rejects_non_intrinsic_price_even_inside_general_bounds() -> None:
    contract = OptionContract(
        option_type=OptionType.CALL,
        spot=90.0,
        strike=100.0,
        maturity_years=0.0,
    )
    observation = ImpliedVolatilityObservation(
        contract=contract,
        market=ImpliedVolatilityMarket(risk_free_rate=0.0),
        observed_price=1.0,
    )
    with pytest.raises(ArbitrageBoundsError, match="must equal intrinsic"):
        solve_implied_volatility(observation)


@pytest.mark.parametrize(
    ("maturity", "volatility"),
    [(1e-6, 0.8), (1.0, 1e-4), (0.5, 4.0)],
)
def test_short_maturity_tiny_and_extreme_finite_volatility(
    maturity: float,
    volatility: float,
) -> None:
    observation = observation_from_volatility(
        OptionType.CALL,
        maturity=maturity,
        rate=0.0 if volatility == 1e-4 else 0.03,
        dividend=0.0 if volatility == 1e-4 else 0.01,
        volatility=volatility,
    )
    result = solve_implied_volatility(observation)
    assert result.implied_volatility == pytest.approx(volatility, abs=2e-8)
    assert result.absolute_price_error is not None
    assert result.absolute_price_error <= 1e-10


def test_zero_volatility_lower_bound_is_recovered_without_fabrication() -> None:
    observation = observation_from_volatility(
        OptionType.CALL,
        strike=90.0,
        rate=0.04,
        dividend=0.01,
        volatility=0.0,
    )
    result = solve_implied_volatility(observation)
    assert result.implied_volatility == 0.0
    assert result.method_used is None
    assert result.absolute_price_error == 0.0


def test_config_rejects_invalid_bounds_tolerances_and_guess() -> None:
    for values in (
        {"min_volatility": 1.0, "max_volatility": 1.0},
        {"initial_volatility": 2.0, "max_volatility": 1.0},
        {"price_tolerance": 0.0},
        {"volatility_tolerance": 0.0},
        {"max_iterations": 0},
    ):
        with pytest.raises(ValidationError):
            ImpliedVolatilityConfig.model_validate(values)


def test_configured_range_without_root_fails_clearly() -> None:
    observation = observation_from_volatility(OptionType.PUT, volatility=1.0)
    with pytest.raises(ImpliedVolatilityBracketError, match="configured range"):
        solve_implied_volatility(
            observation,
            ImpliedVolatilityConfig(max_volatility=0.5, initial_volatility=0.2),
        )


def test_bisection_iteration_limit_is_enforced() -> None:
    observation = observation_from_volatility(OptionType.CALL, volatility=0.4)
    with pytest.raises(ImpliedVolatilityConvergenceError, match="1 iterations"):
        solve_implied_volatility(
            observation,
            ImpliedVolatilityConfig(
                method=ImpliedVolatilityMethod.BISECTION,
                max_iterations=1,
            ),
        )


def test_newton_low_vega_guard_and_auto_fallback() -> None:
    observation = observation_from_volatility(
        OptionType.CALL,
        strike=130.0,
        maturity=0.05,
        rate=0.01,
        dividend=0.0,
        volatility=0.5,
    )
    with pytest.raises(ImpliedVolatilityConvergenceError, match="Vega is too small"):
        solve_implied_volatility(
            observation,
            ImpliedVolatilityConfig(
                method=ImpliedVolatilityMethod.NEWTON,
                vega_tolerance=1e-6,
            ),
        )

    result = solve_implied_volatility(
        observation,
        ImpliedVolatilityConfig(
            method=ImpliedVolatilityMethod.AUTO,
            vega_tolerance=1e-6,
        ),
    )
    assert result.implied_volatility == pytest.approx(0.5, abs=1e-8)
    assert result.method_used is ImpliedVolatilityMethod.BRENT
    assert result.fallback_used is True


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_black_scholes_price_is_nondecreasing_in_volatility(option_type: OptionType) -> None:
    contract = OptionContract(
        option_type=option_type,
        spot=100.0,
        strike=110.0,
        maturity_years=0.75,
    )
    prices = [
        black_scholes_price(
            contract,
            MarketEnvironment(risk_free_rate=-0.01, volatility=vol, dividend_yield=0.02),
        )
        for vol in (0.0, 0.01, 0.1, 0.5, 1.0, 2.0)
    ]
    assert prices == sorted(prices)
