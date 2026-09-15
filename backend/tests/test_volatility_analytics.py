import math

import pytest

from backend.app.analytics.volatility import (
    compare_implied_volatility_solvers,
    run_volatility_smile,
    solve_implied_volatility_batch,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.models.volatility import (
    ImpliedVolatilityMarket,
    ImpliedVolatilityMethod,
    ImpliedVolatilityObservation,
    ImpliedVolatilityStatus,
)
from backend.app.pricing.black_scholes import black_scholes_price
from backend.app.pricing.implied_volatility import forward_price


def make_observation(strike: float, volatility: float) -> ImpliedVolatilityObservation:
    contract = OptionContract(
        option_type=OptionType.CALL,
        spot=100.0,
        strike=strike,
        maturity_years=1.0,
    )
    market = MarketEnvironment(risk_free_rate=0.03, volatility=volatility, dividend_yield=0.01)
    return ImpliedVolatilityObservation(
        contract=contract,
        market=ImpliedVolatilityMarket(risk_free_rate=0.03, dividend_yield=0.01),
        observed_price=black_scholes_price(contract, market),
    )


def test_batch_preserves_order_and_partial_failures() -> None:
    observations = [make_observation(90.0, 0.24), make_observation(100.0, 0.20)]
    impossible = observations[1].model_copy(update={"observed_price": 101.0})
    observations.insert(1, impossible)

    result = solve_implied_volatility_batch(observations)

    assert [item.index for item in result.items] == [0, 1, 2]
    assert [item.success for item in result.items] == [True, False, True]
    assert result.items[1].status is ImpliedVolatilityStatus.ARBITRAGE_VIOLATION
    assert result.success_count == 2
    assert result.failure_count == 1
    assert result.items[0].result is not None
    assert result.items[0].result.implied_volatility == pytest.approx(0.24)
    assert result.items[2].result is not None
    assert result.items[2].result.implied_volatility == pytest.approx(0.20)


def test_batch_is_quantitatively_deterministic() -> None:
    observations = [make_observation(95.0, 0.22), make_observation(105.0, 0.21)]
    first = solve_implied_volatility_batch(observations)
    second = solve_implied_volatility_batch(observations)
    assert [item.result.implied_volatility for item in first.items if item.result] == [
        item.result.implied_volatility for item in second.items if item.result
    ]


def test_synthetic_smile_round_trip_and_coordinates() -> None:
    strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
    volatilities = [0.24, 0.22, 0.20, 0.21, 0.23]
    prices = [
        make_observation(strike, volatility).observed_price
        for strike, volatility in zip(strikes, volatilities, strict=True)
    ]

    result = run_volatility_smile(
        spot=100.0,
        maturity_years=1.0,
        risk_free_rate=0.03,
        dividend_yield=0.01,
        option_type=OptionType.CALL,
        strikes=strikes,
        option_prices=prices,
    )

    expected_forward = forward_price(100.0, 1.0, 0.03, 0.01)
    assert [point.strike for point in result.points] == strikes
    assert result.forward_price == pytest.approx(expected_forward)
    for point, expected_volatility in zip(result.points, volatilities, strict=True):
        assert point.implied_volatility == pytest.approx(expected_volatility, abs=1e-8)
        assert point.moneyness == pytest.approx(point.strike / 100.0)
        assert point.log_moneyness == pytest.approx(math.log(point.strike / expected_forward))


def test_flat_smile_and_invalid_point_status() -> None:
    strikes = [80.0, 100.0, 120.0]
    prices = [make_observation(strike, 0.3).observed_price for strike in strikes]
    prices[1] = 101.0
    result = run_volatility_smile(
        spot=100.0,
        maturity_years=1.0,
        risk_free_rate=0.03,
        dividend_yield=0.01,
        option_type=OptionType.CALL,
        strikes=strikes,
        option_prices=prices,
    )
    assert result.points[0].implied_volatility == pytest.approx(0.3)
    assert result.points[1].implied_volatility is None
    assert result.points[1].status is ImpliedVolatilityStatus.ARBITRAGE_VIOLATION
    assert result.points[2].implied_volatility == pytest.approx(0.3)


def test_smile_requires_matching_nonempty_inputs() -> None:
    with pytest.raises(ValueError, match="matching lengths"):
        run_volatility_smile(
            spot=100.0,
            maturity_years=1.0,
            risk_free_rate=0.03,
            dividend_yield=0.01,
            option_type=OptionType.PUT,
            strikes=[90.0, 100.0],
            option_prices=[3.0],
        )


def test_solver_comparison_agrees_and_serializes() -> None:
    result = compare_implied_volatility_solvers(make_observation(100.0, 0.35))
    assert [point.method for point in result.points] == [
        ImpliedVolatilityMethod.BRENT,
        ImpliedVolatilityMethod.BISECTION,
        ImpliedVolatilityMethod.NEWTON,
    ]
    assert all(point.success for point in result.points)
    recovered = [point.result.implied_volatility for point in result.points if point.result]
    assert recovered == pytest.approx([0.35, 0.35, 0.35], abs=1e-8)
    assert all(
        point.result.absolute_price_error <= 1e-10 for point in result.points if point.result
    )
    assert result.model_dump(mode="json")["points"][0]["method"] == "BRENT"
