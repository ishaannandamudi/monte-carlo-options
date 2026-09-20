"""Portfolio value, dispatch, failure isolation, and model-risk tests."""

import math

import pytest

from backend.app.models.portfolio import (
    DigitalPosition,
    InstrumentClass,
    Portfolio,
    PortfolioCompleteness,
    PortfolioModelRiskRequest,
    PortfolioPricingModel,
    PortfolioRequest,
    PositionStatus,
)
from backend.app.portfolio.engine import portfolio_model_risk, price_portfolio
from backend.app.pricing.black_scholes import black_scholes_price
from backend.tests.portfolio_test_helpers import market, mixed_portfolio, vanilla


def test_one_position_value_is_quantity_times_unit_price() -> None:
    position = vanilla(quantity=7.5)
    result = price_portfolio(PortfolioRequest(portfolio=Portfolio(positions=[position])))
    expected = 7.5 * black_scholes_price(position.instrument, position.market)
    assert result.total_value == pytest.approx(expected)
    assert result.positions[0].position_value == pytest.approx(expected)
    assert result.completeness is PortfolioCompleteness.COMPLETE


def test_long_short_positions_net_value_but_keep_gross_sides() -> None:
    portfolio = Portfolio(positions=[vanilla("long", 3.0), vanilla("short", -3.0)])
    result = price_portfolio(PortfolioRequest(portfolio=portfolio))
    assert result.total_value == pytest.approx(0.0, abs=1e-14)
    assert result.gross_long_value > 0.0
    assert result.gross_short_value == pytest.approx(result.gross_long_value)


def test_mixed_instrument_portfolio_values_all_supported_classes() -> None:
    result = price_portfolio(PortfolioRequest(portfolio=mixed_portfolio()))
    assert result.completeness is PortfolioCompleteness.COMPLETE
    assert result.successful_position_count == 6
    assert result.failed_position_count == 0
    assert result.total_value is not None and math.isfinite(result.total_value)


def test_incompatible_model_is_isolated_as_partial_failure() -> None:
    invalid = DigitalPosition(
        position_id="bad",
        quantity=1.0,
        instrument_class=InstrumentClass.DIGITAL,
        underlying_id="SPY",
        instrument={
            "option_type": "CALL",
            "spot": 100.0,
            "strike": 100.0,
            "maturity_years": 1.0,
        },
        market=market(),
        model=PortfolioPricingModel.BLACK_SCHOLES,
    )
    result = price_portfolio(PortfolioRequest(portfolio=Portfolio(positions=[vanilla(), invalid])))
    assert result.completeness is PortfolioCompleteness.PARTIAL
    assert result.positions[0].status is PositionStatus.SUCCESS
    assert result.positions[1].status is PositionStatus.FAILED
    assert "incompatible" in (result.positions[1].message or "")


def test_model_risk_reuses_compatible_vanilla_models() -> None:
    portfolio = Portfolio(positions=[vanilla()])
    result = portfolio_model_risk(
        PortfolioModelRiskRequest(
            portfolio=portfolio,
            reference_model=PortfolioPricingModel.BLACK_SCHOLES,
            alternate_models=[
                PortfolioPricingModel.CRR_BINOMIAL,
                PortfolioPricingModel.HESTON_ANALYTICAL,
            ],
        )
    )
    assert result.reference_value is not None
    assert [point.model for point in result.alternatives] == [
        PortfolioPricingModel.CRR_BINOMIAL,
        PortfolioPricingModel.HESTON_ANALYTICAL,
    ]
    assert all(point.portfolio_value is not None for point in result.alternatives)
