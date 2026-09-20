"""Canonical Greek scaling, net/gross aggregation, and grouping tests."""

import pytest

from backend.app.models.portfolio import Portfolio, PortfolioRequest
from backend.app.portfolio.engine import portfolio_greeks
from backend.tests.portfolio_test_helpers import mixed_portfolio, vanilla


def test_long_short_greeks_net_to_zero_while_gross_remains() -> None:
    result = portfolio_greeks(
        PortfolioRequest(
            portfolio=Portfolio(positions=[vanilla("long", 2.0), vanilla("short", -2.0)])
        )
    )
    assert result.net_greeks.delta == pytest.approx(0.0, abs=1e-14)
    assert result.net_greeks.vega == pytest.approx(0.0, abs=1e-14)
    assert result.gross_greeks.delta is not None and result.gross_greeks.delta > 0.0
    assert result.gross_greeks.vega is not None and result.gross_greeks.vega > 0.0
    assert result.scaled_gross_greeks.delta_1pct is not None


def test_scaled_greek_units_are_exact_transformations() -> None:
    result = portfolio_greeks(PortfolioRequest(portfolio=Portfolio(positions=[vanilla()])))
    position = result.positions[0]
    assert position.scaled_position_greeks.dollar_delta == pytest.approx(
        position.position_greeks.delta * 100.0  # type: ignore[operator]
    )
    assert position.scaled_position_greeks.delta_1pct == pytest.approx(
        position.position_greeks.delta * 100.0 * 0.01  # type: ignore[operator]
    )
    assert position.scaled_position_greeks.gamma_1pct == pytest.approx(
        0.5 * position.position_greeks.gamma * 1.0**2  # type: ignore[operator]
    )
    assert position.scaled_position_greeks.vega_1vol_point == pytest.approx(
        position.position_greeks.vega * 0.01  # type: ignore[operator]
    )


def test_grouping_is_sorted_and_underlying_delta_identity_holds() -> None:
    portfolio = Portfolio(
        positions=[
            vanilla("z", underlying_id="ZZZ"),
            vanilla("a", underlying_id="AAA"),
        ]
    )
    result = portfolio_greeks(PortfolioRequest(portfolio=portfolio))
    assert [group.key for group in result.by_underlying] == ["AAA", "ZZZ"]
    assert sum(group.greeks.delta for group in result.by_underlying) == pytest.approx(
        result.net_greeks.delta
    )
    assert [group.key for group in result.by_expiry] == ["1"]
    assert [group.key for group in result.by_option_type] == ["CALL"]
    assert result.concentration.largest_absolute_delta is not None


def test_mixed_instrument_greeks_keep_unavailable_basket_metrics_explicit() -> None:
    result = portfolio_greeks(PortfolioRequest(portfolio=mixed_portfolio()))
    assert result.valuation.successful_position_count == 6
    assert len(result.positions) == 6
    basket = next(item for item in result.positions if item.position_id == "basket")
    assert basket.unavailable_metrics == ["delta", "gamma", "vega"]
    assert set(basket.per_asset_delta) == {"AAA", "BBB"}
