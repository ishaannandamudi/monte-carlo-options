"""Opt-in Phase 16 scaling checks; ordinary CI excludes these tests."""

import time

import pytest

from backend.app.models.portfolio import (
    Portfolio,
    PortfolioRequest,
    ScenarioAxis,
    ScenarioGridAxis,
    ScenarioGridRequest,
)
from backend.app.portfolio.engine import portfolio_greeks, price_portfolio, run_scenario_grid
from backend.tests.portfolio_test_helpers import vanilla


@pytest.mark.performance
@pytest.mark.parametrize("count", [10, 100, 1_000])
def test_vanilla_portfolio_scaling(count: int) -> None:
    portfolio = Portfolio(positions=[vanilla(f"p-{index}") for index in range(count)])
    started = time.perf_counter()
    value = price_portfolio(PortfolioRequest(portfolio=portfolio))
    valuation_seconds = time.perf_counter() - started
    assert value.successful_position_count == count
    started = time.perf_counter()
    greeks = portfolio_greeks(PortfolioRequest(portfolio=portfolio))
    greek_seconds = time.perf_counter() - started
    assert greeks.valuation.successful_position_count == count
    assert valuation_seconds < 10.0
    assert greek_seconds < 30.0


@pytest.mark.performance
def test_one_hundred_position_twenty_five_cell_grid() -> None:
    portfolio = Portfolio(positions=[vanilla(f"p-{index}") for index in range(100)])
    result = run_scenario_grid(
        ScenarioGridRequest(
            portfolio=portfolio,
            axes=[
                ScenarioGridAxis(
                    axis=ScenarioAxis.SPOT_RELATIVE,
                    values=[-0.1, -0.05, 0.0, 0.05, 0.1],
                ),
                ScenarioGridAxis(
                    axis=ScenarioAxis.VOLATILITY_ABSOLUTE,
                    values=[-0.04, -0.02, 0.0, 0.02, 0.04],
                ),
            ],
        )
    )
    assert len(result.cells) == 25
