"""Full repricing, local approximation, grid, stress, and P&L-explain tests."""

import math

import pytest

from backend.app.models.barrier import (
    BarrierDirection,
    BarrierMonitoringMode,
    BarrierOptionContract,
    BarrierStyle,
)
from backend.app.models.digital import DigitalOptionContract
from backend.app.models.option import OptionType
from backend.app.models.portfolio import (
    BarrierPosition,
    DigitalPosition,
    InstrumentClass,
    MarketScenario,
    Portfolio,
    PortfolioPnLExplainRequest,
    PortfolioPricingModel,
    PortfolioScenarioRequest,
    PortfolioStressRequest,
    ScenarioAxis,
    ScenarioFlag,
    ScenarioGridAxis,
    ScenarioGridRequest,
)
from backend.app.portfolio.engine import (
    explain_portfolio_pnl,
    run_scenario,
    run_scenario_grid,
    run_stress_scenarios,
)
from backend.tests.portfolio_test_helpers import market, vanilla


@pytest.mark.parametrize(
    "scenario",
    [
        MarketScenario(spot_relative_shift=0.01),
        MarketScenario(spot_absolute_shift=1.0),
        MarketScenario(volatility_absolute_shift=0.01),
        MarketScenario(volatility_relative_shift=0.05),
        MarketScenario(rate_absolute_shift=0.0025),
        MarketScenario(dividend_yield_shift=0.0025),
        MarketScenario(time_shift_years=1.0 / 365.0),
        MarketScenario(
            spot_relative_shift=-0.05,
            volatility_absolute_shift=0.03,
            rate_absolute_shift=-0.005,
            time_shift_years=7.0 / 365.0,
        ),
    ],
)
def test_all_single_asset_scenario_dimensions_full_reprice(scenario: MarketScenario) -> None:
    result = run_scenario(
        PortfolioScenarioRequest(portfolio=Portfolio(positions=[vanilla()]), scenario=scenario)
    )
    assert result.full_repricing_pnl is not None
    assert math.isfinite(result.full_repricing_pnl)
    assert result.scenario_value == pytest.approx(
        result.base_value + result.full_repricing_pnl  # type: ignore[operator]
    )


def test_zero_shock_reproduces_base_value_exactly() -> None:
    result = run_scenario(
        PortfolioScenarioRequest(
            portfolio=Portfolio(positions=[vanilla()]), scenario=MarketScenario()
        )
    )
    assert result.scenario_value == result.base_value
    assert result.full_repricing_pnl == 0.0
    assert result.greek_approximation_pnl == 0.0
    assert result.contributions[0].base_market is not None
    assert result.contributions[0].scenario_market == result.contributions[0].base_market


def test_small_smooth_shock_has_accurate_delta_gamma_approximation() -> None:
    result = run_scenario(
        PortfolioScenarioRequest(
            portfolio=Portfolio(positions=[vanilla()]),
            scenario=MarketScenario(spot_relative_shift=1e-3),
        )
    )
    assert result.approximation_residual is not None
    assert abs(result.approximation_residual) < 1e-5


@pytest.mark.parametrize(
    ("scenario", "tolerance"),
    [
        (MarketScenario(volatility_absolute_shift=1e-4), 1e-5),
        (MarketScenario(rate_absolute_shift=1e-4), 1e-5),
        (MarketScenario(time_shift_years=1.0 / 3650.0), 1e-5),
    ],
)
def test_small_factor_shocks_match_local_greek_approximation(
    scenario: MarketScenario, tolerance: float
) -> None:
    result = run_scenario(
        PortfolioScenarioRequest(portfolio=Portfolio(positions=[vanilla()]), scenario=scenario)
    )
    assert result.approximation_residual is not None
    assert abs(result.approximation_residual) < tolerance


def test_dividend_shock_full_reprices_without_fabricated_approximation() -> None:
    result = run_scenario(
        PortfolioScenarioRequest(
            portfolio=Portfolio(positions=[vanilla()]),
            scenario=MarketScenario(dividend_yield_shift=0.001),
        )
    )
    assert result.full_repricing_pnl is not None
    assert result.greek_approximation_pnl is None


def test_large_shock_reports_exact_residual_and_contribution_identity() -> None:
    portfolio = Portfolio(positions=[vanilla("one", 2.0), vanilla("two", -0.5)])
    result = run_scenario(
        PortfolioScenarioRequest(
            portfolio=portfolio, scenario=MarketScenario(spot_relative_shift=-0.25)
        )
    )
    assert result.approximation_residual == pytest.approx(
        result.full_repricing_pnl - result.greek_approximation_pnl  # type: ignore[operator]
    )
    assert sum(item.full_repricing_pnl for item in result.contributions) == pytest.approx(
        result.full_repricing_pnl
    )
    assert sum(group.pnl for group in result.by_instrument_class) == pytest.approx(
        result.full_repricing_pnl
    )


def test_grid_cartesian_order_and_shape_are_deterministic() -> None:
    result = run_scenario_grid(
        ScenarioGridRequest(
            portfolio=Portfolio(positions=[vanilla()]),
            axes=[
                ScenarioGridAxis(axis=ScenarioAxis.SPOT_RELATIVE, values=[-0.1, 0.1]),
                ScenarioGridAxis(axis=ScenarioAxis.VOLATILITY_ABSOLUTE, values=[-0.02, 0.0, 0.02]),
            ],
        )
    )
    assert result.shape == [2, 3]
    assert len(result.cells) == 6
    assert result.cells[0].coordinates == {
        "SPOT_RELATIVE": -0.1,
        "VOLATILITY_ABSOLUTE": -0.02,
    }
    assert result.cells[-1].coordinates == {
        "SPOT_RELATIVE": 0.1,
        "VOLATILITY_ABSOLUTE": 0.02,
    }


def test_named_stress_and_pnl_explain_identity() -> None:
    portfolio = Portfolio(positions=[vanilla()])
    stresses = run_stress_scenarios(
        PortfolioStressRequest(
            portfolio=portfolio,
            scenarios=[
                MarketScenario(name="sell-off", spot_relative_shift=-0.2),
                MarketScenario(name="vol-spike", volatility_absolute_shift=0.1),
            ],
        )
    )
    assert [point.name for point in stresses.scenarios] == ["sell-off", "vol-spike"]
    explain = explain_portfolio_pnl(
        PortfolioPnLExplainRequest(
            portfolio=portfolio,
            scenario=MarketScenario(
                spot_relative_shift=0.02,
                volatility_absolute_shift=0.01,
                rate_absolute_shift=0.001,
                time_shift_years=1.0 / 365.0,
            ),
        )
    )
    assert explain.terms.explained_total is not None
    assert explain.terms.unexplained_or_higher_order is not None
    assert (
        explain.terms.explained_total + explain.terms.unexplained_or_higher_order
    ) == pytest.approx(explain.total_full_reprice_pnl)
    assert sum(item.full_reprice_pnl for item in explain.positions) == pytest.approx(
        explain.total_full_reprice_pnl
    )
    for term in ("delta", "gamma", "vega", "theta", "rho"):
        assert sum(getattr(item.terms, term) for item in explain.positions) == pytest.approx(
            getattr(explain.terms, term)
        )


def test_digital_and_barrier_discontinuities_do_not_block_full_repricing() -> None:
    digital = DigitalPosition(
        position_id="digital",
        quantity=1.0,
        instrument_class=InstrumentClass.DIGITAL,
        underlying_id="SPY",
        instrument=DigitalOptionContract(
            option_type=OptionType.CALL,
            spot=100.0,
            strike=100.0,
            maturity_years=0.25,
        ),
        market=market(),
    )
    barrier = BarrierPosition(
        position_id="barrier",
        quantity=1.0,
        underlying_id="SPY",
        model=PortfolioPricingModel.BARRIER_ANALYTICAL,
        instrument=BarrierOptionContract(
            option_type=OptionType.CALL,
            barrier_direction=BarrierDirection.UP,
            barrier_style=BarrierStyle.OUT,
            spot=100.0,
            strike=100.0,
            barrier=101.0,
            maturity_years=0.25,
            monitoring_mode=BarrierMonitoringMode.BROWNIAN_BRIDGE,
        ),
        market=market(),
    )
    result = run_scenario(
        PortfolioScenarioRequest(
            portfolio=Portfolio(positions=[digital, barrier]),
            scenario=MarketScenario(spot_relative_shift=0.005),
        )
    )
    assert result.full_repricing_pnl is not None
    assert all(
        ScenarioFlag.DISCONTINUOUS_PAYOFF in contribution.flags
        for contribution in result.contributions
    )
