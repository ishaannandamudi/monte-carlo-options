"""Heston, basket correlation, expiry, and stochastic reproducibility tests."""

import math

import pytest

from backend.app.models.basket import BasketOptionContract, BasketSimulationConfig, BasketType
from backend.app.models.multi_asset import MultiAssetMarketEnvironment
from backend.app.models.option import OptionType
from backend.app.models.portfolio import (
    BasketPosition,
    HestonScenario,
    MarketScenario,
    Portfolio,
    PortfolioCompleteness,
    PortfolioPricingModel,
    PortfolioRequest,
    PortfolioScenarioRequest,
)
from backend.app.models.simulation import SimulationConfig, VarianceReduction
from backend.app.portfolio.engine import portfolio_greeks, price_portfolio, run_scenario
from backend.app.pricing.black_scholes import black_scholes_price
from backend.app.volatility_surface.synthetic import generate_synthetic_surface
from backend.tests.portfolio_test_helpers import vanilla


def test_monte_carlo_portfolio_is_reproducible_by_position_identity() -> None:
    position = vanilla("mc", model=PortfolioPricingModel.MONTE_CARLO_GBM).model_copy(
        update={
            "simulation": SimulationConfig(
                num_simulations=10_000,
                seed=999,
                variance_reduction=VarianceReduction.ANTITHETIC,
            )
        }
    )
    request = PortfolioRequest(portfolio=Portfolio(positions=[position]))
    first = price_portfolio(request)
    second = price_portfolio(request)
    assert first.total_value == second.total_value
    assert first.positions[0].standard_error == second.positions[0].standard_error


def test_monte_carlo_zero_scenario_reuses_common_random_numbers() -> None:
    position = vanilla("mc", model=PortfolioPricingModel.MONTE_CARLO_GBM).model_copy(
        update={
            "simulation": SimulationConfig(
                num_simulations=2_000,
                variance_reduction=VarianceReduction.ANTITHETIC,
            )
        }
    )
    result = run_scenario(
        PortfolioScenarioRequest(
            portfolio=Portfolio(positions=[position]), scenario=MarketScenario()
        )
    )
    assert result.full_repricing_pnl == 0.0
    assert result.contributions[0].pnl_standard_error is None
    assert "CRN" in result.contributions[0].warnings[0]


def test_heston_parameter_scenario_is_finite_and_deterministic() -> None:
    position = vanilla("heston", model=PortfolioPricingModel.HESTON_ANALYTICAL)
    request = PortfolioScenarioRequest(
        portfolio=Portfolio(positions=[position]),
        scenario=MarketScenario(
            heston=HestonScenario(v0_shift=0.005, theta_shift=0.002, rho_shift=0.05)
        ),
    )
    first = run_scenario(request)
    second = run_scenario(request)
    assert first.full_repricing_pnl == second.full_repricing_pnl
    assert first.full_repricing_pnl is not None and math.isfinite(first.full_repricing_pnl)
    assert first.greek_approximation_pnl is None


def test_basket_per_asset_delta_and_correlation_stress() -> None:
    market = MultiAssetMarketEnvironment(
        spots=(100.0, 100.0),
        risk_free_rate=0.02,
        dividend_yields=(0.0, 0.0),
        volatilities=(0.2, 0.25),
        correlation_matrix=((1.0, 0.2), (0.2, 1.0)),
        asset_ids=("AAA", "BBB"),
    )
    position = BasketPosition(
        position_id="basket",
        quantity=2.0,
        instrument=BasketOptionContract(
            option_type=OptionType.CALL,
            strike=100.0,
            maturity_years=1.0,
            weights=(0.5, 0.5),
            basket_type=BasketType.GEOMETRIC,
        ),
        market=market,
        simulation=BasketSimulationConfig(num_simulations=2_000),
    )
    portfolio = Portfolio(positions=[position])
    greeks = portfolio_greeks(PortfolioRequest(portfolio=portfolio))
    assert set(greeks.positions[0].per_asset_delta) == {"AAA", "BBB"}
    assert greeks.net_greeks.delta is None
    scenario = run_scenario(
        PortfolioScenarioRequest(
            portfolio=portfolio,
            scenario=MarketScenario(equicorrelation_shift_by_position={"basket": 0.3}),
        )
    )
    assert scenario.full_repricing_pnl is not None
    assert math.isfinite(scenario.full_repricing_pnl)
    assert scenario.greek_approximation_pnl is None
    asset_stress = run_scenario(
        PortfolioScenarioRequest(
            portfolio=portfolio,
            scenario=MarketScenario(
                per_underlying_spot_relative_shifts={"AAA": -0.10, "BBB": 0.05}
            ),
        )
    )
    assert asset_stress.full_repricing_pnl is not None
    assert math.isfinite(asset_stress.full_repricing_pnl)
    invalid = run_scenario(
        PortfolioScenarioRequest(
            portfolio=portfolio,
            scenario=MarketScenario(
                correlation_matrix_by_position={"basket": ((1.0, 2.0), (2.0, 1.0))}
            ),
        )
    )
    assert invalid.completeness is PortfolioCompleteness.FAILED
    assert "correlation" in invalid.contributions[0].warnings[0]


def test_surface_based_bsm_is_explicit_and_uses_phase6_local_iv() -> None:
    surface = generate_synthetic_surface(
        maturities=[0.5, 1.0, 2.0],
        log_moneyness=[-0.3, 0.0, 0.3],
        base_volatility=0.35,
        spot=100.0,
        risk_free_rate=0.03,
        dividend_yield=0.01,
    )
    position = vanilla().model_copy(
        update={"volatility_surface": surface, "use_surface_volatility": True}
    )
    result = price_portfolio(PortfolioRequest(portfolio=Portfolio(positions=[position])))
    expected = black_scholes_price(
        position.instrument, position.market.model_copy(update={"volatility": 0.35})
    )
    assert result.total_value == pytest.approx(expected)


def test_time_shift_to_expiry_uses_payoff_without_negative_maturity() -> None:
    result = run_scenario(
        PortfolioScenarioRequest(
            portfolio=Portfolio(positions=[vanilla()]),
            scenario=MarketScenario(time_shift_years=5.0),
        )
    )
    assert result.scenario_value == 0.0
