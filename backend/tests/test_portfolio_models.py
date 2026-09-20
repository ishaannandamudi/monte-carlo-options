"""Validation and serialization tests for Phase 16 portfolio contracts."""

import pytest
from pydantic import ValidationError

from backend.app.models.portfolio import (
    HestonScenario,
    MarketScenario,
    Portfolio,
    PortfolioModelRiskRequest,
    PortfolioPricingModel,
    PortfolioStressRequest,
    ScenarioAxis,
    ScenarioGridAxis,
    ScenarioGridRequest,
)
from backend.tests.portfolio_test_helpers import mixed_portfolio, vanilla


def test_portfolio_round_trip_preserves_discriminated_mixed_positions() -> None:
    portfolio = mixed_portfolio()
    restored = Portfolio.model_validate_json(portfolio.model_dump_json())
    assert restored == portfolio
    assert len(restored.positions) == 6


def test_portfolio_rejects_duplicate_position_ids_and_zero_quantity() -> None:
    with pytest.raises(ValidationError, match="unique"):
        Portfolio(positions=[vanilla("same"), vanilla("same")])
    with pytest.raises(ValidationError, match="nonzero"):
        vanilla(quantity=0.0)


def test_stress_requires_names_and_model_risk_requires_unique_alternates() -> None:
    portfolio = Portfolio(positions=[vanilla()])
    with pytest.raises(ValidationError, match="must have a name"):
        PortfolioStressRequest(
            portfolio=portfolio,
            scenarios=[MarketScenario(spot_relative_shift=-0.1)],
        )
    with pytest.raises(ValidationError, match="unique"):
        PortfolioModelRiskRequest(
            portfolio=portfolio,
            reference_model=PortfolioPricingModel.BLACK_SCHOLES,
            alternate_models=[
                PortfolioPricingModel.CRR_BINOMIAL,
                PortfolioPricingModel.CRR_BINOMIAL,
            ],
        )


def test_scenario_grid_contract_supports_more_than_two_axes() -> None:
    request = ScenarioGridRequest(
        portfolio=Portfolio(positions=[vanilla()]),
        axes=[
            ScenarioGridAxis(axis=ScenarioAxis.SPOT_RELATIVE, values=[-0.1, 0.0, 0.1]),
            ScenarioGridAxis(axis=ScenarioAxis.VOLATILITY_ABSOLUTE, values=[0.0, 0.05]),
            ScenarioGridAxis(axis=ScenarioAxis.RATE_ABSOLUTE, values=[0.0, 0.01]),
        ],
    )
    assert ScenarioGridRequest.model_validate(request.model_dump()).portfolio == request.portfolio


def test_heston_scenario_rejects_ambiguous_override_and_shift() -> None:
    with pytest.raises(ValidationError, match="overrides or additive shifts"):
        HestonScenario(
            v0_shift=0.01,
            overrides={"v0": 0.04, "kappa": 1.5, "theta": 0.04, "sigma_v": 0.3, "rho": -0.7},
        )
