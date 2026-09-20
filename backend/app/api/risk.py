"""Bounded HTTP adapters for Phase 17 portfolio market-risk analytics."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.api.portfolio import _enforce_portfolio
from backend.app.core.config import Settings, get_settings
from backend.app.models.portfolio import PortfolioRequest
from backend.app.models.risk import (
    HistoricalRiskRequest,
    MonteCarloRiskRequest,
    ParametricRiskRequest,
    PortfolioRiskReport,
    PortfolioRiskReportRequest,
    PortfolioRiskRequest,
    RiskMethod,
    RiskValuationMode,
    VaRBacktestRequest,
    VaRBacktestResult,
)
from backend.app.risk.backtesting import backtest_var
from backend.app.risk.engine import (
    calculate_historical_risk,
    calculate_monte_carlo_risk,
    calculate_parametric_risk,
    calculate_portfolio_risk,
    calculate_risk_report,
)

router = APIRouter(prefix="/api/risk", tags=["portfolio-market-risk"])


def _reject(detail: str) -> None:
    raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail)


def _scenario_count(request: PortfolioRiskRequest) -> int:
    if request.config.method is RiskMethod.MONTE_CARLO:
        return request.config.simulations
    if request.historical_dataset is not None:
        return len(request.historical_dataset.scenarios)
    return 1


def _enforce_risk(request: PortfolioRiskRequest, settings: Settings) -> None:
    positions = len(request.portfolio.positions)
    if positions > settings.max_risk_positions:
        _reject("risk portfolio position count exceeds API limit")
    scenarios = _scenario_count(request)
    if (
        request.historical_dataset is not None
        and len(request.historical_dataset.scenarios) > settings.max_risk_historical_scenarios
    ):
        _reject("historical risk scenario count exceeds API limit")
    if (
        request.config.method is RiskMethod.MONTE_CARLO
        and request.config.simulations > settings.max_risk_monte_carlo_simulations
    ):
        _reject("risk Monte Carlo simulation count exceeds API limit")
    if request.config.max_tail_scenarios > settings.max_risk_tail_output:
        _reject("risk tail output exceeds API limit")
    if (
        request.config.include_contributions
        and positions > settings.max_risk_contribution_positions
    ):
        _reject("risk contribution position count exceeds API limit")
    if (
        request.config.valuation_mode is RiskValuationMode.FULL_REVALUATION
        and scenarios * positions > settings.max_risk_scenario_position_evaluations
    ):
        _reject("risk scenario-position evaluations exceed API limit")
    repetitions = (
        scenarios + 6 if request.config.valuation_mode is RiskValuationMode.FULL_REVALUATION else 6
    )
    _enforce_portfolio(
        PortfolioRequest(portfolio=request.portfolio, config=request.portfolio_config),
        settings,
        repetitions=repetitions,
    )


def _unified(
    request: HistoricalRiskRequest | ParametricRiskRequest | MonteCarloRiskRequest,
) -> PortfolioRiskRequest:
    return PortfolioRiskRequest(
        portfolio=request.portfolio,
        factor_mappings=request.factor_mappings,
        config=request.config,
        portfolio_config=request.portfolio_config,
        historical_dataset=(
            request.dataset
            if isinstance(request, HistoricalRiskRequest)
            else request.historical_dataset
        ),
        covariance=None if isinstance(request, HistoricalRiskRequest) else request.covariance,
    )


@router.post("/historical", response_model=PortfolioRiskReport)
def historical_risk(
    request: HistoricalRiskRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> PortfolioRiskReport:
    _enforce_risk(_unified(request), settings)
    return calculate_historical_risk(request)


@router.post("/parametric", response_model=PortfolioRiskReport)
def parametric_risk(
    request: ParametricRiskRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> PortfolioRiskReport:
    _enforce_risk(_unified(request), settings)
    return calculate_parametric_risk(request)


@router.post("/monte-carlo", response_model=PortfolioRiskReport)
def monte_carlo_risk(
    request: MonteCarloRiskRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> PortfolioRiskReport:
    _enforce_risk(_unified(request), settings)
    return calculate_monte_carlo_risk(request)


@router.post("/portfolio", response_model=PortfolioRiskReport)
def portfolio_risk(
    request: PortfolioRiskRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> PortfolioRiskReport:
    _enforce_risk(request, settings)
    return calculate_portfolio_risk(request)


@router.post("/contributions", response_model=PortfolioRiskReport)
def risk_contributions(
    request: PortfolioRiskRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> PortfolioRiskReport:
    if not request.config.include_contributions:
        _reject("contributions endpoint requires include_contributions=true")
    _enforce_risk(request, settings)
    return calculate_portfolio_risk(request)


@router.post("/backtest", response_model=VaRBacktestResult)
def risk_backtest(
    request: VaRBacktestRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> VaRBacktestResult:
    if len(request.observations) > settings.max_risk_backtest_observations:
        _reject("risk backtest observation count exceeds API limit")
    return backtest_var(request)


@router.post("/report", response_model=PortfolioRiskReport)
def risk_and_stress_report(
    request: PortfolioRiskReportRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> PortfolioRiskReport:
    if len(request.stress_scenarios) > settings.max_portfolio_stress_scenarios:
        _reject("risk report stress scenario count exceeds API limit")
    if len(request.stress_scenarios) > settings.max_portfolio_scenario_points:
        _reject("risk report scenario count exceeds API limit")
    _enforce_risk(request, settings)
    if request.stress_scenarios:
        _enforce_portfolio(
            PortfolioRequest(portfolio=request.portfolio, config=request.portfolio_config),
            settings,
            repetitions=len(request.stress_scenarios) + 6,
        )
    return calculate_risk_report(request)
