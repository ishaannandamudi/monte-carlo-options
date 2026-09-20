"""Deterministic fixtures shared by Phase 17 market-risk tests."""

from datetime import UTC, datetime, timedelta

from backend.app.models.portfolio import Portfolio
from backend.app.models.risk import (
    CovarianceInput,
    HistoricalRiskDataset,
    HistoricalRiskFactorObservation,
    HistoricalRiskScenario,
    RiskFactorMapping,
    RiskFactorType,
)
from backend.tests.portfolio_test_helpers import vanilla


def single_portfolio() -> Portfolio:
    return Portfolio(portfolio_id="risk", positions=[vanilla("call", quantity=2.0)])


def spot_mapping() -> list[RiskFactorMapping]:
    return [
        RiskFactorMapping(
            factor_id="SPY_SPOT_RETURN",
            factor_type=RiskFactorType.SPOT_RETURN,
            underlying_id="SPY",
        )
    ]


def spot_covariance(variance: float = 0.0004, mean: float = 0.0) -> CovarianceInput:
    return CovarianceInput(
        factor_ids=["SPY_SPOT_RETURN"],
        covariance_matrix=((variance,),),
        mean_changes=(mean,),
    )


def history(values: list[float]) -> HistoricalRiskDataset:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    return HistoricalRiskDataset(
        scenarios=[
            HistoricalRiskScenario(
                timestamp=start + timedelta(days=index),
                observations=[
                    HistoricalRiskFactorObservation(factor_id="SPY_SPOT_RETURN", value=value)
                ],
            )
            for index, value in enumerate(values)
        ]
    )
