"""Opt-in Phase 17 scaling smoke benchmarks."""

import time

import numpy as np
import pytest

from backend.app.models.risk import (
    CovarianceInput,
    MonteCarloRiskRequest,
    ParametricRiskRequest,
    PortfolioRiskConfig,
    RiskFactorMapping,
    RiskFactorType,
    RiskMethod,
    RiskValuationMode,
)
from backend.app.risk.engine import calculate_monte_carlo_risk, calculate_parametric_risk
from backend.tests.risk_test_helpers import single_portfolio, spot_covariance, spot_mapping


@pytest.mark.performance
@pytest.mark.parametrize("factor_count", [10, 50, 100, 500])
def test_parametric_factor_scaling(factor_count: int) -> None:
    factor_ids = [f"f-{index}" for index in range(factor_count)]
    mappings = [
        RiskFactorMapping(
            factor_id=identifier,
            factor_type=RiskFactorType.CUSTOM_GAUSSIAN,
            explicit_position_exposures={"call": 1.0 / factor_count},
        )
        for identifier in factor_ids
    ]
    covariance = CovarianceInput(
        factor_ids=factor_ids,
        covariance_matrix=tuple(
            tuple(float(i == j) * 0.0001 for j in range(factor_count)) for i in range(factor_count)
        ),
    )
    started = time.perf_counter()
    result = calculate_parametric_risk(
        ParametricRiskRequest(
            portfolio=single_portfolio(),
            factor_mappings=mappings,
            covariance=covariance,
            config=PortfolioRiskConfig(
                method=RiskMethod.PARAMETRIC_NORMAL, valuation_mode=RiskValuationMode.DELTA
            ),
        )
    )
    assert result.value_at_risk > 0.0
    assert time.perf_counter() - started < 10.0


@pytest.mark.performance
@pytest.mark.parametrize("simulations", [100_000, 500_000])
def test_monte_carlo_delta_scaling(simulations: int) -> None:
    result = calculate_monte_carlo_risk(
        MonteCarloRiskRequest(
            portfolio=single_portfolio(),
            factor_mappings=spot_mapping(),
            covariance=spot_covariance(),
            config=PortfolioRiskConfig(
                method=RiskMethod.MONTE_CARLO,
                valuation_mode=RiskValuationMode.DELTA,
                simulations=simulations,
                batch_size=25_000,
                include_contributions=False,
                include_tail_scenarios=False,
            ),
        )
    )
    assert result.simulations == simulations
    assert np.isfinite(result.value_at_risk)
