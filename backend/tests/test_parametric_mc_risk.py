"""Analytical parametric and batched correlated-factor Monte Carlo risk tests."""

import math

import numpy as np
import pytest
from scipy.stats import norm

from backend.app.models.portfolio import Portfolio
from backend.app.models.risk import (
    CovarianceInput,
    MonteCarloRiskRequest,
    ParametricRiskRequest,
    PortfolioRiskConfig,
    RiskDistribution,
    RiskFactorMapping,
    RiskFactorType,
    RiskHorizon,
    RiskMethod,
    RiskValuationMode,
)
from backend.app.risk.engine import (
    _mc_shock_batches,
    calculate_monte_carlo_risk,
    calculate_parametric_risk,
)
from backend.tests.portfolio_test_helpers import vanilla
from backend.tests.risk_test_helpers import history, single_portfolio, spot_covariance, spot_mapping


def _parametric(
    covariance: CovarianceInput | None = None, **config: object
) -> ParametricRiskRequest:
    return ParametricRiskRequest(
        portfolio=single_portfolio(),
        factor_mappings=spot_mapping(),
        covariance=covariance or spot_covariance(),
        config=PortfolioRiskConfig(
            method=RiskMethod.PARAMETRIC_NORMAL,
            valuation_mode=RiskValuationMode.DELTA,
            **config,
        ),
    )


def test_single_factor_normal_var_and_es_match_formula() -> None:
    result = calculate_parametric_risk(_parametric(confidence_level=0.99))
    exposure = result.contributions.factors[0].exposure  # type: ignore[union-attr]
    sigma = abs(exposure) * 0.02
    z = norm.ppf(0.99)
    assert result.value_at_risk == pytest.approx(z * sigma)
    assert result.expected_shortfall == pytest.approx(sigma * norm.pdf(z) / 0.01)


def test_component_var_and_es_euler_identities() -> None:
    result = calculate_parametric_risk(_parametric())
    contributions = result.contributions
    assert contributions is not None
    assert contributions.component_var_sum == pytest.approx(result.value_at_risk)
    assert contributions.expected_shortfall_contribution_sum == pytest.approx(
        result.expected_shortfall
    )


def test_incremental_var_leave_one_out() -> None:
    result = calculate_parametric_risk(_parametric(include_incremental_var=True))
    contribution = result.contributions.positions[0]  # type: ignore[union-attr]
    assert contribution.incremental_var == pytest.approx(result.value_at_risk)


def test_zero_covariance_and_mean_produce_zero_risk() -> None:
    result = calculate_parametric_risk(_parametric(spot_covariance(0.0)))
    assert result.value_at_risk == 0.0
    assert result.expected_shortfall == 0.0


def test_horizon_scaling_follows_sqrt_time_for_zero_mean() -> None:
    one = calculate_parametric_risk(_parametric())
    nine = calculate_parametric_risk(_parametric(horizon=RiskHorizon(trading_days=9)))
    assert nine.value_at_risk == pytest.approx(3.0 * one.value_at_risk)


def test_multi_factor_covariance_and_diversification_are_exact() -> None:
    portfolio = Portfolio(
        positions=[vanilla("a", underlying_id="AAA"), vanilla("b", underlying_id="BBB")]
    )
    mappings = [
        RiskFactorMapping(
            factor_id="a", factor_type=RiskFactorType.SPOT_RETURN, underlying_id="AAA"
        ),
        RiskFactorMapping(
            factor_id="b", factor_type=RiskFactorType.SPOT_RETURN, underlying_id="BBB"
        ),
    ]
    covariance = CovarianceInput(
        factor_ids=["a", "b"], covariance_matrix=((0.0004, 0.0001), (0.0001, 0.0009))
    )
    request = ParametricRiskRequest(
        portfolio=portfolio,
        factor_mappings=mappings,
        covariance=covariance,
        config=PortfolioRiskConfig(
            method=RiskMethod.PARAMETRIC_NORMAL, valuation_mode=RiskValuationMode.DELTA
        ),
    )
    result = calculate_parametric_risk(request)
    exposures = np.asarray([item.exposure for item in result.contributions.factors])  # type: ignore[union-attr]
    expected_sigma = math.sqrt(
        float(exposures @ np.asarray(covariance.covariance_matrix) @ exposures)
    )
    assert result.distribution_summary.pnl_standard_deviation == pytest.approx(expected_sigma)


def test_parametric_covariance_can_be_estimated_from_history() -> None:
    request = ParametricRiskRequest(
        portfolio=single_portfolio(),
        factor_mappings=spot_mapping(),
        historical_dataset=history([-0.02, -0.01, 0.0, 0.01, 0.02]),
        config=PortfolioRiskConfig(
            method=RiskMethod.PARAMETRIC_NORMAL,
            valuation_mode=RiskValuationMode.DELTA,
        ),
    )
    result = calculate_parametric_risk(request)
    assert result.covariance_diagnostics.source.value == "HISTORICAL_SAMPLE"  # type: ignore[union-attr]
    assert result.covariance_diagnostics.observation_count == 5  # type: ignore[union-attr]


def _monte_carlo(**config: object) -> MonteCarloRiskRequest:
    values: dict[str, object] = {
        "method": RiskMethod.MONTE_CARLO,
        "valuation_mode": RiskValuationMode.DELTA,
        "simulations": 30_000,
        "batch_size": 3_001,
    }
    values.update(config)
    return MonteCarloRiskRequest(
        portfolio=single_portfolio(),
        factor_mappings=spot_mapping(),
        covariance=spot_covariance(),
        config=PortfolioRiskConfig.model_validate(values),
    )


def test_monte_carlo_same_seed_is_reproducible() -> None:
    first = calculate_monte_carlo_risk(_monte_carlo(root_seed=123))
    second = calculate_monte_carlo_risk(_monte_carlo(root_seed=123))
    assert first.value_at_risk == second.value_at_risk
    assert first.expected_shortfall == second.expected_shortfall
    assert first.worst_scenarios == second.worst_scenarios


def test_monte_carlo_different_seed_changes_tail_estimates() -> None:
    first = calculate_monte_carlo_risk(_monte_carlo(root_seed=1))
    second = calculate_monte_carlo_risk(_monte_carlo(root_seed=2))
    assert first.value_at_risk != second.value_at_risk


def test_mc_delta_normal_converges_to_parametric() -> None:
    analytical = calculate_parametric_risk(_parametric(confidence_level=0.99))
    simulated = calculate_monte_carlo_risk(_monte_carlo(confidence_level=0.99, root_seed=9))
    assert simulated.value_at_risk == pytest.approx(analytical.value_at_risk, rel=0.035)
    assert simulated.expected_shortfall == pytest.approx(analytical.expected_shortfall, rel=0.04)


def test_delta_gamma_mode_changes_distribution_and_reports_limitation() -> None:
    delta = calculate_monte_carlo_risk(_monte_carlo(root_seed=7))
    delta_gamma = calculate_monte_carlo_risk(
        _monte_carlo(root_seed=7, valuation_mode=RiskValuationMode.DELTA_GAMMA_DIAGONAL)
    )
    assert delta_gamma.value_at_risk != delta.value_at_risk
    assert any("diagonal Gamma" in warning for warning in delta_gamma.warnings)


def test_student_t_mode_is_finite_and_heavier_tailed() -> None:
    normal = calculate_monte_carlo_risk(_monte_carlo(root_seed=4))
    student = calculate_monte_carlo_risk(
        _monte_carlo(
            root_seed=4,
            distribution=RiskDistribution.MULTIVARIATE_STUDENT_T,
            student_t_degrees_of_freedom=4.0,
        )
    )
    assert math.isfinite(student.value_at_risk)
    assert student.expected_shortfall > normal.expected_shortfall


def test_mc_es_contributions_sum_exactly() -> None:
    result = calculate_monte_carlo_risk(_monte_carlo(root_seed=5))
    assert result.contributions.expected_shortfall_contribution_sum == pytest.approx(  # type: ignore[union-attr]
        result.expected_shortfall
    )


def test_mc_incremental_var_leave_one_out() -> None:
    result = calculate_monte_carlo_risk(
        _monte_carlo(root_seed=5, include_incremental_var=True, simulations=5_000)
    )
    assert result.contributions.positions[0].incremental_var == pytest.approx(  # type: ignore[union-attr]
        result.value_at_risk
    )


def test_simulated_factor_correlation_matches_target() -> None:
    portfolio = Portfolio(positions=[vanilla("a")])
    mappings = [
        RiskFactorMapping(
            factor_id="a",
            factor_type=RiskFactorType.CUSTOM_GAUSSIAN,
            explicit_position_exposures={"a": 1.0},
        ),
        RiskFactorMapping(
            factor_id="b",
            factor_type=RiskFactorType.CUSTOM_GAUSSIAN,
            explicit_position_exposures={"a": 1.0},
        ),
    ]
    covariance = CovarianceInput(factor_ids=["a", "b"], covariance_matrix=((1.0, 0.7), (0.7, 1.0)))
    request = MonteCarloRiskRequest(
        portfolio=portfolio,
        factor_mappings=mappings,
        covariance=covariance,
        config=PortfolioRiskConfig(
            method=RiskMethod.MONTE_CARLO,
            valuation_mode=RiskValuationMode.DELTA,
            simulations=50_000,
            batch_size=5_000,
            include_contributions=False,
        ),
    )
    samples = np.vstack(
        [
            batch
            for _, batch in _mc_shock_batches(
                request, np.zeros(2), np.asarray(covariance.covariance_matrix)
            )
        ]
    )
    assert np.corrcoef(samples, rowvar=False)[0, 1] == pytest.approx(0.7, abs=0.015)


def test_antithetic_odd_count_returns_exact_requested_count() -> None:
    request = _monte_carlo(simulations=10_001, antithetic=True, batch_size=777)
    result = calculate_monte_carlo_risk(request)
    assert result.simulations == 10_001
    assert result.scenario_count == 10_001


def test_bounded_full_revaluation_monte_carlo() -> None:
    request = _monte_carlo(
        simulations=101,
        batch_size=17,
        valuation_mode=RiskValuationMode.FULL_REVALUATION,
        root_seed=12,
    )
    result = calculate_monte_carlo_risk(request)
    assert result.scenario_count == 101
    assert math.isfinite(result.value_at_risk)
    assert result.contributions.expected_shortfall_contribution_sum == pytest.approx(  # type: ignore[union-attr]
        result.expected_shortfall
    )
