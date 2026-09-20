from backend.app.analytics.qmc import (
    compare_qmc_methods,
    qmc_compute_frontier,
    qmc_convergence,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.models.qmc import QMCComparisonRequest, QMCConvergenceRequest, SamplingMethod


def request() -> QMCConvergenceRequest:
    return QMCConvergenceRequest(
        contract=OptionContract(
            option_type=OptionType.CALL,
            spot=100.0,
            strike=100.0,
            maturity_years=1.0,
        ),
        market=MarketEnvironment(risk_free_rate=0.03, volatility=0.2, dividend_yield=0.01),
        minimum_power=8,
        maximum_power=11,
        randomized_replicates=4,
        root_seed=8,
    )


def test_convergence_points_slopes_and_compute_frontier_metadata() -> None:
    result = qmc_convergence(request())
    assert len(result.points) == 12
    assert {point.method for point in result.points} == set(SamplingMethod)
    assert all(point.error_sqrt_runtime >= 0.0 for point in result.points)
    deterministic = [point for point in result.points if point.method is SamplingMethod.SOBOL]
    assert all(point.standard_error is None for point in deterministic)
    rqmc_slope = next(
        slope for slope in result.slopes if slope.method is SamplingMethod.SCRAMBLED_SOBOL
    )
    assert rqmc_slope.error_points_used >= 2
    assert rqmc_slope.standard_error_points_used >= 2
    frontier = qmc_compute_frontier(request())
    assert [point.metadata.financial_model for point in frontier.points] == [
        "RISK_NEUTRAL_GBM"
    ] * 12
    assert len({point.metadata.reproducibility_signature for point in frontier.points}) > 3


def test_single_budget_method_comparison_is_reproducible() -> None:
    convergence = request()
    comparison_request = QMCComparisonRequest(
        contract=convergence.contract,
        market=convergence.market,
        num_samples=1_024,
        randomized_replicates=4,
        root_seed=7,
    )
    first = compare_qmc_methods(comparison_request)
    second = compare_qmc_methods(comparison_request)
    assert [point.estimate for point in first.points] == [point.estimate for point in second.points]
    assert len(first.points) == 3
