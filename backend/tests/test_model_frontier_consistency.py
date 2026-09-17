import math

from backend.app.analytics.model_comparison import compute_frontier, run_consistency_suite
from backend.app.models.binomial import BinomialTreeConfig
from backend.app.models.heston import HestonParameters, HestonSimulationConfig
from backend.app.models.market import MarketEnvironment
from backend.app.models.model_comparison import (
    ComputeBudgetType,
    ComputeFrontierRequest,
    ConsistencyRequest,
    ModelComparisonConfig,
    ModelComparisonRequest,
    ModelRiskFlag,
    PricingModel,
)
from backend.app.models.option import OptionContract, OptionType
from backend.app.models.simulation import SimulationConfig, VarianceReduction


def inputs() -> tuple[OptionContract, MarketEnvironment]:
    return (
        OptionContract(option_type=OptionType.CALL, spot=100.0, strike=100.0, maturity_years=1.0),
        MarketEnvironment(risk_free_rate=0.03, volatility=0.2, dividend_yield=0.01),
    )


def test_gbm_compute_frontier_has_sorted_reproducible_budgets_and_errors() -> None:
    contract, market = inputs()
    request = ComputeFrontierRequest(
        comparison=ModelComparisonRequest(
            contract=contract,
            market=market,
            config=ModelComparisonConfig(
                selected_models=[PricingModel.MONTE_CARLO_GBM], deterministic_seed=123
            ),
            gbm_monte_carlo=SimulationConfig(
                num_simulations=1_000,
                variance_reduction=VarianceReduction.ANTITHETIC_CONTROL_VARIATE,
            ),
        ),
        model=PricingModel.MONTE_CARLO_GBM,
        budgets=[4_000, 1_000, 2_000],
        reference_model=PricingModel.BLACK_SCHOLES,
    )
    first = compute_frontier(request)
    second = compute_frontier(request)
    assert [point.budget for point in first.points] == [1_000, 2_000, 4_000]
    assert [point.price for point in first.points] == [point.price for point in second.points]
    assert first.natural_convergence_variable is ComputeBudgetType.PATHS
    assert all(point.runtime_ms >= 0.0 for point in first.points)
    assert all(
        point.absolute_error == abs(point.price - point.reference_price) for point in first.points
    )
    assert all(point.standard_error is not None for point in first.points)
    assert all(point.approximate_memory_bytes is not None for point in first.points)


def test_crr_compute_frontier_uses_tree_steps() -> None:
    contract, market = inputs()
    result = compute_frontier(
        ComputeFrontierRequest(
            comparison=ModelComparisonRequest(
                contract=contract,
                market=market,
                config=ModelComparisonConfig(selected_models=[PricingModel.CRR_BINOMIAL]),
            ),
            model=PricingModel.CRR_BINOMIAL,
            budgets=[25, 50, 100],
            reference_model=PricingModel.BLACK_SCHOLES,
        )
    )
    assert result.natural_convergence_variable is ComputeBudgetType.TREE_STEPS
    assert result.points[-1].absolute_error < 0.03


def test_heston_frontier_can_vary_time_steps_at_fixed_paths() -> None:
    contract, market = inputs()
    result = compute_frontier(
        ComputeFrontierRequest(
            comparison=ModelComparisonRequest(
                contract=contract,
                market=market,
                config=ModelComparisonConfig(selected_models=[PricingModel.HESTON_MONTE_CARLO]),
                heston_parameters=HestonParameters(
                    v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7
                ),
                heston_monte_carlo=HestonSimulationConfig(num_simulations=2_000, num_time_steps=5),
            ),
            model=PricingModel.HESTON_MONTE_CARLO,
            budgets=[5, 10],
            budget_type=ComputeBudgetType.TIME_STEPS,
            reference_model=PricingModel.HESTON_ANALYTICAL,
        )
    )
    assert result.natural_convergence_variable is ComputeBudgetType.TIME_STEPS
    assert [point.budget for point in result.points] == [5, 10]
    assert all(point.standard_error is not None for point in result.points)


def test_european_consistency_suite_checks_all_equivalent_engines() -> None:
    contract, market = inputs()
    result = run_consistency_suite(
        ConsistencyRequest(
            contract=contract,
            market=market,
            simulation=SimulationConfig(
                num_simulations=12_000,
                variance_reduction=VarianceReduction.ANTITHETIC_CONTROL_VARIATE,
            ),
            tree=BinomialTreeConfig(steps=500),
            heston_monte_carlo=HestonSimulationConfig(num_simulations=5_000, num_time_steps=20),
            tolerance=0.05,
            seed=44,
        )
    )
    names = {check.name for check in result.checks}
    assert names == {
        "BSM_VS_GBM_MONTE_CARLO",
        "BSM_VS_CRR",
        "BSM_VS_HESTON_DETERMINISTIC_VARIANCE",
        "HESTON_ANALYTICAL_VS_FULL_TRUNCATION_EULER",
        "HESTON_ANALYTICAL_VS_QUADRATIC_EXPONENTIAL",
    }
    assert all(
        check.left_price is not None and math.isfinite(check.left_price) for check in result.checks
    )
    assert all(
        check.right_price is not None and math.isfinite(check.right_price)
        for check in result.checks
    )
    assert ModelRiskFlag.NUMERICAL_CONSISTENCY_WARNING not in result.flags


def test_consistency_failure_generates_flag_without_changing_result() -> None:
    contract, market = inputs()
    result = run_consistency_suite(
        ConsistencyRequest(
            contract=contract,
            market=market,
            simulation=SimulationConfig(num_simulations=2),
            tree=BinomialTreeConfig(steps=1),
            heston_monte_carlo=HestonSimulationConfig(num_simulations=2, num_time_steps=1),
            tolerance=0.0,
            seed=2,
        )
    )
    assert ModelRiskFlag.NUMERICAL_CONSISTENCY_WARNING in result.flags
    assert any(not check.passed for check in result.checks)
