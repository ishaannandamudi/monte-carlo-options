import math

import numpy as np
import pytest
from numpy.random import Generator
from numpy.typing import NDArray
from pydantic import ValidationError

from backend.app.analytics.longstaff_schwartz import compare_longstaff_schwartz
from backend.app.analytics.statistics import antithetic_standard_error, standard_error
from backend.app.core.exceptions import LongstaffSchwartzError
from backend.app.models.binomial import BinomialTreeConfig
from backend.app.models.longstaff_schwartz import LongstaffSchwartzConfig
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import ExerciseStyle, OptionContract, OptionType
from backend.app.models.simulation import PathSimulationConfig, VarianceReduction
from backend.app.pricing import longstaff_schwartz as lsm_module
from backend.app.pricing.binomial import american_binomial_price
from backend.app.pricing.longstaff_schwartz import (
    LongstaffSchwartzPricer,
    LSMCashflows,
    apply_exercise_policy,
    discounted_cashflows,
    empirical_exercise_boundary,
    fit_exercise_policy,
    longstaff_schwartz_price,
)
from backend.app.simulation.base import PathStochasticProcess
from backend.app.simulation.gbm import GBMProcess
from backend.app.simulation.random import create_rng, spawn_child_seeds


def contract(
    option_type: OptionType = OptionType.PUT, spot: float = 100.0, maturity: float = 1.0
) -> OptionContract:
    return OptionContract(
        option_type=option_type,
        exercise_style=ExerciseStyle.AMERICAN,
        spot=spot,
        strike=100.0,
        maturity_years=maturity,
    )


def fixed_paths() -> NDArray[np.float64]:
    return np.array(
        [
            [100.0, 80.0, 70.0, 90.0],
            [100.0, 90.0, 90.0, 60.0],
            [100.0, 100.0, 80.0, 100.0],
            [100.0, 110.0, 110.0, 120.0],
        ]
    )


def tiny_config(**updates: object) -> LongstaffSchwartzConfig:
    return LongstaffSchwartzConfig.model_validate(
        {
            "num_paths": 4,
            "num_time_steps": 3,
            "basis_degree": 0,
            "min_regression_paths": 2,
            "evaluation_mode": "IN_SAMPLE",
            **updates,
        }
    )


def test_fixed_paths_itm_targets_and_earlier_cashflow_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = []
    original = lsm_module.fit_continuation

    def record(*args, **kwargs):
        captured.append((args[0].copy(), args[1].copy()))
        return original(*args, **kwargs)

    monkeypatch.setattr(lsm_module, "fit_continuation", record)
    market = MarketEnvironment(risk_free_rate=math.log(2), volatility=0.2)
    paths = fixed_paths()
    training = fit_exercise_policy(paths, contract(maturity=3), market, tiny_config())
    # At t2: future terminal payments are halved. Every ITM path exercises.
    np.testing.assert_array_equal(captured[0][0], [70, 90, 80])
    np.testing.assert_allclose(captured[0][1], [5, 20, 0])
    # At t1: the t2 payments 30 and 10 are discounted once, rather than terminal values.
    np.testing.assert_array_equal(captured[1][0], [80, 90])
    np.testing.assert_allclose(captured[1][1], [15, 5])
    np.testing.assert_array_equal(training.sample.cashflows, [20, 10, 20, 0])
    np.testing.assert_array_equal(training.sample.exercise_indices, [1, 2, 2, 3])
    present = discounted_cashflows(training.sample, rate=math.log(2), dt=1)
    np.testing.assert_allclose(present, [10, 2.5, 5, 0])
    assert training.policy.training_continuation_price == pytest.approx(4.375)
    assert training.policy.dates[0].coefficients == pytest.approx((10.0,))
    assert training.policy.dates[1].coefficients == pytest.approx((25 / 3,))
    np.testing.assert_array_equal(paths, fixed_paths())


def test_frozen_forward_policy_never_refits_and_stops_at_first_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market = MarketEnvironment(risk_free_rate=math.log(2), volatility=0.2)
    trained = fit_exercise_policy(fixed_paths(), contract(maturity=3), market, tiny_config())
    before = trained.policy.model_dump_json()

    def forbidden(*args, **kwargs):
        raise AssertionError("valuation attempted to refit")

    monkeypatch.setattr(lsm_module, "fit_continuation", forbidden)
    valuation = np.array([[100.0, 70.0, 50.0, 10.0], [100.0, 110.0, 110.0, 150.0]])
    result = apply_exercise_policy(valuation, trained.policy)
    np.testing.assert_array_equal(result.exercise_indices, [1, 3])
    np.testing.assert_array_equal(result.cashflows, [30, 0])
    assert trained.policy.model_dump_json() == before


def test_insufficient_regression_continues_all_candidates() -> None:
    trained = fit_exercise_policy(
        fixed_paths(),
        contract(maturity=3),
        MarketEnvironment(risk_free_rate=0, volatility=0.2),
        tiny_config(min_regression_paths=100),
    )
    np.testing.assert_array_equal(trained.sample.exercise_indices, [3, 3, 3, 3])
    assert all(date.fallback_used for date in trained.policy.dates)


class FixedProcess(PathStochasticProcess):
    def __init__(self, batches: list[NDArray[np.float64]]) -> None:
        self.batches = iter(batches)
        self.draws: list[float] = []

    def simulate_paths(
        self,
        contract: OptionContract,
        market: MarketEnvironment,
        config: PathSimulationConfig,
        rng: Generator,
    ) -> NDArray[np.float64]:
        self.draws.append(float(rng.standard_normal()))
        return next(self.batches).copy()


@pytest.mark.parametrize("mode", [VarianceReduction.NONE, VarianceReduction.ANTITHETIC])
def test_in_sample_se_uses_discounted_final_cashflows(mode: VarianceReduction) -> None:
    result = LongstaffSchwartzPricer(FixedProcess([fixed_paths()])).price(
        contract(maturity=3),
        MarketEnvironment(risk_free_rate=math.log(2), volatility=0.2),
        tiny_config(variance_reduction=mode),
    )
    observations = np.array([10.0, 2.5, 5.0, 0.0])
    expected = (
        standard_error(observations)
        if mode is VarianceReduction.NONE
        else antithetic_standard_error(observations)
    )
    assert result.price == pytest.approx(4.375)
    assert result.standard_error == pytest.approx(expected)
    assert result.estimator_variance == pytest.approx(expected**2)
    assert result.early_exercise_fraction == 0.75
    assert result.zero_cashflow_fraction == 0.25


def test_out_of_sample_se_excludes_training_and_preserves_odd_clusters() -> None:
    valuation = np.array(
        [
            [100.0, 70.0, 50.0, 10.0],
            [100.0, 110.0, 110.0, 120.0],
            [100.0, 90.0, 90.0, 60.0],
            [100.0, 100.0, 80.0, 100.0],
            [100.0, 110.0, 110.0, 130.0],
        ]
    )
    process = FixedProcess([fixed_paths(), valuation])
    config = tiny_config(num_paths=5, regression_paths=4, evaluation_mode="OUT_OF_SAMPLE")
    market = MarketEnvironment(risk_free_rate=math.log(2), volatility=0.2)
    result = LongstaffSchwartzPricer(process).price(contract(maturity=3), market, config)
    observations = np.array([15.0, 0.0, 2.5, 5.0, 0.0])
    assert result.price == pytest.approx(observations.mean())
    assert result.standard_error == pytest.approx(antithetic_standard_error(observations))
    assert result.training_seed != result.valuation_seed
    assert process.draws[0] != process.draws[1]
    assert result.num_paths == 5 and result.regression_paths == 4


def test_out_of_sample_reconstructs_independent_streams_and_is_reproducible() -> None:
    config = LongstaffSchwartzConfig(num_paths=501, regression_paths=601, num_time_steps=8)
    market = MarketEnvironment(risk_free_rate=0.05, volatility=0.2)
    result = longstaff_schwartz_price(contract(), market, config)
    seeds = spawn_child_seeds(config.seed, 2)
    assert (result.training_seed, result.valuation_seed) == tuple(seeds)
    train = GBMProcess().simulate_paths(
        contract(),
        market,
        PathSimulationConfig(num_simulations=601, num_time_steps=8),
        create_rng(seeds[0]),
    )
    val = GBMProcess().simulate_paths(
        contract(),
        market,
        PathSimulationConfig(num_simulations=501, num_time_steps=8),
        create_rng(seeds[1]),
    )
    assert not np.array_equal(train[:501], val)
    policy = fit_exercise_policy(train, contract(), market, config).policy
    cashflows = apply_exercise_policy(val, policy)
    discounted = discounted_cashflows(cashflows, rate=0.05, dt=1 / 8)
    assert result.price == float(discounted.mean())
    assert result.standard_error == antithetic_standard_error(discounted)
    excluded = {
        "runtime_ms",
        "simulation_runtime_ms",
        "regression_runtime_ms",
        "valuation_runtime_ms",
    }
    again = longstaff_schwartz_price(contract(), market, config)
    assert result.model_dump(exclude=excluded) == again.model_dump(exclude=excluded)
    other = longstaff_schwartz_price(contract(), market, config.model_copy(update={"seed": 43}))
    assert result.price != other.price


@pytest.mark.parametrize(
    "option_type,spot,rate,dividend",
    [
        (OptionType.PUT, 80, 0.1, 0),
        (OptionType.CALL, 120, 0.05, 0.2),
        (OptionType.CALL, 100, 0.05, 0),
        (OptionType.PUT, 100, -0.05, 0.01),
    ],
)
def test_zero_volatility_matches_deterministic_crr_and_skips_simulation(
    option_type: OptionType,
    spot: float,
    rate: float,
    dividend: float,
) -> None:
    c = contract(option_type, spot)
    m = MarketEnvironment(risk_free_rate=rate, volatility=0, dividend_yield=dividend)
    result = LongstaffSchwartzPricer(FixedProcess([])).price(
        c, m, LongstaffSchwartzConfig(num_paths=1, num_time_steps=10)
    )
    crr = american_binomial_price(c, m, BinomialTreeConfig(steps=10))
    assert result.price == pytest.approx(crr.price, abs=1e-12)
    assert result.standard_error == 0
    assert result.regression_count == 0 and result.deterministic
    assert result.simulation_runtime_ms == 0


@pytest.mark.parametrize(
    "option_type,spot,expected",
    [(OptionType.PUT, 80, 20), (OptionType.CALL, 120, 20), (OptionType.CALL, 80, 0)],
)
def test_expiry_returns_intrinsic_without_simulation(
    option_type: OptionType, spot: float, expected: float
) -> None:
    result = LongstaffSchwartzPricer(FixedProcess([])).price(
        contract(option_type, spot, 0),
        MarketEnvironment(risk_free_rate=0.05, volatility=0.2),
        LongstaffSchwartzConfig(),
    )
    assert result.price == expected and result.standard_error == 0
    assert not result.early_exercise_occurred and not result.immediate_exercise
    assert not result.boundary_diagnostics and not result.regression_diagnostics


def test_deep_itm_root_decision_replaces_all_stopping_diagnostics() -> None:
    result = longstaff_schwartz_price(
        contract(spot=30),
        MarketEnvironment(risk_free_rate=0.05, volatility=0.2),
        LongstaffSchwartzConfig(num_paths=2000, num_time_steps=10),
    )
    assert result.price == 70
    assert result.immediate_exercise and result.early_exercise_fraction == 1
    assert result.immediate_exercise_fraction == 1 and result.maturity_exercise_fraction == 0
    assert result.mean_exercise_time == 0 and result.standard_error == 0
    assert result.boundary_diagnostics[0].time_index == 0


@pytest.mark.parametrize(
    "option_type,spot,rate,dividend",
    [
        (OptionType.PUT, 100, 0.05, 0),
        (OptionType.PUT, 90, 0.05, 0.01),
        (OptionType.PUT, 110, 0.05, 0),
        (OptionType.PUT, 100, -0.01, 0.01),
        (OptionType.CALL, 100, 0.05, 0),
        (OptionType.CALL, 100, 0.05, 0.15),
    ],
)
@pytest.mark.parametrize("variance", [VarianceReduction.NONE, VarianceReduction.ANTITHETIC])
def test_stochastic_american_values_against_crr(
    option_type: OptionType,
    spot: float,
    rate: float,
    dividend: float,
    variance: VarianceReduction,
) -> None:
    comparison = compare_longstaff_schwartz(
        contract(option_type, spot),
        MarketEnvironment(risk_free_rate=rate, volatility=0.2, dividend_yield=dividend),
        LongstaffSchwartzConfig(
            num_paths=12000, regression_paths=15000, num_time_steps=40, variance_reduction=variance
        ),
        BinomialTreeConfig(steps=800),
    )
    result = comparison.longstaff_schwartz
    # Sampling error plus an explicit 0.15 grid/policy approximation budget; not a CI assertion.
    assert comparison.absolute_error < 5 * result.standard_error + 0.15
    assert math.isfinite(result.price)
    if option_type is OptionType.CALL and dividend == 0 and rate >= 0:
        assert result.no_early_exercise_theorem and result.early_exercise_fraction == 0
    if option_type is OptionType.CALL and dividend == 0.15:
        assert result.early_exercise_fraction > 0.05
        assert comparison.lsm_early_exercise_premium > 0


@pytest.mark.parametrize("option_type", list(OptionType))
def test_empirical_boundaries_use_only_final_stops_and_supported_extremes(
    option_type: OptionType,
) -> None:
    c = contract(option_type)
    paths = np.array(
        [
            [100.0, 80.0, 70.0, 60.0],
            [100.0, 90.0, 80.0, 70.0],
            [100.0, 95.0, 85.0, 75.0],
            [100.0, 110.0, 120.0, 130.0],
        ]
    )
    sample = LSMCashflows(np.array([20.0, 10.0, 5.0, 0.0]), np.array([1, 1, 2, 3], dtype=np.int64))
    points = empirical_exercise_boundary(paths, sample, c, min_observations=2)
    assert len(points) == 1 and points[0].time_index == 1
    assert points[0].exercise_observations == 2 and points[0].sample_support == 0.5
    assert points[0].estimated_boundary_stock == (90 if option_type is OptionType.PUT else 80)
    assert not empirical_exercise_boundary(paths, sample, c, min_observations=5)


def test_sparse_schedule_and_validation() -> None:
    config = tiny_config(exercise_steps=(2, 3))
    trained = fit_exercise_policy(
        fixed_paths(),
        contract(maturity=3),
        MarketEnvironment(risk_free_rate=math.log(2), volatility=0.2),
        config,
    )
    assert [date.time_index for date in trained.policy.dates] == [2]
    assert np.all(trained.sample.exercise_indices >= 2)
    for updates in [
        {"exercise_steps": [1, 2]},
        {"exercise_steps": [2, 1, 3]},
        {"exercise_steps": [0, 3]},
        {"basis_degree": 6},
        {"variance_reduction": "CONTROL_VARIATE"},
        {"num_paths": 0},
        {"num_time_steps": 0},
    ]:
        with pytest.raises(ValidationError):
            tiny_config(**updates)


def test_european_contract_rejected_by_lsm() -> None:
    with pytest.raises(LongstaffSchwartzError, match="AMERICAN"):
        longstaff_schwartz_price(
            contract().model_copy(update={"exercise_style": ExerciseStyle.EUROPEAN}),
            MarketEnvironment(risk_free_rate=0.05, volatility=0.2),
        )


@pytest.mark.parametrize("indices", [np.array([1, 2]), np.array([-1, 2, 3, 3]), np.full(4, 4)])
def test_boundary_rejects_invalid_stopping_indices(indices: NDArray[np.int64]) -> None:
    with pytest.raises(LongstaffSchwartzError, match="cash-flow sample"):
        empirical_exercise_boundary(
            fixed_paths(), LSMCashflows(np.ones(4), indices), contract(), min_observations=1
        )


def test_expiry_does_not_evaluate_irrelevant_extreme_market_arithmetic() -> None:
    result = LongstaffSchwartzPricer(FixedProcess([])).price(
        contract(spot=90, maturity=0),
        MarketEnvironment(risk_free_rate=1e308, dividend_yield=-1e308, volatility=1e308),
        LongstaffSchwartzConfig(),
    )
    assert result.price == 10 and result.standard_error == 0
    assert result.maturity_exercise_fraction == 1
    assert result.simulation_runtime_ms == result.regression_runtime_ms == 0
