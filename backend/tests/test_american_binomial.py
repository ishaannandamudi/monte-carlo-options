import math

import numpy as np
import pytest
from pydantic import ValidationError

from backend.app.analytics.binomial import (
    compare_binomial_to_black_scholes,
    run_binomial_convergence,
)
from backend.app.core.exceptions import UnsupportedExerciseStyleError
from backend.app.models.binomial import (
    AmericanBinomialComparison,
    AmericanBinomialConvergenceResult,
    AmericanBinomialPricingResult,
    BinomialTreeConfig,
    DeterministicBoundaryCase,
    ExerciseRegionDirection,
)
from backend.app.models.greeks import Greek
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import ExerciseStyle, OptionContract, OptionType
from backend.app.models.simulation import SimulationConfig
from backend.app.models.volatility import (
    ImpliedVolatilityMarket,
    ImpliedVolatilityObservation,
)
from backend.app.pricing.binomial import (
    american_exercise_boundary,
    binomial_price,
    crr_parameters,
)
from backend.app.pricing.black_scholes import black_scholes_price
from backend.app.pricing.black_scholes_greeks import black_scholes_greeks
from backend.app.pricing.implied_volatility import (
    european_option_price_bounds,
    solve_implied_volatility,
)
from backend.app.pricing.monte_carlo import MonteCarloPricer
from backend.app.pricing.monte_carlo_greeks import finite_difference_greeks


def american(
    option_type: OptionType,
    *,
    spot: float = 100.0,
    strike: float = 100.0,
    maturity: float = 1.0,
) -> OptionContract:
    return OptionContract(
        option_type=option_type,
        exercise_style=ExerciseStyle.AMERICAN,
        spot=spot,
        strike=strike,
        maturity_years=maturity,
    )


def test_exercise_style_serialization_and_european_engine_rejections() -> None:
    contract = american(OptionType.PUT)
    assert contract.model_dump(mode="json")["exercise_style"] == "AMERICAN"
    european = contract.model_copy(update={"exercise_style": ExerciseStyle.EUROPEAN})
    assert european.model_dump(mode="json")["exercise_style"] == "EUROPEAN"
    market = MarketEnvironment(risk_free_rate=0.05, volatility=0.2)
    with pytest.raises(UnsupportedExerciseStyleError, match="EUROPEAN"):
        black_scholes_price(contract, market)
    with pytest.raises(UnsupportedExerciseStyleError, match="EUROPEAN"):
        black_scholes_greeks(contract, market)
    with pytest.raises(UnsupportedExerciseStyleError, match="EUROPEAN"):
        MonteCarloPricer().price(contract, market, SimulationConfig(num_simulations=10))
    with pytest.raises(UnsupportedExerciseStyleError, match="EUROPEAN"):
        finite_difference_greeks(
            contract,
            market,
            SimulationConfig(num_simulations=10),
            [Greek.DELTA],
        )
    with pytest.raises(UnsupportedExerciseStyleError, match="EUROPEAN"):
        solve_implied_volatility(
            ImpliedVolatilityObservation(
                contract=contract,
                market=ImpliedVolatilityMarket(risk_free_rate=0.05),
                observed_price=6.0,
            )
        )
    with pytest.raises(UnsupportedExerciseStyleError, match="EUROPEAN"):
        european_option_price_bounds(
            contract,
            ImpliedVolatilityMarket(risk_free_rate=0.05),
        )


def test_exercise_diagnostic_tolerance_validation() -> None:
    assert BinomialTreeConfig().exercise_tolerance == 1e-12
    with pytest.raises(ValidationError):
        BinomialTreeConfig(exercise_tolerance=-1.0)
    with pytest.raises(ValidationError):
        BinomialTreeConfig(exercise_tolerance=1e-3)


def test_one_step_american_put_matches_manual_root_max() -> None:
    contract = american(OptionType.PUT, spot=80.0, strike=100.0)
    market = MarketEnvironment(risk_free_rate=0.10, volatility=0.2)
    config = BinomialTreeConfig(steps=1)
    parameters = crr_parameters(contract, market, config)
    down_payoff = max(contract.strike - contract.spot * parameters.down_factor, 0.0)
    up_payoff = max(contract.strike - contract.spot * parameters.up_factor, 0.0)
    continuation = parameters.discount_factor * (
        (1.0 - parameters.risk_neutral_probability) * down_payoff
        + parameters.risk_neutral_probability * up_payoff
    )
    expected = max(contract.strike - contract.spot, continuation)
    result = binomial_price(contract, market, config)
    assert isinstance(result, AmericanBinomialPricingResult)
    assert result.price == pytest.approx(expected, abs=1e-13)


def test_two_step_american_put_matches_manual_backward_induction() -> None:
    contract = american(OptionType.PUT, spot=80.0, strike=100.0)
    market = MarketEnvironment(risk_free_rate=0.10, volatility=0.2)
    config = BinomialTreeConfig(steps=2)
    p = crr_parameters(contract, market, config)
    terminal_stocks = np.array(
        [contract.spot * p.down_factor**2, contract.spot, contract.spot * p.up_factor**2]
    )
    terminal = np.maximum(contract.strike - terminal_stocks, 0.0)
    continuation = p.discount_factor * (
        (1.0 - p.risk_neutral_probability) * terminal[:-1]
        + p.risk_neutral_probability * terminal[1:]
    )
    level_one_stocks = np.array([contract.spot * p.down_factor, contract.spot * p.up_factor])
    level_one = np.maximum(np.maximum(contract.strike - level_one_stocks, 0.0), continuation)
    root_continuation = p.discount_factor * (
        (1.0 - p.risk_neutral_probability) * level_one[0]
        + p.risk_neutral_probability * level_one[1]
    )
    expected = max(contract.strike - contract.spot, root_continuation)
    result = binomial_price(contract, market, config)
    assert isinstance(result, AmericanBinomialPricingResult)
    assert result.price == pytest.approx(expected, abs=1e-13)


def test_non_dividend_call_has_no_early_exercise_and_european_tree_greeks() -> None:
    contract = american(OptionType.CALL)
    market = MarketEnvironment(risk_free_rate=0.05, volatility=0.2, dividend_yield=0.0)
    result = binomial_price(contract, market, BinomialTreeConfig(steps=500))
    assert isinstance(result, AmericanBinomialPricingResult)
    comparison = compare_binomial_to_black_scholes(contract, market, BinomialTreeConfig(steps=500))
    assert isinstance(comparison, AmericanBinomialComparison)
    assert result.price == result.european_tree_price
    assert result.early_exercise_premium == 0.0
    assert result.exercise_node_count == 0
    assert result.early_exercise_occurred is False
    assert comparison.delta.difference == pytest.approx(0.0, abs=1e-14)
    assert comparison.gamma.difference == pytest.approx(0.0, abs=1e-14)
    assert comparison.theta.difference == pytest.approx(0.0, abs=1e-14)


def test_high_dividend_call_and_standard_put_have_early_exercise_value() -> None:
    call_contract = american(OptionType.CALL)
    call_market = MarketEnvironment(risk_free_rate=0.05, volatility=0.2, dividend_yield=0.15)
    call_config = BinomialTreeConfig(steps=400)
    call = binomial_price(call_contract, call_market, call_config)
    put = binomial_price(
        american(OptionType.PUT),
        MarketEnvironment(risk_free_rate=0.05, volatility=0.2),
        BinomialTreeConfig(steps=400),
    )
    assert isinstance(call, AmericanBinomialPricingResult)
    assert isinstance(put, AmericanBinomialPricingResult)
    for result in (call, put):
        assert result.price > result.european_tree_price
        assert result.early_exercise_premium > 0.0
        assert result.early_exercise_premium == result.price - result.european_tree_price
        assert result.early_exercise_occurred
        assert result.boundary_available
        assert result.exercise_node_count > 0
    call_boundary = american_exercise_boundary(call_contract, call_market, call_config)
    assert call_boundary.points
    assert all(
        point.exercise_region_direction is ExerciseRegionDirection.HIGH_STOCK
        for point in call_boundary.points
    )
    assert all(point.region_contiguous for point in call_boundary.points)


def test_deep_itm_put_exercises_at_root_and_boundary_includes_root() -> None:
    contract = american(OptionType.PUT, spot=50.0, strike=100.0)
    market = MarketEnvironment(risk_free_rate=0.05, volatility=0.2)
    config = BinomialTreeConfig(steps=300)
    result = binomial_price(contract, market, config)
    boundary = american_exercise_boundary(contract, market, config)
    assert isinstance(result, AmericanBinomialPricingResult)
    assert result.price == pytest.approx(50.0, abs=1e-13)
    assert result.earliest_exercise_step == 0
    assert result.earliest_exercise_time == 0.0
    assert boundary.points[0].step == 0
    assert boundary.points[0].boundary_stock_price == 50.0


def test_deep_otm_short_low_volatility_options_do_not_get_spurious_exercise() -> None:
    market = MarketEnvironment(risk_free_rate=0.05, volatility=0.01)
    for contract in (
        american(OptionType.CALL, spot=30.0, strike=100.0, maturity=0.1),
        american(OptionType.PUT, spot=200.0, strike=100.0, maturity=0.1),
    ):
        result = binomial_price(contract, market, BinomialTreeConfig(steps=100))
        assert isinstance(result, AmericanBinomialPricingResult)
        assert math.isfinite(result.price)
        assert result.price >= 0.0
        assert result.exercise_node_count == 0


def test_exercise_tolerance_changes_only_diagnostics() -> None:
    contract = american(OptionType.PUT)
    market = MarketEnvironment(risk_free_rate=0.05, volatility=0.2)
    exact = binomial_price(contract, market, BinomialTreeConfig(steps=300, exercise_tolerance=0.0))
    tolerant = binomial_price(
        contract, market, BinomialTreeConfig(steps=300, exercise_tolerance=1e-4)
    )
    assert isinstance(exact, AmericanBinomialPricingResult)
    assert isinstance(tolerant, AmericanBinomialPricingResult)
    assert tolerant.price == exact.price
    assert tolerant.european_tree_price == exact.european_tree_price
    assert tolerant.exercise_node_count <= exact.exercise_node_count


def test_boundary_is_ordered_positive_contiguous_and_excludes_maturity() -> None:
    contract = american(OptionType.PUT)
    config = BinomialTreeConfig(steps=250)
    result = american_exercise_boundary(
        contract,
        MarketEnvironment(risk_free_rate=0.05, volatility=0.2),
        config,
    )
    assert result.points
    assert [point.step for point in result.points] == sorted(point.step for point in result.points)
    assert all(point.step < config.steps for point in result.points)
    assert all(point.time_to_maturity > 0.0 for point in result.points)
    assert all(point.boundary_stock_price > 0.0 for point in result.points)
    assert all(point.region_contiguous for point in result.points)
    assert all(
        point.exercise_region_direction is ExerciseRegionDirection.LOW_STOCK
        for point in result.points
    )
    assert sum(point.number_of_exercise_nodes for point in result.points) == (
        result.exercise_node_count
    )
    assert result.non_contiguous_level_count == 0


def test_boundary_points_match_independently_recomputed_exercise_nodes() -> None:
    contract = american(OptionType.PUT, spot=80.0, strike=100.0)
    market = MarketEnvironment(risk_free_rate=0.10, volatility=0.25)
    config = BinomialTreeConfig(steps=5)
    parameters = crr_parameters(contract, market, config)
    terminal_stocks = np.array(
        [
            contract.spot
            * parameters.up_factor**up_moves
            * parameters.down_factor ** (config.steps - up_moves)
            for up_moves in range(config.steps + 1)
        ]
    )
    values = np.maximum(contract.strike - terminal_stocks, 0.0)
    expected: dict[int, tuple[float, int]] = {}
    for level in range(config.steps - 1, -1, -1):
        continuation = parameters.discount_factor * (
            (1.0 - parameters.risk_neutral_probability) * values[:-1]
            + parameters.risk_neutral_probability * values[1:]
        )
        stocks = np.array(
            [
                contract.spot
                * parameters.up_factor**up_moves
                * parameters.down_factor ** (level - up_moves)
                for up_moves in range(level + 1)
            ]
        )
        intrinsic = np.maximum(contract.strike - stocks, 0.0)
        mask = intrinsic > continuation + config.exercise_tolerance
        if np.any(mask):
            indices = np.flatnonzero(mask)
            expected[level] = (float(stocks[indices[-1]]), int(indices.size))
        values = np.maximum(intrinsic, continuation)

    boundary = american_exercise_boundary(contract, market, config)
    assert [point.step for point in boundary.points] == sorted(expected)
    for point in boundary.points:
        stock, count = expected[point.step]
        assert point.boundary_stock_price == pytest.approx(stock, rel=1e-14)
        assert point.number_of_exercise_nodes == count


@pytest.mark.parametrize(
    ("option_type", "expected"),
    [(OptionType.CALL, 0.0), (OptionType.PUT, 10.0)],
)
def test_zero_maturity_has_no_early_exercise_diagnostics(
    option_type: OptionType,
    expected: float,
) -> None:
    result = binomial_price(
        american(option_type, spot=90.0, strike=100.0, maturity=0.0),
        MarketEnvironment(risk_free_rate=-0.01, volatility=0.3),
        BinomialTreeConfig(steps=25),
    )
    assert isinstance(result, AmericanBinomialPricingResult)
    assert result.price == expected
    assert result.european_tree_price == expected
    assert result.early_exercise_premium == 0.0
    assert result.exercise_node_count == 0
    assert result.deterministic_boundary_case is DeterministicBoundaryCase.ZERO_MATURITY


def test_zero_volatility_optimizes_the_discrete_exercise_grid() -> None:
    contract = american(OptionType.PUT, spot=80.0, strike=100.0)
    market = MarketEnvironment(risk_free_rate=0.10, volatility=0.0)
    config = BinomialTreeConfig(steps=4)
    present_intrinsic = []
    for step in range(config.steps + 1):
        time = contract.maturity_years * step / config.steps
        stock = contract.spot * math.exp(market.risk_free_rate * time)
        present_intrinsic.append(
            math.exp(-market.risk_free_rate * time) * max(contract.strike - stock, 0.0)
        )
    result = binomial_price(contract, market, config)
    assert isinstance(result, AmericanBinomialPricingResult)
    assert result.price == pytest.approx(max(present_intrinsic), abs=1e-13)
    assert result.price == 20.0
    assert result.earliest_exercise_step == 0
    assert result.exercise_node_count == 4
    assert result.deterministic_boundary_case is DeterministicBoundaryCase.ZERO_VOLATILITY


def test_zero_volatility_dividend_call_can_exercise_immediately() -> None:
    result = binomial_price(
        american(OptionType.CALL, spot=120.0, strike=100.0),
        MarketEnvironment(risk_free_rate=0.05, volatility=0.0, dividend_yield=0.20),
        BinomialTreeConfig(steps=4),
    )
    assert isinstance(result, AmericanBinomialPricingResult)
    assert result.price == 20.0
    assert result.early_exercise_premium > 0.0
    assert result.earliest_exercise_step == 0
    assert result.exercise_node_count == 4
    assert result.delta is result.gamma is result.theta is None


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_negative_rate_case_is_finite_when_crr_probability_is_valid(
    option_type: OptionType,
) -> None:
    result = binomial_price(
        american(option_type),
        MarketEnvironment(risk_free_rate=-0.01, volatility=0.25, dividend_yield=0.01),
        BinomialTreeConfig(steps=300),
    )
    assert isinstance(result, AmericanBinomialPricingResult)
    assert math.isfinite(result.price)
    assert result.price + 1e-12 >= result.european_tree_price


def test_american_greeks_are_finite_and_economically_sensible() -> None:
    put = binomial_price(
        american(OptionType.PUT),
        MarketEnvironment(risk_free_rate=0.05, volatility=0.2),
        BinomialTreeConfig(steps=600),
    )
    assert isinstance(put, AmericanBinomialPricingResult)
    assert put.delta is not None and -1.0 <= put.delta <= 0.0
    assert put.gamma is not None and put.gamma >= 0.0
    assert put.theta is not None
    assert all(math.isfinite(value) for value in (put.delta, put.gamma, put.theta))


def test_canonical_american_put_benchmarks() -> None:
    # Widely used CRR reference cases: approximately 6.09 and 4.28 respectively.
    cases = [
        (100.0, 100.0, 0.05, 0.2, 1.0, 6.09),
        (50.0, 50.0, 0.10, 0.4, 5.0 / 12.0, 4.28),
        (36.0, 40.0, 0.06, 0.2, 1.0, 4.486),
        (40.0, 40.0, 0.06, 0.2, 1.0, 2.319),
        (44.0, 40.0, 0.06, 0.2, 1.0, 1.113),
    ]
    for spot, strike, rate, volatility, maturity, expected in cases:
        result = binomial_price(
            american(
                OptionType.PUT,
                spot=spot,
                strike=strike,
                maturity=maturity,
            ),
            MarketEnvironment(risk_free_rate=rate, volatility=volatility),
            BinomialTreeConfig(steps=2_000),
        )
        assert isinstance(result, AmericanBinomialPricingResult)
        assert result.price == pytest.approx(expected, abs=0.01)


def test_american_convergence_reports_stabilization_without_closed_form_claim() -> None:
    result = run_binomial_convergence(
        american(OptionType.PUT),
        MarketEnvironment(risk_free_rate=0.05, volatility=0.2),
        [50, 100, 250, 500],
    )
    assert isinstance(result, AmericanBinomialConvergenceResult)
    assert [point.steps for point in result.points] == [50, 100, 250, 500]
    assert all(point.american_price >= point.european_tree_price for point in result.points)
    assert abs(result.points[-1].american_price - result.points[-2].american_price) < 0.01
    assert all(point.exercise_node_count > 0 for point in result.points)


@pytest.mark.parametrize(
    ("option_type", "dividend_yield"),
    [(OptionType.PUT, 0.0), (OptionType.CALL, 0.0), (OptionType.CALL, 0.15)],
)
def test_american_convergence_stabilizes_for_put_and_call_regimes(
    option_type: OptionType,
    dividend_yield: float,
) -> None:
    result = run_binomial_convergence(
        american(option_type),
        MarketEnvironment(
            risk_free_rate=0.05,
            volatility=0.2,
            dividend_yield=dividend_yield,
        ),
        [100, 250, 500],
    )
    assert isinstance(result, AmericanBinomialConvergenceResult)
    assert abs(result.points[-1].american_price - result.points[-2].american_price) < 0.03
    if option_type is OptionType.CALL and dividend_yield == 0.0:
        assert all(point.early_exercise_premium == 0.0 for point in result.points)
        assert all(point.exercise_node_count == 0 for point in result.points)
    elif option_type is OptionType.CALL:
        assert all(point.early_exercise_premium > 0.0 for point in result.points)
        assert all(point.exercise_node_count > 0 for point in result.points)
