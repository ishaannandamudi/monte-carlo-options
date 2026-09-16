"""Memory-efficient European and American Cox-Ross-Rubinstein pricing."""

import logging
import math
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from backend.app.core.exceptions import (
    BinomialTreeError,
    InvalidRiskNeutralProbabilityError,
    NumericalError,
)
from backend.app.core.numerics import finite_array_exp, finite_exp, require_finite
from backend.app.models.binomial import (
    AmericanBinomialPricingResult,
    BinomialPricingResult,
    BinomialTreeConfig,
    CRRParameters,
    DeterministicBoundaryCase,
    ExerciseBoundaryPoint,
    ExerciseBoundaryResult,
    ExerciseRegionDirection,
    TreeModel,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import ExerciseStyle, OptionContract, OptionType
from backend.app.models.volatility import ImpliedVolatilityMarket
from backend.app.payoffs.vanilla import intrinsic_value, vanilla_payoff
from backend.app.pricing.base import Pricer
from backend.app.pricing.black_scholes import black_scholes_price
from backend.app.pricing.implied_volatility import european_option_price_bounds

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _BackwardInductionState:
    """Root and the small early levels needed for native tree Greeks."""

    root: float
    level_one: NDArray[np.float64] | None
    level_two: NDArray[np.float64] | None


@dataclass(frozen=True)
class _AmericanInductionState:
    """American/equivalent-European roots and O(N) exercise metadata."""

    root: float
    european_root: float
    level_one: NDArray[np.float64] | None
    level_two: NDArray[np.float64] | None
    boundary_points: tuple[ExerciseBoundaryPoint, ...]
    exercise_node_count: int
    non_contiguous_level_count: int


@dataclass(frozen=True)
class _AmericanComputation:
    result: AmericanBinomialPricingResult
    boundary_points: tuple[ExerciseBoundaryPoint, ...]


def crr_parameters(
    contract: OptionContract,
    market: MarketEnvironment,
    config: BinomialTreeConfig,
) -> CRRParameters:
    """Return validated CRR factors for a positive-maturity stochastic tree."""

    if config.model is not TreeModel.CRR:
        raise BinomialTreeError(f"unsupported tree model: {config.model}")
    maturity = contract.maturity_years
    if maturity <= 0.0:
        raise BinomialTreeError("CRR parameters require positive maturity")
    if market.volatility <= 0.0:
        raise BinomialTreeError("CRR parameters require positive volatility")
    dt = maturity / config.steps
    if not math.isfinite(dt) or dt <= 0.0:
        raise NumericalError("CRR time step is outside finite positive range")
    log_up = market.volatility * math.sqrt(dt)
    up = finite_exp(log_up, context="CRR up factor")
    down = finite_exp(-log_up, context="CRR down factor")
    denominator = up - down
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise NumericalError("CRR up/down factors are numerically indistinguishable")
    growth = finite_exp(
        (market.risk_free_rate - market.dividend_yield) * dt,
        context="CRR risk-neutral growth",
    )
    raw_probability = (growth - down) / denominator
    if not math.isfinite(raw_probability):
        raise InvalidRiskNeutralProbabilityError("CRR risk-neutral probability is non-finite")
    tolerance = config.probability_tolerance
    if raw_probability < -tolerance or raw_probability > 1.0 + tolerance:
        raise InvalidRiskNeutralProbabilityError(
            "CRR risk-neutral probability "
            f"{raw_probability:.17g} is outside [0, 1]; increase steps or revise parameters"
        )
    probability = min(max(raw_probability, 0.0), 1.0)
    discount = finite_exp(-market.risk_free_rate * dt, context="CRR discount factor")
    logger.debug(
        "crr_parameters steps=%d dt=%g up=%g down=%g probability=%g discount=%g",
        config.steps,
        dt,
        up,
        down,
        probability,
        discount,
    )
    return CRRParameters(
        steps=config.steps,
        dt=dt,
        up_factor=up,
        down_factor=down,
        risk_neutral_growth=growth,
        risk_neutral_probability=probability,
        discount_factor=discount,
    )


def crr_terminal_stock_prices(
    contract: OptionContract,
    parameters: CRRParameters,
) -> NDArray[np.float64]:
    """Return terminal nodes indexed by up-move count ``j=0,...,N``."""

    indices = np.arange(parameters.steps + 1, dtype=np.float64)
    log_up = math.log(parameters.up_factor)
    log_prices = math.log(contract.spot) + (2.0 * indices - parameters.steps) * log_up
    prices = finite_array_exp(log_prices, context="CRR terminal stock prices")
    return np.asarray(prices, dtype=np.float64)


def _continuation_values(
    next_values: NDArray[np.float64],
    parameters: CRRParameters,
) -> NDArray[np.float64]:
    """Return one exercise-neutral CRR continuation step."""

    probability = parameters.risk_neutral_probability
    with np.errstate(over="ignore", invalid="ignore"):
        continuation = parameters.discount_factor * (
            (1.0 - probability) * next_values[:-1] + probability * next_values[1:]
        )
    require_finite(continuation, context="CRR backward induction")
    return np.asarray(continuation, dtype=np.float64)


def _backward_induction(
    terminal_values: NDArray[np.float64],
    parameters: CRRParameters,
) -> _BackwardInductionState:
    """Reduce one vector in O(N) memory while retaining only levels one and two."""

    values = np.asarray(terminal_values, dtype=np.float64)
    if values.shape != (parameters.steps + 1,):
        raise BinomialTreeError("terminal value vector does not match configured steps")
    level_one = values.copy() if parameters.steps == 1 else None
    level_two = values.copy() if parameters.steps == 2 else None
    for level in range(parameters.steps - 1, -1, -1):
        values = _continuation_values(values, parameters)
        if level == 2:
            level_two = values.copy()
        elif level == 1:
            level_one = values.copy()
    root = float(values[0])
    if not math.isfinite(root) or root < 0.0:
        raise NumericalError("CRR root price is invalid")
    return _BackwardInductionState(root=root, level_one=level_one, level_two=level_two)


def _stock_prices_at_level(
    contract: OptionContract,
    parameters: CRRParameters,
    level: int,
) -> NDArray[np.float64]:
    """Return ascending stock nodes for one CRR level without retaining prior levels."""

    if level == 0:
        return np.asarray([contract.spot], dtype=np.float64)
    indices = np.arange(level + 1, dtype=np.float64)
    log_prices = math.log(contract.spot) + (2.0 * indices - level) * math.log(parameters.up_factor)
    return np.asarray(
        finite_array_exp(log_prices, context=f"CRR stock prices at level {level}"),
        dtype=np.float64,
    )


def _exercise_boundary_point(
    contract: OptionContract,
    parameters: CRRParameters,
    level: int,
    stock_prices: NDArray[np.float64],
    exercise_mask: NDArray[np.bool_],
) -> tuple[ExerciseBoundaryPoint, bool]:
    """Summarize one exercise mask and report whether its stock region is contiguous."""

    indices = np.flatnonzero(exercise_mask)
    if indices.size == 0:
        raise BinomialTreeError("cannot form an exercise boundary from an empty mask")
    if contract.option_type is OptionType.PUT:
        boundary_index = int(indices[-1])
        expected = np.arange(boundary_index + 1)
        direction = ExerciseRegionDirection.LOW_STOCK
    else:
        boundary_index = int(indices[0])
        expected = np.arange(boundary_index, level + 1)
        direction = ExerciseRegionDirection.HIGH_STOCK
    contiguous = bool(np.array_equal(indices, expected))
    time_from_valuation = level * parameters.dt
    time_to_maturity = (parameters.steps - level) * parameters.dt
    return (
        ExerciseBoundaryPoint(
            step=level,
            time_from_valuation=time_from_valuation,
            time_to_maturity=time_to_maturity,
            boundary_stock_price=float(stock_prices[boundary_index]),
            option_type=contract.option_type,
            exercise_region_direction=direction,
            number_of_exercise_nodes=int(indices.size),
            region_contiguous=contiguous,
        ),
        contiguous,
    )


def _american_backward_induction(
    terminal_values: NDArray[np.float64],
    contract: OptionContract,
    parameters: CRRParameters,
    exercise_tolerance: float,
) -> _AmericanInductionState:
    """Apply node exercise in O(N) working memory and retain O(N) boundaries."""

    values = np.asarray(terminal_values, dtype=np.float64)
    if values.shape != (parameters.steps + 1,):
        raise BinomialTreeError("terminal value vector does not match configured steps")
    european_values = values.copy()
    level_one = values.copy() if parameters.steps == 1 else None
    level_two = values.copy() if parameters.steps == 2 else None
    points: list[ExerciseBoundaryPoint] = []
    exercise_node_count = 0
    non_contiguous_level_count = 0

    for level in range(parameters.steps - 1, -1, -1):
        continuation = _continuation_values(values, parameters)
        european_values = _continuation_values(european_values, parameters)
        stock_prices = _stock_prices_at_level(contract, parameters, level)
        exercise_values = vanilla_payoff(stock_prices, contract)
        require_finite(exercise_values, context="American intrinsic values")
        exercise_mask = np.asarray(
            exercise_values > continuation + exercise_tolerance,
            dtype=np.bool_,
        )
        values = np.asarray(np.maximum(exercise_values, continuation), dtype=np.float64)
        require_finite(values, context="American backward induction")
        count = int(np.count_nonzero(exercise_mask))
        if count:
            point, contiguous = _exercise_boundary_point(
                contract,
                parameters,
                level,
                stock_prices,
                exercise_mask,
            )
            points.append(point)
            exercise_node_count += count
            if not contiguous:
                non_contiguous_level_count += 1
        if level == 2:
            level_two = values.copy()
        elif level == 1:
            level_one = values.copy()

    root = float(values[0])
    european_root = float(european_values[0])
    if not math.isfinite(root) or root < 0.0:
        raise NumericalError("American CRR root price is invalid")
    if not math.isfinite(european_root) or european_root < 0.0:
        raise NumericalError("same-tree European root price is invalid")
    points.reverse()
    return _AmericanInductionState(
        root=root,
        european_root=european_root,
        level_one=level_one,
        level_two=level_two,
        boundary_points=tuple(points),
        exercise_node_count=exercise_node_count,
        non_contiguous_level_count=non_contiguous_level_count,
    )


def _tree_greeks(
    contract: OptionContract,
    parameters: CRRParameters,
    state: _BackwardInductionState,
) -> tuple[float | None, float | None, float | None]:
    delta = None
    gamma = None
    theta = None
    if state.level_one is not None:
        stock_down = contract.spot * parameters.down_factor
        stock_up = contract.spot * parameters.up_factor
        denominator = stock_up - stock_down
        if math.isfinite(denominator) and denominator > 0.0:
            delta = float((state.level_one[1] - state.level_one[0]) / denominator)
    if state.level_two is not None:
        stock_dd = contract.spot * parameters.down_factor**2
        stock_ud = contract.spot * parameters.up_factor * parameters.down_factor
        stock_uu = contract.spot * parameters.up_factor**2
        down_spacing = stock_ud - stock_dd
        up_spacing = stock_uu - stock_ud
        center_spacing = 0.5 * (stock_uu - stock_dd)
        if min(down_spacing, up_spacing, center_spacing) > 0.0:
            delta_down = (state.level_two[1] - state.level_two[0]) / down_spacing
            delta_up = (state.level_two[2] - state.level_two[1]) / up_spacing
            gamma = float((delta_up - delta_down) / center_spacing)
        theta = float((state.level_two[1] - state.root) / (2.0 * parameters.dt))
    finite_greeks = [value for value in (delta, gamma, theta) if value is not None]
    if not all(math.isfinite(value) for value in finite_greeks):
        raise NumericalError("CRR native Greeks produced a non-finite value")
    return delta, gamma, theta


def _price_bounds(
    contract: OptionContract,
    market: MarketEnvironment,
) -> tuple[float, float]:
    bounds = european_option_price_bounds(
        contract,
        ImpliedVolatilityMarket(
            risk_free_rate=market.risk_free_rate,
            dividend_yield=market.dividend_yield,
        ),
    )
    return bounds.lower, bounds.upper


def _validate_price_bounds(
    price: float,
    lower: float,
    upper: float,
    tolerance: float,
) -> None:
    if price < lower - tolerance or price > upper + tolerance:
        raise BinomialTreeError(f"CRR price {price} violates European bounds [{lower}, {upper}]")


def _result(
    *,
    price: float,
    contract: OptionContract,
    market: MarketEnvironment,
    config: BinomialTreeConfig,
    runtime_ms: float,
    lower_bound: float,
    upper_bound: float,
    parameters: CRRParameters | None = None,
    delta: float | None = None,
    gamma: float | None = None,
    theta: float | None = None,
    boundary: DeterministicBoundaryCase | None = None,
) -> BinomialPricingResult:
    analytical = black_scholes_price(contract, market)
    signed_error = price - analytical
    absolute_error = abs(signed_error)
    return BinomialPricingResult(
        price=price,
        steps=config.steps,
        model=config.model,
        dt=None if parameters is None else parameters.dt,
        up_factor=None if parameters is None else parameters.up_factor,
        down_factor=None if parameters is None else parameters.down_factor,
        risk_neutral_probability=(
            None if parameters is None else parameters.risk_neutral_probability
        ),
        discount_factor=None if parameters is None else parameters.discount_factor,
        runtime_ms=runtime_ms,
        delta=delta,
        gamma=gamma,
        theta=theta,
        black_scholes_price=analytical,
        signed_error=signed_error,
        absolute_error=absolute_error,
        relative_error=absolute_error / abs(analytical) if analytical != 0.0 else None,
        price_lower_bound=lower_bound,
        price_upper_bound=upper_bound,
        deterministic_boundary_case=boundary,
    )


def _european_contract(contract: OptionContract) -> OptionContract:
    return contract.model_copy(update={"exercise_style": ExerciseStyle.EUROPEAN})


def _american_result(
    *,
    contract: OptionContract,
    market: MarketEnvironment,
    config: BinomialTreeConfig,
    price: float,
    european_tree_price: float,
    runtime_ms: float,
    boundary_points: tuple[ExerciseBoundaryPoint, ...],
    exercise_node_count: int,
    non_contiguous_level_count: int,
    parameters: CRRParameters | None = None,
    dt: float | None = None,
    discount_factor: float | None = None,
    delta: float | None = None,
    gamma: float | None = None,
    theta: float | None = None,
    boundary_case: DeterministicBoundaryCase | None = None,
) -> AmericanBinomialPricingResult:
    """Build and validate compact American diagnostics."""

    if not math.isfinite(price) or price < 0.0:
        raise NumericalError("American CRR price is invalid")
    if not math.isfinite(european_tree_price) or european_tree_price < 0.0:
        raise NumericalError("same-tree European comparator is invalid")
    tolerance = config.arbitrage_tolerance
    intrinsic = intrinsic_value(contract.option_type, contract.spot, contract.strike)
    if price < intrinsic - tolerance:
        raise BinomialTreeError(
            f"American CRR price {price} is below immediate intrinsic value {intrinsic}"
        )
    if (
        contract.option_type is OptionType.CALL
        and market.dividend_yield >= 0.0
        and price > contract.spot + tolerance
    ):
        raise BinomialTreeError(
            "American call price exceeds spot under non-negative dividend yield"
        )
    if (
        contract.option_type is OptionType.PUT
        and market.risk_free_rate >= 0.0
        and price > contract.strike + tolerance
    ):
        raise BinomialTreeError("American put price exceeds strike under a non-negative rate")
    premium = price - european_tree_price
    if premium < -tolerance:
        raise BinomialTreeError("American CRR price is below its same-tree European comparator")
    if premium < 0.0:
        price = european_tree_price
        premium = 0.0
    european = _european_contract(contract)
    analytical = black_scholes_price(european, market)
    earliest = boundary_points[0] if boundary_points else None
    return AmericanBinomialPricingResult(
        price=price,
        exercise_style=ExerciseStyle.AMERICAN,
        steps=config.steps,
        model=config.model,
        dt=parameters.dt if parameters is not None else dt,
        up_factor=None if parameters is None else parameters.up_factor,
        down_factor=None if parameters is None else parameters.down_factor,
        risk_neutral_probability=(
            None if parameters is None else parameters.risk_neutral_probability
        ),
        discount_factor=(parameters.discount_factor if parameters is not None else discount_factor),
        runtime_ms=runtime_ms,
        delta=delta,
        gamma=gamma,
        theta=theta,
        european_tree_price=european_tree_price,
        black_scholes_european_price=analytical,
        early_exercise_premium=premium,
        early_exercise_occurred=earliest is not None,
        earliest_exercise_step=None if earliest is None else earliest.step,
        earliest_exercise_time=(None if earliest is None else earliest.time_from_valuation),
        earliest_boundary_price=(None if earliest is None else earliest.boundary_stock_price),
        exercise_node_count=exercise_node_count,
        boundary_available=earliest is not None,
        non_contiguous_level_count=non_contiguous_level_count,
        deterministic_boundary_case=boundary_case,
    )


def _deterministic_american_computation(
    contract: OptionContract,
    market: MarketEnvironment,
    config: BinomialTreeConfig,
    started: float,
) -> _AmericanComputation:
    """Optimize exercise along the sigma-zero deterministic discrete path."""

    maturity = contract.maturity_years
    dt = maturity / config.steps
    if not math.isfinite(dt) or dt <= 0.0:
        raise NumericalError("deterministic tree time step is outside finite positive range")
    discount = finite_exp(-market.risk_free_rate * dt, context="deterministic tree discount")
    terminal_spot = contract.spot * finite_exp(
        (market.risk_free_rate - market.dividend_yield) * maturity,
        context="deterministic binomial terminal stock",
    )
    terminal_payoff = intrinsic_value(contract.option_type, terminal_spot, contract.strike)
    value = terminal_payoff
    european_tree_price = (
        finite_exp(
            -market.risk_free_rate * maturity,
            context="deterministic binomial full discount",
        )
        * terminal_payoff
    )
    points: list[ExerciseBoundaryPoint] = []
    for level in range(config.steps - 1, -1, -1):
        time = level * dt
        stock = contract.spot * finite_exp(
            (market.risk_free_rate - market.dividend_yield) * time,
            context=f"deterministic stock at level {level}",
        )
        exercise = intrinsic_value(contract.option_type, stock, contract.strike)
        continuation = discount * value
        if not math.isfinite(continuation):
            raise NumericalError("deterministic American continuation value is non-finite")
        if exercise > continuation + config.exercise_tolerance:
            points.append(
                ExerciseBoundaryPoint(
                    step=level,
                    time_from_valuation=time,
                    time_to_maturity=(config.steps - level) * dt,
                    boundary_stock_price=stock,
                    option_type=contract.option_type,
                    exercise_region_direction=(
                        ExerciseRegionDirection.LOW_STOCK
                        if contract.option_type is OptionType.PUT
                        else ExerciseRegionDirection.HIGH_STOCK
                    ),
                    number_of_exercise_nodes=1,
                    region_contiguous=True,
                )
            )
        value = max(exercise, continuation)
    points.reverse()
    runtime_ms = (perf_counter() - started) * 1_000.0
    result = _american_result(
        contract=contract,
        market=market,
        config=config,
        price=value,
        european_tree_price=european_tree_price,
        runtime_ms=runtime_ms,
        boundary_points=tuple(points),
        exercise_node_count=len(points),
        non_contiguous_level_count=0,
        dt=dt,
        discount_factor=discount,
        boundary_case=DeterministicBoundaryCase.ZERO_VOLATILITY,
    )
    return _AmericanComputation(result=result, boundary_points=tuple(points))


def _american_computation(
    contract: OptionContract,
    market: MarketEnvironment,
    config: BinomialTreeConfig,
) -> _AmericanComputation:
    """Compute an American value and compact boundary data in one tree traversal."""

    if contract.exercise_style is not ExerciseStyle.AMERICAN:
        raise BinomialTreeError("American CRR pricing requires AMERICAN exercise style")
    started = perf_counter()
    if contract.maturity_years == 0.0:
        price = intrinsic_value(contract.option_type, contract.spot, contract.strike)
        runtime_ms = (perf_counter() - started) * 1_000.0
        result = _american_result(
            contract=contract,
            market=market,
            config=config,
            price=price,
            european_tree_price=price,
            runtime_ms=runtime_ms,
            boundary_points=(),
            exercise_node_count=0,
            non_contiguous_level_count=0,
            boundary_case=DeterministicBoundaryCase.ZERO_MATURITY,
        )
        return _AmericanComputation(result=result, boundary_points=())
    if market.volatility == 0.0:
        return _deterministic_american_computation(contract, market, config, started)

    parameters = crr_parameters(contract, market, config)
    terminal_prices = crr_terminal_stock_prices(contract, parameters)
    terminal_payoffs = vanilla_payoff(terminal_prices, contract)
    state = _american_backward_induction(
        terminal_payoffs,
        contract,
        parameters,
        config.exercise_tolerance,
    )
    greek_state = _BackwardInductionState(
        root=state.root,
        level_one=state.level_one,
        level_two=state.level_two,
    )
    delta, gamma, theta = _tree_greeks(contract, parameters, greek_state)
    runtime_ms = (perf_counter() - started) * 1_000.0
    result = _american_result(
        contract=contract,
        market=market,
        config=config,
        price=state.root,
        european_tree_price=state.european_root,
        runtime_ms=runtime_ms,
        boundary_points=state.boundary_points,
        exercise_node_count=state.exercise_node_count,
        non_contiguous_level_count=state.non_contiguous_level_count,
        parameters=parameters,
        delta=delta,
        gamma=gamma,
        theta=theta,
    )
    logger.info(
        "american_binomial_pricing_complete model=%s steps=%d price=%g exercise_nodes=%d "
        "runtime_ms=%.3f",
        config.model.value,
        config.steps,
        result.price,
        result.exercise_node_count,
        runtime_ms,
    )
    return _AmericanComputation(result=result, boundary_points=state.boundary_points)


def binomial_price(
    contract: OptionContract,
    market: MarketEnvironment,
    config: BinomialTreeConfig | None = None,
) -> BinomialPricingResult | AmericanBinomialPricingResult:
    """Price one European or American vanilla option under the CRR exercise policy."""

    tree_config = config or BinomialTreeConfig()
    if contract.exercise_style is ExerciseStyle.AMERICAN:
        return _american_computation(contract, market, tree_config).result
    start = perf_counter()
    lower_bound, upper_bound = _price_bounds(contract, market)
    if contract.maturity_years == 0.0:
        price = intrinsic_value(contract.option_type, contract.spot, contract.strike)
        runtime_ms = (perf_counter() - start) * 1_000.0
        return _result(
            price=price,
            contract=contract,
            market=market,
            config=tree_config,
            runtime_ms=runtime_ms,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            boundary=DeterministicBoundaryCase.ZERO_MATURITY,
        )
    if market.volatility == 0.0:
        terminal_spot = contract.spot * finite_exp(
            (market.risk_free_rate - market.dividend_yield) * contract.maturity_years,
            context="deterministic binomial terminal stock",
        )
        payoff = float(vanilla_payoff([terminal_spot], contract)[0])
        price = (
            finite_exp(
                -market.risk_free_rate * contract.maturity_years,
                context="deterministic binomial discount",
            )
            * payoff
        )
        if not math.isfinite(price):
            raise NumericalError("deterministic binomial price is non-finite")
        _validate_price_bounds(price, lower_bound, upper_bound, tree_config.arbitrage_tolerance)
        runtime_ms = (perf_counter() - start) * 1_000.0
        return _result(
            price=price,
            contract=contract,
            market=market,
            config=tree_config,
            runtime_ms=runtime_ms,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            boundary=DeterministicBoundaryCase.ZERO_VOLATILITY,
        )
    parameters = crr_parameters(contract, market, tree_config)
    terminal_prices = crr_terminal_stock_prices(contract, parameters)
    terminal_payoffs = vanilla_payoff(terminal_prices, contract)
    state = _backward_induction(terminal_payoffs, parameters)
    _validate_price_bounds(
        state.root,
        lower_bound,
        upper_bound,
        tree_config.arbitrage_tolerance,
    )
    delta, gamma, theta = _tree_greeks(contract, parameters, state)
    runtime_ms = (perf_counter() - start) * 1_000.0
    logger.info(
        "binomial_pricing_complete model=%s steps=%d price=%g runtime_ms=%.3f",
        tree_config.model.value,
        tree_config.steps,
        state.root,
        runtime_ms,
    )
    return _result(
        price=state.root,
        contract=contract,
        market=market,
        config=tree_config,
        runtime_ms=runtime_ms,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        parameters=parameters,
        delta=delta,
        gamma=gamma,
        theta=theta,
    )


def american_binomial_price(
    contract: OptionContract,
    market: MarketEnvironment,
    config: BinomialTreeConfig | None = None,
) -> AmericanBinomialPricingResult:
    """Return the narrowed American result type."""

    return _american_computation(contract, market, config or BinomialTreeConfig()).result


def american_exercise_boundary(
    contract: OptionContract,
    market: MarketEnvironment,
    config: BinomialTreeConfig | None = None,
) -> ExerciseBoundaryResult:
    """Return the ordered early-exercise frontier for an American contract."""

    tree_config = config or BinomialTreeConfig()
    computation = _american_computation(contract, market, tree_config)
    result = computation.result
    return ExerciseBoundaryResult(
        exercise_style=result.exercise_style,
        option_type=contract.option_type,
        steps=result.steps,
        model=result.model,
        points=list(computation.boundary_points),
        early_exercise_occurred=result.early_exercise_occurred,
        earliest_exercise_step=result.earliest_exercise_step,
        earliest_exercise_time=result.earliest_exercise_time,
        earliest_boundary_price=result.earliest_boundary_price,
        exercise_node_count=result.exercise_node_count,
        non_contiguous_level_count=result.non_contiguous_level_count,
        runtime_ms=result.runtime_ms,
    )


class BinomialPricer(
    Pricer[BinomialTreeConfig, BinomialPricingResult | AmericanBinomialPricingResult]
):
    """Typed European and American CRR pricing engine."""

    def price(
        self,
        contract: OptionContract,
        market: MarketEnvironment,
        config: BinomialTreeConfig,
    ) -> BinomialPricingResult | AmericanBinomialPricingResult:
        return binomial_price(contract, market, config)
