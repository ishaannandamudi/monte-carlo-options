"""Memory-efficient European Cox-Ross-Rubinstein binomial pricing."""

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
    UnsupportedExerciseStyleError,
)
from backend.app.core.numerics import finite_array_exp, finite_exp, require_finite
from backend.app.models.binomial import (
    BinomialPricingResult,
    BinomialTreeConfig,
    CRRParameters,
    DeterministicBoundaryCase,
    TreeModel,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import ExerciseStyle, OptionContract
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
    """Return one European continuation step, isolated for Phase 8 extension."""

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


def binomial_price(
    contract: OptionContract,
    market: MarketEnvironment,
    config: BinomialTreeConfig | None = None,
) -> BinomialPricingResult:
    """Price one European vanilla option with a CRR tree or exact boundary policy."""

    tree_config = config or BinomialTreeConfig()
    if contract.exercise_style is not ExerciseStyle.EUROPEAN:
        raise UnsupportedExerciseStyleError("Phase 7 binomial pricing supports EUROPEAN exercise")
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


class BinomialPricer(Pricer[BinomialTreeConfig, BinomialPricingResult]):
    """Typed European CRR pricing engine."""

    def price(
        self,
        contract: OptionContract,
        market: MarketEnvironment,
        config: BinomialTreeConfig,
    ) -> BinomialPricingResult:
        return binomial_price(contract, market, config)
