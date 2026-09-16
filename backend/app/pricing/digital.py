"""Analytical and terminal-GBM Monte Carlo pricing for European digital options."""

import logging
import math
from time import perf_counter

import numpy as np
from scipy.stats import norm

from backend.app.analytics.statistics import (
    antithetic_standard_error,
    confidence_interval,
    sample_mean,
    standard_error,
)
from backend.app.core.exceptions import NumericalError
from backend.app.core.numerics import finite_exp, require_finite
from backend.app.models.digital import (
    DigitalAnalyticalResult,
    DigitalMonteCarloResult,
    DigitalOptionContract,
    DigitalSimulationConfig,
    DigitalType,
    DigitalVarianceReduction,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.models.simulation import SimulationConfig, VarianceReduction
from backend.app.payoffs.digital import digital_itm, digital_payoff
from backend.app.pricing.black_scholes import black_scholes_d1_d2
from backend.app.simulation.base import StochasticProcess
from backend.app.simulation.gbm import GBMProcess
from backend.app.simulation.random import create_rng

logger = logging.getLogger(__name__)


def _vanilla_contract(contract: DigitalOptionContract) -> OptionContract:
    return OptionContract(
        option_type=contract.option_type,
        exercise_style=contract.exercise_style,
        spot=contract.spot,
        strike=contract.strike,
        maturity_years=contract.maturity_years,
    )


def _deterministic_terminal(contract: DigitalOptionContract, market: MarketEnvironment) -> float:
    growth = finite_exp(
        (market.risk_free_rate - market.dividend_yield) * contract.maturity_years,
        context="deterministic digital growth factor",
    )
    terminal = contract.spot * growth
    if not math.isfinite(terminal) or terminal <= 0.0:
        raise NumericalError("deterministic digital terminal price is invalid")
    return terminal


def digital_analytical_price(
    contract: DigitalOptionContract,
    market: MarketEnvironment,
) -> DigitalAnalyticalResult:
    """Return cash/asset digital value with strict equality at deterministic boundaries."""

    started = perf_counter()
    maturity = contract.maturity_years
    if maturity == 0.0:
        itm = (
            contract.spot > contract.strike
            if contract.option_type is OptionType.CALL
            else (contract.spot < contract.strike)
        )
        probability = float(itm)
        price = (
            (contract.cash_payout or 0.0) * probability
            if contract.digital_type is DigitalType.CASH_OR_NOTHING
            else contract.spot * probability
        )
        deterministic = True
    else:
        terms = black_scholes_d1_d2(_vanilla_contract(contract), market)
        if terms is None:
            terminal = _deterministic_terminal(contract, market)
            itm = (
                terminal > contract.strike
                if contract.option_type is OptionType.CALL
                else (terminal < contract.strike)
            )
            probability = float(itm)
            payoff = (
                (contract.cash_payout or 0.0) * probability
                if contract.digital_type is DigitalType.CASH_OR_NOTHING
                else terminal * probability
            )
            price = (
                finite_exp(
                    -market.risk_free_rate * maturity,
                    context="deterministic digital discount factor",
                )
                * payoff
            )
            deterministic = True
        else:
            d1, d2 = terms
            probability = float(
                norm.cdf(d2) if contract.option_type is OptionType.CALL else norm.cdf(-d2)
            )
            if contract.digital_type is DigitalType.CASH_OR_NOTHING:
                assert contract.cash_payout is not None
                price = (
                    contract.cash_payout
                    * finite_exp(
                        -market.risk_free_rate * maturity,
                        context="cash digital discount factor",
                    )
                    * probability
                )
            else:
                asset_probability = float(
                    norm.cdf(d1) if contract.option_type is OptionType.CALL else norm.cdf(-d1)
                )
                price = (
                    contract.spot
                    * finite_exp(
                        -market.dividend_yield * maturity,
                        context="asset digital dividend discount factor",
                    )
                    * asset_probability
                )
            deterministic = False
    if not math.isfinite(price) or price < 0.0:
        raise NumericalError("digital analytical price is invalid")
    return DigitalAnalyticalResult(
        price=price,
        digital_type=contract.digital_type,
        option_type=contract.option_type,
        cash_payout=contract.cash_payout,
        probability_itm=probability,
        deterministic=deterministic,
        runtime_ms=(perf_counter() - started) * 1_000.0,
    )


class DigitalMonteCarloPricer:
    """Price strict-ITM digital payoffs over the existing terminal GBM process."""

    def __init__(self, process: StochasticProcess | None = None) -> None:
        self._process = process or GBMProcess()

    def price(
        self,
        contract: DigitalOptionContract,
        market: MarketEnvironment,
        config: DigitalSimulationConfig,
    ) -> DigitalMonteCarloResult:
        started = perf_counter()
        variance_reduction = (
            VarianceReduction.ANTITHETIC
            if config.variance_reduction is DigitalVarianceReduction.ANTITHETIC
            else VarianceReduction.NONE
        )
        simulation = SimulationConfig(
            num_simulations=config.num_simulations,
            seed=config.seed,
            variance_reduction=variance_reduction,
            random_engine=config.random_engine,
            confidence_level=config.confidence_level,
        )
        terminal = self._process.simulate_terminal(
            _vanilla_contract(contract),
            market,
            simulation,
            create_rng(config.seed, config.random_engine),
        )
        if terminal.shape != (config.num_simulations,):
            raise NumericalError("digital process returned an unexpected terminal-price shape")
        require_finite(terminal, context="digital terminal prices")
        if np.any(terminal <= 0.0):
            raise NumericalError("digital terminal prices must be strictly positive")
        discount = finite_exp(
            -market.risk_free_rate * contract.maturity_years,
            context="digital payoff discount factor",
        )
        discounted = np.asarray(discount * digital_payoff(terminal, contract), dtype=np.float64)
        require_finite(discounted, context="discounted digital payoffs")
        price = sample_mean(discounted)
        deterministic = contract.maturity_years == 0.0 or market.volatility == 0.0
        if deterministic or np.all(discounted == discounted[0]):
            estimator_se = 0.0
        elif config.variance_reduction is DigitalVarianceReduction.ANTITHETIC:
            estimator_se = antithetic_standard_error(discounted)
        else:
            estimator_se = standard_error(discounted)
        low, high = confidence_interval(price, estimator_se, config.confidence_level)
        analytical = digital_analytical_price(contract, market)
        result = DigitalMonteCarloResult(
            price=price,
            standard_error=estimator_se,
            estimator_variance=estimator_se * estimator_se,
            confidence_interval_low=low,
            confidence_interval_high=high,
            confidence_level=config.confidence_level,
            digital_type=contract.digital_type,
            option_type=contract.option_type,
            cash_payout=contract.cash_payout,
            probability_itm=float(np.mean(digital_itm(terminal, contract))),
            num_simulations=config.num_simulations,
            seed=config.seed,
            variance_reduction=config.variance_reduction,
            runtime_ms=(perf_counter() - started) * 1_000.0,
            deterministic=deterministic,
            analytical_price=analytical.price,
            absolute_error=abs(price - analytical.price),
        )
        logger.info(
            "digital_complete type=%s method=%s price=%g se=%g simulations=%d runtime_ms=%.3f",
            contract.digital_type,
            config.variance_reduction,
            result.price,
            result.standard_error,
            config.num_simulations,
            result.runtime_ms,
        )
        return result


def digital_monte_carlo_price(
    contract: DigitalOptionContract,
    market: MarketEnvironment,
    config: DigitalSimulationConfig | None = None,
) -> DigitalMonteCarloResult:
    return DigitalMonteCarloPricer().price(contract, market, config or DigitalSimulationConfig())
