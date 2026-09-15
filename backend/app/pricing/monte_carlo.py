"""European vanilla Monte Carlo pricing over a terminal stochastic process."""

import logging
import math
from time import perf_counter

import numpy as np

from backend.app.analytics.statistics import (
    antithetic_standard_error,
    confidence_interval,
    sample_mean,
    sample_standard_deviation,
    sample_variance,
    standard_error,
)
from backend.app.core.exceptions import NumericalError
from backend.app.core.numerics import finite_exp, require_finite
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.models.results import MonteCarloResult
from backend.app.models.simulation import SimulationConfig, VarianceReduction
from backend.app.payoffs.vanilla import vanilla_payoff
from backend.app.pricing.base import Pricer
from backend.app.pricing.variance_reduction import (
    apply_control_variate,
    cluster_observations,
    discounted_stock_control,
)
from backend.app.simulation.base import StochasticProcess
from backend.app.simulation.gbm import GBMProcess
from backend.app.simulation.random import create_rng

logger = logging.getLogger(__name__)


class MonteCarloPricer(Pricer[SimulationConfig, MonteCarloResult]):
    """Price discounted European terminal payoffs and quantify sampling error."""

    def __init__(self, process: StochasticProcess | None = None) -> None:
        self._process = process or GBMProcess()

    def price(
        self,
        contract: OptionContract,
        market: MarketEnvironment,
        config: SimulationConfig,
    ) -> MonteCarloResult:
        """Return a reproducible estimate; ``batch_size`` is an inert execution hint.

        The current engine generates one vector so every configured batch size preserves the
        identical PCG64 stream and numerical result. API limits bound allocation.
        """

        started = perf_counter()
        logger.debug("monte_carlo_config=%s", config.model_dump(mode="json"))
        rng = create_rng(config.seed, config.random_engine)
        terminal_prices = self._process.simulate_terminal(contract, market, config, rng)
        if terminal_prices.shape != (config.num_simulations,):
            raise NumericalError(
                "stochastic process returned an unexpected terminal-price shape: "
                f"{terminal_prices.shape}"
            )
        require_finite(terminal_prices, context="terminal prices")
        if np.any(terminal_prices <= 0.0):
            raise NumericalError("terminal prices must be strictly positive")

        payoffs = vanilla_payoff(terminal_prices, contract)
        discount = finite_exp(
            -market.risk_free_rate * contract.maturity_years,
            context="payoff discount factor",
        )
        discounted_payoffs = np.asarray(discount * payoffs, dtype=np.float64)
        require_finite(discounted_payoffs, context="discounted payoffs")

        raw_estimate = sample_mean(discounted_payoffs)
        deterministic = contract.maturity_years == 0.0 or market.volatility == 0.0
        raw_price: float | None = None
        raw_standard_error: float | None = None
        adjusted_standard_error: float | None = None
        control_beta: float | None = None
        control_mean: float | None = None
        control_expected_mean: float | None = None
        raw_estimator_variance: float | None = None
        adjusted_estimator_variance: float | None = None
        variance_reduction_ratio: float | None = None

        if deterministic:
            payoff_variance = 0.0
            payoff_std = 0.0
            estimator_se = 0.0
            terminal_std = 0.0
            estimate = raw_estimate
            if config.variance_reduction.uses_control_variate:
                control, control_expected_mean = discounted_stock_control(
                    terminal_prices, contract, market
                )
                raw_price = raw_estimate
                raw_standard_error = 0.0
                adjusted_standard_error = 0.0
                control_beta = 0.0
                control_mean = sample_mean(control)
                raw_estimator_variance = 0.0
                adjusted_estimator_variance = 0.0
        else:
            payoff_variance = sample_variance(discounted_payoffs)
            payoff_std = math.sqrt(payoff_variance)
            if config.variance_reduction.uses_control_variate:
                control, control_expected_mean = discounted_stock_control(
                    terminal_prices, contract, market
                )
                raw_clusters = cluster_observations(
                    discounted_payoffs,
                    antithetic=config.variance_reduction.uses_antithetic,
                )
                control_clusters = cluster_observations(
                    control,
                    antithetic=config.variance_reduction.uses_antithetic,
                )
                adjustment = apply_control_variate(
                    raw_clusters,
                    control_clusters,
                    control_expected_mean,
                )
                estimate = adjustment.adjusted_price
                estimator_se = adjustment.adjusted_standard_error
                raw_price = adjustment.raw_price
                raw_standard_error = adjustment.raw_standard_error
                adjusted_standard_error = adjustment.adjusted_standard_error
                control_beta = adjustment.beta
                control_mean = adjustment.control_mean
                raw_estimator_variance = adjustment.raw_estimator_variance
                adjusted_estimator_variance = adjustment.adjusted_estimator_variance
                variance_reduction_ratio = adjustment.variance_reduction_ratio
            elif config.variance_reduction is VarianceReduction.ANTITHETIC:
                estimate = raw_estimate
                estimator_se = antithetic_standard_error(discounted_payoffs)
            else:
                estimate = raw_estimate
                estimator_se = standard_error(discounted_payoffs)
            terminal_std = sample_standard_deviation(terminal_prices)
        ci_low, ci_high = confidence_interval(estimate, estimator_se, config.confidence_level)

        if contract.option_type is OptionType.CALL:
            itm = terminal_prices > contract.strike
        else:
            itm = terminal_prices < contract.strike

        runtime_ms = (perf_counter() - started) * 1_000.0
        result = MonteCarloResult(
            price=estimate,
            standard_error=estimator_se,
            payoff_variance=payoff_variance,
            discounted_payoff_std=payoff_std,
            confidence_interval_low=ci_low,
            confidence_interval_high=ci_high,
            confidence_level=config.confidence_level,
            num_simulations=config.num_simulations,
            seed=config.seed,
            variance_reduction=config.variance_reduction,
            runtime_ms=runtime_ms,
            probability_itm=float(np.mean(itm)),
            mean_terminal_price=sample_mean(terminal_prices),
            terminal_price_std=terminal_std,
            raw_price=raw_price,
            raw_standard_error=raw_standard_error,
            adjusted_standard_error=adjusted_standard_error,
            control_variate_beta=control_beta,
            control_variate_mean=control_mean,
            control_variate_expected_mean=control_expected_mean,
            raw_estimator_variance=raw_estimator_variance,
            adjusted_estimator_variance=adjusted_estimator_variance,
            variance_reduction_ratio=variance_reduction_ratio,
        )
        logger.info(
            "monte_carlo_complete simulations=%d variance_reduction=%s runtime_ms=%.3f",
            config.num_simulations,
            config.variance_reduction.value,
            runtime_ms,
        )
        return result
