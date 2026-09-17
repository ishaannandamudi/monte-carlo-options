"""European Heston Monte Carlo pricing."""

import logging
from time import perf_counter

import numpy as np

from backend.app.analytics.statistics import (
    antithetic_standard_error,
    confidence_interval,
    sample_mean,
    sample_standard_deviation,
    standard_error,
)
from backend.app.core.exceptions import NumericalError
from backend.app.core.numerics import finite_exp, require_finite
from backend.app.models.heston import (
    HestonMonteCarloResult,
    HestonParameters,
    HestonSimulationConfig,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.payoffs.vanilla import vanilla_payoff
from backend.app.pricing.base import require_european_exercise
from backend.app.simulation.heston import simulate_heston_terminal
from backend.app.simulation.random import create_rng

logger = logging.getLogger(__name__)


class HestonMonteCarloPricer:
    """Price European terminal payoffs under a configured Heston discretization."""

    def price(
        self,
        contract: OptionContract,
        market: MarketEnvironment,
        parameters: HestonParameters,
        config: HestonSimulationConfig,
    ) -> HestonMonteCarloResult:
        require_european_exercise(contract, engine="Heston Monte Carlo pricing")
        started = perf_counter()
        simulation_started = perf_counter()
        simulation = simulate_heston_terminal(
            contract,
            market,
            parameters,
            config,
            create_rng(config.seed, config.random_engine),
        )
        simulation_ms = (perf_counter() - simulation_started) * 1_000.0
        terminal_spots = simulation.terminal_spots
        terminal_variances = simulation.terminal_variances
        if terminal_spots.shape != (config.num_simulations,) or terminal_variances.shape != (
            config.num_simulations,
        ):
            raise NumericalError("Heston simulator returned an unexpected terminal shape")
        require_finite(terminal_spots, context="Heston terminal spots")
        require_finite(terminal_variances, context="Heston terminal variances")
        if np.any(terminal_spots <= 0.0) or np.any(terminal_variances < 0.0):
            raise NumericalError("Heston terminal state is outside its valid domain")

        discount = finite_exp(
            -market.risk_free_rate * contract.maturity_years,
            context="Heston payoff discount factor",
        )
        discounted_payoffs = np.asarray(
            discount * vanilla_payoff(terminal_spots, contract), dtype=np.float64
        )
        require_finite(discounted_payoffs, context="discounted Heston payoffs")
        estimate = sample_mean(discounted_payoffs)
        deterministic = contract.maturity_years == 0.0 or (
            parameters.v0 == 0.0 and parameters.theta == 0.0
        )
        if deterministic:
            estimator_se = 0.0
            spot_std = 0.0
            variance_std = 0.0
        else:
            estimator_se = (
                antithetic_standard_error(discounted_payoffs)
                if config.antithetic
                else standard_error(discounted_payoffs)
            )
            spot_std = sample_standard_deviation(terminal_spots)
            variance_std = sample_standard_deviation(terminal_variances)
        low, high = confidence_interval(estimate, estimator_se, config.confidence_level)
        itm = (
            terminal_spots > contract.strike
            if contract.option_type is OptionType.CALL
            else terminal_spots < contract.strike
        )
        result = HestonMonteCarloResult(
            price=estimate,
            standard_error=estimator_se,
            estimator_variance=estimator_se * estimator_se,
            confidence_interval_low=low,
            confidence_interval_high=high,
            confidence_level=config.confidence_level,
            num_simulations=config.num_simulations,
            num_time_steps=config.num_time_steps,
            seed=config.seed,
            scheme=config.scheme,
            antithetic=config.antithetic,
            runtime_ms=(perf_counter() - started) * 1_000.0,
            simulation_runtime_ms=simulation_ms,
            mean_terminal_spot=sample_mean(terminal_spots),
            std_terminal_spot=spot_std,
            mean_terminal_variance=sample_mean(terminal_variances),
            std_terminal_variance=variance_std,
            min_terminal_variance=float(np.min(terminal_variances)),
            max_terminal_variance=float(np.max(terminal_variances)),
            negative_variance_proposal_fraction=(simulation.negative_variance_proposal_fraction),
            probability_itm=float(np.mean(itm)),
            feller_satisfied=parameters.feller_satisfied,
            feller_ratio=parameters.feller_ratio,
        )
        logger.info(
            "heston_mc_complete scheme=%s simulations=%d steps=%d price=%g se=%g runtime_ms=%.3f",
            config.scheme,
            config.num_simulations,
            config.num_time_steps,
            result.price,
            result.standard_error,
            result.runtime_ms,
        )
        return result


def heston_monte_carlo_price(
    contract: OptionContract,
    market: MarketEnvironment,
    parameters: HestonParameters,
    config: HestonSimulationConfig | None = None,
) -> HestonMonteCarloResult:
    return HestonMonteCarloPricer().price(
        contract, market, parameters, config or HestonSimulationConfig()
    )
