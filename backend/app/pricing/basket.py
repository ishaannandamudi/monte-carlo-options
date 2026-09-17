"""Analytical geometric and Monte Carlo European basket pricing."""

import logging
import math
from time import perf_counter

import numpy as np
from scipy.stats import norm

from backend.app.analytics.statistics import (
    antithetic_standard_error,
    confidence_interval,
    sample_mean,
    sample_standard_deviation,
    standard_error,
)
from backend.app.core.exceptions import MultiAssetPricingError, NumericalError
from backend.app.core.numerics import finite_exp, require_finite
from backend.app.models.basket import (
    BasketOptionContract,
    BasketPricingResult,
    BasketSimulationConfig,
    BasketType,
    BasketVarianceReduction,
    GeometricBasketAnalyticalResult,
)
from backend.app.models.multi_asset import (
    CorrelationFactorizationDiagnostics,
    MultiAssetMarketEnvironment,
)
from backend.app.models.option import OptionType
from backend.app.payoffs.basket import basket_payoff_from_values, basket_values
from backend.app.pricing.variance_reduction import apply_control_variate, cluster_observations
from backend.app.simulation.correlation import CorrelationFactorization, factorize_correlation
from backend.app.simulation.multi_asset_gbm import simulate_correlated_terminal
from backend.app.simulation.random import create_rng

logger = logging.getLogger(__name__)


def _validate_dimensions(
    contract: BasketOptionContract, market: MultiAssetMarketEnvironment
) -> None:
    if len(contract.weights) != market.num_assets:
        raise MultiAssetPricingError("basket weight count must equal market asset count")


def _factorization_diagnostics(
    market: MultiAssetMarketEnvironment,
    factorization: CorrelationFactorization,
) -> CorrelationFactorizationDiagnostics:
    return CorrelationFactorizationDiagnostics(
        dimension=market.num_assets,
        method=factorization.method,
        min_eigenvalue=factorization.min_eigenvalue,
        max_eigenvalue=factorization.max_eigenvalue,
        effective_rank=factorization.effective_rank,
        condition_number=factorization.condition_number,
        repaired=market.correlation_repaired,
        adjustment_magnitude=market.correlation_adjustment_magnitude,
    )


def geometric_basket_log_moments(
    contract: BasketOptionContract,
    market: MultiAssetMarketEnvironment,
) -> tuple[float, float]:
    """Return exact mean and variance of the terminal geometric basket's log."""

    _validate_dimensions(contract, market)
    weights = np.asarray(contract.weights, dtype=np.float64)
    spots = np.asarray(market.spots, dtype=np.float64)
    volatilities = np.asarray(market.volatilities, dtype=np.float64)
    dividends = np.asarray(market.dividend_yields, dtype=np.float64)
    correlation = np.asarray(market.correlation_matrix, dtype=np.float64)
    maturity = contract.maturity_years
    with np.errstate(over="ignore", invalid="ignore"):
        mean_log = float(
            weights
            @ (
                np.log(spots)
                + (market.risk_free_rate - dividends - 0.5 * volatilities**2) * maturity
            )
        )
        weighted_volatility = weights * volatilities
        variance_log = float(maturity * (weighted_volatility @ correlation @ weighted_volatility))
    if not math.isfinite(mean_log) or not math.isfinite(variance_log):
        raise NumericalError("geometric basket log moments are non-finite")
    if variance_log < -1e-12:
        raise NumericalError("geometric basket variance is materially negative")
    return mean_log, max(variance_log, 0.0)


def geometric_basket_analytical_price(
    contract: BasketOptionContract,
    market: MultiAssetMarketEnvironment,
) -> GeometricBasketAnalyticalResult:
    """Price the geometric basket from its exact terminal lognormal distribution."""

    if contract.basket_type is not BasketType.GEOMETRIC:
        raise MultiAssetPricingError("geometric analytical pricing requires basket_type=GEOMETRIC")
    mean_log, variance_log = geometric_basket_log_moments(contract, market)
    expected = finite_exp(
        mean_log + 0.5 * variance_log,
        context="expected geometric basket value",
    )
    discount = finite_exp(
        -market.risk_free_rate * contract.maturity_years,
        context="geometric basket discount factor",
    )
    deterministic = variance_log == 0.0
    if deterministic:
        terminal = finite_exp(mean_log, context="deterministic geometric basket value")
        payoff = (
            max(terminal - contract.strike, 0.0)
            if contract.option_type is OptionType.CALL
            else max(contract.strike - terminal, 0.0)
        )
        price = discount * payoff
    else:
        deviation = math.sqrt(variance_log)
        d2 = (mean_log - math.log(contract.strike)) / deviation
        d1 = d2 + deviation
        if contract.option_type is OptionType.CALL:
            expectation = expected * float(norm.cdf(d1)) - contract.strike * float(norm.cdf(d2))
        else:
            expectation = contract.strike * float(norm.cdf(-d2)) - expected * float(norm.cdf(-d1))
        price = discount * max(expectation, 0.0)
    if not math.isfinite(price) or price < 0.0:
        raise NumericalError("geometric basket analytical price is invalid")
    return GeometricBasketAnalyticalResult(
        price=price,
        mean_log_basket=mean_log,
        variance_log_basket=variance_log,
        expected_geometric_basket=expected,
        num_assets=market.num_assets,
        weights=contract.weights,
        deterministic=deterministic,
    )


class BasketMonteCarloPricer:
    """Price European arithmetic or geometric baskets under correlated GBM."""

    def price(
        self,
        contract: BasketOptionContract,
        market: MultiAssetMarketEnvironment,
        config: BasketSimulationConfig,
    ) -> BasketPricingResult:
        _validate_dimensions(contract, market)
        if (
            contract.basket_type is BasketType.GEOMETRIC
            and config.variance_reduction.uses_geometric_control
        ):
            raise MultiAssetPricingError(
                "geometric control variates apply only to ARITHMETIC basket options"
            )
        started = perf_counter()
        logger.debug("basket_config=%s", config.model_dump(mode="json"))
        factorization = factorize_correlation(market.correlation_matrix)
        active_stochastic = any(
            weight > 0.0 and volatility > 0.0
            for weight, volatility in zip(contract.weights, market.volatilities, strict=True)
        )
        deterministic = contract.maturity_years == 0.0 or not active_stochastic
        count = 1 if deterministic else config.num_simulations
        simulation_started = perf_counter()
        terminal = simulate_correlated_terminal(
            market,
            contract.maturity_years,
            count,
            create_rng(config.seed, config.random_engine),
            antithetic=config.variance_reduction.uses_antithetic,
            factorization=factorization,
        )
        simulation_ms = (perf_counter() - simulation_started) * 1_000.0
        if terminal.shape != (count, market.num_assets) or np.any(terminal <= 0.0):
            raise NumericalError("multi-asset simulator returned invalid terminal prices")

        values = basket_values(terminal, contract.weights, contract.basket_type)
        payoffs = basket_payoff_from_values(values, contract)
        discount = finite_exp(
            -market.risk_free_rate * contract.maturity_years,
            context="basket payoff discount factor",
        )
        discounted = np.asarray(discount * payoffs, dtype=np.float64)
        require_finite(discounted, context="discounted basket payoffs")
        raw_estimate = sample_mean(discounted)

        raw_price = None
        adjusted_price = None
        raw_se = None
        adjusted_se = None
        raw_variance = None
        adjusted_variance = None
        beta = None
        control_mean = None
        control_expected = None
        variance_ratio = None

        if deterministic:
            estimate = raw_estimate
            estimator_se = 0.0
            if config.variance_reduction.uses_geometric_control:
                geometric_contract = contract.model_copy(
                    update={"basket_type": BasketType.GEOMETRIC}
                )
                control_expected = geometric_basket_analytical_price(
                    geometric_contract, market
                ).price
                geometric_values = basket_values(terminal, contract.weights, BasketType.GEOMETRIC)
                control = discount * basket_payoff_from_values(geometric_values, geometric_contract)
                raw_price = adjusted_price = raw_estimate
                raw_se = adjusted_se = 0.0
                raw_variance = adjusted_variance = 0.0
                beta = 0.0
                control_mean = sample_mean(control)
        elif config.variance_reduction.uses_geometric_control:
            geometric_contract = contract.model_copy(update={"basket_type": BasketType.GEOMETRIC})
            geometric_values = basket_values(terminal, contract.weights, BasketType.GEOMETRIC)
            control = np.asarray(
                discount * basket_payoff_from_values(geometric_values, geometric_contract),
                dtype=np.float64,
            )
            control_expected = geometric_basket_analytical_price(geometric_contract, market).price
            adjustment = apply_control_variate(
                cluster_observations(
                    discounted, antithetic=config.variance_reduction.uses_antithetic
                ),
                cluster_observations(control, antithetic=config.variance_reduction.uses_antithetic),
                control_expected,
            )
            estimate = adjustment.adjusted_price
            estimator_se = adjustment.adjusted_standard_error
            raw_price = adjustment.raw_price
            adjusted_price = adjustment.adjusted_price
            raw_se = adjustment.raw_standard_error
            adjusted_se = adjustment.adjusted_standard_error
            raw_variance = adjustment.raw_estimator_variance
            adjusted_variance = adjustment.adjusted_estimator_variance
            beta = adjustment.beta
            control_mean = adjustment.control_mean
            variance_ratio = adjustment.variance_reduction_ratio
        else:
            estimate = raw_estimate
            estimator_se = (
                antithetic_standard_error(discounted)
                if config.variance_reduction is BasketVarianceReduction.ANTITHETIC
                else standard_error(discounted)
            )

        if not math.isfinite(estimate):
            raise NumericalError("basket Monte Carlo estimate is non-finite")
        low, high = confidence_interval(estimate, estimator_se, config.confidence_level)
        basket_std = 0.0 if deterministic else sample_standard_deviation(values)
        basket_variance = basket_std * basket_std
        if contract.option_type is OptionType.CALL:
            itm = values > contract.strike
        else:
            itm = values < contract.strike
        result = BasketPricingResult(
            price=estimate,
            standard_error=estimator_se,
            estimator_variance=estimator_se * estimator_se,
            confidence_interval_low=low,
            confidence_interval_high=high,
            confidence_level=config.confidence_level,
            num_simulations=config.num_simulations,
            seed=config.seed,
            num_assets=market.num_assets,
            basket_type=contract.basket_type,
            weights=contract.weights,
            variance_reduction=config.variance_reduction,
            runtime_ms=(perf_counter() - started) * 1_000.0,
            simulation_runtime_ms=simulation_ms,
            mean_basket_value=sample_mean(values),
            basket_std=math.sqrt(basket_variance),
            probability_itm=float(np.mean(itm)),
            deterministic=deterministic,
            factorization=_factorization_diagnostics(market, factorization),
            raw_price=raw_price,
            adjusted_price=adjusted_price,
            raw_standard_error=raw_se,
            adjusted_standard_error=adjusted_se,
            raw_estimator_variance=raw_variance,
            adjusted_estimator_variance=adjusted_variance,
            control_variate_beta=beta,
            control_sample_mean=control_mean,
            analytical_geometric_price=control_expected,
            variance_reduction_ratio=variance_ratio,
        )
        logger.info(
            "basket_complete assets=%d type=%s method=%s simulations=%d runtime_ms=%.3f",
            market.num_assets,
            contract.basket_type,
            config.variance_reduction,
            config.num_simulations,
            result.runtime_ms,
        )
        return result


def basket_monte_carlo_price(
    contract: BasketOptionContract,
    market: MultiAssetMarketEnvironment,
    config: BasketSimulationConfig | None = None,
) -> BasketPricingResult:
    """Convenience entry point for correlated-GBM basket Monte Carlo."""

    return BasketMonteCarloPricer().price(contract, market, config or BasketSimulationConfig())
