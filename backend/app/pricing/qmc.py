"""Reusable QMC estimators for European, Asian, and basket option pricing."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from backend.app.core.exceptions import QMCError
from backend.app.core.numerics import finite_exp
from backend.app.models.asian import AveragingType
from backend.app.models.basket import BasketType
from backend.app.models.option import OptionType
from backend.app.models.qmc import (
    QMCAsianRequest,
    QMCBasketRequest,
    QMCConfig,
    QMCEuropeanRequest,
    QMCPathConstruction,
    QMCPricingResult,
    QMCReplicateDiagnostic,
    SamplingMethod,
)
from backend.app.payoffs.asian import asian_average, asian_payoff_from_average
from backend.app.payoffs.basket import basket_payoff_from_values, basket_values
from backend.app.payoffs.vanilla import vanilla_payoff
from backend.app.pricing.asian import geometric_asian_analytical_price
from backend.app.pricing.basket import geometric_basket_analytical_price
from backend.app.pricing.black_scholes import black_scholes_price
from backend.app.simulation.gbm import observation_prices_from_normals, terminal_prices_from_normals
from backend.app.simulation.multi_asset_gbm import (
    terminal_prices_from_normals as multi_asset_terminal_from_normals,
)
from backend.app.simulation.qmc import (
    generate_normal_samples,
    normalized_increment_drivers,
    numerical_method_metadata,
    qmc_replicate_seeds,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class _EstimatorObservations:
    values: FloatArray
    underlying: FloatArray
    itm: FloatArray


Estimator = Callable[[FloatArray], _EstimatorObservations]


def _control_adjusted(
    target: FloatArray, control: FloatArray, expected_control: float
) -> FloatArray:
    if target.shape != control.shape:
        raise QMCError("control and target observations must have the same shape")
    control_variance = float(np.var(control, ddof=1)) if control.size > 1 else 0.0
    if control_variance <= 0.0:
        return target.copy()
    covariance = float(np.cov(target, control, ddof=1)[0, 1])
    beta = covariance / control_variance
    return np.asarray(target - beta * (control - expected_control), dtype=np.float64)


def _relative_error(estimate: float, reference: float | None) -> float | None:
    if reference is None:
        return None
    if reference == 0.0:
        return 0.0 if estimate == 0.0 else None
    return abs(estimate - reference) / abs(reference)


def _run_estimator(
    *,
    product: str,
    financial_model: str,
    config: QMCConfig,
    estimator: Estimator,
    reference_price: float | None,
    reference_label: str | None,
    control_variate: bool,
    signature_input: object,
    observation_times: FloatArray | None = None,
) -> QMCPricingResult:
    started = perf_counter()
    seeds = qmc_replicate_seeds(config)
    replicate_estimates: list[float] = []
    replicate_details: list[QMCReplicateDiagnostic] = []
    underlying_means: list[float] = []
    itm_probabilities: list[float] = []
    pseudo_standard_error: float | None = None
    for replicate_index, seed in enumerate(seeds):
        replicate_started = perf_counter()
        observations = estimator(generate_normal_samples(config, replicate_index))
        if (
            observations.values.ndim != 1
            or observations.values.size != config.effective_num_samples
        ):
            raise QMCError("pricing estimator returned an invalid observation vector")
        if (
            observations.underlying.shape != observations.values.shape
            or observations.itm.shape != observations.values.shape
        ):
            raise QMCError("pricing estimator diagnostics must match the observation vector")
        if not (
            np.all(np.isfinite(observations.values))
            and np.all(np.isfinite(observations.underlying))
            and np.all(np.isfinite(observations.itm))
        ):
            raise QMCError("pricing estimator returned non-finite observations")
        estimate = float(np.mean(observations.values))
        replicate_estimates.append(estimate)
        underlying_means.append(float(np.mean(observations.underlying)))
        itm_probabilities.append(float(np.mean(observations.itm)))
        if config.method is SamplingMethod.PSEUDORANDOM:
            pseudo_standard_error = (
                float(np.std(observations.values, ddof=1) / math.sqrt(observations.values.size))
                if observations.values.size > 1
                else 0.0
            )
        replicate_details.append(
            QMCReplicateDiagnostic(
                replicate_index=replicate_index,
                seed=seed,
                estimate=estimate,
                runtime_ms=(perf_counter() - replicate_started) * 1_000.0,
            )
        )
    replicate_array = np.asarray(replicate_estimates, dtype=np.float64)
    price = float(np.mean(replicate_array))
    replicate_std = float(np.std(replicate_array, ddof=1)) if replicate_array.size > 1 else None
    standard_error: float | None
    if config.method is SamplingMethod.SCRAMBLED_SOBOL:
        assert replicate_std is not None
        standard_error = replicate_std / math.sqrt(replicate_array.size)
    elif config.method is SamplingMethod.PSEUDORANDOM:
        standard_error = pseudo_standard_error
    else:
        standard_error = None
    if standard_error is None:
        low = high = None
    else:
        quantile = float(norm.ppf(0.5 + config.confidence_level / 2.0))
        low = price - quantile * standard_error
        high = price + quantile * standard_error
    absolute_error = None if reference_price is None else abs(price - reference_price)
    return QMCPricingResult(
        product=product,
        price=price,
        standard_error=standard_error,
        confidence_interval_low=low,
        confidence_interval_high=high,
        confidence_level=config.confidence_level,
        requested_samples=config.num_samples,
        effective_samples=config.effective_num_samples,
        sampling_method=config.method,
        qmc_config=config,
        replicate_diagnostics=replicate_details,
        replicate_standard_deviation=replicate_std,
        replicate_minimum=float(np.min(replicate_array)),
        replicate_maximum=float(np.max(replicate_array)),
        runtime_ms=(perf_counter() - started) * 1_000.0,
        reference_price=reference_price,
        reference_label=reference_label,
        absolute_error=absolute_error,
        relative_error=_relative_error(price, reference_price),
        mean_underlying_value=float(np.mean(underlying_means)),
        probability_itm=float(np.mean(itm_probabilities)),
        control_variate=control_variate,
        metadata=numerical_method_metadata(
            config, financial_model, signature_input, observation_times
        ),
    )


def price_european_qmc(request: QMCEuropeanRequest) -> QMCPricingResult:
    if request.qmc.dimension != 1:
        raise QMCError("European terminal GBM QMC requires dimension=1")
    reference = black_scholes_price(request.contract, request.market)
    discount = finite_exp(
        -request.market.risk_free_rate * request.contract.maturity_years,
        context="QMC European discount factor",
    )

    def estimator(normals: FloatArray) -> _EstimatorObservations:
        terminal = terminal_prices_from_normals(request.contract, request.market, normals[:, 0])
        values = np.asarray(discount * vanilla_payoff(terminal, request.contract), dtype=np.float64)
        itm = (
            terminal > request.contract.strike
            if request.contract.option_type is OptionType.CALL
            else terminal < request.contract.strike
        )
        return _EstimatorObservations(values, terminal, np.asarray(itm, dtype=np.float64))

    return _run_estimator(
        product="EUROPEAN_VANILLA",
        financial_model="RISK_NEUTRAL_GBM",
        config=request.qmc,
        estimator=estimator,
        reference_price=reference,
        reference_label="Black-Scholes-Merton analytical price",
        control_variate=False,
        signature_input=request,
    )


def price_asian_qmc(request: QMCAsianRequest) -> QMCPricingResult:
    contract = request.contract
    future_times = np.asarray(contract.future_monitoring_times, dtype=np.float64)
    expected_dimension = max(1, future_times.size)
    if request.qmc.dimension != expected_dimension:
        raise QMCError(
            f"Asian QMC dimension must equal future monitoring count ({expected_dimension})"
        )
    if request.use_geometric_control and contract.averaging_type is AveragingType.GEOMETRIC:
        raise QMCError("geometric control applies only to arithmetic Asian options")
    discount = finite_exp(
        -request.market.risk_free_rate * contract.maturity_years,
        context="QMC Asian discount factor",
    )
    control_contract = contract.model_copy(update={"averaging_type": AveragingType.GEOMETRIC})
    control_expected = (
        geometric_asian_analytical_price(control_contract, request.market).price
        if request.use_geometric_control
        else None
    )

    def estimator(normals: FloatArray) -> _EstimatorObservations:
        if future_times.size:
            drivers = normalized_increment_drivers(
                normals[:, : future_times.size], future_times, request.qmc.path_construction
            )
            future = observation_prices_from_normals(
                contract.spot, request.market, future_times, drivers
            )
        else:
            future = np.empty((normals.shape[0], 0), dtype=np.float64)
        observations = (
            np.column_stack((np.full(normals.shape[0], contract.spot), future))
            if contract.observation_times[0] == 0.0
            else future
        )
        averages = asian_average(observations, contract.averaging_type)
        target = np.asarray(
            discount * asian_payoff_from_average(averages, contract), dtype=np.float64
        )
        if control_expected is not None:
            geometric_average = asian_average(observations, AveragingType.GEOMETRIC)
            control = np.asarray(
                discount * asian_payoff_from_average(geometric_average, control_contract),
                dtype=np.float64,
            )
            target = _control_adjusted(target, control, control_expected)
        itm = (
            averages > contract.strike
            if contract.option_type is OptionType.CALL
            else averages < contract.strike
        )
        return _EstimatorObservations(target, averages, np.asarray(itm, dtype=np.float64))

    reference: float | None
    reference_label: str | None
    if contract.averaging_type is AveragingType.GEOMETRIC:
        reference = geometric_asian_analytical_price(contract, request.market).price
        reference_label = "exact discrete geometric Asian analytical price"
    else:
        reference = request.numerical_reference_price
        reference_label = request.numerical_reference_label if reference is not None else None
    return _run_estimator(
        product=f"{contract.averaging_type.value}_ASIAN",
        financial_model="RISK_NEUTRAL_GBM",
        config=request.qmc,
        estimator=estimator,
        reference_price=reference,
        reference_label=reference_label,
        control_variate=request.use_geometric_control,
        signature_input=request,
        observation_times=future_times if future_times.size else None,
    )


def price_basket_qmc(request: QMCBasketRequest) -> QMCPricingResult:
    contract, market = request.contract, request.market
    if len(contract.weights) != market.num_assets:
        raise QMCError("basket weights must match the market asset count")
    if request.qmc.dimension != market.num_assets:
        raise QMCError(f"basket QMC dimension must equal asset count ({market.num_assets})")
    if request.qmc.path_construction is not QMCPathConstruction.STANDARD:
        raise QMCError("terminal basket QMC supports STANDARD path construction only")
    if request.use_geometric_control and contract.basket_type is BasketType.GEOMETRIC:
        raise QMCError("geometric control applies only to arithmetic baskets")
    discount = finite_exp(
        -market.risk_free_rate * contract.maturity_years,
        context="QMC basket discount factor",
    )
    control_contract = contract.model_copy(update={"basket_type": BasketType.GEOMETRIC})
    control_expected = (
        geometric_basket_analytical_price(control_contract, market).price
        if request.use_geometric_control
        else None
    )

    def estimator(normals: FloatArray) -> _EstimatorObservations:
        terminal = multi_asset_terminal_from_normals(market, contract.maturity_years, normals)
        basket = basket_values(terminal, contract.weights, contract.basket_type)
        target = np.asarray(
            discount * basket_payoff_from_values(basket, contract), dtype=np.float64
        )
        if control_expected is not None:
            geometric = basket_values(terminal, contract.weights, BasketType.GEOMETRIC)
            control = np.asarray(
                discount * basket_payoff_from_values(geometric, control_contract), dtype=np.float64
            )
            target = _control_adjusted(target, control, control_expected)
        itm = (
            basket > contract.strike
            if contract.option_type is OptionType.CALL
            else basket < contract.strike
        )
        return _EstimatorObservations(target, basket, np.asarray(itm, dtype=np.float64))

    reference: float | None
    reference_label: str | None
    if contract.basket_type is BasketType.GEOMETRIC:
        reference = geometric_basket_analytical_price(contract, market).price
        reference_label = "exact geometric basket analytical price"
    else:
        reference = request.numerical_reference_price
        reference_label = request.numerical_reference_label if reference is not None else None
    return _run_estimator(
        product=f"{contract.basket_type.value}_BASKET",
        financial_model="CORRELATED_RISK_NEUTRAL_GBM",
        config=request.qmc,
        estimator=estimator,
        reference_price=reference,
        reference_label=reference_label,
        control_variate=request.use_geometric_control,
        signature_input=request,
    )
