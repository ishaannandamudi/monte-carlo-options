"""Branch-stable Heston characteristic function and Fourier pricing."""

import cmath
import math
from dataclasses import dataclass
from time import perf_counter

from scipy.integrate import quad

from backend.app.core.exceptions import HestonPricingError, NumericalError
from backend.app.core.numerics import finite_exp
from backend.app.models.heston import (
    HestonAnalyticalMethod,
    HestonAnalyticalResult,
    HestonIntegrationConfig,
    HestonParameters,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.payoffs.vanilla import intrinsic_value
from backend.app.pricing.base import require_european_exercise
from backend.app.pricing.black_scholes import black_scholes_price


@dataclass(frozen=True)
class _IntegralResult:
    value: float
    error: float
    evaluations: int
    upper_bound: float


def deterministic_integrated_variance(parameters: HestonParameters, maturity_years: float) -> float:
    """Return integrated variance when vol-of-vol is numerically deterministic."""

    if parameters.kappa == 0.0:
        value = parameters.v0 * maturity_years
    else:
        value = (
            parameters.theta * maturity_years
            + (parameters.v0 - parameters.theta)
            * (-math.expm1(-parameters.kappa * maturity_years))
            / parameters.kappa
        )
    if not math.isfinite(value) or value < 0.0:
        raise NumericalError("deterministic Heston integrated variance is invalid")
    return value


def _uses_deterministic_variance_limit(parameters: HestonParameters, maturity_years: float) -> bool:
    if parameters.sigma_v == 0.0 or parameters.v0 == parameters.theta == 0.0:
        return True
    # C and D contain differences of O(sigma_v^2) divided by sigma_v^2.
    # The fourth-root epsilon boundary keeps at least square-root-epsilon
    # relative resolution in that cancellation before using the exact limit.
    variance_volatility_scale = max(
        math.sqrt(parameters.v0),
        math.sqrt(parameters.theta),
        math.sqrt(math.ulp(1.0)),
    )
    conditioning_boundary = math.sqrt(math.sqrt(math.ulp(1.0)))
    return (
        parameters.sigma_v * math.sqrt(maturity_years)
        <= conditioning_boundary * variance_volatility_scale
    )


def heston_characteristic_function(
    argument: complex,
    contract: OptionContract,
    market: MarketEnvironment,
    parameters: HestonParameters,
) -> complex:
    """Return ``E[exp(i*u*log(S_T))]`` using the Little Heston Trap form."""

    maturity = contract.maturity_years
    if argument == 0:
        return 1.0 + 0.0j
    if maturity == 0.0:
        return cmath.exp(1j * argument * math.log(contract.spot))
    if _uses_deterministic_variance_limit(parameters, maturity):
        integrated = deterministic_integrated_variance(parameters, maturity)
        mean_log = (
            math.log(contract.spot)
            + (market.risk_free_rate - market.dividend_yield) * maturity
            - 0.5 * integrated
        )
        value = cmath.exp(1j * argument * mean_log - 0.5 * argument * argument * integrated)
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise NumericalError("deterministic Heston characteristic function is non-finite")
        return value

    sigma = parameters.sigma_v
    sigma_squared = sigma * sigma
    iu = 1j * argument
    beta = parameters.kappa - parameters.rho * sigma * iu
    discriminant = beta * beta + sigma_squared * (argument * argument + iu)
    root = cmath.sqrt(discriminant)
    if root.real < 0.0:
        root = -root
    denominator = beta + root
    if abs(denominator) < 1e-15:
        raise HestonPricingError("Heston characteristic-function denominator is singular")
    g = (beta - root) / denominator
    decay = cmath.exp(-root * maturity)
    one_minus_g = 1.0 - g
    one_minus_g_decay = 1.0 - g * decay
    if abs(one_minus_g) < 1e-15 or abs(one_minus_g_decay) < 1e-15:
        raise HestonPricingError("Heston characteristic-function logarithm is singular")
    log_ratio = cmath.log(one_minus_g_decay / one_minus_g)
    c_term = iu * (
        market.risk_free_rate - market.dividend_yield
    ) * maturity + parameters.kappa * parameters.theta / sigma_squared * (
        (beta - root) * maturity - 2.0 * log_ratio
    )
    d_term = (beta - root) / sigma_squared * ((1.0 - decay) / one_minus_g_decay)
    exponent = iu * math.log(contract.spot) + c_term + d_term * parameters.v0
    value = cmath.exp(exponent)
    if not math.isfinite(value.real) or not math.isfinite(value.imag):
        raise NumericalError("Heston characteristic function produced a non-finite value")
    return value


def _probability_integrand(
    frequency: float,
    probability_index: int,
    contract: OptionContract,
    market: MarketEnvironment,
    parameters: HestonParameters,
    forward_mean: float,
) -> float:
    if frequency <= 0.0:
        frequency = 1e-10
    phase = cmath.exp(-1j * frequency * math.log(contract.strike))
    if probability_index == 1:
        characteristic = (
            heston_characteristic_function(complex(frequency, -1.0), contract, market, parameters)
            / forward_mean
        )
    else:
        characteristic = heston_characteristic_function(
            complex(frequency, 0.0), contract, market, parameters
        )
    value = (phase * characteristic / (1j * frequency)).real
    if not math.isfinite(value):
        raise NumericalError("Heston Fourier integrand is non-finite")
    return value


def _segmented_integral(
    probability_index: int,
    contract: OptionContract,
    market: MarketEnvironment,
    parameters: HestonParameters,
    config: HestonIntegrationConfig,
    forward_mean: float,
) -> _IntegralResult:
    lower = 1e-8
    total = 0.0
    error = 0.0
    evaluations = 0
    quiet_segments = 0
    end = 0.0
    while end < config.upper_bound:
        start = max(lower, end)
        end = min(config.upper_bound, end + config.segment_width)
        quadrature = quad(
            _probability_integrand,
            start,
            end,
            args=(probability_index, contract, market, parameters, forward_mean),
            epsabs=config.absolute_tolerance,
            epsrel=config.relative_tolerance,
            limit=config.quadrature_limit,
            full_output=1,
        )
        value, segment_error, information = quadrature[:3]
        if len(quadrature) > 3:
            raise HestonPricingError(
                f"Heston quadrature failed to converge on frequency interval [{start}, {end}]"
            )
        if not math.isfinite(value) or not math.isfinite(segment_error):
            raise NumericalError("Heston quadrature produced a non-finite result")
        total += float(value)
        error += float(segment_error)
        evaluations += int(information["neval"])
        if end >= 3.0 * config.segment_width and abs(value) <= config.tail_tolerance:
            quiet_segments += 1
        else:
            quiet_segments = 0
        if quiet_segments >= 2:
            break
    if end >= config.upper_bound and abs(value) > 10.0 * config.tail_tolerance:
        raise HestonPricingError(
            "Heston Fourier tail did not converge within integration upper_bound"
        )
    return _IntegralResult(
        value=total,
        error=error + abs(float(value)),
        evaluations=evaluations,
        upper_bound=end,
    )


def _deterministic_variance_price(
    contract: OptionContract,
    market: MarketEnvironment,
    parameters: HestonParameters,
) -> float:
    maturity = contract.maturity_years
    integrated = deterministic_integrated_variance(parameters, maturity)
    volatility = math.sqrt(integrated / maturity) if maturity > 0.0 else 0.0
    return black_scholes_price(
        contract,
        MarketEnvironment(
            risk_free_rate=market.risk_free_rate,
            dividend_yield=market.dividend_yield,
            volatility=volatility,
        ),
    )


def heston_analytical_price(
    contract: OptionContract,
    market: MarketEnvironment,
    parameters: HestonParameters,
    config: HestonIntegrationConfig | None = None,
) -> HestonAnalyticalResult:
    """Price a European option by branch-stable Heston Fourier inversion."""

    require_european_exercise(contract, engine="Heston semi-analytical pricing")
    started = perf_counter()
    integration = config or HestonIntegrationConfig()
    if contract.maturity_years == 0.0:
        return HestonAnalyticalResult(
            price=intrinsic_value(contract.option_type, contract.spot, contract.strike),
            integration_error=0.0,
            integration_evaluations=0,
            integration_upper_bound=0.0,
            runtime_ms=(perf_counter() - started) * 1_000.0,
            method=HestonAnalyticalMethod.INTRINSIC,
            feller_satisfied=parameters.feller_satisfied,
            feller_ratio=parameters.feller_ratio,
        )
    if _uses_deterministic_variance_limit(parameters, contract.maturity_years):
        return HestonAnalyticalResult(
            price=_deterministic_variance_price(contract, market, parameters),
            integration_error=0.0,
            integration_evaluations=0,
            integration_upper_bound=0.0,
            runtime_ms=(perf_counter() - started) * 1_000.0,
            method=HestonAnalyticalMethod.DETERMINISTIC_VARIANCE_LIMIT,
            feller_satisfied=parameters.feller_satisfied,
            feller_ratio=parameters.feller_ratio,
        )

    maturity = contract.maturity_years
    discounted_spot = contract.spot * finite_exp(
        -market.dividend_yield * maturity,
        context="Heston dividend discount factor",
    )
    discounted_strike = contract.strike * finite_exp(
        -market.risk_free_rate * maturity,
        context="Heston strike discount factor",
    )
    forward_mean = contract.spot * finite_exp(
        (market.risk_free_rate - market.dividend_yield) * maturity,
        context="Heston terminal spot mean",
    )
    first = _segmented_integral(1, contract, market, parameters, integration, forward_mean)
    second = _segmented_integral(2, contract, market, parameters, integration, forward_mean)
    probability_one = 0.5 + first.value / math.pi
    probability_two = 0.5 + second.value / math.pi
    if not (-1e-10 <= probability_one <= 1.0 + 1e-10) or not (
        -1e-10 <= probability_two <= 1.0 + 1e-10
    ):
        raise NumericalError("Heston Fourier probabilities are outside [0, 1]")
    probability_one = min(max(probability_one, 0.0), 1.0)
    probability_two = min(max(probability_two, 0.0), 1.0)
    call = max(discounted_spot * probability_one - discounted_strike * probability_two, 0.0)
    if contract.option_type is OptionType.CALL:
        price = call
    else:
        price = max(call - discounted_spot + discounted_strike, 0.0)
    integration_error = (discounted_spot * first.error + discounted_strike * second.error) / math.pi
    if not math.isfinite(price) or not math.isfinite(integration_error):
        raise NumericalError("Heston analytical pricing produced a non-finite result")
    return HestonAnalyticalResult(
        price=price,
        integration_error=integration_error,
        integration_evaluations=first.evaluations + second.evaluations,
        integration_upper_bound=max(first.upper_bound, second.upper_bound),
        runtime_ms=(perf_counter() - started) * 1_000.0,
        method=HestonAnalyticalMethod.LITTLE_TRAP_FOURIER,
        feller_satisfied=parameters.feller_satisfied,
        feller_ratio=parameters.feller_ratio,
    )
