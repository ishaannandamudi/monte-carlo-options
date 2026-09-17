"""Heston calibration dataset conversion, filtering, deduplication, and weights."""

import math
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from backend.app.core.exceptions import HestonCalibrationError, QuantitativeError
from backend.app.models.heston import HestonParameters
from backend.app.models.heston_calibration import (
    CalibrationExclusion,
    CalibrationSelectionPolicy,
    CalibrationWeighting,
    HestonCalibrationBounds,
    HestonCalibrationConfig,
    HestonCalibrationDataset,
    HestonCalibrationObservation,
    HestonCalibrationTarget,
    finite_parameter_vector,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.market_data import QuoteIssue
from backend.app.models.option import OptionContract, OptionType
from backend.app.models.results import EnrichedOptionChain
from backend.app.models.volatility import ImpliedVolatilityMarket, ImpliedVolatilityObservation
from backend.app.models.volatility_surface import VolatilitySurface
from backend.app.pricing.black_scholes import black_scholes_price
from backend.app.pricing.black_scholes_greeks import black_scholes_greeks
from backend.app.pricing.implied_volatility import (
    european_option_price_bounds,
    forward_price,
    solve_implied_volatility,
)

HESTON_PARAMETER_ORDER = ("v0", "kappa", "theta", "sigma_v", "rho")


@dataclass(frozen=True)
class PreparedCalibrationObservation:
    original_index: int
    observation: HestonCalibrationObservation
    contract: OptionContract
    market_price: float
    market_implied_volatility: float
    target_value: float
    forward: float
    moneyness: float
    log_moneyness: float
    weight: float


@dataclass(frozen=True)
class PreparedCalibrationData:
    dataset: HestonCalibrationDataset
    observations: tuple[PreparedCalibrationObservation, ...]
    exclusions: tuple[CalibrationExclusion, ...]
    exclusion_reasons: dict[str, int]


def heston_parameters_to_vector(parameters: HestonParameters) -> NDArray[np.float64]:
    """Return the canonical ``v0,kappa,theta,sigma_v,rho`` native-space vector."""

    return np.asarray([float(getattr(parameters, name)) for name in HESTON_PARAMETER_ORDER])


def vector_to_heston_parameters(values: NDArray[np.float64] | list[float]) -> HestonParameters:
    """Construct parameters from the canonical native-space vector."""

    vector = [float(value) for value in values]
    if not finite_parameter_vector(vector):
        raise HestonCalibrationError(
            "Heston calibration parameter vector must contain five finite values"
        )
    return HestonParameters(**dict(zip(HESTON_PARAMETER_ORDER, vector, strict=True)))


def scipy_parameter_bounds(
    bounds: HestonCalibrationBounds,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    lower = np.asarray([getattr(bounds, name).lower for name in HESTON_PARAMETER_ORDER])
    upper = np.asarray([getattr(bounds, name).upper for name in HESTON_PARAMETER_ORDER])
    return lower, upper


def parameters_within_bounds(
    parameters: HestonParameters,
    bounds: HestonCalibrationBounds,
) -> bool:
    lower, upper = scipy_parameter_bounds(bounds)
    vector = heston_parameters_to_vector(parameters)
    return bool(np.all(vector >= lower) and np.all(vector <= upper))


def _spot_from_chain(chain: EnrichedOptionChain) -> float:
    quote = chain.underlying_quote
    spot = quote.midpoint if quote.midpoint is not None else quote.last
    if spot is None or spot <= 0.0:
        raise HestonCalibrationError("enriched option chain has no positive underlying price")
    return spot


def calibration_dataset_from_chain(chain: EnrichedOptionChain) -> HestonCalibrationDataset:
    """Convert Phase 5 enriched quotes without changing their quality decisions."""

    observations = []
    for item in chain.quotes:
        quote = item.quote
        observations.append(
            HestonCalibrationObservation(
                strike=quote.instrument.strike,
                maturity_years=max(item.maturity_years, 1e-15),
                option_type=quote.instrument.option_type,
                observed_price=item.market_price,
                observed_implied_volatility=item.implied_volatility,
                bid=quote.bid,
                ask=quote.ask,
                mid=quote.midpoint,
                source=quote.instrument.provider_symbol,
                quote_timestamp=quote.timestamp,
                quality_issues=[issue.value for issue in item.quality_issues],
                quote_is_fresh=(
                    QuoteIssue.STALE not in item.quality_issues
                    and QuoteIssue.FUTURE_TIMESTAMP not in item.quality_issues
                ),
                arbitrage_valid=(
                    QuoteIssue.PRICE_BELOW_LOWER_BOUND not in item.quality_issues
                    and QuoteIssue.PRICE_ABOVE_UPPER_BOUND not in item.quality_issues
                ),
            )
        )
    return HestonCalibrationDataset(
        spot=_spot_from_chain(chain),
        risk_free_rate=chain.risk_free_rate,
        dividend_yield=chain.dividend_yield,
        observations=observations,
        valuation_timestamp=chain.valuation_time,
        underlying_symbol=chain.underlying_symbol,
        metadata={"source": chain.source.value, "input": "PHASE5_ENRICHED_CHAIN"},
    )


def calibration_dataset_from_surface(surface: VolatilitySurface) -> HestonCalibrationDataset:
    """Flatten Phase 6 European-BSM volatility points into calibration observations."""

    observations = []
    for smile in surface.smiles:
        for point in smile.points:
            contract = OptionContract(
                option_type=point.option_type,
                spot=surface.spot,
                strike=point.strike,
                maturity_years=point.maturity_years,
            )
            reconstructed_price = black_scholes_price(
                contract,
                MarketEnvironment(
                    risk_free_rate=surface.risk_free_rate,
                    dividend_yield=surface.dividend_yield,
                    volatility=point.implied_volatility,
                ),
            )
            observations.append(
                HestonCalibrationObservation(
                    strike=point.strike,
                    maturity_years=point.maturity_years,
                    option_type=point.option_type,
                    observed_price=(
                        point.market_price
                        if point.market_price is not None
                        else reconstructed_price
                    ),
                    observed_implied_volatility=point.implied_volatility,
                    bid=point.bid,
                    ask=point.ask,
                    mid=point.market_price,
                    source=point.quality.provider_symbol or point.source,
                    quote_timestamp=point.quality.quote_timestamp,
                    quality_issues=point.quality.quality_issues,
                    quote_is_fresh=point.quality.is_fresh,
                    arbitrage_valid=point.quality.arbitrage_valid,
                )
            )
    return HestonCalibrationDataset(
        spot=surface.spot,
        risk_free_rate=surface.risk_free_rate,
        dividend_yield=surface.dividend_yield,
        observations=observations,
        valuation_timestamp=surface.valuation_timestamp,
        underlying_symbol=surface.underlying_symbol,
        metadata={"input": "PHASE6_VOLATILITY_SURFACE"},
    )


def _exclusion(
    index: int, observation: HestonCalibrationObservation, reason: str
) -> CalibrationExclusion:
    return CalibrationExclusion(
        observation_index=index,
        strike=observation.strike,
        maturity_years=observation.maturity_years,
        option_type=observation.option_type,
        reason=reason,
    )


def _market_values(
    dataset: HestonCalibrationDataset,
    observation: HestonCalibrationObservation,
) -> tuple[OptionContract, float | None, float | None]:
    contract = OptionContract(
        option_type=observation.option_type,
        spot=dataset.spot,
        strike=observation.strike,
        maturity_years=observation.maturity_years,
    )
    price = observation.observed_price
    if price is None:
        price = observation.mid
    implied = observation.observed_implied_volatility
    iv_market = ImpliedVolatilityMarket(
        risk_free_rate=dataset.risk_free_rate,
        dividend_yield=dataset.dividend_yield,
    )
    if price is None and implied is not None:
        price = black_scholes_price(
            contract,
            MarketEnvironment(
                risk_free_rate=dataset.risk_free_rate,
                dividend_yield=dataset.dividend_yield,
                volatility=implied,
            ),
        )
    if implied is None and price is not None:
        try:
            implied = solve_implied_volatility(
                ImpliedVolatilityObservation(
                    contract=contract,
                    market=iv_market,
                    observed_price=price,
                )
            ).implied_volatility
        except QuantitativeError:
            implied = None
    return contract, price, implied


def _base_filter_reason(
    dataset: HestonCalibrationDataset,
    observation: HestonCalibrationObservation,
    config: HestonCalibrationConfig,
    log_moneyness: float,
    price: float | None,
) -> str | None:
    filters = config.filters
    issues = set(observation.quality_issues)
    if filters.exclude_stale and (
        observation.quote_is_fresh is False or "STALE" in issues or "FUTURE_TIMESTAMP" in issues
    ):
        return "STALE_OR_FUTURE_QUOTE"
    if filters.exclude_crossed and (
        (
            observation.bid is not None
            and observation.ask is not None
            and observation.ask < observation.bid
        )
        or "CROSSED_MARKET" in issues
    ):
        return "CROSSED_MARKET"
    if "EXPIRED" in issues:
        return "EXPIRED"
    if "NEGATIVE_PRICE" in issues:
        return "NEGATIVE_PRICE"
    if filters.exclude_missing_mid and observation.mid is None:
        return "MISSING_MIDPOINT"
    if filters.exclude_arbitrage_invalid and observation.arbitrage_valid is False:
        return "ARBITRAGE_INVALID"
    if filters.minimum_log_moneyness is not None and log_moneyness < filters.minimum_log_moneyness:
        return "BELOW_MINIMUM_LOG_MONEYNESS"
    if filters.maximum_log_moneyness is not None and log_moneyness > filters.maximum_log_moneyness:
        return "ABOVE_MAXIMUM_LOG_MONEYNESS"
    strike_to_spot = observation.strike / dataset.spot
    if (
        filters.minimum_strike_to_spot is not None
        and strike_to_spot < filters.minimum_strike_to_spot
    ):
        return "BELOW_MINIMUM_STRIKE_TO_SPOT"
    if (
        filters.maximum_strike_to_spot is not None
        and strike_to_spot > filters.maximum_strike_to_spot
    ):
        return "ABOVE_MAXIMUM_STRIKE_TO_SPOT"
    if filters.minimum_price is not None and (price is None or price < filters.minimum_price):
        return "BELOW_MINIMUM_PRICE"
    if filters.allowed_maturities is not None and not any(
        math.isclose(observation.maturity_years, value, rel_tol=0.0, abs_tol=1e-12)
        for value in filters.allowed_maturities
    ):
        return "MATURITY_NOT_ALLOWED"
    if filters.maximum_relative_spread is not None:
        if observation.bid is None or observation.ask is None:
            return "RELATIVE_SPREAD_UNAVAILABLE"
        midpoint = observation.mid
        if midpoint is None or midpoint <= 0.0:
            return "RELATIVE_SPREAD_UNAVAILABLE"
        if (observation.ask - observation.bid) / midpoint > filters.maximum_relative_spread:
            return "RELATIVE_SPREAD_TOO_WIDE"
    return None


def _otm_selected(
    observation: HestonCalibrationObservation,
    forward: float,
) -> bool:
    tolerance = 1e-12 * max(1.0, forward)
    if observation.strike < forward - tolerance:
        return observation.option_type is OptionType.PUT
    if observation.strike > forward + tolerance:
        return observation.option_type is OptionType.CALL
    return observation.option_type is OptionType.CALL


def _duplicate_score(item: PreparedCalibrationObservation) -> tuple[float, int, float, str]:
    observation = item.observation
    if (
        observation.bid is not None
        and observation.ask is not None
        and observation.mid is not None
        and observation.mid > 0.0
        and observation.ask >= observation.bid
    ):
        spread = (observation.ask - observation.bid) / observation.mid
    else:
        spread = math.inf
    issue_count = len(observation.quality_issues)
    timestamp = (
        -observation.quote_timestamp.timestamp()
        if observation.quote_timestamp is not None
        else math.inf
    )
    return spread, issue_count, timestamp, observation.source or ""


def _raw_weight(
    item: PreparedCalibrationObservation,
    config: HestonCalibrationConfig,
    dataset: HestonCalibrationDataset,
) -> float | None:
    observation = item.observation
    if config.weighting is CalibrationWeighting.UNIFORM:
        return 1.0
    if config.weighting is CalibrationWeighting.CUSTOM:
        return observation.weight
    if config.weighting is CalibrationWeighting.BID_ASK:
        if observation.bid is None or observation.ask is None or observation.ask < observation.bid:
            return None
        spread = max(observation.ask - observation.bid, config.spread_floor)
        return 1.0 / (spread * spread)
    vega = black_scholes_greeks(
        item.contract,
        MarketEnvironment(
            risk_free_rate=dataset.risk_free_rate,
            dividend_yield=dataset.dividend_yield,
            volatility=item.market_implied_volatility,
        ),
    ).vega
    floored = max(vega, config.vega_floor)
    if config.weighting is CalibrationWeighting.VEGA:
        return floored * floored
    return 1.0 / (floored * floored)


def prepare_calibration_data(
    dataset: HestonCalibrationDataset,
    config: HestonCalibrationConfig,
) -> PreparedCalibrationData:
    """Apply explicit filtering, OTM selection, deduplication, and normalized weights."""

    candidates: list[PreparedCalibrationObservation] = []
    exclusions: list[CalibrationExclusion] = []
    iv_market = ImpliedVolatilityMarket(
        risk_free_rate=dataset.risk_free_rate,
        dividend_yield=dataset.dividend_yield,
    )
    for index, observation in enumerate(dataset.observations):
        forward = forward_price(
            dataset.spot,
            observation.maturity_years,
            dataset.risk_free_rate,
            dataset.dividend_yield,
        )
        log_moneyness = math.log(observation.strike / forward)
        contract, market_price, market_iv = _market_values(dataset, observation)
        reason = _base_filter_reason(dataset, observation, config, log_moneyness, market_price)
        if (
            reason is None
            and config.selection_policy is CalibrationSelectionPolicy.OTM_ONLY
            and not _otm_selected(observation, forward)
        ):
            reason = "NOT_SELECTED_BY_OTM_POLICY"
        if reason is None and market_price is None:
            reason = "MARKET_PRICE_UNAVAILABLE"
        if reason is None and market_iv is None:
            reason = "MARKET_IMPLIED_VOLATILITY_UNAVAILABLE"
        if reason is None and market_price is not None:
            bounds = european_option_price_bounds(contract, iv_market)
            if market_price < bounds.lower - 1e-8 or market_price > bounds.upper + 1e-8:
                reason = "PRICE_BOUND_VIOLATION"
        if reason is not None:
            exclusions.append(_exclusion(index, observation, reason))
            continue
        assert market_price is not None and market_iv is not None
        if config.target is HestonCalibrationTarget.PRICE:
            target = market_price
        elif config.target is HestonCalibrationTarget.IMPLIED_VOLATILITY:
            target = market_iv
        else:
            target = market_iv * market_iv * observation.maturity_years
        candidates.append(
            PreparedCalibrationObservation(
                original_index=index,
                observation=observation,
                contract=contract,
                market_price=market_price,
                market_implied_volatility=market_iv,
                target_value=target,
                forward=forward,
                moneyness=observation.strike / dataset.spot,
                log_moneyness=log_moneyness,
                weight=1.0,
            )
        )

    deduplicated: list[PreparedCalibrationObservation] = []
    groups: dict[tuple[float, float, OptionType], list[PreparedCalibrationObservation]] = {}
    for item in candidates:
        key = (
            item.observation.maturity_years,
            item.observation.strike,
            item.observation.option_type,
        )
        groups.setdefault(key, []).append(item)
    for key in sorted(groups, key=lambda value: (value[0], value[1], value[2].value)):
        ordered = sorted(groups[key], key=_duplicate_score)
        deduplicated.append(ordered[0])
        for duplicate in ordered[1:]:
            exclusions.append(
                _exclusion(
                    duplicate.original_index,
                    duplicate.observation,
                    "DUPLICATE_OBSERVATION",
                )
            )

    weighted: list[PreparedCalibrationObservation] = []
    raw_weights: list[float] = []
    for item in deduplicated:
        weight = _raw_weight(item, config, dataset)
        if weight is None or not math.isfinite(weight) or weight < 0.0:
            exclusions.append(
                _exclusion(item.original_index, item.observation, "WEIGHT_UNAVAILABLE")
            )
            continue
        weighted.append(item)
        raw_weights.append(weight)
    if not weighted:
        raise HestonCalibrationError("no calibration observations remain after filtering")
    if not any(weight > 0.0 for weight in raw_weights):
        raise HestonCalibrationError("calibration weights must contain at least one positive value")

    weights = np.asarray(raw_weights, dtype=np.float64)
    if config.expiry_balancing:
        maturities = sorted({item.observation.maturity_years for item in weighted})
        for maturity in maturities:
            indices = [
                index
                for index, item in enumerate(weighted)
                if item.observation.maturity_years == maturity
            ]
            total = float(np.sum(weights[indices]))
            if total > 0.0:
                weights[indices] /= total
    weights *= len(weights) / float(np.sum(weights))
    observations = tuple(
        replace(item, weight=float(weight)) for item, weight in zip(weighted, weights, strict=True)
    )
    exclusions.sort(key=lambda item: item.observation_index)
    reason_counts: dict[str, int] = {}
    for exclusion in exclusions:
        reason_counts[exclusion.reason] = reason_counts.get(exclusion.reason, 0) + 1
    return PreparedCalibrationData(
        dataset=dataset,
        observations=observations,
        exclusions=tuple(exclusions),
        exclusion_reasons=reason_counts,
    )


def prepared_dataset(data: PreparedCalibrationData) -> HestonCalibrationDataset:
    """Create a public dataset containing exactly the selected observations."""

    return data.dataset.model_copy(
        update={"observations": [item.observation for item in data.observations]}
    )


def clone_dataset_with_observations(
    dataset: HestonCalibrationDataset,
    observations: list[HestonCalibrationObservation],
) -> HestonCalibrationDataset:
    return dataset.model_copy(update={"observations": observations})
