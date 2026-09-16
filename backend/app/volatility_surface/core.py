"""Construction and total-variance interpolation for volatility smiles and surfaces."""

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import datetime

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator

from backend.app.analytics.market_data import enrich_option_chain, is_arbitrage_valid, is_fresh
from backend.app.core.exceptions import VolatilitySurfaceError
from backend.app.core.numerics import finite_exp
from backend.app.models.market_data import OptionChainSnapshot, QuoteQualityConfig
from backend.app.models.option import OptionType
from backend.app.models.results import EnrichedOptionChain, EnrichedOptionQuote
from backend.app.models.volatility import ImpliedVolatilityConfig
from backend.app.models.volatility_surface import (
    ExtrapolationPolicy,
    MaturityInterpolationMethod,
    SmileBuildResult,
    SmileConstructionConfig,
    SmileConstructionDiagnostics,
    SmileConstructionMetadata,
    SmileInterpolationConfig,
    SmileInterpolationMethod,
    SmileQueryResult,
    SmileSelectionPolicy,
    SurfaceQueryResult,
    VolatilityPoint,
    VolatilityPointQuality,
    VolatilitySmile,
    VolatilitySurface,
)


def implied_vol_to_total_variance(implied_volatility: float, maturity_years: float) -> float:
    """Return ``sigma² T`` after explicit finite domain validation."""

    if not math.isfinite(implied_volatility) or implied_volatility < 0.0:
        raise VolatilitySurfaceError("implied volatility must be finite and non-negative")
    if not math.isfinite(maturity_years) or maturity_years <= 0.0:
        raise VolatilitySurfaceError("maturity must be finite and positive")
    result = implied_volatility * implied_volatility * maturity_years
    if not math.isfinite(result):
        raise VolatilitySurfaceError("total implied variance exceeded floating-point range")
    return result


def total_variance_to_implied_vol(total_variance: float, maturity_years: float) -> float:
    """Return ``sqrt(w / T)`` without permitting invalid square roots."""

    if not math.isfinite(total_variance) or total_variance < 0.0:
        raise VolatilitySurfaceError("total variance must be finite and non-negative")
    if not math.isfinite(maturity_years) or maturity_years <= 0.0:
        raise VolatilitySurfaceError("maturity must be finite and positive")
    result = math.sqrt(total_variance / maturity_years)
    if not math.isfinite(result):
        raise VolatilitySurfaceError("implied volatility exceeded floating-point range")
    return result


def calculate_forward(
    spot: float,
    maturity_years: float,
    risk_free_rate: float,
    dividend_yield: float,
) -> float:
    """Return ``S exp((r-q)T)`` for the flat-rate Phase 6 convention."""

    if not math.isfinite(spot) or spot <= 0.0:
        raise VolatilitySurfaceError("spot must be finite and positive")
    if not math.isfinite(maturity_years) or maturity_years <= 0.0:
        raise VolatilitySurfaceError("maturity must be finite and positive")
    if not math.isfinite(risk_free_rate) or not math.isfinite(dividend_yield):
        raise VolatilitySurfaceError("rates must be finite")
    result = spot * finite_exp(
        (risk_free_rate - dividend_yield) * maturity_years,
        context="volatility-surface forward",
    )
    if not math.isfinite(result) or result <= 0.0:
        raise VolatilitySurfaceError("forward price must be finite and positive")
    return result


def make_volatility_point(
    *,
    strike: float,
    maturity_years: float,
    implied_volatility: float,
    option_type: OptionType,
    spot: float,
    forward: float,
    market_price: float | None = None,
    bid: float | None = None,
    ask: float | None = None,
    source: str | None = None,
    expiration: datetime | None = None,
    quality: VolatilityPointQuality | None = None,
) -> VolatilityPoint:
    """Normalize one observation into strike, spot, forward, and log coordinates."""

    if not math.isfinite(spot) or spot <= 0.0:
        raise VolatilitySurfaceError("spot must be finite and positive")
    if not math.isfinite(forward) or forward <= 0.0:
        raise VolatilitySurfaceError("forward must be finite and positive")
    forward_moneyness = strike / forward
    return VolatilityPoint(
        strike=strike,
        maturity_years=maturity_years,
        implied_volatility=implied_volatility,
        option_type=option_type,
        market_price=market_price,
        bid=bid,
        ask=ask,
        forward_price=forward,
        moneyness=strike / spot,
        forward_moneyness=forward_moneyness,
        log_moneyness=math.log(forward_moneyness),
        source=source,
        expiration=expiration,
        quality=quality or VolatilityPointQuality(),
    )


def build_smile(
    *,
    points: Sequence[VolatilityPoint],
    maturity_years: float,
    spot: float,
    risk_free_rate: float,
    dividend_yield: float,
    underlying_symbol: str | None = None,
    valuation_timestamp: datetime | None = None,
    expiration: datetime | None = None,
    minimum_points: int = 1,
    source: str | None = None,
) -> SmileBuildResult:
    """Build one normalized slice, rejecting duplicate canonical coordinates."""

    if len(points) < minimum_points:
        raise VolatilitySurfaceError(
            f"smile requires at least {minimum_points} points; received {len(points)}"
        )
    forward = calculate_forward(spot, maturity_years, risk_free_rate, dividend_yield)
    metadata = SmileConstructionMetadata(
        minimum_points=minimum_points,
        accepted_points=len(points),
        source=source,
    )
    smile = VolatilitySmile(
        underlying_symbol=underlying_symbol,
        valuation_timestamp=valuation_timestamp,
        expiration=expiration,
        maturity_years=maturity_years,
        spot=spot,
        forward=forward,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        points=list(points),
        construction_metadata=metadata,
    )
    diagnostics = SmileConstructionDiagnostics(
        total_quotes_considered=len(points),
        quotes_excluded=0,
        iv_points_accepted=len(points),
        smiles_created=1,
        expirations_rejected=0,
    )
    return SmileBuildResult(smile=smile, diagnostics=diagnostics)


def _exclusion_reason(
    item: EnrichedOptionQuote,
    config: SmileConstructionConfig,
) -> str | None:
    volatility = item.implied_volatility
    if item.maturity_years <= 0.0:
        return "INVALID_MATURITY"
    if volatility is None or not math.isfinite(volatility):
        return "IV_UNAVAILABLE"
    if volatility < config.min_implied_volatility or volatility > config.max_implied_volatility:
        return "IV_OUTSIDE_CONFIGURED_RANGE"
    if item.forward_price is None or not math.isfinite(item.forward_price):
        return "FORWARD_UNAVAILABLE"
    if config.require_fresh_quote and not is_fresh(item.quality_issues):
        return "STALE_OR_FUTURE_QUOTE"
    if config.require_arbitrage_valid and not is_arbitrage_valid(item):
        return "PRICE_ARBITRAGE_INVALID"
    if config.maximum_relative_spread is not None and (
        item.relative_spread is None or item.relative_spread > config.maximum_relative_spread
    ):
        return "RELATIVE_SPREAD_UNACCEPTABLE"
    return None


def _selection_reason(
    item: EnrichedOptionQuote,
    config: SmileConstructionConfig,
) -> str | None:
    option_type = item.quote.instrument.option_type
    if config.selection_policy is SmileSelectionPolicy.CALLS_ONLY:
        return None if option_type is OptionType.CALL else "OPTION_SIDE_NOT_SELECTED"
    if config.selection_policy is SmileSelectionPolicy.PUTS_ONLY:
        return None if option_type is OptionType.PUT else "OPTION_SIDE_NOT_SELECTED"
    if config.selection_policy is SmileSelectionPolicy.ALL_VALID:
        return None
    if item.log_moneyness is None:
        return "LOG_MONEYNESS_UNAVAILABLE"
    if abs(item.log_moneyness) <= config.atm_log_moneyness_tolerance:
        return None
    preferred = OptionType.PUT if item.log_moneyness < 0.0 else OptionType.CALL
    return None if option_type is preferred else "OPTION_SIDE_NOT_OTM"


def _candidate_rank(
    item: EnrichedOptionQuote,
    config: SmileConstructionConfig,
) -> tuple[int, float, int, float, str]:
    """Preferred OTM side; at ATM use spread then call; finish deterministically."""

    spread = item.relative_spread if item.relative_spread is not None else math.inf
    call_rank = 0 if item.quote.instrument.option_type is OptionType.CALL else 1
    log_moneyness = item.log_moneyness
    side_rank = 0
    if log_moneyness is not None and abs(log_moneyness) > config.atm_log_moneyness_tolerance:
        preferred = OptionType.PUT if log_moneyness < 0.0 else OptionType.CALL
        side_rank = 0 if item.quote.instrument.option_type is preferred else 1
    else:
        # ATM is explicitly spread-first; calls are only the final option-side tie-break.
        side_rank = 0
    freshness_rank = -item.quote.timestamp.timestamp()
    return side_rank, spread, call_rank, freshness_rank, item.quote.instrument.provider_symbol


def _market_point(item: EnrichedOptionQuote, *, spot: float) -> VolatilityPoint:
    instrument = item.quote.instrument
    volatility = item.implied_volatility
    forward = item.forward_price
    if volatility is None or forward is None:
        raise VolatilitySurfaceError("market point was not fully enriched")
    quality = VolatilityPointQuality(
        quality_issues=[issue.value for issue in item.quality_issues],
        relative_spread=item.relative_spread,
        quote_timestamp=item.quote.timestamp,
        provider_symbol=instrument.provider_symbol,
        is_fresh=is_fresh(item.quality_issues),
        arbitrage_valid=is_arbitrage_valid(item),
    )
    return make_volatility_point(
        strike=instrument.strike,
        maturity_years=item.maturity_years,
        implied_volatility=volatility,
        option_type=instrument.option_type,
        spot=spot,
        forward=forward,
        market_price=item.market_price,
        bid=item.quote.bid,
        ask=item.quote.ask,
        source=item.quote.source.value,
        expiration=instrument.expiration,
        quality=quality,
    )


def construct_market_smiles(
    chain: EnrichedOptionChain,
    config: SmileConstructionConfig | None = None,
) -> tuple[list[VolatilitySmile], SmileConstructionDiagnostics]:
    """Convert an enriched Phase 5 chain into deterministically cleaned slices."""

    policy = config or SmileConstructionConfig()
    spot = chain.underlying_quote.midpoint or chain.underlying_quote.last
    if spot is None or spot <= 0.0:
        raise VolatilitySurfaceError("the enriched chain has no positive underlying price")
    reasons: Counter[str] = Counter()
    eligible: dict[datetime, list[EnrichedOptionQuote]] = defaultdict(list)
    for item in chain.quotes:
        reason = _exclusion_reason(item, policy) or _selection_reason(item, policy)
        if reason is not None:
            reasons[reason] += 1
            continue
        eligible[item.quote.instrument.expiration].append(item)

    all_expirations = sorted({item.quote.instrument.expiration for item in chain.quotes})
    smiles: list[VolatilitySmile] = []
    rejected_expirations: list[datetime] = []
    accepted_count = 0
    for expiration in all_expirations:
        by_strike: dict[float, list[EnrichedOptionQuote]] = defaultdict(list)
        for item in eligible[expiration]:
            by_strike[item.quote.instrument.strike].append(item)
        chosen: list[EnrichedOptionQuote] = []
        for candidates in by_strike.values():
            ranked = sorted(candidates, key=lambda item: _candidate_rank(item, policy))
            chosen.append(ranked[0])
            reasons["DUPLICATE_STRIKE"] += len(ranked) - 1
        chosen.sort(key=lambda item: item.quote.instrument.strike)
        if len(chosen) < policy.minimum_points:
            rejected_expirations.append(expiration)
            reasons["INSUFFICIENT_POINTS"] += len(chosen)
            continue
        points = [_market_point(item, spot=spot) for item in chosen]
        accepted_count += len(points)
        first = chosen[0]
        maturity = first.maturity_years
        forward = first.forward_price
        if forward is None:
            raise VolatilitySurfaceError("eligible market slice has no forward")
        smiles.append(
            VolatilitySmile(
                underlying_symbol=chain.underlying_symbol,
                valuation_timestamp=chain.valuation_time,
                expiration=expiration,
                maturity_years=maturity,
                spot=spot,
                forward=forward,
                risk_free_rate=chain.risk_free_rate,
                dividend_yield=chain.dividend_yield,
                points=points,
                construction_metadata=SmileConstructionMetadata(
                    selection_policy=policy.selection_policy,
                    minimum_points=policy.minimum_points,
                    accepted_points=len(points),
                    excluded_points=(
                        sum(item.quote.instrument.expiration == expiration for item in chain.quotes)
                        - len(points)
                    ),
                    source=chain.source.value,
                ),
            )
        )
    excluded = len(chain.quotes) - accepted_count
    diagnostics = SmileConstructionDiagnostics(
        total_quotes_considered=len(chain.quotes),
        quotes_excluded=excluded,
        iv_points_accepted=accepted_count,
        smiles_created=len(smiles),
        expirations_rejected=len(rejected_expirations),
        exclusion_reasons=dict(sorted(reasons.items())),
        rejected_expirations=rejected_expirations,
    )
    return smiles, diagnostics


def _interpolated_total_variance(
    coordinates: NDArray[np.float64],
    variances: NDArray[np.float64],
    target: float,
    method: SmileInterpolationMethod,
) -> float:
    if method is SmileInterpolationMethod.LINEAR:
        result = float(np.interp(target, coordinates, variances))
    else:
        if coordinates.size < 2:
            raise VolatilitySurfaceError("PCHIP interpolation requires at least two points")
        result = float(PchipInterpolator(coordinates, variances, extrapolate=False)(target))
    if not math.isfinite(result) or result < 0.0:
        raise VolatilitySurfaceError("smile interpolation produced invalid total variance")
    return result


def query_smile(
    smile: VolatilitySmile,
    *,
    strike: float | None = None,
    log_moneyness: float | None = None,
    config: SmileInterpolationConfig | None = None,
) -> SmileQueryResult:
    """Query one smile by strike or canonical coordinate in total-variance space."""

    if (strike is None) == (log_moneyness is None):
        raise VolatilitySurfaceError("provide exactly one of strike or log_moneyness")
    if strike is not None:
        if not math.isfinite(strike) or strike <= 0.0:
            raise VolatilitySurfaceError("query strike must be finite and positive")
        target = math.log(strike / smile.forward)
    else:
        if log_moneyness is None or not math.isfinite(log_moneyness):
            raise VolatilitySurfaceError("query log_moneyness must be finite")
        target = log_moneyness
    policy = config or SmileInterpolationConfig()
    coordinates = np.asarray([point.log_moneyness for point in smile.points], dtype=np.float64)
    variances = np.asarray(
        [
            implied_vol_to_total_variance(point.implied_volatility, smile.maturity_years)
            for point in smile.points
        ],
        dtype=np.float64,
    )
    lower = float(coordinates[0])
    upper = float(coordinates[-1])
    exact_index = next(
        (
            index
            for index, coordinate in enumerate(coordinates)
            if math.isclose(target, float(coordinate), rel_tol=0.0, abs_tol=1e-13)
        ),
        None,
    )
    diagnostics: list[str] = []
    extrapolated = False
    if exact_index is not None:
        point = smile.points[exact_index]
        volatility = point.implied_volatility
        total_variance = variances[exact_index]
        exact = True
    elif target < lower or target > upper:
        if policy.extrapolation is ExtrapolationPolicy.NONE:
            raise VolatilitySurfaceError(
                f"log-moneyness {target} is outside smile domain [{lower}, {upper}]"
            )
        endpoint = smile.points[0] if target < lower else smile.points[-1]
        volatility = endpoint.implied_volatility
        total_variance = implied_vol_to_total_variance(volatility, smile.maturity_years)
        exact = False
        extrapolated = True
        diagnostics.append("flat-volatility wing extrapolation used")
    else:
        total_variance = _interpolated_total_variance(
            coordinates,
            variances,
            target,
            policy.method,
        )
        volatility = total_variance_to_implied_vol(total_variance, smile.maturity_years)
        exact = False
    return SmileQueryResult(
        requested_strike=strike,
        log_moneyness=target,
        implied_volatility=volatility,
        total_variance=float(total_variance),
        interpolation_method=policy.method,
        exact_observation=exact,
        interpolated=not exact and not extrapolated,
        extrapolation_used=extrapolated,
        domain_lower=lower,
        domain_upper=upper,
        diagnostics=diagnostics,
    )


def build_surface(
    smiles: Sequence[VolatilitySmile],
    *,
    interpolation_config: SmileInterpolationConfig | None = None,
    construction_diagnostics: SmileConstructionDiagnostics | None = None,
) -> VolatilitySurface:
    """Build a maturity-sorted surface from existing immutable smile slices."""

    if not smiles:
        raise VolatilitySurfaceError("at least one smile is required")
    first = smiles[0]
    diagnostics = construction_diagnostics or SmileConstructionDiagnostics(
        total_quotes_considered=sum(len(smile.points) for smile in smiles),
        quotes_excluded=0,
        iv_points_accepted=sum(len(smile.points) for smile in smiles),
        smiles_created=len(smiles),
        expirations_rejected=0,
    )
    return VolatilitySurface(
        underlying_symbol=first.underlying_symbol,
        valuation_timestamp=first.valuation_timestamp,
        spot=first.spot,
        risk_free_rate=first.risk_free_rate,
        dividend_yield=first.dividend_yield,
        smiles=list(smiles),
        interpolation_config=interpolation_config or SmileInterpolationConfig(),
        construction_diagnostics=diagnostics,
    )


def query_surface(
    surface: VolatilitySurface,
    *,
    maturity_years: float,
    strike: float | None = None,
    log_moneyness: float | None = None,
    config: SmileInterpolationConfig | None = None,
) -> SurfaceQueryResult:
    """Query neighboring smiles at fixed ``k``, then interpolate total variance in time."""

    if not math.isfinite(maturity_years) or maturity_years <= 0.0:
        raise VolatilitySurfaceError("query maturity must be finite and positive")
    if (strike is None) == (log_moneyness is None):
        raise VolatilitySurfaceError("provide exactly one of strike or log_moneyness")
    forward = calculate_forward(
        surface.spot,
        maturity_years,
        surface.risk_free_rate,
        surface.dividend_yield,
    )
    if strike is not None:
        if not math.isfinite(strike) or strike <= 0.0:
            raise VolatilitySurfaceError("query strike must be finite and positive")
        target_k = math.log(strike / forward)
    else:
        if log_moneyness is None or not math.isfinite(log_moneyness):
            raise VolatilitySurfaceError("query log_moneyness must be finite")
        target_k = log_moneyness
    policy = config or surface.interpolation_config
    maturities = surface.maturities
    exact_index = next(
        (
            index
            for index, maturity in enumerate(maturities)
            if math.isclose(maturity_years, maturity, rel_tol=0.0, abs_tol=1e-13)
        ),
        None,
    )
    if exact_index is not None:
        smile_result = query_smile(
            surface.smiles[exact_index],
            log_moneyness=target_k,
            config=policy,
        )
        maturity = maturities[exact_index]
        return SurfaceQueryResult(
            requested_strike=strike,
            requested_maturity=maturity_years,
            forward=forward,
            log_moneyness=target_k,
            implied_volatility=smile_result.implied_volatility,
            total_variance=smile_result.total_variance,
            lower_maturity=maturity,
            upper_maturity=maturity,
            smile_interpolation_method=policy.method,
            maturity_interpolation_method=policy.maturity_method,
            exact_maturity=True,
            exact_observation=smile_result.exact_observation,
            smile_extrapolation_used=smile_result.extrapolation_used,
            maturity_extrapolation_used=False,
            diagnostics=smile_result.diagnostics,
        )

    lower_index = (
        next(
            (index for index, maturity in enumerate(maturities) if maturity > maturity_years),
            len(maturities),
        )
        - 1
    )
    upper_index = lower_index + 1
    maturity_extrapolated = lower_index < 0 or upper_index >= len(maturities)
    if maturity_extrapolated:
        if policy.maturity_extrapolation is ExtrapolationPolicy.NONE:
            raise VolatilitySurfaceError(
                f"maturity {maturity_years} is outside surface domain "
                f"[{maturities[0]}, {maturities[-1]}]"
            )
        nearest_index = 0 if maturity_years < maturities[0] else len(maturities) - 1
        nearest = query_smile(
            surface.smiles[nearest_index],
            log_moneyness=target_k,
            config=policy,
        )
        volatility = nearest.implied_volatility
        total_variance = implied_vol_to_total_variance(volatility, maturity_years)
        maturity = maturities[nearest_index]
        diagnostics = [*nearest.diagnostics, "flat-volatility maturity extrapolation used"]
        return SurfaceQueryResult(
            requested_strike=strike,
            requested_maturity=maturity_years,
            forward=forward,
            log_moneyness=target_k,
            implied_volatility=volatility,
            total_variance=total_variance,
            lower_maturity=maturity,
            upper_maturity=maturity,
            smile_interpolation_method=policy.method,
            maturity_interpolation_method=policy.maturity_method,
            exact_maturity=False,
            exact_observation=False,
            smile_extrapolation_used=nearest.extrapolation_used,
            maturity_extrapolation_used=True,
            diagnostics=diagnostics,
        )

    lower_smile = surface.smiles[lower_index]
    upper_smile = surface.smiles[upper_index]
    lower_result = query_smile(lower_smile, log_moneyness=target_k, config=policy)
    upper_result = query_smile(upper_smile, log_moneyness=target_k, config=policy)
    weight = (maturity_years - lower_smile.maturity_years) / (
        upper_smile.maturity_years - lower_smile.maturity_years
    )
    total_variance = lower_result.total_variance + weight * (
        upper_result.total_variance - lower_result.total_variance
    )
    volatility = total_variance_to_implied_vol(total_variance, maturity_years)
    return SurfaceQueryResult(
        requested_strike=strike,
        requested_maturity=maturity_years,
        forward=forward,
        log_moneyness=target_k,
        implied_volatility=volatility,
        total_variance=total_variance,
        lower_maturity=lower_smile.maturity_years,
        upper_maturity=upper_smile.maturity_years,
        smile_interpolation_method=policy.method,
        maturity_interpolation_method=MaturityInterpolationMethod.TOTAL_VARIANCE_LINEAR,
        exact_maturity=False,
        exact_observation=False,
        smile_extrapolation_used=(
            lower_result.extrapolation_used or upper_result.extrapolation_used
        ),
        maturity_extrapolation_used=False,
        diagnostics=[*lower_result.diagnostics, *upper_result.diagnostics],
    )


def build_market_surface(
    chain: EnrichedOptionChain,
    *,
    construction_config: SmileConstructionConfig | None = None,
    interpolation_config: SmileInterpolationConfig | None = None,
) -> tuple[VolatilitySurface, SmileConstructionDiagnostics]:
    """End-to-end Phase 5 enriched chain to Phase 6 surface integration."""

    smiles, diagnostics = construct_market_smiles(chain, construction_config)
    if not smiles:
        raise VolatilitySurfaceError("no expiration had enough eligible points for a surface")
    return (
        build_surface(
            smiles,
            interpolation_config=interpolation_config,
            construction_diagnostics=diagnostics,
        ),
        diagnostics,
    )


def build_surface_from_option_chain(
    snapshot: OptionChainSnapshot,
    *,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
    valuation_time: datetime | None = None,
    quote_quality_config: QuoteQualityConfig | None = None,
    iv_config: ImpliedVolatilityConfig | None = None,
    construction_config: SmileConstructionConfig | None = None,
    interpolation_config: SmileInterpolationConfig | None = None,
) -> tuple[VolatilitySurface, SmileConstructionDiagnostics]:
    """Enrich, filter, and construct a surface from a normalized Phase 5 chain."""

    enriched = enrich_option_chain(
        snapshot,
        valuation_time=valuation_time,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        quality_config=quote_quality_config,
        iv_config=iv_config,
    )
    return build_market_surface(
        enriched,
        construction_config=construction_config,
        interpolation_config=interpolation_config,
    )
