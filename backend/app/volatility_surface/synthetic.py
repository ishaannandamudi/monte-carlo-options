"""Deterministic synthetic smile and surface generators for research tests."""

import math
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from backend.app.core.exceptions import VolatilitySurfaceError
from backend.app.models.option import OptionType
from backend.app.models.volatility_surface import (
    SmileConstructionMetadata,
    SmileInterpolationConfig,
    VolatilitySmile,
    VolatilitySurface,
)
from backend.app.volatility_surface.core import (
    build_surface,
    calculate_forward,
    make_volatility_point,
)


def generate_synthetic_smile(
    *,
    maturity_years: float,
    log_moneyness: Sequence[float],
    base_volatility: float = 0.20,
    skew: float = 0.0,
    curvature: float = 0.0,
    term_slope: float = 0.0,
    spot: float = 100.0,
    risk_free_rate: float = 0.02,
    dividend_yield: float = 0.0,
    valuation_timestamp: datetime | None = None,
    underlying_symbol: str = "SYNTHETIC",
) -> VolatilitySmile:
    """Generate ``sigma(k,T)=base+skew*k+curvature*k²+term_slope*T``."""

    if not log_moneyness:
        raise VolatilitySurfaceError("synthetic smile requires at least one coordinate")
    valuation = valuation_timestamp or datetime(2026, 1, 1, tzinfo=UTC)
    forward = calculate_forward(spot, maturity_years, risk_free_rate, dividend_yield)
    expiration = valuation + timedelta(days=365.0 * maturity_years)
    points = []
    for coordinate in log_moneyness:
        volatility = (
            base_volatility
            + skew * coordinate
            + curvature * coordinate * coordinate
            + term_slope * maturity_years
        )
        if volatility < 0.0:
            raise VolatilitySurfaceError("synthetic volatility formula produced a negative value")
        strike = forward * math.exp(coordinate)
        points.append(
            make_volatility_point(
                strike=strike,
                maturity_years=maturity_years,
                implied_volatility=volatility,
                option_type=OptionType.CALL,
                spot=spot,
                forward=forward,
                source="SYNTHETIC",
                expiration=expiration,
            )
        )
    return VolatilitySmile(
        underlying_symbol=underlying_symbol,
        valuation_timestamp=valuation,
        expiration=expiration,
        maturity_years=maturity_years,
        spot=spot,
        forward=forward,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        points=points,
        construction_metadata=SmileConstructionMetadata(
            minimum_points=1,
            accepted_points=len(points),
            source="SYNTHETIC",
        ),
    )


def generate_synthetic_surface(
    *,
    maturities: Sequence[float],
    log_moneyness: Sequence[float],
    base_volatility: float = 0.20,
    skew: float = 0.0,
    curvature: float = 0.0,
    term_slope: float = 0.0,
    spot: float = 100.0,
    risk_free_rate: float = 0.02,
    dividend_yield: float = 0.0,
    interpolation_config: SmileInterpolationConfig | None = None,
) -> VolatilitySurface:
    """Generate a deterministic multi-maturity surface from the smile formula."""

    smiles = [
        generate_synthetic_smile(
            maturity_years=maturity,
            log_moneyness=log_moneyness,
            base_volatility=base_volatility,
            skew=skew,
            curvature=curvature,
            term_slope=term_slope,
            spot=spot,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
        )
        for maturity in maturities
    ]
    return build_surface(smiles, interpolation_config=interpolation_config)
