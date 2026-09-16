"""Volatility smile/surface interpolation and synthetic research helpers."""

from backend.app.volatility_surface.core import (
    build_market_surface,
    build_smile,
    build_surface,
    build_surface_from_option_chain,
    calculate_forward,
    construct_market_smiles,
    implied_vol_to_total_variance,
    make_volatility_point,
    query_smile,
    query_surface,
    total_variance_to_implied_vol,
)
from backend.app.volatility_surface.synthetic import (
    generate_synthetic_smile,
    generate_synthetic_surface,
)

__all__ = [
    "build_market_surface",
    "build_smile",
    "build_surface",
    "build_surface_from_option_chain",
    "calculate_forward",
    "construct_market_smiles",
    "generate_synthetic_smile",
    "generate_synthetic_surface",
    "implied_vol_to_total_variance",
    "make_volatility_point",
    "query_smile",
    "query_surface",
    "total_variance_to_implied_vol",
]
