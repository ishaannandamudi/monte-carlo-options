"""Typed Greek names, estimator methods, and finite-difference configuration."""

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class Greek(StrEnum):
    """Canonical raw-unit option sensitivities."""

    DELTA = "DELTA"
    GAMMA = "GAMMA"
    VEGA = "VEGA"
    THETA = "THETA"
    RHO = "RHO"
    DIVIDEND_RHO = "DIVIDEND_RHO"


class GreekMethod(StrEnum):
    """Supported analytical and stochastic Greek estimators."""

    ANALYTICAL = "ANALYTICAL"
    FINITE_DIFFERENCE = "FINITE_DIFFERENCE"
    PATHWISE = "PATHWISE"


class DifferenceScheme(StrEnum):
    """Finite-difference stencil used for a reported estimate."""

    CENTRAL = "CENTRAL"
    FORWARD = "FORWARD"


class SensitivityParameter(StrEnum):
    """One-dimensional Black-Scholes sensitivity-grid axes."""

    SPOT = "SPOT"
    VOLATILITY = "VOLATILITY"
    MATURITY = "MATURITY"
    RISK_FREE_RATE = "RISK_FREE_RATE"
    DIVIDEND_YIELD = "DIVIDEND_YIELD"


PositiveFloat = Annotated[float, Field(gt=0.0)]
RelativeSpotBump = Annotated[float, Field(gt=0.0, lt=1.0)]


class GreekBumpConfig(BaseModel):
    """Finite-difference increments in canonical model units."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    spot_relative_bump: RelativeSpotBump = 1e-3
    volatility_absolute_bump: PositiveFloat = 1e-4
    rate_absolute_bump: PositiveFloat = 1e-4
    dividend_yield_absolute_bump: PositiveFloat = 1e-4
    maturity_absolute_bump: PositiveFloat = 1.0 / 3650.0
