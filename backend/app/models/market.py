"""Constant market assumptions kept separate from option contracts."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

NonNegativeFloat = Annotated[float, Field(ge=0.0)]


class MarketEnvironment(BaseModel):
    """Constant risk-neutral inputs using continuously compounded rates."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    risk_free_rate: float
    volatility: NonNegativeFloat
    dividend_yield: float = 0.0
