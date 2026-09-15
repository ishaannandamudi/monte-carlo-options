"""Typed inputs and policies for European implied-volatility inversion."""

from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.models.option import OptionContract


class ImpliedVolatilityMethod(StrEnum):
    """Supported one-dimensional Black-Scholes root solvers."""

    BRENT = "BRENT"
    BISECTION = "BISECTION"
    NEWTON = "NEWTON"
    AUTO = "AUTO"


class ImpliedVolatilityStatus(StrEnum):
    """Serializable outcome categories for solves and partial-failure analytics."""

    CONVERGED = "CONVERGED"
    NON_IDENTIFIABLE = "NON_IDENTIFIABLE"
    ARBITRAGE_VIOLATION = "ARBITRAGE_VIOLATION"
    NO_FINITE_ROOT = "NO_FINITE_ROOT"
    NON_CONVERGENCE = "NON_CONVERGENCE"
    NUMERICAL_FAILURE = "NUMERICAL_FAILURE"


NonNegativeFloat = Annotated[float, Field(ge=0.0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]
PositiveInt = Annotated[int, Field(gt=0)]


class ImpliedVolatilityMarket(BaseModel):
    """Known continuously compounded rates when volatility is the unknown."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    risk_free_rate: float
    dividend_yield: float = 0.0


class ImpliedVolatilityObservation(BaseModel):
    """A European option contract and its observed price under known rates."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    contract: OptionContract
    market: ImpliedVolatilityMarket
    observed_price: NonNegativeFloat


class ImpliedVolatilityConfig(BaseModel):
    """Explicit solver choice, bracket, tolerances, and iteration protection."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    method: ImpliedVolatilityMethod = ImpliedVolatilityMethod.AUTO
    initial_volatility: NonNegativeFloat = 0.20
    min_volatility: NonNegativeFloat = 0.0
    max_volatility: PositiveFloat = 5.0
    price_tolerance: PositiveFloat = 1e-10
    volatility_tolerance: PositiveFloat = 1e-10
    vega_tolerance: PositiveFloat = 1e-12
    max_iterations: PositiveInt = 100

    @model_validator(mode="after")
    def validate_volatility_range(self) -> Self:
        """Keep the bracket ordered and the Newton starting point inside it."""

        if self.max_volatility <= self.min_volatility:
            raise ValueError("max_volatility must be greater than min_volatility")
        if not self.min_volatility <= self.initial_volatility <= self.max_volatility:
            raise ValueError("initial_volatility must lie inside the volatility bounds")
        return self
