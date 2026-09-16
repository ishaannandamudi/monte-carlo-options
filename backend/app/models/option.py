"""Vanilla-option contract definitions."""

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class OptionType(StrEnum):
    """Supported vanilla payoff direction."""

    CALL = "CALL"
    PUT = "PUT"


class ExerciseStyle(StrEnum):
    """Supported vanilla exercise rules."""

    EUROPEAN = "EUROPEAN"
    AMERICAN = "AMERICAN"


PositiveFloat = Annotated[float, Field(gt=0.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]


class OptionContract(BaseModel):
    """Continuous option-model inputs, not accounting ledger values."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    option_type: OptionType
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN
    spot: PositiveFloat
    strike: PositiveFloat
    maturity_years: NonNegativeFloat
