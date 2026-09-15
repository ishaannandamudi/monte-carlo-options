"""Explicit simulation behavior and random-engine configuration."""

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class VarianceReduction(StrEnum):
    """Phase 1 Monte Carlo variance-reduction choices."""

    NONE = "NONE"
    ANTITHETIC = "ANTITHETIC"


class RandomEngine(StrEnum):
    """Supported pseudo-random bit generator."""

    PCG64 = "PCG64"


PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
ConfidenceLevel = Annotated[float, Field(gt=0.0, lt=1.0)]


class SimulationConfig(BaseModel):
    """All user-controlled Monte Carlo execution and uncertainty parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    num_simulations: PositiveInt = 100_000
    seed: NonNegativeInt = 42
    variance_reduction: VarianceReduction = VarianceReduction.ANTITHETIC
    random_engine: RandomEngine = RandomEngine.PCG64
    confidence_level: ConfidenceLevel = 0.95
    batch_size: PositiveInt | None = None
