"""Explicit simulation behavior and random-engine configuration."""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class VarianceReduction(StrEnum):
    """Supported Monte Carlo variance-reduction choices."""

    NONE = "NONE"
    ANTITHETIC = "ANTITHETIC"
    CONTROL_VARIATE = "CONTROL_VARIATE"
    ANTITHETIC_CONTROL_VARIATE = "ANTITHETIC_CONTROL_VARIATE"

    @property
    def uses_antithetic(self) -> bool:
        """Whether terminal normals must be generated as antithetic pairs."""

        return self in {self.ANTITHETIC, self.ANTITHETIC_CONTROL_VARIATE}

    @property
    def uses_control_variate(self) -> bool:
        """Whether discounted terminal stock controls the payoff estimator."""

        return self in {self.CONTROL_VARIATE, self.ANTITHETIC_CONTROL_VARIATE}


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


class PathSimulationConfig(BaseModel):
    """Exact GBM path sampling; adjacent rows are antithetic partners when selected."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    num_simulations: PositiveInt = 25_000
    num_time_steps: PositiveInt = 50
    seed: NonNegativeInt = 42
    variance_reduction: Literal[VarianceReduction.NONE, VarianceReduction.ANTITHETIC] = (
        VarianceReduction.ANTITHETIC
    )
    random_engine: RandomEngine = RandomEngine.PCG64
    confidence_level: ConfidenceLevel = 0.95
