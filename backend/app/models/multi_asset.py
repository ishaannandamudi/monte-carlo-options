"""Typed constant-parameter multi-asset market inputs and diagnostics."""

import math
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.core.exceptions import MultiAssetPricingError
from backend.app.models.simulation import NonNegativeInt, RandomEngine
from backend.app.simulation.correlation import (
    CorrelationFactorizationMethod,
    prepare_correlation_matrix,
)

PositiveFloat = Annotated[float, Field(gt=0.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]


class MultiAssetModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class MultiAssetMarketEnvironment(MultiAssetModel):
    """Risk-neutral spots, carry, volatility, and effective Gaussian correlation."""

    spots: tuple[PositiveFloat, ...] = Field(min_length=1)
    risk_free_rate: float
    dividend_yields: tuple[float, ...]
    volatilities: tuple[NonNegativeFloat, ...]
    correlation_matrix: tuple[tuple[float, ...], ...]
    asset_ids: tuple[str, ...] | None = None
    correlation_repaired: bool = False
    correlation_adjustment_magnitude: NonNegativeFloat = 0.0
    original_min_eigenvalue: float = 0.0

    @model_validator(mode="before")
    @classmethod
    def prepare_correlation(cls, value: object) -> object:
        if isinstance(value, cls) or not isinstance(value, dict):
            return value
        data = dict(value)
        if "correlation_matrix" not in data:
            return data
        try:
            prepared = prepare_correlation_matrix(data["correlation_matrix"])
        except MultiAssetPricingError as exc:
            raise ValueError(str(exc)) from exc
        data["correlation_matrix"] = tuple(
            tuple(float(item) for item in row) for row in prepared.matrix
        )
        data["correlation_repaired"] = prepared.repaired
        data["correlation_adjustment_magnitude"] = prepared.adjustment_magnitude
        data["original_min_eigenvalue"] = prepared.original_min_eigenvalue
        return data

    @model_validator(mode="after")
    def validate_dimensions(self) -> Self:
        count = len(self.spots)
        if len(self.dividend_yields) != count or len(self.volatilities) != count:
            raise ValueError("spots, dividend_yields, and volatilities must have equal length")
        if len(self.correlation_matrix) != count or any(
            len(row) != count for row in self.correlation_matrix
        ):
            raise ValueError("correlation matrix dimension must equal asset count")
        if self.asset_ids is not None:
            if len(self.asset_ids) != count:
                raise ValueError("asset_ids length must equal asset count")
            if any(not identifier.strip() for identifier in self.asset_ids):
                raise ValueError("asset identifiers must be non-empty")
            if len(set(self.asset_ids)) != count:
                raise ValueError("asset identifiers must be unique")
        if not math.isfinite(self.risk_free_rate) or any(
            not math.isfinite(value) for value in self.dividend_yields
        ):
            raise ValueError("rates and dividend yields must be finite")
        return self

    @property
    def num_assets(self) -> int:
        return len(self.spots)


class CorrelationFactorizationDiagnostics(MultiAssetModel):
    dimension: int = Field(gt=0)
    method: CorrelationFactorizationMethod
    min_eigenvalue: float
    max_eigenvalue: float
    effective_rank: int = Field(gt=0)
    condition_number: NonNegativeFloat | None
    repaired: bool
    adjustment_magnitude: NonNegativeFloat
    symmetric: bool = True
    positive_semidefinite: bool = True


class CorrelationDiagnosticConfig(MultiAssetModel):
    num_simulations: int = Field(default=100_000, ge=2)
    seed: NonNegativeInt = 42
    random_engine: RandomEngine = RandomEngine.PCG64
    antithetic: bool = False


class EmpiricalCorrelationResult(MultiAssetModel):
    num_simulations: int = Field(gt=0)
    seed: int = Field(ge=0)
    target_matrix: list[list[float]]
    empirical_matrix: list[list[float]]
    absolute_error_matrix: list[list[NonNegativeFloat]]
    maximum_absolute_error: NonNegativeFloat
    frobenius_error: NonNegativeFloat
    factorization: CorrelationFactorizationDiagnostics
