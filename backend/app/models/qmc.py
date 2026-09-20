"""Typed contracts for Sobol, randomized QMC, and numerical convergence research."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.models.asian import AsianOptionContract
from backend.app.models.basket import BasketOptionContract
from backend.app.models.market import MarketEnvironment
from backend.app.models.multi_asset import MultiAssetMarketEnvironment
from backend.app.models.option import OptionContract


class QMCModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class SamplingMethod(StrEnum):
    PSEUDORANDOM = "PSEUDORANDOM"
    SOBOL = "SOBOL"
    SCRAMBLED_SOBOL = "SCRAMBLED_SOBOL"


class PowerOfTwoPolicy(StrEnum):
    REQUIRE_POWER_OF_TWO = "REQUIRE_POWER_OF_TWO"
    ROUND_UP = "ROUND_UP"
    ALLOW_ARBITRARY = "ALLOW_ARBITRARY"


class QMCNormalTransform(StrEnum):
    INVERSE_CDF = "INVERSE_CDF"


class QMCPathConstruction(StrEnum):
    STANDARD = "STANDARD"
    BROWNIAN_BRIDGE = "BROWNIAN_BRIDGE"
    PCA = "PCA"


class QMCConfig(QMCModel):
    method: SamplingMethod = SamplingMethod.PSEUDORANDOM
    dimension: int = Field(default=1, gt=0, le=21_201)
    num_samples: int = Field(default=4_096, gt=0, le=1_073_741_824)
    scramble: bool = False
    seed: int = Field(default=42, ge=0)
    num_randomizations: int = Field(default=1, gt=0)
    skip: int = Field(default=0, ge=0)
    power_of_two_policy: PowerOfTwoPolicy = PowerOfTwoPolicy.REQUIRE_POWER_OF_TWO
    transform: QMCNormalTransform = QMCNormalTransform.INVERSE_CDF
    path_construction: QMCPathConstruction = QMCPathConstruction.STANDARD
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_method(self) -> QMCConfig:
        power = self.num_samples > 0 and self.num_samples & (self.num_samples - 1) == 0
        if (
            self.method is not SamplingMethod.PSEUDORANDOM
            and self.power_of_two_policy is PowerOfTwoPolicy.REQUIRE_POWER_OF_TWO
            and not power
        ):
            raise ValueError("Sobol num_samples must be a power of two under REQUIRE_POWER_OF_TWO")
        if self.method is SamplingMethod.SOBOL:
            if self.scramble:
                raise ValueError("deterministic SOBOL requires scramble=false")
            if self.num_randomizations != 1:
                raise ValueError("deterministic SOBOL requires num_randomizations=1")
        elif self.method is SamplingMethod.SCRAMBLED_SOBOL:
            if not self.scramble:
                raise ValueError("SCRAMBLED_SOBOL requires scramble=true")
            if self.num_randomizations < 2:
                raise ValueError(
                    "SCRAMBLED_SOBOL needs at least two randomizations for uncertainty"
                )
        else:
            if self.scramble:
                raise ValueError("PSEUDORANDOM does not use Sobol scrambling")
            if self.num_randomizations != 1:
                raise ValueError("PSEUDORANDOM uses one path sample in this interface")
            if self.skip != 0:
                raise ValueError("PSEUDORANDOM does not support Sobol fast-forward offsets")
        return self

    @property
    def effective_num_samples(self) -> int:
        if (
            self.method is not SamplingMethod.PSEUDORANDOM
            and self.power_of_two_policy is PowerOfTwoPolicy.ROUND_UP
        ):
            return 1 << (self.num_samples - 1).bit_length()
        return self.num_samples

    @property
    def is_power_of_two(self) -> bool:
        count = self.effective_num_samples
        return count > 0 and count & (count - 1) == 0


class QMCReplicateDiagnostic(QMCModel):
    replicate_index: int = Field(ge=0)
    seed: int = Field(ge=0)
    estimate: float
    runtime_ms: float = Field(ge=0.0)


class NumericalMethodMetadata(QMCModel):
    financial_model: str
    sampling_method: SamplingMethod
    scramble: bool
    seed: int = Field(ge=0)
    dimension: int = Field(gt=0)
    requested_samples: int = Field(gt=0)
    effective_samples: int = Field(gt=0)
    replicate_count: int = Field(gt=0)
    path_construction: QMCPathConstruction
    path_observation_times: list[float] | None = None
    path_coordinate_order: list[int] | None = None
    pca_cumulative_variance_explained: list[float] | None = None
    skip: int = Field(ge=0)
    reproducibility_signature: str


class QMCPricingResult(QMCModel):
    product: str
    price: float
    standard_error: float | None = Field(default=None, ge=0.0)
    confidence_interval_low: float | None = None
    confidence_interval_high: float | None = None
    confidence_level: float = Field(gt=0.0, lt=1.0)
    requested_samples: int = Field(gt=0)
    effective_samples: int = Field(gt=0)
    sampling_method: SamplingMethod
    qmc_config: QMCConfig
    replicate_diagnostics: list[QMCReplicateDiagnostic]
    replicate_standard_deviation: float | None = Field(default=None, ge=0.0)
    replicate_minimum: float
    replicate_maximum: float
    runtime_ms: float = Field(ge=0.0)
    reference_price: float | None = Field(default=None, ge=0.0)
    reference_label: str | None = None
    absolute_error: float | None = Field(default=None, ge=0.0)
    relative_error: float | None = Field(default=None, ge=0.0)
    mean_underlying_value: float = Field(gt=0.0)
    probability_itm: float = Field(ge=0.0, le=1.0)
    control_variate: bool = False
    metadata: NumericalMethodMetadata


class SamplingDiagnosticsRequest(QMCModel):
    config: QMCConfig
    include_points: bool = False
    max_output_points: int = Field(default=16, gt=0)
    calculate_discrepancy: bool = True
    observation_times: list[float] | None = None

    @model_validator(mode="after")
    def validate_observation_times(self) -> SamplingDiagnosticsRequest:
        if self.observation_times is None:
            return self
        if len(self.observation_times) != self.config.dimension:
            raise ValueError("observation_times length must equal the configured dimension")
        if any(value <= 0.0 for value in self.observation_times) or any(
            right <= left
            for left, right in zip(self.observation_times, self.observation_times[1:], strict=False)
        ):
            raise ValueError("observation_times must be strictly increasing and positive")
        return self


class SamplingDiagnosticsResult(QMCModel):
    method: SamplingMethod
    dimension: int = Field(gt=0)
    requested_samples: int = Field(gt=0)
    effective_samples: int = Field(gt=0)
    power_of_two: bool
    scramble: bool
    seed: int = Field(ge=0)
    replicate_count: int = Field(gt=0)
    replicate_seeds: list[int]
    skip: int = Field(ge=0)
    path_construction: QMCPathConstruction
    path_observation_times: list[float]
    path_coordinate_order: list[int]
    pca_cumulative_variance_explained: list[float] | None = None
    minimum_uniform: float = Field(ge=0.0, lt=1.0)
    maximum_uniform: float = Field(ge=0.0, lt=1.0)
    mean_uniform: float = Field(ge=0.0, le=1.0)
    discrepancy: float | None = Field(default=None, ge=0.0)
    first_points: list[list[float]]
    normal_mean: float
    normal_variance: float = Field(ge=0.0)
    runtime_ms: float = Field(ge=0.0)


class QMCEuropeanRequest(QMCModel):
    contract: OptionContract
    market: MarketEnvironment
    qmc: QMCConfig = Field(default_factory=QMCConfig)


class QMCAsianRequest(QMCModel):
    contract: AsianOptionContract
    market: MarketEnvironment
    qmc: QMCConfig
    use_geometric_control: bool = False
    numerical_reference_price: float | None = Field(default=None, ge=0.0)
    numerical_reference_label: str = "caller-supplied numerical reference"


class QMCBasketRequest(QMCModel):
    contract: BasketOptionContract
    market: MultiAssetMarketEnvironment
    qmc: QMCConfig
    use_geometric_control: bool = False
    numerical_reference_price: float | None = Field(default=None, ge=0.0)
    numerical_reference_label: str = "caller-supplied numerical reference"


class QMCConvergenceRequest(QMCModel):
    contract: OptionContract
    market: MarketEnvironment
    minimum_power: int = Field(default=8, ge=1, le=30)
    maximum_power: int = Field(default=14, ge=1, le=30)
    root_seed: int = Field(default=42, ge=0)
    randomized_replicates: int = Field(default=8, ge=2)
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_powers(self) -> QMCConvergenceRequest:
        if self.maximum_power < self.minimum_power:
            raise ValueError("maximum_power must be at least minimum_power")
        return self


class QMCMethodPoint(QMCModel):
    method: SamplingMethod
    num_samples: int = Field(gt=0)
    estimate: float
    reference_price: float = Field(ge=0.0)
    absolute_error: float = Field(ge=0.0)
    relative_error: float = Field(ge=0.0)
    standard_error: float | None = Field(default=None, ge=0.0)
    runtime_ms: float = Field(ge=0.0)
    error_sqrt_runtime: float = Field(ge=0.0)
    metadata: NumericalMethodMetadata


class QMCMethodSlope(QMCModel):
    method: SamplingMethod
    error_slope: float | None = None
    error_points_used: int = Field(ge=0)
    standard_error_slope: float | None = None
    standard_error_points_used: int = Field(ge=0)


class QMCConvergenceResult(QMCModel):
    reference_price: float = Field(ge=0.0)
    reference_label: str
    points: list[QMCMethodPoint]
    slopes: list[QMCMethodSlope]


class QMCComparisonRequest(QMCModel):
    contract: OptionContract
    market: MarketEnvironment
    num_samples: int = Field(default=4_096, gt=0, le=1_073_741_824)
    root_seed: int = Field(default=42, ge=0)
    randomized_replicates: int = Field(default=8, ge=2)
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    power_of_two_policy: PowerOfTwoPolicy = PowerOfTwoPolicy.REQUIRE_POWER_OF_TWO

    @model_validator(mode="after")
    def validate_sample_policy(self) -> QMCComparisonRequest:
        if (
            self.power_of_two_policy is PowerOfTwoPolicy.REQUIRE_POWER_OF_TWO
            and self.num_samples & (self.num_samples - 1) != 0
        ):
            raise ValueError("comparison num_samples must be a power of two under REQUIRE policy")
        return self


class QMCComparisonResult(QMCModel):
    reference_price: float = Field(ge=0.0)
    reference_label: str
    points: list[QMCMethodPoint]
