"""Reusable deterministic observation schedules for path-dependent contracts."""

import math
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

NonNegativeTime = Annotated[float, Field(ge=0.0)]
PositiveInt = Annotated[int, Field(gt=0)]


class MonitoringSchedule(BaseModel):
    """Either regular future observations or explicit year-fraction times.

    ``monitoring_count`` creates equally spaced observations in ``(0, T]``.
    Explicit times may contain zero only when the enclosing contract elects to
    include the initial spot. Explicit input is normalized into sorted order.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    monitoring_count: PositiveInt | None = None
    monitoring_times: tuple[NonNegativeTime, ...] | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_choice(cls, value: object) -> object:
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            return value
        data = dict(value)
        count_supplied = data.get("monitoring_count") is not None
        times_supplied = data.get("monitoring_times") is not None
        if count_supplied and times_supplied:
            raise ValueError("choose monitoring_count or monitoring_times, not both")
        if not count_supplied and not times_supplied:
            data["monitoring_count"] = 12
        if times_supplied:
            raw_times = tuple(float(item) for item in data["monitoring_times"])
            ordered = tuple(sorted(raw_times))
            if len(set(ordered)) != len(ordered):
                raise ValueError("monitoring_times must be unique")
            if not ordered:
                raise ValueError("monitoring_times must not be empty")
            data["monitoring_times"] = ordered
        return data

    @model_validator(mode="after")
    def validate_choice(self) -> Self:
        if (self.monitoring_count is None) == (self.monitoring_times is None):
            raise ValueError("choose exactly one monitoring schedule representation")
        return self

    @classmethod
    def regular(cls, monitoring_count: int) -> "MonitoringSchedule":
        return cls(monitoring_count=monitoring_count)

    @classmethod
    def explicit(cls, monitoring_times: tuple[float, ...] | list[float]) -> "MonitoringSchedule":
        return cls(monitoring_times=tuple(monitoring_times))

    def validate_for_contract(self, maturity_years: float, *, include_initial_spot: bool) -> None:
        """Validate maturity-dependent rules without allocating a regular schedule."""

        if not math.isfinite(maturity_years) or maturity_years < 0.0:
            raise ValueError("maturity_years must be finite and non-negative")
        if maturity_years == 0.0:
            if self.monitoring_times is not None and any(
                time != 0.0 for time in self.monitoring_times
            ):
                raise ValueError("zero-maturity monitoring times must equal zero")
            return

        if self.monitoring_count is not None:
            if maturity_years / self.monitoring_count == 0.0:
                raise ValueError("regular monitoring interval underflowed floating-point range")
            return
        assert self.monitoring_times is not None
        if self.monitoring_times[-1] > maturity_years:
            raise ValueError("monitoring time cannot exceed maturity")
        has_zero = self.monitoring_times[0] == 0.0
        if has_zero and not include_initial_spot:
            raise ValueError("explicit time zero requires include_initial_spot_in_average=true")
        if not include_initial_spot and not any(time > 0.0 for time in self.monitoring_times):
            raise ValueError("averaging schedule must contain at least one observation")

    def resolve(self, maturity_years: float, *, include_initial_spot: bool) -> tuple[float, ...]:
        """Return the exact averaging times, including zero only by explicit policy."""

        self.validate_for_contract(maturity_years, include_initial_spot=include_initial_spot)
        if maturity_years == 0.0:
            return (0.0,)
        if self.monitoring_count is not None:
            count = self.monitoring_count
            future = tuple(maturity_years * index / count for index in range(1, count + 1))
        else:
            assert self.monitoring_times is not None
            future = tuple(time for time in self.monitoring_times if time > 0.0)

        resolved = ((0.0,) if include_initial_spot else ()) + future
        if not resolved:
            raise ValueError("averaging schedule must contain at least one observation")
        return resolved
