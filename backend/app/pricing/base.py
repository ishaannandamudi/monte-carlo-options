"""Minimal pricing abstraction shared by Phase 1 engines."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from backend.app.core.exceptions import UnsupportedExerciseStyleError
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import ExerciseStyle, OptionContract

ConfigT = TypeVar("ConfigT")
ResultT = TypeVar("ResultT")


def require_european_exercise(contract: OptionContract, *, engine: str) -> None:
    """Reject American contracts at European-only engine boundaries."""

    if contract.exercise_style is not ExerciseStyle.EUROPEAN:
        raise UnsupportedExerciseStyleError(f"{engine} supports EUROPEAN exercise only")


class Pricer(ABC, Generic[ConfigT, ResultT]):
    """Price one contract in one market environment."""

    @abstractmethod
    def price(
        self,
        contract: OptionContract,
        market: MarketEnvironment,
        config: ConfigT,
    ) -> ResultT:
        """Return an engine-specific typed pricing result."""
