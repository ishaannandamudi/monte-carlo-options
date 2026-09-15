"""Minimal pricing abstraction shared by Phase 1 engines."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract

ConfigT = TypeVar("ConfigT")
ResultT = TypeVar("ResultT")


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
