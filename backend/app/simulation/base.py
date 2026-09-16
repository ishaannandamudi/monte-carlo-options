"""Interface implemented by terminal-value stochastic processes."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract
from backend.app.models.simulation import PathSimulationConfig, SimulationConfig


class StochasticProcess(ABC):
    """Generate model state independently of payoff and discounting logic."""

    @abstractmethod
    def simulate_terminal(
        self,
        contract: OptionContract,
        market: MarketEnvironment,
        config: SimulationConfig,
        rng: Generator,
    ) -> NDArray[np.float64]:
        """Return exactly ``config.num_simulations`` terminal spot observations."""


class PathStochasticProcess(ABC):
    """Separate full-path capability; terminal-only implementations stay valid."""

    @abstractmethod
    def simulate_paths(
        self,
        contract: OptionContract,
        market: MarketEnvironment,
        config: PathSimulationConfig,
        rng: Generator,
    ) -> NDArray[np.float64]:
        """Return (num_simulations, num_time_steps + 1), starting at spot."""
