"""Risk-neutral geometric Brownian motion terminal simulation."""

import math

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from backend.app.core.numerics import finite_array_exp, finite_exp, require_finite
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract
from backend.app.models.simulation import SimulationConfig, VarianceReduction
from backend.app.simulation.base import StochasticProcess


class GBMProcess(StochasticProcess):
    """Terminal-only risk-neutral GBM process with vectorized normal sampling."""

    def simulate_terminal(
        self,
        contract: OptionContract,
        market: MarketEnvironment,
        config: SimulationConfig,
        rng: Generator,
    ) -> NDArray[np.float64]:
        """Simulate terminal values, preserving exact requested path count.

        Antithetic output is interleaved as ``Z, -Z``. For odd counts, the last
        independently drawn ``Z`` is retained as a documented unpaired outcome.
        """

        count = config.num_simulations
        maturity = contract.maturity_years
        volatility = market.volatility

        if maturity == 0.0:
            return np.full(count, contract.spot, dtype=np.float64)

        drift = (market.risk_free_rate - market.dividend_yield - 0.5 * volatility**2) * maturity
        if volatility == 0.0:
            growth = finite_exp(drift, context="deterministic GBM growth factor")
            deterministic_terminal = contract.spot * growth
            if not math.isfinite(deterministic_terminal) or deterministic_terminal <= 0.0:
                from backend.app.core.exceptions import NumericalError

                raise NumericalError(
                    "deterministic GBM terminal price is outside positive finite range"
                )
            return np.full(count, deterministic_terminal, dtype=np.float64)

        if config.variance_reduction is VarianceReduction.ANTITHETIC:
            normals = self._antithetic_normals(count, rng)
        else:
            normals = np.asarray(rng.standard_normal(count), dtype=np.float64)

        log_returns = drift + volatility * math.sqrt(maturity) * normals
        terminal = contract.spot * finite_array_exp(log_returns, context="GBM terminal prices")
        require_finite(terminal, context="GBM terminal prices")
        return terminal

    @staticmethod
    def _antithetic_normals(count: int, rng: Generator) -> NDArray[np.float64]:
        pair_count, remainder = divmod(count, 2)
        primary = np.asarray(rng.standard_normal(pair_count + remainder), dtype=np.float64)
        normals = np.empty(count, dtype=np.float64)
        normals[0 : 2 * pair_count : 2] = primary[:pair_count]
        normals[1 : 2 * pair_count : 2] = -primary[:pair_count]
        if remainder:
            normals[-1] = primary[-1]
        return normals


def theoretical_terminal_mean(
    contract: OptionContract,
    market: MarketEnvironment,
) -> float:
    """Return E[S_T] under the risk-neutral GBM measure."""

    growth = finite_exp(
        (market.risk_free_rate - market.dividend_yield) * contract.maturity_years,
        context="GBM theoretical mean growth factor",
    )
    result = contract.spot * growth
    if not math.isfinite(result) or result <= 0.0:
        from backend.app.core.exceptions import NumericalError

        raise NumericalError("GBM theoretical mean overflowed floating-point range")
    return result
