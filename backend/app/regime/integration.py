"""Bounded integrations with model-fit and scenario research outputs."""

from __future__ import annotations

import math

import numpy as np

from backend.app.models.model_comparison import ModelMarketFit
from backend.app.models.portfolio import PortfolioStressResult
from backend.app.models.regime import RegimeAwareStressContext, RegimeGroupedMetric


def group_market_fit_by_regime(
    model_fit: ModelMarketFit, state_assignments: list[int]
) -> list[RegimeGroupedMetric]:
    """Group existing Phase 15 price residuals by externally inferred HMM state."""

    if any(state < 0 for state in state_assignments):
        raise ValueError("state assignments must be nonnegative")
    if any(
        point.price_residual is not None and point.observation_index >= len(state_assignments)
        for point in model_fit.residuals
    ):
        raise ValueError("state assignments must cover every grouped market-fit observation")
    grouped: list[RegimeGroupedMetric] = []
    for state in sorted(set(state_assignments)):
        residuals = [
            point.price_residual
            for point in model_fit.residuals
            if point.observation_index < len(state_assignments)
            and state_assignments[point.observation_index] == state
            and point.price_residual is not None
        ]
        if residuals:
            values = np.asarray(residuals, dtype=np.float64)
            grouped.append(
                RegimeGroupedMetric(
                    state=state,
                    observation_count=values.size,
                    rmse=math.sqrt(float(np.mean(values * values))),
                    mae=float(np.mean(np.abs(values))),
                    bias=float(np.mean(values)),
                )
            )
        else:
            grouped.append(RegimeGroupedMetric(state=state, observation_count=0))
    return grouped


def contextualize_stress_results(
    stress_result: PortfolioStressResult, state_probabilities: list[float]
) -> RegimeAwareStressContext:
    """Attach regime context without changing any deterministic Phase 16 stress value."""

    if not state_probabilities or any(value < 0.0 for value in state_probabilities):
        raise ValueError("state probabilities must be nonnegative and non-empty")
    if not math.isclose(sum(state_probabilities), 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("state probabilities must sum to one")
    return RegimeAwareStressContext(
        state_probabilities=state_probabilities,
        stress_result=stress_result.model_dump(mode="json"),
    )
