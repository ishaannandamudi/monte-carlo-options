"""Opt-in CPU checks for representative bounded option chains."""

from datetime import UTC, datetime
from time import perf_counter

import pytest

from backend.app.analytics.options_analyzer import analyze_american_options
from backend.app.core.config import Settings
from backend.app.models.options_analyzer import OptionsAnalysisRequest
from backend.tests.test_options_analyzer import _provider, _snapshot


@pytest.mark.performance
@pytest.mark.parametrize("contract_count", [20, 50, 100])
def test_representative_chain_analysis_runtime(contract_count: int) -> None:
    strikes_per_side = contract_count // 2
    strikes = tuple(
        75.0 + 50.0 * index / (strikes_per_side - 1) for index in range(strikes_per_side)
    )
    snapshot = _snapshot(strikes=strikes, now=datetime.now(UTC))
    started = perf_counter()
    result = analyze_american_options(
        OptionsAnalysisRequest(
            symbol="XYZ",
            max_contracts=contract_count,
            strike_window_pct=0.50,
            tree_steps=100,
        ),
        provider=_provider(snapshot),
        settings=Settings(max_analysis_contracts=100),
    )
    wall_ms = (perf_counter() - started) * 1_000.0
    assert result.summary.contracts_analyzed == contract_count
    assert result.timings.total_ms <= wall_ms * 1.05
    assert result.timings.implied_volatility_ms >= 0.0
    assert result.timings.smile_construction_ms >= 0.0
    assert result.timings.pricing_and_greeks_ms >= 0.0
    assert wall_ms < 30_000.0
