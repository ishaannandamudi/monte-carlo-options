import pytest

from backend.app.analytics.market_data import enrich_option_chain
from backend.app.calibration.datasets import (
    calibration_dataset_from_chain,
    calibration_dataset_from_surface,
)
from backend.app.models.market_data import QuoteQualityConfig
from backend.app.models.option import OptionType
from backend.app.volatility_surface.synthetic import generate_synthetic_surface
from backend.tests.test_market_data_analytics import (
    VALUATION,
    snapshot,
    theoretical_quote,
)


def test_enriched_chain_conversion_preserves_market_values_and_quality() -> None:
    chain = enrich_option_chain(
        snapshot(
            [
                theoretical_quote(OptionType.PUT, 90.0, 0.23),
                theoretical_quote(OptionType.CALL, 100.0, 0.20),
                theoretical_quote(OptionType.CALL, 110.0, 0.22),
            ]
        ),
        valuation_time=VALUATION,
        risk_free_rate=0.03,
        dividend_yield=0.01,
        quality_config=QuoteQualityConfig(),
    )
    dataset = calibration_dataset_from_chain(chain)
    assert dataset.spot == pytest.approx(100.0)
    assert dataset.risk_free_rate == 0.03
    assert dataset.underlying_symbol == "XYZ"
    assert len(dataset.observations) == 3
    assert all(item.observed_price is not None for item in dataset.observations)
    assert all(item.observed_implied_volatility is not None for item in dataset.observations)
    assert dataset.metadata["input"] == "PHASE5_ENRICHED_CHAIN"


def test_phase6_surface_conversion_reconstructs_missing_prices() -> None:
    surface = generate_synthetic_surface(
        maturities=(0.5, 1.0),
        log_moneyness=(-0.1, 0.0, 0.1),
        base_volatility=0.22,
        skew=-0.1,
    )
    assert all(point.market_price is None for smile in surface.smiles for point in smile.points)
    dataset = calibration_dataset_from_surface(surface)
    assert len(dataset.observations) == 6
    assert all(item.observed_price is not None for item in dataset.observations)
    assert [item.observed_implied_volatility for item in dataset.observations] == pytest.approx(
        [point.implied_volatility for smile in surface.smiles for point in smile.points]
    )
    assert dataset.metadata["input"] == "PHASE6_VOLATILITY_SURFACE"
