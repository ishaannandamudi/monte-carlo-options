"""Full-revaluation mapping for volatility, rates, dividends, Heston, and correlation."""

from datetime import UTC, datetime, timedelta

import pytest

from backend.app.core.exceptions import PortfolioRiskError
from backend.app.models.merton import MertonJumpParameters
from backend.app.models.portfolio import (
    BasketPosition,
    Portfolio,
    PortfolioPricingModel,
)
from backend.app.models.risk import (
    HistoricalRiskDataset,
    HistoricalRiskFactorObservation,
    HistoricalRiskRequest,
    HistoricalRiskScenario,
    PortfolioRiskConfig,
    RiskFactorMapping,
    RiskFactorType,
    RiskMethod,
    RiskValuationMode,
)
from backend.app.risk.engine import calculate_historical_risk
from backend.tests.portfolio_test_helpers import mixed_portfolio, vanilla


def _dataset(factor_id: str, values: list[float]) -> HistoricalRiskDataset:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    return HistoricalRiskDataset(
        scenarios=[
            HistoricalRiskScenario(
                timestamp=start + timedelta(days=index),
                observations=[
                    HistoricalRiskFactorObservation(
                        timestamp=start + timedelta(days=index),
                        factor_id=factor_id,
                        value=value,
                    )
                ],
            )
            for index, value in enumerate(values)
        ]
    )


def _full_request(
    portfolio: Portfolio,
    mapping: RiskFactorMapping,
    values: list[float],
) -> HistoricalRiskRequest:
    return HistoricalRiskRequest(
        portfolio=portfolio,
        factor_mappings=[mapping],
        dataset=_dataset(mapping.factor_id, values),
        config=PortfolioRiskConfig(
            method=RiskMethod.HISTORICAL,
            confidence_level=0.8,
            valuation_mode=RiskValuationMode.FULL_REVALUATION,
        ),
    )


@pytest.mark.parametrize(
    ("factor_type", "target"),
    [
        (RiskFactorType.VOLATILITY_CHANGE, {"underlying_id": "SPY"}),
        (RiskFactorType.RATE_CHANGE, {}),
        (RiskFactorType.DIVIDEND_YIELD_CHANGE, {"underlying_id": "SPY"}),
    ],
)
def test_scalar_market_factor_full_revaluation(
    factor_type: RiskFactorType, target: dict[str, str]
) -> None:
    mapping = RiskFactorMapping(factor_id="factor", factor_type=factor_type, **target)
    result = calculate_historical_risk(
        _full_request(Portfolio(positions=[vanilla()]), mapping, [-0.01, 0.01])
    )
    assert result.distribution_summary.minimum_pnl != result.distribution_summary.maximum_pnl


def test_unavailable_dividend_greek_requires_explicit_exposure_in_approximation_mode() -> None:
    mapping = RiskFactorMapping(
        factor_id="dividend",
        factor_type=RiskFactorType.DIVIDEND_YIELD_CHANGE,
        underlying_id="SPY",
    )
    request = _full_request(Portfolio(positions=[vanilla()]), mapping, [-0.01, 0.01])
    request = request.model_copy(
        update={
            "config": request.config.model_copy(update={"valuation_mode": RiskValuationMode.DELTA})
        }
    )
    with pytest.raises(PortfolioRiskError, match="explicit dividend-yield exposure"):
        calculate_historical_risk(request)


def test_explicit_heston_parameter_history_is_replayed() -> None:
    position = vanilla(model=PortfolioPricingModel.HESTON_ANALYTICAL)
    mapping = RiskFactorMapping(
        factor_id="v0", factor_type=RiskFactorType.HESTON_V0_CHANGE, position_id="vanilla"
    )
    result = calculate_historical_risk(
        _full_request(Portfolio(positions=[position]), mapping, [-0.005, 0.005])
    )
    assert result.distribution_summary.minimum_pnl != result.distribution_summary.maximum_pnl


def test_explicit_merton_parameter_history_is_replayed() -> None:
    position = vanilla().model_copy(
        update={
            "model": PortfolioPricingModel.MERTON_ANALYTICAL,
            "merton_parameters": MertonJumpParameters(
                jump_intensity=0.7, jump_mean=-0.12, jump_volatility=0.25
            ),
        }
    )
    mapping = RiskFactorMapping(
        factor_id="jump_intensity",
        factor_type=RiskFactorType.MERTON_JUMP_INTENSITY_CHANGE,
        position_id="vanilla",
    )
    result = calculate_historical_risk(
        _full_request(Portfolio(positions=[position]), mapping, [-0.1, 0.1])
    )
    assert result.distribution_summary.minimum_pnl != result.distribution_summary.maximum_pnl


def test_explicit_basket_correlation_history_is_replayed_and_psd_checked() -> None:
    basket = mixed_portfolio().positions[-1]
    assert isinstance(basket, BasketPosition)
    mapping = RiskFactorMapping(
        factor_id="corr",
        factor_type=RiskFactorType.CORRELATION_CHANGE,
        position_id="basket",
    )
    result = calculate_historical_risk(
        _full_request(Portfolio(positions=[basket]), mapping, [-0.05, 0.05])
    )
    assert result.value_at_risk != 0.0
    with pytest.raises(PortfolioRiskError):
        calculate_historical_risk(
            _full_request(Portfolio(positions=[basket]), mapping, [-1.5, 1.5])
        )


def test_custom_factor_requires_greek_mode_for_nonzero_shock() -> None:
    mapping = RiskFactorMapping(
        factor_id="custom",
        factor_type=RiskFactorType.CUSTOM_GAUSSIAN,
        explicit_position_exposures={"vanilla": 1.0},
    )
    with pytest.raises(PortfolioRiskError, match="cannot be used for full revaluation"):
        calculate_historical_risk(
            _full_request(Portfolio(positions=[vanilla()]), mapping, [-0.01, 0.01])
        )


def test_time_passage_is_explicitly_opt_in() -> None:
    mapping = RiskFactorMapping(
        factor_id="spot", factor_type=RiskFactorType.SPOT_RETURN, underlying_id="SPY"
    )
    request = _full_request(Portfolio(positions=[vanilla()]), mapping, [0.0, 0.0])
    without = calculate_historical_risk(request)
    with_time = calculate_historical_risk(
        request.model_copy(
            update={"config": request.config.model_copy(update={"include_time_passage": True})}
        )
    )
    assert without.value_at_risk == 0.0
    assert with_time.value_at_risk != 0.0


def test_worst_scenario_position_contributions_use_position_ids() -> None:
    mapping = RiskFactorMapping(
        factor_id="spot", factor_type=RiskFactorType.SPOT_RETURN, underlying_id="SPY"
    )
    result = calculate_historical_risk(
        _full_request(Portfolio(positions=[vanilla("known-id")]), mapping, [-0.1, 0.1])
    )
    assert result.worst_scenarios[0].position_contributions[0].position_id == "known-id"


def test_large_full_revaluation_shock_reports_approximation_error_diagnostic() -> None:
    mapping = RiskFactorMapping(
        factor_id="spot", factor_type=RiskFactorType.SPOT_RETURN, underlying_id="SPY"
    )
    result = calculate_historical_risk(
        _full_request(Portfolio(positions=[vanilla()]), mapping, [-0.4, 0.4])
    )
    assert any("maximum local Greek approximation" in warning for warning in result.warnings)
