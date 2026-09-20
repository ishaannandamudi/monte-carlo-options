from backend.app.models.model_comparison import MarketFitPoint, ModelMarketFit
from backend.app.models.portfolio import PortfolioStressResult
from backend.app.regime.integration import (
    contextualize_stress_results,
    group_market_fit_by_regime,
)


def test_phase15_market_fit_residuals_group_by_inferred_state() -> None:
    points = [
        MarketFitPoint.model_construct(observation_index=0, price_residual=1.0),
        MarketFitPoint.model_construct(observation_index=1, price_residual=-1.0),
        MarketFitPoint.model_construct(observation_index=2, price_residual=4.0),
    ]
    fit = ModelMarketFit.model_construct(residuals=points)
    groups = group_market_fit_by_regime(fit, [0, 0, 1])
    assert groups[0].observation_count == 2
    assert groups[0].bias == 0.0
    assert groups[0].mae == 1.0
    assert groups[1].rmse == 4.0


def test_regime_context_does_not_change_phase16_stress_output() -> None:
    stress = PortfolioStressResult(portfolio_id="p", scenarios=[], runtime_ms=1.5)
    result = contextualize_stress_results(stress, [0.7, 0.3])
    assert result.stress_result == stress.model_dump(mode="json")
    assert result.state_probabilities == [0.7, 0.3]
