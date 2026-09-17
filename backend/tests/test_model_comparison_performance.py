import pytest

from backend.app.analytics.model_comparison import compare_models, evaluate_market_fit
from backend.app.calibration.synthetic import (
    FELLER_SATISFIED_PARAMETERS,
    generate_synthetic_heston_dataset,
)
from backend.app.models.heston import HestonSimulationConfig
from backend.app.models.heston_calibration import (
    CalibrationSelectionPolicy,
    HestonCalibrationConfig,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.model_comparison import (
    MarketFitRequest,
    ModelComparisonConfig,
    ModelComparisonRequest,
    PricingModel,
)
from backend.app.models.option import OptionContract, OptionType
from backend.app.models.simulation import SimulationConfig


@pytest.mark.performance
def test_phase15_representative_multi_model_workload() -> None:
    result = compare_models(
        ModelComparisonRequest(
            contract=OptionContract(
                option_type=OptionType.CALL,
                spot=100.0,
                strike=100.0,
                maturity_years=1.0,
            ),
            market=MarketEnvironment(risk_free_rate=0.03, volatility=0.2),
            config=ModelComparisonConfig(
                selected_models=[
                    PricingModel.BLACK_SCHOLES,
                    PricingModel.MONTE_CARLO_GBM,
                    PricingModel.CRR_BINOMIAL,
                    PricingModel.HESTON_ANALYTICAL,
                    PricingModel.HESTON_MONTE_CARLO,
                ],
                runtime_repetitions=3,
            ),
            heston_parameters=FELLER_SATISFIED_PARAMETERS,
            gbm_monte_carlo=SimulationConfig(num_simulations=100_000),
            heston_monte_carlo=HestonSimulationConfig(num_simulations=30_000, num_time_steps=50),
        )
    )
    assert len(result.successful_models) == 5
    assert result.experiment.total_runtime_ms > 0.0
    assert result.experiment.total_runtime_ms < 10_000.0


@pytest.mark.performance
def test_phase15_heston_market_fit_batch_workload() -> None:
    dataset = generate_synthetic_heston_dataset(
        maturities=(0.25, 0.5, 1.0, 2.0),
        log_moneyness=tuple(index / 100.0 for index in range(-20, 21, 2)),
    )
    result = evaluate_market_fit(
        MarketFitRequest(
            dataset=dataset,
            selected_models=[PricingModel.HESTON_ANALYTICAL],
            heston_parameters=FELLER_SATISFIED_PARAMETERS,
            preprocessing=HestonCalibrationConfig(selection_policy=CalibrationSelectionPolicy.ALL),
        )
    )
    assert result.models[0].metrics.observation_count == len(dataset.observations)
    assert result.models[0].metrics.rmse == pytest.approx(0.0, abs=1e-10)
    assert result.models[0].runtime_ms < 10_000.0
