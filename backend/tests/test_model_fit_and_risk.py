import pytest

from backend.app.analytics.heston_calibration import analyze_heston_calibration_stability
from backend.app.analytics.model_comparison import (
    build_model_risk_report,
    compare_models_out_of_sample,
    compare_surface,
    evaluate_market_fit,
    heston_calibration_ensemble,
    propagate_heston_parameter_uncertainty,
)
from backend.app.analytics.volatility_surface import smile_local_metrics
from backend.app.calibration.synthetic import (
    FELLER_SATISFIED_PARAMETERS,
    generate_synthetic_heston_dataset,
)
from backend.app.models.heston import HestonParameters
from backend.app.models.heston_calibration import (
    CalibrationHoldoutPolicy,
    CalibrationSelectionPolicy,
    HestonCalibrationConfig,
    HestonCalibrationHoldoutConfig,
    HestonCalibrationTarget,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.model_comparison import (
    MarketFitRequest,
    ModelComparisonConfig,
    ModelComparisonRequest,
    ModelRiskRequest,
    PricingModel,
    SurfaceComparisonRequest,
)
from backend.app.models.option import OptionContract, OptionType
from backend.app.volatility_surface.synthetic import generate_synthetic_surface


def dataset_request() -> MarketFitRequest:
    return MarketFitRequest(
        dataset=generate_synthetic_heston_dataset(
            maturities=(0.5, 1.0), log_moneyness=(-0.1, 0.0, 0.1)
        ),
        selected_models=[PricingModel.BLACK_SCHOLES, PricingModel.HESTON_ANALYTICAL],
        heston_parameters=FELLER_SATISFIED_PARAMETERS,
        use_observed_iv_for_black_scholes=True,
        preprocessing=HestonCalibrationConfig(
            target=HestonCalibrationTarget.PRICE,
            selection_policy=CalibrationSelectionPolicy.ALL,
            identifiability_diagnostics=False,
        ),
    )


def test_synthetic_market_fit_metrics_bid_ask_and_grouping() -> None:
    request = dataset_request().model_copy(
        update={
            "observation_labels": {
                0: "CALM",
                1: "CALM",
                2: "CALM",
                3: "STRESSED",
                4: "STRESSED",
                5: "STRESSED",
            }
        }
    )
    result = evaluate_market_fit(request)
    assert result.observations_used == 6
    assert result.observations_excluded == 0
    by_model = {item.model: item for item in result.models}
    for model in (PricingModel.BLACK_SCHOLES, PricingModel.HESTON_ANALYTICAL):
        fit = by_model[model]
        assert fit.metrics.rmse == pytest.approx(0.0, abs=5e-11)
        assert fit.metrics.mae == pytest.approx(0.0, abs=5e-11)
        assert fit.metrics.bias == pytest.approx(0.0, abs=5e-11)
        assert fit.metrics.weighted_rmse == pytest.approx(0.0, abs=5e-11)
        assert fit.metrics.fraction_inside_bid_ask == 1.0
        assert fit.metrics.average_distance_outside_spread == 0.0
        assert fit.metrics.iv_rmse == pytest.approx(0.0, abs=1e-10)
        assert len(fit.per_expiry) == 2
        assert fit.per_moneyness
        assert [item.label for item in fit.per_metadata_group] == ["CALM", "STRESSED"]
        assert all(item.metrics.observation_count == 3 for item in fit.per_metadata_group)
        assert all(point.market_atm_iv is not None for point in fit.smile_fit)
        assert all(point.market_skew is not None for point in fit.smile_fit)
        assert all(point.market_curvature is not None for point in fit.smile_fit)


def test_market_fit_uses_floored_relative_scale_and_isolates_incompatible_model() -> None:
    request = dataset_request().model_copy(
        update={"selected_models": [PricingModel.BLACK_SCHOLES, PricingModel.LONGSTAFF_SCHWARTZ]}
    )
    result = evaluate_market_fit(request)
    assert result.models[0].metrics.observation_count == 6
    assert result.models[1].status.value == "EXCLUDED"
    assert result.models[1].metrics.observation_count == 0


def test_structured_holdout_keeps_training_and_validation_metrics_separate() -> None:
    result = compare_models_out_of_sample(
        dataset_request(),
        HestonCalibrationHoldoutConfig(
            policy=CalibrationHoldoutPolicy.EVERY_NTH_STRIKE,
            every_nth_strike=3,
        ),
    )
    assert result.training_observations == 4
    assert result.validation_observations == 2
    assert len(result.models) == 2
    assert result.models[0].training_metrics.observation_count == 4
    assert result.models[0].validation_metrics.observation_count == 2


def test_calibration_ensemble_and_local_covariance_propagate_to_value() -> None:
    contract = OptionContract(
        option_type=OptionType.CALL, spot=100.0, strike=100.0, maturity_years=1.0
    )
    market = MarketEnvironment(risk_free_rate=0.02, volatility=0.2, dividend_yield=0.01)
    parameter_sets = [
        FELLER_SATISFIED_PARAMETERS,
        FELLER_SATISFIED_PARAMETERS.model_copy(update={"rho": -0.6}),
        FELLER_SATISFIED_PARAMETERS.model_copy(update={"sigma_v": 0.35}),
    ]
    ensemble = heston_calibration_ensemble(contract, market, parameter_sets)
    assert len(ensemble.members) == 3
    assert ensemble.price_dispersion.range is not None
    assert ensemble.price_dispersion.range > 0.0
    assert ensemble.implied_volatility_dispersion.count == 3
    covariance = [[0.0] * 5 for _ in range(5)]
    for index in range(5):
        covariance[index][index] = 1e-6
    uncertainty = propagate_heston_parameter_uncertainty(
        contract, market, FELLER_SATISFIED_PARAMETERS, covariance
    )
    assert uncertainty.approximate_price_standard_deviation > 0.0
    assert len(uncertainty.price_gradient) == 5
    assert "not a confidence interval" in uncertainty.interpretation


def test_model_risk_report_keeps_uncertainty_categories_separate() -> None:
    contract = OptionContract(
        option_type=OptionType.CALL, spot=100.0, strike=100.0, maturity_years=1.0
    )
    market = MarketEnvironment(risk_free_rate=0.02, volatility=0.2, dividend_yield=0.01)
    comparison = ModelComparisonRequest(
        contract=contract,
        market=market,
        config=ModelComparisonConfig(
            selected_models=[PricingModel.BLACK_SCHOLES, PricingModel.HESTON_ANALYTICAL]
        ),
        heston_parameters=FELLER_SATISFIED_PARAMETERS,
    )
    report = build_model_risk_report(
        ModelRiskRequest(
            comparison=comparison,
            market_fit=dataset_request(),
            calibration_ensemble=[
                FELLER_SATISFIED_PARAMETERS,
                FELLER_SATISFIED_PARAMETERS.model_copy(update={"rho": -0.65}),
            ],
        )
    )
    assert report.market_fit is not None
    assert report.calibration_stability is not None
    assert report.uncertainty.calibration_dispersion is not None
    assert report.uncertainty.model_dispersion.interpretation.startswith("Descriptive")
    assert "not combined" in report.uncertainty.note
    assert len(report.complexity_vs_fit) == 2


def test_phase14_stability_output_integrates_with_model_risk() -> None:
    dataset = generate_synthetic_heston_dataset(maturities=(0.5,), log_moneyness=(-0.1, 0.0, 0.1))
    config = HestonCalibrationConfig(
        target=HestonCalibrationTarget.PRICE,
        selection_policy=CalibrationSelectionPolicy.ALL,
        initial_parameters=HestonParameters(
            v0=0.041, kappa=1.9, theta=0.041, sigma_v=0.31, rho=-0.68
        ),
        max_iterations=8,
        identifiability_diagnostics=False,
    )
    stability = analyze_heston_calibration_stability(dataset, config, 1)
    contract = OptionContract(
        option_type=OptionType.CALL, spot=100.0, strike=100.0, maturity_years=1.0
    )
    report = build_model_risk_report(
        ModelRiskRequest(
            comparison=ModelComparisonRequest(
                contract=contract,
                market=MarketEnvironment(risk_free_rate=0.02, volatility=0.2),
                config=ModelComparisonConfig(selected_models=[PricingModel.HESTON_ANALYTICAL]),
                heston_parameters=stability.runs[0].parameters,
            ),
            calibration_stability=stability,
        )
    )
    assert report.calibration_stability is not None
    assert report.calibration_stability.ensemble.members[0].provenance.startswith("phase14")


def test_phase6_surface_comparison_preserves_atm_skew_and_curvature() -> None:
    surface = generate_synthetic_surface(
        maturities=(0.5, 1.0),
        log_moneyness=(-0.1, 0.0, 0.1),
        base_volatility=0.2,
        skew=-0.1,
        curvature=0.2,
    )
    result = compare_surface(
        SurfaceComparisonRequest(
            surface=surface,
            selected_models=[PricingModel.BLACK_SCHOLES],
        )
    )
    fit = result.models[0]
    assert fit.metrics.iv_rmse == pytest.approx(0.0, abs=1e-10)
    assert len(fit.smile_fit) == 2
    phase6 = smile_local_metrics(surface.smiles[0])
    assert fit.smile_fit[0].market_skew == pytest.approx(phase6.skew_dvol_dlog_moneyness, abs=1e-12)
    assert fit.smile_fit[0].market_curvature == pytest.approx(
        phase6.curvature_d2vol_dlog_moneyness2, abs=1e-12
    )
