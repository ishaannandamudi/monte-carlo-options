import math

import pytest

from backend.app.analytics.model_comparison import (
    analyze_model_stability,
    compare_models,
    greek_dispersion,
    pairwise_matrix,
    price_dispersion,
    scalar_dispersion,
)
from backend.app.models.binomial import BinomialTreeConfig
from backend.app.models.heston import HestonParameters, HestonSimulationConfig
from backend.app.models.longstaff_schwartz import LongstaffSchwartzConfig
from backend.app.models.market import MarketEnvironment
from backend.app.models.model_comparison import (
    ModelComparisonConfig,
    ModelComparisonRequest,
    ModelGreekValues,
    ModelOutput,
    ModelRunStatus,
    PriceRangeNormalization,
    PricingModel,
    StabilityInput,
    StabilityPerturbation,
)
from backend.app.models.option import ExerciseStyle, OptionContract, OptionType
from backend.app.models.simulation import SimulationConfig, VarianceReduction


def european() -> tuple[OptionContract, MarketEnvironment]:
    return (
        OptionContract(option_type=OptionType.CALL, spot=100.0, strike=100.0, maturity_years=1.0),
        MarketEnvironment(risk_free_rate=0.05, volatility=0.2, dividend_yield=0.01),
    )


def test_exact_dispersion_pairwise_and_missing_greeks() -> None:
    outputs = [
        ModelOutput(
            model=PricingModel.BLACK_SCHOLES,
            status=ModelRunStatus.SUCCESS,
            price=10.0,
            runtime_ms=1.0,
            deterministic=True,
            greeks=ModelGreekValues(delta=0.5, gamma=0.02),
        ),
        ModelOutput(
            model=PricingModel.CRR_BINOMIAL,
            status=ModelRunStatus.SUCCESS,
            price=12.0,
            runtime_ms=2.0,
            deterministic=True,
            greeks=ModelGreekValues(delta=0.7),
        ),
    ]
    scalar = scalar_dispersion([10.0, 12.0])
    assert scalar.mean == 11.0
    assert scalar.standard_deviation == 1.0
    dispersion = price_dispersion(
        [10.0, 12.0],
        normalization=PriceRangeNormalization.MEDIAN_PRICE,
        spot=100.0,
        reference_price=10.0,
        floor=1e-8,
    )
    assert dispersion.range == 2.0
    assert dispersion.relative_range == pytest.approx(2.0 / 11.0)
    matrix = pairwise_matrix(outputs, 1e-8)
    assert matrix is not None
    assert matrix.signed_differences == [[0.0, -2.0], [2.0, 0.0]]
    greeks = {item.greek: item for item in greek_dispersion(outputs)}
    assert greeks["DELTA"].dispersion.count == 2
    assert greeks["GAMMA"].dispersion.count == 1
    assert greeks["VEGA"].dispersion.count == 0


def test_bsm_gbm_mc_and_crr_comparison_is_reproducible() -> None:
    contract, market = european()
    request = ModelComparisonRequest(
        contract=contract,
        market=market,
        config=ModelComparisonConfig(
            selected_models=[
                PricingModel.BLACK_SCHOLES,
                PricingModel.MONTE_CARLO_GBM,
                PricingModel.CRR_BINOMIAL,
            ],
            reference_model=PricingModel.BLACK_SCHOLES,
            include_greeks=True,
            deterministic_seed=91,
        ),
        gbm_monte_carlo=SimulationConfig(
            num_simulations=20_000,
            variance_reduction=VarianceReduction.ANTITHETIC_CONTROL_VARIATE,
        ),
        tree=BinomialTreeConfig(steps=500),
    )
    first = compare_models(request)
    second = compare_models(request)
    assert first.experiment.input_signature == second.experiment.input_signature
    assert [item.price for item in first.outputs] == [item.price for item in second.outputs]
    by_model = {item.model: item for item in first.outputs}
    bsm = by_model[PricingModel.BLACK_SCHOLES]
    mc = by_model[PricingModel.MONTE_CARLO_GBM]
    crr = by_model[PricingModel.CRR_BINOMIAL]
    assert bsm.price is not None and mc.price is not None and mc.standard_error is not None
    assert abs(mc.price - bsm.price) <= 5.0 * mc.standard_error
    assert crr.price == pytest.approx(bsm.price, abs=0.01)
    assert len(first.reference_deviations) == 3
    assert first.pairwise is not None
    assert len(first.pairwise.signed_differences) == 3
    assert {item.name for item in first.consistency_diagnostics} == {
        "BSM_VS_GBM_MONTE_CARLO",
        "BSM_VS_CRR",
    }
    assert all(math.isfinite(item.runtime_ms or 0.0) for item in first.outputs)


def test_heston_limit_and_generalized_heston_mc_comparison() -> None:
    contract, market = european()
    parameters = HestonParameters(v0=0.04, kappa=1.5, theta=0.04, sigma_v=0.0, rho=-0.5)
    result = compare_models(
        ModelComparisonRequest(
            contract=contract,
            market=market,
            config=ModelComparisonConfig(
                selected_models=[
                    PricingModel.BLACK_SCHOLES,
                    PricingModel.HESTON_ANALYTICAL,
                    PricingModel.HESTON_MONTE_CARLO,
                ],
                include_greeks=True,
            ),
            heston_parameters=parameters,
            heston_monte_carlo=HestonSimulationConfig(
                num_simulations=8_000, num_time_steps=20, seed=3
            ),
        )
    )
    by_model = {item.model: item for item in result.outputs}
    assert by_model[PricingModel.HESTON_ANALYTICAL].price == pytest.approx(
        by_model[PricingModel.BLACK_SCHOLES].price, abs=1e-10
    )
    mc = by_model[PricingModel.HESTON_MONTE_CARLO]
    analytical = by_model[PricingModel.HESTON_ANALYTICAL]
    assert mc.price is not None and analytical.price is not None and mc.standard_error is not None
    assert abs(mc.price - analytical.price) <= 5.0 * mc.standard_error
    assert analytical.greeks.vega is None
    assert analytical.greeks.delta is not None
    assert any(
        item.name == "HESTON_ANALYTICAL_VS_MONTE_CARLO" for item in result.consistency_diagnostics
    )


def test_american_crr_lsm_and_incompatible_models_are_explicit() -> None:
    contract = OptionContract(
        option_type=OptionType.PUT,
        exercise_style=ExerciseStyle.AMERICAN,
        spot=40.0,
        strike=40.0,
        maturity_years=1.0,
    )
    market = MarketEnvironment(risk_free_rate=0.06, volatility=0.2)
    result = compare_models(
        ModelComparisonRequest(
            contract=contract,
            market=market,
            config=ModelComparisonConfig(
                selected_models=[
                    PricingModel.BLACK_SCHOLES,
                    PricingModel.CRR_AMERICAN,
                    PricingModel.LONGSTAFF_SCHWARTZ,
                ],
                reference_model=PricingModel.CRR_AMERICAN,
            ),
            tree=BinomialTreeConfig(steps=300),
            longstaff_schwartz=LongstaffSchwartzConfig(
                num_paths=12_000, regression_paths=12_000, num_time_steps=30, seed=8
            ),
        )
    )
    assert result.excluded_models == [PricingModel.BLACK_SCHOLES]
    by_model = {item.model: item for item in result.outputs}
    lsm = by_model[PricingModel.LONGSTAFF_SCHWARTZ]
    crr = by_model[PricingModel.CRR_AMERICAN]
    assert lsm.price is not None and crr.price is not None and lsm.standard_error is not None
    assert abs(lsm.price - crr.price) <= max(5.0 * lsm.standard_error, 0.2)
    assert lsm.early_exercise_fraction is not None


def test_one_model_numerical_failure_is_isolated() -> None:
    contract, _ = european()
    market = MarketEnvironment(risk_free_rate=10.0, volatility=0.01)
    result = compare_models(
        ModelComparisonRequest(
            contract=contract,
            market=market,
            config=ModelComparisonConfig(
                selected_models=[PricingModel.BLACK_SCHOLES, PricingModel.CRR_BINOMIAL]
            ),
            tree=BinomialTreeConfig(steps=1),
        )
    )
    assert result.successful_models == [PricingModel.BLACK_SCHOLES]
    assert result.failed_models == [PricingModel.CRR_BINOMIAL]
    assert result.outputs[1].error_type is not None


def test_input_stability_returns_separate_perturbation_changes() -> None:
    contract, market = european()
    result = analyze_model_stability(
        ModelComparisonRequest(
            contract=contract,
            market=market,
            config=ModelComparisonConfig(selected_models=[PricingModel.BLACK_SCHOLES]),
        ),
        [StabilityPerturbation(input=StabilityInput.SPOT)],
    )
    assert len(result.points) == 1
    assert result.points[0].status is ModelRunStatus.SUCCESS
    assert result.points[0].upward_change > 0.0
    assert result.points[0].central_sensitivity is not None
