import pytest

from backend.app.analytics.model_comparison import (
    compare_models,
    compare_surface,
    evaluate_market_fit,
)
from backend.app.calibration.synthetic import generate_synthetic_heston_dataset
from backend.app.model_comparison.registry import model_capabilities
from backend.app.models.heston_calibration import (
    CalibrationSelectionPolicy,
    HestonCalibrationConfig,
    HestonCalibrationTarget,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.merton import MertonJumpParameters
from backend.app.models.model_comparison import (
    MarketFitRequest,
    ModelComparisonConfig,
    ModelComparisonRequest,
    PricingModel,
    SurfaceComparisonRequest,
)
from backend.app.models.option import ExerciseStyle, OptionContract, OptionType
from backend.app.models.portfolio import (
    InstrumentClass,
    MarketScenario,
    MertonScenario,
    Portfolio,
    PortfolioPricingModel,
    PortfolioRequest,
    PortfolioScenarioRequest,
    VanillaPosition,
)
from backend.app.portfolio.engine import price_portfolio, run_scenario
from backend.app.volatility_surface.synthetic import generate_synthetic_surface

CONTRACT = OptionContract(
    option_type=OptionType.PUT,
    exercise_style=ExerciseStyle.EUROPEAN,
    spot=100.0,
    strike=105.0,
    maturity_years=0.5,
)
MARKET = MarketEnvironment(risk_free_rate=0.02, dividend_yield=0.01, volatility=0.18)
PARAMETERS = MertonJumpParameters(jump_intensity=0.7, jump_mean=-0.15, jump_volatility=0.22)


def test_registry_capabilities() -> None:
    analytical = model_capabilities(PricingModel.MERTON_ANALYTICAL)
    monte_carlo = model_capabilities(PricingModel.MERTON_MONTE_CARLO)
    assert analytical.supports_jump_diffusion
    assert analytical.supports_european and not analytical.supports_american
    assert analytical.supports_closed_form_or_semi_analytical
    assert analytical.supports_greeks
    assert monte_carlo.supports_monte_carlo and not monte_carlo.deterministic


def test_model_comparison_includes_merton_engines() -> None:
    request = ModelComparisonRequest(
        contract=CONTRACT,
        market=MARKET,
        merton_parameters=PARAMETERS,
        merton_monte_carlo={"num_simulations": 80_000, "seed": 7},
        config=ModelComparisonConfig(
            selected_models=[
                PricingModel.BLACK_SCHOLES,
                PricingModel.MERTON_ANALYTICAL,
                PricingModel.MERTON_MONTE_CARLO,
            ],
            include_greeks=True,
        ),
    )
    result = compare_models(request)
    by_model = {item.model: item for item in result.outputs}
    assert by_model[PricingModel.MERTON_ANALYTICAL].greeks.delta is not None
    mc = by_model[PricingModel.MERTON_MONTE_CARLO]
    analytical = by_model[PricingModel.MERTON_ANALYTICAL]
    assert mc.price is not None and analytical.price is not None and mc.standard_error is not None
    assert abs(mc.price - analytical.price) <= 4.0 * mc.standard_error
    assert any(
        item.name == "MERTON_ANALYTICAL_VS_MONTE_CARLO" for item in result.consistency_diagnostics
    )


def test_market_fit_produces_merton_expiry_and_moneyness_metrics() -> None:
    dataset = generate_synthetic_heston_dataset(
        maturities=(0.25, 0.5), log_moneyness=(-0.1, 0.0, 0.1)
    )
    result = evaluate_market_fit(
        MarketFitRequest(
            dataset=dataset,
            selected_models=[PricingModel.MERTON_ANALYTICAL],
            merton_parameters=PARAMETERS,
            preprocessing=HestonCalibrationConfig(
                target=HestonCalibrationTarget.PRICE,
                selection_policy=CalibrationSelectionPolicy.ALL,
                identifiability_diagnostics=False,
            ),
        )
    )
    fit = result.models[0]
    assert fit.metrics.observation_count == 6
    assert len(fit.per_expiry) == 2
    assert fit.per_moneyness


def test_phase6_surface_flows_into_merton_fit() -> None:
    surface = generate_synthetic_surface(
        maturities=(0.25, 0.5),
        log_moneyness=(-0.1, 0.0, 0.1),
        base_volatility=0.2,
        skew=-0.1,
        curvature=0.1,
    )
    result = compare_surface(
        SurfaceComparisonRequest(
            surface=surface,
            selected_models=[PricingModel.MERTON_ANALYTICAL],
            merton_parameters=PARAMETERS,
        )
    )
    assert result.models[0].metrics.observation_count == 6
    assert len(result.models[0].smile_fit) == 2


def position(
    model: PortfolioPricingModel = PortfolioPricingModel.MERTON_ANALYTICAL,
) -> VanillaPosition:
    return VanillaPosition(
        position_id="merton-put",
        quantity=3.0,
        model=model,
        instrument_class=InstrumentClass.VANILLA_EUROPEAN,
        underlying_id="ABC",
        instrument=CONTRACT,
        market=MARKET,
        merton_parameters=PARAMETERS,
        merton_simulation={"num_simulations": 50_000, "seed": 3},
    )


def test_portfolio_value_and_parameter_scenario() -> None:
    portfolio = Portfolio(positions=[position()])
    base = price_portfolio(PortfolioRequest(portfolio=portfolio))
    scenario = run_scenario(
        PortfolioScenarioRequest(
            portfolio=portfolio,
            scenario=MarketScenario(
                merton=MertonScenario(
                    jump_intensity_shift=0.5,
                    jump_mean_shift=-0.05,
                    jump_volatility_shift=0.04,
                )
            ),
        )
    )
    repeated = run_scenario(
        PortfolioScenarioRequest(
            portfolio=portfolio,
            scenario=MarketScenario(
                merton=MertonScenario(
                    jump_intensity_shift=0.5,
                    jump_mean_shift=-0.05,
                    jump_volatility_shift=0.04,
                )
            ),
        )
    )
    assert base.total_value is not None
    assert scenario.full_repricing_pnl is not None
    assert scenario.full_repricing_pnl == repeated.full_repricing_pnl


def test_realized_jump_stress_changes_spot_deterministically() -> None:
    portfolio = Portfolio(positions=[position()])
    result = run_scenario(
        PortfolioScenarioRequest(
            portfolio=portfolio,
            scenario=MarketScenario(realized_jump_log_size=-0.2),
        )
    )
    assert result.contributions[0].scenario_market is not None
    assert result.contributions[0].scenario_market.spots["ABC"] == pytest.approx(81.87307530779819)
