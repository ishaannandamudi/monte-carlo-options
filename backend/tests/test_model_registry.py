from backend.app.model_comparison.registry import (
    capability_registry,
    compatibility_exclusion,
    model_capabilities,
    registered_models,
)
from backend.app.models.model_comparison import ModelCostClass, PricingModel
from backend.app.models.option import ExerciseStyle, OptionContract, OptionType


def contract(style: ExerciseStyle = ExerciseStyle.EUROPEAN) -> OptionContract:
    return OptionContract(
        option_type=OptionType.PUT,
        exercise_style=style,
        spot=100.0,
        strike=105.0,
        maturity_years=1.0,
    )


def test_registry_is_complete_deterministic_and_capabilities_are_descriptive() -> None:
    assert registered_models() == tuple(PricingModel)
    assert [item.model for item in capability_registry()] == list(PricingModel)
    bsm = model_capabilities(PricingModel.BLACK_SCHOLES)
    assert bsm.deterministic and bsm.supports_greeks
    assert bsm.typical_cost is ModelCostClass.LOW
    assert not bsm.supports_american
    heston = model_capabilities(PricingModel.HESTON_ANALYTICAL)
    assert heston.requires_calibration
    assert heston.calibrated_parameter_count == 5
    assert heston.numerical_integration_required
    lsm = model_capabilities(PricingModel.LONGSTAFF_SCHWARTZ)
    assert lsm.supports_american and lsm.supports_monte_carlo
    assert lsm.early_exercise_capability


def test_compatibility_returns_structured_exclusions_without_substitution() -> None:
    assert (
        compatibility_exclusion(
            PricingModel.BLACK_SCHOLES, contract(), heston_parameters_available=False
        )
        is None
    )
    american_exclusion = compatibility_exclusion(
        PricingModel.BLACK_SCHOLES,
        contract(ExerciseStyle.AMERICAN),
        heston_parameters_available=False,
    )
    assert american_exclusion is not None
    assert "American" in american_exclusion.reason
    heston_exclusion = compatibility_exclusion(
        PricingModel.HESTON_ANALYTICAL, contract(), heston_parameters_available=False
    )
    assert heston_exclusion is not None
    assert "Heston parameters" in heston_exclusion.reason
