"""Deterministic registry of implemented pricing engines and honest capabilities."""

from backend.app.models.model_comparison import (
    ModelCapabilities,
    ModelCostClass,
    ModelExclusion,
    PricingModel,
)
from backend.app.models.option import ExerciseStyle, OptionContract


def _cap(
    model: PricingModel,
    *,
    european: bool,
    american: bool,
    stochastic_volatility: bool,
    analytical: bool,
    monte_carlo: bool,
    greeks: bool,
    calibration: bool,
    deterministic: bool,
    cost: ModelCostClass,
    states: int,
    factors: int,
    parameter_count: int,
    integration: bool = False,
) -> ModelCapabilities:
    return ModelCapabilities(
        model=model,
        supports_european=european,
        supports_american=american,
        supports_stochastic_volatility=stochastic_volatility,
        supports_closed_form_or_semi_analytical=analytical,
        supports_monte_carlo=monte_carlo,
        supports_greeks=greeks,
        requires_calibration=calibration,
        deterministic=deterministic,
        typical_cost=cost,
        state_variables=states,
        stochastic_factors=factors,
        calibrated_parameter_count=parameter_count,
        numerical_integration_required=integration,
        simulation_required=monte_carlo,
        early_exercise_capability=american,
    )


_CAPABILITIES: dict[PricingModel, ModelCapabilities] = {
    PricingModel.BLACK_SCHOLES: _cap(
        PricingModel.BLACK_SCHOLES,
        european=True,
        american=False,
        stochastic_volatility=False,
        analytical=True,
        monte_carlo=False,
        greeks=True,
        calibration=False,
        deterministic=True,
        cost=ModelCostClass.LOW,
        states=1,
        factors=1,
        parameter_count=1,
    ),
    PricingModel.MONTE_CARLO_GBM: _cap(
        PricingModel.MONTE_CARLO_GBM,
        european=True,
        american=False,
        stochastic_volatility=False,
        analytical=False,
        monte_carlo=True,
        greeks=True,
        calibration=False,
        deterministic=False,
        cost=ModelCostClass.MEDIUM,
        states=1,
        factors=1,
        parameter_count=1,
    ),
    PricingModel.CRR_BINOMIAL: _cap(
        PricingModel.CRR_BINOMIAL,
        european=True,
        american=False,
        stochastic_volatility=False,
        analytical=False,
        monte_carlo=False,
        greeks=True,
        calibration=False,
        deterministic=True,
        cost=ModelCostClass.MEDIUM,
        states=1,
        factors=1,
        parameter_count=1,
    ),
    PricingModel.HESTON_ANALYTICAL: _cap(
        PricingModel.HESTON_ANALYTICAL,
        european=True,
        american=False,
        stochastic_volatility=True,
        analytical=True,
        monte_carlo=False,
        greeks=True,
        calibration=True,
        deterministic=True,
        cost=ModelCostClass.HIGH,
        states=2,
        factors=2,
        parameter_count=5,
        integration=True,
    ),
    PricingModel.HESTON_MONTE_CARLO: _cap(
        PricingModel.HESTON_MONTE_CARLO,
        european=True,
        american=False,
        stochastic_volatility=True,
        analytical=False,
        monte_carlo=True,
        greeks=False,
        calibration=True,
        deterministic=False,
        cost=ModelCostClass.VERY_HIGH,
        states=2,
        factors=2,
        parameter_count=5,
    ),
    PricingModel.CRR_AMERICAN: _cap(
        PricingModel.CRR_AMERICAN,
        european=False,
        american=True,
        stochastic_volatility=False,
        analytical=False,
        monte_carlo=False,
        greeks=True,
        calibration=False,
        deterministic=True,
        cost=ModelCostClass.MEDIUM,
        states=1,
        factors=1,
        parameter_count=1,
    ),
    PricingModel.LONGSTAFF_SCHWARTZ: _cap(
        PricingModel.LONGSTAFF_SCHWARTZ,
        european=False,
        american=True,
        stochastic_volatility=False,
        analytical=False,
        monte_carlo=True,
        greeks=False,
        calibration=False,
        deterministic=False,
        cost=ModelCostClass.VERY_HIGH,
        states=1,
        factors=1,
        parameter_count=1,
    ),
}


def registered_models() -> tuple[PricingModel, ...]:
    """Return stable registry order for serialization and research tables."""

    return tuple(_CAPABILITIES)


def model_capabilities(model: PricingModel) -> ModelCapabilities:
    return _CAPABILITIES[model]


def capability_registry() -> tuple[ModelCapabilities, ...]:
    return tuple(_CAPABILITIES[model] for model in registered_models())


def compatibility_exclusion(
    model: PricingModel,
    contract: OptionContract,
    *,
    heston_parameters_available: bool,
) -> ModelExclusion | None:
    capability = model_capabilities(model)
    if contract.exercise_style is ExerciseStyle.EUROPEAN and not capability.supports_european:
        return ModelExclusion(model=model, reason="model does not support European exercise")
    if contract.exercise_style is ExerciseStyle.AMERICAN and not capability.supports_american:
        return ModelExclusion(model=model, reason="model does not support American exercise")
    if capability.requires_calibration and not heston_parameters_available:
        return ModelExclusion(
            model=model,
            reason="model requires explicit Heston parameters or a calibration result",
        )
    return None
