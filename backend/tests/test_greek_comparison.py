import pytest

from backend.app.analytics.greeks import compare_greek_estimators
from backend.app.models.greeks import Greek, GreekMethod
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.models.simulation import SimulationConfig, VarianceReduction


def inputs() -> tuple[OptionContract, MarketEnvironment]:
    return (
        OptionContract(
            option_type=OptionType.CALL,
            spot=100.0,
            strike=100.0,
            maturity_years=1.0,
        ),
        MarketEnvironment(risk_free_rate=0.05, volatility=0.2, dividend_yield=0.01),
    )


@pytest.mark.parametrize("greek", [Greek.DELTA, Greek.VEGA])
def test_delta_and_vega_comparison_include_finite_difference_and_pathwise(
    greek: Greek,
) -> None:
    contract, market = inputs()
    result = compare_greek_estimators(
        contract,
        market,
        SimulationConfig(
            num_simulations=100_000,
            seed=42,
            variance_reduction=VarianceReduction.ANTITHETIC,
        ),
        greek,
    )
    assert result.greek is greek
    assert [estimate.method for estimate in result.estimates] == [
        GreekMethod.FINITE_DIFFERENCE,
        GreekMethod.PATHWISE,
    ]
    for estimate in result.estimates:
        assert estimate.absolute_error == pytest.approx(
            abs(estimate.estimate - result.analytical_benchmark)
        )
        assert estimate.z_score == pytest.approx(
            (estimate.estimate - result.analytical_benchmark) / estimate.standard_error
        )


@pytest.mark.parametrize(
    "greek",
    [Greek.GAMMA, Greek.THETA, Greek.RHO, Greek.DIVIDEND_RHO],
)
def test_other_greeks_expose_only_finite_difference(greek: Greek) -> None:
    contract, market = inputs()
    result = compare_greek_estimators(
        contract,
        market,
        SimulationConfig(
            num_simulations=20_000,
            seed=13,
            variance_reduction=VarianceReduction.NONE,
        ),
        greek,
    )
    assert [estimate.method for estimate in result.estimates] == [GreekMethod.FINITE_DIFFERENCE]


def test_control_variate_comparison_omits_unsupported_pathwise_method() -> None:
    contract, market = inputs()
    result = compare_greek_estimators(
        contract,
        market,
        SimulationConfig(
            num_simulations=20_000,
            seed=13,
            variance_reduction=VarianceReduction.CONTROL_VARIATE,
        ),
        Greek.DELTA,
    )
    assert [estimate.method for estimate in result.estimates] == [GreekMethod.FINITE_DIFFERENCE]


def test_zero_se_comparison_uses_unavailable_z_score() -> None:
    contract, market = inputs()
    expired = contract.model_copy(update={"maturity_years": 0.0, "spot": 120.0})
    result = compare_greek_estimators(
        expired,
        market,
        SimulationConfig(num_simulations=5, variance_reduction=VarianceReduction.NONE),
        Greek.DELTA,
    )
    assert all(estimate.standard_error == 0.0 for estimate in result.estimates)
    assert all(estimate.z_score is None for estimate in result.estimates)
