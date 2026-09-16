import math

import pytest
from pydantic import ValidationError

from backend.app.core.exceptions import VolatilitySurfaceError
from backend.app.models.option import OptionType
from backend.app.models.volatility_surface import (
    ExtrapolationPolicy,
    SmileInterpolationConfig,
    SmileInterpolationMethod,
    VolatilityPoint,
)
from backend.app.volatility_surface.core import (
    build_surface,
    calculate_forward,
    implied_vol_to_total_variance,
    make_volatility_point,
    query_smile,
    query_surface,
    total_variance_to_implied_vol,
)
from backend.app.volatility_surface.synthetic import (
    generate_synthetic_smile,
    generate_synthetic_surface,
)


@pytest.mark.parametrize(
    ("volatility", "maturity", "expected"),
    [(0.0, 1.0, 0.0), (0.2, 2.0, 0.08), (3.0, 0.25, 2.25), (0.4, 1e-12, 1.6e-13)],
)
def test_total_variance_round_trip(volatility: float, maturity: float, expected: float) -> None:
    variance = implied_vol_to_total_variance(volatility, maturity)
    assert variance == pytest.approx(expected)
    assert total_variance_to_implied_vol(variance, maturity) == pytest.approx(volatility)


def test_total_variance_helpers_reject_invalid_values() -> None:
    with pytest.raises(VolatilitySurfaceError):
        implied_vol_to_total_variance(-0.2, 1.0)
    with pytest.raises(VolatilitySurfaceError):
        implied_vol_to_total_variance(0.2, 0.0)
    with pytest.raises(VolatilitySurfaceError):
        total_variance_to_implied_vol(-1.0, 1.0)
    with pytest.raises(VolatilitySurfaceError):
        total_variance_to_implied_vol(0.04, 0.0)


@pytest.mark.parametrize(
    ("rate", "dividend", "maturity"),
    [(0.05, 0.0, 1.0), (-0.02, 0.01, 2.0), (0.01, 0.04, 0.25)],
)
def test_forward_and_log_forward_moneyness(rate: float, dividend: float, maturity: float) -> None:
    forward = calculate_forward(100.0, maturity, rate, dividend)
    assert forward == pytest.approx(100.0 * math.exp((rate - dividend) * maturity))
    point = make_volatility_point(
        strike=105.0,
        maturity_years=maturity,
        implied_volatility=0.2,
        option_type=OptionType.CALL,
        spot=100.0,
        forward=forward,
    )
    assert point.moneyness == pytest.approx(1.05)
    assert point.forward_moneyness == pytest.approx(105.0 / forward)
    assert point.log_moneyness == pytest.approx(math.log(105.0 / forward))


def test_point_rejects_inconsistent_and_nonfinite_coordinates() -> None:
    with pytest.raises(ValidationError):
        VolatilityPoint(
            strike=100.0,
            maturity_years=1.0,
            implied_volatility=0.2,
            option_type=OptionType.CALL,
            forward_price=100.0,
            moneyness=1.0,
            forward_moneyness=2.0,
            log_moneyness=0.0,
        )
    with pytest.raises(ValidationError):
        VolatilityPoint(
            strike=100.0,
            maturity_years=1.0,
            implied_volatility=float("nan"),
            option_type=OptionType.CALL,
            forward_price=100.0,
            moneyness=1.0,
            forward_moneyness=1.0,
            log_moneyness=0.0,
        )


@pytest.mark.parametrize("method", list(SmileInterpolationMethod))
def test_smile_interpolation_exact_nodes_and_flat_surface(
    method: SmileInterpolationMethod,
) -> None:
    smile = generate_synthetic_smile(
        maturity_years=1.0,
        log_moneyness=[0.2, -0.2, 0.0, 0.1, -0.1],
        base_volatility=0.25,
    )
    assert [point.log_moneyness for point in smile.points] == sorted(
        point.log_moneyness for point in smile.points
    )
    config = SmileInterpolationConfig(method=method)
    exact = query_smile(smile, log_moneyness=0.1, config=config)
    between = query_smile(smile, log_moneyness=0.05, config=config)
    assert exact.exact_observation is True
    assert exact.implied_volatility == pytest.approx(0.25)
    assert between.interpolated is True
    assert between.implied_volatility == pytest.approx(0.25)


def test_smile_rejects_duplicate_coordinate() -> None:
    smile = generate_synthetic_smile(
        maturity_years=1.0,
        log_moneyness=[-0.1, 0.0, 0.1],
    )
    with pytest.raises(ValidationError, match="unique"):
        smile.model_copy(update={"points": [smile.points[0], smile.points[0]]}).model_validate(
            smile.model_copy(update={"points": [smile.points[0], smile.points[0]]}).model_dump()
        )


def test_smile_no_extrapolation_by_default_and_flat_vol_when_enabled() -> None:
    smile = generate_synthetic_smile(
        maturity_years=1.0,
        log_moneyness=[-0.1, 0.0, 0.1],
        base_volatility=0.2,
        skew=-0.1,
    )
    with pytest.raises(VolatilitySurfaceError, match="outside smile domain"):
        query_smile(smile, log_moneyness=0.2)
    result = query_smile(
        smile,
        log_moneyness=0.2,
        config=SmileInterpolationConfig(extrapolation=ExtrapolationPolicy.FLAT_VOL),
    )
    assert result.extrapolation_used is True
    assert result.implied_volatility == pytest.approx(smile.points[-1].implied_volatility)


def test_surface_maturity_interpolation_uses_total_variance() -> None:
    lower = generate_synthetic_smile(
        maturity_years=0.5,
        log_moneyness=[-0.1, 0.0, 0.1],
        base_volatility=0.2,
    )
    upper = generate_synthetic_smile(
        maturity_years=1.5,
        log_moneyness=[-0.1, 0.0, 0.1],
        base_volatility=0.3,
    )
    result = query_surface(build_surface([upper, lower]), maturity_years=1.0, log_moneyness=0.0)
    expected_variance = 0.5 * (0.2**2 * 0.5 + 0.3**2 * 1.5)
    assert result.total_variance == pytest.approx(expected_variance)
    assert result.implied_volatility == pytest.approx(math.sqrt(expected_variance))
    assert result.implied_volatility != pytest.approx(0.25)


def test_surface_exact_grid_point_preserves_observed_iv() -> None:
    surface = generate_synthetic_surface(
        maturities=[0.5, 1.0],
        log_moneyness=[-0.1, 0.0, 0.1],
        base_volatility=0.2,
        skew=-0.15,
        curvature=0.4,
    )
    observed = surface.smiles[1].points[2]
    result = query_surface(
        surface,
        maturity_years=1.0,
        strike=observed.strike,
    )
    assert result.exact_maturity is True
    assert result.exact_observation is True
    assert result.implied_volatility == observed.implied_volatility


def test_flat_synthetic_surface_round_trip() -> None:
    surface = generate_synthetic_surface(
        maturities=[0.25, 1.0, 2.0],
        log_moneyness=[-0.2, -0.05, 0.1, 0.2],
        base_volatility=0.23,
    )
    for maturity, coordinate in [(0.5, -0.1), (1.5, 0.0), (2.0, 0.15)]:
        result = query_surface(
            surface,
            maturity_years=maturity,
            log_moneyness=coordinate,
        )
        assert result.implied_volatility == pytest.approx(0.23, abs=1e-12)


def test_surface_rejects_maturity_extrapolation_by_default() -> None:
    surface = generate_synthetic_surface(
        maturities=[0.5, 1.0],
        log_moneyness=[-0.1, 0.0, 0.1],
    )
    with pytest.raises(VolatilitySurfaceError, match="outside surface domain"):
        query_surface(surface, maturity_years=2.0, log_moneyness=0.0)
