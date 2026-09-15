import math

import pytest
from pydantic import ValidationError

from backend.app.core.config import Settings
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.models.simulation import SimulationConfig


def option_data() -> dict[str, object]:
    return {
        "option_type": "CALL",
        "exercise_style": "EUROPEAN",
        "spot": 100.0,
        "strike": 100.0,
        "maturity_years": 1.0,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("spot", 0.0),
        ("spot", -1.0),
        ("strike", 0.0),
        ("strike", -1.0),
        ("maturity_years", -1e-12),
        ("spot", math.inf),
        ("strike", math.nan),
    ],
)
def test_option_validation_rejects_invalid_numbers(field: str, value: float) -> None:
    data = option_data()
    data[field] = value
    with pytest.raises(ValidationError):
        OptionContract.model_validate(data)


@pytest.mark.parametrize("option_type", ["DIGITAL", "call", 1])
def test_option_validation_rejects_invalid_enum(option_type: object) -> None:
    data = option_data()
    data["option_type"] = option_type
    with pytest.raises(ValidationError):
        OptionContract.model_validate(data)


def test_market_rejects_negative_or_non_finite_volatility() -> None:
    with pytest.raises(ValidationError):
        MarketEnvironment(risk_free_rate=0.01, volatility=-0.1)
    with pytest.raises(ValidationError):
        MarketEnvironment(risk_free_rate=0.01, volatility=math.inf)


def test_market_allows_negative_rates_and_dividend_yields() -> None:
    market = MarketEnvironment(
        risk_free_rate=-0.03,
        volatility=0.2,
        dividend_yield=-0.01,
    )
    assert market.risk_free_rate == -0.03
    assert market.dividend_yield == -0.01


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_simulations", 0),
        ("num_simulations", -1),
        ("confidence_level", 0.0),
        ("confidence_level", 1.0),
        ("confidence_level", -0.1),
        ("batch_size", 0),
        ("batch_size", -1),
        ("seed", -1),
    ],
)
def test_simulation_config_rejects_invalid_values(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        SimulationConfig.model_validate({field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("variance_reduction", "CONTROL_VARIATE"),
        ("random_engine", "MT19937"),
    ],
)
def test_simulation_config_rejects_unknown_enums(field: str, value: str) -> None:
    with pytest.raises(ValidationError):
        SimulationConfig.model_validate({field: value})


def test_models_forbid_unknown_fields() -> None:
    data = option_data()
    data["currency"] = "USD"
    with pytest.raises(ValidationError):
        OptionContract.model_validate(data)


def test_settings_limits_are_positive() -> None:
    with pytest.raises(ValidationError):
        Settings(max_api_simulations=0)
    with pytest.raises(ValidationError):
        Settings(max_convergence_points=0)


def test_settings_load_environment_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_API_SIMULATIONS", "1234")
    monkeypatch.setenv("MAX_CONVERGENCE_POINTS", "7")
    monkeypatch.setenv("LOG_LEVEL", "debug")
    settings = Settings.from_environment()
    assert settings.max_api_simulations == 1_234
    assert settings.max_convergence_points == 7
    assert settings.log_level == "DEBUG"


def test_contract_accepts_zero_maturity_and_float_values() -> None:
    option = OptionContract(
        option_type=OptionType.PUT,
        spot=100.25,
        strike=101.75,
        maturity_years=0.0,
    )
    assert option.maturity_years == 0.0
