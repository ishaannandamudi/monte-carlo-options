"""Offline synthetic Heston calibration datasets with known parameters."""

import math
from collections.abc import Sequence

from backend.app.models.heston import HestonParameters
from backend.app.models.heston_calibration import (
    HestonCalibrationDataset,
    HestonCalibrationObservation,
)
from backend.app.models.market import MarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.models.volatility import ImpliedVolatilityMarket, ImpliedVolatilityObservation
from backend.app.pricing.black_scholes import black_scholes_price
from backend.app.pricing.heston_analytical import heston_analytical_price
from backend.app.pricing.implied_volatility import forward_price, solve_implied_volatility
from backend.app.simulation.random import create_rng

FELLER_SATISFIED_PARAMETERS = HestonParameters(
    v0=0.04,
    kappa=2.0,
    theta=0.04,
    sigma_v=0.3,
    rho=-0.7,
)
FELLER_VIOLATING_PARAMETERS = HestonParameters(
    v0=0.04,
    kappa=0.8,
    theta=0.03,
    sigma_v=0.6,
    rho=-0.75,
)


def generate_synthetic_heston_dataset(
    *,
    parameters: HestonParameters = FELLER_SATISFIED_PARAMETERS,
    maturities: Sequence[float] = (0.25, 0.5, 1.0, 2.0),
    log_moneyness: Sequence[float] = (-0.2, -0.1, 0.0, 0.1, 0.2),
    spot: float = 100.0,
    risk_free_rate: float = 0.02,
    dividend_yield: float = 0.01,
    implied_volatility_noise_std: float = 0.0,
    seed: int = 42,
    bid_ask_half_spread: float | None = 0.02,
) -> HestonCalibrationDataset:
    """Generate OTM calls/puts from Phase 13 prices, optionally perturbing BSM IVs."""

    if not maturities or not log_moneyness:
        raise ValueError("synthetic Heston calibration grids must be nonempty")
    if any(value <= 0.0 or not math.isfinite(value) for value in maturities):
        raise ValueError("synthetic Heston maturities must be positive and finite")
    if any(not math.isfinite(value) for value in log_moneyness):
        raise ValueError("synthetic Heston log-moneyness coordinates must be finite")
    if implied_volatility_noise_std < 0.0 or not math.isfinite(implied_volatility_noise_std):
        raise ValueError("synthetic Heston IV noise must be finite and non-negative")
    if bid_ask_half_spread is not None and bid_ask_half_spread < 0.0:
        raise ValueError("synthetic bid/ask half-spread must be non-negative")
    rng = create_rng(seed)
    market = MarketEnvironment(
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        volatility=0.0,
    )
    iv_market = ImpliedVolatilityMarket(
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
    )
    observations = []
    for maturity in sorted(set(maturities)):
        forward = forward_price(spot, maturity, risk_free_rate, dividend_yield)
        for coordinate in sorted(set(log_moneyness)):
            strike = forward * math.exp(coordinate)
            option_type = OptionType.PUT if coordinate < 0.0 else OptionType.CALL
            contract = OptionContract(
                option_type=option_type,
                spot=spot,
                strike=strike,
                maturity_years=maturity,
            )
            heston_price = heston_analytical_price(contract, market, parameters).price
            implied = solve_implied_volatility(
                ImpliedVolatilityObservation(
                    contract=contract,
                    market=iv_market,
                    observed_price=heston_price,
                )
            ).implied_volatility
            if implied is None:
                raise ValueError("synthetic Heston price did not have a finite implied volatility")
            observed_iv = max(implied + float(rng.normal(0.0, implied_volatility_noise_std)), 1e-6)
            observed_price = (
                heston_price
                if implied_volatility_noise_std == 0.0
                else black_scholes_price(
                    contract,
                    MarketEnvironment(
                        risk_free_rate=risk_free_rate,
                        dividend_yield=dividend_yield,
                        volatility=observed_iv,
                    ),
                )
            )
            if bid_ask_half_spread is None:
                bid = ask = None
            else:
                bid = max(observed_price - bid_ask_half_spread, 0.0)
                ask = observed_price + bid_ask_half_spread
            observations.append(
                HestonCalibrationObservation(
                    strike=strike,
                    maturity_years=maturity,
                    option_type=option_type,
                    observed_price=observed_price,
                    observed_implied_volatility=observed_iv,
                    bid=bid,
                    ask=ask,
                    mid=observed_price,
                    source="SYNTHETIC_HESTON",
                    weight=1.0,
                    quote_is_fresh=True,
                    arbitrage_valid=True,
                )
            )
    return HestonCalibrationDataset(
        spot=spot,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        observations=observations,
        underlying_symbol="SYNTHETIC",
        metadata={
            "generator": "PHASE13_HESTON_ANALYTICAL",
            "true_parameters": parameters.model_dump(),
            "noise_standard_deviation": implied_volatility_noise_std,
        },
    )
