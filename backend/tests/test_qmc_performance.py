import pytest

from backend.app.models.asian import AsianOptionContract, AveragingType
from backend.app.models.basket import BasketOptionContract, BasketType
from backend.app.models.market import MarketEnvironment
from backend.app.models.multi_asset import MultiAssetMarketEnvironment
from backend.app.models.option import OptionContract, OptionType
from backend.app.models.qmc import (
    QMCAsianRequest,
    QMCBasketRequest,
    QMCConfig,
    QMCEuropeanRequest,
    SamplingMethod,
)
from backend.app.pricing.qmc import price_asian_qmc, price_basket_qmc, price_european_qmc


@pytest.mark.performance
def test_european_rqmc_262k_samples() -> None:
    result = price_european_qmc(
        QMCEuropeanRequest(
            contract=OptionContract(
                option_type=OptionType.CALL,
                spot=100.0,
                strike=100.0,
                maturity_years=1.0,
            ),
            market=MarketEnvironment(risk_free_rate=0.03, volatility=0.2),
            qmc=QMCConfig(
                method=SamplingMethod.SCRAMBLED_SOBOL,
                dimension=1,
                num_samples=262_144,
                scramble=True,
                num_randomizations=8,
            ),
        )
    )
    assert result.runtime_ms >= 0.0


@pytest.mark.performance
def test_asian_bridge_rqmc_32k_by_52() -> None:
    result = price_asian_qmc(
        QMCAsianRequest(
            contract=AsianOptionContract(
                option_type=OptionType.CALL,
                spot=100.0,
                strike=100.0,
                maturity_years=1.0,
                averaging_type=AveragingType.GEOMETRIC,
                averaging_schedule={"monitoring_count": 52},
            ),
            market=MarketEnvironment(risk_free_rate=0.03, volatility=0.2),
            qmc=QMCConfig(
                method=SamplingMethod.SCRAMBLED_SOBOL,
                dimension=52,
                num_samples=32_768,
                scramble=True,
                num_randomizations=4,
                path_construction="BROWNIAN_BRIDGE",
            ),
        )
    )
    assert result.runtime_ms >= 0.0


@pytest.mark.performance
def test_basket_rqmc_131k_by_10() -> None:
    assets = 10
    market = MultiAssetMarketEnvironment(
        asset_ids=tuple(f"A{i}" for i in range(assets)),
        spots=(100.0,) * assets,
        volatilities=(0.2,) * assets,
        risk_free_rate=0.03,
        dividend_yields=(0.0,) * assets,
        correlation_matrix=tuple(
            tuple(1.0 if i == j else 0.2 for j in range(assets)) for i in range(assets)
        ),
    )
    result = price_basket_qmc(
        QMCBasketRequest(
            contract=BasketOptionContract(
                option_type=OptionType.CALL,
                strike=100.0,
                maturity_years=1.0,
                weights=(0.1,) * assets,
                basket_type=BasketType.GEOMETRIC,
            ),
            market=market,
            qmc=QMCConfig(
                method=SamplingMethod.SCRAMBLED_SOBOL,
                dimension=assets,
                num_samples=131_072,
                scramble=True,
                num_randomizations=4,
            ),
        )
    )
    assert result.runtime_ms >= 0.0
