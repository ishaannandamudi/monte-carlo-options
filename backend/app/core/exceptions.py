"""Domain-specific failures surfaced by numerical services."""


class QuantitativeError(Exception):
    """Base class for quantitative-domain errors."""


class NumericalError(QuantitativeError):
    """Raised when valid model inputs exceed safe numerical representation."""


class LongstaffSchwartzError(QuantitativeError):
    """Raised for invalid paths, policies, or LSM experiments."""


class ExoticPricingError(QuantitativeError):
    """Raised for invalid Asian/digital pricing experiments or method combinations."""


class BarrierPricingError(ExoticPricingError):
    """Raised for unsupported or numerically invalid barrier pricing operations."""


class MultiAssetPricingError(ExoticPricingError):
    """Raised for invalid correlation, simulation, or basket pricing operations."""


class HestonPricingError(QuantitativeError):
    """Raised for invalid or numerically unsafe Heston operations."""


class MertonPricingError(QuantitativeError):
    """Raised for invalid or numerically unsafe Merton jump-diffusion operations."""


class HestonCalibrationError(HestonPricingError):
    """Raised for invalid datasets or unsuccessful Heston calibration operations."""


class StatisticalError(QuantitativeError):
    """Raised when an estimator cannot be formed from the supplied observations."""


class UnsupportedGreekMethodError(QuantitativeError):
    """Raised when a Greek is unavailable for the requested estimator configuration."""


class SensitivityGridError(QuantitativeError):
    """Raised when a sensitivity grid contains an invalid model value."""


class PortfolioRiskError(QuantitativeError):
    """Raised for invalid portfolio dispatch, scenario, or aggregation operations."""


class ImpliedVolatilityError(QuantitativeError):
    """Base class for an implied-volatility inversion failure."""


class ArbitrageBoundsError(ImpliedVolatilityError):
    """Raised when an observed price violates European no-arbitrage bounds."""


class ImpliedVolatilityBracketError(ImpliedVolatilityError):
    """Raised when no finite root exists in the configured volatility range."""


class ImpliedVolatilityConvergenceError(ImpliedVolatilityError):
    """Raised when a valid bracketed or Newton solve does not converge."""


class MarketDataError(QuantitativeError):
    """Base class for normalized market-data and provider failures."""


class MarketDataValidationError(MarketDataError):
    """Raised when normalized market data cannot support requested analytics."""


class VolatilitySurfaceError(QuantitativeError):
    """Raised when a smile or surface cannot be constructed or queried safely."""


class BinomialTreeError(QuantitativeError):
    """Base class for invalid or numerically unsafe binomial trees."""


class InvalidRiskNeutralProbabilityError(BinomialTreeError):
    """Raised when CRR inputs do not admit a risk-neutral probability."""


class UnsupportedExerciseStyleError(BinomialTreeError):
    """Raised when a pricing or analytics engine cannot honor the exercise style."""


class ProviderError(MarketDataError):
    """Base class for translated external-provider failures."""


class ProviderAuthenticationError(ProviderError):
    """Raised for missing or rejected provider credentials."""


class ProviderRateLimitError(ProviderError):
    """Raised when the provider rejects a request due to rate limits."""


class ProviderSymbolNotFoundError(ProviderError):
    """Raised when the requested market symbol is unavailable."""


class ProviderMalformedResponseError(ProviderError):
    """Raised when a provider response cannot be normalized safely."""


class ProviderTimeoutError(ProviderError):
    """Raised when an external provider request times out."""


class ProviderNetworkError(ProviderError):
    """Raised for other provider transport failures."""


class ProviderServerError(ProviderError):
    """Raised when the provider reports a server-side failure."""
