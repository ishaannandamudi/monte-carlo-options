"""Domain-specific failures surfaced by numerical services."""


class QuantitativeError(Exception):
    """Base class for quantitative-domain errors."""


class NumericalError(QuantitativeError):
    """Raised when valid model inputs exceed safe numerical representation."""


class StatisticalError(QuantitativeError):
    """Raised when an estimator cannot be formed from the supplied observations."""


class UnsupportedGreekMethodError(QuantitativeError):
    """Raised when a Greek is unavailable for the requested estimator configuration."""


class SensitivityGridError(QuantitativeError):
    """Raised when a sensitivity grid contains an invalid model value."""


class ImpliedVolatilityError(QuantitativeError):
    """Base class for an implied-volatility inversion failure."""


class ArbitrageBoundsError(ImpliedVolatilityError):
    """Raised when an observed price violates European no-arbitrage bounds."""


class ImpliedVolatilityBracketError(ImpliedVolatilityError):
    """Raised when no finite root exists in the configured volatility range."""


class ImpliedVolatilityConvergenceError(ImpliedVolatilityError):
    """Raised when a valid bracketed or Newton solve does not converge."""
