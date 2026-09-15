"""Domain-specific failures surfaced by numerical services."""


class QuantitativeError(Exception):
    """Base class for quantitative-domain errors."""


class NumericalError(QuantitativeError):
    """Raised when valid model inputs exceed safe numerical representation."""


class StatisticalError(QuantitativeError):
    """Raised when an estimator cannot be formed from the supplied observations."""
