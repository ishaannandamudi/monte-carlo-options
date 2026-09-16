"""Vectorized vanilla and exotic payoff functions."""

from backend.app.payoffs.asian import asian_average, asian_payoff, asian_payoff_from_average
from backend.app.payoffs.digital import digital_itm, digital_payoff
from backend.app.payoffs.vanilla import call_payoff, intrinsic_value, put_payoff, vanilla_payoff

__all__ = [
    "asian_average",
    "asian_payoff",
    "asian_payoff_from_average",
    "call_payoff",
    "digital_itm",
    "digital_payoff",
    "intrinsic_value",
    "put_payoff",
    "vanilla_payoff",
]
