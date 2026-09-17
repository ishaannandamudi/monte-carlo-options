"""Vectorized vanilla and exotic payoff functions."""

from backend.app.payoffs.asian import asian_average, asian_payoff, asian_payoff_from_average
from backend.app.payoffs.barrier import (
    BarrierEventEvaluation,
    barrier_breaches,
    barrier_vanilla_payoff,
    bridge_discounted_barrier_payoffs,
    brownian_bridge_crossing_probabilities,
    brownian_bridge_survival_probabilities,
    discrete_discounted_barrier_payoffs,
    evaluate_discrete_barrier,
)
from backend.app.payoffs.basket import basket_payoff, basket_payoff_from_values, basket_values
from backend.app.payoffs.digital import digital_itm, digital_payoff
from backend.app.payoffs.vanilla import call_payoff, intrinsic_value, put_payoff, vanilla_payoff

__all__ = [
    "asian_average",
    "asian_payoff",
    "asian_payoff_from_average",
    "BarrierEventEvaluation",
    "barrier_breaches",
    "barrier_vanilla_payoff",
    "basket_payoff",
    "basket_payoff_from_values",
    "basket_values",
    "bridge_discounted_barrier_payoffs",
    "brownian_bridge_crossing_probabilities",
    "brownian_bridge_survival_probabilities",
    "call_payoff",
    "digital_itm",
    "digital_payoff",
    "discrete_discounted_barrier_payoffs",
    "evaluate_discrete_barrier",
    "intrinsic_value",
    "put_payoff",
    "vanilla_payoff",
]
