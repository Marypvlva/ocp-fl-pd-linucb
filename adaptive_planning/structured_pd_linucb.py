"""Experimental shared linear model for placement and verification effects.

Uses the same credit charge, dual update and design-now/reward-later convention
as the article adapter. No regret guarantee for this extension is asserted.
"""

import math

import numpy as np

from .pd_linucb import CreditBudget, decimal_number


def action_features(context, action):
    """Common reward + incremental replication effect + incremental B effect."""
    x = np.asarray(context, dtype=float)
    return np.concatenate((x, x * action.startswith("R"), x * action.endswith("B")))


class SharedPrimalDualLinUCB:
    """One ridge model on [x, replication*x, full_recalculation*x].

    Additivity is an explicit modeling assumption, not a protocol guarantee.
    Only the actual selected action contributes a reward label.
    """

    def __init__(self, dimension, costs, budget, horizon, alpha=1.0, ridge=1.0, eta=0.05):
        if type(dimension) is not int or not 1 <= dimension <= 128:
            raise ValueError("Invalid shared context dimension")
        values = [
            float(decimal_number(v, n))
            for v, n in ((alpha, "alpha"), (ridge, "ridge"), (eta, "eta"))
        ]
        if values[0] < 0 or values[1] <= 0 or values[2] < 0:
            raise ValueError("Invalid shared model parameters")
        self.alpha, self.ridge, self.eta = values
        self.ledger = CreditBudget(costs, budget, horizon)
        if not set(costs) <= {"F+B", "F+P", "R+B", "R+P"}:
            raise ValueError("Expected execution actions")
        self.dimension = dimension
        self.actions = tuple(costs)
        self.A_inv = np.eye(3 * dimension) / self.ridge
        self.b = np.zeros(3 * dimension)
        self.observed = np.zeros(len(costs), dtype=int)
        self.dual = 0.0
        self.pending = {}
        self.last_decision = {}

    def select(self, context, *, forced_action=None):
        feasible = self.ledger.feasible_actions()
        if not feasible:
            return None
        x = np.array(context, dtype=float, copy=True)
        if x.shape != (self.dimension,) or not np.isfinite(x).all():
            raise ValueError("Invalid shared context")
        if forced_action is not None and forced_action not in feasible:
            raise ValueError("Unaffordable or unknown forced action")
        features = np.array([action_features(x, action) for action in self.actions])
        vectors = features @ self.A_inv
        variance = np.sum(vectors * features, axis=1)
        if not np.isfinite(variance).all() or np.any(variance < -1e-10):
            raise ValueError("Invalid shared confidence geometry")
        means = features @ (self.A_inv @ self.b)
        radii = self.alpha * np.sqrt(np.maximum(0, variance))
        costs = np.array([float(self.ledger.costs[a]) for a in self.actions])
        scores = means + radii - self.dual * costs
        if not np.isfinite(scores).all():
            raise ValueError("Invalid shared scores")
        index = (
            self.actions.index(forced_action)
            if forced_action
            else int(np.argmax(np.where([a in feasible for a in self.actions], scores, -np.inf)))
        )
        action = self.actions[index]
        chosen = features[index]
        vector = vectors[index]
        denominator = 1 + chosen @ vector
        updated = self.A_inv - np.outer(vector, vector) / denominator
        target = self.ledger.target_rate()
        dual = max(0.0, self.dual + self.eta * (costs[index] - target))
        if denominator <= 0 or not np.isfinite(updated).all() or not math.isfinite(dual):
            raise ValueError("Invalid shared update")
        identity = self.ledger.steps + 1
        self.ledger.commit(action)
        self.A_inv = updated
        self.pending[identity] = (index, chosen.copy())
        self.last_decision = dict(
            predicted_reward=means.tolist(),
            confidence_bonus=radii.tolist(),
            dual_before=self.dual,
            dual_after=dual,
            target_cost_rate=target,
        )
        self.dual = dual
        return action

    def reward(self, decision_id, value):
        if type(decision_id) is not int or decision_id not in self.pending:
            raise ValueError("Unknown or duplicate shared reward")
        reward = float(decimal_number(value, "reward"))
        if not 0 <= reward <= 1:
            raise ValueError("Shared reward must be in [0,1]")
        index, feature = self.pending[decision_id]
        updated = self.b + reward * feature
        if not np.isfinite(updated).all():
            raise ValueError("Invalid shared reward vector")
        self.b = updated
        self.observed[index] += 1
        del self.pending[decision_id]
