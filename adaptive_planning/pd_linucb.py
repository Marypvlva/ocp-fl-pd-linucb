"""PD-LinUCB from Pavlova's article, section 4: design now, reward later.

This module implements a single known-cost resource, not a bound on unobserved
wall-clock time or wire bytes. Ridge regularization and the dual price are
different quantities. No reward clipping is applied to the UCB score.
"""

from decimal import Decimal
from fractions import Fraction
from collections.abc import Mapping
import math

import numpy as np


def decimal_number(value, name):
    if isinstance(value, (bool, str)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = Decimal(str(value))
    except (ValueError, TypeError, ArithmeticError) as error:
        raise ValueError(f"Invalid {name}") from error
    if not result.is_finite() or not math.isfinite(float(result)):
        raise ValueError(f"Invalid {name}")
    return result


class CreditBudget:
    """Exact decimal accounting for known action charges, before execution."""

    def __init__(self, costs, budget, horizon):
        if type(horizon) is not int or not 1 <= horizon <= 1_000_000:
            raise ValueError("Horizon must be an integer in 1..1000000")
        if (
            not isinstance(costs, Mapping)
            or not costs
            or any(not isinstance(a, str) or not a for a in costs)
        ):
            raise ValueError("Named actions and costs are required")
        self.costs = {a: Fraction(decimal_number(c, "action cost")) for a, c in costs.items()}
        self.budget = Fraction(decimal_number(budget, "budget"))
        if self.budget <= 0 or any(c < 0 for c in self.costs.values()):
            raise ValueError("Budget must be positive and costs nonnegative")
        self.horizon = horizon
        self.spent = Fraction(0)
        self.steps = 0

    @property
    def remaining(self):
        return self.budget - self.spent

    def feasible_actions(self):
        if self.steps >= self.horizon:
            return ()
        return tuple(a for a, c in self.costs.items() if c <= self.remaining)

    def stop_reason(self):
        if self.steps >= self.horizon:
            return "horizon_exhausted"
        if not self.feasible_actions():
            return "budget_exhausted"
        return None

    def target_rate(self):
        if self.steps >= self.horizon:
            raise ValueError("Horizon exhausted")
        return float(self.remaining / (self.horizon - self.steps))

    def commit(self, action):
        if action not in self.feasible_actions():
            raise ValueError("Action exceeds remaining budget or horizon")
        self.spent += self.costs[action]
        self.steps += 1

    def state(self):
        return dict(
            budget_credits=float(self.budget),
            spent_credits=float(self.spent),
            remaining_credits=float(self.remaining),
            charged_actions=self.steps,
        )


class PrimalDualLinUCB:
    """Disjoint linear UCB, immediate geometry/charge, delayed labels.

    Score = theta_a @ x + alpha * sqrt(x @ A_inv[a] @ x) - dual * c(a).
    Target = remaining_budget_before_choice / remaining_horizon_including_choice.
    The first maximizer in configured action order wins, as in numpy.argmax.
    eta=0 is the LinUCB ablation with the identical hard feasible set.
    """

    def __init__(self, dimension, costs, budget, horizon, alpha=1.0, ridge=1.0, eta=0.05):
        if type(dimension) is not int or not 1 <= dimension <= 1024:
            raise ValueError("Invalid context dimension")
        values = [
            decimal_number(v, name)
            for v, name in ((alpha, "alpha"), (ridge, "ridge"), (eta, "eta"))
        ]
        if values[0] < 0 or values[1] <= 0 or values[2] < 0:
            raise ValueError("Require alpha>=0, ridge>0, eta>=0")
        self.dimension = dimension
        self.alpha, self.ridge, self.eta = map(float, values)
        self.ledger = CreditBudget(costs, budget, horizon)
        self.actions = tuple(costs)
        self.index = {a: i for i, a in enumerate(self.actions)}
        self.A_inv = np.repeat((np.eye(dimension) / self.ridge)[None, :, :], len(costs), axis=0)
        if not np.all(np.isfinite(self.A_inv)):
            raise ValueError("Invalid ridge geometry")
        self.b = np.zeros((len(costs), dimension))
        self.selected = np.zeros(len(costs), dtype=np.int64)
        self.observed = np.zeros(len(costs), dtype=np.int64)
        self.dual = 0.0
        self.pending = {}
        self.last_decision = {}

    def select(self, context, *, forced_action=None, allowed_actions=None):
        """Charge one action, optionally from a precommitted experimental control."""
        feasible = self.ledger.feasible_actions()
        if allowed_actions is not None:
            allowed = tuple(allowed_actions)
            if len(allowed) != len(set(allowed)) or any(a not in self.index for a in allowed):
                raise ValueError("Unknown or duplicate externally permitted action")
            feasible = tuple(a for a in feasible if a in allowed)
        if not feasible:
            return None
        if forced_action is not None and forced_action not in feasible:
            raise ValueError("Scheduled action exceeds the budget or is unknown")
        x = np.array(context, dtype=np.float64, copy=True)
        if x.shape != (self.dimension,) or not np.all(np.isfinite(x)):
            raise ValueError("Invalid context")
        theta = np.einsum("kij,kj->ki", self.A_inv, self.b)
        vectors = self.A_inv @ x
        variance = np.sum(vectors * x[None, :], axis=1)
        if not np.all(np.isfinite(variance)) or np.any(variance < -1e-10):
            raise ValueError("Invalid confidence geometry")
        mean = theta @ x
        radius = self.alpha * np.sqrt(np.maximum(0.0, variance))
        costs = np.array([float(self.ledger.costs[a]) for a in self.actions])
        ucb = mean + radius
        scores = ucb - self.dual * costs
        if not np.all(np.isfinite(scores)):
            raise ValueError("Nonfinite score")
        mask = np.array([a in feasible for a in self.actions])
        arm = (
            self.index[forced_action]
            if forced_action is not None
            else int(np.argmax(np.where(mask, scores, -np.inf)))
        )
        action = self.actions[arm]
        v = vectors[arm]
        denominator = 1.0 + float(x @ v)
        updated = self.A_inv[arm] - np.outer(v, v) / denominator
        target = self.ledger.target_rate()
        dual_before = self.dual
        dual_after = max(0.0, dual_before + self.eta * (costs[arm] - target))
        if denominator <= 0 or not np.all(np.isfinite(updated)) or not math.isfinite(dual_after):
            raise ValueError("Numerically invalid update")
        # Everything above is checked before committing the action or its charge.
        decision_id = self.ledger.steps + 1
        self.ledger.commit(action)
        self.A_inv[arm] = updated
        self.selected[arm] += 1
        self.pending[decision_id] = (arm, x)
        self.dual = dual_after
        self.last_decision = dict(
            decision_id=decision_id,
            context=x.tolist(),
            feasible_actions=list(feasible),
            predicted_reward=mean.tolist(),
            confidence_bonus=radius.tolist(),
            ucb=ucb.tolist(),
            penalized_scores=scores.tolist(),
            dual_before=dual_before,
            dual_after=dual_after,
            target_cost_rate=target,
            action_cost_credits=costs[arm],
            **self.ledger.state(),
        )
        if forced_action is not None:
            self.last_decision["selection_rule"] = "precommitted_schedule"
        return action

    def reward(self, decision_id, value):
        if type(decision_id) is not int or decision_id not in self.pending:
            raise ValueError("Unknown, future or duplicate reward ID")
        reward = float(decimal_number(value, "reward"))
        if not 0 <= reward <= 1:
            raise ValueError("Reward must lie in [0,1]")
        arm, context = self.pending[decision_id]
        updated = self.b[arm] + reward * context
        if not np.all(np.isfinite(updated)):
            raise ValueError("Nonfinite reward update")
        self.b[arm] = updated
        self.observed[arm] += 1
        del self.pending[decision_id]

    def state(self):
        return dict(
            **self.ledger.state(),
            dual=self.dual,
            selected_by_action=self.selected.tolist(),
            observed_by_action=self.observed.tolist(),
            pending_decision_ids=sorted(self.pending),
            reward_vectors=self.b.tolist(),
            inverse_diagonal=np.diagonal(self.A_inv, axis1=1, axis2=2).tolist(),
        )
