"""Strong non-bandit controls for budgeted remedy selection on certified rosters.

PlugInRemedyControl generalizes the reserve-benefit control of the 17 Nov study
to several remedies. It fits ridge models of the observable fault indicators
on the same certificate-attributed features as the PD-LinUCB planner and
evaluates each action's expected gain over F+P under the known acceptance rule
(assuming conditional independence of indicators given the features). Pacing
is either the quantile cutoff of the original control (``pacing='quantile'``)
or the dual price of PD-LinUCB (``pacing='dual'``).

ReactiveRemedyControl applies the remedy matching the previous round's observed
indicators. Both keep the cheapest action funded for every later round.
"""

import heapq
import math
from types import SimpleNamespace

import numpy as np

from .attributed_pd_linucb import CertificateEvidence, CertificateFilter, FundedBudget, observable


def _ridge(design, labels, prior=0.0, ridge=1.0):
    d = design.shape[1]
    target = design.T @ labels if len(labels) else np.zeros(d)
    target = np.array(target, dtype=float)
    target[0] += prior * ridge
    return np.linalg.solve(ridge * np.eye(d) + design.T @ design, target)


class _Base:
    def __init__(
        self,
        *,
        budget,
        horizon,
        population,
        costs,
        actions,
        feedback_delay=0,
        decays=(0.5, 0.9),
        attributed=True,
        attribution="evidence",
        secondary_costs=None,
        secondary_budget=None,
    ):
        self.actions = tuple(actions)
        self.costs = {a: float(costs[a]) for a in self.actions}
        self.ledger = FundedBudget(self.costs, budget, horizon)
        self.secondary = None
        if secondary_costs is not None:
            self.secondary = FundedBudget(
                {a: float(secondary_costs[a]) for a in self.actions}, secondary_budget, horizon
            )
            self.ledger.extra_filter = self.secondary.feasible_actions
        self.population = population
        self.attributed = attributed
        self.evidence = (
            CertificateEvidence(population, decays)
            if attribution == "evidence"
            else CertificateFilter(population, decays)
        )
        self.delay = feedback_delay
        self.queue = []
        self.records = {}
        self.active = None
        self.ready = False
        self.previous_observed = {"E": False, "S": None, "M": False, "U": None}
        self.last_decision = {}
        self.last_reward = None
        self.engine = SimpleNamespace(pending={})
        self.feedback = 0

    def feedback_count(self):
        return self.feedback

    @property
    def stats(self):
        state = dict(
            budget_credits=float(self.ledger.budget),
            spent_credits=float(self.ledger.spent),
            remaining_credits=float(self.ledger.remaining),
            charged_actions=self.ledger.steps,
            queued_labels=len(self.queue),
        )
        if self.secondary is not None:
            state.update(secondary_spent=float(self.secondary.spent))
        return state

    def features(self, selected):
        membership = np.zeros(self.population)
        membership[list(selected)] = 1.0
        prev = self.previous_observed
        parts = [
            np.array([1.0, float(bool(prev["E"])), float(bool(prev["M"]))]),
            (membership - len(selected) / self.population) / math.sqrt(len(selected)),
        ]
        if self.attributed:
            parts.append(self.evidence.features(selected))
        return np.concatenate(parts)

    def begin_round(self, observation, capture_attempt, attempt):
        if attempt != self.ledger.steps + 1 or self.active is not None or self.ready:
            raise ValueError("Noncausal or unfinished decision")
        self.ready = True
        if attempt > 1:
            self.decay()
        while self.queue and self.queue[0][0] <= attempt:
            _, identity = heapq.heappop(self.queue)
            self.learn(*self.records.pop(identity))
            self.feedback += 1

    def decay(self):
        pass

    def learn(self, context, action, observed, success):
        pass

    def observe_round(
        self,
        action,
        success,
        cycle_seconds,
        switch_wall_seconds,
        *,
        primary_edge_error=None,
        backup_edge_error=None,
        projection_missing=None,
        full_update_missing=None,
        stage_observations=None,
    ):
        identity, _, selected, context = self.active
        seen = observable(action)
        observed = {
            "E": primary_edge_error,
            "S": backup_edge_error if "S" in seen else None,
            "M": projection_missing if "M" in seen else None,
            "U": full_update_missing if "U" in seen else None,
        }
        self.evidence.round_finished(selected, observed)
        self.previous_observed = observed
        self.records[identity] = (context, action, observed, bool(success))
        heapq.heappush(self.queue, (identity + self.delay, identity))
        while self.queue and self.queue[0][0] <= identity:
            _, i = heapq.heappop(self.queue)
            self.learn(*self.records.pop(i))
            self.feedback += 1
        self.last_reward = float(bool(success))
        self.active = None

    def commit(self, action, selected, context):
        self.ledger.commit(action)
        if self.secondary is not None:
            self.secondary.commit(action)
        self.ready = False
        self.active = (self.ledger.steps, action, tuple(selected), context)
        return action

    @property
    def spare_rate(self):
        left = self.ledger.horizon - self.ledger.steps
        spare = float(self.ledger.remaining) - left * min(self.costs.values())
        return max(0.0, spare) / left


class PlugInRemedyControl(_Base):
    def __init__(self, *, pacing="quantile", eta=0.05, forget=1.0, **kw):
        super().__init__(**kw)
        if pacing not in ("quantile", "dual"):
            raise ValueError("Unknown pacing")
        if not 0 < float(forget) <= 1:
            raise ValueError("Forgetting factor must lie in (0, 1]")
        # Same discounting of sufficient statistics as the factored PD-LinUCB engine.
        self.forget = float(forget)
        self.pacing = pacing
        self.eta = eta
        self.dual = 0.0
        self.dual2 = 0.0
        self.suff = {}
        self.history = []
        self.labels = 0

    def _add(self, name, x, y):
        d = x.size
        A, b, n = self.suff.get(name, (np.eye(d), np.zeros(d), 0))
        self.suff[name] = (A + np.outer(x, x), b + y * x, n + 1)

    def decay(self):
        if self.forget >= 1.0:
            return
        for name, (A, b, n) in self.suff.items():
            eye = np.eye(A.shape[0])
            self.suff[name] = (eye + self.forget * (A - eye), self.forget * b, n)

    def _predict(self, name, x, prior):
        if name not in self.suff:
            return prior
        A, b, n = self.suff[name]
        target = b.copy()
        target[0] += prior
        return float(np.clip(x @ np.linalg.solve(A, target), 0, 1))

    def learn(self, context, action, observed, success):
        self.labels += 1
        if observed["E"] is not None:
            self._add("E", context, float(observed["E"]))
        if observed["M"] is not None:
            self._add("M", context, float(observed["M"]))
        if observed["S"] is not None and observed["E"]:
            self._add("S", context, float(observed["S"]))
        if observed["U"] is not None:
            self._add("U", context, float(observed["U"]))

    def gains(self, x):
        pE = self._predict("E", x, 0.0)
        pM = self._predict("M", x, 0.0)
        qS = self._predict("S", x, 0.5)
        pU = self._predict("U", x, 0.0)
        value = {}
        for a in self.actions:
            lane = 1 - pE * qS if a[0] == "R" else 1 - pE
            proof = 1 - pU if a[-1] == "B" else 1 - pM
            value[a] = lane * proof
        g = {a: value[a] - value["F+P"] for a in self.actions}
        return g, dict(pE=pE, pM=pM, qS=qS, pU=pU)

    def choose(self, observation, previous, *, selected=None):
        x = self.features(selected)
        feasible = self.ledger.feasible_actions()
        g, est = self.gains(x)
        base = min(self.costs.values())
        if self.pacing == "dual":
            c2 = self.secondary.costs if self.secondary is not None else None
            score = {
                a: g[a] - self.dual * self.costs[a] - (self.dual2 * float(c2[a]) if c2 else 0.0)
                for a in feasible
            }
            action = max(score, key=score.get)
            target = self.ledger.target_rate()
            if c2:
                target2 = self.secondary.target_rate()
                self.dual2 = max(0.0, self.dual2 + self.eta * (float(c2[action]) - target2))
            self.dual = max(0.0, self.dual + self.eta * (self.costs[action] - target))
        else:
            if self.secondary is not None:
                # Linear-schedule pacing of the second resource: an action is
                # allowed only if cumulative use stays on the proportional schedule.
                step = self.ledger.steps + 1
                share = float(self.secondary.budget) * step / self.ledger.horizon
                used = float(self.secondary.spent)
                feasible = tuple(
                    a
                    for a in feasible
                    if a == "F+P" or used + float(self.secondary.costs[a]) <= share
                ) or ("F+P",)
            eff = {a: g[a] / (self.costs[a] - base) for a in feasible if self.costs[a] > base}
            best = max(eff, key=eff.get) if eff else None
            score = eff[best] if best else 0.0
            cutoff = (
                float(np.quantile(self.history, 1 - min(1.0, self.spare_rate)))
                if self.history
                else 0.0
            )
            self.history.append(score)
            action = best if best is not None and score > cutoff else "F+P"
        self.last_decision = dict(estimates=est, gains=g)
        return self.commit(action, selected, x)


class ReactiveRemedyControl(_Base):
    def choose(self, observation, previous, *, selected=None):
        x = self.features(selected)
        feasible = self.ledger.feasible_actions()
        prev = self.previous_observed
        want_r = bool(prev["E"]) and not bool(prev["S"])
        want_b = bool(prev["M"])
        wish = ("R" if want_r else "F") + "+" + ("B" if want_b else "P")
        order = [wish, "R+P" if want_r else None, "F+B" if want_b else None, "F+P"]
        action = next(a for a in order if a and a in feasible and a in self.actions)
        return self.commit(action, selected, x)
