"""Literal disjoint PD-LinUCB engine with collector credits and external site guards."""

import numpy as np
from adaptive_planning.pd_linucb import PrimalDualLinUCB
from model import EXTRA, USE


class ArticlePolicy:
    def __init__(
        self,
        settings,
        runs,
        population,
        k,
        horizon,
        budget,
        seed=53,
        gateway_caps=None,
        fixed_prices=None,
    ):
        if runs != 1 or gateway_caps is None:
            raise ValueError("Article adapter is single-episode")
        self.s = settings
        self.n = population
        self.k = k
        self.h = horizon
        self.budget = budget
        self.spent = np.zeros(1, dtype=int)
        self.gateway_spent = np.zeros((1, 4), dtype=int)
        self.caps = np.asarray(gateway_caps)
        self.t = 0
        self.pending = []
        self.engine = PrimalDualLinUCB(
            1 + population + 3,
            {str(a): 1 + int(EXTRA[a]) for a in range(10)},
            horizon + budget,
            horizon,
            settings.alpha,
            settings.ridge,
            settings.eta,
        )
        self.engine.dual = settings.initial_price

    def context(self, c, roster):
        x = np.zeros(1 + self.n + 3)
        x[0] = 1
        x[1 + roster[0]] = 1 / np.sqrt(self.k)
        x[1 + self.n + c] = 1
        return x

    def decide(self, c, roster, forced_arm=None):
        while self.pending and self.pending[0][0] <= self.t:
            _, ident, value = self.pending.pop(0)
            self.engine.reward(ident, value)
        feasible = np.all(
            self.gateway_spent[:, None, :] + USE[None, :, :] <= self.caps[:, None, :], axis=2
        )[0]
        feasible &= (~EXTRA) | (self.spent[0] < self.budget)
        action = self.engine.select(
            self.context(c, roster),
            allowed_actions=[str(i) for i in np.flatnonzero(feasible)],
            forced_action=str(forced_arm) if forced_arm is not None else None,
        )
        if action is None:
            raise ValueError("No feasible article-engine action")
        arm = int(action)
        self.gateway_spent += USE[arm]
        self.spent += EXTRA[arm]
        d = self.engine.last_decision
        self.last_decision = dict(
            step=self.t + 1,
            window_class=int(c),
            roster=roster.tolist(),
            arm=[arm],
            predictions=[d["ucb"]],
            feasible_arms=[feasible.tolist()],
            penalized_scores=[
                [s if feasible[i] else None for i, s in enumerate(d["penalized_scores"])]
            ],
            effective_mirror_price=[d["dual_before"]],
            effective_gateway_prices=[[0.0] * 4],
            price_before=[d["dual_before"]],
            price_after=[d["dual_after"]],
            gateway_prices_before=[[0.0] * 4],
            gateway_prices_after=[[0.0] * 4],
            mirror_spent=self.spent.tolist(),
            gateway_spent=self.gateway_spent.tolist(),
            path_mean_prediction=None,
            path_pooled_prediction=None,
            chosen_probability=None,
            pending_labels=len(self.pending),
            article_engine=d,
            cost_scope="1 base credit + 1 mirror; budget=H+M. Site quotas are external hard guards; no site dual. Application bytes guarded independently.",
        )
        self.last_id = d["decision_id"]
        return np.array([arm])

    def observe(self, readings, reward, attempted=None):
        self.pending.append((self.t + self.s.delay, self.last_id, float(reward[0])))
        self.t += 1
