"""Known-model resource heuristic with causal masks and delayed remedy telemetry."""

import itertools
import numpy as np
from adaptive_planning.remedy_world import ACTION_COSTS

STATES = list(itertools.product((0, 1), repeat=3))


def trans(pi, phi):
    p11 = phi + (1 - phi) * pi
    p01 = (1 - phi) * pi
    return np.array([[1 - p01, p01], [1 - p11, p11]])


class BayesReference:
    def __init__(self, world, actions=tuple(ACTION_COSTS), delay=2):
        sp = world.spec
        self.world, self.actions, self.delay = world, actions, delay
        ts = [trans(*sp.repairable), trans(*sp.common), trans(*sp.proof)]
        self.T = np.array(
            [[np.prod([ts[k][s[k], u[k]] for k in range(3)]) for u in STATES] for s in STATES]
        )
        stat = [sp.repairable[0], sp.common[0], sp.proof[0]]
        self.initial = np.array(
            [np.prod([stat[k] if s[k] else 1 - stat[k] for k in range(3)]) for s in STATES]
        )
        self.belief = self.initial.copy()
        self.costs = np.array([ACTION_COSTS[a] for a in actions])
        self.budget, self.T_h = world.budget, sp.horizon
        self.bS, self.bU = sp.backup_failure, sp.full_update_failure
        self.spent, self.t, self.hist = 0.0, 0, []
        self.records, self.posts, self.pending = [], [], []

    def hits(self, s, members, t):
        ar, ac, am = self.world.groups["schedule"][t]
        return (
            bool(s[0]) and not set(members).isdisjoint(ar),
            bool(s[1]) and not set(members).isdisjoint(ac),
            bool(s[2]) and not set(members).isdisjoint(am),
        )

    def accept_prob(self, a, hr, hc, m):
        e = hr or hc
        ps = 1.0 if hc else (self.bS if hr else 0.0)
        lane = 1.0 if not e else ((1 - ps) if a[0] == "R" else 0.0)
        proof = (1 - (self.bU if m else 0.0)) if a[-1] == "B" else float(not m)
        return lane * proof

    def _posterior(self, prior, index):
        members, action, seen = self.records[index]
        post = prior.copy()
        e, s, m, u = seen
        for i, state in enumerate(STATES):
            hr, hc, proof = self.hits(state, members, index)
            like = float((hr or hc) == e) if e is not None else 1.0
            if e is True and s is not None:
                ps = 1.0 if hc else (self.bS if hr else 0.0)
                like *= ps if s else 1 - ps
            if m is not None:
                like *= float(proof == m)
            if u is not None:
                pu = self.bU if proof else 0.0
                like *= pu if u else 1 - pu
            post[i] *= like
        if post.sum() <= 0:
            raise ValueError("Observed labels inconsistent with known model")
        return post / post.sum()

    def _release(self, step):
        due = [r for r in self.pending if r[0] <= step]
        self.pending = [r for r in self.pending if r[0] > step]
        if not due:
            return
        for _, index, s, u in due:
            self.records[index][2][1] = s
            self.records[index][2][3] = u
        for index in range(min(r[1] for r in due), len(self.records)):
            prior = self.initial if index == 0 else self.posts[index - 1] @ self.T
            self.posts[index] = self._posterior(prior, index)
        self.belief = self.posts[-1]

    def choose(self, members):
        self._release(self.t)
        self.prior = self.belief @ self.T if self.t else self.initial
        v = np.zeros(len(self.actions))
        for i, state in enumerate(STATES):
            hr, hc, m = self.hits(state, members, self.t)
            for k, action in enumerate(self.actions):
                v[k] += self.prior[i] * self.accept_prob(action, hr, hc, m)
        self.hist.append(v)
        left = self.T_h - self.t
        target = (self.budget - self.spent) / left
        history = np.array(self.hist)

        def spend(price):
            return float(np.mean(self.costs[np.argmax(history - price * self.costs, axis=1)]))

        price = 0.0
        if spend(0.0) > target:
            low, high = 0.0, 4.0
            for _ in range(40):
                mid = (low + high) / 2
                if spend(mid) > target:
                    low = mid
                else:
                    high = mid
            price = high
        floor = (left - 1) * min(self.costs)
        feasible = [
            k for k in range(len(self.actions)) if self.costs[k] <= self.budget - self.spent - floor
        ]
        k = max(feasible, key=lambda k: v[k] - price * self.costs[k])
        self.spent += self.costs[k]
        return self.actions[k]

    def observe(self, members, action, E, S, M, U):
        self.records.append([tuple(members), action, [E, None, M, None]])
        self.posts.append(self._posterior(self.prior, self.t))
        self.belief = self.posts[-1]
        self.pending.append((self.t + self.delay, self.t, S, U))
        self._release(self.t)
        self.t += 1
