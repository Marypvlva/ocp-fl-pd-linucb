"""PD-LinUCB on OCP-FL certificates with protocol-implied feedback.

Two extensions of PD-LinUCB (Pavlova, MOTOR 2026) for verifiable federated
rounds whose roster is fixed by OCP-FL (Pavlova et al.):

1. Certificate-attributed context. The OCP-FL certificate fixes the roster
   before execution and is never resampled after a failure, so each observed
   fault indicator can be attributed to the certified members of its round.
   Per-client, exponentially forgotten evidence of each indicator, masked by
   the current roster, is appended to the PD-LinUCB context.

2. Protocol-implied counterfactual feedback. The acceptance rule of the
   verification protocol is a known Boolean function of fault indicators
   (primary aggregation error E, backup-lane error S, missing projection M).
   After a round, every action whose outcome is determined by the indicators
   actually observed receives a label; unobserved indicators are enumerated,
   so no label is imputed unless all completions agree.

The UCB score, the dual price update, the hard credit ledger and the
design-now / reward-later handling of the chosen arm are unchanged from
PD-LinUCB. Labels of non-chosen arms are applied, design and reward together,
when the chosen arm's label matures, so all feedback obeys the same delay.
Only causal information is used.
"""

import heapq
from itertools import product
import math

import numpy as np

from .pd_linucb import CreditBudget, PrimalDualLinUCB, decimal_number
from .pd_linucb_planner import make_context, outcome_reward

INDICATORS = ("E", "S", "M", "U")


def accepted_under(action, E, S, M, U=False):
    """Acceptance rule of the protocol for one action.

    E: primary aggregation lane failed. S: backup lane failed.
    M: a certified worker cannot answer the projection challenge.
    U: a signed full update for recomputation did not arrive.
    Full recomputation (B) needs signed full updates instead of projection
    replies; a reserve (R) replaces the primary lane unless the backup failed.
    """
    lane_ok = (not E) or (action[0] == "R" and not S)
    proof_ok = (not U) if action[-1] == "B" else (not M)
    return lane_ok and proof_ok


def observable(action):
    """Indicators revealed by executing an action."""
    seen = {"E"}
    if action[0] == "R":
        seen.add("S")
    seen.add("U" if action[-1] == "B" else "M")
    return seen


def implied_labels(actions, observed):
    """Labels of all actions that the observed indicators determine.

    observed maps indicator name to bool or None (not observed).
    """
    unknown = [k for k in INDICATORS if observed.get(k) is None]
    labels = {}
    for action in actions:
        outcomes = set()
        for values in product((False, True), repeat=len(unknown)):
            state = dict(observed) | dict(zip(unknown, values))
            outcomes.add(accepted_under(action, state["E"], state["S"], state["M"], state["U"]))
            if len(outcomes) > 1:
                break
        if len(outcomes) == 1:
            labels[action] = float(outcomes.pop())
    return labels


class CertificateEvidence:
    """Per-client evidence of fault indicators attributed through certificates."""

    def __init__(self, population, decays=(0.5, 0.9), indicators=INDICATORS):
        if type(population) is not int or not 2 <= population <= 256:
            raise ValueError("Invalid population")
        decays = tuple(float(d) for d in decays)
        if not decays or any(not 0 <= d < 1 for d in decays):
            raise ValueError("Decays must lie in [0,1)")
        self.population = population
        self.indicators = tuple(indicators)
        self.decays = np.array(decays)
        self.values = np.zeros((len(self.indicators), len(decays), population))

    def round_finished(self, selected, observed):
        self.values *= self.decays[None, :, None]
        members = list(selected)
        for k, name in enumerate(self.indicators):
            value = observed.get(name)
            if value is not None:
                self.values[k][:, members] = float(bool(value))

    def features(self, selected):
        membership = np.zeros(self.population)
        membership[list(selected)] = 1.0
        return (self.values * membership[None, None, :]).ravel()

    @property
    def dimension(self):
        return self.values.size


class CertificateFilter:
    """Per-client fault beliefs filtered through certified rosters (noisy-OR).

    For each indicator k and persistence phi, client i has a belief b[k,phi,i]
    that its fault state for k is active. Between rounds the belief relaxes to
    the client's estimated base rate: b <- phi * b + (1 - phi) * base. When k is
    observed for a certified roster S, the indicator is modelled as a leaky
    noisy-OR of the members' states: if it is absent, members' beliefs drop to
    zero; if present, each member's belief is divided by the predicted
    probability of the indicator (assumed-density update of the marginals).
    Base rates are running means of members' posterior beliefs.
    Features: the noisy-OR prediction for the current roster and the
    roster-masked member beliefs, for every indicator and persistence.
    """

    def __init__(self, population, persistences=(0.5, 0.9), indicators=INDICATORS, leak=0.01):
        self.population = population
        self.indicators = tuple(indicators)
        self.phis = np.array([float(p) for p in persistences])
        if not len(self.phis) or np.any((self.phis < 0) | (self.phis >= 1)):
            raise ValueError("Persistences must lie in [0,1)")
        self.leak = float(leak)
        shape = (len(self.indicators), len(self.phis), population)
        self.belief = np.full(shape, 0.05)
        self.mass = np.full((len(self.indicators), population), 0.1)
        self.exposure = np.full((len(self.indicators), population), 1.0)

    @property
    def base(self):
        return self.mass / self.exposure

    def predicted(self):
        base = self.base[:, None, :]
        return self.phis[None, :, None] * self.belief + (1 - self.phis[None, :, None]) * base

    def round_finished(self, selected, observed):
        prior = self.predicted()
        members = list(selected)
        for k, name in enumerate(self.indicators):
            value = observed.get(name)
            if value is None:
                continue
            b = prior[k][:, members]
            if value:
                quiet = (1 - self.leak) * np.prod(1 - b, axis=1, keepdims=True)
                prior[k][:, members] = np.clip(b / np.maximum(1 - quiet, 1e-9), 0.0, 1.0)
            else:
                prior[k][:, members] = 0.0
            self.mass[k, members] += prior[k][-1, members]
            self.exposure[k, members] += 1.0
        self.belief = prior

    def features(self, selected):
        b = self.predicted()
        members = list(selected)
        mask = np.zeros(self.population)
        mask[members] = 1.0
        noisy_or = 1 - np.prod(1 - b[:, :, members], axis=2)
        masked = b * mask[None, None, :]
        return np.concatenate((noisy_or.ravel(), masked.ravel()))

    @property
    def dimension(self):
        return len(self.indicators) * len(self.phis) * (1 + self.population)


class LatentFaultFilter:
    """Certificate-attributed latent fault filtering (attribution module "latent").

    Every client carries a latent outage belief for the primary lane (E) and for
    the projection proof (M). Beliefs follow a Markov persistence prior and are
    updated through the certified roster with a leaky noisy-OR likelihood: an
    absent indicator clears all certified members, a present one raises them in
    proportion to their prior belief. One filter runs per persistence value in
    PERSISTENCES; the one with the smallest discounted predictive log-loss is
    used. Base rates are running means of members' posterior beliefs.

    Whether a lane outage also defeats the reserve (S) and whether a signed full
    update is missing (U) are learned per client from delayed, discounted
    pseudo-counts, weighting each certified member by its predicted share of the
    lane outage (its responsibility). Unobserved indicators (None) only advance
    the prediction step.

    Features for the current roster: intercept, the filtered probabilities of E
    and M, the responsibility-weighted probability of S given E, the members'
    mean probability of U, and the previous round's observed E and M flags.
    """

    PERSISTENCES = (0.5, 0.7, 0.85, 0.95)

    def __init__(self, population, leak=0.005, prior=0.05, count_forget=0.99):
        self.population = population
        self.leak = float(leak)
        self.count_forget = float(count_forget)
        self.phis = np.array(self.PERSISTENCES)
        self.state = {}
        for k in ("E", "M"):
            self.state[k] = dict(
                belief=np.full((len(self.phis), population), prior),
                loss=np.zeros(len(self.phis)),
                mass=np.full(population, prior),
                exposure=np.ones(population),
            )
        self.backup = np.ones((population, 2))
        self.upload_prior = np.array([0.1, 1.9])
        self.upload = np.tile(self.upload_prior, (population, 1))
        self.last_responsibility = None
        self.previous = (0.0, 0.0)

    def _predicted(self, k):
        st = self.state[k]
        base = st["mass"] / st["exposure"]
        return self.phis[:, None] * st["belief"] + (1 - self.phis[:, None]) * base[None, :]

    def _best(self, k):
        return int(np.argmin(self.state[k]["loss"]))

    def probability(self, k, selected):
        b = self._predicted(k)[self._best(k)][list(selected)]
        return float(1 - (1 - self.leak) * np.prod(1 - b))

    def responsibility(self, selected):
        b = self._predicted("E")[self._best("E")][list(selected)]
        total = b.sum()
        return b / total if total > 0 else np.full(len(selected), 1 / len(selected))

    def features(self, selected):
        members = list(selected)
        resp = self.responsibility(selected)
        self.last_responsibility = resp
        q_s = float(resp @ (self.backup[members, 0] / self.backup[members].sum(axis=1)))
        p_u = float(np.mean(self.upload[members, 0] / self.upload[members].sum(axis=1)))
        return np.array(
            [
                self.probability("E", selected),
                q_s,
                self.probability("M", selected),
                p_u,
                *self.previous,
            ]
        )

    @property
    def dimension(self):
        return 6

    def round_finished(self, selected, observed):
        members = list(selected)
        for k in ("E", "M"):
            st = self.state[k]
            prior = self._predicted(k)
            value = observed.get(k)
            if value is not None:
                q = 1 - (1 - self.leak) * np.prod(1 - prior[:, members], axis=1)
                q = np.clip(q, 1e-6, 1 - 1e-6)
                st["loss"] = 0.995 * st["loss"] - (np.log(q) if value else np.log(1 - q))
                if value:
                    prior[:, members] = np.clip(prior[:, members] / q[:, None], 0.0, 1.0)
                else:
                    prior[:, members] = 0.0
                st["mass"][members] += prior[self._best(k), members]
                st["exposure"][members] += 1.0
            st["belief"] = prior
        self.previous = (float(bool(observed.get("E"))), float(bool(observed.get("M"))))

    def age_counts(self):
        """Discount the delayed pseudo-counts once per round."""
        self.backup = 1 + self.count_forget * (self.backup - 1)
        self.upload = self.upload_prior + self.count_forget * (self.upload - self.upload_prior)

    def outcome_matured(self, selected, responsibility, observed):
        members = list(selected)
        if observed.get("E") and observed.get("S") is not None:
            self.backup[members, 0] += responsibility * float(observed["S"])
            self.backup[members, 1] += responsibility * (1 - float(observed["S"]))
        if observed.get("U") is not None:
            share = 1 / len(members)
            self.upload[members, 0] += share * float(observed["U"])
            self.upload[members, 1] += share * (1 - float(observed["U"]))


class FundedBudget(CreditBudget):
    """Credit ledger that keeps the cheapest action affordable for every later round."""

    extra_filter = None

    def feasible_actions(self):
        if self.steps >= self.horizon:
            return ()
        floor = (self.horizon - self.steps - 1) * min(self.costs.values())
        actions = tuple(a for a, c in self.costs.items() if c <= self.remaining - floor)
        if self.extra_filter is not None:
            allowed = self.extra_filter()
            actions = tuple(a for a in actions if a in allowed)
        return actions


class FeedbackPDLinUCB(PrimalDualLinUCB):
    """PD-LinUCB (disjoint arms) that also accepts labels of non-chosen arms."""

    def side_reward(self, action, context, value):
        arm = self.index[action]
        x = np.asarray(context, dtype=np.float64)
        v = self.A_inv[arm] @ x
        denominator = 1.0 + float(x @ v)
        if denominator <= 0:
            raise ValueError("Numerically invalid side update")
        self.A_inv[arm] = self.A_inv[arm] - np.outer(v, v) / denominator
        self.b[arm] = self.b[arm] + float(value) * x
        self.observed[arm] += 1


def action_value(action, pE, qS, pM, pU=0.0):
    """Acceptance probability of an action, indicators conditionally independent."""
    lane = 1.0 - pE * qS if action[0] == "R" else 1.0 - pE
    proof = 1.0 - pU if action[-1] == "B" else 1.0 - pM
    return lane * proof


class FactoredPDLinUCB:
    """PD-LinUCB whose reward model is composed from protocol indicators.

    Ridge models estimate P(E|x), P(S|E,x) and P(M|x) from the rounds that
    reveal each indicator. The value of an action follows the protocol's
    acceptance rule; its upper confidence bound adds alpha times the
    delta-method width of that value. Selection maximizes UCB - dual * cost and
    the dual price follows the PD-LinUCB update with the same target rate.
    """

    PRIORS = {"E": 0.0, "S": 0.5, "M": 0.0, "U": 0.0}

    def __init__(
        self,
        dimension,
        costs,
        budget,
        horizon,
        alpha=1.0,
        ridge=1.0,
        eta=0.05,
        dual="gradient",
        forget=1.0,
    ):
        if dual not in ("gradient", "resolve"):
            raise ValueError("Unknown dual price rule")
        forget = float(decimal_number(forget, "forget"))
        if not 0 < forget <= 1:
            raise ValueError("Forgetting factor must lie in (0, 1]")
        # forget < 1: exponentially discounted ridge statistics (Gram matrix and
        # response vector decay once per round, the ridge prior does not), so
        # estimates of rarely revealed indicators return to their priors and the
        # LinUCB width grows again. forget == 1 is the undiscounted original.
        self.forget = forget
        self.dual_rule = dual
        self.history = []
        self.secondary = None
        values = [
            float(decimal_number(v, n))
            for v, n in ((alpha, "alpha"), (ridge, "ridge"), (eta, "eta"))
        ]
        if values[0] < 0 or values[1] <= 0 or values[2] < 0:
            raise ValueError("Require alpha>=0, ridge>0, eta>=0")
        self.alpha, self.ridge, self.eta = values
        self.ledger = CreditBudget(costs, budget, horizon)
        self.actions = tuple(costs)
        self.dimension = dimension
        self.A_inv = {k: np.eye(dimension) / self.ridge for k in INDICATORS}
        self.b = {k: np.zeros(dimension) for k in INDICATORS}
        self.b_prior = {k: np.eye(dimension)[0] * self.ridge * self.PRIORS[k] for k in INDICATORS}
        self.gram = {k: np.zeros((dimension, dimension)) for k in INDICATORS}
        # Optional offsets: column of the context holding a calibrated prior
        # probability for each indicator. The ridge model then learns a residual
        # correction (zero prior) instead of the probability itself.
        self.offsets = None
        self.residual_learning = True
        self.observed = np.zeros(len(self.actions), dtype=np.int64)
        self.labels = {k: 0 for k in INDICATORS}
        self.dual = 0.0
        self.pending = {}
        self.last_decision = {}

    def decay(self):
        """Discount all regression statistics by one round (no-op without forgetting)."""
        if self.forget >= 1.0:
            return
        eye = np.eye(self.dimension) * self.ridge
        for k in self.gram:
            self.gram[k] *= self.forget
            self.b[k] = self.b[k] * self.forget
            self.A_inv[k] = np.linalg.inv(eye + self.gram[k])

    def estimate(self, x):
        p, w = {}, {}
        for k in self.b:
            v = self.A_inv[k] @ x
            offset = float(x[self.offsets[k]]) if self.offsets else 0.0
            p[k] = float(np.clip(offset + v @ (self.b[k] + self.b_prior[k]), 0.0, 1.0))
            w[k] = math.sqrt(max(0.0, float(x @ v)))
        return p, w

    def scores_for(self, X):
        """Composed means and delta-method bonuses for a batch of contexts."""
        X = np.atleast_2d(X)
        p, w = {}, {}
        for k in INDICATORS:
            V = X @ self.A_inv[k]
            offset = X[:, self.offsets[k]] if self.offsets else 0.0
            p[k] = np.clip(offset + V @ (self.b[k] + self.b_prior[k]), 0.0, 1.0)
            w[k] = np.sqrt(np.maximum(0.0, np.sum(V * X, axis=1)))
        pE, qS, pM, pU = p["E"], p["S"], p["M"], p["U"]
        means, bonus = [], []
        zero = np.zeros_like(pE)
        for a in self.actions:
            reserve, full = a[0] == "R", a[-1] == "B"
            lane = 1.0 - pE * qS if reserve else 1.0 - pE
            proof = 1.0 - pU if full else 1.0 - pM
            dE = -(qS if reserve else 1.0) * proof
            dS = -pE * proof if reserve else zero
            dM = zero if full else -lane
            dU = -lane if full else zero
            means.append(lane * proof)
            bonus.append(
                self.alpha
                * np.sqrt(
                    (dE * w["E"]) ** 2
                    + (dS * w["S"]) ** 2
                    + (dM * w["M"]) ** 2
                    + (dU * w["U"]) ** 2
                )
            )
        return np.array(means).T, np.array(bonus).T, p, w

    def resolve_price(self, values, costs, target):
        """Smallest price at which the sample-average greedy spend meets the target rate."""

        def spend(price):
            return float(np.mean(costs[np.argmax(values - price * costs[None, :], axis=1)]))

        if spend(0.0) <= target:
            return 0.0
        low, high = 0.0, 1.0
        while spend(high) > target and high < 1e6:
            high *= 2
        for _ in range(40):
            mid = 0.5 * (low + high)
            if spend(mid) > target:
                low = mid
            else:
                high = mid
        return high

    def attach_secondary(self, ledger):
        """Add a second known-cost resource with its own dual price."""
        self.secondary = ledger
        self.dual2 = 0.0

    def resolve_prices(self, values, costs, costs2, target, target2, start):
        """Sample-average dual of the fluid LP with two resources (projected subgradient)."""
        lam = np.array(start, dtype=float)
        C = np.stack((costs, costs2), axis=1)
        rho = np.array([target, target2])
        best, best_val = lam.copy(), np.inf
        for it in range(80):
            pen = values - (C @ lam)[None, :]
            choice = np.argmax(pen, axis=1)
            dual_value = float(np.mean(pen[np.arange(len(pen)), choice]) + lam @ rho)
            if dual_value < best_val:
                best, best_val = lam.copy(), dual_value
            grad = rho - C[choice].mean(axis=0)
            lam = np.maximum(0.0, lam - 0.5 / math.sqrt(it + 1) * grad)
        return best

    def select(self, context, *, forced_action=None, allowed=None):
        feasible = self.ledger.feasible_actions()
        if allowed is not None:
            feasible = tuple(a for a in feasible if a in allowed)
        if not feasible:
            return None
        x = np.array(context, dtype=np.float64, copy=True)
        if x.shape != (self.dimension,) or not np.all(np.isfinite(x)):
            raise ValueError("Invalid context")
        costs = np.array([float(self.ledger.costs[a]) for a in self.actions])
        self.history.append(x)
        batch = np.asarray(self.history) if self.dual_rule == "resolve" else x[None, :]
        M, Bn, pb, wb = self.scores_for(batch)
        means, bonus = M[-1], Bn[-1]
        p = {k: float(v[-1]) for k, v in pb.items()}
        w = {k: float(v[-1]) for k, v in wb.items()}
        penalty = np.zeros(len(self.actions))
        if self.secondary is not None:
            costs2 = np.array([float(self.secondary.costs[a]) for a in self.actions])
            target2 = self.secondary.target_rate()
            if self.dual_rule == "resolve":
                self.dual, self.dual2 = self.resolve_prices(
                    M + Bn,
                    costs,
                    costs2,
                    self.ledger.target_rate(),
                    target2,
                    (self.dual, self.dual2),
                )
            penalty = self.dual2 * costs2
        elif self.dual_rule == "resolve":
            self.dual = self.resolve_price(M + Bn, costs, self.ledger.target_rate())
        scores = means + bonus - self.dual * costs - penalty
        mask = np.array([a in feasible for a in self.actions])
        arm = (
            self.actions.index(forced_action)
            if forced_action is not None
            else int(np.argmax(np.where(mask, scores, -np.inf)))
        )
        action = self.actions[arm]
        target = self.ledger.target_rate()
        dual_before = self.dual
        if self.dual_rule == "gradient":
            self.dual = max(0.0, dual_before + self.eta * (costs[arm] - target))
            if self.secondary is not None:
                self.dual2 = max(0.0, self.dual2 + self.eta * (costs2[arm] - target2))
        decision_id = self.ledger.steps + 1
        self.ledger.commit(action)
        self.pending[decision_id] = (arm, x)
        self.last_decision = dict(
            decision_id=decision_id,
            indicator_estimates=p,
            indicator_widths=w,
            predicted_reward=means.tolist(),
            confidence_bonus=bonus.tolist(),
            penalized_scores=scores.tolist(),
            dual_before=dual_before,
            dual_after=self.dual,
            target_cost_rate=target,
            **self.ledger.state(),
        )
        return action

    def learn(self, decision_id, observed):
        arm, x = self.pending.pop(decision_id)
        self.observed[arm] += 1
        for k in INDICATORS:
            if observed.get(k) is None or (k == "S" and not observed.get("E")):
                continue
            v = self.A_inv[k] @ x
            self.A_inv[k] = self.A_inv[k] - np.outer(v, v) / (1.0 + float(x @ v))
            if self.forget < 1.0:
                self.gram[k] += np.outer(x, x)
            target = float(bool(observed[k])) - (float(x[self.offsets[k]]) if self.offsets else 0.0)
            if self.residual_learning:
                self.b[k] = self.b[k] + target * x
            self.labels[k] += 1

    def state(self):
        return dict(**self.ledger.state(), dual=self.dual, indicator_labels=dict(self.labels))


class AttributedPDLinUCBPlanner:
    """Drop-in planner: OCP-FL certificate evidence + PD-LinUCB decisions."""

    def __init__(
        self,
        *,
        budget,
        horizon,
        population,
        costs=None,
        actions=("F+P", "R+P"),
        alpha=1.0,
        ridge=1.0,
        eta=0.05,
        feedback_delay=0,
        decays=(0.5, 0.9),
        attributed=True,
        implied_feedback=True,
        base_context=True,
        fund_horizon=True,
        reward_model="arms",
        dual="gradient",
        attribution="evidence",
        residual_learning=True,
        secondary_costs=None,
        secondary_budget=None,
        secondary_price=True,
        forget=1.0,
    ):
        default = {"F+P": 1.0, "F+B": 2.0, "R+P": 2.0, "R+B": 3.0}
        costs = default if costs is None else dict(costs)
        actions = tuple(actions)
        if not actions or "F+P" not in actions or any(a not in costs for a in actions):
            raise ValueError("Actions must include F+P and have known costs")
        if type(feedback_delay) is not int or not 0 <= feedback_delay <= 1000:
            raise ValueError("Invalid reward delay")
        self.population = population
        self.attributed = attributed
        self.implied_feedback = implied_feedback
        self.base_context = base_context
        if attribution not in ("evidence", "filter", "latent"):
            raise ValueError("Unknown attribution module")
        self.attribution = attribution
        if attribution == "evidence":
            self.evidence = CertificateEvidence(population, decays)
        elif attribution == "filter":
            self.evidence = CertificateFilter(population, decays)
        else:
            self.evidence = LatentFaultFilter(population)
        self.previous_observed = {"E": False, "S": None, "M": False, "U": None}
        self.latent_records = {}
        if attribution == "latent":
            if base_context or not attributed:
                raise ValueError("Latent attribution uses its own compact context")
            dimension = 1 + self.evidence.dimension
        else:
            dimension = (17 if base_context else 3) + population
            dimension += self.evidence.dimension if attributed else 0
        if reward_model not in ("arms", "factored", "staged", "branched"):
            raise ValueError("Unknown reward model")
        self.reward_model = reward_model
        if reward_model == "arms":
            if dual != "gradient":
                raise ValueError("The arms model uses the article's gradient dual price")
            self.engine = FeedbackPDLinUCB(
                dimension, {a: costs[a] for a in actions}, budget, horizon, alpha, ridge, eta
            )
        else:
            from .staged_pd_linucb import StagedPDLinUCB
            from .branched_pd_linucb import BranchedPDLinUCB

            engine_class = {
                "staged": StagedPDLinUCB,
                "branched": BranchedPDLinUCB,
                "factored": FactoredPDLinUCB,
            }[reward_model]
            extra = {"forget": forget} if reward_model == "factored" else {}
            if forget != 1.0 and reward_model != "factored":
                raise ValueError("Forgetting is implemented for the factored model")
            self.engine = engine_class(
                dimension,
                {a: costs[a] for a in actions},
                budget,
                horizon,
                alpha,
                ridge,
                eta,
                dual,
                **extra,
            )
        if type(residual_learning) is not bool:
            raise ValueError("residual_learning must be bool")
        if not residual_learning and (attribution != "latent" or reward_model != "factored"):
            raise ValueError("Filter-only mode requires the latent factored model")
        if reward_model == "factored":
            self.engine.residual_learning = residual_learning
        if attribution == "latent" and reward_model == "factored":
            # Context = [1, pE, qS, pM, pU, prevE, prevM]: residual PD-LinUCB on filter output.
            self.engine.offsets = {"E": 1, "S": 2, "M": 3, "U": 4}
            self.engine.b_prior = {k: np.zeros(dimension) for k in INDICATORS}
        if fund_horizon:
            self.engine.ledger = FundedBudget({a: costs[a] for a in actions}, budget, horizon)
            if self.engine.ledger.budget < horizon * min(self.engine.ledger.costs.values()):
                raise ValueError("The funded horizon requires budget for the cheapest action")
        self.ledger = self.engine.ledger
        self.secondary = None
        self.secondary_priced = False
        if secondary_costs is not None:
            self.secondary = FundedBudget(
                {a: secondary_costs[a] for a in actions}, secondary_budget, horizon
            )
            if reward_model in ("factored", "staged", "branched") and secondary_price:
                self.engine.attach_secondary(self.secondary)
                self.secondary_priced = True
            else:
                if not isinstance(self.ledger, FundedBudget):
                    raise ValueError("A secondary feasibility guard needs the funded ledger")
                self.ledger.extra_filter = self.secondary.feasible_actions
        self.actions = actions
        self.fund_horizon = fund_horizon
        self.delay = feedback_delay
        self.queue = []
        self.records = {}
        self.attempt = 0
        self.active = None
        self.ready = False
        self.age = 0
        self.last_decision = {}
        self.last_reward = None
        self.side_labels = 0

    def _release(self, step):
        while self.queue and self.queue[0][0] <= step:
            _, identity, reward = heapq.heappop(self.queue)
            context, action, observed = self.records.pop(identity)
            matured = self.latent_records.pop(identity, None)
            if matured is not None:
                self.evidence.outcome_matured(*matured, observed)
            if self.reward_model in ("factored", "staged", "branched"):
                self.engine.learn(identity, observed)
                continue
            self.engine.reward(identity, reward)
            if self.implied_feedback:
                for other, label in implied_labels(self.actions, observed).items():
                    if other != action:
                        self.engine.side_reward(other, context, label)
                        self.side_labels += 1

    def begin_round(self, observation, capture_attempt, attempt):
        if attempt != self.ledger.steps + 1 or self.active is not None or self.ready:
            raise ValueError("Noncausal or unfinished decision")
        self.age = attempt - capture_attempt
        if self.ledger.stop_reason():
            raise ValueError("Cannot begin an unavailable decision")
        self.attempt = attempt
        self.ready = True
        if attempt > 1 and hasattr(self.engine, "decay"):
            self.engine.decay()
        if attempt > 1 and self.attribution == "latent":
            self.evidence.age_counts()
        self._release(attempt)

    def make_context(self, observation, previous, selected):
        membership = np.zeros(self.population)
        membership[list(selected)] = 1.0
        size = len(selected)
        if self.attribution == "latent":
            return np.concatenate(([1.0], self.evidence.features(selected)))
        if self.base_context:
            head = make_context(observation, previous, self.age)
        else:
            prev = self.previous_observed
            head = np.array([1.0, float(bool(prev["E"])), float(bool(prev["M"]))])
        parts = [head, (membership - size / self.population) / math.sqrt(size)]
        if self.attributed:
            parts.append(self.evidence.features(selected))
        return np.concatenate(parts)

    def choose(self, observation, previous, *, selected=None):
        if self.active is not None or not self.ready:
            raise ValueError("begin_round must precede one selection")
        if selected is None:
            raise ValueError("The certified roster is required")
        context = self.make_context(observation, previous, selected)
        if self.secondary_priced:
            action = self.engine.select(context, allowed=self.secondary.feasible_actions())
        else:
            action = self.engine.select(context)
        if action is not None and self.secondary is not None:
            self.secondary.commit(action)
        if action is not None:
            self.ready = False
            self.active = (self.ledger.steps, action, tuple(selected), context)
            if self.attribution == "latent":
                self.latent_records[self.ledger.steps] = (
                    tuple(selected),
                    np.array(self.evidence.last_responsibility),
                )
            self.last_decision = dict(self.engine.last_decision)
        return action

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
        if self.active is None or action != self.active[1]:
            raise ValueError("Invalid or duplicate selected-action outcome")
        for value in (
            primary_edge_error,
            backup_edge_error,
            projection_missing,
            full_update_missing,
        ):
            if value is not None and type(value) is not bool:
                raise ValueError("Protocol indicators must be bool or None")
        identity, _, selected, context = self.active
        seen = observable(action)
        observed = {
            "E": primary_edge_error,
            "S": bool(backup_edge_error) if "S" in seen and backup_edge_error is not None else None,
            "M": bool(projection_missing)
            if "M" in seen and projection_missing is not None
            else None,
            "U": full_update_missing if "U" in seen else None,
        }
        if stage_observations is not None:
            if set(stage_observations) != {"compute_ok", "lane_ok", "verification_ok"}:
                raise ValueError("Unknown protocol stage observations")
            if any(
                value is not None and type(value) is not bool
                for value in stage_observations.values()
            ):
                raise ValueError("Protocol stages must be bool or None")
            if (
                stage_observations["lane_ok"] is not None
                and stage_observations["compute_ok"] is not True
            ):
                raise ValueError("Aggregation requires completed computation")
            if (
                stage_observations["verification_ok"] is not None
                and stage_observations["lane_ok"] is not True
            ):
                raise ValueError("Verification requires a candidate aggregate")
            observed["stages"] = dict(stage_observations)
        elif self.reward_model in ("staged", "branched"):
            raise ValueError("The staged model requires executed-stage observations")
        reward = outcome_reward(success, cycle_seconds)
        self.evidence.round_finished(selected, observed)
        self.previous_observed = observed
        self.records[identity] = (context, action, observed)
        self.last_reward = reward
        heapq.heappush(self.queue, (identity + self.delay, identity, reward))
        self._release(identity)
        self.active = None

    def feedback_count(self):
        return int(self.engine.observed.sum())

    @property
    def stats(self):
        return dict(
            **self.engine.state(), queued_rewards=len(self.queue), side_labels=self.side_labels
        )
