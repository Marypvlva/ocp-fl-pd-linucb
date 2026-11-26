"""Independent normal-equation checks for attributed reward models and delays."""

import heapq
import math

import numpy as np

from .attributed_pd_linucb import INDICATORS, implied_labels
from .staged_pd_linucb import StagedPDLinUCB
from .branched_pd_linucb import branch_keys, branch_labels


class RewardEquationAudit:
    def __init__(self, planner):
        self.planner = planner
        self.model = planner.reward_model
        self.engine = planner.engine
        self.actions = self.engine.actions
        self.ridge = self.engine.ridge
        self.dimension = self.engine.dimension
        keys = self.actions if self.model == "arms" else self.engine.b.keys()
        self.A = {key: np.eye(self.dimension) * self.ridge for key in keys}
        self.b = {key: np.zeros(self.dimension) for key in keys}
        self.queue = []
        self.dual = 0.0
        self.dual2 = 0.0
        # Latent attribution: PD-LinUCB learns residuals around filter offsets.
        self.offsets = getattr(self.engine, "offsets", None) or {}
        if getattr(self.engine, "forget", 1.0) < 1.0:
            raise ValueError("The equation audit covers stationary designs (forget = 1) only")

    def release(self, step):
        while self.queue and self.queue[0][0] <= step:
            _, identity, context, action, reward, observed = heapq.heappop(self.queue)
            if self.model == "arms":
                self.b[action] += context * reward
                if self.planner.implied_feedback:
                    for other, label in implied_labels(self.actions, observed).items():
                        if other != action:
                            self.A[other] += np.outer(context, context)
                            self.b[other] += context * label
            else:
                if self.model == "branched":
                    labels = branch_labels(action, observed)
                elif self.model == "staged":
                    labels = dict(
                        zip(
                            StagedPDLinUCB.model_names(action),
                            (
                                observed["stages"][name]
                                for name in ("compute_ok", "lane_ok", "verification_ok")
                            ),
                        )
                    )
                else:
                    labels = {key: observed.get(key) for key in INDICATORS}
                    if not observed.get("E"):
                        labels["S"] = None
                for key, value in labels.items():
                    if value is not None:
                        self.A[key] += np.outer(context, context)
                        offset = float(context[self.offsets[key]]) if key in self.offsets else 0.0
                        if getattr(self.engine, "residual_learning", True):
                            self.b[key] += (float(value) - offset) * context

    def check(self, context, action, decision):
        probabilities, widths = {}, {}
        for key in self.A:
            arm = self.actions.index(key) if self.model == "arms" else key
            prior = np.zeros(self.dimension) if self.model == "arms" else self.engine.b_prior[key]
            coefficient = np.linalg.solve(self.A[key], self.b[key] + prior)
            probabilities[key] = float(context @ coefficient)
            if key in self.offsets:
                probabilities[key] += float(context[self.offsets[key]])
            if self.model != "arms":
                probabilities[key] = float(np.clip(probabilities[key], 0.0, 1.0))
            widths[key] = math.sqrt(
                max(0.0, float(context @ np.linalg.solve(self.A[key], context)))
            )
            expected_design = self.A[key]
            if self.model == "arms" and key == action:
                expected_design = expected_design + np.outer(context, context)
            if not np.allclose(
                self.engine.A_inv[arm] @ expected_design,
                np.eye(self.dimension),
                rtol=1e-9,
                atol=1e-9,
            ):
                raise ValueError("Attributed reward design matrix changed")
            if not np.allclose(self.engine.b[arm], self.b[key], rtol=1e-10, atol=1e-10):
                raise ValueError("Attributed delayed labels changed")
        means, bonuses = [], []
        for candidate in self.actions:
            if self.model == "arms":
                mean = probabilities[candidate]
                width = widths[candidate]
            elif self.model == "branched":
                names = branch_keys(candidate)
                c, e, s, g0, g1 = (probabilities[name] for name in names)
                reserve = candidate[0] == "R"
                path = (1 - e) * g0 + (e * (1 - s) * g1 if reserve else 0)
                mean = c * path
                derivatives = (
                    path,
                    c * (-g0 + ((1 - s) * g1 if reserve else 0)),
                    -c * e * g1 if reserve else 0,
                    c * (1 - e),
                    c * e * (1 - s) if reserve else 0,
                )
                width = math.sqrt(
                    sum(
                        (gradient * widths[name]) ** 2 for name, gradient in zip(names, derivatives)
                    )
                )
            elif self.model == "staged":
                names = StagedPDLinUCB.model_names(candidate)
                values = [probabilities[name] for name in names]
                mean = math.prod(values)
                width = math.sqrt(
                    sum(
                        (math.prod(values[j] for j in range(3) if j != i) * widths[name]) ** 2
                        for i, name in enumerate(names)
                    )
                )
            else:
                pE, qS, pM, pU = (probabilities[key] for key in INDICATORS)
                reserve, full = candidate[0] == "R", candidate[-1] == "B"
                lane = 1.0 - pE * qS if reserve else 1.0 - pE
                proof = 1.0 - (pU if full else pM)
                mean = lane * proof
                width = math.sqrt(
                    ((qS if reserve else 1.0) * proof * widths["E"]) ** 2
                    + ((pE * proof * widths["S"]) ** 2 if reserve else 0.0)
                    + (lane * widths["U" if full else "M"]) ** 2
                )
            means.append(mean)
            bonuses.append(self.engine.alpha * width)
        if not np.allclose(means, decision["predicted_reward"], atol=1e-9, rtol=1e-9):
            raise ValueError("Independent attributed reward equation failed")
        if not np.allclose(bonuses, decision["confidence_bonus"], atol=1e-9, rtol=1e-9):
            raise ValueError("Independent attributed uncertainty equation failed")
        costs = np.array([float(self.planner.ledger.costs[a]) for a in self.actions])
        gradient = getattr(self.engine, "dual_rule", "gradient") == "gradient"
        if gradient and not np.isclose(decision["dual_before"], self.dual, atol=1e-10):
            raise ValueError("Attributed dual price changed")
        scores = np.array(means) + bonuses - decision["dual_before"] * costs
        if self.planner.secondary_priced:
            costs2 = np.array([float(self.planner.secondary.costs[a]) for a in self.actions])
            price2 = self.dual2 if gradient else self.engine.dual2
            scores -= price2 * costs2
        if not np.allclose(scores, decision["penalized_scores"], rtol=1e-9, atol=1e-9):
            raise ValueError("Independent attributed resource penalty failed")
        if gradient:
            self.dual = max(
                0.0,
                self.dual
                + self.engine.eta
                * (float(self.planner.ledger.costs[action]) - decision["target_cost_rate"]),
            )
            if self.planner.secondary_priced:
                ledger = self.planner.secondary
                left = ledger.horizon - ledger.steps + 1
                target = float((ledger.remaining + ledger.costs[action]) / left)
                self.dual2 = max(
                    0.0, self.dual2 + self.engine.eta * (float(ledger.costs[action]) - target)
                )
            if not np.isclose(self.dual, decision["dual_after"], atol=1e-10):
                raise ValueError("Independent attributed dual update failed")

    def record(self, step, context, action, reward, observed):
        if self.model == "arms":
            self.A[action] += np.outer(context, context)
        heapq.heappush(
            self.queue, (step + self.planner.delay, step, context.copy(), action, reward, observed)
        )
        self.release(step)
