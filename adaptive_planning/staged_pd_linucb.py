"""PD-LinUCB values composed by the chain rule over executed protocol stages."""

import numpy as np

from .attributed_pd_linucb import FactoredPDLinUCB


class StagedPDLinUCB(FactoredPDLinUCB):
    """Estimate computation, conditional lane success and final verification.

    Conditional stage models avoid assuming independence of physical faults.
    Linear probability models and delta-method widths remain approximations.
    The action score, resource prices and delayed updates use the base engine.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        names = {name for action in self.actions for name in self.model_names(action)}
        self.model_keys = tuple(sorted(names))
        self.A_inv = {name: np.eye(self.dimension) / self.ridge for name in self.model_keys}
        self.b = {name: np.zeros(self.dimension) for name in self.model_keys}
        self.b_prior = {name: np.eye(self.dimension)[0] * self.ridge for name in self.model_keys}
        self.labels = dict.fromkeys(self.model_keys, 0)

    @staticmethod
    def model_names(action):
        return ("compute_" + action[-1], "lane_" + action, "verify_" + action)

    def scores_for(self, contexts):
        contexts = np.atleast_2d(contexts)
        probabilities, widths = {}, {}
        for name in self.model_keys:
            projected = contexts @ self.A_inv[name]
            probabilities[name] = np.clip(projected @ (self.b[name] + self.b_prior[name]), 0.0, 1.0)
            widths[name] = np.sqrt(np.maximum(0.0, np.sum(projected * contexts, axis=1)))
        means, bonuses = [], []
        for action in self.actions:
            compute, lane, verify = self.model_names(action)
            p_compute, p_lane, p_verify = (probabilities[name] for name in (compute, lane, verify))
            means.append(p_compute * p_lane * p_verify)
            variance = (
                (p_lane * p_verify * widths[compute]) ** 2
                + (p_compute * p_verify * widths[lane]) ** 2
                + (p_compute * p_lane * widths[verify]) ** 2
            )
            bonuses.append(self.alpha * np.sqrt(variance))
        return np.array(means).T, np.array(bonuses).T, probabilities, widths

    def learn(self, decision_id, observed):
        arm, context = self.pending.pop(decision_id)
        self.observed[arm] += 1
        stages = observed["stages"]
        for name, stage in zip(
            self.model_names(self.actions[arm]), ("compute_ok", "lane_ok", "verification_ok")
        ):
            value = stages[stage]
            if value is None:
                continue
            projected = self.A_inv[name] @ context
            self.A_inv[name] -= np.outer(projected, projected) / (1.0 + float(context @ projected))
            self.b[name] += float(value) * context
            self.labels[name] += 1
