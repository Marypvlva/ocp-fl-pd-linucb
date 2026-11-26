"""Conditional protocol branches with shared observations across placements."""

import numpy as np

from .attributed_pd_linucb import FactoredPDLinUCB


def branch_keys(action):
    method = action[-1]
    return tuple(
        name + "_" + method
        for name in (
            "compute",
            "primary_failed",
            "rescue_failed",
            "verify_primary",
            "verify_rescued",
        )
    )


def branch_labels(action, observed):
    """Targets of stages that actually ran, conditioned on the observed branch."""
    compute, primary, rescue, quiet, recovered = branch_keys(action)
    stages = observed["stages"]
    labels = {compute: stages["compute_ok"]}
    if stages["compute_ok"] is not True or observed.get("E") is None:
        return labels
    labels[primary] = observed["E"]
    if not observed["E"]:
        labels[quiet] = stages["verification_ok"]
    elif action[0] == "R":
        lane = stages["lane_ok"]
        labels[rescue] = not lane if lane is not None else None
        if lane:
            labels[recovered] = stages["verification_ok"]
    return labels


class BranchedPDLinUCB(FactoredPDLinUCB):
    """Use total probability over primary and rescued execution branches.

    Verification success is conditional on how a candidate was obtained.
    A successful primary path reveals the same target under F and R, so its
    observations can be pooled without inventing labels for unexecuted stages.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_keys = tuple(sorted({name for a in self.actions for name in branch_keys(a)}))
        priors = {
            "compute": 1.0,
            "primary_failed": 0.0,
            "rescue_failed": 0.5,
            "verify_primary": 1.0,
            "verify_rescued": 1.0,
        }
        self.A_inv = {name: np.eye(self.dimension) / self.ridge for name in self.model_keys}
        self.b = {name: np.zeros(self.dimension) for name in self.model_keys}
        self.b_prior = {
            name: np.eye(self.dimension)[0] * self.ridge * priors[name.rsplit("_", 1)[0]]
            for name in self.model_keys
        }
        self.labels = dict.fromkeys(self.model_keys, 0)

    def scores_for(self, contexts):
        contexts = np.atleast_2d(contexts)
        probabilities, widths = {}, {}
        for name in self.model_keys:
            projected = contexts @ self.A_inv[name]
            probabilities[name] = np.clip(projected @ (self.b[name] + self.b_prior[name]), 0, 1)
            widths[name] = np.sqrt(np.maximum(0.0, np.sum(projected * contexts, axis=1)))
        means, bonuses = [], []
        for action in self.actions:
            keys = branch_keys(action)
            c, e, s, g0, g1 = (probabilities[name] for name in keys)
            reserved = action[0] == "R"
            path = (1 - e) * g0
            if reserved:
                path = path + e * (1 - s) * g1
            derivatives = [
                path,
                c * (-g0 + ((1 - s) * g1 if reserved else 0)),
                -c * e * g1 if reserved else 0,
                c * (1 - e),
                c * e * (1 - s) if reserved else 0,
            ]
            variance = sum(
                (gradient * widths[name]) ** 2 for name, gradient in zip(keys, derivatives)
            )
            means.append(c * path)
            bonuses.append(self.alpha * np.sqrt(variance))
        return np.array(means).T, np.array(bonuses).T, probabilities, widths

    def learn(self, decision_id, observed):
        arm, context = self.pending.pop(decision_id)
        self.observed[arm] += 1
        for name, value in branch_labels(self.actions[arm], observed).items():
            if value is None:
                continue
            projected = self.A_inv[name] @ context
            self.A_inv[name] -= np.outer(projected, projected) / (1 + float(context @ projected))
            self.b[name] += float(value) * context
            self.labels[name] += 1
