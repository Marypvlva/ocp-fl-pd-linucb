"""Features and a reference policy for a verified, fixed client roster."""

import math

import numpy as np


def roster_features(selected, population, size):
    """Encode membership independently of order and keep its norm at most one."""
    if (
        not isinstance(selected, (list, tuple))
        or not 1 <= size <= population
        or len(selected) != size
        or any(type(i) is not int or not 0 <= i < population for i in selected)
        or len(set(selected)) != size
    ):
        raise ValueError("Expected the verified client roster")
    membership = np.zeros(population)
    membership[list(selected)] = 1
    return (membership - size / population) / math.sqrt(size)


class RosterRiskControl:
    """Forecast primary edge errors from the same causal observations as PD.

    The next observation supplies the error label for the previous roster.
    Predicting edge trouble is a proxy for the benefit of a reserve, since the
    primary can fail even when its reserve or a required proof also fails.
    """

    def __init__(self):
        self.previous = None
        self.features = []
        self.errors = []

    def prefer_reserve(self, observation, membership, reserve_fraction):
        if self.previous is not None:
            self.features.append(self.previous)
            self.errors.append(float(observation.edge_trouble))
        current = np.r_[1, observation.edge_trouble, observation.projection_missing, membership]
        self.previous = current
        if not any(self.errors):
            return observation.edge_trouble
        design = np.asarray(self.features)
        coefficients = np.linalg.solve(
            np.eye(current.size) + design.T @ design, design.T @ self.errors
        )
        historical_scores = np.clip(design @ coefficients, 0, 1)
        cutoff = np.quantile(historical_scores, 1 - min(1, reserve_fraction))
        return np.clip(current @ coefficients, 0, 1) > cutoff


class RosterBenefitControl:
    """Rank opportunities by primary error risk times conditional rescue chance.

    All attempts reveal the observed primary error flag. Only attempts using
    the reserve on a primary error reveal whether this reserve succeeded.
    Ridge models use matured labels and apply a neutral rescue prior until
    enough conditional outcomes are available.
    """

    def __init__(self):
        self.pending = {}
        self.features = []
        self.errors = []
        self.rescue_features = []
        self.rescues = []

    def score(self, observation, membership, reserve_fraction):
        current = np.r_[1, observation.edge_trouble, observation.projection_missing, membership]
        design = np.asarray(self.features).reshape(-1, current.size)
        errors = self._fit(design, self.errors, intercept_prior=0.0)
        reserve_design = np.asarray(self.rescue_features).reshape(-1, current.size)
        rescues = self._fit(reserve_design, self.rescues, intercept_prior=0.5)
        historical_scores = np.clip(design @ errors, 0, 1) * np.clip(design @ rescues, 0, 1)
        cutoff = (
            float(np.quantile(historical_scores, 1 - min(1, reserve_fraction)))
            if len(historical_scores)
            else 0.0
        )
        risk = float(np.clip(current @ errors, 0, 1))
        rescue = float(np.clip(current @ rescues, 0, 1))
        decision = dict(
            primary_error_risk=risk,
            conditional_rescue=rescue,
            estimated_benefit=risk * rescue,
            cutoff=cutoff,
            matured_primary_labels=len(self.errors),
            matured_conditional_labels=len(self.rescues),
        )
        prefer_reserve = (
            decision["estimated_benefit"] > cutoff if any(self.errors) else observation.edge_trouble
        )
        return prefer_reserve, current, decision

    @staticmethod
    def _fit(design, labels, *, intercept_prior):
        target = design.T @ labels
        target[0] += intercept_prior
        return np.linalg.solve(np.eye(design.shape[1]) + design.T @ design, target)

    def release(self, identity):
        context, action, primary_error, accepted = self.pending.pop(identity)
        self.features.append(context)
        self.errors.append(float(primary_error))
        if action == "R+P" and primary_error:
            self.rescue_features.append(context)
            self.rescues.append(float(accepted))
