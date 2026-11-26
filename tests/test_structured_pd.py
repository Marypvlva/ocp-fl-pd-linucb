"""Independent normal-equation and accounting checks for the research model."""

import unittest
import numpy as np
from adaptive_planning.pd_linucb_planner import DEFAULT_COSTS
from adaptive_planning.structured_pd_linucb import SharedPrimalDualLinUCB, action_features
from adaptive_planning.ocp_availability_sim import simulate


class StructuredPDTests(unittest.TestCase):
    def test_delayed_ridge_matches_batch_geometry_and_selected_labels(self):
        learner = SharedPrimalDualLinUCB(2, DEFAULT_COSTS, 8, 4, alpha=1, eta=0.05)
        A = np.eye(6)
        b = np.zeros(6)
        pending = []
        for step, action in enumerate(["F+P", "R+P", "F+B", "R+B"], 1):
            x = np.array([1, step / 4])
            features = action_features(x, action)
            predicted = np.array(
                [action_features(x, a) @ np.linalg.solve(A, b) for a in DEFAULT_COSTS]
            )
            self.assertEqual(learner.select(x, forced_action=action), action)
            np.testing.assert_allclose(
                learner.last_decision["predicted_reward"], predicted, atol=1e-12
            )
            A += np.outer(features, features)
            pending.append((step, features))
            if step >= 2:
                identity, old = pending.pop(0)
                learner.reward(identity, 1)
                b += old
            np.testing.assert_allclose(learner.A_inv, np.linalg.inv(A), atol=1e-12)
            np.testing.assert_allclose(learner.b, b, atol=1e-12)
        self.assertEqual(float(learner.ledger.spent), 8)
        self.assertEqual(int(learner.observed.sum()), 3)
        with self.assertRaises(ValueError):
            learner.reward(1, 1)

    def test_unaffordable_and_invalid_inputs_do_not_mutate_budget(self):
        learner = SharedPrimalDualLinUCB(1, DEFAULT_COSTS, 1, 3)
        for context, action in (([1], "R+B"), ([float("nan")], "F+P")):
            with self.assertRaises(ValueError):
                learner.select(context, forced_action=action)
            self.assertEqual(learner.ledger.steps, 0)
        self.assertEqual(learner.select([1]), "F+P")
        self.assertIsNone(learner.select([1]))

    def test_additive_action_structure_and_causal_simulation(self):
        x = np.array([1, 0.4])
        np.testing.assert_allclose(
            action_features(x, "R+B"),
            action_features(x, "R+P") + action_features(x, "F+B") - action_features(x, "F+P"),
        )
        for policy in ("pd_shared", "pd_joint_compact"):
            original = simulate(policy, [0, 1, 2] * 10, 43)
            changed = simulate(policy, [0, 1, 2] * 4 + [2] * 18, 43)
            self.assertEqual(original["actions"][:13], changed["actions"][:13])
            self.assertLessEqual(original["spent"], 43)
            self.assertEqual(simulate(policy, [0] * 30, 43, delay=100)["labels_observed"], 0)
