"""Check the exploratory simulator against the real planner adapter."""

import unittest

import numpy as np

from adaptive_planning.adaptive_scheduler import Observation
from adaptive_planning.ocp_availability_sim import (
    JOINT_POLICIES,
    POLICIES,
    generate_trace,
    simulate,
)
from adaptive_planning.pd_linucb_planner import PDLinUCBPlanner


class AvailabilitySimulationTests(unittest.TestCase):
    def test_pd_trajectory_and_delayed_labels_match_runtime_adapter(self):
        failures = generate_trace(8000, 40, 0.85, 0.25)
        result = simulate("pd", failures, 58)
        planner = PDLinUCBPlanner(
            budget=58,
            horizon=40,
            available_actions=["F+P", "R+P"],
            feedback_delay=2,
            reward_mode="success",
        )
        actions = []
        previous_issue = 0
        for step, issue in enumerate(failures, 1):
            if planner.ledger.stop_reason():
                break
            obs = Observation(3, 4810, 0, 0, 0, 0, previous_issue == 1, previous_issue == 2)
            planner.begin_round(obs, step, step)
            action = planner.choose(obs, actions[-1] if actions else None)
            planner.observe_round(
                action, bool(issue == 0 or (issue == 1 and action == "R+P")), 0, 0
            )
            actions.append(action)
            previous_issue = issue
        self.assertEqual(actions, result["actions"])
        self.assertEqual(planner.feedback_count(), result["labels_observed"])
        self.assertEqual(float(planner.ledger.spent), result["spent"])

    def test_current_and_future_faults_do_not_change_current_action(self):
        original = generate_trace(8001, 20, 0.5, 0.25)
        changed = original.copy()
        changed[9:] = (changed[9:] + 1) % 3
        for policy in (*POLICIES, *JOINT_POLICIES):
            with self.subTest(policy=policy):
                first = simulate(policy, original, 29)
                second = simulate(policy, changed, 29)
                self.assertEqual(first["actions"][:10], second["actions"][:10])

    def test_budget_and_unrepairable_failure_at_extremes(self):
        for budget in (12, 17, 24):
            for policy in POLICIES:
                with self.subTest(budget=budget, policy=policy):
                    result = simulate(policy, np.full(12, 2), budget)
                    self.assertEqual(result["accepted"], 0)
                    self.assertLessEqual(result["spent"], budget)
                    self.assertEqual(
                        result["reserve_on_unrepairable"],
                        result["reserves"],
                    )

    def test_onehot_uses_selected_feedback_only_and_delay_releases_no_future_labels(self):
        result = simulate("pd_type_onehot", np.zeros(12), 17, delay=100)
        self.assertEqual(result["labels_observed"], 0)
        self.assertEqual(result["accepted"], result["attempted"])
        no_delay = simulate("pd_type_onehot", np.zeros(12), 17, delay=0)
        self.assertEqual(no_delay["labels_observed"], no_delay["attempted"])

    def test_baseline_mode_does_not_reveal_hidden_projection_failure(self):
        # A reactive controller sees the missing P proof after round one, uses B,
        # then sees no P error because B does not request P proofs.
        result = simulate("reactive_joint", np.full(6, 2), 9)
        self.assertEqual(result["actions"][:3], ["F+P", "F+B", "F+P"])
        self.assertGreater(result["projection_recoveries"], 0)
