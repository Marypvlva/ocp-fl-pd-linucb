"""Regressions for censored protocol observations and byte-limit enforcement."""

from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import unittest

import numpy as np

from adaptive_planning.byte_budget import ByteBudgetExceeded, SharedByteBudget, reserve_frame
from adaptive_planning.protocol_feedback import feedback_from_round
from adaptive_planning.staged_pd_linucb import StagedPDLinUCB
from adaptive_planning.branched_pd_linucb import BranchedPDLinUCB, branch_labels
from adaptive_planning.remedy_controls import PlugInRemedyControl
from adaptive_planning.adaptive_scheduler import Observation
from adaptive_planning.remedy_world import WorldSpec, make_world
from adaptive_planning.ocp_experiment import remedy_fault


class FeedbackTests(unittest.TestCase):
    def test_aggregate_failure_does_not_invent_a_projection_label(self):
        feedback = feedback_from_round(
            "F+P",
            False,
            ["e0: unavailable"],
            dict(
                compute_started=True,
                compute_completed=True,
                aggregate_started=True,
                aggregate_completed=False,
                projection_started=False,
                verification_attempts=0,
            ),
        )
        self.assertTrue(feedback.primary_edge_error)
        self.assertIsNone(feedback.projection_missing)
        self.assertFalse(feedback.lane_ok)
        self.assertIsNone(feedback.verification_ok)

    def test_failed_computation_leaves_aggregation_unknown(self):
        feedback = feedback_from_round(
            "R+B",
            False,
            [],
            dict(
                compute_started=True,
                compute_completed=False,
                aggregate_started=False,
                verification_attempts=0,
            ),
        )
        self.assertTrue(feedback.full_update_missing)
        self.assertIsNone(feedback.primary_edge_error)
        self.assertIsNone(feedback.backup_edge_error)
        self.assertIsNone(feedback.lane_ok)

    def test_projection_is_observed_only_after_the_challenge_is_sent(self):
        feedback = feedback_from_round(
            "F+P",
            False,
            [],
            dict(
                compute_started=True,
                compute_completed=True,
                aggregate_started=True,
                aggregate_completed=True,
                projection_started=False,
                verification_attempts=1,
                verification_completed=False,
            ),
        )
        self.assertIsNone(feedback.projection_missing)
        self.assertFalse(feedback.verification_ok)

    def test_control_skips_unknown_primary_and_projection_labels(self):
        planner = PlugInRemedyControl(
            budget=2,
            horizon=2,
            population=2,
            actions=("F+P",),
            costs={"F+P": 1},
            feedback_delay=0,
        )
        observation = Observation(1, 10, 0, 0, 0, 0, False, False)
        planner.begin_round(observation, 1, 1)
        action = planner.choose(observation, None, selected=(0,))
        planner.observe_round(action, False, 0, 0, primary_edge_error=None, projection_missing=None)
        self.assertNotIn("E", planner.suff)
        self.assertNotIn("M", planner.suff)


class StageModelTests(unittest.TestCase):
    def test_branch_model_shares_quiet_primary_verification_between_placements(self):
        engine = BranchedPDLinUCB(1, {"F+P": 1, "R+P": 2}, 900, 600, alpha=0)
        for t in range(600):
            action = "F+P" if t % 2 else "R+P"
            failed = t % 5 < 2
            engine.select(np.ones(1), forced_action=action)
            engine.learn(
                t + 1,
                dict(
                    E=failed,
                    stages=dict(
                        compute_ok=True,
                        lane_ok=not failed or action[0] == "R",
                        verification_ok=False
                        if failed and action[0] == "R"
                        else True
                        if not failed
                        else None,
                    ),
                ),
            )
        means, _, _, _ = engine.scores_for(np.ones((1, 1)))
        self.assertAlmostEqual(means[0, 0], 0.6, delta=0.01)
        self.assertAlmostEqual(means[0, 1], 0.6, delta=0.02)
        self.assertEqual(engine.labels["verify_primary_P"], 360)
        self.assertEqual(engine.labels["verify_rescued_P"], 120)

    def test_branch_labels_do_not_invent_a_rescued_verification(self):
        labels = branch_labels(
            "R+P",
            dict(
                E=True,
                stages=dict(
                    compute_ok=True,
                    lane_ok=False,
                    verification_ok=None,
                ),
            ),
        )
        self.assertTrue(labels["rescue_failed_P"])
        self.assertNotIn("verify_rescued_P", labels)

    def test_signed_update_fault_is_preserved_in_the_live_trace(self):
        world = make_world(WorldSpec(horizon=40, proof=(1.0, 0.0), full_update_failure=1.0), 193999)
        config = {"remedy_traces": {"193999": world.trace_config()}}
        found = 0
        for t, members in enumerate(world.rosters):
            fault = remedy_fault(config, t + 1, members, 193999)
            self.assertEqual(fault, world.faults[t])
            self.assertEqual(bool(fault.missing_full_update), world.indicators[t][3])
            found += bool(fault.missing_full_update)
        self.assertGreater(found, 0)

    def test_conditional_stage_model_handles_correlated_faults(self):
        engine = StagedPDLinUCB(1, {"F+P": 1}, budget=500, horizon=500, alpha=0)
        for t in range(500):
            # The same outage prevents aggregation and a projection response.
            # Projection availability is never queried on the failed lane.
            lane_ok = t % 5 >= 2
            engine.select(np.array([1.0]))
            engine.learn(
                t + 1,
                dict(
                    stages=dict(
                        compute_ok=True,
                        lane_ok=lane_ok,
                        verification_ok=True if lane_ok else None,
                    )
                ),
            )
        means, _, _, _ = engine.scores_for(np.array([[1.0]]))
        self.assertAlmostEqual(means[0, 0], 0.6, delta=0.005)
        self.assertEqual(engine.labels["lane_F+P"], 500)
        self.assertEqual(engine.labels["verify_F+P"], 300)


class ByteBudgetTests(unittest.TestCase):
    def setUp(self):
        self.budget = SharedByteBudget(mp.get_context("spawn"), 100)

    def test_partial_send_settles_sent_bytes_and_releases_the_tail(self):
        with self.assertRaises(ConnectionError):
            with reserve_frame(self.budget, 80) as sent:
                sent(25)
                raise ConnectionError("simulated peer failure")
        self.assertEqual(self.budget.state()["spent"], 25)
        self.assertEqual(self.budget.state()["reserved"], 0)
        with reserve_frame(self.budget, 75) as sent:
            sent(75)
        with self.assertRaises(ByteBudgetExceeded):
            with reserve_frame(self.budget, 1):
                self.fail("Exhausted reservation was accepted")

    def test_concurrent_reservations_cannot_overspend(self):
        def send():
            try:
                with reserve_frame(self.budget, 17) as sent:
                    sent(17)
                return True
            except ByteBudgetExceeded:
                return False

        with ThreadPoolExecutor(8) as pool:
            accepted = list(pool.map(lambda _: send(), range(20)))
        self.assertEqual(sum(accepted), 5)
        self.assertEqual(self.budget.state()["spent"], 85)
        self.assertEqual(self.budget.state()["reserved"], 0)

    def test_sender_cannot_settle_another_frames_reservation(self):
        with reserve_frame(self.budget, 40) as sent:
            with self.assertRaises(ValueError):
                sent(41)
        self.assertEqual(self.budget.state()["reserved"], 0)


if __name__ == "__main__":
    unittest.main()
