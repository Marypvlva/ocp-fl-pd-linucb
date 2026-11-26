import unittest

from adaptive_planning.adaptive_scheduler import (
    ACTIONS,
    NetworkOnlinePlanner,
    Observation,
    overhead_prior,
    transfer_prior,
)
from adaptive_planning.aligned_scheduler import AlignedNetworkPlanner
from adaptive_planning.network_telemetry import NetworkTelemetry, TELEMETRY_MODES
from adaptive_planning.tcp_dynamic_experiment import observation_for
from adaptive_planning.tcp_network import PROFILES


class AlignmentTests(unittest.TestCase):
    def test_exact_reports_equal_original_for_every_decision_and_feedback(self):
        old, new = NetworkOnlinePlanner(102, 0.025), AlignedNetworkPlanner(102, 0.025)
        previous = None
        for attempt in range(1, 101):
            obs = observation_for(
                PROFILES[("edge_limited", "verifier_limited")[attempt % 2]], 12, 50260
            )
            new.begin_round(obs, attempt, attempt)
            a, b = old.choose(obs, previous), new.choose(obs, previous)
            self.assertEqual(a, b)
            self.assertEqual(old.last_decision, new.last_decision)
            duration = 0.03 + transfer_prior(a, obs) + attempt / 1000
            success = attempt % 7 != 0
            old.observe(obs, a, success, duration, 0.0)
            new.observe_round(b, success, duration, 0.0)
            for action in ACTIONS:
                for key, value in old.stats[action].items():
                    self.assertEqual(value, new.stats[action][key])
            self.assertEqual(new.cost_feedback_count(), attempt)
            self.assertEqual(new.pending, {})
            previous = a

    def test_delayed_report_matches_historical_action_and_is_used_once(self):
        sensor = NetworkTelemetry(TELEMETRY_MODES["stale"], 123)
        planner = AlignedNetworkPlanner(5, 0.0)
        names = [
            "edge_limited",
            "verifier_limited",
            "edge_limited",
            "verifier_limited",
            "edge_limited",
        ]
        costs = [1.0, 2.0, 3.0, 4.0, 5.0]
        actions = ["F+B", "R+P", "F+P", "R+B", "R+P"]
        for attempt, (name, action, duration) in enumerate(zip(names, actions, costs), 1):
            report = sensor.observe(PROFILES[name], attempt)
            planner.begin_round(observation_for(report, 12, 50260), report.capture_attempt, attempt)
            planner.observe_round(action, True, duration, 0.0)
            self.assertEqual(planner.cost_feedback_count(), max(1, attempt - 2))
            self.assertEqual(planner.feedback_count(), attempt)
        self.assertEqual(set(planner.pending), {4, 5})
        for i, action in enumerate(actions[:3]):
            stat = planner.stats[action]
            obs = observation_for(PROFILES[names[i]], 12, 50260)
            self.assertAlmostEqual(
                stat["residual_sum"],
                2 * overhead_prior(action) + max(0.001, costs[i] - transfer_prior(action, obs)),
            )
            self.assertEqual(stat["cost_n"], 1)
        self.assertEqual(planner.stats["R+B"]["cost_n"], 0)

    def test_pending_feedback_has_no_effect_before_report_arrival(self):
        obs = observation_for(PROFILES["edge_limited"], 12, 50260)
        planners = [AlignedNetworkPlanner(1, 0.0) for _ in range(2)]
        for planner in planners:
            planner.begin_round(obs, 1, 1)
            planner.observe_round("F+B", True, 1.0, 0.0)
            planner.begin_round(obs, 1, 2)
        for planner, duration in zip(planners, (2.0, 200.0)):
            planner.observe_round("R+P", True, duration, 0.0)
            planner.begin_round(obs, 1, 3)
        self.assertEqual(planners[0].stats, planners[1].stats)
        self.assertEqual(planners[0].choose(obs, "F+B"), planners[1].choose(obs, "F+B"))

    def test_feedback_validation_and_bounded_pending(self):
        planner = AlignedNetworkPlanner(1, 0.0)
        obs = Observation(4, 100, 0.0, 40.0, 0.0, 2.0)
        for capture, attempt in ((2, 1), (True, 1), (0, 1), (1, 2)):
            with self.assertRaises(ValueError):
                planner.begin_round(obs, capture, attempt)
        with self.assertRaises(ValueError):
            planner.observe_round("F+B", True, 1.0, 0.0)
        for attempt in range(1, 101):
            capture = max(1, attempt - 20)
            planner.begin_round(obs, capture, attempt)
            if attempt == 1:
                with self.assertRaises(ValueError):
                    planner.observe_round("F+B", True, float("nan"), 0.0)
                with self.assertRaises(ValueError):
                    planner.begin_round(obs, 1, 2)
            planner.observe_round("F+B", True, 1.0, 0.025)
            self.assertLessEqual(len(planner.pending), 20)
            with self.assertRaises(ValueError):
                planner.observe_round("F+B", True, 1.0, 0.0)
        self.assertEqual(planner.cost_feedback_count(), 80)

    def test_skipped_reports_are_not_filled_with_a_different_context(self):
        obs = Observation(4, 100, 0.0, 40.0, 0.0, 2.0)
        planner = AlignedNetworkPlanner(1, 0.0)
        for attempt, capture in ((1, 1), (2, 1), (3, 1), (4, 3)):
            planner.begin_round(obs, capture, attempt)
            planner.observe_round("F+B", True, 1.0, 0.0)
        self.assertEqual(planner.cost_feedback_count(), 2)
        self.assertEqual(set(planner.pending), {4})
        with self.assertRaises(ValueError):
            planner.begin_round(obs, 2, 5)


class AnalysisTests(unittest.TestCase):
    def rows(self):
        rows = []
        for repeat in range(2):
            for mode in ("exact", "stale_noise"):
                for policy, duration in (
                    ("network_aligned", 4.0),
                    ("network_online", 6.0),
                    ("network_prior", 5.0),
                ):
                    rows.append(
                        dict(
                            workers="12",
                            repeat=str(repeat),
                            condition=mode,
                            policy=policy,
                            model_seed=str(16 + repeat),
                            learning_rate="0.5",
                            reached="1",
                            t_epsilon_wall_s=str(duration),
                        )
                    )
            rows.append(
                dict(
                    workers="12",
                    repeat=str(repeat),
                    condition="control",
                    policy="F+P",
                    model_seed=str(16 + repeat),
                    learning_rate="0.5",
                    reached="1",
                    t_epsilon_wall_s="5.5",
                )
            )
        return rows

    def test_primary_is_unique_and_shared_control_is_not_multiplied(self):
        from adaptive_planning.telemetry_alignment_analyze import comparisons

        rows = self.rows()
        results = comparisons(rows, ("exact", "stale_noise"))
        primary = [r for r in results if r["primary"]]
        self.assertEqual(len(primary), 1)
        self.assertEqual(primary[0]["mean_delta_wall_s"], -2.0)
        self.assertEqual(primary[0]["bootstrap95_high_s"], -2.0)
        self.assertEqual(sum(r["condition"] == "control" for r in rows), 2)
        self.assertTrue(all(r["pairs"] == 2 for r in results))

    def test_censoring_is_not_imputed_and_model_pairs_must_match(self):
        from adaptive_planning.telemetry_alignment_analyze import comparisons

        rows = self.rows()
        record = next(
            r for r in rows if r["condition"] == "stale_noise" and r["policy"] == "network_aligned"
        )
        record.update(reached="0", t_epsilon_wall_s="")
        primary = next(r for r in comparisons(rows, ("exact", "stale_noise")) if r["primary"])
        self.assertEqual(primary["pairs"], 2)
        self.assertEqual(primary["complete_pairs"], 1)
        self.assertIsNone(primary["bootstrap95_low_s"])
        record["model_seed"] = "100"
        with self.assertRaises(ValueError):
            comparisons(rows, ("exact", "stale_noise"))

    def test_duplicate_control_rejected(self):
        from adaptive_planning.telemetry_alignment_analyze import comparisons

        rows = self.rows()
        with self.assertRaises(ValueError):
            comparisons([*rows, rows[-1]], ("exact", "stale_noise"))


if __name__ == "__main__":
    unittest.main()
