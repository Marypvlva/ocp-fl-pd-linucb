import unittest

from adaptive_planning.adaptive_scheduler import Observation, transfer_prior
from adaptive_planning.aligned_scheduler import AlignedNetworkPlanner
from adaptive_planning.freshness_scheduler import FreshnessMixturePlanner


def observation(edge=40.0, verifier=0.0):
    return Observation(12, 50260, 0.0, edge, verifier, 2.0)


class FreshnessTests(unittest.TestCase):
    def test_fresh_reports_preserve_aligned_choices_and_feedback(self):
        a, b = AlignedNetworkPlanner(42, 0.025), FreshnessMixturePlanner(42, 0.025)
        previous = None
        for attempt in range(1, 81):
            obs = observation() if attempt % 2 else observation(0.0, 40.0)
            for planner in (a, b):
                planner.begin_round(obs, attempt, attempt)
            aa, bb = a.choose(obs, previous), b.choose(obs, previous)
            self.assertEqual(aa, bb)
            for name, value in a.last_decision.items():
                self.assertEqual(b.last_decision[name], value)
            for planner in (a, b):
                planner.observe_round(aa, attempt % 9 != 0, 0.1 + transfer_prior(aa, obs), 0.01)
            self.assertEqual(a.stats, b.stats)
            previous = aa

    def test_old_report_uses_only_distinct_delivered_history(self):
        planner = FreshnessMixturePlanner(42, 0.0)
        edge, verifier = observation(), observation(0.0, 40.0)
        for attempt, capture, obs in ((1, 1, edge), (2, 1, edge), (3, 1, edge), (4, 2, verifier)):
            planner.begin_round(obs, capture, attempt)
            if attempt == 4:
                expected = (
                    transfer_prior("F+P", verifier) / 3
                    + 2 / 3 * (transfer_prior("F+P", edge) + transfer_prior("F+P", verifier)) / 2
                )
                self.assertAlmostEqual(planner.transmission_budget("F+P", obs), expected)
                self.assertEqual([c for c, _ in planner.history], [1, 2])
                planner.choose(obs, "F+P")
                self.assertEqual(planner.last_decision["latest_report_weight"], 1 / 3)
            planner.observe_round("F+P", True, 0.2, 0.0)

    def test_age_shrinkage_is_bounded_by_observed_budgets(self):
        planner = FreshnessMixturePlanner(1, 0.0)
        for attempt in range(1, 41):
            capture = max(1, attempt - 2)
            obs = observation(10.0 + capture, 0.0)
            planner.begin_round(obs, capture, attempt)
            values = [transfer_prior("R+B", old) for _, old in planner.history]
            budget = planner.transmission_budget("R+B", obs)
            self.assertGreaterEqual(budget + 1e-12, min(values))
            self.assertLessEqual(budget - 1e-12, max(values))
            self.assertLessEqual(len(planner.history), 16)
            planner.observe_round("F+B", True, 1.0, 0.0)
        self.assertEqual([c for c, _ in planner.history], list(range(23, 39)))

    def test_no_decision_before_observation_and_no_future_report(self):
        planner = FreshnessMixturePlanner(1, 0.0)
        with self.assertRaises(RuntimeError):
            planner.choose(observation(), None)
        with self.assertRaises(ValueError):
            planner.begin_round(observation(), 2, 1)
        self.assertFalse(planner.history)
        planner.begin_round(observation(), 1, 1)
        planner.observe_round("F+B", True, 0.2, 0.0)
        with self.assertRaises(ValueError):
            planner.begin_round(observation(20.0, 0.0), 1, 2)

    def test_causal_prefix_does_not_depend_on_later_reports(self):
        planners = [FreshnessMixturePlanner(4, 0.025) for _ in range(2)]
        prefix = []
        for planner, length in zip(planners, (6, 12)):
            decisions, previous = [], None
            for attempt in range(1, length + 1):
                capture = max(1, attempt - 2)
                obs = observation(20.0 + capture, 0.0)
                planner.begin_round(obs, capture, attempt)
                action = planner.choose(obs, previous)
                decisions.append((action, planner.last_decision.copy()))
                planner.observe_round(action, True, 0.2, 0.0)
                previous = action
            prefix.append(decisions)
        self.assertEqual(prefix[0], prefix[1][:6])


if __name__ == "__main__":
    unittest.main()
