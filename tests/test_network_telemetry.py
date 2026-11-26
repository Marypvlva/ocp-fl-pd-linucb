import unittest
import numpy as np

from adaptive_planning.network_telemetry import NetworkTelemetry, TELEMETRY_MODES, TelemetryMode
from adaptive_planning.tcp_network import PROFILES
from adaptive_planning.tcp_runtime import Cluster, flatten_model, make_model


class SensorTests(unittest.TestCase):
    def test_exact_and_delayed_capture_prefix(self):
        exact = NetworkTelemetry(TELEMETRY_MODES["exact"], 42)
        delayed = NetworkTelemetry(TELEMETRY_MODES["stale"], 42)
        names = ["edge_limited", "verifier_limited", "edge_limited", "verifier_limited"]
        reports = []
        for attempt, name in enumerate(names, 1):
            current = exact.observe(PROFILES[name], attempt)
            self.assertEqual(current.capture_attempt, attempt)
            self.assertEqual(current.edge_mbps, PROFILES[name].edge_mbps)
            reports.append(delayed.observe(PROFILES[name], attempt))
        self.assertEqual([r.capture_attempt for r in reports], [1, 1, 1, 2])
        self.assertEqual(
            [r.source_profile for r in reports], [names[0], names[0], names[0], names[1]]
        )

    def test_noise_is_captured_before_delay_not_regenerated(self):
        noisy = NetworkTelemetry(TELEMETRY_MODES["noise"], 123)
        stale = NetworkTelemetry(TELEMETRY_MODES["stale_noise"], 123)
        original = noisy.observe(PROFILES["edge_limited"], 1)
        delivered = [stale.observe(PROFILES["edge_limited"], i) for i in range(1, 4)]
        self.assertTrue(all(r == original for r in delivered))
        self.assertEqual(original.verifier_mbps, 0.0)
        self.assertNotEqual(original.edge_mbps, 40.0)
        self.assertGreater(original.edge_mbps, 0.0)

    def test_reproducible_common_prefix(self):
        a = NetworkTelemetry(TELEMETRY_MODES["stale_noise"], 42)
        b = NetworkTelemetry(TELEMETRY_MODES["stale_noise"], 42)
        prefix = [a.observe(PROFILES["verifier_limited"], i) for i in range(1, 5)]
        longer = [b.observe(PROFILES["verifier_limited"], i) for i in range(1, 9)]
        self.assertEqual(prefix, longer[:4])

    def test_invalid_settings_or_future_attempt(self):
        for kwargs in (
            {"delay_rounds": -1},
            {"delay_rounds": True},
            {"log_sigma": float("nan")},
            {"log_sigma": -1},
        ):
            with self.assertRaises(ValueError):
                TelemetryMode("bad", **kwargs)
        sensor = NetworkTelemetry(TELEMETRY_MODES["exact"], 1)
        with self.assertRaises(ValueError):
            sensor.observe(PROFILES["edge_limited"], 2)


class LearningRateTests(unittest.TestCase):
    def test_explicit_rate_and_legacy_fallback(self):
        full = make_model(16, hidden=8, learning_rate=1.0)
        half = make_model(16, hidden=8, learning_rate=0.5)
        legacy = make_model(16, hidden=8)
        del legacy.learning_rate
        before = [p.copy() for p in full.params]
        update = np.full(full.nparam, 1000, dtype=np.int64)
        full.update(update)
        half.update(update)
        legacy.update(update)
        for initial, a, b, old in zip(before, full.params, half.params, legacy.params):
            np.testing.assert_allclose(initial - a, 2 * (initial - b), atol=1e-14)
            np.testing.assert_array_equal(a, old)

    def test_invalid_rate_fails_before_mutation(self):
        model = make_model(16, hidden=8)
        before = flatten_model(model)
        for rate in (-1.0, float("nan"), 0.0):
            model.learning_rate = rate
            with self.assertRaises(ValueError):
                model.update(np.ones(model.nparam, dtype=np.int64))
            self.assertEqual(flatten_model(model), before)

    def test_real_tcp_update_respects_coordinator_learning_rate(self):
        from adaptive_planning.training import make_workers

        groups, _, _ = make_workers(4)
        full, half = make_model(16, 8, 1.0), make_model(16, 8, 0.5)
        before = [p.copy() for p in full.params]
        with Cluster(groups, hidden=8) as cluster:
            a, b = cluster.run_round(full, "F+B"), cluster.run_round(half, "F+B")
            self.assertTrue(a.accepted and b.accepted, a.errors + b.errors)
            np.testing.assert_array_equal(a.vector, b.vector)
            for initial, p, q in zip(before, full.params, half.params):
                np.testing.assert_allclose(initial - p, 2 * (initial - q), atol=1e-14)


class RobustnessAnalysisTests(unittest.TestCase):
    def rows(self):
        result = []
        for mode in TELEMETRY_MODES:
            for policy, time in (("network_online", "4"), ("network_prior", "6")):
                result.append(
                    dict(
                        workers="12",
                        repeat="0",
                        condition=mode,
                        policy=policy,
                        model_seed="16",
                        learning_rate="0.5",
                        reached="1",
                        t_epsilon_wall_s=time,
                    )
                )
        result.append(
            dict(
                workers="12",
                repeat="0",
                condition="control",
                policy="F+P",
                model_seed="16",
                learning_rate="0.5",
                reached="1",
                t_epsilon_wall_s="5",
            )
        )
        return result

    def test_shared_control_is_not_replicated(self):
        from adaptive_planning.telemetry_robustness_analyze import paired_comparisons

        rows = self.rows()
        result = paired_comparisons(rows, list(TELEMETRY_MODES))
        self.assertEqual(len(rows), 9)
        controls = [r for r in result if r["second_condition"] == "control"]
        self.assertEqual(len(controls), 4)
        self.assertTrue(all(r["pairs"] == 1 and r["mean_delta_wall_s"] == -1 for r in controls))
        primary = [r for r in result if r["primary"]]
        self.assertEqual(len(primary), 1)
        self.assertEqual(primary[0]["first_condition"], "stale_noise")

    def test_censored_and_mismatched_initialization(self):
        from adaptive_planning.telemetry_robustness_analyze import paired_comparisons

        rows = self.rows()
        row = next(
            r for r in rows if r["condition"] == "stale_noise" and r["policy"] == "network_online"
        )
        row.update(reached="0", t_epsilon_wall_s="")
        primary = next(r for r in paired_comparisons(rows, list(TELEMETRY_MODES)) if r["primary"])
        self.assertEqual(primary["complete_pairs"], 0)
        self.assertIsNone(primary["mean_delta_wall_s"])
        row["model_seed"] = "17"
        with self.assertRaises(ValueError):
            paired_comparisons(rows, list(TELEMETRY_MODES))


if __name__ == "__main__":
    unittest.main()
