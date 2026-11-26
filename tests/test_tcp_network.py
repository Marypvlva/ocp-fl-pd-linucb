import multiprocessing as mp
from time import monotonic
import unittest

from adaptive_planning.tcp_network import NetworkProfile, PROFILES, SharedPacer
from adaptive_planning.tcp_network_calibrate import calibrate
from adaptive_planning.tcp_runtime import Cluster, flatten_model, make_model


class PacingTests(unittest.TestCase):
    def test_invalid_settings(self):
        for bad in (-1.0, float("nan"), float("inf"), "40", True):
            with self.assertRaises(ValueError):
                NetworkProfile(edge_mbps=bad)

    def test_failed_reservation_does_not_poison_queue(self):
        pacer = SharedPacer(mp.get_context("spawn"), 1_000_000.0)
        with self.assertRaises(TimeoutError):
            pacer.wait(1_000_000, monotonic() + 0.02)
        self.assertEqual(pacer.next_free.value, 0.0)
        pacer.wait(1000, monotonic() + 0.1)

    def test_real_tcp_shared_across_spawned_processes(self):
        result = calibrate(streams=3, payload_size=60_000, mbps=1.2)
        # Total throughput is limited, even with three independent processes.
        self.assertGreaterEqual(result["elapsed_wall_s"], 0.95 * result["ideal_budget_s"])
        self.assertLessEqual(result["measured_mb_s"], 1.2 * 1.05)
        self.assertGreater(result["frame_bytes"], 3 * 60_000)


class AnalysisTests(unittest.TestCase):
    def test_censored_pair_is_not_replaced_by_last_observation(self):
        from adaptive_planning.tcp_network_analyze import paired_rows

        def row(repeat, policy, reached, value):
            return dict(
                network_profile="unlimited",
                workers="12",
                repeat=str(repeat),
                policy=policy,
                reached=str(reached),
                t_epsilon_wall_s=value,
            )

        summary = [
            row(0, "F+P", 0, ""),
            row(0, "F+B", 1, "2"),
            row(1, "F+P", 1, "1"),
            row(1, "F+B", 1, "3"),
        ]
        result = next(r for r in paired_rows(summary) if r["contrast"] == "F+P - F+B")
        self.assertEqual((result["pairs"], result["complete_pairs"]), (2, 1))
        self.assertEqual((result["first_reached"], result["second_reached"]), (1, 2))
        self.assertEqual(result["mean_delta_wall_s"], -2.0)
        self.assertIsNone(result["bootstrap95_low_s"])

    def test_duplicate_runs_rejected(self):
        from adaptive_planning.tcp_network_analyze import paired_rows

        row = dict(
            network_profile="unlimited",
            workers="12",
            repeat="0",
            policy="F+B",
            reached="1",
            t_epsilon_wall_s="2",
        )
        with self.assertRaises(ValueError):
            paired_rows([row, row])


class NetworkTCPTests(unittest.TestCase):
    def test_full_protocol_with_shared_verifier_limit(self):
        from adaptive_planning.training import make_workers

        groups, _, _ = make_workers(4)
        with Cluster(groups, hidden=8, network=PROFILES["verifier_limited"]) as cluster:
            for action in ("F+B", "R+P"):
                result = cluster.run_round(make_model(hidden=8), action)
                self.assertTrue(result.accepted, result.errors)
                metrics = cluster.collect_metrics(cluster.round_id)
                self.assertGreater(metrics["pacing_wait_wall_sum_s"], 0.0)
                self.assertEqual(metrics["app_tx_bytes"], metrics["app_rx_bytes"])
                expected = sum(
                    metrics[k + "_tx_bytes"]
                    for k in ("worker_compute_reply", "aggregate_transfer", "proof_transfer")
                )
                self.assertEqual(metrics["pacing_reserved_bytes"], expected)
        self.assertTrue(all(not p.is_alive() for p in cluster.processes.values()))

    def test_slow_sender_obeys_rpc_deadline(self):
        from adaptive_planning.training import make_workers

        groups, _, _ = make_workers(4)
        profile = NetworkProfile("test_slow", verifier_mbps=0.005)
        with Cluster(groups, hidden=8, network=profile) as cluster:
            model = make_model(hidden=8)
            before = flatten_model(model)
            started = monotonic()
            result = cluster.run_round(model, "F+B", deadline_s=0.05)
            self.assertLess(monotonic() - started, 0.5)
            self.assertFalse(result.accepted)
            self.assertEqual(flatten_model(model), before)
            # Whichever side reaches its deadline first ends the exchange: the caller
            # times out, or the slow peer closes a truncated frame. Both enforce the
            # deadline; the earlier check accepted only the first (flaky, ~50 %).
            self.assertTrue(result.metrics["timeouts"] > 0 or result.metrics["deadline_exceeded"])
        self.assertTrue(all(not p.is_alive() for p in cluster.processes.values()))


if __name__ == "__main__":
    unittest.main()
