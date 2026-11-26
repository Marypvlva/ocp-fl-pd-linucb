import unittest
from unittest.mock import patch
from time import monotonic

import numpy as np

from adaptive_planning.tcp_runtime import ACTIONS, Cluster, Fault, flatten_model, make_model
from adaptive_planning.tcp_transport import ProtocolError
from adaptive_planning.tcp_closed_loop_pilot import OnlinePlanner, make_scenario


class TCPIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from adaptive_planning.training import make_workers

        cls.groups, cls.validation, _ = make_workers(4)
        # 8 hidden units -> 610 parameters, not divisible by twelve blocks.
        cls.cluster = Cluster(cls.groups, hidden=8, rpc_timeout=0.5).__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.cluster.close()
        assert all(not p.is_alive() for p in cls.cluster.processes.values())

    def model(self):
        return make_model(hidden=8)

    def test_12_pd_linucb_charges_before_execution_and_stops_at_budget(self):
        import tempfile
        from pathlib import Path
        from adaptive_planning.tcp_dynamic_experiment import parse_args, DynamicStep, run_policy

        config = parse_args(
            [
                "--workers",
                "4",
                "--policies",
                "pd_linucb",
                "--rounds",
                "6",
                "--budget-credits",
                "3",
                "--hidden",
                "8",
                "--model-seed",
                "16",
                "--learning-rate",
                ".5",
                "--pd-feedback-delay",
                "2",
            ]
        )
        trace = [DynamicStep("unlimited", Fault()) for _ in range(6)]
        before = self.cluster.round_id
        with tempfile.TemporaryDirectory() as tmp:
            rows, summary = run_policy(
                self.cluster, "pd_linucb", trace, self.validation, config, 0, 0, Path(tmp)
            )
        self.assertEqual(summary["spent_credits"], 3.0)
        self.assertEqual(summary["stop_reason"], "budget_exhausted")
        self.assertEqual(len(rows), 2)
        self.assertEqual(self.cluster.round_id - before, 2)
        self.assertEqual(summary["observed_rewards"], 0)
        self.assertEqual(summary["pending_rewards"], 2)
        self.assertTrue(all(row["accepted"] for row in rows))

    def test_13_no_affordable_action_means_no_tcp_round(self):
        import tempfile
        from pathlib import Path
        from adaptive_planning.tcp_dynamic_experiment import parse_args, make_trace, run_policy

        config = parse_args(
            [
                "--workers",
                "4",
                "--policies",
                "pd_linucb",
                "--rounds",
                "3",
                "--budget-credits",
                ".5",
                "--hidden",
                "8",
            ]
        )
        before = self.cluster.round_id
        with tempfile.TemporaryDirectory() as tmp:
            rows, summary = run_policy(
                self.cluster,
                "pd_linucb",
                make_trace(42, 3, 4, "clean"),
                self.validation,
                config,
                0,
                0,
                Path(tmp),
            )
        self.assertEqual(rows, [])
        self.assertEqual(self.cluster.round_id, before)
        self.assertEqual(summary["spent_credits"], 0.0)

    def test_05b_projection_failure_delay_is_physical_and_B_is_unaffected(self):
        from adaptive_planning.availability_experiment import AvailabilityFault

        model = self.model()
        before = flatten_model(model)
        fault = AvailabilityFault(missing_projection=(0,), projection_failure_delay_s=0.06)
        result = self.cluster.run_round(model, "F+P", fault=fault)
        self.assertFalse(result.accepted)
        self.assertEqual(before, flatten_model(model))
        self.assertGreaterEqual(result.metrics["projection_rpc_wall_s"], 0.05)
        result = self.cluster.run_round(model, "F+B", fault=fault)
        self.assertTrue(result.accepted, result.errors)
        self.assertEqual(result.metrics["projection_rpc_wall_s"], 0.0)

    def test_05c_projection_failure_delay_respects_deadline(self):
        from adaptive_planning.availability_experiment import AvailabilityFault

        model = self.model()
        before = flatten_model(model)
        started = monotonic()
        result = self.cluster.run_round(
            model,
            "F+P",
            deadline_s=0.06,
            fault=AvailabilityFault(missing_projection=(0,), projection_failure_delay_s=1.0),
        )
        self.assertFalse(result.accepted)
        self.assertEqual(before, flatten_model(model))
        self.assertLess(monotonic() - started, 0.5)

    def test_01_real_round_all_actions_and_exact_bytes(self):
        # Independent oracle only in this test, after the protocol's decision.
        reference_model = self.model()
        total = sum(len(y) for _, y in self.groups)
        expected = np.stack(
            [
                np.rint(reference_model.gradient(X, y) * len(y) / total * 1_000_000).astype(
                    np.int64
                )
                for X, y in self.groups
            ]
        ).sum(axis=0)
        for action in ACTIONS:
            with self.subTest(action=action):
                model = self.model()
                result = self.cluster.run_round(model, action)
                self.assertTrue(result.accepted, result.errors)
                np.testing.assert_array_equal(result.vector, expected)
                metrics = self.cluster.collect_metrics(self.cluster.round_id)
                self.assertEqual(metrics["app_tx_bytes"], metrics["app_rx_bytes"])
                self.assertGreater(metrics["update_transfer_tx_bytes"], 0)
                self.assertGreater(metrics["aggregate_transfer_tx_bytes"], 0)

    def test_02_tamper_never_updates_model(self):
        for method in ("B", "P"):
            model = self.model()
            before = flatten_model(model)
            result = self.cluster.run_round(model, "F+" + method, fault=Fault(corrupt_edges=(0, 1)))
            self.assertFalse(result.accepted)
            self.assertEqual(flatten_model(model), before)
            self.assertGreater(result.metrics["candidate_rejections"], 0)

    def test_03_reserve_recovers_bad_primary(self):
        for method in ("B", "P"):
            result = self.cluster.run_round(
                self.model(),
                "R+" + method,
                fault=Fault(corrupt_edges=(0, 1), edge_delay_s=(0.0, 0.0, 0.04, 0.04)),
            )
            self.assertTrue(result.accepted, result.errors)
            self.assertGreater(result.metrics["verification_attempts"], 1)
            self.assertEqual(result.metrics["selected_edges"], "2,3")

    def test_04_reserve_recovers_unavailable_primary(self):
        for method in ("B", "P"):
            for placement in ("F", "R"):
                result = self.cluster.run_round(
                    self.model(), placement + "+" + method, fault=Fault(failed_edges=(0, 1))
                )
                self.assertEqual(result.accepted, placement == "R", result.errors)

    def test_05_projection_requires_participant_baseline_cached(self):
        for method in ("B", "P"):
            model = self.model()
            before = flatten_model(model)
            result = self.cluster.run_round(
                model, "R+" + method, fault=Fault(missing_projection=(0,))
            )
            self.assertEqual(result.accepted, method == "B", result.errors)
            if method == "P":
                self.assertEqual(flatten_model(model), before)

    def test_06_missing_compute(self):
        for method in ("B", "P"):
            result = self.cluster.run_round(
                self.model(), "R+" + method, fault=Fault(missing_compute=(0,))
            )
            self.assertFalse(result.accepted)

    def test_07_stalled_primary_is_bounded_and_reserve_works(self):
        start = monotonic()
        result = self.cluster.run_round(
            self.model(), "R+P", deadline_s=2.0, fault=Fault(stalled_edges=(0, 1))
        )
        self.assertLess(monotonic() - start, 2.5)
        self.assertTrue(result.accepted, result.errors)
        self.assertGreater(result.metrics["timeouts"], 0)

    def test_08_deadline_and_switch_prevent_model_update(self):
        model = self.model()
        before = flatten_model(model)
        result = self.cluster.run_round(model, "F+B", deadline_s=0.02, switched=True, switch_s=0.1)
        self.assertFalse(result.accepted)
        self.assertEqual(flatten_model(model), before)
        result = self.cluster.run_round(
            model, "R+P", deadline_s=0.03, fault=Fault(edge_delay_s=(0.1, 0.1, 0.1, 0.1))
        )
        self.assertFalse(result.accepted)
        self.assertEqual(flatten_model(model), before)

    def test_09_replayed_command_and_stale_round_rejected(self):
        self.cluster.run_round(self.model(), "F+B")
        rid = self.cluster.round_id
        with self.assertRaises(ProtocolError):
            self.cluster.call(
                "w0",
                rid - 1,
                "projection",
                {"hash": "00" * 32, "dimension": 610, "seeds": ["11" * 32, "22" * 32]},
            )
        nonce = "24" * 16
        self.cluster.call("w0", 0, "stats", {"round": rid}, nonce=nonce)
        with self.assertRaisesRegex(ProtocolError, "Replay"):
            self.cluster.call("w0", 0, "stats", {"round": rid}, nonce=nonce)

    def test_10_training_depends_on_current_model(self):
        model = self.model()
        first = self.cluster.run_round(model, "F+B")
        second = self.cluster.run_round(model, "F+P")
        self.assertTrue(first.accepted and second.accepted)
        self.assertFalse(np.array_equal(first.vector, second.vector))

    def test_11_malformed_replicas_do_not_hide_good_reserve(self):
        original = self.cluster.parallel

        def altered(calls, deadline):
            responses, errors = original(calls, deadline)
            if calls[0][1] == "aggregate":
                (body, data), elapsed = responses["e0"]
                responses["e0"] = ((dict(body, aggregate_s=float("inf")), data), elapsed)
                (body, data), elapsed = responses["e1"]
                responses["e1"] = ((body, data[:-1]), elapsed)
            return responses, errors

        # Fault after authenticated receive isolates validation of replica contents.
        with patch.object(self.cluster, "parallel", side_effect=altered):
            result = self.cluster.run_round(self.model(), "R+P")
        self.assertTrue(result.accepted, result.errors)
        self.assertEqual(result.metrics["selected_edges"], "2,3")
        self.assertEqual(sum("malformed replica" in e for e in result.errors), 2)

    def test_98_out_of_range_gradient_is_rejected_without_clipping(self):
        model = self.model()
        for p in model.params:
            p[:] = 0.0
        model.params[-1][0] = 100.0
        before = flatten_model(model)
        result = self.cluster.run_round(model, "R+B")
        self.assertFalse(result.accepted)
        self.assertEqual(result.metrics["update_bound_exceeded"], 1)
        self.assertEqual(flatten_model(model), before)

    def test_99_dead_process_and_cleanup(self):
        process = self.cluster.processes["e0"]
        process.terminate()
        process.join(1.0)
        result = self.cluster.run_round(self.model(), "R+B")
        self.assertTrue(result.accepted, result.errors)
        result = self.cluster.run_round(self.model(), "F+B")
        self.assertFalse(result.accepted)


class PlannerTests(unittest.TestCase):
    def test_only_chosen_action_receives_feedback(self):
        planner = OnlinePlanner(1, 0.025)
        state = (False, False, False)
        action = planner.choose(state, None)
        planner.observe(state, action, True, 0.1, 0.025)
        self.assertEqual(sum(planner.total.values()), 1)
        self.assertEqual(planner.stats[(state, action)], [1, 1, 0.1 - 0.025])
        self.assertEqual(sum(v[0] for v in planner.stats.values()), 1)

    def test_scenario_independent_of_policy(self):
        self.assertEqual(make_scenario(42, 30, 12, "faults"), make_scenario(42, 30, 12, "faults"))


if __name__ == "__main__":
    unittest.main()
