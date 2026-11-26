from dataclasses import fields
import multiprocessing as mp
from time import monotonic
import unittest

from adaptive_planning.adaptive_scheduler import (
    ACTIONS,
    Observation,
    NetworkOnlinePlanner,
    FrozenNetworkPlanner,
    network_rule,
    predicted_seconds,
    transfer_prior,
)
from adaptive_planning.tcp_dynamic_experiment import make_trace, observation_for, observed_issues
from adaptive_planning.tcp_network import NetworkShaper, PROFILES
from adaptive_planning.tcp_runtime import Cluster, RoundResult, make_model


def hold_config_lock(shaper, ready):
    shaper.config_lock.acquire()
    ready.send("locked")
    ready.close()
    from time import sleep

    sleep(10)


class RoundProfileTests(unittest.TestCase):
    def test_terminated_lock_owner_does_not_cause_infinite_wait(self):
        context = mp.get_context("spawn")
        shaper = NetworkShaper(context, PROFILES["unlimited"], dynamic=True)
        receive, send = context.Pipe(duplex=False)
        process = context.Process(target=hold_config_lock, args=(shaper, send))
        try:
            process.start()
            send.close()
            self.assertTrue(receive.poll(5))
            self.assertEqual(receive.recv(), "locked")
            process.terminate()
            process.join(2)
            started = monotonic()
            with self.assertRaises(TimeoutError):
                shaper.configure_round(1, PROFILES["edge_limited"], deadline=monotonic() + 0.03)
            with self.assertRaises(TimeoutError):
                shaper.profile_for_round(1, deadline=monotonic() + 0.03)
            self.assertLess(monotonic() - started, 0.5)
        finally:
            receive.close()
            if process.is_alive():
                process.kill()
            process.join(2)

    def test_round_binding_immutable_and_retained(self):
        shaper = NetworkShaper(mp.get_context("spawn"), PROFILES["unlimited"], dynamic=True)
        with self.assertRaises(ValueError):
            shaper.profile_for_round(1)
        shaper.configure_round(1, PROFILES["edge_limited"])
        shaper.configure_round(2, PROFILES["verifier_limited"])
        self.assertEqual(shaper.profile_for_round(1), PROFILES["edge_limited"])
        self.assertEqual(shaper.profile_for_round(2), PROFILES["verifier_limited"])
        for rid in (1, 2, 0, 65536):
            with self.assertRaises(ValueError):
                shaper.configure_round(rid, PROFILES["unlimited"])

    def test_real_processes_observe_each_rounds_profile(self):
        from adaptive_planning.training import make_workers

        groups, _, _ = make_workers(4)
        with Cluster(groups, hidden=8, dynamic_network=True) as cluster:
            for name in ("edge_limited", "verifier_limited", "unlimited"):
                result = cluster.run_round(
                    make_model(hidden=8), "R+B", network_profile=PROFILES[name]
                )
                self.assertTrue(result.accepted, result.errors)
                stats = cluster.collect_metrics(cluster.round_id)
                expected = (
                    stats["update_transfer_tx_bytes"]
                    if name == "edge_limited"
                    else stats["worker_compute_reply_tx_bytes"]
                    + stats["aggregate_transfer_tx_bytes"]
                    if name == "verifier_limited"
                    else 0
                )
                self.assertEqual(stats["pacing_reserved_bytes"], expected)
                self.assertEqual(stats["app_tx_bytes"], stats["app_rx_bytes"])
        self.assertTrue(all(not p.is_alive() for p in cluster.processes.values()))


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.edge = observation_for(PROFILES["edge_limited"], 12, 50260)
        self.verify = observation_for(PROFILES["verifier_limited"], 12, 50260)

    def test_rule_reacts_to_current_rates(self):
        self.assertEqual(network_rule(self.edge, None, 0.025), "F+B")
        self.assertEqual(network_rule(self.verify, None, 0.025), "F+P")

    def test_rule_hysteresis_and_observable_errors(self):
        obs = observation_for(PROFILES["unlimited"], 12, 50260)
        self.assertEqual(network_rule(obs, "R+B", 0.025), "R+B")
        troubled = observation_for(PROFILES["verifier_limited"], 12, 50260, True, True)
        self.assertEqual(network_rule(troubled, None, 0.025), "R+B")

    def test_reserve_prior_counts_both_candidate_copies(self):
        # In verifier_limited mode R sends one additional full candidate sum.
        delta = transfer_prior("R+B", self.verify) - transfer_prior("F+B", self.verify)
        self.assertAlmostEqual(delta, 50260 * 8 / 40e6)

    def test_only_chosen_action_updates(self):
        planner = NetworkOnlinePlanner(1, 0.025, exploration=0.0)
        before = {a: s.copy() for a, s in planner.stats.items()}
        action = planner.choose(self.verify, None)
        self.assertEqual(action, "F+P")
        planner.observe(self.verify, action, True, 0.2, 0.025)
        self.assertEqual(planner.feedback_count(), 1)
        self.assertEqual(planner.stats[action]["success"], 1)
        for other in ACTIONS:
            if other != action:
                self.assertEqual(planner.stats[other], before[other])
        residual = max(0.001, 0.2 - 0.025 - transfer_prior(action, self.verify))
        self.assertAlmostEqual(
            planner.stats[action]["residual_sum"], before[action]["residual_sum"] + residual
        )

    def test_switch_cost_added_once(self):
        planner = NetworkOnlinePlanner(1, 0.025, exploration=0.0)
        before = planner.estimate("F+P", self.verify, None)[1]
        after = planner.estimate("F+P", self.verify, "F+B")[1]
        self.assertAlmostEqual(after - before, 0.025)
        self.assertAlmostEqual(
            predicted_seconds("F+P", self.verify, "F+B", 0.025)
            - predicted_seconds("F+P", self.verify, None, 0.025),
            0.025,
        )

    def test_observation_does_not_expose_trace_or_future(self):
        self.assertEqual(
            {f.name for f in fields(Observation)},
            {
                "workers",
                "dimension",
                "model_mbps",
                "edge_mbps",
                "verifier_mbps",
                "latency_ms",
                "edge_trouble",
                "projection_missing",
            },
        )

    def test_missing_projection_is_not_misclassified_as_bad_edge(self):
        missing = RoundResult(
            errors=["verification: ProtocolError: Missing projection participant"],
            metrics={"candidate_rejections": 1},
        )
        self.assertEqual(observed_issues(missing), (False, True))
        corrupt = RoundResult(
            errors=["verification: ValueError: Подмена агрегированной суммы"],
            metrics={"candidate_rejections": 1},
        )
        self.assertEqual(observed_issues(corrupt), (True, False))

    def test_frozen_ablation_keeps_priors_and_exploration_rng(self):
        learned = NetworkOnlinePlanner(42, 0.025, exploration=1.0)
        frozen = FrozenNetworkPlanner(42, 0.025, exploration=1.0)
        original = {a: s.copy() for a, s in frozen.stats.items()}
        for _ in range(10):
            action = learned.choose(self.verify, None)
            self.assertEqual(frozen.choose(self.verify, None), action)
            learned.observe(self.verify, action, True, 0.2, 0.0)
            frozen.observe(self.verify, action, True, 0.2, 0.0)
        self.assertEqual(frozen.stats, original)
        self.assertEqual(frozen.feedback_count(), 0)
        self.assertEqual(learned.feedback_count(), 10)


class TraceTests(unittest.TestCase):
    def test_reproducible_prefix_independent_of_horizon(self):
        short = make_trace(33026, 20, 12, "faults")
        long = make_trace(33026, 80, 12, "faults")
        self.assertEqual(short, long[:20])
        self.assertEqual(short, make_trace(33026, 20, 12, "faults"))

    def test_phase_lengths_and_profile_changes(self):
        trace = make_trace(33026, 100, 12, "clean")
        runs = []
        last, count = None, 0
        for step in trace:
            if step.profile != last:
                if last is not None:
                    runs.append(count)
                last, count = step.profile, 0
            count += 1
        self.assertTrue(all(4 <= n <= 8 for n in runs))
        self.assertGreater(len(runs), 10)


class AnalysisTests(unittest.TestCase):
    def test_primary_pair_does_not_impute_censored_time(self):
        from adaptive_planning.tcp_dynamic_analyze import paired_comparisons

        def row(repeat, policy, reached, time):
            return dict(
                workers="12",
                repeat=str(repeat),
                policy=policy,
                reached=str(reached),
                t_epsilon_wall_s=time,
            )

        rows = [
            row(0, "network_online", 0, ""),
            row(0, "network_rule", 1, "8"),
            row(1, "network_online", 1, "4"),
            row(1, "network_rule", 1, "5"),
        ]
        primary = next(r for r in paired_comparisons(rows) if r["primary"])
        self.assertEqual((primary["pairs"], primary["complete_pairs"]), (2, 1))
        self.assertEqual(primary["mean_delta_wall_s"], -1.0)
        self.assertIsNone(primary["bootstrap95_low_s"])

    def test_duplicate_run_rejected(self):
        from adaptive_planning.tcp_dynamic_analyze import paired_comparisons

        row = dict(
            workers="12", repeat="0", policy="network_online", reached="1", t_epsilon_wall_s="4"
        )
        with self.assertRaises(ValueError):
            paired_comparisons([row, row])

    def test_ablation_primary_selected_from_metadata(self):
        from adaptive_planning.tcp_dynamic_analyze import paired_comparisons

        def row(policy, time):
            return dict(
                workers="12", repeat="0", policy=policy, reached="1", t_epsilon_wall_s=str(time)
            )

        result = paired_comparisons(
            [row("network_online", 5), row("network_prior", 6), row("network_rule", 7)],
            "network_online - network_prior",
        )
        primary = [r for r in result if r["primary"]]
        self.assertEqual(len(primary), 1)
        self.assertEqual(primary[0]["mean_delta_wall_s"], -1.0)


if __name__ == "__main__":
    unittest.main()
