"""Scientific diagnostics: causal feedback, matched controls, plans and block inference."""

from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest import mock
import unittest
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/collector_v3/kit"))
from model import Policy, Settings
from program import policies, payload_bound, validate_scenario, effects, freeze
from statistics_program import paired, holm, power_plan, required_blocks
from design_tools import grid, costs
from analysis_program import unpack, contrasts
from program_suite import cases, steps, plan_suite, load_suite, FAMILIES
from adaptive_planning import collector_routes
from adaptive_planning.clock import now


class ProgramTests(unittest.TestCase):
    def calibration(self):
        return json.loads(
            (ROOT / "experiments/collector_v3/data/gateway_calibration.json").read_text()
        )

    def policy(self, **settings):
        return Policy(Settings(delay=0, **settings), 1, 4, 2, 8, 2, 53, np.full((1, 4), 8))

    def test_matched_ablations_change_exactly_one_mechanism(self):
        spec = policies(self.calibration())
        for name, expected in [
            ("no_exploration", {"alpha"}),
            ("no_dual", {"price_rule"}),
            ("no_site_price", {"price_rule"}),
            ("no_roster", {"use_current_roster"}),
            ("matched_recent", {"mode"}),
        ]:
            delta = {key for key in spec["joint"] if spec["joint"][key] != spec[name][key]}
            self.assertEqual(delta, expected)
        self.assertNotEqual(spec["legacy_no_ids"]["eta"], spec["joint"]["eta"])

    def test_unexecuted_projection_is_unknown_not_negative(self):
        p = self.policy()
        roster = np.array([[0, 1]])
        p.decide(1, roster, forced_arm=0)
        p.observe(np.full((1, 2, 2), -1.0), np.array([False]), np.zeros((1, 2, 2), bool))
        self.assertEqual(p.done.sum(), 0)
        p.decide(1, roster, forced_arm=0)
        self.assertEqual(p.done.sum(), 0)
        self.assertEqual(p.wins.sum(), 0)
        self.assertEqual(p.design[0, 0, 0, 1], 1)  # only the newly selected pending design

    def test_real_attempt_failure_censors_only_longer_windows(self):
        p = self.policy()
        p.decide(1, np.array([[0, 1]]), forced_arm=0)
        p.observe(np.array([[[50.0, -1.0], [-1.0, -1.0]]]), np.array([False]))
        p.decide(1, np.array([[0, 1]]), forced_arm=0)
        self.assertEqual(p.done[0, 0, 0].tolist(), [1, 1, 1])
        self.assertEqual(p.done[0, 1, 0].tolist(), [1, 1, 0])
        self.assertEqual(p.wins[0, 0, 0].tolist(), [1, 1, 1])

    def test_delayed_labels_do_not_arrive_early(self):
        p = Policy(Settings(delay=2), 1, 4, 2, 8, 2, 53, np.full((1, 4), 8))
        for rd in range(3):
            p.decide(1, np.array([[0, 1]]), forced_arm=0)
            if rd < 2:
                self.assertEqual(p.done.sum(), 0)
            else:
                self.assertGreater(p.done.sum(), 0)
            p.observe(np.full((1, 2, 2), 50.0), np.array([True]))

    def test_uniform_support_and_hard_quota_guard(self):
        p = self.policy(mode="uniform", price_rule="zero")
        p.decide(1, np.array([[0, 1]]))
        self.assertEqual(p.last_decision["chosen_probability"], [0.1])
        p.spent[:] = 2
        p.decide(1, np.array([[0, 1]]))
        self.assertEqual(p.last_decision["chosen_probability"], [0.25])
        with self.assertRaises(ValueError):
            p.decide(1, np.array([[0, 1]]), forced_arm=4)

    def test_payload_lower_bound_matches_protocol_model_dimensions(self):
        from adaptive_planning.tcp_runtime import make_model

        self.assertEqual(payload_bound(2), 192400)
        self.assertEqual(
            payload_bound(2),
            5
            * len(
                __import__(
                    "adaptive_planning.tcp_runtime", fromlist=["flatten_model"]
                ).flatten_model(make_model(17, 64, 0.5))
            ),
        )

    def test_controlled_events_are_frozen_and_not_natural(self):
        s = dict(
            id="shock",
            horizon=8,
            network_mode="controlled",
            random_outage_probability=0.2,
            events=[dict(start=3, end=5, cpu_iterations=[10, 0, 0, 0], failed_edges=[2])],
        )
        validate_scenario(s)
        self.assertEqual(effects(s, 902001), effects(s, 902001))
        self.assertEqual(effects(s, 902001)[1]["cpu_iterations"], [0] * 4)
        self.assertEqual(effects(s, 902001)[2]["cpu_iterations"], [10, 0, 0, 0])
        with self.assertRaises(ValueError):
            validate_scenario(dict(s, network_mode="natural"))
        with self.assertRaises(ValueError):
            validate_scenario(dict(s, deadline_s=float("nan")))

    def test_real_collector_cpu_work_preserves_signed_response(self):
        class Edge:
            workers = ["unused"]

        class Server:
            identity = "e0"
            keys = {}
            session = "test"
            meter = None
            shaper = None

        header = dict(
            sender="coordinator",
            round=1,
            deadline_ns=int((now() + 3) * 1e9),
            body=dict(
                worker=0,
                command={},
                token="test",
                emulated_ms=0,
                workload=dict(cpu_iterations=16, service_ms=0),
            ),
        )
        with mock.patch.object(collector_routes, "rpc", return_value=({"tags": [1]}, b"signed")):
            body, raw = collector_routes.forward(Edge(), header, b"", Server())
        self.assertEqual(raw, b"signed")
        self.assertEqual(body["tags"], [1])
        self.assertEqual(body["controlled_work"]["cpu_iterations"], 16)
        self.assertGreater(body["controlled_work"]["cpu_s"], 0)
        header["body"]["workload"]["unavailable"] = True
        with mock.patch.object(collector_routes, "rpc") as rpc:
            with self.assertRaises(ValueError):
                collector_routes.forward(Edge(), header, b"", Server())
            rpc.assert_not_called()

    def test_article_adapter_matches_original_engine_without_site_pressure(self):
        from article_policy import ArticlePolicy
        from adaptive_planning.pd_linucb import PrimalDualLinUCB

        s = Settings(mode="article_pd", alpha=1.0, ridge=1.0, eta=0.05, initial_price=0.0, delay=0)
        p = ArticlePolicy(s, 1, 4, 2, 8, 8, gateway_caps=np.full((1, 4), 8))
        reference = PrimalDualLinUCB(
            8, {str(i): 1 + int(i >= 4) for i in range(10)}, 16, 8, 1.0, 1.0, 0.05
        )
        for rd in range(8):
            roster = np.array([[rd % 4, (rd + 1) % 4]])
            arm = p.decide(rd % 3, roster)[0]
            expected = reference.select(p.context(rd % 3, roster))
            self.assertEqual(str(arm), expected)
            np.testing.assert_allclose(p.engine.A_inv, reference.A_inv)
            self.assertAlmostEqual(p.engine.dual, reference.dual)
            value = rd % 2
            p.observe(None, np.array([value]))
            reference.reward(rd + 1, value)
        # The last label remains pending until the next decision, by design.
        p.engine.reward(p.pending[-1][1], p.pending[-1][2])
        p.pending.clear()
        np.testing.assert_allclose(p.engine.b, reference.b)

    def test_original_engine_external_mask_is_transactional(self):
        from adaptive_planning.pd_linucb import PrimalDualLinUCB

        p = PrimalDualLinUCB(1, {"cheap": 1, "costly": 2}, 5, 3)
        with self.assertRaises(ValueError):
            p.select([1], allowed_actions=["unknown"])
        self.assertEqual(p.ledger.steps, 0)
        self.assertIsNone(p.select([1], allowed_actions=[]))
        self.assertEqual(p.select([1], allowed_actions=["costly"]), "costly")

    def test_grid_scales_horizon_budget_and_event_times(self):
        base = dict(
            id="x",
            horizon=120,
            budget_bytes=24000000,
            network_mode="controlled",
            events=[dict(start=41, end=80, failed_edges=[0])],
        )
        longer = grid(base, "horizon")["500"]
        self.assertEqual(longer["budget_bytes"], 100000000)
        self.assertEqual(longer["events"][0]["start"], 167)
        self.assertEqual(len(grid(base, "budget")), 6)
        self.assertEqual(costs(10, 10000, [1, 1, 1], 24)[-1]["new_rounds"], 12000)

    def test_all_scenario_families_and_randomized_block_order(self):
        for family in FAMILIES:
            cells = cases(family)
            self.assertTrue(cells)
            for cell in cells:
                validate_scenario(cell["scenario"])
            suite = dict(seeds=[902001, 902002], cells=cells)
            order = list(steps(suite))
            self.assertEqual(order, list(steps(suite)))
            self.assertEqual(len(order), 2 * len(cells))
            self.assertTrue(all(seed == 902001 for seed, cell in order[: len(cells)]))
        self.assertEqual(sum(len(c["modes"]) for c in cases("E8")), 9)

    def test_pair_statistics_keep_blocks_and_correct_family(self):
        a = paired([1, 2, 3, 4], family=3)
        self.assertEqual(a["blocks"], 4)
        self.assertEqual(a["p_exchangeable"], 0.125)
        self.assertLess(a["ci_familywise_t"][0], a["ci_t"][0])
        self.assertEqual(holm({"a": 0.01, "b": 0.04, "c": 0.03}), {"a": 0.03, "b": 0.06, "c": 0.06})
        self.assertEqual(holm({"a": 0.02, "untestable": None}), {"a": 0.04, "untestable": None})
        self.assertIsNone(paired([1])["ci_t"])
        self.assertEqual(paired([0] * 8)["p_exchangeable"], 1)
        self.assertFalse(power_plan([0] * 8, 1)["estimable"])
        self.assertGreater(required_blocks(2, 1, 0.05 / 6), required_blocks(2, 1, 0.05))

    def test_incomplete_pairs_remain_visible(self):
        rows = [
            dict(
                series="s",
                seed=seed,
                policy=p,
                complete=not (seed == 2 and p == "control"),
                accepted_observed=5,
                expected_rounds=10,
            )
            for seed in (1, 2)
            for p in ("joint", "control")
        ]
        out, _ = contrasts(rows)
        self.assertTrue(out[0]["conditional_on_complete_pairs"])
        self.assertEqual(out[0]["missing_pair_seeds"], [2])

    def test_checked_archive_rejects_corruption_and_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            good = tmp / "good.zip"
            with zipfile.ZipFile(good, "w") as z:
                z.writestr("plan.json", b"{}")
                z.writestr("SHA256SUMS", sha256(b"{}").hexdigest() + "  plan.json\n")
            self.assertEqual(unpack(good, tmp / "good")["files"], 2)
            with self.assertRaises(ValueError):
                unpack(good, tmp / "good")
            for name, path, data in [
                ("bad", "plan.json", b"changed"),
                ("traverse", "../evil", b"{}"),
            ]:
                archive = tmp / (name + ".zip")
                with zipfile.ZipFile(archive, "w") as z:
                    z.writestr(path, data)
                    z.writestr("SHA256SUMS", sha256(b"{}").hexdigest() + "  " + path + "\n")
                with self.assertRaises(ValueError):
                    unpack(archive, tmp / name)
                self.assertFalse((tmp / name).exists())

    def test_frozen_plan_preserves_old_data_and_holds_out_seeds(self):
        from distributed_live import dump

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            old = tmp / "old"
            old.mkdir()
            base = dict(
                seeds=[81101],
                horizon=120,
                bytes_per_episode=30000000,
                windows_ms=[80, 130, 200],
                deadline_s=3,
                end_to_end_deadline_s=600,
                population=4,
                k=2,
                calibration=self.calibration(),
            )
            dump(old / "plan.json", base)
            original = (old / "plan.json").read_bytes()
            (old / "plan.sha256").write_text(sha256(original).hexdigest())
            result = freeze(
                old / "plan.json",
                dict(id="x", horizon=120),
                tmp / "new",
                "pilot",
                [902001],
                ["joint", "no_dual"],
            )
            self.assertEqual((old / "plan.json").read_bytes(), original)
            self.assertEqual(result["bytes_per_episode"], 30000000)
            self.assertEqual(
                result["policy_specs"]["joint"]["eta"], result["policy_specs"]["no_dual"]["eta"]
            )
            for phase, seeds, forbidden in [
                ("confirmatory", [904001], []),
                ("pilot", [904001], []),
                ("pilot", [902001], [902001]),
            ]:
                with self.assertRaises(ValueError):
                    freeze(
                        old / "plan.json",
                        dict(id="x"),
                        tmp / "reject",
                        phase,
                        seeds,
                        ["joint"],
                        forbidden,
                    )
            self.assertEqual(asdict(Settings())["delay"], 2)
            long = freeze(
                old / "plan.json",
                dict(id="long", horizon=250, feedback_delay=4),
                tmp / "long",
                "pilot",
                [902001],
                ["joint", "legacy_lp_mean"],
            )
            self.assertEqual(
                result["window_classes"]["902001"], long["window_classes"]["902001"][:120]
            )
            self.assertEqual(long["policy_specs"]["joint"]["delay"], 4)
            self.assertEqual(long["policy_specs"]["legacy_lp_mean"]["delay"], 4)
            suite = plan_suite(
                old / "plan.json",
                tmp / "suite",
                "E1",
                "pilot",
                [902001, 902002],
                ["B24000000", "B30000000"],
            )
            self.assertEqual(suite["new_rounds"], 1440)
            self.assertEqual(load_suite(tmp / "suite")["family"], "E1")

    def test_extended_local_distributed_episode_replay_analysis_and_resume(self):
        # Actual loopback processes/EVM, expressly not an interhost/cloud experiment.
        code = """
from pathlib import Path
from hashlib import sha256
import json
import sys
from distributed_live import plan,run
from program import freeze,verify_complete
from replay_program import reconstruct
from analysis_program import analyze
root=Path.cwd();tmp=Path(sys.argv[1]);placement=tmp/'placement.json'
placement.write_text(json.dumps(dict(workers=['local'],edges=['local']*4)))
plan(root,placement,tmp/'base','pilot',rounds=2,seeds=[81101],modes=['joint'],windows=[1000,1500,2000])
config=freeze(tmp/'base/plan.json',dict(id='local_integration',horizon=2,network_mode='controlled',
    events=[dict(start=1,end=1,cpu_iterations=[16,0,0,0],unavailable_collectors=[0])]),tmp/'run','pilot',[902001,902002],['joint','no_dual','article_pd'])
runtime=dict(url='loopback',bind='127.0.0.1',port=0,network_mode='controlled')
run(root,tmp/'run',runtime)
before={str(p):sha256(p.read_bytes()).hexdigest() for p in (tmp/'run').rglob('*') if p.is_file()}
for seed in config['seeds']:
 for mode in config['modes']:
  directory=tmp/'run'/f'{seed}_{mode}'
  verify_complete(directory,config)
  assert len(reconstruct(tmp/'run',seed,mode))==2
  for path in directory.glob('round-*.json'):
   row=json.loads(path.read_text());assert row['policy_decision']
   assert row['emulated_link_ms'] is None
   if row['accepted']:assert row['telemetry']['app_tx_bytes']>=192400
result=analyze([('local_only',tmp/'run')],tmp/'analysis')
assert result['complete'] and result['observed_rounds']==12
assert (tmp/'analysis/figures/13_model_quality.svg').exists()
assert before=={str(p):sha256(p.read_bytes()).hexdigest() for p in (tmp/'run').rglob('*') if p.is_file()}
run(root,tmp/'run',runtime)
old_directory=tmp/'run/902001_joint'
old_attempt=json.loads((old_directory/'attempt.json').read_text())['id']
(old_directory/'complete.json').unlink()
(old_directory/'failure.json').write_text(json.dumps(dict(progress={'stage':'test_interruption'},partial_result=None)))
try:run(root,tmp/'run',runtime)
except ValueError:pass
else:raise AssertionError('Interrupted episode silently reset')
run(root,tmp/'run',dict(runtime,retry_incomplete=True))
assert json.loads((old_directory/'attempt.json').read_text())['id']!=old_attempt
archives=list((tmp/'run/attempts/902001_joint').iterdir());assert len(archives)==1
assert len(list(archives[0].glob('round-*.json')))==2
retry_analysis=analyze([('local_only',tmp/'run')],tmp/'analysis-with-attempt')
assert not retry_analysis['no_retries'] and retry_analysis['prior_attempt_records']==3
path=tmp/'run/902001_joint/round-0001.json';path.write_text('{}')
try:verify_complete(path.parent,config)
except ValueError:pass
else:raise AssertionError('Tampered resume accepted')
"""
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, "-c", code, tmp],
                cwd=ROOT,
                capture_output=True,
                text=True,
                env=dict(
                    os.environ,
                    ADAPTIVE_CLOCK="wall",
                    PYTHONPATH=str(ROOT / "experiments/collector_v3/kit") + os.pathsep + str(ROOT),
                    OPENBLAS_NUM_THREADS="1",
                    OMP_NUM_THREADS="1",
                    MPLBACKEND="Agg",
                ),
                timeout=240,
            )
            self.assertEqual(result.returncode, 0, result.stdout[-1000:] + result.stderr[-5000:])


if __name__ == "__main__":
    unittest.main()
