import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def probe(latency, outage):
    env = dict(os.environ, ADAPTIVE_CLOCK="wall")
    done = subprocess.run([sys.executable, "-m", "adaptive_planning.stand_check", "--latency-ms",
                           str(latency), "--outage", outage, "--repeats", "2"],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
    if done.returncode:
        raise AssertionError(done.stdout[-2000:] + done.stderr[-2000:])
    return json.loads(done.stdout)


class DistributedStandTests(unittest.TestCase):
    def check(self, out):
        self.assertTrue(out["passed"])
        self.assertEqual(out["role_processes"], 8)  # four workers and four edges, own processes
        self.assertEqual(out["primary_down_F"]["outages"], ["e0"])
        self.assertTrue(out["primary_down_F"]["model_unchanged"])
        self.assertEqual(out["gate_switches"], 6)
        self.assertGreater(out["app_tx_bytes_last_round"], 0)
        self.assertEqual(out["alive_after_close"], 0)

    def test_refused_edges_over_wan_like_relays(self):
        out = probe(25, "refuse")
        self.check(out)
        self.assertGreater(out["healthy"]["median_wall_s"], 0.1)  # relay latency is real
        self.assertGreater(min(out["control_rtt_ms"].values()), 40)

    def test_silent_edges_cost_the_rpc_timeout(self):
        out = probe(0, "silent")
        self.check(out)
        self.assertGreater(out["primary_down_R"]["wall_s"], 0.9)  # waited for the hung edge

    def test_ssh_launch_path(self):
        # A test double for ssh runs the exact remote command line locally.
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            hosts = {n: "127.0.0.1" for n in ("w0", "w1", "w2", "w3", "e0", "e1", "e2", "e3")}
            placement = Path(tmp) / "placement.json"
            placement.write_text(json.dumps(dict(roles=hosts, ssh=dict(
                ssh_user="ubuntu", ssh_options=["-i", "~/.ssh/stand", "-o", "BatchMode=yes"],
                remote_dir=str(ROOT), remote_python=sys.executable))))
            log = Path(tmp) / "targets.log"
            env = dict(os.environ, ADAPTIVE_CLOCK="wall", FAKE_SSH_LOG=str(log),
                       PATH=str(ROOT / "tests/fixtures/fake_ssh") + os.pathsep + os.environ["PATH"])
            done = subprocess.run([sys.executable, "-m", "adaptive_planning.stand_check", "--placement",
                                   str(placement), "--repeats", "1"], cwd=ROOT, env=env,
                                  capture_output=True, text=True, timeout=180)
            self.assertEqual(done.returncode, 0, done.stderr[-2000:])
            self.assertTrue(json.loads(done.stdout)["passed"])
            self.assertEqual(len(log.read_text().splitlines()), 8)  # one ssh session per role

    def test_wall_clock_is_required(self):
        env = dict(os.environ, ADAPTIVE_CLOCK="monotonic")
        code = "from adaptive_planning.distributed import RemoteCluster; RemoteCluster([], placement={})"
        done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=60)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("ADAPTIVE_CLOCK=wall", done.stderr)


if __name__ == "__main__":
    unittest.main()
