"""Ensure the v0.6.1 update preserves old scientific settings and SQLite results."""

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class UpdatePreservationTests(unittest.TestCase):
    def test_clone_preserves_scientific_plan_and_never_overwrites_old_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "original"
            old.mkdir()
            settings = dict(
                horizon=120,
                seeds=list(range(8)),
                modes=["joint", "no_ids", "scalar", "recent", "round_lin", "lp_mean", "greedy"],
                order={"0": ["recent", "joint"]},
                windows_ms=[80, 130, 200],
                bytes_per_episode=30000000,
                placement={
                    "workers": ["worker-a", "worker-b"],
                    "edges": ["worker-a", "worker-a", "worker-b", "worker-b"],
                },
                calibration={"opaque": "unchanged"},
                network_mode="natural",
                source_hashes={"old": "snapshot"},
                commit="3850ae5",
                created=1,
            )
            raw = (json.dumps(settings, sort_keys=True) + "\n").encode()
            (old / "plan.json").write_bytes(raw)
            (old / "plan.sha256").write_text(sha256(raw).hexdigest() + "\n")
            (old / "round-0001.json").write_text("original result")
            new = root / "new"
            done = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "deploy/timeweb/clone_plan.py"),
                    str(old / "plan.json"),
                    str(new),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(done.returncode, 0, done.stderr)
            copied = json.loads((new / "plan.json").read_text())
            for key, value in settings.items():
                if key not in ("source_hashes", "commit", "created"):
                    self.assertEqual(copied[key], value)
            self.assertEqual((old / "plan.json").read_bytes(), raw)
            self.assertEqual((old / "round-0001.json").read_text(), "original result")
            self.assertEqual(copied["parent_plan_sha256"], sha256(raw).hexdigest())
            self.assertEqual(
                (new / "plan.sha256").read_text().strip(),
                sha256((new / "plan.json").read_bytes()).hexdigest(),
            )
            again = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "deploy/timeweb/clone_plan.py"),
                    str(old / "plan.json"),
                    str(old),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(again.returncode, 0)
            self.assertEqual((old / "plan.json").read_bytes(), raw)

    def test_read_only_audit_does_not_migrate_or_refund_old_database(self):
        spec = importlib.util.spec_from_file_location(
            "audit_budget", ROOT / "deploy/timeweb/audit_budget.py"
        )
        audit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(audit)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "budget.sqlite"
            with sqlite3.connect(path) as db:
                db.executescript(
                    "CREATE TABLE config(limit_bytes INTEGER); INSERT INTO config VALUES (1000); CREATE TABLE frames(id TEXT PRIMARY KEY,owner TEXT,size INTEGER,sent INTEGER,created REAL);"
                )
                db.executemany(
                    "INSERT INTO frames VALUES (?,?,?,?,0)",
                    [("a", "w0", 80, 31), ("b", "w1", 90, None)],
                )
            before = path.read_bytes()
            result = audit.audit(path)
            self.assertTrue(result["valid"])
            self.assertEqual(
                (result["spent"], result["reserved"], result["unsettled_frames"]), (31, 90, 1)
            )
            self.assertEqual(path.read_bytes(), before)
            with sqlite3.connect(path) as db:
                self.assertIsNone(
                    db.execute("SELECT name FROM sqlite_master WHERE name='totals'").fetchone()
                )


if __name__ == "__main__":
    unittest.main()
