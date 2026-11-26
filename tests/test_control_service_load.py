"""Failure injection and long-journal regressions for the real HTTPS control service."""

from concurrent.futures import ThreadPoolExecutor
import socketserver
import threading
from contextlib import contextmanager
import json
from pathlib import Path
import socket
import sqlite3
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from adaptive_planning.distributed_control import (
    Ledger,
    ControlServer,
    ControlClient,
    RemoteBudget,
    _ConnectionPool,
)


@contextmanager
def service_fixture(**options):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        key = root / "tls.key"
        cert = root / "tls.crt"
        config = root / "tls.cnf"
        config.write_text(
            "[req]\ndistinguished_name=dn\nx509_extensions=ext\nprompt=no\n[dn]\nCN=loopback\n[ext]\nsubjectAltName=IP:127.0.0.1\nbasicConstraints=critical,CA:TRUE\n"
        )
        subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-nodes",
                "-days",
                "1",
                "-config",
                str(config),
                "-keyout",
                str(key),
                "-out",
                str(cert),
            ],
            check=True,
            capture_output=True,
        )
        ledger = Ledger(root / "ledger.sqlite", 10000000)
        secret = bytes([7]) * 32
        server = ControlServer(
            ("127.0.0.1", 0),
            ledger,
            {"worker": secret},
            options.pop("evidence", lambda _: {}),
            str(cert),
            str(key),
            log_path=root / "server.jsonl",
            **options,
        ).start()
        client = ControlClient(
            f"https://127.0.0.1:{server.http.server_port}",
            "worker",
            secret.hex(),
            str(cert),
            timeout=2,
            log_path=root / "client.jsonl",
        )
        try:
            yield server, client, ledger, root
        finally:
            client.close()
            server.close()
            ledger.close()


class ServiceRegressionTests(unittest.TestCase):
    def test_header_write_disconnect_is_logged_after_commit_and_retry_is_idempotent(self):
        with service_fixture() as (server, client, ledger, root):
            request = dict(id="lost-header", size=80)
            with patch.object(
                server.http.RequestHandlerClass,
                "end_headers",
                side_effect=BrokenPipeError(32, "Broken pipe"),
            ):
                with self.assertRaises(OSError):
                    client.call("reserve", request)
            state = ledger.apply("worker", "state", {})
            self.assertEqual((state["reserved"], state["frames"]), (80, 1))
            self.assertTrue(client.call("reserve", request)["admitted"])
            client.call("settle", dict(id="lost-header", sent=23))
            self.assertEqual(ledger.audit(), dict(spent=23, reserved=0, frames=1))
            rows = [json.loads(s) for s in (root / "server.jsonl").read_text().splitlines()]
            failures = [r for r in rows if r["event"] == "response_lost"]
            self.assertEqual(len(failures), 3)
            self.assertTrue(
                all(r["stage"] == "headers" and r["operation_applied"] for r in failures)
            )
            self.assertEqual(client.telemetry.snapshot()["counts"]["client_retry"], 2)

    def test_truncated_body_is_retried_and_unsent_frame_is_canceled(self):
        original = socketserver._SocketWriter.write

        def disconnect_body(writer, data):
            if data.startswith(b"{"):
                raise BrokenPipeError(32, "Broken pipe")
            return original(writer, data)

        with service_fixture() as (server, client, ledger, root):
            with patch.object(socketserver._SocketWriter, "write", disconnect_body):
                with self.assertRaises(OSError):
                    RemoteBudget(client).reserve(80)
            self.assertEqual(ledger.audit(), dict(spent=0, reserved=0, frames=1))
            rows = [json.loads(s) for s in (root / "server.jsonl").read_text().splitlines()]
            self.assertTrue(
                any(r["event"] == "response_lost" and r["stage"] == "body" for r in rows)
            )
            self.assertGreater(client.telemetry.snapshot()["counts"]["client_retry"], 0)

    def test_slow_tls_peer_cannot_block_the_accept_loop(self):
        with service_fixture(max_workers=4) as (server, client, _, _root):
            slow = socket.create_connection(("127.0.0.1", server.http.server_port))
            try:
                for _ in range(100):
                    if server.http.load()["active_connections"]:
                        break
                    time.sleep(0.005)
                start = time.perf_counter()
                self.assertEqual(client.call("state", {})["spent"], 0)
                self.assertLess(time.perf_counter() - start, 1.5)
            finally:
                slow.close()

    def test_thread_creation_failure_releases_capacity_and_recovers(self):
        with service_fixture(max_workers=2) as (server, client, ledger, _):
            with patch.object(
                threading.Thread, "start", side_effect=RuntimeError("can't start new thread")
            ):
                with self.assertRaises(OSError):
                    client.call("state", {})
            self.assertEqual(server.http.load()["active_connections"], 0)
            self.assertEqual(server.http.load()["queued_connections"], 0)
            self.assertGreater(server.telemetry.snapshot()["counts"]["worker_start_error"], 0)
            self.assertEqual(client.call("state", {})["spent"], 0)
            self.assertEqual(ledger.audit()["reserved"], 0)

    def test_total_deadline_is_not_multiplied_by_retries(self):
        def delayed(_):
            time.sleep(0.15)
            return {}

        with service_fixture(evidence=delayed) as (server, client, _, root):
            start = time.perf_counter()
            with self.assertRaises(OSError):
                client.call("evidence", {}, timeout=0.04)
            self.assertLess(time.perf_counter() - start, 0.12)
            time.sleep(0.2)
            rows = [json.loads(s) for s in (root / "server.jsonl").read_text().splitlines()]
            self.assertTrue(any(r["event"] in ("response_lost", "response_written") for r in rows))
            self.assertEqual(client.telemetry.snapshot()["counts"]["client_transport_error"], 1)

    def test_reuse_and_multithreaded_fair_credit(self):
        pool = _ConnectionPool(1)
        pool.get(0.1)
        accepted = threading.Event()

        def waiter():
            connection = pool.get(2)
            accepted.set()
            pool.put(connection)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(waiter)
            for _ in range(200):
                with pool.condition:
                    if pool.waiters:
                        break
                time.sleep(0.001)
            pool.put(None)
            connection = pool.get(2)
            self.assertTrue(accepted.is_set())
            pool.put(connection)
            future.result()
        with service_fixture(max_workers=4) as (server, client, ledger, _):

            def send(i):
                body = dict(id=str(i), size=40)
                client.call("reserve", body)
                client.call("settle", dict(id=str(i), sent=31))

            with ThreadPoolExecutor(max_workers=12) as pool:
                list(pool.map(send, range(240)))
            self.assertEqual(ledger.audit(), dict(spent=240 * 31, reserved=0, frames=240))
            counts = server.telemetry.snapshot()["counts"]
            self.assertLess(counts["connection_open"], 20)
            self.assertLessEqual(server.http.load()["peak_connections"], 4)

    def test_trickled_body_cannot_extend_the_absolute_deadline(self):
        original = socketserver._SocketWriter.write

        def trickle(writer, data):
            if data.startswith(b"{"):
                for part in data[:12]:
                    original(writer, bytes([part]))
                    time.sleep(0.015)
                return original(writer, data[12:])
            return original(writer, data)

        with service_fixture() as (_server, client, _ledger, _root):
            client.call("state", {})
            with patch.object(socketserver._SocketWriter, "write", trickle):
                start = time.perf_counter()
                with self.assertRaises(OSError):
                    client.call("state", {}, timeout=0.05)
                self.assertLess(time.perf_counter() - start, 0.15)

    def test_legacy_migration_and_no_full_journal_scans_per_operation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.sqlite"
            with sqlite3.connect(path) as db:
                db.executescript(
                    "CREATE TABLE config(limit_bytes INTEGER); INSERT INTO config VALUES (1000000); CREATE TABLE denials(owner TEXT,created REAL); CREATE TABLE frames(id TEXT PRIMARY KEY,owner TEXT,size INTEGER,sent INTEGER,created REAL);"
                )
                db.executemany(
                    "INSERT INTO frames VALUES (?,?,?,?,0)",
                    [(str(i), "worker", 40, 30 if i % 2 else None) for i in range(3000)],
                )
            ledger = Ledger(path, 1000000)
            try:
                self.assertEqual(ledger.audit(), dict(spent=45000, reserved=60000, frames=3000))
                traced = []
                ledger._db.set_trace_callback(traced.append)
                ledger.apply("worker", "reserve", dict(id="next", size=50))
                ledger.apply("worker", "settle", dict(id="next", sent=41))
                ledger.apply("worker", "state", {})
                self.assertFalse(any("SUM(" in q.upper() or "COUNT(" in q.upper() for q in traced))
                self.assertEqual(ledger.audit(), dict(spent=45041, reserved=60000, frames=3001))
            finally:
                ledger.close()


if __name__ == "__main__":
    unittest.main()
