"""Safety and recovery of the distributed credit and independent lease checks."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
import subprocess
import os
import sys
import tempfile
import unittest

from adaptive_planning.distributed_control import Ledger, ControlServer, ControlClient, RemoteBudget
from adaptive_planning.byte_budget import ByteBudgetExceeded, reserve_frame
from adaptive_planning.lease_guard import LeaseGuard, read_evidence
from adaptive_planning.ocp_chain import OcpChain
from eth_hash.auto import keccak

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/collector_v3/kit"))
from lease import deploy, reserve


class LedgerTests(unittest.TestCase):
    def test_atomic_concurrent_admission_and_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.sqlite"
            ledger = Ledger(path, 1000)

            def admit(i):
                try:
                    ledger.apply(str(i), "reserve", dict(id=str(i), size=60))
                    return True
                except ByteBudgetExceeded:
                    return False

            with ThreadPoolExecutor(max_workers=24) as pool:
                accepted = list(pool.map(admit, range(60)))
            self.assertEqual(sum(accepted), 16)
            state = Ledger(path, 1000).apply("x", "state", {})
            self.assertEqual(state["reserved"], 960)
            with self.assertRaises(ValueError):
                Ledger(path, 1001)

    def test_retries_conflicts_and_crashed_sender_keep_credit(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(Path(tmp) / "credit", 100)
            args = dict(id="a", size=80)
            ledger.apply("w0", "reserve", args)
            ledger.apply("w0", "reserve", args)
            with self.assertRaises(ValueError):
                ledger.apply("w1", "settle", dict(id="a", sent=0))
            with self.assertRaises(ByteBudgetExceeded):
                ledger.apply("w1", "reserve", dict(id="b", size=21))
            ledger.apply("w0", "settle", dict(id="a", sent=31))
            ledger.apply("w0", "settle", dict(id="a", sent=31))
            with self.assertRaises(ValueError):
                ledger.apply("w0", "settle", dict(id="a", sent=0))
            self.assertEqual(ledger.apply("w0", "state", {})["spent"], 31)
            # Cancellation can precede the delayed admission, without granting credit twice.
            for ident in ("late", "early"):
                if ident == "early":
                    ledger.apply("w0", "reserve", dict(id=ident, size=10))
                ledger.apply("w0", "cancel", dict(id=ident, size=10))
                self.assertFalse(ledger.apply("w0", "reserve", dict(id=ident, size=10))["admitted"])
            with self.assertRaises(ValueError):
                ledger.apply("w0", "cancel", dict(id="a", size=80))
            self.assertEqual(ledger.apply("w0", "state", {})["reserved"], 0)

            class LostAckClient:
                timeout = 0.1

                def call(self, operation, body, timeout=None):
                    reply = ledger.apply("w0", operation, body)
                    if operation == "reserve":
                        raise TimeoutError("Admission reply lost")
                    return reply

            with self.assertRaises(TimeoutError):
                RemoteBudget(LostAckClient()).reserve(10)
            self.assertEqual(ledger.apply("w0", "state", {})["reserved"], 0)

    def test_http_auth_and_partial_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            key = bytes([1]) * 32
            service = ControlServer(
                ("127.0.0.1", 0), Ledger(Path(tmp) / "credit", 100), {"w0": key}, lambda _: {}
            ).start()
            try:
                client = ControlClient(
                    f"http://127.0.0.1:{service.http.server_port}", "w0", key.hex()
                )
                remote = RemoteBudget(client)
                with self.assertRaises(OSError):
                    with reserve_frame(remote, 80) as sent:
                        sent(23)
                        raise OSError("disconnect")
                state = remote.state()
                self.assertEqual((state["spent"], state["reserved"]), (23, 0))
                with self.assertRaises(ByteBudgetExceeded):
                    ControlClient(client.url, "w0", bytes(32).hex()).call("state", {})
            finally:
                service.close()

    def test_tls_pin_and_unreachable_admission_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cert = base / "cert.pem"
            key = base / "key.pem"
            conf = base / "openssl.cnf"
            conf.write_text(
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
                    str(conf),
                    "-keyout",
                    str(key),
                    "-out",
                    str(cert),
                ],
                check=True,
                capture_output=True,
            )
            secret = bytes([4]) * 32
            server = ControlServer(
                ("127.0.0.1", 0),
                Ledger(base / "ledger", 100),
                {"w0": secret},
                lambda _: {},
                str(cert),
                str(key),
            ).start()
            url = f"https://127.0.0.1:{server.http.server_port}"
            client = ControlClient(url, "w0", secret.hex(), str(cert), timeout=0.5)
            try:
                with reserve_frame(RemoteBudget(client), 20) as sent:
                    sent(12)
                self.assertEqual(client.call("state", {})["spent"], 12)
                with self.assertRaises(OSError):
                    ControlClient(url, "w0", secret.hex(), timeout=0.5).call("state", {})
            finally:
                server.close()
            with self.assertRaises((OSError, TimeoutError)):
                with reserve_frame(RemoteBudget(client), 10):
                    self.fail("Sending without admission")

    def test_node_source_mismatch_rejected_before_start(self):
        from adaptive_planning.node import build

        with self.assertRaises(ValueError):
            build({"source_hashes": {"adaptive_planning/node.py": "00" * 32}})

    def test_non_tls_remote_service_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                ControlServer(("0.0.0.0", 0), Ledger(Path(tmp) / "credit", 100), {}, lambda _: {})
        with self.assertRaises(ValueError):
            ControlClient("http://192.0.2.1:9443", "w0", bytes(32).hex())


class LeaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chain = OcpChain.local(4, 2)
        cls.book, cfg = deploy(cls.chain, {"test": "distributed"}, 2, [4] * 4)
        cls.pin = dict(
            chain_id=cls.chain.chain_id,
            source=cls.chain.contract.address,
            lease=cls.book.address,
            source_code=cls.chain.code_hash.hex(),
            lease_code=keccak(bytes(cls.chain.w3.eth.get_code(cls.book.address))).hex(),
            cfg=cfg.hex(),
            confirmations=2,
        )
        cls.ticket = cls.chain.run_local_ceremony(
            1, seeds=[sha256(str(i).encode()).digest() for i in range(4)]
        )
        granted = reserve(cls.chain, cls.book, cfg, cls.ticket, (0, 1), 130)
        cls.chain.w3.provider.ethereum_tester.mine_blocks(1)
        cls.request = dict(
            round=1,
            mask=3,
            window=130,
            digest=granted["digest"],
            block=granted["block"],
            block_hash=cls.chain.w3.eth.get_block(granted["block"]).hash.hex(),
            selected=cls.ticket["selected"],
        )

    def guard(self, identity):
        return LeaseGuard(identity, self.pin, lambda x: read_evidence(self.chain.w3, self.pin, x))

    def test_roles_verify_actual_contract_and_reject_changed_grants(self):
        guard = self.guard("e0")
        guard.install(self.request)
        for key, value in [
            ("mask", 5),
            ("window", 131),
            ("digest", "ab" * 32),
            ("block_hash", "ab" * 32),
            ("selected", [0, 1, 2]),
        ]:
            changed = dict(self.request, **{key: value})
            with self.assertRaises(ValueError):
                self.guard("e0").install(changed)
        with self.assertRaises(ValueError):
            self.guard("e0").authorize(dict(round=1, kind="aggregate"))
        with self.assertRaises(ValueError):
            guard.authorize(dict(round=2, kind="aggregate"))
        worker_absent = next(i for i in range(4) if i not in self.ticket["selected"])
        missing = self.guard(f"w{worker_absent}")
        missing.install(self.request)
        with self.assertRaises(ValueError):
            missing.authorize(dict(round=1, kind="compute"))
        bad = self.guard("e3")
        bad.install(self.request)
        with self.assertRaises(ValueError):
            bad.authorize(
                dict(round=1, kind="relay_projection", body={"worker": self.ticket["selected"][0]})
            )
        i = self.ticket["selected"][0]
        worker = self.guard(f"w{i}")
        worker.install(self.request)
        worker.authorize(dict(round=1, kind="relayed_projection", sender=f"w{i}", recipient="e0"))

    def test_external_mode_requires_consensus_finalized_grant(self):
        pin = dict(self.pin, require_finalized=True)
        evidence = read_evidence(self.chain.w3, self.pin, self.request)
        guard = LeaseGuard("e0", pin, lambda _: evidence)
        with self.assertRaises(ValueError):
            guard.install(self.request)
        evidence["finalized_block"] = self.request["block"] - 1
        with self.assertRaises(ValueError):
            guard.install(self.request)
        evidence["finalized_block"] = self.request["block"]
        guard.install(self.request)

    def test_local_distributed_collector_entry_point(self):
        code = """
from pathlib import Path
import json
from distributed_live import plan,run
root=Path.cwd();out=Path(__import__('sys').argv[1])/'run'
p=out.parent/'placement.json';p.write_text(json.dumps({'workers':['local'],'edges':['local']*4}))
plan(root,p,out,'pilot',rounds=2,seeds=[81101],modes=['joint'])
run(root,out,dict(url='loopback',bind='127.0.0.1',port=0,network_mode='natural'))
rows=[json.loads(f.read_text()) for f in sorted((out/'81101_joint').glob('round-*.json'))]
assert len(rows)==2
assert any(r['accepted'] for r in rows), [(r['errors'],r['metrics']) for r in rows]
for row in rows:
 assert row['network_mode']=='natural' and row['emulated_link_ms'] is None
 assert len(row['node_grant_checks'])==8
 assert row['byte_ledger']['spent']==sum(x['telemetry']['app_tx_bytes'] for x in rows[:row['round']])
 assert row['total_round_s']>=row['data_and_update_s']
from importlib.util import spec_from_file_location,module_from_spec
spec=spec_from_file_location('audit',root/'deploy/timeweb/summarize.py');audit=module_from_spec(spec);spec.loader.exec_module(audit)
assert audit.audit(out)['complete']
"""
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(
                os.environ,
                ADAPTIVE_CLOCK="wall",
                PYTHONPATH=str(ROOT / "experiments/collector_v3/kit") + os.pathsep + str(ROOT),
                OPENBLAS_NUM_THREADS="1",
                OMP_NUM_THREADS="1",
            )
            result = subprocess.run(
                [sys.executable, "-c", code, tmp],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=180,
            )
            self.assertEqual(result.returncode, 0, result.stdout[-1000:] + result.stderr[-4000:])


if __name__ == "__main__":
    unittest.main()
