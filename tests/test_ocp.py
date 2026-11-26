import copy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from time import time
from unittest.mock import patch
from web3.exceptions import BlockNotFound
import unittest

from eth_hash.auto import keccak

from adaptive_planning.ocp_chain import OcpChain, verify_certificate
from adaptive_planning.ocp_randomness import BeaconSpec, LocalTestBeacon, commitment, select_clients
from adaptive_planning.paths import ROOT


class RandomnessTests(unittest.TestCase):
    def test_keccak_is_ethereum_not_nist_sha3(self):
        self.assertEqual(
            keccak(b"").hex(), "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
        )

    def test_commitment_binds_identity_round_chain_and_contract(self):
        values = [bytes(32), "0x" + "11" * 20, 1, 1, "0x" + "22" * 20]
        original = commitment(*values)
        for i, v in enumerate([b"x" * 32, "0x" + "33" * 20, 2, 2, "0x" + "44" * 20]):
            changed = values.copy()
            changed[i] = v
            self.assertNotEqual(original, commitment(*changed))

    def test_shuffle_rejects_modulo_bias(self):
        calls = []

        def words(counter):
            calls.append(counter)
            return [0, 1, 2][counter]

        selected = select_clients(bytes(32), 3, 3, words)
        self.assertEqual(calls, [0, 1, 2])
        self.assertEqual(set(selected), {0, 1, 2})
        self.assertEqual(selected, (2, 0, 1))

    def test_real_published_drand_vector(self):
        spec = BeaconSpec(**json.loads((ROOT / "configs/drand_mainnet.json").read_text()))
        proof = json.loads(
            (Path(__file__).parent / "fixtures/drand_default_1000000.json").read_text()
        )
        value = spec.verify(proof, 1000000, spec.time_for(1000000) - 1)
        self.assertEqual(value.hex(), proof["randomness"])
        bad = copy.deepcopy(proof)
        bad["signature"] = "00" + proof["signature"][2:]
        with self.assertRaises(ValueError):
            spec.verify(bad, 1000000, 0)
        with self.assertRaises(ValueError):
            spec.verify(proof, 1000001, 0)
        with self.assertRaises(ValueError):
            spec.verify(proof, 1000000, spec.time_for(1000000))
        with self.assertRaises(ValueError):
            replace(spec, scheme="sha256-only")

    def test_predictable_beacon_does_not_prevent_last_revealer_bias(self):
        # Regression counterexample to the article's overly broad "one honest
        # revealed seed suffices" wording. With a known beacon, aborts can matter.
        honest = sha256(b"honest").digest()
        base = select_clients(honest, 2, 1)
        for i in range(100):
            malicious = sha256(str(i).encode()).digest()
            mixed = bytes(a ^ b for a, b in zip(honest, malicious))
            if select_clients(mixed, 2, 1) != base:
                break
        else:
            self.fail("Expected a selective-abort counterexample")
        self.assertNotEqual(select_clients(mixed, 2, 1), base)


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chain = OcpChain.local(4, 2)

    def setUp(self):
        self.snapshot = self.chain.w3.provider.ethereum_tester.take_snapshot()
        self.chain.receipts = self.chain.receipts[:1]
        self.chain.validated.clear()
        self.chain.open_blocks.clear()
        self.chain.final_blocks.clear()
        self.chain.model_blocks.clear()
        self.chain.model_events.clear()
        self.chain.late_events.clear()

    def tearDown(self):
        self.chain.w3.provider.ethereum_tester.revert_to_snapshot(self.snapshot)

    def test_local_clock_and_bounded_canonical_event_queries(self):
        c = self.chain
        self.assertGreater(c.timestamp, time() + 3600)
        ticket = c.run_local_ceremony(1)
        with patch.object(c.w3.eth, "get_logs", wraps=c.w3.eth.get_logs) as queries:
            self.assertEqual(c.certificate(1), ticket)
        self.assertTrue(queries.call_args_list)
        for call in queries.call_args_list:
            params = call.args[0]
            self.assertGreaterEqual(params["fromBlock"], c.open_blocks[1])
            self.assertLessEqual(params["toBlock"], c.final_blocks[1])

    def test_export_uses_actual_receipts_without_rescanning_events(self):
        c = self.chain
        c.run_local_ceremony(1)
        digest = c.record_model(1, b"model")
        expected = c._events("ModelRecorded", 1)
        with patch.object(c, "_events", side_effect=AssertionError("Historical rescan")):
            exported = c.export()
        self.assertEqual(exported["model_events"], expected)
        self.assertEqual(exported["model_events"][0]["modelHash"], digest)

    def test_ceremony_contract_python_parity_and_non_reveal(self):
        c = self.chain
        t = c.run_local_ceremony(1, missing=(2,))
        self.assertEqual(
            verify_certificate(t, c.spec), tuple(c.contract.functions.selected(1).call())
        )
        self.assertEqual(t["non_reveals"], [c.clients[2]])
        self.assertEqual(c.contract.functions.missedReveals(c.clients[2]).call(), 1)
        self.assertEqual(c.authorize(t, 1, 4), tuple(t["selected"]))
        self.assertTrue(all(r["status"] == 1 for r in c.receipts))
        self.assertGreater(c.export()["gas_used"], 0)

    def test_no_reveals_uses_beacon_and_records_all_missed(self):
        c = self.chain
        t = c.run_local_ceremony(1, missing=(0, 1, 2, 3))
        self.assertEqual(t["mixed"], t["beacon_proof"]["randomness"])
        self.assertEqual(len(t["non_reveals"]), 4)
        self.assertTrue(c._events("RoundFinal", 1)[0]["noReveals"])

    def test_bad_phase_wrong_opening_and_duplicate_commits_revert(self):
        c = self.chain
        seed = bytes(32)
        commit_end = c.timestamp + 20
        reveal_end = commit_end + 20
        c.open_round(1, commit_end, reveal_end, c.spec.round_after(reveal_end))
        c.commit(1, 0, seed)
        for fn in [lambda: c.commit(1, 0, seed), lambda: c.reveal(1, 0, seed)]:
            with self.assertRaises(ValueError):
                fn()
        c.advance_local(commit_end)
        with self.assertRaises(ValueError):
            c.commit(1, 1, seed)
        with self.assertRaises(ValueError):
            c.reveal(1, 0, b"x" * 32)
        c.reveal(1, 0, seed)
        with self.assertRaises(ValueError):
            c.reveal(1, 0, seed)

    def test_late_reveal_logged_but_excluded(self):
        c = self.chain
        seed = bytes(32)
        a = c.timestamp + 20
        b = a + 20
        c.open_round(1, a, b, c.spec.round_after(b))
        c.commit(1, 0, seed)
        c.advance_local(b)
        c.reveal(1, 0, seed)
        self.assertFalse(c.contract.functions.revealed(1, c.clients[0]).call())
        self.assertEqual(len(c._events("LateReveal", 1)), 1)
        c.finalize(1, LocalTestBeacon().fetch(c.spec.round_after(b)))
        ticket = c.certificate(1)
        verify_certificate(ticket, c.spec)
        self.assertEqual(ticket["non_reveals"], [c.clients[0]])

    def test_only_oracle_and_future_beacon_round(self):
        c = self.chain
        a = c.timestamp + 20
        b = a + 20
        with self.assertRaises(ValueError):
            c.open_round(1, a, b, c.spec.round_after(a))
        with self.assertRaises(ValueError):
            c.transact(
                c.contract.functions.openRound(1, a, b, c.spec.round_after(b), b + 1), c.clients[0]
            )
        c.open_round(1, a, b, c.spec.round_after(b))
        with self.assertRaises(ValueError):
            c.finalize(1, LocalTestBeacon().fetch(c.spec.round_after(b)))

    def test_coordinator_forgery_and_replayed_roster_rejected(self):
        c = self.chain
        t = c.run_local_ceremony(1)
        for field, value in [
            ("selected", list(reversed(t["selected"]))),
            ("mixed", "00" * 32),
            ("round", 2),
            ("clients", list(reversed(t["clients"]))),
            ("config_hash", "ff" * 32),
        ]:
            bad = copy.deepcopy(t)
            bad[field] = value
            with self.assertRaises(ValueError):
                c.authorize(bad, 1, 4)
        bad = copy.deepcopy(t)
        bad["reveals"][0]["seed"] = "00" * 32
        with self.assertRaises(ValueError):
            verify_certificate(bad, c.spec)
        with self.assertRaises(ValueError):
            c.authorize(t, 2, 4)

    def test_oracle_bls_forgery_blocked_by_offchain_verifier(self):
        c = self.chain
        a = c.timestamp + 20
        b = a + 20
        r = c.spec.round_after(b)
        c.open_round(1, a, b, r)
        c.advance_local(c.spec.time_for(r))
        c.transact(c.contract.functions.finalize(1, bytes(32), bytes(96), bytes(96)))
        c.w3.provider.ethereum_tester.mine_blocks(1)
        with self.assertRaises(ValueError):
            c.authorize(c.certificate(1), 1, 4)

    def test_model_hash_one_time_and_reorg_detection(self):
        c = self.chain
        t = c.run_local_ceremony(1)
        self.assertEqual(c.record_model(1, b"model"), keccak(b"model").hex())
        with self.assertRaises(ValueError):
            c.record_model(1, b"other")
        c.w3.provider.ethereum_tester.revert_to_snapshot(self.snapshot)
        with self.assertRaises((ValueError, BlockNotFound)):
            c.authorize(t, 1, 4)


if __name__ == "__main__":
    unittest.main()
