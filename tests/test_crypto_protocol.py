from dataclasses import replace
import hmac
import json
import secrets
import socket
import struct
from time import monotonic
import unittest

import numpy as np

from adaptive_planning import crypto_baseline as base
from adaptive_planning import randomized_check as proj
from adaptive_planning.tcp_transport import (
    Meter,
    ProtocolError,
    encode,
    receive_message,
    recv_exact,
)


class CryptoTests(unittest.TestCase):
    def setUp(self):
        self.session = secrets.token_bytes(16)
        self.updates = np.random.default_rng(42).integers(-100, 101, (4, 17), dtype=np.int64)
        self.workers = [proj.Worker(i, v) for i, v in enumerate(self.updates)]
        self.keys = {w.worker_id: w.public_key for w in self.workers}
        self.candidate = self.updates.sum(axis=0)
        self.packets = [
            base.make_packet(w.private_key, w.worker_id, v, self.session, 1, 2)
            for w, v in zip(self.workers, self.updates)
        ]

    def baseline(self, packets=None, candidate=None, **kwargs):
        return base.verify_and_reconstruct(
            self.packets if packets is None else packets,
            self.keys,
            self.session,
            1,
            2,
            np.array_split(self.candidate if candidate is None else candidate, 12),
            vector_size=17,
            max_abs_update=100,
            **kwargs,
        )

    def verifier(self, candidate=None, round_id=2):
        verifier = proj.Verifier(self.keys, self.session, 1, round_id)
        challenge = verifier.begin(self.candidate if candidate is None else candidate)
        return verifier, [w.reply(challenge) for w in self.workers]

    def test_unequal_blocks_both_protocols(self):
        np.testing.assert_array_equal(self.baseline(), self.candidate)
        verifier, replies = self.verifier()
        np.testing.assert_array_equal(verifier.finish(replies), self.candidate)

    def test_wrong_aggregate(self):
        changed = self.candidate.copy()
        changed[-1] += 1
        with self.assertRaises(ValueError):
            self.baseline(candidate=changed)
        verifier, replies = self.verifier(changed)
        with self.assertRaisesRegex(ValueError, "Подмена агрегированной"):
            verifier.finish(replies)

    def test_missing_and_duplicate_workers(self):
        for packets in (self.packets[:-1], [self.packets[0]] * 4):
            with self.assertRaises(ValueError):
                self.baseline(packets)
        for duplicate in (False, True):
            verifier, replies = self.verifier()
            with self.assertRaises(ValueError):
                verifier.finish([replies[0]] * 4 if duplicate else replies[:-1])

    def test_modified_signatures_and_payloads(self):
        packet = self.packets[0]
        damaged = bytes([packet.payload[0] ^ 1]) + packet.payload[1:]
        for altered in (replace(packet, payload=damaged), replace(packet, signature=b"x" * 64)):
            with self.assertRaises(ValueError):
                self.baseline([altered, *self.packets[1:]])
        verifier, replies = self.verifier()
        altered = replace(replies[0], tags=((replies[0].tags[0] + 1) % proj.P, replies[0].tags[1]))
        with self.assertRaisesRegex(ValueError, "подпись"):
            verifier.finish([altered, *replies[1:]])

    def test_projection_replay_and_single_use(self):
        first, replies = self.verifier()
        first.finish(replies)
        with self.assertRaises(ValueError):
            first.finish(replies)
        second, _ = self.verifier(round_id=3)
        with self.assertRaisesRegex(ValueError, "подпись"):
            second.finish(replies)
        third, _ = self.verifier()  # same round, different challenge also rejects replay
        with self.assertRaisesRegex(ValueError, "подпись"):
            third.finish(replies)
        with self.assertRaises(ValueError):
            first.begin(self.candidate)

    def test_baseline_context_replay(self):
        for field, value in (
            ("round_id", 1),
            ("epoch", 2),
            ("session_id", b"a" * 16),
            ("worker_id", 9),
        ):
            with self.assertRaises(ValueError):
                self.baseline([replace(self.packets[0], **{field: value}), *self.packets[1:]])

    def test_bounds_prevent_overflow_and_modular_alias(self):
        for bad in (np.array([2**64 - 1], dtype=np.uint64), np.array([1.5]), np.zeros((2, 2))):
            with self.assertRaises(ValueError):
                base.integer_vector(bad)
        huge = np.full(17, np.iinfo(np.int64).max, dtype=np.int64)
        packets = [
            base.make_packet(w.private_key, w.worker_id, huge, self.session, 1, 2)
            for w in self.workers
        ]
        with self.assertRaises(ValueError):
            base.verify_and_reconstruct(
                packets, self.keys, self.session, 1, 2, np.array_split(huge, 12), vector_size=17
            )
        with self.assertRaises(ValueError):
            proj.Verifier(self.keys, self.session, 1, 2, max_abs_update=proj.P)
        vector = np.array([1_000_000_000, -1_000_000_000] * 4097, dtype=np.int64)
        weights = proj.random_weights(b"z" * 32, len(vector))
        expected = sum(int(a) * int(b) for a, b in zip(vector, weights)) % proj.P
        self.assertEqual(proj.dot_mod(vector, weights), expected)

    def test_formats_and_candidate_snapshot(self):
        for bad in (self.candidate.astype(float), self.candidate.reshape(1, -1)):
            with self.assertRaises(ValueError):
                self.verifier(bad)
        external = self.candidate.copy()
        verifier, replies = self.verifier(external)
        external[:] = 0
        np.testing.assert_array_equal(verifier.finish(replies), self.candidate)
        bad_packet = replace(self.packets[0], payload=self.packets[0].payload[:-1])
        with self.assertRaises(ValueError):
            self.baseline([bad_packet, *self.packets[1:]])
        with self.assertRaises(ValueError):
            proj.dot_mod(np.array([1, 2]), np.array([1]))


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.key = b"k" * 32
        self.header = dict(
            version=2,
            session="ab" * 16,
            epoch=1,
            round=1,
            sender="worker",
            recipient="coordinator",
            nonce="12" * 16,
            kind="fragment",
            response=True,
            body={},
            deadline_ns=int((monotonic() + 2) * 1e9),
        )

    def decode(self, frame):
        left, right = socket.socketpair()
        with left, right:
            left.sendall(frame)
            left.shutdown(socket.SHUT_WR)
            return receive_message(
                right, {"worker": self.key}, "coordinator", "ab" * 16, monotonic() + 0.2, Meter()
            )

    def test_authenticated_frame(self):
        header, payload = self.decode(encode(self.header, b"abc", self.key))
        self.assertEqual(header, self.header)
        self.assertEqual(payload, b"abc")

    def test_modified_and_unknown_sender(self):
        data = encode(self.header, b"abc", self.key)
        with self.assertRaises(ProtocolError):
            self.decode(data[:-1] + bytes([data[-1] ^ 1]))
        for changes in (
            {"sender": "stranger"},
            {"session": "cd" * 16},
            {"recipient": "another"},
            {"epoch": 2},
            {"round": -1},
            {"response": 1},
        ):
            with self.assertRaises(ProtocolError):
                self.decode(encode(dict(self.header, **changes), b"", self.key))

    def test_truncated_oversized_duplicate_json(self):
        with self.assertRaises(ConnectionError):
            self.decode(encode(self.header, b"abc", self.key)[:-2])
        with self.assertRaises(ProtocolError):
            self.decode(struct.pack("!II", 33_000, 0))
        with self.assertRaises(ProtocolError):
            self.decode(struct.pack("!II", 10, 2**32 - 1))
        raw = json.dumps(self.header)[:-1].encode() + b', "epoch": 1}'
        frame = struct.pack("!II", len(raw), 0) + raw
        with self.assertRaises(ProtocolError):
            self.decode(frame + hmac.digest(self.key, frame, "sha256"))

    def test_idle_socket_timeout(self):
        left, right = socket.socketpair()
        with left, right:
            with self.assertRaises(TimeoutError):
                recv_exact(right, 1, monotonic() + 0.02)

    def test_expired_authenticated_request_deadline(self):
        header = dict(self.header, deadline_ns=int((monotonic() - 1) * 1e9))
        with self.assertRaises(TimeoutError):
            self.decode(encode(header, b"abc", self.key))


if __name__ == "__main__":
    unittest.main()
