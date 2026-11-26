"""Bounded, authenticated RPC for the research harness.

Pairwise HMAC keys are provisioned by the launcher: over spawn IPC on one
host, or over the SSH channel of the distributed launcher (distributed.py).
No pickle is accepted from TCP. Frames are authenticated, not encrypted: run
distributed stands inside a private network (VPC), not on the public Internet.
"""

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import hmac
import json
import secrets
import socket
import struct
import threading
from time import perf_counter
from .clock import now as monotonic
from .tcp_network import CHUNK_BYTES
from .byte_budget import reserve_frame

MAX_HEADER = 32_768
MAX_PAYLOAD = 16_000_000
PREFIX = struct.Struct("!II")
FIELDS = {
    "version",
    "session",
    "epoch",
    "round",
    "sender",
    "recipient",
    "nonce",
    "kind",
    "response",
    "body",
    "deadline_ns",
}


class ProtocolError(ValueError):
    pass


class RemoteError(ProtocolError):
    pass


def require(condition, message):
    if not condition:
        raise ProtocolError(message)


def exact_fields(value, names):
    require(type(value) is dict and set(value) == set(names), "Unexpected message fields")


def integer(value, low, high):
    require(type(value) is int and low <= value <= high, "Invalid integer")
    return value


def hex_bytes(value, length):
    require(type(value) is str and len(value) == length * 2, "Invalid hex length")
    try:
        data = bytes.fromhex(value)
    except ValueError as exc:
        raise ProtocolError("Invalid hex") from exc
    require(len(data) == length, "Invalid hex encoding")
    return data


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


class Meter:
    def __init__(self):
        self.lock = threading.Lock()
        self.rounds = defaultdict(lambda: defaultdict(float))

    def add(self, round_id, **values):
        with self.lock:
            for key, value in values.items():
                self.rounds[round_id][key] += value

    def snapshot(self, round_id):
        with self.lock:
            return dict(self.rounds[round_id])


def remaining(deadline):
    value = deadline - monotonic()
    if value <= 0:
        raise TimeoutError("RPC deadline exceeded")
    return value


def recv_exact(sock, size, deadline):
    data = bytearray()
    while len(data) < size:
        sock.settimeout(remaining(deadline))
        part = sock.recv(min(size - len(data), 65536))
        if not part:
            raise ConnectionError("Truncated frame / peer closed connection")
        data.extend(part)
    return bytes(data)


def encode(header, payload, key):
    raw = json.dumps(header, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    require(0 < len(raw) <= MAX_HEADER and len(payload) <= MAX_PAYLOAD, "Frame too large")
    prefix = PREFIX.pack(len(raw), len(payload))
    frame = prefix + raw + payload
    return frame + hmac.digest(key, frame, "sha256")


def send_message(sock, header, payload, key, deadline, meter, shaper=None):
    if shaper and hasattr(shaper, "authorize"):
        shaper.authorize(header)
    started = perf_counter()
    data = encode(header, payload, key)
    meter.add(header["round"], serialization_s=perf_counter() - started)
    view = memoryview(data)
    socket_s = 0.0
    category = {
        ("compute", False): "model_broadcast",
        ("compute", True): "worker_compute_reply",
        ("fragment", True): "update_transfer",
        ("aggregate", True): "aggregate_transfer",
        ("projection", True): "proof_transfer",
        ("signed_recovery", True): "signed_recovery_transfer",
        ("open_committed", True): "signed_recovery_transfer",
    }.get((header["kind"], header["response"]), "control")
    budget = getattr(shaper, "byte_budget", None) if header["round"] else None
    with reserve_frame(budget, len(data)) as record_sent:
        try:
            if shaper:
                started = perf_counter()
                try:
                    shaper.latency(header, deadline)
                finally:
                    meter.add(header["round"], pacing_wait_s=perf_counter() - started)
            while view:
                size = (
                    min(len(view), CHUNK_BYTES) if shaper and shaper.limited(header) else len(view)
                )
                chunk = view[:size]
                if shaper and shaper.limited(header):
                    started = perf_counter()
                    try:
                        shaper.pace(header, size, deadline)
                        meter.add(header["round"], pacing_reserved_bytes=size)
                    finally:
                        meter.add(header["round"], pacing_wait_s=perf_counter() - started)
                while chunk:
                    sock.settimeout(remaining(deadline))
                    started = perf_counter()
                    try:
                        count = sock.send(chunk)
                    finally:
                        socket_s += perf_counter() - started
                    if not count:
                        raise ConnectionError("Send closed")
                    record_sent(count)
                    meter.add(header["round"], tx_bytes=count)
                    meter.add(header["round"], **{category + "_tx_bytes": count})
                    chunk = chunk[count:]
                view = view[size:]
        finally:
            meter.add(header["round"], socket_send_s=socket_s)
            meter.add(header["round"], **{category + "_send_sum_s": socket_s})


def receive_message(sock, keys, recipient, session, deadline, meter):
    started = perf_counter()
    prefix = recv_exact(sock, PREFIX.size, deadline)
    hlen, plen = PREFIX.unpack(prefix)
    require(0 < hlen <= MAX_HEADER and plen <= MAX_PAYLOAD, "Invalid frame length")
    raw = recv_exact(sock, hlen, deadline)
    decode_start = perf_counter()
    try:
        header = json.loads(
            raw,
            object_pairs_hook=unique_object,
            parse_constant=lambda _: (_ for _ in ()).throw(ProtocolError("Nonfinite JSON")),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("Invalid JSON header") from exc
    exact_fields(header, FIELDS)
    integer(header["version"], 2, 2)
    integer(header["epoch"], 1, 1)
    integer(header["round"], 0, 2**32 - 1)
    integer(header["deadline_ns"], 1, 2**63 - 1)
    # All processes share one clock: monotonic on one host, NTP wall clock across hosts.
    deadline = min(deadline, header["deadline_ns"] / 1e9)
    require(type(header["response"]) is bool and type(header["body"]) is dict, "Invalid envelope")
    require(type(header["kind"]) is str and 0 < len(header["kind"]) < 32, "Invalid kind")
    require(type(header["sender"]) is str and header["sender"] in keys, "Unknown sender")
    require(
        header["recipient"] == recipient and header["session"] == session, "Wrong recipient/session"
    )
    hex_bytes(header["nonce"], 16)
    decode_s = perf_counter() - decode_start
    payload = recv_exact(sock, plen, deadline)
    mac = recv_exact(sock, 32, deadline)
    verify_start = perf_counter()
    expected = hmac.digest(keys[header["sender"]], prefix + raw + payload, "sha256")
    require(hmac.compare_digest(mac, expected), "Invalid message authentication")
    meter.add(
        header["round"],
        rx_bytes=8 + hlen + plen + 32,
        socket_receive_s=perf_counter() - started,
        serialization_s=decode_s + perf_counter() - verify_start,
    )
    return header, payload


@dataclass(frozen=True)
class Peer:
    name: str
    port: int
    host: str = "127.0.0.1"


def rpc(
    peer,
    sender,
    keys,
    session,
    round_id,
    kind,
    body,
    payload,
    deadline,
    meter,
    *,
    nonce=None,
    shaper=None,
):
    header = dict(
        version=2,
        session=session,
        epoch=1,
        round=round_id,
        sender=sender,
        recipient=peer.name,
        nonce=nonce or secrets.token_hex(16),
        kind=kind,
        response=False,
        body=body,
        deadline_ns=int(deadline * 1e9),
    )
    with socket.create_connection((peer.host, peer.port), remaining(deadline)) as sock:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        send_message(sock, header, payload, keys[peer.name], deadline, meter, shaper)
        reply, data = receive_message(sock, keys, sender, session, deadline, meter)
    for field in ("nonce", "kind", "round", "epoch", "deadline_ns"):
        require(reply[field] == header[field], "Mismatched/stale response")
    require(reply["sender"] == peer.name and reply["response"], "Wrong response sender")
    if "error" in reply["body"]:
        exact_fields(reply["body"], {"error"})
        raise RemoteError(str(reply["body"]["error"]))
    return reply["body"], data


def request_digest(header, payload):
    """Useful in tests/logs; no message content or keys are logged."""
    return hashlib.sha256(header["nonce"].encode() + payload).hexdigest()
