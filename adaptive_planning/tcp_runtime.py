"""Separate worker/edge processes and a real verified TCP round.

The coordinator never receives raw worker updates in projection mode.
Each edge pulls its assigned fragments directly from worker TCP servers.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import product
import multiprocessing as mp
import math
import os
import resource
import secrets
import socket
import threading
from time import perf_counter, sleep, thread_time
from .clock import now as monotonic

# One BLAS thread per process; otherwise 12 workers may oversubscribe the Mac.
for _name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from . import crypto_baseline as baseline
from . import randomized_check as projection
from . import collector_routes
from .tcp_network import NetworkShaper, PROFILES
from .tcp_transport import (
    Meter,
    Peer,
    ProtocolError,
    exact_fields,
    hex_bytes,
    integer,
    receive_message,
    remaining,
    require,
    rpc,
    send_message,
)

ACTIONS = ("F+B", "F+P", "R+B", "R+P")
MAX_UPDATE = 200_000
BLOCKS = 12


def timing_fields(body, names):
    for name in names:
        value = body[name]
        require(
            type(value) in (int, float) and math.isfinite(value) and value >= 0,
            f"Invalid timing field: {name}",
        )


def make_model(seed=2026, hidden=670, learning_rate=1.0):
    # Reuse the original MLP's gradient/update/score implementation unchanged.
    from .training import MLP

    model = MLP.__new__(MLP)
    rng = np.random.default_rng(seed)
    model.params = [
        rng.normal(0, np.sqrt(1 / 64), (64, hidden)),
        np.zeros(hidden),
        rng.normal(0, np.sqrt(1 / hidden), (hidden, 10)),
        np.zeros(10),
    ]
    model.nparam = sum(p.size for p in model.params)
    require(
        type(learning_rate) in (int, float) and math.isfinite(learning_rate) and learning_rate > 0,
        "Invalid learning rate",
    )
    model.learning_rate = learning_rate
    return model


def flatten_model(model):
    return np.concatenate([p.ravel() for p in model.params]).astype("<f8").tobytes()


def load_model(model, payload):
    require(len(payload) == model.nparam * 8, "Wrong model length")
    values = np.frombuffer(payload, dtype="<f8")
    require(np.all(np.isfinite(values)), "Nonfinite parameters")
    offset = 0
    for p in model.params:
        p[:] = values[offset : offset + p.size].reshape(p.shape)
        offset += p.size


def block_indices(dimension):
    return np.array_split(np.arange(dimension), min(BLOCKS, dimension))


def lane_indices(dimension, lane):
    blocks = block_indices(dimension)
    selected = blocks[lane::2]
    return np.concatenate(selected) if selected else np.empty(0, dtype=np.int64)


@dataclass(frozen=True)
class Fault:
    failed_edges: tuple = ()
    corrupt_edges: tuple = ()
    stalled_edges: tuple = ()
    missing_projection: tuple = ()
    missing_compute: tuple = ()
    edge_delay_s: tuple = (0.0, 0.0, 0.0, 0.0)
    missing_full_update: tuple = ()
    # Test harness controls endpoint admission responses independently of eventual data service.
    unavailable_at_probe: tuple | None = None


@dataclass
class RoundResult:
    accepted: bool = False
    vector: object = None
    metrics: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)


class WorkerService:
    def __init__(self, identity, group, total_examples, hidden):
        self.worker_id = int(identity[1:])
        self.group = group
        self.total_examples = total_examples
        self.model = make_model(hidden=hidden)
        self.worker = projection.Worker(
            self.worker_id, np.zeros(self.model.nparam, dtype=np.int64), max_abs_update=MAX_UPDATE
        )
        self.lock = threading.Lock()
        self.round_id = 0
        self.missing_projection = False
        self.projection_failure_delay_s = 0.0
        self.challenges = set()

    def handle(self, header, payload, server):
        kind, body, round_id = header["kind"], header["body"], header["round"]
        if kind == "compute":
            require(header["sender"] == "coordinator", "Compute requires coordinator")
            fields = {"method", "missing_projection", "missing_compute"}
            if "selected_examples" in body:
                fields.add("selected_examples")
            if "projection_failure_delay_s" in body:
                fields.add("projection_failure_delay_s")
            for flag in ("cross_spares", "commit_updates", "late_map"):
                if flag in body:
                    require(type(body[flag]) is bool, "Invalid round option")
                    fields.add(flag)
            if "allow_signed_recovery" in body:
                fields.add("allow_signed_recovery")
            if "missing_full_update" in body:
                fields.add("missing_full_update")
            exact_fields(body, fields)
            delay = body.get("projection_failure_delay_s", 0.0)
            require(
                type(delay) in (int, float) and math.isfinite(delay) and 0 <= delay <= 5,
                "Invalid projection failure delay",
            )
            require(body["method"] in ("B", "P"), "Invalid method")
            require(
                type(body.get("missing_full_update", False)) is bool,
                "Invalid signed-update fault control",
            )
            require(
                type(body["missing_projection"]) is bool and type(body["missing_compute"]) is bool,
                "Invalid fault control",
            )
            with self.lock:
                require(round_id > self.round_id, "Repeated or stale compute")
                self.round_id = round_id
                self.challenges.clear()
                self.relay_reply = None
                self.missing_projection = body["missing_projection"]
                self.allow_signed_recovery = body.get("allow_signed_recovery") is True
                self.cross_spares = body.get("cross_spares", False)
                self.late_map = body.get("late_map", False)
                self.map_granted = not self.late_map
                self.commit_updates = body.get("commit_updates", False)
                self.missing_full_update = body.get("missing_full_update", False)
                self.projection_failure_delay_s = delay
                load_model(self.model, payload)
                self.ready = False
                if body["missing_compute"]:
                    raise ProtocolError("Injected unavailable worker before compute")
                from .training import Q_SCALE

                start, cpu = perf_counter(), thread_time()
                X, y = self.group
                denominator = integer(
                    body.get("selected_examples", self.total_examples), len(y), self.total_examples
                )
                weighted = self.model.gradient(X, y) * (len(y) / denominator) * Q_SCALE
                require(np.all(np.isfinite(weighted)), "Nonfinite gradient")
                rounded = np.rint(weighted)
                require(np.all(np.abs(rounded) <= MAX_UPDATE), "UPDATE_OUT_OF_RANGE")
                update = baseline.integer_vector(rounded.astype(np.int64), MAX_UPDATE)
                self.worker.update = update.copy()
                gradient_s, gradient_cpu_s = perf_counter() - start, thread_time() - cpu
                self.ready = True
                if body["method"] == "B" and body.get("missing_full_update", False):
                    raise ProtocolError("Injected missing signed full update")
                start, cpu = perf_counter(), thread_time()
                if body["method"] == "B":
                    packet = baseline.make_packet(
                        self.worker.private_key,
                        self.worker_id,
                        update,
                        bytes.fromhex(server.session),
                        1,
                        round_id,
                    )
                    data = packet.payload + packet.signature
                elif self.commit_updates:
                    raw = self.worker.update.astype("<i8", copy=False).tobytes()
                    digest = baseline.hashlib.sha256(raw).digest()
                    message = baseline.digest_message(self.worker_id, bytes.fromhex(server.session),
                                                      1, self.round_id, update.size, digest)
                    data = digest + self.worker.private_key.sign(message)
                else:
                    data = b""
                return dict(
                    gradient_s=gradient_s,
                    gradient_cpu_s=gradient_cpu_s,
                    prepare_s=perf_counter() - start,
                    prepare_cpu_s=thread_time() - cpu,
                ), data

        require(not payload, "Unexpected worker payload")
        with self.lock:
            require(
                round_id == self.round_id and getattr(self, "ready", False),
                "No update for this round",
            )
            if kind == "grant_spares":
                require(header["sender"] == "coordinator" and self.late_map,
                        "The coordinator must authorize a late placement")
                exact_fields(body, {"cross"})
                require(type(body["cross"]) is bool, "Invalid spare map")
                require(not self.map_granted or self.cross_spares == body["cross"],
                        "A conflicting spare map was already granted")
                self.cross_spares = body["cross"]
                self.map_granted = True
                return {}, b""
            if kind == "open_committed":
                exact_fields(body, set())
                require(header["sender"] == "coordinator" and self.allow_signed_recovery and
                        self.commit_updates, "Opening not authorized for this round")
                require(not self.missing_full_update, "Signed update unavailable")
                return {}, self.worker.update.astype("<i8", copy=False).tobytes()
            if kind == "signed_recovery":
                exact_fields(body, set())
                require(header["sender"] == "coordinator" and self.allow_signed_recovery,
                        "This round does not authorize signed full recovery")
                require(not self.missing_full_update, "Signed update unavailable")
                packet = baseline.make_packet(self.worker.private_key, self.worker_id,
                                              self.worker.update, bytes.fromhex(server.session),
                                              1, self.round_id)
                return {}, packet.payload + packet.signature
            if kind == "fragment":
                exact_fields(body, {"lane"})
                lane = integer(body["lane"], 0, 1)
                spare = 2 + (1 - lane if self.cross_spares else lane)
                require(header["sender"] == f"e{lane}" or
                        (self.map_granted and header["sender"] == f"e{spare}"), "Wrong edge lane")
                indices = lane_indices(self.model.nparam, lane)
                return {"size": len(indices)}, self.worker.update[indices].astype("<i8").tobytes()
            if kind in ("projection","relayed_projection"):
                relay = kind=="relayed_projection"
                if relay:
                    exact_fields(body, {"command","token"})
                    sender=header["sender"]
                    require(sender in ("e0","e1","e2","e3"), "Unregistered proof collector")
                    expected=collector_routes.token(server.keys["coordinator"],server.session,round_id,
                                                    self.worker_id,int(sender[1:]),body["command"])
                    require(type(body["token"]) is str and secrets.compare_digest(body["token"],expected),
                            "Invalid collector grant")
                    body=body["command"]
                else:
                    require(header["sender"] == "coordinator", "Challenge requires coordinator")
                exact_fields(body, {"hash", "seeds", "dimension"})
                dimension = integer(body["dimension"], self.model.nparam, self.model.nparam)
                require(
                    type(body["seeds"]) is list and len(body["seeds"]) == projection.CHECKS,
                    "Invalid challenge count",
                )
                challenge = projection.Challenge(
                    bytes.fromhex(server.session),
                    1,
                    round_id,
                    hex_bytes(body["hash"], 32),
                    dimension,
                    tuple(hex_bytes(s, 32) for s in body["seeds"]),
                )
                identity = (challenge.candidate_hash, challenge.seeds)
                if relay and self.relay_reply is not None:
                    require(self.relay_reply[0]==identity,"Conflicting relayed challenge")
                    return self.relay_reply[1]
                require(identity not in self.challenges, "Repeated projection challenge")
                self.challenges.add(identity)
                if self.missing_projection:
                    sleep(
                        min(self.projection_failure_delay_s, remaining(header["deadline_ns"] / 1e9))
                    )
                    raise ProtocolError("Injected unavailable worker during projection")
                start, cpu = perf_counter(), thread_time()
                reply = self.worker.reply(challenge)
                response = dict(
                    tags=list(reply.tags),
                    prepare_s=perf_counter() - start,
                    prepare_cpu_s=thread_time() - cpu,
                ), reply.signature
                if relay:
                    self.relay_reply=(identity,response)
                return response
        raise ProtocolError("Unknown worker command")


class EdgeService:
    def __init__(self, identity, workers, dimension, rpc_timeout):
        self.node = int(identity[1:])
        self.workers, self.dimension, self.rpc_timeout = workers, dimension, rpc_timeout
        self.lock = threading.Lock()
        self.round_id = 0

    def handle(self, header, payload, server):
        if header["kind"]=="relay_projection":
            return collector_routes.forward(self,header,payload,server)
        if header["kind"] == "ready_spare":
            require(header["sender"] == "coordinator" and self.node in (2,3) and not payload,
                    "A readiness request requires an authorized spare")
            exact_fields(header["body"], {"unavailable"})
            require(header["body"]["unavailable"] is False, "Spare refused the readiness request")
            return {"ready": True}, b""
        require(
            header["sender"] == "coordinator" and header["kind"] == "aggregate",
            "Invalid edge command",
        )
        require(not payload, "Unexpected edge payload")
        body = header["body"]
        fields = {"fail", "corrupt", "stall", "delay_s", "budget_s"}
        lane = self.node % 2
        if "assigned_lane" in body:
            fields.add("assigned_lane")
            lane = integer(body["assigned_lane"], 0, 1)
        if "worker_ids" in body:
            fields.add("worker_ids")
        exact_fields(body, fields)
        identities = body.get("worker_ids", list(range(len(self.workers))))
        require(
            type(identities) is list
            and identities
            and all(type(i) is int and 0 <= i < len(self.workers) for i in identities)
            and len(set(identities)) == len(identities),
            "Invalid selected worker roster",
        )
        workers = [self.workers[i] for i in identities]
        require(
            all(type(body[k]) is bool for k in ("fail", "corrupt", "stall")),
            "Invalid fault controls",
        )
        require(
            type(body["delay_s"]) in (int, float) and 0 <= body["delay_s"] <= 5, "Invalid delay"
        )
        require(
            type(body["budget_s"]) in (int, float) and 0 < body["budget_s"] <= 60, "Invalid budget"
        )
        round_id = header["round"]
        with self.lock:
            require(round_id > self.round_id, "Repeated or stale aggregation")
            self.round_id = round_id
        if body["fail"]:
            raise ProtocolError("Injected unavailable edge")
        if body["stall"]:
            sleep(min(body["budget_s"], self.rpc_timeout) + 0.05)
            raise TimeoutError("Injected stalled edge")
        sleep(body["delay_s"])
        deadline = min(
            header["deadline_ns"] / 1e9, monotonic() + min(body["budget_s"], self.rpc_timeout)
        )
        indices = lane_indices(self.dimension, lane)
        fetch_start = perf_counter()

        def fetch(peer):
            info, data = rpc(
                peer,
                server.identity,
                server.keys,
                server.session,
                round_id,
                "fragment",
                {"lane": lane},
                b"",
                deadline,
                server.meter,
                shaper=server.shaper,
            )
            exact_fields(info, {"size"})
            require(
                info["size"] == len(indices) and len(data) == len(indices) * 8,
                "Wrong fragment length",
            )
            return baseline.integer_vector(np.frombuffer(data, dtype="<i8"), MAX_UPDATE)

        with ThreadPoolExecutor(max_workers=len(workers)) as pool:
            fragments = list(pool.map(fetch, workers))
        fetch_s = perf_counter() - fetch_start
        start, cpu = perf_counter(), thread_time()
        # Worker bounds and <= 16 workers make this int64 sum safe.
        candidate = np.stack(fragments).sum(axis=0, dtype=np.int64)
        if body["corrupt"]:
            candidate[0] += 1
        return dict(
            size=len(indices),
            fetch_s=fetch_s,
            aggregate_s=perf_counter() - start,
            aggregate_cpu_s=thread_time() - cpu,
        ), candidate.astype("<i8").tobytes()


class Server:
    def __init__(self, identity, session, keys, service, timeout, shaper=None):
        self.identity, self.session, self.keys = identity, session, keys
        self.service, self.timeout = service, timeout
        self.meter = Meter()
        self.shaper = shaper
        self.stopped = threading.Event()
        self.seen = set()
        self.lock = threading.Lock()

    def connection(self, sock):
        header = None
        with sock:
            deadline = monotonic() + self.timeout
            try:
                header, payload = receive_message(
                    sock, self.keys, self.identity, self.session, deadline, self.meter
                )
                deadline = min(deadline, header["deadline_ns"] / 1e9)
                require(not header["response"], "Request expected")
                nonce = (header["sender"], header["nonce"])
                with self.lock:
                    require(nonce not in self.seen, "Replay detected")
                    self.seen.add(nonce)
                if header["kind"] in ("stop", "stats"):
                    require(
                        header["sender"] == "coordinator" and not payload and header["round"] == 0,
                        "Invalid control command",
                    )
                    if header["kind"] == "stop":
                        exact_fields(header["body"], set())
                        self.stopped.set()
                        info, data = {}, b""
                    else:
                        exact_fields(header["body"], {"round"})
                        target = integer(header["body"]["round"], 1, 2**32 - 1)
                        info, data = self.meter.snapshot(target), b""
                        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                        info["process_peak_rss_bytes"] = (
                            peak if os.uname().sysname == "Darwin" else peak * 1024
                        )
                else:
                    require(header["round"] > 0, "Round must be positive")
                    info, data = self.service.handle(header, payload, self)
            except (ValueError, OSError, TypeError, KeyError, OverflowError) as exc:
                self.meter.add(header["round"] if header else 0, protocol_errors=1)
                if header is None:
                    return
                info, data = {"error": f"{type(exc).__name__}: {exc}"[:250]}, b""
            reply = dict(
                header, sender=self.identity, recipient=header["sender"], response=True, body=info
            )
            try:
                send_message(
                    sock,
                    reply,
                    data,
                    self.keys[header["sender"]],
                    deadline,
                    self.meter,
                    self.shaper,
                )
            except (ValueError, OSError):
                self.meter.add(header["round"], protocol_errors=1)


def serve_process(identity, session, keys, role_args, ready_pipe, rpc_timeout, shaper):
    listener = None
    try:
        if identity.startswith("w"):
            service = WorkerService(identity, *role_args)
            public_key = service.worker.public_key.public_bytes_raw().hex()
        else:
            service = EdgeService(identity, *role_args, rpc_timeout)
            public_key = None
        server = Server(identity, session, keys, service, max(2.0, rpc_timeout + 0.5), shaper)
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(64)
        listener.settimeout(0.1)
        ready_pipe.send({"port": listener.getsockname()[1], "key": public_key, "pid": os.getpid()})
        ready_pipe.close()
        with ThreadPoolExecutor(max_workers=8) as pool:
            while not server.stopped.is_set():
                try:
                    sock, _ = listener.accept()
                except socket.timeout:
                    continue
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                pool.submit(server.connection, sock)
    except Exception as exc:
        try:
            ready_pipe.send({"error": repr(exc)})
        except (OSError, EOFError):
            pass
        raise
    finally:
        if listener:
            listener.close()
        ready_pipe.close()


class Cluster:
    def __init__(
        self,
        groups,
        *,
        hidden=670,
        rpc_timeout=2.0,
        startup_timeout=60.0,
        network=None,
        dynamic_network=False,
        selection_authority=None,
        app_byte_budget=None,
    ):
        require(2 <= len(groups) <= 16, "Supported worker count: 2..16")
        require(0 < rpc_timeout <= 30, "RPC timeout must be in (0, 30]")
        self.groups, self.hidden, self.rpc_timeout = groups, hidden, rpc_timeout
        self.selection_authority = selection_authority
        self.startup_timeout = startup_timeout
        self.dimension = make_model(hidden=hidden).nparam
        self.session = secrets.token_hex(16)
        self.meter = Meter()
        self.processes, self.peers, self.public_keys = {}, {}, {}
        self.context = mp.get_context("spawn")
        self.network = network or PROFILES["unlimited"]
        self.dynamic_network = dynamic_network
        self.shaper = (
            NetworkShaper(
                self.context, self.network, dynamic=dynamic_network, byte_limit=app_byte_budget
            )
            if dynamic_network
            or self.network != PROFILES["unlimited"]
            or app_byte_budget is not None
            else None
        )
        self.keys = {"coordinator": {}}
        self.round_id = 0
        names = [f"w{i}" for i in range(len(groups))] + [f"e{i}" for i in range(4)]
        for name in names:
            self.keys[name] = {}
            self.keys["coordinator"][name] = self.keys[name]["coordinator"] = secrets.token_bytes(
                32
            )
        for w in [n for n in names if n.startswith("w")]:
            for e in [n for n in names if n.startswith("e")]:
                self.keys[w][e] = self.keys[e][w] = secrets.token_bytes(32)

    def __enter__(self):
        start = perf_counter()
        try:
            total = sum(len(y) for _, y in self.groups)
            self._start(
                [(f"w{i}", (group, total, self.hidden)) for i, group in enumerate(self.groups)]
            )
            workers = [self.peers[f"w{i}"] for i in range(len(self.groups))]
            self._start([(f"e{i}", (workers, self.dimension)) for i in range(4)])
            self.startup_s = perf_counter() - start
            return self
        except BaseException:
            self.close()
            raise

    def _start(self, specs):
        pipes = []
        try:
            for name, args in specs:
                receive, send = self.context.Pipe(duplex=False)
                process = self.context.Process(
                    target=serve_process,
                    args=(
                        name,
                        self.session,
                        self.keys[name],
                        args,
                        send,
                        self.rpc_timeout,
                        self.shaper,
                    ),
                    name=f"adaptive-{name}",
                )
                self.processes[name] = process
                process.start()
                send.close()
                pipes.append((name, receive))
            deadline = monotonic() + self.startup_timeout
            for name, pipe in pipes:
                if not pipe.poll(remaining(deadline)):
                    raise TimeoutError(f"Startup timeout: {name}")
                info = pipe.recv()
                require("error" not in info, f"Startup failure: {info}")
                self.peers[name] = Peer(name, info["port"])
                if info["key"]:
                    self.public_keys[int(name[1:])] = Ed25519PublicKey.from_public_bytes(
                        bytes.fromhex(info["key"])
                    )
        finally:
            for _, pipe in pipes:
                pipe.close()

    def call(self, name, round_id, kind, body, payload=b"", deadline=None, **kwargs):
        return rpc(
            self.peers[name],
            "coordinator",
            self.keys["coordinator"],
            self.session,
            round_id,
            kind,
            body,
            payload,
            deadline or monotonic() + self.rpc_timeout,
            self.meter,
            shaper=self.shaper,
            **kwargs,
        )

    def parallel(self, calls, deadline):
        results, errors = {}, []

        def invoke(item):
            name, kind, body, payload = item
            started = perf_counter()
            result = self.call(
                name,
                self.round_id,
                kind,
                body,
                payload,
                min(deadline, monotonic() + self.rpc_timeout),
            )
            return result, perf_counter() - started

        with ThreadPoolExecutor(max_workers=len(calls)) as pool:
            futures = {pool.submit(invoke, item): item[0] for item in calls}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except (ValueError, OSError) as exc:
                    errors.append(f"{name}: {type(exc).__name__}: {exc}")
        return results, errors

    def run_round(
        self,
        model,
        action,
        *,
        deadline_s=5.0,
        fault=None,
        switched=False,
        switch_s=0.0,
        network_profile=None,
        selection=None,
        recovery_gate=None,
        cross_spares=False,
        commit_updates=False,
        probe_spares=False,
        proof_collectors=None,
        proof_budget_ms=None,
        emulated_link_ms=None,
    ):
        require(action in ACTIONS, "Unknown action")
        if proof_collectors is not None:
            require(action.endswith("+P") and self.selection_authority is not None,"Collector needs certified P")
            require(isinstance(proof_collectors,(list,tuple)) and 1<=len(proof_collectors)<=2 and
                    all(type(g) is int and 0<=g<4 for g in proof_collectors) and
                    len(set(proof_collectors))==len(proof_collectors),"Invalid collector grant")
            require(type(proof_budget_ms) in (float,int) and 0<proof_budget_ms<=60000,"Invalid proof window")
        require(type(cross_spares) is bool and type(commit_updates) is bool, "Invalid round mode")
        require(not commit_updates or recovery_gate is not None, "Selective repair requires a gate")
        require(type(probe_spares) is bool and (not probe_spares or recovery_gate is not None),
                "Late placement requires an enabled gate")
        require(recovery_gate is None or action == "F+P", "Deferred recovery starts at F+P")
        require(0 < deadline_s <= 60 and 0 <= switch_s <= 5, "Invalid deadline/switch cost")
        profile = network_profile or self.network
        require(
            self.dynamic_network or profile == self.network,
            "Changing network requires dynamic mode",
        )
        if self.selection_authority is not None:
            selected = self.selection_authority.authorize(
                selection, self.round_id + 1, len(self.groups)
            )
            require(
                selected
                and len(set(selected)) == len(selected)
                and all(type(i) is int and 0 <= i < len(self.groups) for i in selected),
                "Invalid authorized selection",
            )
        else:
            require(selection is None, "A schedule requires a configured authority")
            selected = tuple(range(len(self.groups)))
        selected_examples = sum(len(self.groups[i][1]) for i in selected)
        selected_keys = {i: self.public_keys[i] for i in selected}
        fault = fault or Fault()
        self.round_id += 1
        result = RoundResult()
        metrics = result.metrics
        metrics.update(
            protocol_round=self.round_id,
            action=action,
            switched=int(switched),
            feedback_schema=3,
            recovery_attempts=[],
            opened_workers=[],
            spare_assignments=[],
            ready_spares=None,
            grant_complete=None,
            accepted_check=None,
            compute_started=False,
            compute_completed=False,
            aggregate_started=False,
            aggregate_completed=False,
            projection_started=False,
            verification_completed=False,
            verification_attempts=0,
            candidate_rejections=0,
            verification_wall_s=0.0,
            verification_cpu_s=0.0,
            proof_prepare_max_s=0.0,
            proof_prepare_cpu_sum_s=0.0,
            projection_rpc_wall_s=0.0,
            edge_compute_max_s=0.0,
            edge_cpu_sum_s=0.0,
            update_fetch_max_s=0.0,
            gradient_max_s=0.0,
            gradient_cpu_sum_s=0.0,
            worker_rpc_wall_s=0.0,
            edge_rpc_wall_s=0.0,
            update_wall_s=0.0,
            switch_wall_s=0.0,
        )
        if self.selection_authority is not None:
            metrics["selected_workers"] = list(selected)
            metrics["selected_examples"] = selected_examples
        start = perf_counter()
        deadline = monotonic() + deadline_s
        try:
            if self.dynamic_network:
                self.shaper.configure_round(self.round_id, profile, deadline=deadline)
            if switched:
                t = perf_counter()
                sleep(min(switch_s, remaining(deadline)))
                metrics["switch_wall_s"] = perf_counter() - t
            remaining(deadline)
            params = flatten_model(model)
            calls = [
                (
                    f"w{i}",
                    "compute",
                    {
                        "method": action[-1],
                        **({"cross_spares": True} if cross_spares else {}),
                        **({"late_map": True} if probe_spares else {}),
                        **({"commit_updates": True} if commit_updates else {}),
                        **({"allow_signed_recovery": True} if recovery_gate is not None else {}),
                        **(
                            {"selected_examples": selected_examples}
                            if self.selection_authority is not None
                            else {}
                        ),
                        "missing_projection": i in fault.missing_projection,
                        "missing_compute": i in fault.missing_compute,
                        "missing_full_update": i in fault.missing_full_update,
                        "projection_failure_delay_s": getattr(
                            fault, "projection_failure_delay_s", 0.0
                        ),
                    },
                    params,
                )
                for i in selected
            ]
            t = perf_counter()
            metrics["compute_started"] = True
            responses, errors = self.parallel(calls, deadline)
            metrics["worker_rpc_wall_s"] = perf_counter() - t
            result.errors.extend(errors)
            packets = []
            commitments = {}
            for name, ((body, data), _) in responses.items():
                exact_fields(body, {"gradient_s", "gradient_cpu_s", "prepare_s", "prepare_cpu_s"})
                timing_fields(body, body)
                metrics["gradient_max_s"] = max(metrics["gradient_max_s"], body["gradient_s"])
                metrics["gradient_cpu_sum_s"] += body["gradient_cpu_s"]
                metrics["proof_prepare_max_s"] = max(
                    metrics["proof_prepare_max_s"], body["prepare_s"]
                )
                metrics["proof_prepare_cpu_sum_s"] += body["prepare_cpu_s"]
                if action[-1] == "B":
                    require(len(data) == self.dimension * 8 + 64, "Wrong signed update length")
                    packets.append(
                        baseline.SignedUpdate(
                            int(name[1:]),
                            bytes.fromhex(self.session),
                            1,
                            self.round_id,
                            self.dimension,
                            data[:-64],
                            data[-64:],
                        )
                    )
                elif commit_updates:
                    require(len(data) == 96, "Wrong signed commitment length")
                    worker_id = int(name[1:])
                    digest, signature = data[:32], data[32:]
                    msg = baseline.digest_message(worker_id, bytes.fromhex(self.session), 1,
                                                  self.round_id, self.dimension, digest)
                    try:
                        selected_keys[worker_id].verify(signature, msg)
                    except baseline.InvalidSignature as exc:
                        raise ValueError("Invalid update commitment signature") from exc
                    commitments[worker_id] = (digest, signature)
                else:
                    require(not data, "Projection mode must not return raw updates")
            require(len(responses) == len(selected), "Not all workers computed updates")
            metrics["compute_completed"] = True
            metrics["aggregate_started"] = True
            lanes = {0: [], 1: []}
            def assigned_lane(node):
                return (1-node%2) if cross_spares and node >= 2 else node%2

            def fetch_lanes(nodes):
                responses, errors = self.parallel(
                    [
                        (
                            f"e{i}",
                            "aggregate",
                            {
                                "fail": i in fault.failed_edges,
                                **({"assigned_lane": assigned_lane(i)} if cross_spares and i >= 2 else {}),
                                "corrupt": i in fault.corrupt_edges,
                                "stall": i in fault.stalled_edges,
                                "delay_s": fault.edge_delay_s[i],
                                "budget_s": remaining(deadline),
                                **(
                                    {"worker_ids": list(selected)}
                                    if self.selection_authority is not None
                                    else {}
                                ),
                            },
                            b"",
                        )
                        for i in nodes
                    ],
                    deadline,
                )
                result.errors.extend(errors)
                for name, ((body, data), elapsed) in responses.items():
                    try:
                        exact_fields(body, {"size", "fetch_s", "aggregate_s", "aggregate_cpu_s"})
                        timing_fields(body, ("fetch_s", "aggregate_s", "aggregate_cpu_s"))
                        node = int(name[1:])
                        indices = lane_indices(self.dimension, assigned_lane(node))
                        integer(body["size"], len(indices), len(indices))
                        require(len(data) == len(indices) * 8, "Wrong aggregate fragment length")
                        candidate = baseline.integer_vector(
                            np.frombuffer(data, dtype="<i8"), len(selected) * MAX_UPDATE
                        )
                        lanes[assigned_lane(node)].append((elapsed, node, candidate))
                        metrics["edge_compute_max_s"] = max(
                            metrics["edge_compute_max_s"], body["aggregate_s"]
                        )
                        metrics["update_fetch_max_s"] = max(
                            metrics["update_fetch_max_s"], body["fetch_s"]
                        )
                        metrics["edge_cpu_sum_s"] += body["aggregate_cpu_s"]
                    except ValueError as exc:
                        result.errors.append(f"{name}: malformed replica: {exc}")
            t = perf_counter()
            fetch_lanes(range(4 if action[0] == "R" else 2))
            missing = [i for i in (0, 1) if not lanes[i]]
            if missing and recovery_gate is not None and remaining(deadline) > 0:
                if recovery_gate("edge", tuple(missing), remaining(deadline)):
                    metrics["recovery_attempts"].append("edge")
                    if probe_spares:
                        unavailable = (fault.failed_edges if fault.unavailable_at_probe is None
                                       else fault.unavailable_at_probe)
                        # Fault flags go only to the isolated service. The scheduling decision
                        # below consumes the actual authenticated replies, never those flags.
                        offers, offer_errors = self.parallel(
                            [(f"e{j}", "ready_spare", {"unavailable": j in unavailable}, b"")
                             for j in (2, 3)], min(deadline, monotonic() + .06))
                        result.errors.extend(offer_errors)
                        ready = {int(name[1:]) for name, ((body, data), _) in offers.items()
                                 if body == {"ready": True} and not data}
                        metrics["ready_spares"] = sorted(ready)
                        eligible = [m for m in (cross_spares, not cross_spares)
                                    if all(2+(1-i if m else i) in ready for i in missing)]
                        require(bool(eligible), "No responding spare for this fragment set")
                        cross_spares = eligible[0]
                        grants, grant_errors = self.parallel(
                            [(f"w{i}", "grant_spares", {"cross": cross_spares}, b"")
                             for i in selected], deadline)
                        result.errors.extend(grant_errors)
                        metrics["grant_complete"] = (
                            set(grants) == {f"w{i}" for i in selected}
                            and all(body == {} and not data for (body, data), _ in grants.values()))
                        require(metrics["grant_complete"], "Incomplete spare authorization")
                    nodes = [2 + (1 - i if cross_spares else i) for i in missing]
                    metrics["spare_assignments"] = list(zip(nodes, missing))
                    fetch_lanes(nodes)
            metrics["edge_rpc_wall_s"] = perf_counter() - t
            require(all(lanes.values()), "Missing aggregate lane")
            metrics["aggregate_completed"] = True
            # All copies have arrived or timed out. Ordering uses only observed response time.
            # Four combinations suffice for two lanes; no trusted sum chooses the replica.
            choices = sorted(product(*lanes.values()), key=lambda pair: max(x[0] for x in pair))
            tried = set()
            for pair in choices:
                remaining(deadline)
                candidate = np.empty(self.dimension, dtype=np.int64)
                for lane, (_, _, cell) in enumerate(pair):
                    candidate[lane_indices(self.dimension, lane)] = cell
                digest = projection.sha256(candidate.astype("<i8").tobytes()).digest()
                if digest in tried:
                    continue
                tried.add(digest)
                metrics["verification_attempts"] += 1
                try:
                    t, cpu = perf_counter(), thread_time()
                    if action[-1] == "B":
                        try:
                            verified = baseline.verify_and_reconstruct(
                                packets,
                                selected_keys,
                                bytes.fromhex(self.session),
                                1,
                                self.round_id,
                                np.array_split(candidate, min(BLOCKS, self.dimension)),
                                vector_size=self.dimension,
                                blocks=min(BLOCKS, self.dimension),
                                max_abs_update=MAX_UPDATE,
                            )
                        finally:
                            metrics["verification_wall_s"] += perf_counter() - t
                            metrics["verification_cpu_s"] += thread_time() - cpu
                    else:
                        verifier = projection.Verifier(
                            selected_keys,
                            bytes.fromhex(self.session),
                            1,
                            self.round_id,
                            max_abs_update=MAX_UPDATE,
                            commitments=commitments if commit_updates else None,
                        )
                        challenge = verifier.begin(candidate)
                        metrics["verification_wall_s"] += perf_counter() - t
                        metrics["verification_cpu_s"] += thread_time() - cpu
                        command = {
                            "hash": challenge.candidate_hash.hex(),
                            "dimension": self.dimension,
                            "seeds": [s.hex() for s in challenge.seeds],
                        }
                        t = perf_counter()
                        metrics["projection_started"] = True
                        if proof_collectors is None:
                            replies, errors = self.parallel(
                                [(f"w{i}", "projection", command, b"") for i in selected], deadline)
                        else:
                            phase_end=min(deadline,monotonic()+proof_budget_ms/1000)
                            replies,errors=collector_routes.collect(self,selected,proof_collectors,command,
                                challenge,phase_end,emulated_link_ms,metrics)
                        metrics["projection_rpc_wall_s"] += perf_counter() - t
                        result.errors.extend(errors)
                        signed = []
                        for name, ((body, data), _) in replies.items():
                            exact_fields(body, {"tags", "prepare_s", "prepare_cpu_s"})
                            timing_fields(body, ("prepare_s", "prepare_cpu_s"))
                            require(
                                len(data) == 64 and type(body["tags"]) is list,
                                "Wrong projection reply",
                            )
                            metrics["proof_prepare_max_s"] = max(
                                metrics["proof_prepare_max_s"], body["prepare_s"]
                            )
                            metrics["proof_prepare_cpu_sum_s"] += body["prepare_cpu_s"]
                            signed.append(
                                projection.Reply(int(name[1:]), tuple(body["tags"]), data)
                            )
                        missing = sorted(set(selected_keys) - {r.worker_id for r in signed})
                        if missing and recovery_gate is not None and remaining(deadline) > 0:
                            if recovery_gate("proof", tuple(missing), remaining(deadline)):
                                metrics["recovery_attempts"].append("proof")
                                recover_ids = missing if commit_updates else selected
                                rpc_name = "open_committed" if commit_updates else "signed_recovery"
                                full, errors = self.parallel(
                                    [(f"w{i}", rpc_name, {}, b"") for i in recover_ids], deadline)
                                metrics["opened_workers"] = sorted(int(n[1:]) for n in full)
                                result.errors.extend(errors)
                                packets = []
                                openings = {}
                                for name, ((body, data), _) in full.items():
                                    exact_fields(body, set())
                                    if commit_updates:
                                        worker_id = int(name[1:])
                                        require(worker_id in missing, "Unexpected opening")
                                        openings[worker_id] = data
                                        continue
                                    require(len(data) == self.dimension * 8 + 64,
                                            "Wrong signed recovery length")
                                    packets.append(baseline.SignedUpdate(
                                        int(name[1:]), bytes.fromhex(self.session), 1, self.round_id,
                                        self.dimension, data[:-64], data[-64:]))
                                t, cpu = perf_counter(), thread_time()
                                try:
                                    if commit_updates:
                                        require(set(openings) == set(missing), "Incomplete opening set")
                                        verified = verifier.finish(signed, openings=openings)
                                        metrics["accepted_check"] = "mixed_projection"
                                    else:
                                        verified = baseline.verify_and_reconstruct(
                                            packets, selected_keys, bytes.fromhex(self.session), 1,
                                            self.round_id, np.array_split(candidate, min(BLOCKS, self.dimension)),
                                            vector_size=self.dimension, blocks=min(BLOCKS, self.dimension),
                                            max_abs_update=MAX_UPDATE)
                                        metrics["accepted_check"] = "exact_full"
                                finally:
                                    metrics["verification_wall_s"] += perf_counter() - t
                                    metrics["verification_cpu_s"] += thread_time() - cpu
                            else:
                                require(False, "Missing projection participant")
                        else:
                            require(len(signed) == len(selected), "Missing projection participant")
                            t, cpu = perf_counter(), thread_time()
                            try:
                                verified = verifier.finish(signed)
                                metrics["accepted_check"] = "projection"
                            finally:
                                metrics["verification_wall_s"] += perf_counter() - t
                                metrics["verification_cpu_s"] += thread_time() - cpu
                except ValueError as exc:
                    metrics["candidate_rejections"] += 1
                    result.errors.append(f"verification: {type(exc).__name__}: {exc}")
                    if "Missing projection participant" in str(exc):
                        break
                    continue
                # Transactional model update: if applying it crosses the deadline, restore.
                remaining(deadline)
                before = [p.copy() for p in model.params]
                t = perf_counter()
                model.update(verified)
                metrics["update_wall_s"] = perf_counter() - t
                if monotonic() > deadline:
                    for p, old in zip(model.params, before):
                        p[:] = old
                    raise TimeoutError("Round expired during model update")
                metrics["verification_completed"] = True
                result.accepted, result.vector = True, verified
                metrics["selected_edges"] = ",".join(str(x[1]) for x in pair)
                break
        except (ValueError, OSError) as exc:
            result.errors.append(f"round: {type(exc).__name__}: {exc}")
        metrics["round_wall_s"] = perf_counter() - start
        metrics["deadline_exceeded"] = int(monotonic() > deadline)
        metrics["accepted"] = int(result.accepted)
        metrics["timeouts"] = sum("Timeout" in e or "timed out" in e for e in result.errors)
        metrics["update_bound_exceeded"] = int(
            any("UPDATE_OUT_OF_RANGE" in e for e in result.errors)
        )
        return result

    def collect_metrics(self, round_id):
        """Telemetry is outside the round but inside end-to-end T_epsilon."""
        snapshots = {"coordinator": self.meter.snapshot(round_id)}
        errors = []

        def get(name):
            return self.call(name, 0, "stats", {"round": round_id})[0]

        with ThreadPoolExecutor(max_workers=len(self.peers)) as pool:
            futures = {pool.submit(get, name): name for name in self.peers}
            for future in as_completed(futures):
                try:
                    snapshots[futures[future]] = future.result()
                except (ValueError, OSError) as exc:
                    errors.append(str(exc))
        metrics = {
            "app_tx_bytes": int(sum(s.get("tx_bytes", 0) for s in snapshots.values())),
            "app_rx_bytes": int(sum(s.get("rx_bytes", 0) for s in snapshots.values())),
            "telemetry_missing_nodes": len(errors),
            "serialization_wall_sum_s": sum(
                s.get("serialization_s", 0) for s in snapshots.values()
            ),
            "socket_send_wall_sum_s": sum(s.get("socket_send_s", 0) for s in snapshots.values()),
            "socket_receive_wall_sum_s": sum(
                s.get("socket_receive_s", 0) for s in snapshots.values()
            ),
            "server_protocol_errors": sum(s.get("protocol_errors", 0) for s in snapshots.values()),
        }
        metrics["child_peak_rss_sum_bytes"] = sum(
            s.get("process_peak_rss_bytes", 0) for s in snapshots.values()
        )
        metrics["pacing_wait_wall_sum_s"] = sum(
            s.get("pacing_wait_s", 0) for s in snapshots.values()
        )
        metrics["pacing_reserved_bytes"] = sum(
            s.get("pacing_reserved_bytes", 0) for s in snapshots.values()
        )
        if self.shaper is not None and self.shaper.byte_budget is not None:
            metrics["app_byte_budget"] = self.shaper.byte_budget.state()
        for category in (
            "model_broadcast",
            "worker_compute_reply",
            "update_transfer",
            "aggregate_transfer",
            "proof_transfer",
            "control",
        ):
            for suffix in ("tx_bytes", "send_sum_s"):
                key = f"{category}_{suffix}"
                metrics[key] = sum(s.get(key, 0) for s in snapshots.values())
        return metrics

    def close(self):
        def stop(name):
            try:
                self.call(name, 0, "stop", {}, deadline=monotonic() + 0.5)
            except (ValueError, OSError):
                pass

        if self.peers:
            with ThreadPoolExecutor(max_workers=len(self.peers)) as pool:
                list(pool.map(stop, self.peers))
        deadline = monotonic() + 2.0
        for process in self.processes.values():
            if process.pid is not None:
                process.join(max(0.0, deadline - monotonic()))
        for process in self.processes.values():
            if process.pid is not None and process.is_alive():
                process.terminate()
        for process in self.processes.values():
            if process.pid is not None:
                process.join(1.0)
                if process.is_alive():
                    process.kill()
                    process.join(1.0)
                require(not process.is_alive(), "Child process could not be stopped")

    def __exit__(self, *args):
        self.close()
