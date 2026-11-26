"""Distributed stand: worker and edge roles on separate hosts.

RemoteCluster keeps the protocol, keys, certificates and measurements of the
local Cluster and replaces process spawning: every role runs `python -m
adaptive_planning.node` on the host given by a placement map, started over SSH
(cloud VMs) or as a local subprocess (emulation). Pairwise HMAC keys and the
role data are sent once over the launcher's stdin channel; peers address each
other by host and port. All processes use the NTP-synchronized wall clock for
deadlines (ADAPTIVE_CLOCK=wall).

For local rehearsals, NetemProxy places a TCP relay in front of every role and
adds one-way latency, jitter and outages (connections are accepted and then
left silent), so timeouts, deadline misses and retries happen in the real
socket stack. On cloud VMs the network is real; the same proxy can still be
used to add WAN-like latency between zones.

Legacy app_byte_budget uses shared memory. Collector experiments supply an
authenticated durable control service and a RemoteBudget at every sender.
"""

import json
import os
import random
import select
import shlex
import socket
import subprocess
import sys
import threading
import time

from . import clock
from dataclasses import replace

from .tcp_runtime import Cluster, Fault, PROFILES, require
from .tcp_transport import Peer


class NetemProxy:
    """Threaded TCP relay with latency, jitter and scheduled silence."""

    def __init__(self, target_host, target_port, *, latency_ms=0.0, jitter_ms=0.0, seed=0):
        self.target = (target_host, target_port)
        self.latency, self.jitter = latency_ms / 1000.0, jitter_ms / 1000.0
        self.random = random.Random(seed)
        self.silent = threading.Event()
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(64)
        self.listener.settimeout(0.1)
        self.port = self.listener.getsockname()[1]
        self.stopped = threading.Event()
        threading.Thread(target=self._accept, daemon=True).start()

    def delay(self):
        return max(0.0, self.latency + self.random.uniform(-self.jitter, self.jitter))

    def _accept(self):
        while not self.stopped.is_set():
            try:
                client, _ = self.listener.accept()
            except socket.timeout:
                continue
            threading.Thread(target=self._relay, args=(client,), daemon=True).start()

    def _pipe(self, src, dst, done):
        try:
            while not done.is_set():
                ready, _, _ = select.select([src], [], [], 0.1)
                if not ready:
                    continue
                data = src.recv(65536)
                if not data:
                    break
                time.sleep(self.delay())
                dst.sendall(data)
        except OSError:
            pass
        finally:
            done.set()

    def _relay(self, client):
        if self.silent.is_set():
            # Outage: accept, never answer, close later; the caller times out.
            time.sleep(30)
            client.close()
            return
        try:
            upstream = socket.create_connection(self.target, timeout=5)
        except OSError:
            client.close()
            return
        done = threading.Event()
        threads = [threading.Thread(target=self._pipe, args=(a, b, done), daemon=True)
                   for a, b in ((client, upstream), (upstream, client))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        client.close()
        upstream.close()

    def close(self):
        self.stopped.set()
        self.listener.close()


class RemoteCluster(Cluster):
    """Cluster whose roles run on the hosts of a placement map.

    placement: {"w0": "10.128.0.11", ..., "e0": "10.129.0.5", ...}; the value
    "local" starts the role as a subprocess of this machine.

    physical_faults=True turns the trace's edge outages into real ones: before
    the round the launcher switches the gate of each failed edge to
    `edge_outage` ("refuse": the port is closed, "silent": the port accepts and
    never answers) and reopens it after the round. Worker data withholding
    (missing projection or full update) stays an application-level behaviour of
    the worker role, as in the threat model.
    """

    def __init__(self, groups, *, placement, ssh_user=None, ssh_options=(),
                 remote_dir="~/adaptive_planning", remote_python=".venv/bin/python",
                 bind_host="0.0.0.0", proxies=None, physical_faults=True,
                 edge_outage="refuse", control_timeout=10.0, control=None, lease_pin=None, chain_rpc=None, **kwargs):
        require(clock.MODE == "wall", "Distributed runs require ADAPTIVE_CLOCK=wall")
        require(kwargs.get("app_byte_budget") is None, "Hard byte budgets need one host")
        require(kwargs.get("network") in (None, PROFILES["unlimited"]),
                "Use the real network or NetemProxy instead of shared-memory pacing")
        require(edge_outage in ("refuse", "silent"), "Unknown outage mode")
        super().__init__(groups, **kwargs)
        names = [f"w{i}" for i in range(len(groups))] + [f"e{i}" for i in range(4)]
        missing = [n for n in names if n not in placement]
        require(not missing, f"Placement misses roles: {missing}")
        self.placement = dict(placement)
        self.ssh_user, self.ssh_options = ssh_user, tuple(ssh_options)
        self.remote_dir, self.remote_python = remote_dir, remote_python
        self.bind_host = bind_host
        self.proxy_settings = proxies or {}
        self.physical_faults, self.edge_outage = physical_faults, edge_outage
        self.control_timeout = control_timeout
        self.proxies, self.remote, self.replies = {}, {}, {}
        self.sequence = 0
        self.gate_log = []
        self.control, self.lease_pin, self.chain_rpc = control, lease_pin, chain_rpc

    def _command(self, host):
        if host == "local":
            return [sys.executable, "-m", "adaptive_planning.node"], dict(
                os.environ, ADAPTIVE_CLOCK="wall", PYTHONUNBUFFERED="1")
        target = f"{self.ssh_user}@{host}" if self.ssh_user else host
        script = (f"cd {self.remote_dir} && ADAPTIVE_CLOCK=wall PYTHONUNBUFFERED=1 "
                  f"{self.remote_python} -m adaptive_planning.node")
        return ["ssh", *self.ssh_options, target, "bash -lc " + shlex.quote(script)], None

    def _spec(self, name, args):
        host = self.placement[name]
        spec = dict(identity=name, session=self.session, rpc_timeout=self.rpc_timeout,
                    keys={peer: key.hex() for peer, key in self.keys[name].items()},
                    bind_host="127.0.0.1" if host == "local" else self.bind_host,
                    advertise_host="127.0.0.1" if host == "local" else host,
                    port=23000+(int(name[1:]) if name.startswith("w") else 16+int(name[1:])) if self.control else 0)
        if self.control:
            spec.update(control=dict(self.control[name]), lease_pin=self.lease_pin, chain_rpc=self.chain_rpc)
        if name.startswith("w"):
            (X, y), total, hidden = args
            spec.update(role="worker", worker=dict(X=X.tolist(), y=[int(v) for v in y],
                                                   total_examples=int(total), hidden=int(hidden)))
        else:
            workers, dimension = args
            spec.update(role="edge", edge=dict(
                workers=[[p.name, p.host, p.port] for p in workers], dimension=int(dimension)))
        return spec

    def _reader(self, name, process):
        replies = self.replies[name]
        for line in process.stdout:
            replies.put(json.loads(line))
        replies.put(None)

    def _start(self, specs):
        import queue

        launched = []
        for name, args in specs:
            command, env = self._command(self.placement[name])
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.DEVNULL, env=env, text=True, bufsize=1)
            process.stdin.write(json.dumps(self._spec(name, args)) + "\n")
            process.stdin.flush()
            self.remote[name] = process
            self.replies[name] = queue.Queue()
            threading.Thread(target=self._reader, args=(name, process), daemon=True).start()
            launched.append(name)
        deadline = time.monotonic() + self.startup_timeout
        for name in launched:
            try:
                info = self.replies[name].get(timeout=max(0.0, deadline - time.monotonic()))
            except Exception:
                raise TimeoutError(f"Startup timeout: {name}") from None
            require(info is not None and "port" in info, f"Role {name} failed to start")
            host, port = info["host"], int(info["port"])
            settings = self.proxy_settings.get(name)
            if settings is not None:
                proxy = NetemProxy(host, port, seed=sum(map(ord, name)), **settings)
                self.proxies[name] = proxy
                host, port = "127.0.0.1", proxy.port
            self.peers[name] = Peer(name, port, host)
            if info["key"]:
                from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
                self.public_keys[int(name[1:])] = Ed25519PublicKey.from_public_bytes(
                    bytes.fromhex(info["key"]))

    def install_grant(self, request):
        """Require independent contract validation from every role before traffic."""
        pending = {}
        for name, process in self.remote.items():
            self.sequence += 1
            pending[name] = self.sequence
            process.stdin.write(json.dumps(dict(grant=request, seq=self.sequence)) + "\n")
            process.stdin.flush()
        answers = {}
        deadline = time.monotonic() + self.control_timeout
        for name, seq in pending.items():
            ack = self.replies[name].get(timeout=max(.01, deadline-time.monotonic()))
            require(ack is not None and ack.get('seq') == seq and 'error' not in ack,
                    f'Grant validation failed at {name}: {ack}')
            require(ack['grant']['digest'] == request['digest'], 'Grant acknowledgement mismatch')
            answers[name] = ack['grant']
        return answers

    def gates(self, modes):
        """Switch real availability of roles and wait for every acknowledgement."""
        sent = {}
        for name, mode in modes.items():
            self.sequence += 1
            sent[name] = self.sequence
            stream = self.remote[name].stdin
            stream.write(json.dumps(dict(gate=mode, seq=self.sequence)) + "\n")
            stream.flush()
        deadline = time.monotonic() + self.control_timeout
        for name, seq in sent.items():
            while True:
                reply = self.replies[name].get(timeout=max(0.01, deadline - time.monotonic()))
                require(reply is not None and "error" not in reply, f"Gate failure: {name}")
                if reply.get("seq") == seq:
                    require(reply["gate"] == modes[name], f"Gate mismatch: {name}")
                    break
        self.gate_log.append(dict(round=self.round_id + 1, modes=dict(modes), t=clock.now()))

    def run_round(self, model, action, *, fault=None, **kwargs):
        fault = fault or Fault()
        if not self.physical_faults or not (fault.failed_edges or fault.stalled_edges):
            return super().run_round(model, action, fault=fault, **kwargs)
        modes = {f"e{i}": self.edge_outage for i in fault.failed_edges}
        modes.update({f"e{i}": "silent" for i in fault.stalled_edges})
        self.gates(modes)
        try:
            result = super().run_round(
                model, action, fault=replace(fault, failed_edges=(), stalled_edges=()), **kwargs)
            result.metrics["physical_outages"] = sorted(modes)
            result.metrics["outage_mode"] = self.edge_outage
            return result
        finally:
            self.gates({name: "up" for name in modes})

    def set_outage(self, name, silent):
        """Silence a role's relay (local rehearsal of a network partition)."""
        (self.proxies[name].silent.set if silent else self.proxies[name].silent.clear)()

    def close(self):
        for name in list(self.remote):
            try:
                self.call(name, 0, "stop", {}, deadline=clock.now() + 1.0)
            except (ValueError, OSError, KeyError):
                pass
        for process in self.remote.values():
            try:
                process.stdin.close()
            except OSError:
                pass
        for process in self.remote.values():
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
        for proxy in self.proxies.values():
            proxy.close()
        self.remote.clear()


def load_placement(path_or_dict, workers):
    """Placement from a JSON file (written by deploy/yandex/inventory.py) or a dict.

    Accepts {"roles": {"w0": host, ...}} or {"workers": [hosts], "edges": [4 hosts]}
    (workers are assigned round-robin), or the string "local".
    """
    if path_or_dict == "local":
        return {**{f"w{i}": "local" for i in range(workers)}, **{f"e{i}": "local" for i in range(4)}}
    data = path_or_dict
    if not isinstance(data, dict):
        from pathlib import Path

        data = json.loads(Path(data).read_text())
    if "roles" in data:
        return dict(data["roles"])
    hosts, edges = data["workers"], data["edges"]
    require(len(edges) == 4 and hosts, "Placement needs worker hosts and four edge hosts")
    return {**{f"w{i}": hosts[i % len(hosts)] for i in range(workers)},
            **{f"e{i}": edges[i] for i in range(4)}}
