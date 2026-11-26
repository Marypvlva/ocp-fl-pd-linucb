"""Role agent for distributed stands: runs one worker or edge service on this host.

Started by the distributed launcher (directly or over SSH). It reads one JSON
line with the role specification from stdin, binds the requested address,
answers with one JSON line {"port", "key", "pid", "host"} on stdout and then
serves authenticated RPCs until the coordinator sends "stop" or the launcher
closes stdin. Every process of a distributed run uses the shared wall clock
(ADAPTIVE_CLOCK=wall), see clock.py.
"""

from concurrent.futures import ThreadPoolExecutor
import json
import os
import socket
import sys
import threading
from time import sleep, perf_counter

import numpy as np

from .tcp_runtime import EdgeService, Server, WorkerService
from .tcp_transport import Peer


def build(spec):
    keys = {name: bytes.fromhex(value) for name, value in spec["keys"].items()}
    identity, rpc_timeout = spec["identity"], float(spec["rpc_timeout"])
    if spec["role"] == "worker":
        w = spec["worker"]
        group = (np.asarray(w["X"], dtype=np.float64), np.asarray(w["y"], dtype=np.int64))
        service = WorkerService(identity, group, int(w["total_examples"]), int(w["hidden"]))
        key = service.worker.public_key.public_bytes_raw().hex()
    elif spec["role"] == "edge":
        e = spec["edge"]
        workers = [Peer(name, int(port), host) for name, host, port in e["workers"]]
        service = EdgeService(identity, workers, int(e["dimension"]), rpc_timeout)
        key = None
    else:
        raise ValueError("Unknown role")
    shaper = None
    if 'control' in spec:
        from .distributed_control import ControlClient, RemoteBudget
        from .lease_guard import LeaseGuard, DistributedShaper, read_evidence
        client = ControlClient(**spec['control'])
        pin = spec['lease_pin']
        if spec.get('chain_rpc'):
            from web3 import Web3
            w3 = Web3(Web3.HTTPProvider(spec['chain_rpc'], request_kwargs={'timeout': 5}))
            reader = lambda request: read_evidence(w3, pin, request)
        else:
            reader = lambda request: client.call('evidence', request)
        shaper = DistributedShaper(RemoteBudget(client), LeaseGuard(identity, pin, reader))
    server = Server(identity, spec["session"], keys, service, max(2.0, rpc_timeout + 0.5), shaper)
    return server, key


class Gate:
    """Real availability of this role, switched by the launcher between rounds.

    up      the role listens and answers;
    refuse  the listening socket is closed: the kernel refuses connections
            (RST), as after a crash of the role;
    silent  the socket stays open but nothing is accepted: callers connect,
            send and wait until their own deadline, as with a hung host.
    """

    MODES = ("up", "refuse", "silent")

    def __init__(self, bind):
        self.bind = bind
        self.mode = "up"
        self.lock = threading.Lock()
        self.listener = None
        self.port = 0
        self.open()

    def open(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.bind[0], self.port or self.bind[1]))
        listener.listen(64)
        listener.settimeout(0.1)
        self.listener = listener
        self.port = listener.getsockname()[1]

    def set(self, mode):
        if mode not in self.MODES:
            raise ValueError("Unknown gate mode")
        with self.lock:
            if mode == "refuse" and self.listener is not None:
                self.listener.close()
                self.listener = None
            elif mode != "refuse" and self.listener is None:
                self.open()
            self.mode = mode

    def accept(self):
        # The lock is held while waiting (<= 0.1 s), so a gate switch is
        # acknowledged only after any pending accept has returned.
        with self.lock:
            if self.listener is not None and self.mode == "up":
                try:
                    return self.listener.accept()[0]
                except (socket.timeout, OSError):
                    return None
        sleep(0.02)
        return None

    def close(self):
        with self.lock:
            if self.listener is not None:
                self.listener.close()
                self.listener = None


def serve(spec, out=sys.stdout, stdin=sys.stdin):
    if os.environ.get("ADAPTIVE_CLOCK") != "wall":
        raise RuntimeError("Distributed roles require ADAPTIVE_CLOCK=wall")
    server, key = build(spec)
    gate = Gate((spec.get("bind_host", "0.0.0.0"), int(spec.get("port", 0))))
    write_lock = threading.Lock()

    def emit(message):
        with write_lock:
            out.write(json.dumps(message) + "\n")
            out.flush()

    emit(dict(port=gate.port, key=key, pid=os.getpid(), host=spec.get("advertise_host", "127.0.0.1")))

    def control():
        # One JSON command per line; EOF means the launcher is gone.
        try:
            for line in stdin:
                command = json.loads(line)
                if 'grant' in command:
                    try:
                        ack = server.shaper.guard.install(command['grant'])
                        emit(dict(seq=command['seq'], grant=ack))
                    except Exception as exc:
                        emit(dict(seq=command['seq'], error=repr(exc)))
                else:
                    gate.set(command["gate"])
                    emit(dict(gate=gate.mode, seq=command.get("seq")))
        except (ValueError, KeyError, OSError) as exc:
            emit(dict(error=repr(exc)))
        finally:
            server.stopped.set()

    threading.Thread(target=control, daemon=True).start()
    try:
        with ThreadPoolExecutor(max_workers=16) as pool:
            while not server.stopped.is_set():
                sock = gate.accept()
                if sock is None:
                    continue
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                pool.submit(server.connection, sock, perf_counter())
    finally:
        gate.close()


def main():
    spec = json.loads(sys.stdin.readline())
    serve(spec)


if __name__ == "__main__":
    main()
