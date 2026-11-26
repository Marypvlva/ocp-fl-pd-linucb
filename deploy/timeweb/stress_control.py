"""Real loopback HTTPS credit-service load, with optional one-CPU limits and externally applied cgroup memory limits.

This is a control-plane benchmark. Logical settled bytes are test inputs;
there is no FL training and no claim of interhost measurements.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from adaptive_planning.distributed_control import Ledger, ControlServer, ControlClient
from adaptive_planning.control_telemetry import process_load


def percentile(values, p):
    values = sorted(values)
    return values[int((len(values) - 1) * p)] if values else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--frames", type=int, default=60480)
    p.add_argument("--owners", type=int, default=9)
    p.add_argument("--concurrency", type=int, default=18)
    p.add_argument("--one-cpu", action="store_true")
    p.add_argument("--memory-mb", type=int)
    a = p.parse_args()
    if min(a.frames, a.owners, a.concurrency) <= 0:
        raise ValueError("Positive workload required")
    if a.one_cpu:
        if not hasattr(os, "sched_setaffinity"):
            raise ValueError("CPU affinity requires Linux")
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    if a.memory_mb:
        resource.setrlimit(resource.RLIMIT_AS, (a.memory_mb * 1024**2, a.memory_mb * 1024**2))
    a.out.mkdir(parents=True, exist_ok=False)
    key = a.out / "tls.key"
    cert = a.out / "tls.crt"
    conf = a.out / "tls.cnf"
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
    ledger = Ledger(a.out / "budget.sqlite", a.frames * 128)
    keys = {f"role-{i}": os.urandom(32) for i in range(a.owners)}
    server = ControlServer(
        ("127.0.0.1", 0),
        ledger,
        keys,
        lambda _: {},
        str(cert),
        str(key),
        log_path=a.out / "server.jsonl",
    ).start()
    clients = [
        ControlClient(
            f"https://127.0.0.1:{server.http.server_port}",
            name,
            secret.hex(),
            str(cert),
            log_path=a.out / f"{name}.jsonl",
        )
        for name, secret in keys.items()
    ]
    started = time.perf_counter()

    def send(i):
        begin = time.perf_counter()
        client = clients[i % len(clients)]
        try:
            reply = client.call("reserve", dict(id=str(i), size=128))
            if not reply["admitted"]:
                raise ValueError("Unexpected stale admission")
            client.call("settle", dict(id=str(i), sent=80))
            return dict(frame=i, elapsed_s=time.perf_counter() - begin)
        except Exception as exc:
            return dict(
                frame=i, error_type=type(exc).__name__, elapsed_s=time.perf_counter() - begin
            )

    try:
        with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
            rows = list(pool.map(send, range(a.frames)))
        elapsed = time.perf_counter() - started
        state = ledger.audit()
        failures = [r for r in rows if "error_type" in r]
        latencies = [r["elapsed_s"] for r in rows]
        cgroup = {}
        if Path("/proc/self/cgroup").exists():
            for line in Path("/proc/self/cgroup").read_text().splitlines():
                if line.startswith("0::"):
                    base = Path("/sys/fs/cgroup") / line[3:].lstrip("/")
                    cgroup = {
                        name: (base / name).read_text().strip()
                        for name in ("memory.max", "cpu.max", "memory.peak")
                        if (base / name).exists()
                    }
        result = dict(
            cgroup=cgroup,
            kind="synthetic_credit_service_load_over_real_loopback_https",
            frames=a.frames,
            owners=a.owners,
            concurrency=a.concurrency,
            affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            memory_limit_mb=a.memory_mb,
            elapsed_s=elapsed,
            frames_per_s=a.frames / elapsed,
            p50_p95_p99_s=[percentile(latencies, x) for x in (0.5, 0.95, 0.99)],
            first_1000_p95_s=percentile(latencies[:1000], 0.95),
            last_1000_p95_s=percentile(latencies[-1000:], 0.95),
            failures=len(failures),
            failure_samples=failures[:20],
            ledger=state,
            load=process_load(),
            server=server.telemetry.snapshot(),
            connections=server.http.load(),
            clients=[c.telemetry.snapshot() for c in clients],
        )
        (a.out / "measurements.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (a.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {k: v for k, v in result.items() if k not in ("clients", "server")}, indent=2
            )
        )
        if failures or state != dict(spent=80 * a.frames, reserved=0, frames=a.frames):
            raise SystemExit(1)
    finally:
        for client in clients:
            client.close()
        server.close()
        ledger.close()
        key.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
