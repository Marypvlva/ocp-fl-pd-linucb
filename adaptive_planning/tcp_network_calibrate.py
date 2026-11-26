"""Measure actual multi-process TCP throughput of one shared paced bottleneck."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import multiprocessing as mp
from pathlib import Path
import secrets
import socket
from time import monotonic, perf_counter

from .tcp_network import NetworkProfile, NetworkShaper
from .tcp_transport import Meter, receive_message, send_message


def sender(port, worker_id, payload_size, shaper, barrier, key, session):
    with socket.create_connection(("127.0.0.1", port), timeout=15) as sock:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        barrier.wait(timeout=15)
        deadline = monotonic() + 15
        header = dict(
            version=2,
            session=session,
            epoch=1,
            round=1,
            sender=f"w{worker_id}",
            recipient="e0",
            nonce=secrets.token_hex(16),
            kind="fragment",
            response=True,
            body={},
            deadline_ns=int(deadline * 1e9),
        )
        send_message(sock, header, b"x" * payload_size, key, deadline, Meter(), shaper)


def calibrate(streams=4, payload_size=500_000, mbps=40.0):
    context = mp.get_context("spawn")
    shaper = NetworkShaper(context, NetworkProfile("calibration", edge_mbps=mbps))
    barrier = context.Barrier(streams + 1)
    keys = {f"w{i}": secrets.token_bytes(32) for i in range(streams)}
    session = secrets.token_hex(16)
    processes, sockets = [], []
    meter = Meter()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(streams)
        listener.settimeout(15)
        try:
            for i in range(streams):
                process = context.Process(
                    target=sender,
                    args=(
                        listener.getsockname()[1],
                        i,
                        payload_size,
                        shaper,
                        barrier,
                        keys[f"w{i}"],
                        session,
                    ),
                )
                process.start()
                processes.append(process)
            for _ in range(streams):
                sock, _ = listener.accept()
                sockets.append(sock)

            def receive(sock):
                header, payload = receive_message(
                    sock, keys, "e0", session, monotonic() + 15, meter
                )
                if payload != b"x" * payload_size:
                    raise ValueError("Calibration payload corrupted")
                return header["sender"]

            with ThreadPoolExecutor(max_workers=streams) as pool:
                futures = [pool.submit(receive, sock) for sock in sockets]
                start = perf_counter()
                barrier.wait(timeout=15)
                senders = [future.result() for future in futures]
                elapsed = perf_counter() - start
            if len(set(senders)) != streams:
                raise ValueError("Duplicate calibration sender")
            for process in processes:
                process.join(2)
                if process.exitcode != 0:
                    raise RuntimeError(f"Calibration process exit {process.exitcode}")
            frame_bytes = int(meter.snapshot(1)["rx_bytes"])
            return dict(
                streams=streams,
                payload_bytes_per_stream=payload_size,
                frame_bytes=frame_bytes,
                elapsed_wall_s=elapsed,
                ideal_budget_s=frame_bytes / (mbps * 1e6),
                measured_mb_s=frame_bytes / elapsed / 1e6,
                configured_mb_s=mbps,
            )
        finally:
            for sock in sockets:
                sock.close()
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(2)
                if process.is_alive():
                    process.kill()
                    process.join(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streams", nargs="+", type=int, default=[1, 4, 12])
    parser.add_argument("--payload-bytes", type=int, default=1_000_000)
    parser.add_argument("--mbps", type=float, default=40.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (
        not all(1 <= n <= 16 for n in args.streams)
        or not 0 < args.payload_bytes <= 15_000_000
        or not 0 < args.mbps < 10000
        or args.repeats < 1
    ):
        parser.error("Invalid calibration limits")
    rows = []
    for streams in args.streams:
        for repeat in range(args.repeats):
            row = dict(repeat=repeat, **calibrate(streams, args.payload_bytes, args.mbps))
            rows.append(row)
            print(row, flush=True)
    with args.output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
