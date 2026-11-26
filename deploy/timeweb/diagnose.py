"""Small peer-to-peer TCP RTT probe for an explicitly supplied VM inventory."""

import argparse
import json
import socket
import time
import threading
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["listen", "probe"])
    p.add_argument("--hosts", nargs="*")
    a = p.parse_args()
    if a.action == "listen":
        with socket.create_server(("0.0.0.0", 23100), reuse_port=False) as server:
            stopped = threading.Event()
            threading.Thread(target=lambda: (sys.stdin.read(), stopped.set()), daemon=True).start()
            server.settimeout(0.2)
            print("READY", flush=True)
            while not stopped.is_set():
                try:
                    sock, _ = server.accept()
                except socket.timeout:
                    continue
                with sock:
                    sock.sendall(b"OCP diagnostic\n")
    else:
        result = {}
        for host in a.hosts:
            start = time.perf_counter()
            with socket.create_connection((host, 23100), timeout=5) as s:
                if s.recv(100) != b"OCP diagnostic\n":
                    raise ValueError("Unexpected peer")
            result[host] = 1000 * (time.perf_counter() - start)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
