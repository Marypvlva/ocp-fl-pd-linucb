"""Control-plane counters and credential-free JSONL events."""

from collections import Counter, deque
import json
from pathlib import Path
import resource
import threading
import time
import os


class ControlTelemetry:
    def __init__(self, path=None):
        self.lock = threading.Lock()
        self.counts = Counter()
        self.recent_errors = deque(maxlen=16)
        self.samples = deque(maxlen=1024)
        self.file = None
        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.file = path.open("a", buffering=1)

    def event(self, event, **fields):
        record = dict(at=time.time(), event=event, **fields)
        with self.lock:
            self.counts[event] += 1
            if "error_type" in fields:
                self.recent_errors.append(record)
            if "elapsed_s" in fields:
                self.samples.append(fields["elapsed_s"])
            if self.file:
                self.file.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    def snapshot(self):
        with self.lock:
            samples = sorted(self.samples)
            return dict(
                counts=dict(self.counts),
                recent_errors=list(self.recent_errors),
                recent_latency_samples=len(samples),
                recent_p50_p95_s=[samples[int((len(samples) - 1) * p)] for p in (0.5, 0.95)]
                if samples
                else [],
            )

    def close(self):
        with self.lock:
            if self.file:
                self.file.close()
                self.file = None


def process_load():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(
        cpu_s=time.process_time(),
        threads=threading.active_count(),
        peak_rss_bytes=rss if os.uname().sysname == "Darwin" else rss * 1024,
    )
