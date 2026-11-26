"""Read all control-service events without deleting failed or incomplete rounds."""

import argparse
from collections import Counter
import json
from pathlib import Path


def report(directory):
    counters = Counter()
    lost_by_stage = Counter()
    lost_by_operation = Counter()
    lost_by_peer = Counter()
    missing = []
    files = []
    requests = 0
    latencies = []
    peak_threads = 0
    peak_rss = 0
    for episode in sorted(p for p in directory.iterdir() if p.is_dir()):
        if not list(episode.glob("round-*.json")) and not (episode / "failure.json").exists():
            continue
        log = episode / "control-server.jsonl"
        if not log.exists():
            missing.append(episode.name)
            continue
        files.append(str(log))
        with log.open() as stream:
            for line in stream:
                event = json.loads(line)
                name = event["event"]
                counters[name] += 1
                peak_threads = max(peak_threads, event.get("threads", 0))
                peak_rss = max(peak_rss, event.get("peak_rss_bytes", 0))
                if name in ("response_written", "response_lost"):
                    requests += 1
                    latencies.append(event["elapsed_s"])
                if name == "response_lost":
                    lost_by_stage[event["stage"]] += 1
                    lost_by_operation[event.get("operation", "unknown")] += 1
                    lost_by_peer[event["peer"]] += 1
    latencies.sort()
    return dict(
        server_logs=files,
        missing_server_logs=missing,
        event_counts=dict(counters),
        requests=requests,
        lost_by_stage=dict(lost_by_stage),
        lost_by_operation=dict(lost_by_operation),
        lost_by_peer=dict(lost_by_peer),
        p50_p95_p99_response_processing_s=[
            latencies[int((len(latencies) - 1) * p)] for p in (0.5, 0.95, 0.99)
        ]
        if latencies
        else None,
        peak_threads=peak_threads,
        peak_rss_bytes=peak_rss,
        interpretation="Response writes are not application acknowledgements. Missing v0.6.0 logs do not mean zero errors. All round and error records are retained.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    result = json.dumps(report(a.directory), indent=2) + "\n"
    if a.output:
        with a.output.open("x") as stream:
            stream.write(result)
    else:
        print(result, end="")
