"""Acceptance check of a distributed stand before the paired live study.

Runs roles on the hosts of a placement (or as local processes), then checks:
healthy rounds, a real outage of the primary edge without and with the backup
lane, a double outage, recovery after the gates reopen, RPC round-trip times
and the wall-clock offsets reported by every host. Must run with
ADAPTIVE_CLOCK=wall. Writes JSON to stdout (and to --output).

  ADAPTIVE_CLOCK=wall python -m adaptive_planning.stand_check --placement placement.json
"""

import argparse
import json
from pathlib import Path
import statistics
from time import perf_counter

from . import clock
from .distributed import RemoteCluster, load_placement
from .tcp_runtime import Fault, flatten_model, make_model
from .training import make_workers


def timed(cluster, model, action, fault=None, deadline_s=3.0):
    t = perf_counter()
    result = cluster.run_round(model, action, deadline_s=deadline_s, fault=fault)
    return result, perf_counter() - t


def check(placement="local", workers=4, latency_ms=0.0, outage="refuse", rpc_timeout=1.0,
          repeats=5, ssh=None):
    groups, _, _ = make_workers(workers)
    roles = load_placement(placement, len(groups))
    proxies = ({n: dict(latency_ms=latency_ms, jitter_ms=latency_ms / 4) for n in roles}
               if latency_ms else None)
    out = dict(placement=roles, outage=outage, clock=clock.MODE)
    with RemoteCluster(groups, placement=roles, hidden=8, rpc_timeout=rpc_timeout, proxies=proxies,
                       edge_outage=outage, **(ssh or {})) as cluster:
        out["role_processes"] = len(cluster.remote)
        model = make_model(hidden=8)
        healthy = [timed(cluster, model, a) for a in ("F+P", "F+B", "R+P", "R+B") * repeats]
        out["healthy"] = dict(accepted=sum(r.accepted for r, _ in healthy), of=len(healthy),
                              median_wall_s=statistics.median(w for _, w in healthy))
        before = flatten_model(model)
        down_f, wall_f = timed(cluster, model, "F+P", Fault(failed_edges=(0,)))
        out["primary_down_F"] = dict(accepted=down_f.accepted, wall_s=wall_f,
                                     outages=down_f.metrics.get("physical_outages"),
                                     model_unchanged=flatten_model(model) == before,
                                     errors=[e[:80] for e in down_f.errors][:3])
        down_r, wall_r = timed(cluster, model, "R+P", Fault(failed_edges=(0,)))
        out["primary_down_R"] = dict(accepted=down_r.accepted, wall_s=wall_r,
                                     outages=down_r.metrics.get("physical_outages"))
        both, wall_b = timed(cluster, model, "R+B", Fault(failed_edges=(0, 2)))
        out["both_down_R"] = dict(accepted=both.accepted, wall_s=wall_b)
        missing, _ = timed(cluster, model, "F+P", Fault(missing_projection=(0, 1)))
        out["missing_projection_F+P"] = dict(accepted=missing.accepted)
        signed, _ = timed(cluster, model, "F+B", Fault(missing_projection=(0, 1)))
        out["missing_projection_F+B"] = dict(accepted=signed.accepted)
        after, _ = timed(cluster, model, "F+P")
        out["recovered"] = dict(accepted=after.accepted)
        out["gate_switches"] = len(cluster.gate_log)
        out["app_tx_bytes_last_round"] = cluster.collect_metrics(cluster.round_id)["app_tx_bytes"]
        rtt = {}
        for name in sorted(cluster.peers):
            samples = []
            for _ in range(repeats):
                t = perf_counter()
                cluster.call(name, 0, "stats", {"round": 1})
                samples.append(perf_counter() - t)
            rtt[name] = round(1000 * statistics.median(samples), 2)
        out["control_rtt_ms"] = rtt
    out["alive_after_close"] = sum(p.poll() is None for p in cluster.remote.values())
    ok = (out["healthy"]["accepted"] == out["healthy"]["of"]
          and not out["primary_down_F"]["accepted"] and out["primary_down_F"]["model_unchanged"]
          and out["primary_down_R"]["accepted"] and not out["both_down_R"]["accepted"]
          and not out["missing_projection_F+P"]["accepted"]
          and out["missing_projection_F+B"]["accepted"] and out["recovered"]["accepted"])
    out["passed"] = bool(ok)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--placement", default="local")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--latency-ms", type=float, default=0.0)
    parser.add_argument("--outage", choices=["refuse", "silent"], default="refuse")
    parser.add_argument("--rpc-timeout", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    placement, ssh = args.placement, None
    if placement != "local":
        data = json.loads(Path(placement).read_text())
        placement, ssh = {"roles": data["roles"]}, data.get("ssh")
    out = check(placement, args.workers, args.latency_ms, args.outage, args.rpc_timeout,
                args.repeats, ssh)
    text = json.dumps(out, indent=1, default=str)
    if args.output:
        Path(args.output).write_text(text)
    print(text)
    raise SystemExit(0 if out["passed"] else 1)


if __name__ == "__main__":
    main()
