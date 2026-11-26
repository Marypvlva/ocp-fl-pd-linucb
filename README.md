# OCP-FL + PD-LinUCB

Resource-constrained scheduling of verifiable federated-learning rounds on edge nodes.

OCP-FL fixes a publicly verifiable random client roster. The collector experiment then selects one or two gateways for that roster, charges every attempted route, and learns from individually verified replies under delayed feedback and per-gateway quotas. A model update is applied after every selected client's reply passes the existing signature and projection checks.

The current software release is **v0.6.1**, dated 26 November 2026. It includes the distributed Collector v3 execution for Timeweb Cloud and a control-service reliability fix. Existing stands should follow the [v0.6.1 update instructions](deploy/timeweb/UPDATE_061_RU.md). The archived collector study remains version v3. Its code, frozen plans, archived measurements, per-draw outcomes and TCP/EVM logs are in [`experiments/collector_v3`](experiments/collector_v3). The [Russian report](reports/collector-v3-report-ru.pdf) describes the results and implementation work required for a multi-host confirmatory study.

## Install

Use Python 3.13 on Linux or macOS. From a fresh checkout

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python project.py doctor
```

The compiled EVM artifacts are supplied. Rebuilding the original scheduling contract requires Node.js 22 and `npm ci --prefix contracts && npm run build --prefix contracts`.

## Verify the published collector results

```bash
python experiments/collector_v3/kit/analyze_v3.py \
  experiments/collector_v3/data reproduced.json
cmp experiments/collector_v3/data/summary.json reproduced.json
python experiments/collector_v3/kit/check_bounds.py
python experiments/collector_v3/kit/check_feedback.py
python experiments/collector_v3/kit/check_collector.py .
python experiments/collector_v3/kit/check_leases.py .
```

To recompute the replay, including April-only calibration and all A/B/C comparisons, from the archived matrices

```bash
python experiments/collector_v3/kit/replay.py \
  . experiments/collector_v3/data results/collector-v3-replay
```

The output directory must be new. This uses the stored measurements and needs no network access. The source acquisition programs and provenance are included. RIPE Atlas [measurement documentation](https://atlas.ripe.net/docs/apis/rest-api-manual/anchors/anchor-measurements/) describes the source service.

## Run experiments in the cloud

Start with the detailed [Timeweb runbook in Russian](deploy/timeweb/RUNBOOK_RU.md). It covers Ubuntu 24.04 x86_64 bootstrap, a verified Linux hash lock, coordinator-side SSH deployment, clock and peer diagnostics, frozen plans, natural latency runs and HTTPS result export.

The distributed entry point is `experiments/collector_v3/kit/live.py --plan ...`. It uses `RemoteCluster`, a durable atomic application-byte admission service and node-side CollectorLeases verification.

See also [the previous cloud instructions](docs/CLOUD_COLLECTOR_V3.md) for a Linux VM run and the existing multi-host deployment entry points. The repository includes the original TCP runtime and its distributed launcher, Yandex Cloud infrastructure templates, and the current collector experiment. Use a fresh output directory for each independent study and record `git rev-parse HEAD` with its configuration.

## What the current results establish

Three disjoint probe panels with separate dates were evaluated using archived network measurements. The original joint collector policy exceeded an optimistic pre-roster reservation bound by 32.02, 36.16 and 27.16 accepted rounds per 480, when the bound allows discarding reserved resources after reveal. A matched factorial study found a positive interaction between roster information and per-resource prices. In a third held-out series, paid individual binary receipts yielded 216.03 acceptances versus 176.22 for the best tested aggregate-only method selected separately per panel.

These results depend on the workload, resource regime and measurement replay described in the report. Individually informed greedy and recency controls remain strong. The UCB bonus alone has no established substantial advantage in these series. Local EVM and TCP runs validate protocol execution. Distributed spending admission and queue telemetry are implemented. Real interhost performance, end-to-end external consensus finality and hardware sizing require measurement on the operator-provided cloud VMs. In the replay, an exclusive extra 5 ms for late booking preserves the chosen practical-effect threshold in all three series. At larger delays the evidence weakens.

## Project layout

| Directory | Contents |
| --- | --- |
| `adaptive_planning/` | TCP protocol, client sampling, verification, training and distributed roles |
| `contracts/` | OCP scheduling contract, pinned compiler and compiled artifact |
| `experiments/collector_v3/kit/` | Collector policies, reservation contract, replay and checks |
| `experiments/collector_v3/data/` | Measurements, frozen plans' outcomes, published logs and summary |
| `deploy/` | Existing distributed study driver and Yandex Cloud templates |
| `tests/` | Protocol and scheduler regression suite |
| `reports/` | Current research report |
| `docs/` | Cloud instructions and earlier experiment documentation |

The [previous project guide](docs/README_PREVIOUS_RU.md) documents the older CLI studies. Their outcomes have separate workloads and comparison bases. The collector experiment's primary entry points are the commands above.

## Reuse in a paper

Use a tagged version together with its exact commit SHA, frozen plan, configuration, seeds and raw output files. [`CITATION.cff`](CITATION.cff) provides software citation metadata. Report new cloud outcomes with their own hardware, clock synchronization, network placement, complete latency and spending measurements. The supplied report records the original research handoff. The [Timeweb implementation report](deploy/timeweb/IMPLEMENTATION_REPORT_RU.md) describes subsequent changes, checks and remaining empirical limits.
