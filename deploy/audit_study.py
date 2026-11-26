"""Run the standard independent audit (ocp_analyze.validate) on every live episode.

  python deploy/audit_study.py study_cloud

The audit re-verifies OCP certificates (commit-reveal, BLS beacon, shuffle),
EVM receipts and gas, journal/round consistency, causal planner contexts and,
for AF planners, the reward equations (including latent residual offsets).
"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deploy.live_study import check_plan, require_complete
from adaptive_planning.ocp_analyze import validate  # noqa: E402


def main():
    out = Path(sys.argv[1])
    plan = json.loads((out / "plan.json").read_text())
    check_plan(plan)
    require_complete(out, plan)
    expected = {
        f"runs/{c}/{s}/{p}" for c in plan["cells"] for s in plan["seeds"] for p in plan["policies"]
    }
    results, failed = [], 0
    for folder in sorted(out.glob("runs/*/*/*")):
        if str(folder.relative_to(out)) not in expected:
            continue
        try:
            _, summary, rows = validate(folder)
            results.append(
                dict(
                    episode=str(folder.relative_to(out)),
                    status="pass",
                    rounds=len(rows),
                    accepted=int(summary[0]["accepted_updates"]),
                )
            )
        except Exception as exc:  # noqa: BLE001 - every failure is reported
            failed += 1
            results.append(
                dict(episode=str(folder.relative_to(out)), status="fail", error=repr(exc)[:500])
            )
    (out / "audit.json").write_text(
        json.dumps(dict(episodes=len(results), failed=failed, results=results), indent=1)
    )
    print(f"audited {len(results)} episodes, failed {failed}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
