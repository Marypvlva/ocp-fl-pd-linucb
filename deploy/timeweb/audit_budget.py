"""Read-only audit of a stopped episode's v0.6.0 or v0.6.1 SQLite frame journal."""

import argparse
import json
from pathlib import Path
import sqlite3


def audit(path, *, immutable=False):
    if immutable:
        wal = path.with_name(path.name + "-wal")
        if wal.exists() and wal.stat().st_size:
            raise ValueError(
                "Immutable audit requires a stopped checkpointed database; preserve the WAL"
            )
    db = sqlite3.connect(
        path.resolve().as_uri() + "?mode=ro" + ("&immutable=1" if immutable else ""), uri=True
    )
    try:
        db.execute("BEGIN")
        limit = db.execute("SELECT limit_bytes FROM config").fetchone()[0]
        spent, reserved, frames = db.execute(
            "SELECT COALESCE(SUM(sent),0),COALESCE(SUM(CASE WHEN sent IS NULL THEN size ELSE 0 END),0),COUNT(*) FROM frames"
        ).fetchone()
        invalid = db.execute(
            "SELECT COUNT(*) FROM frames WHERE size<0 OR sent<0 OR sent>size"
        ).fetchone()[0]
        unsettled = db.execute("SELECT COUNT(*) FROM frames WHERE sent IS NULL").fetchone()[0]
        result = dict(
            unsettled_frames=unsettled,
            path=str(path),
            limit=limit,
            spent=spent,
            reserved=reserved,
            frames=frames,
            valid=not invalid and 0 <= spent + reserved <= limit,
        )
        if db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='totals'"
        ).fetchone():
            actual = db.execute("SELECT spent,reserved,frames FROM totals WHERE id=1").fetchone()
            result["totals_match"] = actual == (spent, reserved, frames)
            result["valid"] &= result["totals_match"]
        return result
    finally:
        db.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    a = p.parse_args()
    rows = [audit(path) for path in sorted(a.directory.rglob("budget.sqlite"))]
    print(json.dumps(rows, indent=2))
    raise SystemExit(0 if rows and all(r["valid"] for r in rows) else 2)
