"""Per-problem regrade for the multi-turn audit step.

Run as: uv run python scripts/audit_one_multi_turn.py <path_to_problem_json> <category>

Returns exit code 0 with one line: "PASS_<bool>" or "ERROR_<msg>".
Designed to be called in a fresh subprocess so bfcl_eval's globals()
instance cache doesn't carry over from earlier regrades.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.bfcl.adapter import load_ground_truth, load_problems  # noqa: E402
from benchmarks.bfcl.grade import grade_multi_turn  # noqa: E402


def main() -> int:
    if len(sys.argv) != 3:
        print("ERROR_usage: audit_one_multi_turn.py <problem_json> <category>")
        return 1
    path = Path(sys.argv[1])
    category = sys.argv[2]
    row = json.loads(path.read_text())

    # Load the original problem + GT for this id.
    problems = load_problems(category) or []
    by_id = {p.get("id"): p for p in problems}
    p = by_id.get(row["id"])
    if p is None:
        print(f"ERROR_problem_not_found:{row['id']}")
        return 1
    gt = load_ground_truth(category).get(row["id"])
    if gt is None:
        print(f"ERROR_gt_not_found:{row['id']}")
        return 1

    test_entry = dict(p)
    test_entry["ground_truth"] = gt

    # Use a unique model_name per call to defeat the globals() cache even
    # within the subprocess (this is belt-and-suspenders — fresh subprocess
    # alone is sufficient, but the unique name protects against any future
    # in-process reuse).
    res = grade_multi_turn(
        per_turn_steps=row.get("per_turn_steps") or [],
        test_entry=test_entry,
        category=category,
        model_name=f"audit_{uuid.uuid4().hex}",
    )
    print(f"PASS_{res.passed}|REASON_{res.reason[:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
