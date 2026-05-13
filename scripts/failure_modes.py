"""Aggregate per-problem failure reasons from BFCL bake-off outputs.

Walks ``acceptance/bfcl/<model>/rep_<n>/<category>/*.json`` and groups
fail rows by (model, category, reason-bucket). Prints a markdown
breakdown so readers can see *why* each model lost the points it lost
without having to grep through the raw outputs themselves.

Usage:
    uv run python scripts/failure_modes.py
    uv run python scripts/failure_modes.py --rep 0 --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

CATS = [
    "simple_python", "multiple", "parallel", "parallel_multiple", "irrelevance",
    "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple",
    "live_irrelevance", "live_relevance",
]


def bucket_reason(category: str, reason: str, row: dict[str, Any]) -> str:
    """Map a grader 'reason' string to a coarse failure bucket."""
    r = reason or ""
    n_calls = len(row.get("actual_calls") or [])

    if r.startswith("emitted_") and "_expected_" in r:
        m = re.match(r"emitted_(\d+)_calls_expected_(\d+)", r)
        if m:
            got, want = int(m.group(1)), int(m.group(2))
            if got == 0:
                return "no_calls_emitted"
            if got < want:
                return f"under_called_{got}_of_{want}"
            return f"over_called_{got}_of_{want}"

    if r.startswith("called_tool_when_irrelevant"):
        return "over_called_when_irrelevant"

    if r.startswith("wrong_tool"):
        return "wrong_tool_name"

    if r.startswith("missing_required_arg") or "required_arg" in r:
        return "missing_required_arg"

    if r.startswith("arg_mismatch") or r.startswith("arg_value_mismatch") or "value_mismatch" in r:
        return "arg_value_mismatch"

    if r.startswith("extra_call") or "extra" in r:
        return "extra_call"

    if "no calls" in r or r == "no_calls" or (n_calls == 0 and category != "irrelevance"):
        return "no_calls_emitted"

    if not r:
        return "passed_or_no_reason"

    return f"other:{r[:60]}"


def walk(rep: int) -> dict[str, dict[str, dict[str, Any]]]:
    """model -> category -> {fail_count, fail_buckets, examples}"""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    bfcl = ROOT / "acceptance" / "bfcl"
    for model_dir in sorted(bfcl.iterdir()):
        if not model_dir.is_dir():
            continue
        rep_dir = model_dir / f"rep_{rep}"
        if not rep_dir.is_dir():
            continue
        per_cat: dict[str, dict[str, Any]] = {}
        for cat in CATS:
            cat_dir = rep_dir / cat
            if not cat_dir.is_dir():
                continue
            buckets: Counter[str] = Counter()
            examples: dict[str, list[str]] = defaultdict(list)
            n = 0
            n_pass = 0
            n_with_calls = 0
            for path in sorted(cat_dir.glob("*.json")):
                try:
                    row = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                n += 1
                if row.get("actual_calls"):
                    n_with_calls += 1
                if row.get("passed"):
                    n_pass += 1
                    continue
                bucket = bucket_reason(cat, row.get("reason", ""), row)
                buckets[bucket] += 1
                if len(examples[bucket]) < 2:
                    examples[bucket].append(row.get("id", path.stem))
            per_cat[cat] = {
                "n": n,
                "passed": n_pass,
                "n_with_calls": n_with_calls,
                "buckets": dict(buckets),
                "examples": {k: v for k, v in examples.items()},
            }
        out[model_dir.name] = per_cat
    return out


def render_md(data: dict[str, dict[str, dict[str, Any]]]) -> str:
    lines: list[str] = []
    lines.append("# BFCL failure-mode breakdown")
    lines.append("")
    lines.append(
        "Per-model, per-category counts of *why* each non-passing row failed. "
        "Buckets are derived from the grader's `reason` string. Pass criteria "
        "are the same as the leaderboard: simple/multiple = exactly one matching call; "
        "parallel/parallel_multiple = full set match (order-insensitive); "
        "irrelevance = zero calls."
    )
    lines.append("")
    for model in sorted(data.keys()):
        lines.append(f"## {model}")
        lines.append("")
        for cat in CATS:
            cs = data[model].get(cat)
            if not cs or cs["n"] == 0:
                continue
            n, passed = cs["n"], cs["passed"]
            failed = n - passed
            if failed == 0:
                lines.append(f"- **{cat}**: {passed}/{n} — clean sweep")
                continue
            lines.append(
                f"- **{cat}**: {passed}/{n} ({failed} failed, "
                f"{cs['n_with_calls']}/{n} emitted ≥1 call)"
            )
            for bucket, count in sorted(cs["buckets"].items(), key=lambda x: -x[1]):
                ex = cs["examples"].get(bucket, [])
                ex_str = f" e.g. {', '.join(ex)}" if ex else ""
                lines.append(f"    - `{bucket}` ×{count}{ex_str}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    data = walk(args.rep)
    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print(render_md(data))
    return 0


if __name__ == "__main__":
    sys.exit(main())
