"""Post-hoc grading + leaderboard for BFCL bake-off runs.

Walks ``acceptance/bfcl/<model>/rep_<n>/<category>/*.json``, applies
``benchmarks.bfcl.grade.grade`` against BFCL ground truth, and prints a
graded leaderboard. Optionally writes the graded summary back to disk
(``--write-back``) so future re-runs of ``aggregate.py`` see ``passed``
fields populated.

Usage:
    uv run python scripts/grade_bakeoff.py
    uv run python scripts/grade_bakeoff.py --rep 0 --write-back
    uv run python scripts/grade_bakeoff.py --models qwen25-coder-1.5b-instruct
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.bfcl.adapter import ALL_CATEGORIES, load_ground_truth, load_problems  # noqa: E402
from benchmarks.bfcl.grade import grade, grade_multi_turn  # noqa: E402
from benchmarks.bfcl.multi_turn import MULTI_TURN_CATEGORIES, is_multi_turn  # noqa: E402

GRADED_CATEGORIES = tuple(ALL_CATEGORIES) + MULTI_TURN_CATEGORIES


@dataclass
class CatStats:
    n: int = 0
    passed: int = 0
    n_with_calls: int = 0
    wall_s: float = 0.0
    completion_tokens: int = 0
    errors: int = 0


def _grade_one_problem(
    category: str,
    row: dict[str, Any],
    gt: list | None,
) -> tuple[bool, str]:
    """Return (passed, reason) by re-running the grader against an on-disk row.

    Per-problem JSONs store actual_calls as ``[[name, args], ...]`` (JSON tuples
    serialize as lists). The grader takes ``list[tuple[str, dict]]`` so we
    convert here.
    """
    raw = row.get("actual_calls", []) or []
    calls: list[tuple[str, dict[str, Any]]] = []
    for entry in raw:
        if isinstance(entry, list) and len(entry) == 2:
            name, args = entry
            calls.append((str(name), dict(args) if isinstance(args, dict) else {}))
        elif isinstance(entry, dict) and "name" in entry:
            calls.append((str(entry["name"]),
                          dict(entry.get("arguments") or entry.get("parameters") or {})))
    res = grade(category, calls, gt)
    return res.passed, res.reason


def _grade_one_multi_turn(
    category: str,
    row: dict[str, Any],
    test_entry: dict[str, Any] | None,
    model_id: str,
) -> tuple[bool, str]:
    """Multi-turn: hand per-turn-step call-strings to bfcl_eval's state checker.

    `test_entry` must include the problem record + a `ground_truth` key
    (merged from the possible_answer file). Returns (False, "missing_*")
    if either the row or test_entry is malformed.
    """
    if test_entry is None:
        return False, "missing_test_entry"
    if "ground_truth" not in test_entry:
        return False, "missing_ground_truth"
    per_turn_steps = row.get("per_turn_steps") or []
    res = grade_multi_turn(
        per_turn_steps=per_turn_steps,
        test_entry=test_entry,
        category=category,
        model_name=model_id,
    )
    return res.passed, res.reason


def grade_model(
    model_dir: Path,
    rep: int,
    write_back: bool,
) -> dict[str, CatStats]:
    """Walk one model's rep dir and return per-category graded stats."""
    rep_dir = model_dir / f"rep_{rep}"
    if not rep_dir.is_dir():
        return {}
    out: dict[str, CatStats] = {}
    gt_cache: dict[str, dict[str, list]] = {}
    # Multi-turn needs the full problem record (initial_config, involved_classes)
    # in addition to GT. Built lazily per-category.
    mt_entry_cache: dict[str, dict[str, dict]] = {}
    model_id = model_dir.name
    for cat in GRADED_CATEGORIES:
        cat_dir = rep_dir / cat
        if not cat_dir.is_dir():
            continue
        cs = CatStats()
        if is_multi_turn(cat):
            if cat not in mt_entry_cache:
                problems = load_problems(cat) or []
                try:
                    gt_for_cat = load_ground_truth(cat)
                except FileNotFoundError:
                    gt_for_cat = {}
                # test_entry per id = problem fields + ground_truth list.
                merged: dict[str, dict] = {}
                for p in problems:
                    pid = p.get("id")
                    if not pid:
                        continue
                    entry = dict(p)
                    entry["ground_truth"] = gt_for_cat.get(pid)
                    merged[pid] = entry
                mt_entry_cache[cat] = merged
            for path in sorted(cat_dir.glob("*.json")):
                try:
                    row = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                cs.n += 1
                cs.wall_s += float(row.get("wall_s") or 0)
                cs.completion_tokens += int(row.get("completion_tokens") or 0)
                if row.get("error"):
                    cs.errors += 1
                if any(row.get("per_turn_steps") or []):
                    cs.n_with_calls += 1
                test_entry = mt_entry_cache[cat].get(row.get("id", ""))
                passed, reason = _grade_one_multi_turn(cat, row, test_entry, model_id)
                if passed:
                    cs.passed += 1
                if write_back:
                    row["passed"] = passed
                    row["reason"] = reason
                    path.write_text(json.dumps(row))
            out[cat] = cs
            continue

        if cat not in gt_cache:
            try:
                gt_cache[cat] = load_ground_truth(cat)
            except FileNotFoundError:
                gt_cache[cat] = {}
        for path in sorted(cat_dir.glob("*.json")):
            try:
                row = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            cs.n += 1
            cs.wall_s += float(row.get("wall_s") or 0)
            cs.completion_tokens += int(row.get("completion_tokens") or 0)
            if row.get("error"):
                cs.errors += 1
            if row.get("actual_calls"):
                cs.n_with_calls += 1
            gt_for_pid = gt_cache[cat].get(row.get("id", "")) if cat != "irrelevance" else None
            passed, reason = _grade_one_problem(cat, row, gt_for_pid)
            if passed:
                cs.passed += 1
            if write_back:
                row["passed"] = passed
                row["reason"] = reason
                path.write_text(json.dumps(row))
        out[cat] = cs
    return out


def render(per_model: dict[str, dict[str, CatStats]]) -> str:
    cats = list(GRADED_CATEGORIES)
    short = {"simple_python": "simple", "multiple": "multi",
             "parallel": "par", "parallel_multiple": "par_mul",
             "irrelevance": "irrel",
             "live_simple": "L.simple", "live_multiple": "L.multi",
             "live_parallel": "L.par", "live_parallel_multiple": "L.par_mul",
             "live_irrelevance": "L.irrel", "live_relevance": "L.rel",
             "multi_turn_base": "MT.base",
             "multi_turn_long_context": "MT.lc",
             "multi_turn_miss_func": "MT.miss_fn",
             "multi_turn_miss_param": "MT.miss_pa"}
    lines: list[str] = []
    lines.append("# BFCL graded leaderboard")
    lines.append("")
    lines.append("Pass criteria: simple/multiple = exactly one matching call; "
                 "parallel/parallel_multiple = full set match (order-insensitive); "
                 "irrelevance = zero calls.")
    lines.append("")
    header = "| model | overall | " + " | ".join(short[c] for c in cats) + " | wall (s) | comp tok |"
    sep = "|---" * (2 + len(cats) + 2) + "|"
    lines.append(header)
    lines.append(sep)

    rows = []
    for model, stats in per_model.items():
        total_n = sum(s.n for s in stats.values())
        total_pass = sum(s.passed for s in stats.values())
        total_wall = sum(s.wall_s for s in stats.values())
        total_comp = sum(s.completion_tokens for s in stats.values())
        if not total_n:
            continue
        cat_cells = []
        for c in cats:
            cs = stats.get(c)
            if cs is None or cs.n == 0:
                cat_cells.append("-")
            else:
                cat_cells.append(f"{cs.passed}/{cs.n}")
        rows.append((
            total_pass / total_n,
            f"| {model} | **{total_pass}/{total_n} ({total_pass/total_n:.0%})** | "
            + " | ".join(cat_cells) + f" | {total_wall:.0f} | {total_comp} |"
        ))
    rows.sort(key=lambda r: -r[0])
    lines.extend(r for _, r in rows)
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--acceptance-dir", type=Path, default=ROOT / "acceptance")
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--models", nargs="+", default=None,
                   help="Limit to these model ids (default: all under acceptance/bfcl).")
    p.add_argument("--write-back", action="store_true",
                   help="Write 'passed'/'reason' back into each per-problem JSON.")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of the markdown table.")
    args = p.parse_args()

    bfcl_root = args.acceptance_dir / "bfcl"
    if not bfcl_root.is_dir():
        print(f"No BFCL output found at {bfcl_root}", file=sys.stderr)
        return 2

    model_dirs = sorted(p for p in bfcl_root.iterdir() if p.is_dir())
    if args.models:
        wanted = set(args.models)
        model_dirs = [d for d in model_dirs if d.name in wanted]

    per_model: dict[str, dict[str, CatStats]] = {}
    for d in model_dirs:
        stats = grade_model(d, args.rep, args.write_back)
        if stats:
            per_model[d.name] = stats

    if args.json:
        payload = {
            m: {c: cs.__dict__ for c, cs in stats.items()}
            for m, stats in per_model.items()
        }
        print(json.dumps(payload, indent=2))
    else:
        print(render(per_model))
    return 0


if __name__ == "__main__":
    sys.exit(main())
