"""Generate a markdown leaderboard from acceptance/<bench>/<model>/rep_<n>/summary.json.

Usage:
    uv run python scripts/report.py acceptance/ > report.md
    uv run python scripts/report.py acceptance/ --rep 0 > report.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


def _load_summaries(root: Path, rep: int) -> dict[str, dict[str, dict]]:
    """Return {bench: {model: summary_dict}}."""
    out: dict[str, dict[str, dict]] = {}
    for bench_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        bench = bench_dir.name
        out.setdefault(bench, {})
        # Both layouts are supported:
        #   acceptance/<bench>/<model>/rep_<n>/summary.json   (per-bench rooted)
        #   acceptance/<model>/<bench>/rep_<n>/summary.json   (per-model rooted)
        # Our runner uses the first; allow the second for hand-organized data.
        for model_dir in sorted(p for p in bench_dir.iterdir() if p.is_dir()):
            rep_dir = model_dir / f"rep_{rep}"
            sf = rep_dir / "summary.json"
            if sf.is_file():
                out[bench][model_dir.name] = json.loads(sf.read_text())
    return out


def _bfcl_row(s: dict[str, Any]) -> dict[str, Any]:
    cats = s.get("categories", {})
    parts = []
    n_total = 0
    n_with_calls = 0
    n_errors = 0
    wall_total = 0.0
    comp_total = 0
    for cat, cs in cats.items():
        n_total += cs["n_problems"]
        n_with_calls += cs["n_with_calls"]
        n_errors += cs["n_errors"]
        wall_total += cs["wall_s"]
        comp_total += cs.get("completion_tokens", 0)
        # Irrelevance: pass = no call. simple/multiple/parallel*: pass = at least one call.
        # We don't have ground-truth grading here yet — that's the BFCL grade step;
        # this report shows raw "emitted-call rate" per category.
    return {
        "n_problems": n_total,
        "n_with_calls": n_with_calls,
        "n_errors": n_errors,
        "wall_s": wall_total,
        "completion_tokens": comp_total,
        "per_cat": cats,
    }


def render_markdown(summaries: dict[str, dict[str, dict]], rep: int) -> str:
    lines: list[str] = [f"# llama-bench leaderboard (rep {rep})", ""]

    bfcl = summaries.get("bfcl", {})
    he = summaries.get("humaneval", {})
    models = sorted(set(bfcl) | set(he))

    if not models:
        return "# llama-bench leaderboard\n\nNo summaries found.\n"

    lines += ["## Overview", ""]
    lines += [
        "| model | bfcl emit-rate | bfcl irrelevance refusal | humaneval pass@1 | bfcl wall (s) | humaneval wall (s) | comp-tok/problem |",
        "|---|---|---|---|---|---|---|",
    ]
    rows: list[tuple[str, float, str, str]] = []
    median_comp_per_prob: list[float] = []
    for m in models:
        bfcl_s = bfcl.get(m)
        he_s = he.get(m)

        if bfcl_s:
            row = _bfcl_row(bfcl_s)
            n = row["n_problems"]
            emit = (row["n_with_calls"] / max(1, n)) if n else 0.0
            irrel = row["per_cat"].get("irrelevance", {})
            irrel_n = irrel.get("n_problems", 0)
            irrel_refused = irrel_n - irrel.get("n_with_calls", 0)
            irrel_rate = (irrel_refused / max(1, irrel_n)) if irrel_n else 0.0
            bfcl_wall_s = f"{row['wall_s']:.0f}"
            non_irrel_n = n - irrel_n
            comp_per_prob = (row["completion_tokens"] / max(1, n)) if n else 0.0
            median_comp_per_prob.append(comp_per_prob)
        else:
            emit = 0.0
            irrel_rate = 0.0
            bfcl_wall_s = "-"
            comp_per_prob = 0.0

        if he_s:
            pa1 = he_s.get("pass_at_1", he_s.get("n_passed", 0) / max(1, he_s.get("n_problems", 1)))
            he_wall_s = f"{he_s.get('wall_s', 0):.0f}"
        else:
            pa1 = 0.0
            he_wall_s = "-"

        lines.append(
            f"| {m} | {emit:.0%} | {irrel_rate:.0%} | {pa1:.1%} | {bfcl_wall_s} | {he_wall_s} | "
            f"{comp_per_prob:.0f} |"
        )
        rows.append((m, pa1, bfcl_wall_s, he_wall_s))

    # Flag thinking-mode candidates (deluxe lesson #11.1: >2x median completion-tokens-per-problem)
    if median_comp_per_prob:
        med = statistics.median(median_comp_per_prob)
        flagged = [m for m, ctpp in zip(models, median_comp_per_prob) if ctpp > 2 * med and med > 0]
        if flagged:
            lines += ["", f"> ⚠ Thinking-mode warning (>2× median {med:.0f} tok/problem): "
                          + ", ".join(flagged)]

    # BFCL per-category breakdown
    if bfcl:
        lines += ["", "## BFCL per-category emit-rate", ""]
        cats = sorted({c for s in bfcl.values() for c in s.get("categories", {})})
        header = "| model | " + " | ".join(cats) + " |"
        sep = "|---" * (len(cats) + 1) + "|"
        lines += [header, sep]
        for m in sorted(bfcl):
            row_cats = bfcl[m].get("categories", {})
            cells = []
            for c in cats:
                cs = row_cats.get(c)
                if not cs:
                    cells.append("-")
                    continue
                n = cs["n_problems"]
                if c == "irrelevance":
                    refused = n - cs["n_with_calls"]
                    cells.append(f"{refused/max(1,n):.0%} refusal")
                else:
                    cells.append(f"{cs['n_with_calls']}/{n}")
            lines.append(f"| {m} | " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("acceptance_dir", type=Path,
                   help="Root directory containing <bench>/<model>/rep_<n>/summary.json")
    p.add_argument("--rep", type=int, default=0)
    args = p.parse_args()

    if not args.acceptance_dir.is_dir():
        print(f"Not a directory: {args.acceptance_dir}", file=sys.stderr)
        return 1
    summaries = _load_summaries(args.acceptance_dir, args.rep)
    print(render_markdown(summaries, args.rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
