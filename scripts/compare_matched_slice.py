"""Cross-model BFCL comparison on a methodologically controlled problem-ID slice.

The bake-off harness writes per-problem `<problem_id>.json` files
under `acceptance/bfcl/<model>/rep_<n>/<category>/`. When two models'
rep dirs contain different sets of problem IDs in the same category
(common when one model ran the full live cat and another ran with
`--bfcl-limit 100`), any cross-model pass-rate comparison must either:

  1. restrict to the intersection of problem IDs (apples-to-apples), or
  2. acknowledge the asymmetry and report it as a distribution claim
     rather than a head-to-head one.

This script implements (1). It is the *only* sanctioned path for
publishing a cross-model BFCL delta in the reports — see
`memory/feedback_slicing_methodology.md`.

Why this script exists: a previous round of Phase H + Round 3 analyses
used `sorted(cat_dir.glob('*.json'))[:100]` to compute a "first-100"
slice. Python lexicographic order returns `2-2-0` after `10-3-6`, so
the resulting "first 100" was a chaotic subset that overlapped the
companion model's "first 100" by only 7-11/100 problems. The
correction is to use *set intersection* of stems, not sliced sorts.

Usage (cross-model on same rep):
    uv run python scripts/compare_matched_slice.py \\
        --rep 1 \\
        --models smollm3-3b-instruct qwen25-1.5b-instruct \\
        --cats live_simple live_multiple live_irrelevance \\
        --write-ids acceptance/audits/phase_h_n1106_matched_ids.json

Usage (cross-rep, same or different models — for Round 3 v3a/v3c
ablations and rep_1 vs rep_7 distribution drops):
    uv run python scripts/compare_matched_slice.py \\
        --targets qwen25-1.5b-instruct:1 qwen25-1.5b-instruct:6 \\
        --cats irrelevance live_simple live_irrelevance \\
        --write-ids acceptance/audits/branch_a_matched_ids.json

Output: per-category overlap counts, per-target passed/n on the matched
slice with Wilson 95% CIs, and (optionally) a JSON file documenting
the exact matched-ID set used for each category.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Target:
    """A (model, rep) pair to compare against others.

    Model dirs and reps are independent variables; using a Target lets
    the same helper drive cross-model-same-rep comparisons (Phase H
    BFCL) and cross-rep-same-model comparisons (Round 3 v3a/v3c
    ablations, rep_1 vs rep_7 distribution-robustness checks).
    """

    model: str
    rep: int

    def label(self) -> str:
        return f"{self.model}:rep_{self.rep}"

    def rep_dir(self, bfcl_root: Path) -> Path:
        return bfcl_root / self.model / f"rep_{self.rep}"


@dataclass
class ModelCatStats:
    model: str  # for backward-compat with existing render(); holds target.label()
    passed: int
    n: int

    @property
    def rate(self) -> float:
        return self.passed / self.n if self.n else 0.0

    def wilson(self, z: float = 1.96) -> tuple[float, float]:
        """Wilson 95% score interval, returned as (lo, hi) in percent."""
        if self.n == 0:
            return 0.0, 0.0
        phat = self.rate
        denom = 1 + z * z / self.n
        center = (phat + z * z / (2 * self.n)) / denom
        half = (
            z
            * math.sqrt(phat * (1 - phat) / self.n + z * z / (4 * self.n * self.n))
            / denom
        )
        lo = max(0.0, center - half) * 100
        hi = min(1.0, center + half) * 100
        return lo, hi


@dataclass
class MatchedSliceResult:
    category: str
    policy: Literal["intersection", "union"]
    matched_ids: list[str]
    per_model_totals: dict[str, int]
    per_model: list[ModelCatStats]

    @property
    def overlap_n(self) -> int:
        return len(self.matched_ids)


def _per_problem_passed(rep_dir: Path, category: str) -> dict[str, bool]:
    """Return {problem_id_stem: passed} for one model's category dir.

    Reads the `passed` field from each per-problem JSON. Rows without a
    `passed` field (ungraded) are skipped, surfacing as missing IDs in
    the matched-slice intersection.
    """
    cat_dir = rep_dir / category
    if not cat_dir.is_dir():
        return {}
    out: dict[str, bool] = {}
    for path in cat_dir.glob("*.json"):
        try:
            rec = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "passed" not in rec:
            continue
        out[path.stem] = bool(rec["passed"])
    return out


def matched_slice(
    targets: list[Target],
    bfcl_root: Path,
    category: str,
    *,
    policy: Literal["intersection", "union"] = "intersection",
) -> MatchedSliceResult:
    """Compute a per-target passed/n on a matched problem-ID slice.

    `policy='intersection'` (default): only IDs present in every target.
    This is the apples-to-apples slice for head-to-head claims.

    `policy='union'`: all IDs across any target. Used for missing-data
    diagnostics ("which targets are missing which IDs?"); rows for IDs
    not present in a given target are excluded from that target's count
    rather than imputed.
    """
    per_target_pass: dict[str, dict[str, bool]] = {}
    for t in targets:
        per_target_pass[t.label()] = _per_problem_passed(t.rep_dir(bfcl_root), category)

    id_sets = [set(p.keys()) for p in per_target_pass.values()]
    if not id_sets:
        return MatchedSliceResult(category, policy, [], {}, [])

    if policy == "intersection":
        matched = set.intersection(*id_sets)
    elif policy == "union":
        matched = set.union(*id_sets)
    else:
        raise ValueError(f"unknown policy: {policy}")

    matched_sorted = sorted(matched)
    per_target_stats: list[ModelCatStats] = []
    for t in targets:
        pmap = per_target_pass[t.label()]
        n = 0
        p = 0
        for pid in matched_sorted:
            if pid not in pmap:
                continue
            n += 1
            if pmap[pid]:
                p += 1
        per_target_stats.append(ModelCatStats(model=t.label(), passed=p, n=n))

    per_target_totals = {t.label(): len(per_target_pass[t.label()]) for t in targets}
    return MatchedSliceResult(
        category=category,
        policy=policy,
        matched_ids=matched_sorted,
        per_model_totals=per_target_totals,
        per_model=per_target_stats,
    )


def render(res: MatchedSliceResult) -> str:
    lines: list[str] = []
    lines.append(f"=== {res.category} ===")
    totals_str = ", ".join(
        f"{m}={n}" for m, n in res.per_model_totals.items()
    )
    lines.append(
        f"  policy={res.policy}  overlap_n={res.overlap_n}  per-model totals: {totals_str}"
    )
    for s in res.per_model:
        lo, hi = s.wilson()
        pct = s.rate * 100
        lines.append(
            f"  - {s.model}: {s.passed}/{s.n} ({pct:.1f}%)  [{lo:.1f}, {hi:.1f}]"
        )
    return "\n".join(lines)


def aggregate_overall(
    results: list[MatchedSliceResult],
) -> list[ModelCatStats]:
    """Sum passed/n across categories per model. Use only when the
    matched-slice policy is identical across all categories (typical:
    'intersection')."""
    by_model: dict[str, tuple[int, int]] = {}
    for r in results:
        for s in r.per_model:
            p, n = by_model.get(s.model, (0, 0))
            by_model[s.model] = (p + s.passed, n + s.n)
    return [ModelCatStats(model=m, passed=p, n=n) for m, (p, n) in by_model.items()]


def _parse_target(s: str) -> Target:
    """Parse a CLI target spec `<model>:<rep>`."""
    if ":" not in s:
        raise argparse.ArgumentTypeError(
            f"target {s!r} must be '<model>:<rep>' (e.g. 'smollm3-3b-instruct:1')"
        )
    model, rep_str = s.rsplit(":", 1)
    try:
        rep = int(rep_str)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"target {s!r}: bad rep number") from e
    return Target(model=model, rep=rep)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "BFCL comparison on a matched problem-ID slice across "
            "(model, rep) targets. Use this script for any cross-model "
            "or cross-rep BFCL delta — never ad-hoc sorted(glob)[:N] slicing."
        )
    )
    ap.add_argument("--acceptance-dir", type=Path, default=ROOT / "acceptance")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--targets",
        type=_parse_target,
        nargs="+",
        help=(
            "Explicit (model, rep) targets as '<model>:<rep>' pairs. "
            "Use for cross-rep comparisons (Round 3 v3a vs rep_1) or "
            "mixed model-rep comparisons. Mutually exclusive with --rep+--models."
        ),
    )
    grp.add_argument(
        "--rep",
        type=int,
        help="Single rep number; pair with --models for cross-model comparison.",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        help="Models to compare on the same --rep (broadcast shortcut).",
    )
    ap.add_argument("--cats", nargs="+", required=True)
    ap.add_argument(
        "--policy",
        choices=("intersection", "union"),
        default="intersection",
    )
    ap.add_argument(
        "--write-ids",
        type=Path,
        default=None,
        help=(
            "Persist the matched-ID set per category to this JSON path. "
            "Recommended: acceptance/audits/<section>_matched_ids.json. "
            "Future re-runs can diff against this for reproducibility."
        ),
    )
    args = ap.parse_args()

    if args.targets is not None:
        targets = list(args.targets)
    else:
        if not args.models:
            ap.error("--rep requires --models")
        targets = [Target(model=m, rep=args.rep) for m in args.models]

    bfcl_root = args.acceptance_dir / "bfcl"
    for t in targets:
        if not t.rep_dir(bfcl_root).is_dir():
            print(f"missing: {t.rep_dir(bfcl_root)}", file=sys.stderr)
            return 2

    results: list[MatchedSliceResult] = []
    for cat in args.cats:
        res = matched_slice(targets, bfcl_root, cat, policy=args.policy)
        results.append(res)
        print(render(res))
        print()

    # Aggregate (only meaningful for intersection policy where every
    # target has been counted on the same set of IDs)
    if args.policy == "intersection":
        overall = aggregate_overall(results)
        if overall:
            print("=== TOTAL (sum across categories) ===")
            for s in overall:
                lo, hi = s.wilson()
                pct = s.rate * 100
                print(
                    f"  - {s.model}: {s.passed}/{s.n} ({pct:.1f}%)  [{lo:.1f}, {hi:.1f}]"
                )

    if args.write_ids is not None:
        payload = {
            "targets": [t.label() for t in targets],
            "policy": args.policy,
            "categories": {
                r.category: {
                    "matched_ids": r.matched_ids,
                    "overlap_n": r.overlap_n,
                    "per_target_totals": r.per_model_totals,
                }
                for r in results
            },
        }
        args.write_ids.parent.mkdir(parents=True, exist_ok=True)
        args.write_ids.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"\nWrote matched-ID set to {args.write_ids}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
