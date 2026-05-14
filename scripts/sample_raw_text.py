"""Reproducible sampling + mechanism taxonomy for BFCL `raw_text`.

Phase J motivation: Phase H asserted "smollm3 emits Python code blocks
in agent mode" without on-disk evidence. After raw_text persistence
landed (2026-05-14), we can inspect what the model actually produces.

This script:
1. Walks `acceptance/bfcl/<model>/rep_<n>/<category>/*.json` for the
   given model+rep.
2. Filters to rows with non-empty `raw_text`.
3. Samples N filenames with a fixed `--seed` (deterministic Random).
4. Classifies each `raw_text` into one of six buckets (taxonomy below).
5. Persists the classified samples + aggregate counts to a JSON file
   (default: `acceptance/audits/phase_j_<model>_mechanism_samples.json`).

Taxonomy (in evaluation order; first matching wins):

- `empty`         blank or whitespace-only text
- `code_block`    fenced ```python ... ``` or ``` ... ``` blocks, or
                  bare `def \\w+\\(...\\)` declarations
- `pseudo_tool`   non-Python function-call syntax outside a code block
                  (`function_name(arg=value)` / `area(10, 5)`)
- `malformed_json` attempted JSON tool-call shape but invalid
- `partial_tool`  recognized tool-call shape but truncated mid-emission
                  (open brace, no close; or open code fence, no close)
- `prose_only`    natural language only, no call shapes detected

Usage:
    uv run python scripts/sample_raw_text.py \\
        --rep 4 --model smollm3-3b-instruct \\
        --seed 1337 --n 20 \\
        --write acceptance/audits/phase_j_smollm3_mechanism_samples.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# ---------- taxonomy ----------

_CODE_FENCE_OPEN = re.compile(r"```(?:python|json)?")
_CODE_FENCE_CLOSE = re.compile(r"```\s*$", re.MULTILINE)
_DEF_LINE = re.compile(r"\bdef\s+\w+\s*\(.*?\)\s*:", re.DOTALL)
# Tool-call JSON hints — used to decide we're looking at a structured
# emission attempt before trying to parse.
_JSON_TOOL_HINT = re.compile(r'"name"\s*:|"arguments"\s*:|"parameters"\s*:')
# Pseudo-call: `identifier(...)` anywhere in the text. Used after the
# JSON check, so we only land here for non-JSON call shapes like
# `area(base=10, height=5)` emitted as prose.
_PSEUDO_CALL = re.compile(r"\b([A-Za-z_]\w*)\s*\(\s*\w+\s*[=:]")


BUCKETS = (
    "empty",
    "code_block",
    "pseudo_tool",
    "malformed_json",
    "partial_tool",
    "prose_only",
)


def _extract_top_level_json_object(text: str) -> str | None:
    """Find the first balanced `{...}` region by brace-counting.

    re.findall with a non-greedy `\\{...?\\}` returns the innermost {}
    pair, which strips the outer object when nested. Brace-counting
    returns the outermost — which is what we need to detect a tool-call
    JSON shape that wraps `{"name": ..., "arguments": {...}}`.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None  # unbalanced (open without close)


def classify(text: str) -> str:
    """Return one of BUCKETS describing the dominant shape of `text`."""
    if not text or not text.strip():
        return "empty"

    has_open_fence = bool(_CODE_FENCE_OPEN.search(text))
    has_close_fence = bool(_CODE_FENCE_CLOSE.search(text))
    has_def = bool(_DEF_LINE.search(text))

    # Open fence with no close → partial; balanced → code_block; bare def → code_block
    if has_open_fence and not has_close_fence:
        return "partial_tool"
    if has_open_fence or has_def:
        return "code_block"

    # JSON-tool-call attempt: must look like a tool call (has name +
    # arguments/parameters hint somewhere) AND we can isolate the
    # outermost JSON object via brace-counting.
    if _JSON_TOOL_HINT.search(text):
        obj_str = _extract_top_level_json_object(text)
        if obj_str is None:
            # Open brace without close — malformed mid-emission, but
            # could also be a partial tool. Treat as malformed since
            # the trailing portion is missing, not the leading.
            return "partial_tool"
        try:
            parsed = json.loads(obj_str)
        except json.JSONDecodeError:
            return "malformed_json"
        if isinstance(parsed, dict) and (
            "name" in parsed and ("arguments" in parsed or "parameters" in parsed)
        ):
            return "pseudo_tool"
        # Has tool-call shape hints but no top-level name+args — likely
        # malformed even though it parses (e.g. nested in something else)
        return "malformed_json"

    # `identifier(arg=val)` or `identifier(arg: val)` outside JSON,
    # outside a code block — Gemma 2 / pseudo-Python call shape.
    if _PSEUDO_CALL.search(text):
        return "pseudo_tool"

    return "prose_only"


# ---------- sampling ----------


@dataclass
class Sample:
    path: str
    bucket: str
    text_excerpt: str  # first 400 chars
    text_length: int


@dataclass
class TaxonomyReport:
    model: str
    rep: int
    seed: int
    n_sampled: int
    n_with_raw_text: int
    n_total_rows: int
    bucket_counts: dict[str, int] = field(default_factory=dict)
    samples: list[Sample] = field(default_factory=list)


def collect_rows(
    model: str,
    rep: int,
    bfcl_root: Path,
    cats: list[str] | None = None,
) -> tuple[list[tuple[Path, dict]], int, int]:
    """Walk per-problem JSONs; return [(path, row)] where raw_text is
    non-empty, plus (n_with_raw_text, n_total)."""
    rep_dir = bfcl_root / model / f"rep_{rep}"
    if not rep_dir.is_dir():
        raise SystemExit(f"missing: {rep_dir}")

    cat_dirs = (
        [rep_dir / c for c in cats] if cats else
        [p for p in rep_dir.iterdir() if p.is_dir()]
    )
    rows: list[tuple[Path, dict]] = []
    n_with = 0
    n_total = 0
    for cd in cat_dirs:
        if not cd.is_dir():
            continue
        for p in cd.glob("*.json"):
            try:
                rec = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            n_total += 1
            rt = rec.get("raw_text")
            if rt:
                n_with += 1
                rows.append((p, rec))
    return rows, n_with, n_total


def sample_and_classify(
    model: str,
    rep: int,
    bfcl_root: Path,
    *,
    seed: int,
    n: int,
    cats: list[str] | None = None,
) -> TaxonomyReport:
    rows, n_with, n_total = collect_rows(model, rep, bfcl_root, cats)
    rng = random.Random(seed)
    sampled = rng.sample(rows, k=min(n, len(rows))) if rows else []

    samples: list[Sample] = []
    counts = {b: 0 for b in BUCKETS}
    for path, rec in sampled:
        rt = rec.get("raw_text", "")
        bucket = classify(rt)
        counts[bucket] += 1
        samples.append(Sample(
            path=str(path.relative_to(bfcl_root.parent)),
            bucket=bucket,
            text_excerpt=rt[:400],
            text_length=len(rt),
        ))

    # Also classify every row (not just the sample) for the aggregate
    # distribution — cheap and gives a fuller picture.
    aggregate_counts = {b: 0 for b in BUCKETS}
    for path, rec in rows:
        aggregate_counts[classify(rec.get("raw_text", ""))] += 1

    return TaxonomyReport(
        model=model,
        rep=rep,
        seed=seed,
        n_sampled=len(samples),
        n_with_raw_text=n_with,
        n_total_rows=n_total,
        # Sample-only bucket counts in `bucket_counts`; aggregate in
        # `bucket_counts_full` (serialized below). Keeps the sample
        # statistics + full-population stats both visible.
        bucket_counts={**counts, "_aggregate": dict(aggregate_counts)},
        samples=samples,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acceptance-dir", type=Path, default=ROOT / "acceptance")
    ap.add_argument("--model", required=True)
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--cats", nargs="+", default=None,
                    help="restrict to these categories (default: all)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--write", type=Path, default=None,
                    help="persist the report JSON to this path "
                         "(default: print to stdout)")
    args = ap.parse_args()

    report = sample_and_classify(
        args.model,
        args.rep,
        args.acceptance_dir / "bfcl",
        seed=args.seed,
        n=args.n,
        cats=args.cats,
    )

    print(f"=== {args.model} rep_{args.rep} mechanism taxonomy ===")
    print(f"  total rows: {report.n_total_rows}")
    print(f"  rows with non-empty raw_text: {report.n_with_raw_text}")
    print(f"  sampled (seed={args.seed}, n={args.n}): {report.n_sampled}")
    print()
    print(f"  sample bucket counts ({args.n}):")
    for b in BUCKETS:
        c = report.bucket_counts.get(b, 0)
        print(f"    {b:18s} {c}")
    print()
    print(f"  full-population bucket counts ({report.n_with_raw_text}):")
    for b in BUCKETS:
        c = report.bucket_counts["_aggregate"].get(b, 0)
        print(f"    {b:18s} {c}")

    if args.write:
        payload = {
            "model": report.model,
            "rep": report.rep,
            "seed": report.seed,
            "n_total_rows": report.n_total_rows,
            "n_with_raw_text": report.n_with_raw_text,
            "n_sampled": report.n_sampled,
            "bucket_counts_sample": {
                b: report.bucket_counts.get(b, 0) for b in BUCKETS
            },
            "bucket_counts_full_population": dict(
                report.bucket_counts["_aggregate"]
            ),
            "samples": [
                {
                    "path": s.path,
                    "bucket": s.bucket,
                    "text_length": s.text_length,
                    "text_excerpt": s.text_excerpt,
                }
                for s in report.samples
            ],
        }
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.write}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
