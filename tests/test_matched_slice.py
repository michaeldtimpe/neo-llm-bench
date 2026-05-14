"""Tests for scripts/compare_matched_slice.py + scripts/_problem_ids.py.

Regression coverage for the Phase H slicing bug: lexicographic sort
returns `live_simple_10` before `live_simple_2`, which broke
"first-100 apples-to-apples" comparisons in graded_report.md when one
side had more files than the other (overlap was 7-11/100, not 100/100).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# scripts/ is not a package; add to path explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from _problem_ids import natural_problem_key, sorted_problem_files  # noqa: E402
from compare_matched_slice import (  # noqa: E402
    Target,
    matched_slice,
)


# ---------- natural_problem_key ----------


def test_natural_key_orders_single_int_correctly():
    """Multi-digit IDs sort numerically, not lexicographically."""
    stems = ["a_2", "a_10", "a_100", "a_9", "a_1"]
    out = sorted(stems, key=natural_problem_key)
    assert out == ["a_1", "a_2", "a_9", "a_10", "a_100"]


def test_natural_key_orders_triple_correctly():
    """BFCL live-cat IDs use `<a>-<b>-<c>` shape."""
    stems = [
        "live_simple_10-3-6",
        "live_simple_2-2-0",
        "live_simple_100-58-0",
        "live_simple_1-1-0",
    ]
    out = sorted(stems, key=natural_problem_key)
    assert out == [
        "live_simple_1-1-0",
        "live_simple_2-2-0",
        "live_simple_10-3-6",
        "live_simple_100-58-0",
    ]


def test_natural_key_stable_across_first_n_slice():
    """Regression for the original bug: first-100 by natural key picks
    IDs 0..99, not the lex-mangled set."""
    stems = [f"live_simple_{i}-0-0" for i in range(150)]
    # Shuffle to defeat any incidental ordering.
    import random

    random.Random(0).shuffle(stems)
    first_100 = sorted(stems, key=natural_problem_key)[:100]
    ids = {int(s.split("_")[2].split("-")[0]) for s in first_100}
    assert ids == set(range(100)), "natural-sort first-100 must be IDs 0..99"


def test_natural_key_stems_with_no_digits():
    """Fallback for malformed stems — sort to a stable position, not crash."""
    assert natural_problem_key("garbage") == (0,)


def test_sorted_problem_files_helper(tmp_path: Path):
    files = []
    for stem in ["live_simple_10-0-0", "live_simple_2-0-0", "live_simple_1-0-0"]:
        f = tmp_path / f"{stem}.json"
        f.write_text("{}")
        files.append(f)
    out = sorted_problem_files(files)
    assert [p.stem for p in out] == [
        "live_simple_1-0-0",
        "live_simple_2-0-0",
        "live_simple_10-0-0",
    ]


# ---------- matched_slice ----------


def _make_rep(
    root: Path,
    model: str,
    rep: int,
    category: str,
    rows: dict[str, bool],
) -> None:
    """Write a fake rep dir with per-problem JSONs containing `passed`."""
    d = root / "bfcl" / model / f"rep_{rep}" / category
    d.mkdir(parents=True)
    for pid, passed in rows.items():
        (d / f"{pid}.json").write_text(json.dumps({"id": pid, "passed": passed}))


def test_intersection_restricts_to_overlapping_ids(tmp_path: Path):
    """Cross-model: model A has IDs 0-149, model B has IDs 0-99.
    Intersection must be IDs 0-99 (the smaller set)."""
    # A: 150 problems, all pass for IDs <100, all fail for IDs ≥100
    a_rows = {f"live_simple_{i}-0-0": (i < 100) for i in range(150)}
    b_rows = {f"live_simple_{i}-0-0": True for i in range(100)}
    _make_rep(tmp_path, "model_a", 1, "live_simple", a_rows)
    _make_rep(tmp_path, "model_b", 1, "live_simple", b_rows)
    res = matched_slice(
        [Target("model_a", 1), Target("model_b", 1)],
        tmp_path / "bfcl",
        "live_simple",
    )
    # The 50 problems with IDs ≥100 are A-only and must be excluded.
    assert res.overlap_n == 100
    by_label = {s.model: s for s in res.per_model}
    assert by_label["model_a:rep_1"].passed == 100  # all 100 matched pass for A
    assert by_label["model_a:rep_1"].n == 100
    assert by_label["model_b:rep_1"].passed == 100
    assert by_label["model_b:rep_1"].n == 100
    # Per-model totals (full dir size, not intersection) surface for diagnostics
    assert res.per_model_totals == {"model_a:rep_1": 150, "model_b:rep_1": 100}


def test_intersection_does_not_pick_lex_mangled_first_100(tmp_path: Path):
    """The original bug: lex first-100 of [0..149] picks IDs
    {0,1,10,100..149,11,110..119, ...}. Intersection with [0..99] gives
    only 11 problems. Confirm our helper avoids this by using set
    intersection (not slice-by-position)."""
    a_rows = {f"live_simple_{i}-0-0": True for i in range(150)}
    b_rows = {f"live_simple_{i}-0-0": True for i in range(100)}
    _make_rep(tmp_path, "lex_a", 1, "live_simple", a_rows)
    _make_rep(tmp_path, "lex_b", 1, "live_simple", b_rows)
    res = matched_slice(
        [Target("lex_a", 1), Target("lex_b", 1)],
        tmp_path / "bfcl",
        "live_simple",
    )
    # All 100 of B's IDs are in A. Intersection == 100, NOT 11.
    assert res.overlap_n == 100, (
        "intersection collapsed to lex-first-100 overlap; "
        "helper has regressed to the original bug"
    )


def test_cross_rep_comparison(tmp_path: Path):
    """Same model, two reps. Used for Round 3 branch ablations
    (rep_1 vs rep_6 v3a)."""
    _make_rep(
        tmp_path,
        "qwen",
        1,
        "live_simple",
        {f"p_{i}": True for i in range(100)},
    )
    # rep_6 has 150 problems, IDs 0-149, with the first 50 failing.
    _make_rep(
        tmp_path,
        "qwen",
        6,
        "live_simple",
        {f"p_{i}": (i >= 50) for i in range(150)},
    )
    res = matched_slice(
        [Target("qwen", 1), Target("qwen", 6)],
        tmp_path / "bfcl",
        "live_simple",
    )
    assert res.overlap_n == 100  # rep_1 only has 100; intersection bounded
    by_label = {s.model: s for s in res.per_model}
    # rep_1: 100 passes; rep_6 (restricted to IDs 0..99): 50 pass (IDs 50..99).
    assert by_label["qwen:rep_1"].passed == 100
    assert by_label["qwen:rep_6"].passed == 50


def test_ungraded_rows_excluded(tmp_path: Path):
    """Per-problem JSONs without a `passed` field are skipped — they
    surface as missing IDs in the matched intersection."""
    d = tmp_path / "bfcl" / "m" / "rep_1" / "live_simple"
    d.mkdir(parents=True)
    (d / "ok.json").write_text(json.dumps({"id": "ok", "passed": True}))
    (d / "ungraded.json").write_text(json.dumps({"id": "ungraded"}))
    (d / "ok2.json").write_text(json.dumps({"id": "ok2", "passed": False}))
    res = matched_slice(
        [Target("m", 1)],
        tmp_path / "bfcl",
        "live_simple",
    )
    # ungraded row excluded; only ok + ok2 surface.
    assert res.overlap_n == 2
    assert set(res.matched_ids) == {"ok", "ok2"}


def test_union_policy_includes_non_overlapping(tmp_path: Path):
    """Union policy is for missing-data diagnostics — surfaces IDs
    present in only one target."""
    _make_rep(tmp_path, "a", 1, "x", {"p1": True, "p2": True})
    _make_rep(tmp_path, "b", 1, "x", {"p2": False, "p3": True})
    res = matched_slice(
        [Target("a", 1), Target("b", 1)],
        tmp_path / "bfcl",
        "x",
        policy="union",
    )
    assert set(res.matched_ids) == {"p1", "p2", "p3"}
    by_label = {s.model: s for s in res.per_model}
    # Each target only counts IDs it has — no imputation.
    assert by_label["a:rep_1"].n == 2
    assert by_label["b:rep_1"].n == 2


def test_wilson_ci_returns_reasonable_range():
    """Spot-check: 89/100 should produce ~89% with a ~10pp half-width."""
    from compare_matched_slice import ModelCatStats

    s = ModelCatStats(model="x", passed=89, n=100)
    lo, hi = s.wilson()
    assert 80.0 < lo < 84.0
    assert 92.0 < hi < 95.0
    # Sanity: published Phase H number was 89/100 → [81.4, 93.7]
    assert abs(lo - 81.4) < 0.2 and abs(hi - 93.7) < 0.2


def test_missing_rep_dir_returns_empty_result(tmp_path: Path):
    """Gracefully handle missing rep dir — no crash, empty result."""
    res = matched_slice(
        [Target("nonexistent", 1)],
        tmp_path / "bfcl",
        "live_simple",
    )
    assert res.overlap_n == 0


# ---------- CLI smoke ----------


def test_cli_target_parsing():
    """`<model>:<rep>` → Target. Plain string without colon must error."""
    from compare_matched_slice import _parse_target
    import argparse

    t = _parse_target("smollm3-3b-instruct:1")
    assert t.model == "smollm3-3b-instruct"
    assert t.rep == 1
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_target("missing-colon")
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_target("model:not_a_number")
