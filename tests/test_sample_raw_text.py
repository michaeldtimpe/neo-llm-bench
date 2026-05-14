"""Tests for scripts/sample_raw_text.py taxonomy + sampling determinism."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from sample_raw_text import (  # noqa: E402
    BUCKETS,
    classify,
    sample_and_classify,
)


# ---------- classify() ----------


def test_empty_is_empty():
    assert classify("") == "empty"
    assert classify("   \n  ") == "empty"


def test_fenced_python_is_code_block():
    txt = "Sure, here's the function:\n\n```python\ndef foo(x):\n    return x\n```\n"
    assert classify(txt) == "code_block"


def test_bare_def_is_code_block():
    txt = "def calculate_triangle_area(base, height):\n    return 0.5 * base * height"
    assert classify(txt) == "code_block"


def test_unclosed_fence_is_partial_tool():
    txt = "Here's how I'd start:\n\n```python\ndef foo(x):\n    return x"
    assert classify(txt) == "partial_tool"


def test_valid_json_tool_call_is_pseudo_tool():
    txt = '{"name": "get_user_info", "arguments": {"user_id": 7890}}'
    assert classify(txt) == "pseudo_tool"


def test_truncated_json_tool_call_is_partial_tool():
    """Unbalanced braces → partial_tool (truncated mid-emission)."""
    txt = '{"name": "get_user_info", "arguments": {"user_id": 7890,'
    assert classify(txt) == "partial_tool"


def test_malformed_json_tool_call_is_malformed_json():
    """Has tool-call hint ('name'/'arguments') AND balanced braces but
    invalid JSON syntax (single quotes inside) → malformed_json."""
    txt = "{\"name\": \"get_user_info\", \"arguments\": {'user_id': 7890}}"
    assert classify(txt) == "malformed_json"


def test_pseudo_call_shape_outside_json():
    """Common shape from Gemma 2 and similar — `func(arg=val)` as text."""
    txt = "I'll call calculate_triangle_area(base=10, height=5) for you."
    assert classify(txt) == "pseudo_tool"


def test_pure_prose_is_prose_only():
    txt = (
        "I'm not sure what tools are available, but I can help you "
        "manually if you describe what you need."
    )
    assert classify(txt) == "prose_only"


def test_buckets_constant_is_complete():
    """If someone adds a bucket, make sure BUCKETS lists it."""
    required = {"empty", "code_block", "pseudo_tool",
                "malformed_json", "partial_tool", "prose_only"}
    assert set(BUCKETS) == required


# ---------- sample_and_classify ----------


def _make_row(stem: str, raw_text: str) -> dict:
    return {
        "id": stem,
        "actual_calls": [],
        "wall_s": 0.1,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "error": "",
        "passed": False,
        "raw_text": raw_text,
    }


def _setup_rep(tmp_path: Path, rows_by_cat: dict[str, list[tuple[str, str]]]):
    bfcl_root = tmp_path / "bfcl"
    rep_dir = bfcl_root / "test-model" / "rep_4"
    for cat, rows in rows_by_cat.items():
        cd = rep_dir / cat
        cd.mkdir(parents=True)
        for stem, rt in rows:
            (cd / f"{stem}.json").write_text(json.dumps(_make_row(stem, rt)))
    return bfcl_root


def test_sampling_is_deterministic_for_seed(tmp_path):
    bfcl_root = _setup_rep(tmp_path, {
        "simple_python": [
            (f"p_{i}", f"prose {i}" if i % 2 else "")
            for i in range(50)
        ],
    })
    r1 = sample_and_classify("test-model", 4, bfcl_root, seed=1337, n=5)
    r2 = sample_and_classify("test-model", 4, bfcl_root, seed=1337, n=5)
    assert [s.path for s in r1.samples] == [s.path for s in r2.samples], (
        "same seed must pick the same samples"
    )


def test_sampling_changes_with_different_seed(tmp_path):
    bfcl_root = _setup_rep(tmp_path, {
        "simple_python": [(f"p_{i}", f"text {i}") for i in range(50)],
    })
    r1 = sample_and_classify("test-model", 4, bfcl_root, seed=1, n=10)
    r2 = sample_and_classify("test-model", 4, bfcl_root, seed=999, n=10)
    assert [s.path for s in r1.samples] != [s.path for s in r2.samples]


def test_sampling_excludes_empty_raw_text(tmp_path):
    """Rows without raw_text or with empty raw_text aren't sampled.
    Only meaningful text contributes to the mechanism taxonomy."""
    bfcl_root = _setup_rep(tmp_path, {
        "simple_python": [
            ("with_text", "some prose"),
            ("empty_text", ""),
        ],
    })
    # Also write a legacy row without raw_text at all
    legacy = bfcl_root / "test-model/rep_4/simple_python/legacy.json"
    legacy.write_text(json.dumps({
        "id": "legacy", "actual_calls": [], "wall_s": 0.1,
        "prompt_tokens": 10, "completion_tokens": 5, "error": "",
    }))
    r = sample_and_classify("test-model", 4, bfcl_root, seed=0, n=5)
    paths = [s.path for s in r.samples]
    assert all("legacy" not in p for p in paths)
    assert all("empty_text" not in p for p in paths)
    assert r.n_with_raw_text == 1
    assert r.n_total_rows == 3


def test_sample_count_capped_at_population(tmp_path):
    """Requesting n=20 when only 3 rows have text returns 3 samples."""
    bfcl_root = _setup_rep(tmp_path, {
        "simple_python": [
            ("a", "alpha"), ("b", "beta"), ("c", "gamma"),
        ],
    })
    r = sample_and_classify("test-model", 4, bfcl_root, seed=0, n=20)
    assert r.n_sampled == 3
    assert len(r.samples) == 3
