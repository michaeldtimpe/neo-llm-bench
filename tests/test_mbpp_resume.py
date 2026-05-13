"""Per-problem resume + adapter tests for the MBPP runner.

Mirrors the structure of `tests/test_humaneval_resume.py` — we mock the
adapter so tests don't depend on a live llama-server, and assert that:
  * cold runs cover every task_id
  * resume keeps existing rows and only re-runs missing ones
  * a torn last line is dropped and re-inferred
  * the completion normalizer survives real-world model noise

Also covers `_entry_point_from_test` (function-name extraction from the
first assert) and `normalize_completion` (markdown-fence + prose-trim +
main-guard drop) since both are MBPP-specific and brittle.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import benchmarks.mbpp.adapter as mbpp_adapter
from llamabench.config import ModelConfig
from llamabench.runner import RunRequest, _run_mbpp


def _problem(task_id: int, fn_name: str = "f"):
    """Minimal MBPP-shaped problem record."""
    return {
        "task_id": task_id,
        "prompt": f"Write a function {fn_name} that returns 0.",
        "test_list": [f"assert {fn_name}() == 0"],
        "test_imports": [],
        "code": f"def {fn_name}():\n    return 0",
    }


def _make_result(task_id: int, *, passed: bool = True, tokens: int = 100, wall: float = 1.0):
    return mbpp_adapter.MbppResult(
        task_id=f"Mbpp/{task_id}",
        passed=passed,
        wall_s=wall,
        extract_ok=True,
        completion_tokens=tokens,
        prompt_tokens=50,
        error="",
        raw_text="",
        completion="def f():\n    return 0",
    )


@pytest.fixture
def model_cfg() -> ModelConfig:
    return ModelConfig(id="fake-model", gguf_path="/tmp/does-not-exist.gguf")


@pytest.fixture
def run_request(tmp_path: Path, model_cfg: ModelConfig) -> RunRequest:
    return RunRequest(
        model=model_cfg,
        benchmarks=["mbpp"],
        output_dir=tmp_path,
        rep=0,
        limit=5,
    )


@pytest.fixture
def patched_mbpp(monkeypatch):
    """Monkeypatch load_problems + run_problem. Tracks every task_id inferred."""
    inferred: list[str] = []
    problems = [_problem(i) for i in range(5)]

    def fake_load_problems(limit=None):
        return problems if limit is None else problems[:limit]

    def fake_run_problem(backend, problem, *, max_tokens, temperature, test_timeout_s=10.0):
        tid = problem["task_id"]
        inferred.append(f"Mbpp/{tid}")
        # Even task_ids pass; odd fail. Token sentinel = 200 + idx.
        return _make_result(tid, passed=(tid % 2 == 0), tokens=200 + tid)

    monkeypatch.setattr(mbpp_adapter, "load_problems", fake_load_problems)
    monkeypatch.setattr(mbpp_adapter, "run_problem", fake_run_problem)
    return inferred


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# --- resume semantics ------------------------------------------------------


def test_mbpp_cold_start_runs_every_problem(run_request, patched_mbpp):
    summary = _run_mbpp(backend=None, req=run_request)
    assert patched_mbpp == [f"Mbpp/{i}" for i in range(5)]
    assert summary["n_problems"] == 5
    assert summary["n_passed"] == 3  # 0, 2, 4 even pass


def test_mbpp_resume_skips_existing_rows(run_request, patched_mbpp):
    out_dir = run_request.output_dir / "mbpp" / "fake-model" / "rep_0"
    out_dir.mkdir(parents=True)
    pre_existing = [
        {"task_id": "Mbpp/0", "passed": True, "extract_ok": True,
         "wall_s": 5.0, "prompt_tokens": 50, "completion_tokens": 999, "error": ""},
        {"task_id": "Mbpp/1", "passed": False, "extract_ok": True,
         "wall_s": 7.0, "prompt_tokens": 50, "completion_tokens": 888, "error": "x"},
    ]
    (out_dir / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pre_existing))

    _run_mbpp(backend=None, req=run_request)

    # Only the missing task_ids ran.
    assert patched_mbpp == [f"Mbpp/{i}" for i in (2, 3, 4)]
    rows = _read_rows(out_dir / "results.jsonl")
    assert [r["task_id"] for r in rows] == [f"Mbpp/{i}" for i in range(5)]
    # Pre-existing token sentinels survive (not overwritten by the fake runner).
    assert rows[0]["completion_tokens"] == 999
    assert rows[1]["completion_tokens"] == 888


def test_mbpp_torn_last_line_dropped(run_request, patched_mbpp):
    out_dir = run_request.output_dir / "mbpp" / "fake-model" / "rep_0"
    out_dir.mkdir(parents=True)
    good = {"task_id": "Mbpp/0", "passed": True, "extract_ok": True,
            "wall_s": 1.0, "prompt_tokens": 10, "completion_tokens": 999, "error": ""}
    torn = '{"task_id": "Mbpp/1", "passed": tru'
    (out_dir / "results.jsonl").write_text(json.dumps(good) + "\n" + torn)

    _run_mbpp(backend=None, req=run_request)

    assert patched_mbpp == [f"Mbpp/{i}" for i in (1, 2, 3, 4)]
    rows = _read_rows(out_dir / "results.jsonl")
    assert [r["task_id"] for r in rows] == [f"Mbpp/{i}" for i in range(5)]
    assert rows[0]["completion_tokens"] == 999


# --- adapter primitives ----------------------------------------------------


def test_entry_point_from_test_skips_builtin_wrappers():
    # `set(similar_elements(...))` should find `similar_elements`, not `set`.
    line = "assert set(similar_elements((3,4),(5,4))) == set((4,))"
    assert mbpp_adapter._entry_point_from_test(line) == "similar_elements"


def test_entry_point_from_test_handles_nested_builtins():
    line = "assert len(list(my_func(7))) == 3"
    assert mbpp_adapter._entry_point_from_test(line) == "my_func"


def test_entry_point_from_test_returns_none_on_syntax_error():
    assert mbpp_adapter._entry_point_from_test("not valid python @@") is None


def test_entry_point_from_test_returns_none_when_no_user_function():
    # Pure builtin calls only — no user function to find.
    assert mbpp_adapter._entry_point_from_test("assert set() == set()") is None


def test_normalize_completion_strips_fence():
    text = "Here is the function:\n\n```python\ndef foo():\n    return 1\n```\n\nThis works."
    assert mbpp_adapter.normalize_completion(text, "foo") == "def foo():\n    return 1"


def test_normalize_completion_anchors_to_first_def():
    text = "Sure, here goes.\n\ndef bar(x):\n    return x + 1\n\nDone."
    out = mbpp_adapter.normalize_completion(text, "bar")
    assert out.startswith("def bar(x):")
    # Trailing "Done." prose isn't recognized as code — the anchor catches
    # the def, then rstrip handles outer whitespace.
    assert "def bar(x):" in out


def test_normalize_completion_drops_main_guard():
    text = (
        "```python\n"
        "def f():\n    return 1\n\n"
        "if __name__ == \"__main__\":\n"
        "    print(f())\n"
        "```"
    )
    out = mbpp_adapter.normalize_completion(text, "f")
    assert "__main__" not in out
    assert "def f():" in out


def test_normalize_completion_prefers_block_containing_entry_point():
    """If the model emits a usage-example block AND a definition block,
    the one mentioning entry_point should win."""
    text = (
        "Example usage:\n```python\nprint('hi')\n```\n"
        "Definition:\n```python\ndef target():\n    return 42\n```\n"
    )
    out = mbpp_adapter.normalize_completion(text, "target")
    assert "def target():" in out
    assert "print('hi')" not in out


def test_normalize_completion_empty_input():
    assert mbpp_adapter.normalize_completion("", "x") == ""


# --- registered spec smoke ------------------------------------------------


def test_mbpp_is_registered_with_per_problem_resume():
    from llamabench.runner import _BENCH_RUNNERS
    spec = _BENCH_RUNNERS["mbpp"]
    assert spec.supports_per_problem_resume is True
    assert "results.jsonl" in spec.force_clean_filenames
    assert "summary.json" in spec.force_clean_filenames
