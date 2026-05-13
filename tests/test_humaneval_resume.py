"""Per-problem resume tests for the HumanEval runner.

The runner truncated ``results.jsonl`` on every start until 2026-05-11; an
interrupted overnight bake-off lost all its in-flight rows on restart.
``_run_humaneval`` now reads the existing jsonl, drops any torn last line,
and only re-runs task_ids that aren't already recorded.

Also covers the `_clear_stale_for_force` helper in `scripts/run_bakeoff.py`
that ensures `--force` actually starts over — the per-problem resume path
will happily reuse a stale `results.jsonl` if the file isn't cleared first.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import benchmarks.humaneval.adapter as humaneval_adapter
from llamabench.config import ModelConfig
from llamabench.runner import RunRequest, _run_humaneval

# `scripts/` isn't a package; load the helper by path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import run_bakeoff  # noqa: E402  — helper module under test


def _problem(task_id: str) -> dict:
    return {
        "task_id": task_id,
        "prompt": f"def f_{task_id.split('/')[-1]}():\n    ",
        "test": "def check(f): assert f() == 0",
        "entry_point": f"f_{task_id.split('/')[-1]}",
    }


def _make_result(task_id: str, *, passed: bool = True, tokens: int = 100, wall: float = 1.0):
    """Build a HumanEvalResult-shaped object compatible with what _run_humaneval reads."""
    return humaneval_adapter.HumanEvalResult(
        task_id=task_id,
        passed=passed,
        wall_s=wall,
        extract_ok=True,
        completion_tokens=tokens,
        prompt_tokens=50,
        error="",
        raw_text="",
    )


@pytest.fixture
def model_cfg() -> ModelConfig:
    return ModelConfig(id="fake-model", gguf_path="/tmp/does-not-exist.gguf")


@pytest.fixture
def run_request(tmp_path: Path, model_cfg: ModelConfig) -> RunRequest:
    return RunRequest(
        model=model_cfg,
        benchmarks=["humaneval"],
        output_dir=tmp_path,
        rep=0,
        limit=5,
    )


@pytest.fixture
def patched_humaneval(monkeypatch):
    """Patch load_problems + run_problem; return a list that records every
    task_id passed to run_problem so tests can assert which problems re-ran.
    """
    inferred: list[str] = []
    problems = [_problem(f"HumanEval/{i}") for i in range(5)]

    def fake_load_problems(limit=None):
        return problems if limit is None else problems[:limit]

    def fake_run_problem(backend, problem, *, max_tokens, temperature):
        inferred.append(problem["task_id"])
        # task index drives passed/tokens so tests can spot which row came
        # from disk vs from this call.
        idx = int(problem["task_id"].split("/")[-1])
        return _make_result(problem["task_id"], passed=(idx % 2 == 0), tokens=200 + idx)

    monkeypatch.setattr(humaneval_adapter, "load_problems", fake_load_problems)
    monkeypatch.setattr(humaneval_adapter, "run_problem", fake_run_problem)
    return inferred


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def test_cold_start_runs_every_problem(run_request, patched_humaneval):
    summary = _run_humaneval(backend=None, req=run_request)

    assert patched_humaneval == [f"HumanEval/{i}" for i in range(5)]
    assert summary["n_problems"] == 5
    assert summary["n_passed"] == 3  # 0, 2, 4 even-indexed pass
    rows = _read_rows(run_request.output_dir / "humaneval" / "fake-model" / "rep_0" / "results.jsonl")
    assert [r["task_id"] for r in rows] == [f"HumanEval/{i}" for i in range(5)]


def test_resume_skips_existing_rows(run_request, patched_humaneval):
    """Three rows already on disk → only the missing two should be inferred."""
    out_dir = run_request.output_dir / "humaneval" / "fake-model" / "rep_0"
    out_dir.mkdir(parents=True)
    pre_existing = [
        {"task_id": "HumanEval/0", "passed": True, "extract_ok": True,
         "wall_s": 5.0, "prompt_tokens": 50, "completion_tokens": 999, "error": ""},
        {"task_id": "HumanEval/1", "passed": False, "extract_ok": True,
         "wall_s": 7.0, "prompt_tokens": 50, "completion_tokens": 888, "error": "x"},
        {"task_id": "HumanEval/2", "passed": True, "extract_ok": False,
         "wall_s": 3.0, "prompt_tokens": 50, "completion_tokens": 777, "error": ""},
    ]
    (out_dir / "results.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in pre_existing)
    )

    summary = _run_humaneval(backend=None, req=run_request)

    # Only the missing task_ids got inferred.
    assert patched_humaneval == ["HumanEval/3", "HumanEval/4"]

    # All 5 rows are now in the file, in resume order (preserved first, new appended).
    rows = _read_rows(out_dir / "results.jsonl")
    assert [r["task_id"] for r in rows] == [
        "HumanEval/0", "HumanEval/1", "HumanEval/2", "HumanEval/3", "HumanEval/4",
    ]
    # Pre-existing rows kept their original values — completion_tokens of 999/888/777
    # are sentinel markers that prove they came from disk, not the fake runner.
    assert rows[0]["completion_tokens"] == 999
    assert rows[1]["completion_tokens"] == 888
    assert rows[2]["completion_tokens"] == 777

    # Summary tallies old + new.
    assert summary["n_problems"] == 5
    assert summary["n_passed"] == 3  # HE/0 (kept), HE/2 (kept), HE/4 (new even idx)
    assert summary["n_extract_ok"] == 4  # all except HE/2
    assert summary["completion_tokens"] == 999 + 888 + 777 + 203 + 204  # disk + fake(200+idx)
    assert summary["wall_s"] == pytest.approx(5.0 + 7.0 + 3.0 + 1.0 + 1.0)
    assert summary["pass_at_1"] == 3 / 5


def test_torn_last_line_is_dropped_and_rerun(run_request, patched_humaneval):
    """A jsonl with one truncated trailing line should be cleaned: the parsed
    rows are kept, the torn row is re-inferred, and the file ends up valid jsonl.
    """
    out_dir = run_request.output_dir / "humaneval" / "fake-model" / "rep_0"
    out_dir.mkdir(parents=True)
    good = {"task_id": "HumanEval/0", "passed": True, "extract_ok": True,
            "wall_s": 1.0, "prompt_tokens": 10, "completion_tokens": 999, "error": ""}
    torn = '{"task_id": "HumanEval/1", "passed": tru'  # mid-write kill
    (out_dir / "results.jsonl").write_text(json.dumps(good) + "\n" + torn)

    summary = _run_humaneval(backend=None, req=run_request)

    # HumanEval/1 must be re-inferred (its torn row was dropped); /2,/3,/4 too.
    assert patched_humaneval == [f"HumanEval/{i}" for i in (1, 2, 3, 4)]

    rows = _read_rows(out_dir / "results.jsonl")
    assert [r["task_id"] for r in rows] == [f"HumanEval/{i}" for i in range(5)]
    # The kept row is unchanged (sentinel completion_tokens=999 survives).
    assert rows[0]["completion_tokens"] == 999
    assert summary["n_problems"] == 5


def test_clear_stale_for_force_removes_summary(tmp_path: Path) -> None:
    out_dir = tmp_path / "humaneval" / "fake-model" / "rep_0"
    out_dir.mkdir(parents=True)
    (out_dir / "summary.json").write_text('{"stale": true}')
    (out_dir / "results.jsonl").write_text('{"task_id": "HumanEval/0"}\n')

    run_bakeoff._clear_stale_for_force(out_dir)

    assert not (out_dir / "summary.json").exists()
    assert not (out_dir / "results.jsonl").exists()


def test_clear_stale_for_force_leaves_unrelated_files(tmp_path: Path) -> None:
    out_dir = tmp_path / "bfcl" / "fake-model" / "rep_1"
    out_dir.mkdir(parents=True)
    (out_dir / "summary.json").write_text('{}')
    # Per-category subdir + per-problem JSON (BFCL's shape) — must survive.
    cat = out_dir / "simple_python"
    cat.mkdir()
    (cat / "simple_python_0.json").write_text('{"id": "simple_python_0"}')

    run_bakeoff._clear_stale_for_force(out_dir)

    assert not (out_dir / "summary.json").exists()
    assert cat.is_dir()
    assert (cat / "simple_python_0.json").is_file()


def test_clear_stale_for_force_is_idempotent(tmp_path: Path) -> None:
    # Missing dir
    run_bakeoff._clear_stale_for_force(tmp_path / "does" / "not" / "exist")
    # Empty dir
    empty = tmp_path / "empty"
    empty.mkdir()
    run_bakeoff._clear_stale_for_force(empty)
    # Dir with only unrelated files
    odd = tmp_path / "odd"
    odd.mkdir()
    (odd / "notes.txt").write_text("hi")
    run_bakeoff._clear_stale_for_force(odd)
    assert (odd / "notes.txt").is_file()
    # Double-call on a populated dir leaves no residue and doesn't raise.
    pop = tmp_path / "pop"
    pop.mkdir()
    (pop / "summary.json").write_text('{}')
    run_bakeoff._clear_stale_for_force(pop)
    run_bakeoff._clear_stale_for_force(pop)
    assert not (pop / "summary.json").exists()


def test_resume_with_smaller_limit_ignores_extra_rows(tmp_path, model_cfg, patched_humaneval):
    """If the user resumes with a smaller --humaneval-limit, rows for tasks
    beyond the new limit stay on disk but don't contribute to the summary.
    """
    out_dir = tmp_path / "humaneval" / "fake-model" / "rep_0"
    out_dir.mkdir(parents=True)
    # Five rows on disk — sentinel completion_tokens 100..104 so we can prove
    # the summary only counts the first three.
    existing = [
        {"task_id": f"HumanEval/{i}", "passed": True, "extract_ok": True,
         "wall_s": 1.0, "prompt_tokens": 0, "completion_tokens": 100 + i, "error": ""}
        for i in range(5)
    ]
    (out_dir / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in existing))

    req = RunRequest(model=model_cfg, benchmarks=["humaneval"],
                     output_dir=tmp_path, rep=0, limit=3)
    summary = _run_humaneval(backend=None, req=req)

    # Nothing new ran.
    assert patched_humaneval == []
    # Summary counts only the first three (limit=3).
    assert summary["n_problems"] == 3
    assert summary["n_passed"] == 3
    assert summary["completion_tokens"] == 100 + 101 + 102
    # But the extra rows are still preserved on disk for a future larger-limit resume.
    rows = _read_rows(out_dir / "results.jsonl")
    assert {r["task_id"] for r in rows} == {f"HumanEval/{i}" for i in range(5)}
