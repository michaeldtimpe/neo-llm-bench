"""Unit tests for the multi-turn adapter + grader.

We mock backend.chat() and bfcl_eval's multi_turn_checker so the tests
exercise *our* code (conversation driving, call-string conversion, tool-
spec loading, grader wiring) without depending on a live llama-server
or the heavy bfcl_eval evaluation pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from benchmarks.bfcl import grade as grade_mod
from benchmarks.bfcl import multi_turn as mt
from benchmarks.bfcl.grade import grade_multi_turn


# ---- call-string conversion ----------------------------------------------


def test_calls_to_call_strings_simple():
    calls = [("cd", {"folder": "document"})]
    assert mt.calls_to_call_strings(calls) == ["cd(folder='document')"]


def test_calls_to_call_strings_multiple_args():
    calls = [("mv", {"source": "x.pdf", "destination": "temp"})]
    out = mt.calls_to_call_strings(calls)
    # repr() preserves source/destination ordering by dict insertion (Py3.7+)
    assert out == ["mv(source='x.pdf', destination='temp')"]


def test_calls_to_call_strings_various_arg_types():
    calls = [("f", {"n": 5, "name": "abc", "active": True, "items": [1, 2]})]
    s = mt.calls_to_call_strings(calls)[0]
    # repr() round-trips these correctly for eval().
    assert "n=5" in s and "name='abc'" in s and "active=True" in s and "items=[1, 2]" in s


def test_calls_to_call_strings_handles_special_chars_via_repr():
    calls = [("grep", {"pattern": "year's data", "file": "a\nb"})]
    s = mt.calls_to_call_strings(calls)[0]
    # repr escapes — eval() of this string would produce the same args.
    assert eval(s.replace("grep(", "dict(")) == {"pattern": "year's data", "file": "a\nb"}


def test_calls_to_call_strings_multiple_calls_preserved():
    calls = [("a", {"x": 1}), ("b", {"y": 2})]
    assert mt.calls_to_call_strings(calls) == ["a(x=1)", "b(y=2)"]


# ---- tool spec loading ---------------------------------------------------


def test_load_tool_specs_for_GorillaFileSystem():
    """Verifies the actual install ships func_doc files for our classes."""
    tool_defs = mt.load_tool_specs_for_classes(["GorillaFileSystem"])
    names = {td.name for td in tool_defs}
    # GorillaFileSystem has at least these filesystem-shaped methods.
    for expected in ("ls", "cd", "mv", "cp"):
        assert expected in names, f"{expected} should be in {names}"


def test_load_tool_specs_excludes_function():
    """excluded_function names should be filtered out of the tool surface."""
    full = mt.load_tool_specs_for_classes(["GorillaFileSystem"])
    full_names = {td.name for td in full}
    assert "cp" in full_names

    filtered = mt.load_tool_specs_for_classes(["GorillaFileSystem"], excluded_function=["cp"])
    filtered_names = {td.name for td in filtered}
    assert "cp" not in filtered_names
    # And other methods are still present.
    assert "ls" in filtered_names


def test_load_tool_specs_unknown_class_silently_skipped():
    # Class not in MULTI_TURN_FUNC_DOC_FILE_MAPPING → nothing emitted, no raise.
    tool_defs = mt.load_tool_specs_for_classes(["NotARealAPI"])
    assert tool_defs == []


def test_load_tool_specs_unions_multiple_classes():
    fs_only = mt.load_tool_specs_for_classes(["GorillaFileSystem"])
    math_only = mt.load_tool_specs_for_classes(["MathAPI"])
    both = mt.load_tool_specs_for_classes(["GorillaFileSystem", "MathAPI"])
    assert len(both) == len(fs_only) + len(math_only)


# ---- grader wrapper -------------------------------------------------------


def test_grade_multi_turn_pass_path(monkeypatch):
    """multi_turn_checker says valid=True -> GradeResult(passed=True)."""
    def fake_checker(**kwargs):
        return {"valid": True}
    # Patch the import inside grade_multi_turn — it imports lazily.
    import bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker as mtc
    monkeypatch.setattr(mtc, "multi_turn_checker", fake_checker)

    res = grade_multi_turn(
        per_turn_steps=[[["cd(folder='x')"]]],
        test_entry={"id": "multi_turn_base_0",
                    "ground_truth": [["cd(folder='x')"]],
                    "initial_config": {}, "involved_classes": ["GorillaFileSystem"]},
        category="multi_turn_base",
        model_name="fake-model",
    )
    assert res.passed is True
    assert res.reason == "multi_turn_pass"


def test_grade_multi_turn_fail_path_carries_error_type(monkeypatch):
    def fake_checker(**kwargs):
        return {"valid": False, "error_type": "multi_turn:state_mismatch",
                "error_message": "fs differs at /workspace/temp/a.pdf"}
    import bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker as mtc
    monkeypatch.setattr(mtc, "multi_turn_checker", fake_checker)

    res = grade_multi_turn(
        per_turn_steps=[[]],
        test_entry={"id": "multi_turn_base_0", "ground_truth": [["cd(folder='x')"]],
                    "initial_config": {}, "involved_classes": ["GorillaFileSystem"]},
        category="multi_turn_base",
        model_name="fake-model",
    )
    assert res.passed is False
    # The bucket name is preserved so failure_modes.py can group by it.
    assert "state_mismatch" in res.reason


def test_grade_multi_turn_missing_gt_returns_fail():
    res = grade_multi_turn(
        per_turn_steps=[[]],
        test_entry={"id": "x", "initial_config": {}, "involved_classes": []},
        category="multi_turn_base",
        model_name="fake-model",
    )
    assert res.passed is False
    assert "missing_ground_truth" in res.reason


def test_grade_multi_turn_grader_crash_caught(monkeypatch):
    """If multi_turn_checker raises, we surface a clear reason rather than
    crashing the whole bench step."""
    def fake_checker(**kwargs):
        raise RuntimeError("checker exploded on something internal")
    import bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker as mtc
    monkeypatch.setattr(mtc, "multi_turn_checker", fake_checker)

    res = grade_multi_turn(
        per_turn_steps=[[]],
        test_entry={"id": "x", "ground_truth": [["cd()"]],
                    "initial_config": {}, "involved_classes": []},
        category="multi_turn_base",
        model_name="fake-model",
    )
    assert res.passed is False
    assert res.reason.startswith("grader_crash:RuntimeError:")


# ---- conversation driver (mocked backend) ---------------------------------


@dataclass
class _FakeTiming:
    prompt_tokens: int = 50
    completion_tokens: int = 20


@dataclass
class _FakeToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class _FakeChatResp:
    text: str = ""
    tool_calls: list[_FakeToolCall] = field(default_factory=list)
    timing: _FakeTiming = field(default_factory=_FakeTiming)


class _ScriptedBackend:
    """Returns the next pre-scripted ChatResponse each time chat() is called."""
    def __init__(self, script: list[_FakeChatResp]):
        self.script = list(script)
        self.calls: list[dict] = []

    def chat(self, *, messages, tools=None, max_tokens, temperature):
        self.calls.append({
            "n_messages": len(messages),
            "had_tools": tools is not None,
            "last_role": messages[-1].get("role") if messages else None,
        })
        if not self.script:
            return _FakeChatResp()
        return self.script.pop(0)


def test_run_problem_multi_turn_terminates_on_no_tool_call(monkeypatch):
    """One turn, model emits no tool calls → ends cleanly with 0 steps recorded."""
    backend = _ScriptedBackend([_FakeChatResp(text="I'd rather not.", tool_calls=[])])
    problem = {
        "id": "test_no_call_0",
        "question": [[{"role": "user", "content": "list the files"}]],
        "involved_classes": ["GorillaFileSystem"],
        "initial_config": {"GorillaFileSystem": {"root": {"workspace": {"type": "directory", "contents": {}}}}},
    }
    res = mt.run_problem_multi_turn(
        backend, problem, max_tokens=128, temperature=0.0,
        max_steps_per_turn=3, category="multi_turn_base",
    )
    assert res.error == ""
    assert len(res.per_turn_steps) == 1
    assert res.per_turn_steps[0] == []  # no steps emitted calls
    assert res.per_turn_trace[0].stop_reason == "no_tool_call"
    assert res.per_turn_trace[0].steps[0].assistant_raw_text == "I'd rather not."


def test_run_problem_multi_turn_executes_calls_and_advances(monkeypatch):
    """Two-step turn: model emits ls, sees results, emits no more, done."""
    backend = _ScriptedBackend([
        _FakeChatResp(tool_calls=[_FakeToolCall("ls", {"a": False})]),
        _FakeChatResp(text="here are the files", tool_calls=[]),
    ])
    problem = {
        "id": "test_ls_0",
        "question": [[{"role": "user", "content": "list"}]],
        "involved_classes": ["GorillaFileSystem"],
        "initial_config": {"GorillaFileSystem": {"root": {"workspace": {"type": "directory", "contents": {"a.txt": {"type": "file", "content": "hello"}}}}}},
    }
    res = mt.run_problem_multi_turn(
        backend, problem, max_tokens=128, temperature=0.0,
        max_steps_per_turn=3, category="multi_turn_base",
    )
    assert res.error == ""
    assert len(res.per_turn_steps) == 1
    assert res.per_turn_steps[0] == [["ls(a=False)"]]
    # exec_results should be a non-empty list — we ran ls on a fresh FS.
    exec_res = res.per_turn_trace[0].steps[0].execution_results
    assert len(exec_res) == 1
    assert "a.txt" in exec_res[0]  # ls() should mention the file


def test_run_problem_multi_turn_max_steps_caps_runaway():
    """Model that keeps emitting tool calls should be capped at max_steps_per_turn."""
    runaway = [_FakeChatResp(tool_calls=[_FakeToolCall("ls", {"a": False})])] * 5
    backend = _ScriptedBackend(runaway)
    problem = {
        "id": "test_loop_0",
        "question": [[{"role": "user", "content": "loop"}]],
        "involved_classes": ["GorillaFileSystem"],
        "initial_config": {"GorillaFileSystem": {"root": {"workspace": {"type": "directory", "contents": {}}}}},
    }
    res = mt.run_problem_multi_turn(
        backend, problem, max_tokens=128, temperature=0.0,
        max_steps_per_turn=3, category="multi_turn_base",
    )
    assert res.per_turn_trace[0].stop_reason == "max_steps"
    assert len(res.per_turn_trace[0].steps) == 3
    assert len(backend.calls) == 3  # didn't exceed cap


def test_run_problem_multi_turn_multiple_turns_advance_history():
    """Two turns, each ending immediately with text response. Verify the
    second turn's user message is in the message history seen by chat()."""
    backend = _ScriptedBackend([
        _FakeChatResp(text="turn 0 reply", tool_calls=[]),
        _FakeChatResp(text="turn 1 reply", tool_calls=[]),
    ])
    problem = {
        "id": "test_2_turns_0",
        "question": [
            [{"role": "user", "content": "hi turn 0"}],
            [{"role": "user", "content": "hi turn 1"}],
        ],
        "involved_classes": [],
        "initial_config": {},
    }
    res = mt.run_problem_multi_turn(
        backend, problem, max_tokens=128, temperature=0.0,
        max_steps_per_turn=3, category="multi_turn_base",
    )
    assert len(res.per_turn_steps) == 2
    # Second chat() call should see more messages than the first (history grew).
    assert backend.calls[1]["n_messages"] > backend.calls[0]["n_messages"]
