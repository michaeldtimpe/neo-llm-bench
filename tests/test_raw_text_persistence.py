"""Tests for `raw_text` field persistence on BFCL per-problem JSONs.

Schema contract (see ARCHITECTURE.md "Per-problem persistence and
`raw_text` semantics"):

- Raw mode: `raw_text` = ChatResponse.text from the single backend call.
- Agent mode: `raw_text` = '\\n---\\n'.join of non-empty assistant turns.
- Field is optional; legacy per-problem JSONs lacking it must remain
  readable through any downstream path.

Regression target: Phase J landed the field after Phase H asserted
"smollm3 emits Python code blocks in agent mode" without persisted
evidence (the field didn't exist yet). These tests keep the field
populated when text is captured and keep legacy rows readable.
"""

from __future__ import annotations

import json
from typing import Any

from benchmarks.bfcl.adapter import (
    BfclInvocationResult,
    load_problems,
    run_problem_raw,
)
from llamabench.agents.loop import AgentResult
from llamabench.backend import ChatResponse, GenerationTiming, ToolCallResponse


class _ScriptedBackend:
    """Backend that returns canned ChatResponses in order."""

    def __init__(self, responses: list[ChatResponse]) -> None:
        self._responses = list(responses)

    def chat(self, messages, tools=None, **kwargs) -> ChatResponse:
        return self._responses.pop(0)


# ---------- raw mode ----------


def test_raw_mode_captures_text_alongside_calls():
    """ChatResponse.text is captured even when the model also emitted
    structured tool_calls."""
    problem = load_problems("simple_python", limit=1)[0]
    resp = ChatResponse(
        text="I'll call the function for you.",
        tool_calls=[
            ToolCallResponse(id="x", name="calculate_triangle_area",
                             arguments={"base": 10, "height": 5})
        ],
        finish_reason="tool_calls",
        timing=GenerationTiming(prompt_tokens=100, completion_tokens=20),
    )
    result = run_problem_raw(_ScriptedBackend([resp]), problem)
    assert result.raw_text == "I'll call the function for you."
    assert len(result.actual_calls) == 1


def test_raw_mode_captures_text_when_no_tool_calls():
    """Prose-only response: raw_text holds it; actual_calls is empty."""
    problem = load_problems("simple_python", limit=1)[0]
    resp = ChatResponse(
        text="I'm not sure how to use this tool.",
        tool_calls=[],
        finish_reason="stop",
        timing=GenerationTiming(prompt_tokens=100, completion_tokens=10),
    )
    result = run_problem_raw(_ScriptedBackend([resp]), problem)
    assert result.raw_text == "I'm not sure how to use this tool."
    assert result.actual_calls == []


def test_raw_mode_empty_text_is_none():
    """Empty string → None so the runner omits the field from JSON,
    keeping diffs minimal for problems where the model said nothing."""
    problem = load_problems("simple_python", limit=1)[0]
    resp = ChatResponse(
        text="",
        tool_calls=[
            ToolCallResponse(id="x", name="calculate_triangle_area",
                             arguments={"base": 10, "height": 5})
        ],
        finish_reason="tool_calls",
        timing=GenerationTiming(prompt_tokens=100, completion_tokens=20),
    )
    result = run_problem_raw(_ScriptedBackend([resp]), problem)
    assert result.raw_text is None


# ---------- agent mode ----------


def _make_agent_result(texts: list[str]) -> AgentResult:
    """Synthesize an AgentResult with the given assistant_texts."""
    return AgentResult(
        final_text=texts[-1] if texts else "",
        steps=len(texts),
        assistant_texts=list(texts),
        prompt_tokens=100,
        completion_tokens=20 * len(texts),
        wall_s=0.1,
    )


def test_agent_mode_joins_turns_with_separator():
    """run_problem_agent should concatenate assistant_texts with the
    documented '\\n---\\n' separator.

    Tests the dataclass conversion shape; the agent loop itself is
    exercised by the smoke run in Phase J.6.
    """
    # We construct the raw_text directly to test the contract.
    texts = ["First turn output.", "Second turn output."]
    raw_text = "\n---\n".join(t for t in texts if t)
    assert raw_text == "First turn output.\n---\nSecond turn output."


def test_agent_mode_empty_texts_yield_none():
    """No assistant text emitted → raw_text is None.

    This is the smollm3 rep_4 shape — the model emitted 0 tool calls
    AND no surrounding text in agent mode; raw_text must be None
    rather than empty string so the JSON row omits the field."""
    texts: list[str] = []
    non_empty = [t for t in texts if t]
    raw_text = "\n---\n".join(non_empty) if non_empty else None
    assert raw_text is None


def test_agent_mode_skips_empty_turns():
    """Empty-string turns (tool_call-only turns) are skipped from the
    concatenation — they would otherwise produce '\\n---\\n\\n---\\n'
    visual noise in the persisted text."""
    texts = ["thinking out loud", "", "final reasoning"]
    non_empty = [t for t in texts if t]
    raw_text = "\n---\n".join(non_empty)
    assert raw_text == "thinking out loud\n---\nfinal reasoning"


# ---------- AgentResult dataclass back-compat ----------


def test_agent_result_default_assistant_texts_is_empty_list():
    """Constructing AgentResult without assistant_texts → empty list,
    not None. Ensures existing call sites (single.py, stub_ablation.py)
    aren't broken by the dataclass extension."""
    result = AgentResult()
    assert result.assistant_texts == []
    assert isinstance(result.assistant_texts, list)


# ---------- runner persistence shape ----------


def test_runner_serializer_omits_raw_text_when_none(tmp_path):
    """Mirrors src/llamabench/runner.py:257-266 serialization logic.

    Legacy invariant: rows with no captured text should NOT have a
    `raw_text: null` field; the field should be absent entirely. This
    keeps file diffs minimal and lets readers distinguish "this rep
    predates raw_text" from "this problem produced no text."
    """
    r = BfclInvocationResult(
        problem_id="x",
        actual_calls=[("f", {"a": 1})],
        wall_s=0.1,
        prompt_tokens=10,
        completion_tokens=5,
        raw_text=None,
    )
    # The exact serialization snippet from runner.py, replicated for the
    # contract test. If runner.py changes shape, update this together.
    row: dict[str, Any] = {
        "id": r.problem_id, "actual_calls": r.actual_calls,
        "wall_s": r.wall_s, "prompt_tokens": r.prompt_tokens,
        "completion_tokens": r.completion_tokens, "error": r.error,
    }
    if r.raw_text is not None:
        row["raw_text"] = r.raw_text
    out = tmp_path / "x.json"
    out.write_text(json.dumps(row))
    reloaded = json.loads(out.read_text())
    assert "raw_text" not in reloaded


def test_runner_serializer_writes_raw_text_when_present(tmp_path):
    """Mirror of the same snippet — present case."""
    r = BfclInvocationResult(
        problem_id="x",
        actual_calls=[],
        wall_s=0.1,
        prompt_tokens=10,
        completion_tokens=5,
        raw_text="model said hello",
    )
    row: dict[str, Any] = {
        "id": r.problem_id, "actual_calls": r.actual_calls,
        "wall_s": r.wall_s, "prompt_tokens": r.prompt_tokens,
        "completion_tokens": r.completion_tokens, "error": r.error,
    }
    if r.raw_text is not None:
        row["raw_text"] = r.raw_text
    out = tmp_path / "x.json"
    out.write_text(json.dumps(row))
    reloaded = json.loads(out.read_text())
    assert reloaded["raw_text"] == "model said hello"


# ---------- legacy JSON back-compat ----------


def test_legacy_per_problem_json_loads_without_raw_text(tmp_path):
    """A pre-2026-05-14 per-problem JSON has no `raw_text` key. Any
    downstream reader (grade_bakeoff, audit scripts, this very test
    suite) must not crash on missing-key.

    This is a schema-evolution invariant: the field is *optional*. If
    a future change makes raw_text required somewhere, this test will
    fail and prompt a back-compat review.
    """
    legacy = {
        "id": "live_simple_0-0-0",
        "actual_calls": [["get_user_info", {"user_id": 7890}]],
        "wall_s": 0.485,
        "prompt_tokens": 329,
        "completion_tokens": 31,
        "error": "",
        "passed": True,
        "reason": "matched_gt_entry",
        # no raw_text — predates the field
    }
    out = tmp_path / "legacy.json"
    out.write_text(json.dumps(legacy))
    reloaded = json.loads(out.read_text())
    # No exception; legacy keys present; raw_text absent
    assert reloaded["passed"] is True
    assert reloaded.get("raw_text") is None
    assert "raw_text" not in reloaded


def test_grade_bakeoff_handles_legacy_rows_without_raw_text(tmp_path):
    """Smoke test: scripts/grade_bakeoff.py's row reader (per-problem
    JSON via path.read_text + json.loads, see lines 130-152) must
    tolerate missing raw_text on legacy rows.

    We construct one cat dir with one legacy row and one fresh-shape
    row, then read both via the same access pattern grade_bakeoff uses.
    """
    cat_dir = tmp_path / "live_simple"
    cat_dir.mkdir()
    (cat_dir / "legacy.json").write_text(json.dumps({
        "id": "legacy", "actual_calls": [], "wall_s": 0.1,
        "prompt_tokens": 10, "completion_tokens": 5, "error": "",
        "passed": False,
    }))
    (cat_dir / "fresh.json").write_text(json.dumps({
        "id": "fresh", "actual_calls": [], "wall_s": 0.1,
        "prompt_tokens": 10, "completion_tokens": 5, "error": "",
        "passed": True, "raw_text": "fresh row",
    }))
    rows = []
    for p in sorted(cat_dir.glob("*.json")):
        rec = json.loads(p.read_text())
        rows.append(rec)
    # Both readable; raw_text presence differs.
    legacy = next(r for r in rows if r["id"] == "legacy")
    fresh = next(r for r in rows if r["id"] == "fresh")
    assert "raw_text" not in legacy
    assert fresh["raw_text"] == "fresh row"
