"""Tests for benchmarks/bfcl/adapter.py — problem loading + dispatch shape.

PRELIMINARY (2026-05-03). Validates message construction and tool-spec
extraction against the installed bfcl_eval data layout. Backend
interaction is mocked.
"""

from __future__ import annotations

from typing import Any

import pytest

from benchmarks.bfcl.adapter import (
    BFCL_SYSTEM_PROMPT,
    BFCL_SYSTEM_PROMPTS,
    SUPPORTED_CATEGORIES,
    _problem_messages,
    _problem_tools,
    get_bfcl_system_prompt,
    load_ground_truth,
    load_problems,
    run_problem_raw,
)
from llamabench.backend import ChatResponse, GenerationTiming, ToolCallResponse


def test_load_problems_each_category():
    """All five supported categories load without error."""
    for cat in SUPPORTED_CATEGORIES:
        problems = load_problems(cat, limit=2)
        assert len(problems) >= 1, f"{cat} returned 0 problems"
        first = problems[0]
        assert "id" in first
        assert "question" in first
        assert "function" in first


def test_load_ground_truth_simple_python():
    gt = load_ground_truth("simple_python")
    assert len(gt) > 0
    # Each entry maps id → list of GT call options
    sample_id = next(iter(gt))
    assert isinstance(gt[sample_id], list)


def test_load_ground_truth_irrelevance_returns_empty():
    """Irrelevance has no positive ground truth; the grader expects no
    tool calls. The loader should return an empty dict cleanly."""
    assert load_ground_truth("irrelevance") == {}


def test_problem_messages_unwraps_nested_question():
    """BFCL wraps question in [[{...}]] — we should flatten to single-turn.
    With the default system prompt, the output is [system, user]."""
    problem = {
        "id": "test_0",
        "question": [[{"role": "user", "content": "hello"}]],
        "function": [{"name": "f", "parameters": {}}],
    }
    msgs = _problem_messages(problem)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == BFCL_SYSTEM_PROMPT
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "hello"


def test_problem_messages_no_system_prompt_when_disabled():
    """Passing system_prompt=None gives the raw BFCL turn back."""
    problem = {
        "id": "test_0",
        "question": [[{"role": "user", "content": "hello"}]],
        "function": [{"name": "f", "parameters": {}}],
    }
    msgs = _problem_messages(problem, system_prompt=None)
    assert msgs == [{"role": "user", "content": "hello"}]


def test_problem_messages_preserves_problem_supplied_system_role():
    """If a future BFCL problem ships its own system message, don't double-stack."""
    problem = {
        "id": "test_0",
        "question": [[
            {"role": "system", "content": "BFCL-shipped sys"},
            {"role": "user", "content": "hello"},
        ]],
        "function": [],
    }
    msgs = _problem_messages(problem)
    # Only the BFCL-supplied system survives; ours is suppressed.
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == "BFCL-shipped sys"


def test_problem_messages_handles_missing_question():
    problem = {"id": "x", "function": []}
    msgs = _problem_messages(problem)
    # Default system + empty user.
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"


def test_problem_tools_extracts_all():
    problem = {
        "function": [
            {"name": "a", "description": "a", "parameters": {"type": "object"}},
            {"name": "b", "description": "b", "parameters": {"type": "object"}},
        ]
    }
    tools = _problem_tools(problem)
    assert len(tools) == 2
    assert {t.name for t in tools} == {"a", "b"}


def test_problem_tools_handles_dict_function_field():
    """Some BFCL entries put a single function dict (not list) — tolerate."""
    problem = {"function": {"name": "f", "parameters": {}}}
    tools = _problem_tools(problem)
    assert len(tools) == 1
    assert tools[0].name == "f"


class _MockBackend:
    """Minimal backend that returns scripted ChatResponses."""

    def __init__(self, response: ChatResponse) -> None:
        self._response = response
        self.last_messages: list[dict[str, Any]] | None = None
        self.last_tools: list[dict[str, Any]] | None = None

    def chat(self, messages, tools=None, **kwargs) -> ChatResponse:
        self.last_messages = list(messages)
        self.last_tools = tools
        return self._response


def test_run_problem_raw_captures_tool_calls():
    """Backend returns one tool call → adapter surfaces (name, args)."""
    problem = load_problems("simple_python", limit=1)[0]
    fake_resp = ChatResponse(
        text="",
        tool_calls=[ToolCallResponse(id="x", name="calculate_triangle_area",
                                     arguments={"base": 10, "height": 5})],
        finish_reason="tool_calls",
        timing=GenerationTiming(prompt_tokens=100, completion_tokens=20),
    )
    backend = _MockBackend(fake_resp)
    result = run_problem_raw(backend, problem)
    assert result.problem_id == problem["id"]
    assert len(result.actual_calls) == 1
    name, args = result.actual_calls[0]
    assert name == "calculate_triangle_area"
    assert args == {"base": 10, "height": 5}
    # Backend was called with the tool spec
    assert backend.last_tools is not None
    assert len(backend.last_tools) >= 1


def test_run_problem_raw_handles_no_tool_calls():
    """Model returns prose only — adapter records empty actual_calls."""
    problem = load_problems("simple_python", limit=1)[0]
    fake_resp = ChatResponse(
        text="I'm not sure what to do here.",
        tool_calls=[],
        finish_reason="stop",
        timing=GenerationTiming(prompt_tokens=100, completion_tokens=10),
    )
    result = run_problem_raw(_MockBackend(fake_resp), problem)
    assert result.actual_calls == []
    assert result.error == ""


def test_run_problem_raw_captures_backend_errors():
    class _FailingBackend:
        def chat(self, *args, **kwargs):
            raise RuntimeError("oMLX is down")

    problem = load_problems("simple_python", limit=1)[0]
    result = run_problem_raw(_FailingBackend(), problem)
    assert result.actual_calls == []
    assert "oMLX is down" in result.error


# ---- System-prompt variant registry (round 3 prereq) ---------------------


def test_system_prompt_registry_contains_round3_variants():
    """All five named variants are registered."""
    assert set(BFCL_SYSTEM_PROMPTS) == {
        "v2", "v3a", "v3b", "v3c", "v2_fewshot_parallel",
    }


def test_system_prompt_v2_alias_matches_constant():
    """BFCL_SYSTEM_PROMPT must equal the v2 entry — back-compat with rounds 1/2."""
    assert BFCL_SYSTEM_PROMPT == BFCL_SYSTEM_PROMPTS["v2"]


def test_system_prompt_variants_share_rules_1_and_3():
    """Rules 1 (parallel) and 3 (math syntax) are kept intact across all
    variants — only rule 2 (decline boundary) is the experimental knob."""
    parallel_marker = "emit N separate tool calls"
    math_marker = "Python operator syntax"
    for name, text in BFCL_SYSTEM_PROMPTS.items():
        assert parallel_marker in text, f"{name} lost rule 1"
        assert math_marker in text, f"{name} lost rule 3"


def test_system_prompt_v3a_uses_imperative():
    """v3a is the stronger-imperative variant for branch A."""
    assert "MUST NOT call any tool" in BFCL_SYSTEM_PROMPTS["v3a"]


def test_system_prompt_v3b_uses_decision_tree():
    """v3b is the decision-tree variant for branch A."""
    assert "ask yourself" in BFCL_SYSTEM_PROMPTS["v3b"]


def test_system_prompt_v3c_loosens_decline():
    """v3c is the looser decline rule for branch C — use-best-available."""
    txt = BFCL_SYSTEM_PROMPTS["v3c"]
    assert "reasonably satisfies" in txt
    assert "best-matching" in txt


def test_system_prompt_v2_fewshot_parallel_appends_examples():
    """v2_fewshot_parallel keeps v2 wholesale and appends two examples."""
    txt = BFCL_SYSTEM_PROMPTS["v2_fewshot_parallel"]
    assert BFCL_SYSTEM_PROMPTS["v2"] in txt  # v2 is a prefix
    assert "Example 1 — parallel:" in txt
    assert "Example 2 — parallel_multiple:" in txt


def test_get_bfcl_system_prompt_unknown_name_raises():
    """Misconfigured variant fails loudly at startup, not silently."""
    with pytest.raises(KeyError, match="unknown bfcl_system_prompt variant"):
        get_bfcl_system_prompt("v99_does_not_exist")


def test_run_problem_raw_uses_supplied_system_prompt():
    """The variant text reaches the backend's system message."""
    problem = load_problems("simple_python", limit=1)[0]
    custom = "SENTINEL-PROMPT-FOR-TEST"
    fake_resp = ChatResponse(
        text="",
        tool_calls=[],
        finish_reason="stop",
        timing=GenerationTiming(prompt_tokens=10, completion_tokens=1),
    )
    backend = _MockBackend(fake_resp)
    run_problem_raw(backend, problem, system_prompt=custom)
    assert backend.last_messages is not None
    sys_msgs = [m for m in backend.last_messages if m.get("role") == "system"]
    assert any(custom in m["content"] for m in sys_msgs)
