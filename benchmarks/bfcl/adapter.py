"""BFCL problem → llamabench Backend invocation adapter (raw + agent modes).

PRELIMINARY (2026-05-03). Loads BFCL v4 problems from the installed
`bfcl_eval` package, runs them against the llamabench backend, returns
(actual_tool_calls, timing) per problem.

Two modes (per `~/.claude/plans/fancy-honking-lerdorf.md`):

- `raw`: single-turn `backend.chat()` with the BFCL function as a tool.
  Captures the model's first emitted tool calls. Comparable to public
  BFCL numbers (fair model-only baseline).
- `agent`: full `run_agent()` loop with the BFCL spec as the only ToolDef
  and a stub executor. Captures all tool calls from the loop. Measures
  whether llamabench's prompt scaffolding helps or hurts.

For irrelevance category: tools are still passed but the model must
correctly NOT call them. Both modes apply.
"""

from __future__ import annotations

import importlib.resources
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llamabench.backend import Backend, ChatResponse
from llamabench.config import RoleConfig
from llamabench.tools.base import ToolDef

from .schemas import bfcl_func_spec_to_tool_def, make_stub_executor


# Categories we run. Subset chosen for Python-relevance and Mode-B parity.
SUPPORTED_CATEGORIES = (
    "simple_python",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
    # multi_turn deferred — needs state-tracking grader.
)

# Live (user-submitted) categories — added 2026-05-12. Same per-row shape
# as the curated set above, so they reuse load_problems / _problem_messages
# unchanged. live_irrelevance has no possible_answer file (pass = no call,
# handled in grade.grade_irrelevance); live_relevance also has none (pass =
# at least one call to any provided tool, handled in grade.grade_relevance).
SUPPORTED_LIVE_CATEGORIES = (
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "live_irrelevance",
    "live_relevance",
)

ALL_CATEGORIES = SUPPORTED_CATEGORIES + SUPPORTED_LIVE_CATEGORIES


# System prompt prepended to every BFCL problem in raw mode. Three rules:
#   (1) targets the "single-call collapse" failure mode (qwen25-coder packed
#       parallel problems into one call with array args; BFCL expects N calls).
#   (2) targets irrelevance over-calling (instruct-tuned models call the tool
#       even when the user's request can't be satisfied by it).
#   (3) targets math-notation: BFCL ground truth uses Python operator syntax
#       (`x**2`, `3*x`); models that emit `x^2` are scored wrong even though
#       a human reader would consider it equivalent.
BFCL_SYSTEM_PROMPT = (
    "You are a function-calling assistant.\n"
    "- To invoke a function on N inputs, emit N separate tool calls. "
    "Do not pack multiple inputs into array arguments unless the function "
    "spec explicitly accepts arrays.\n"
    "- If the available tools cannot satisfy the user's request, do not "
    "call any tool — answer in plain text.\n"
    "- Use Python operator syntax for math expressions: `x**2`, `3*x`. "
    "Do not use `^` for exponentiation."
)


def _bfcl_data_dir() -> Path:
    """Locate the installed bfcl_eval package's data dir."""
    import bfcl_eval
    return Path(bfcl_eval.__file__).parent / "data"


def _category_filename(category: str) -> str:
    return f"BFCL_v4_{category}.json"


def load_problems(category: str, limit: int | None = None) -> list[dict[str, Any]]:
    """Load problems for a category, optionally capped at `limit`."""
    data_dir = _bfcl_data_dir()
    path = data_dir / _category_filename(category)
    if not path.is_file():
        raise FileNotFoundError(f"BFCL category data not found: {path}")
    out: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def load_ground_truth(category: str) -> dict[str, list[Any]]:
    """Load ground-truth for a category as a {problem_id: gt_list} dict.
    Returns empty dict for categories that don't have a possible_answer
    file (irrelevance, live_irrelevance, live_relevance — pass is judged
    purely on call-count semantics, not GT match).
    """
    if category in ("irrelevance", "live_irrelevance", "live_relevance"):
        return {}
    data_dir = _bfcl_data_dir()
    path = data_dir / "possible_answer" / _category_filename(category)
    if not path.is_file():
        return {}
    out: dict[str, list[Any]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            out[entry["id"]] = entry.get("ground_truth", [])
    return out


def _problem_messages(
    problem: dict[str, Any],
    *,
    system_prompt: str | None = BFCL_SYSTEM_PROMPT,
) -> list[dict[str, Any]]:
    """Convert BFCL `question` (list-of-list-of-messages) to a flat
    OpenAI-style message list. Single-turn problems wrap the user
    message in `[[{...}]]`; we take the first turn.

    If ``system_prompt`` is non-empty and the BFCL problem doesn't ship
    its own system message, prepend ours. (BFCL v4 single-turn problems
    don't include a system role today; this is the safe path. If a future
    problem does, we leave its system message intact.)
    """
    question = problem.get("question") or []
    msgs: list[dict[str, Any]] = []
    if not question:
        msgs = [{"role": "user", "content": ""}]
    else:
        first_turn = question[0]
        if not isinstance(first_turn, list):
            first_turn = [first_turn]
        for m in first_turn:
            if isinstance(m, dict) and "role" in m and "content" in m:
                msgs.append({"role": m["role"], "content": m["content"]})
        if not msgs:
            msgs = [{"role": "user", "content": str(question)}]

    if system_prompt and not any(m.get("role") == "system" for m in msgs):
        msgs = [{"role": "system", "content": system_prompt}] + msgs
    return msgs


def _problem_tools(problem: dict[str, Any]) -> list[ToolDef]:
    """Extract function specs from a BFCL problem and convert to ToolDefs."""
    funcs = problem.get("function") or []
    if not isinstance(funcs, list):
        funcs = [funcs]
    return [bfcl_func_spec_to_tool_def(f) for f in funcs]


@dataclass
class BfclInvocationResult:
    problem_id: str
    actual_calls: list[tuple[str, dict[str, Any]]]
    wall_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str = ""
    # Agent-mode-only fields (left at 0 for raw mode). Used for the
    # raw-vs-agent overhead comparison — pass rate alone is misleading
    # without these.
    n_turns: int = 0
    n_tool_calls_total: int = 0   # incl. duplicates/errors emitted by the loop
    n_schema_rejects: int = 0


def _format_tool_prompt(tool_defs: list[ToolDef]) -> str:
    """Render tool specs as text for prompt-injection mode.

    Used when the model's chat template can't grammar-parse structured
    tool calls (Llama-3.2 + llama-server emits a non-standard shape that
    server-side --jinja parsing rejects with HTTP 500), and as the
    primary mechanism for models without a native tool template (Phi-1.5).
    """
    if not tool_defs:
        return ""
    lines = [
        "You have access to the following tool(s). Decide whether to call one.",
        "",
    ]
    for td in tool_defs:
        spec = td.to_openai()["function"]
        params = json.dumps(spec.get("parameters", {}), separators=(",", ":"))
        lines.append(f"- name: {spec['name']}")
        lines.append(f"  description: {spec.get('description','')}")
        lines.append(f"  parameters: {params}")
    lines += [
        "",
        "When you decide to call a tool, output ONLY a JSON object on its own "
        'line in the form: {"name": "<tool_name>", "arguments": {...}}',
        "Output one such object per call. To call multiple tools in parallel, "
        "emit multiple objects, each on its own line.",
        "If no tool is needed, answer the user directly in plain text.",
    ]
    return "\n".join(lines)


def _is_template_parse_error(err: Exception) -> bool:
    """llama-server returns 500 with body 'Failed to parse input at pos N'
    when the model emits a tool-call shape its grammar can't decode.
    Detect this so we can fall back to prompt-injection mode. We do NOT
    fall back on 4xx (our request bug) or on transient 5xx (loading etc).
    """
    s = str(err)
    return "Failed to parse input at pos" in s or "5xx-empty-post-warmup" in s


def run_problem_raw(
    backend: Backend,
    problem: dict[str, Any],
    *,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    mode: str = "auto",
) -> BfclInvocationResult:
    """Raw mode: single chat call, capture emitted tool calls.

    Modes:
      - "auto" (default): try structured tools first; if llama-server's
        chat-template grammar 500s on the response, retry prompt-injected.
      - "structured": only structured tools.
      - "inject": only prompt-injected (no `tools` body param).
    """
    pid = problem.get("id", "unknown")
    messages = _problem_messages(problem)
    tools = _problem_tools(problem)

    t0 = time.monotonic()
    resp: ChatResponse | None = None
    used_mode = mode

    if mode in ("auto", "structured"):
        openai_tools = [t.to_openai() for t in tools] if tools else None
        try:
            resp = backend.chat(
                messages=messages,
                tools=openai_tools,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            used_mode = "structured"
        except Exception as e:  # noqa: BLE001
            if mode == "structured" or not _is_template_parse_error(e):
                return BfclInvocationResult(
                    problem_id=pid, actual_calls=[],
                    wall_s=time.monotonic() - t0,
                    error=f"{type(e).__name__}: {e}",
                )
            # Auto-mode fallback: chat template grammar choked on the
            # model's output shape. Retry without `tools` and inject the
            # spec into the system message instead.
            resp = None

    if resp is None:
        used_mode = "inject"
        tool_text = _format_tool_prompt(tools)
        # `messages` already has BFCL_SYSTEM_PROMPT as messages[0]. In inject
        # mode we append the tool-spec text to that single system message so
        # the model sees one coherent system role rather than two.
        injected = list(messages)
        if tool_text:
            if injected and injected[0].get("role") == "system":
                injected[0] = {
                    "role": "system",
                    "content": injected[0]["content"] + "\n\n" + tool_text,
                }
            else:
                injected = [{"role": "system", "content": tool_text}] + injected
        try:
            resp = backend.chat(
                messages=injected, tools=None,
                max_tokens=max_tokens, temperature=temperature,
            )
        except Exception as e:  # noqa: BLE001
            return BfclInvocationResult(
                problem_id=pid, actual_calls=[],
                wall_s=time.monotonic() - t0,
                error=f"{type(e).__name__}: {e}",
            )

    if resp.tool_calls:
        calls = [(tc.name, tc.arguments) for tc in resp.tool_calls]
    elif resp.text and tools:
        from llamabench.agents.loop import _parse_text_tool_calls
        known = {td.name for td in tools}
        parsed = _parse_text_tool_calls(resp.text, known)
        calls = [(p.name, p.arguments) for p in parsed]
    else:
        calls = []
    return BfclInvocationResult(
        problem_id=pid,
        actual_calls=calls,
        wall_s=time.monotonic() - t0,
        prompt_tokens=resp.timing.prompt_tokens,
        completion_tokens=resp.timing.completion_tokens,
    )


def run_problem_agent(
    backend: Backend,
    role_cfg: RoleConfig,
    problem: dict[str, Any],
    *,
    system_prompt: str = BFCL_SYSTEM_PROMPT,
) -> BfclInvocationResult:
    """Agent mode: full run_agent() loop with the BFCL spec as the only
    ToolDef and stub executor. Captures all tool calls from the loop.
    """
    from llamabench.agents.loop import run_agent

    pid = problem.get("id", "unknown")
    messages_seed = _problem_messages(problem)
    user_text = "\n\n".join(m["content"] for m in messages_seed if m.get("role") == "user")
    tool_defs = _problem_tools(problem)
    tool_fns = {td.name: make_stub_executor({"name": td.name}) for td in tool_defs}

    t0 = time.monotonic()
    try:
        result = run_agent(
            backend=backend,
            role_cfg=role_cfg,
            system_prompt=system_prompt,
            task_prompt=user_text,
            tool_defs=tool_defs,
            tool_fns=tool_fns,
        )
    except Exception as e:  # noqa: BLE001
        return BfclInvocationResult(
            problem_id=pid,
            actual_calls=[],
            wall_s=time.monotonic() - t0,
            error=f"{type(e).__name__}: {e}",
        )

    calls = [(tc.name, tc.arguments) for tc in result.tool_calls
             if not tc.duplicate and not tc.error]
    return BfclInvocationResult(
        problem_id=pid,
        actual_calls=calls,
        wall_s=result.wall_s or (time.monotonic() - t0),
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        n_turns=result.steps,
        n_tool_calls_total=result.tool_calls_total,
        n_schema_rejects=result.schema_rejects,
    )
