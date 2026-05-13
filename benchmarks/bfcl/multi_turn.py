"""Multi-turn BFCL adapter — reuses bfcl_eval's multi_turn_checker.

BFCL multi-turn problems differ structurally from single-turn:

- `question` is a list of *turns*, each containing user messages. The
  conversation advances by injecting the next turn's user message after
  the model has finished responding to the prior one.
- Tools aren't embedded in the problem — they're determined by
  `involved_classes` (e.g. `GorillaFileSystem`, `MathAPI`), with tool
  specs loaded from `bfcl_eval/data/multi_turn_func_doc/<file>.json`.
- The mock APIs are *stateful* Python classes whose behavior depends on
  prior calls (filesystem, message inbox, vehicle settings, etc).
- Grading isn't call-shape match: bfcl_eval's `multi_turn_checker`
  actually executes the model's and ground-truth call sequences against
  the same stateful mock APIs and compares the resulting state.

We drive the conversation, executing the model's calls against the
mock APIs via `execute_multi_turn_func_call` between turns so the model
sees real tool results. At the end we hand the per-turn-per-step
call-string lists to `multi_turn_checker` for grading.

Tool delivery uses the same auto/structured-with-text-fallback path as
raw mode. Message history maintains structured tool_calls + tool roles
when the model uses structured output, and plain text otherwise.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llamabench.backend import Backend, ChatResponse
from llamabench.tools.base import ToolDef

from .adapter import (
    BFCL_SYSTEM_PROMPT, _format_tool_prompt, _is_template_parse_error,
)
from .schemas import bfcl_func_spec_to_tool_def


# The four BFCL v4 multi-turn categories.
MULTI_TURN_CATEGORIES = (
    "multi_turn_base",
    "multi_turn_long_context",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
)


def is_multi_turn(category: str) -> bool:
    return category in MULTI_TURN_CATEGORIES


# ---- Tool spec loading ----------------------------------------------------


def _multi_turn_func_doc_dir() -> Path:
    import bfcl_eval
    return Path(bfcl_eval.__file__).parent / "data" / "multi_turn_func_doc"


def _class_file_mapping() -> dict[str, str]:
    from bfcl_eval.constants.executable_backend_config import (
        MULTI_TURN_FUNC_DOC_FILE_MAPPING,
    )
    return dict(MULTI_TURN_FUNC_DOC_FILE_MAPPING)


def load_tool_specs_for_classes(
    involved_classes: list[str],
    excluded_function: list[str] | None = None,
) -> list[ToolDef]:
    """Load OpenAI-style tool specs for each involved mock class.

    `excluded_function` (from miss_func problems) is the set of methods
    we deliberately strip from the tool surface to force the "no good
    tool available" condition. Names are method-only (e.g. "cp"); we
    drop any spec whose `name` matches.
    """
    fd_dir = _multi_turn_func_doc_dir()
    mapping = _class_file_mapping()
    exclude = set(excluded_function or [])
    out: list[ToolDef] = []
    for cls in involved_classes:
        fname = mapping.get(cls)
        if not fname:
            continue
        path = fd_dir / fname
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            spec = json.loads(line)
            if spec.get("name") in exclude:
                continue
            out.append(bfcl_func_spec_to_tool_def(spec))
    return out


# ---- Call-tuple <-> bfcl_eval call-string conversion ---------------------


def _repr_arg(v: Any) -> str:
    """Render an argument value for the bfcl_eval `eval()` call-string.

    bfcl_eval's executor `eval`s `name(arg=value)` strings. We must
    produce values that `eval` accepts in the executor's namespace.
    Use repr() — handles str / int / float / bool / list / dict /
    None correctly. Tuples become tuple literals, which is fine.
    """
    return repr(v)


def calls_to_call_strings(calls: list[tuple[str, dict[str, Any]]]) -> list[str]:
    out: list[str] = []
    for name, args in calls:
        parts = [f"{k}={_repr_arg(v)}" for k, v in args.items()]
        out.append(f"{name}({', '.join(parts)})")
    return out


# ---- Result types ---------------------------------------------------------


@dataclass
class StepTrace:
    """One iteration of the per-turn inner loop."""
    assistant_raw_text: str
    parsed_calls: list[tuple[str, dict[str, Any]]]
    call_strings: list[str]
    execution_results: list[str]
    structured_tool_calls: bool  # True if the model used native tool_calls


@dataclass
class TurnTrace:
    """One user turn, possibly several model steps within it."""
    turn_idx: int
    user_message: dict[str, Any]
    steps: list[StepTrace] = field(default_factory=list)
    stop_reason: str = ""       # "no_tool_call" | "max_steps" | "error"
    prompt_tokens_at_turn_end: int = 0
    completion_tokens_this_turn: int = 0


@dataclass
class BfclMultiTurnInvocationResult:
    problem_id: str
    # The shape bfcl_eval's multi_turn_checker consumes:
    # outer: turns, inner: steps, innermost: list of call-strings emitted
    # in that step. Empty inner lists OK.
    per_turn_steps: list[list[list[str]]] = field(default_factory=list)
    per_turn_trace: list[TurnTrace] = field(default_factory=list)
    wall_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str = ""


# ---- Conversation driver --------------------------------------------------


def _try_structured_then_inject_first_step(
    backend: Backend,
    messages: list[dict[str, Any]],
    tool_defs: list[ToolDef],
    max_tokens: int,
    temperature: float,
) -> tuple[ChatResponse, str, list[dict[str, Any]]]:
    """First-step call: try structured tools, fall back to inject on
    template grammar errors. Returns (response, mode_used, messages_used).

    For subsequent steps we stick with the chosen mode — flipping
    between structured and inject mid-conversation would create message-
    history confusion.
    """
    openai_tools = [t.to_openai() for t in tool_defs] if tool_defs else None
    try:
        resp = backend.chat(
            messages=messages, tools=openai_tools,
            max_tokens=max_tokens, temperature=temperature,
        )
        return resp, "structured", messages
    except Exception as e:  # noqa: BLE001
        if not _is_template_parse_error(e):
            raise
        # Inject fallback: append tool-spec text to the system message.
        tool_text = _format_tool_prompt(tool_defs)
        injected = list(messages)
        if tool_text and injected and injected[0].get("role") == "system":
            injected[0] = {
                "role": "system",
                "content": injected[0]["content"] + "\n\n" + tool_text,
            }
        resp = backend.chat(
            messages=injected, tools=None,
            max_tokens=max_tokens, temperature=temperature,
        )
        return resp, "inject", injected


def _extract_calls(
    resp: ChatResponse, tool_defs: list[ToolDef],
) -> list[tuple[str, dict[str, Any]]]:
    """Extract (name, args) tuples from a chat response.

    Prefer structured `tool_calls`; fall back to text parsing if the
    model emitted calls inline (common in inject mode).
    """
    if resp.tool_calls:
        return [(tc.name, tc.arguments) for tc in resp.tool_calls]
    if resp.text and tool_defs:
        from llamabench.agents.loop import _parse_text_tool_calls
        known = {td.name for td in tool_defs}
        parsed = _parse_text_tool_calls(resp.text, known)
        return [(p.name, p.arguments) for p in parsed]
    return []


_SAFE_PID_RE = re.compile(r"[^a-zA-Z0-9_]+")


def _safe_namespace_for(problem_id: str) -> str:
    """Make a unique, eval-runner-namespace-distinct name.

    bfcl_eval's executor caches class instances in `globals()` under
    `{model_name}_{test_entry_id}_{class_name}_instance`. We use a
    distinct prefix so our conversation-time instances never collide
    with the eval-time instances bfcl_eval creates inside the checker
    (which suffixes with "_eval").
    """
    return "neollmbench_runtime_" + _SAFE_PID_RE.sub("_", problem_id)


def run_problem_multi_turn(
    backend: Backend,
    problem: dict[str, Any],
    *,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    max_steps_per_turn: int = 6,
    system_prompt: str | None = BFCL_SYSTEM_PROMPT,
    category: str,
) -> BfclMultiTurnInvocationResult:
    """Drive the multi-turn conversation, executing the model's calls
    against the stateful mock APIs between turns.
    """
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
        execute_multi_turn_func_call,
    )

    pid = problem["id"]
    involved_classes = list(problem.get("involved_classes", []))
    initial_config = problem.get("initial_config", {})
    excluded_function = problem.get("excluded_function", [])
    if isinstance(excluded_function, str):
        excluded_function = [excluded_function]
    tool_defs = load_tool_specs_for_classes(involved_classes, excluded_function)
    long_context = (category == "multi_turn_long_context")

    turns = problem.get("question", [])
    if not turns:
        return BfclMultiTurnInvocationResult(
            problem_id=pid, error="empty question list",
        )

    # Build initial messages with system prompt.
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    result = BfclMultiTurnInvocationResult(problem_id=pid)
    t0 = time.monotonic()
    chosen_mode: str | None = None  # "structured" or "inject", locked after step 0

    namespace = _safe_namespace_for(pid)

    try:
        for turn_idx, turn_msgs in enumerate(turns):
            # Inject this turn's user message(s) into history.
            user_message: dict[str, Any] = {}
            for m in turn_msgs:
                if isinstance(m, dict) and "role" in m and "content" in m:
                    messages.append({"role": m["role"], "content": m["content"]})
                    if m.get("role") == "user":
                        user_message = {"role": m["role"], "content": m["content"]}

            trace = TurnTrace(turn_idx=turn_idx, user_message=user_message)
            step_call_lists: list[list[str]] = []
            stop_reason = "no_tool_call"
            ct_before = result.completion_tokens

            for step_idx in range(max_steps_per_turn):
                # First step ever: try structured; on jinja failure, fall
                # back to inject and lock that mode for the rest of the
                # problem.
                if chosen_mode is None:
                    resp, chosen_mode, messages = _try_structured_then_inject_first_step(
                        backend, messages, tool_defs, max_tokens, temperature,
                    )
                else:
                    if chosen_mode == "structured":
                        openai_tools = [t.to_openai() for t in tool_defs] if tool_defs else None
                        resp = backend.chat(
                            messages=messages, tools=openai_tools,
                            max_tokens=max_tokens, temperature=temperature,
                        )
                    else:  # inject
                        resp = backend.chat(
                            messages=messages, tools=None,
                            max_tokens=max_tokens, temperature=temperature,
                        )

                result.prompt_tokens += resp.timing.prompt_tokens
                result.completion_tokens += resp.timing.completion_tokens

                parsed = _extract_calls(resp, tool_defs)
                used_structured = bool(resp.tool_calls)

                if not parsed:
                    # End of turn: model emitted text without calls.
                    messages.append({"role": "assistant", "content": resp.text or ""})
                    trace.steps.append(StepTrace(
                        assistant_raw_text=resp.text or "",
                        parsed_calls=[], call_strings=[], execution_results=[],
                        structured_tool_calls=used_structured,
                    ))
                    stop_reason = "no_tool_call"
                    break

                call_strings = calls_to_call_strings(parsed)
                try:
                    exec_results, _ = execute_multi_turn_func_call(
                        func_call_list=call_strings,
                        initial_config=initial_config,
                        involved_classes=involved_classes,
                        model_name=namespace,
                        test_entry_id=pid,
                        long_context=long_context,
                        is_evaL_run=False,
                    )
                except Exception as e:  # noqa: BLE001
                    exec_results = [f"Error during execution: {type(e).__name__}: {e}"]

                # Append the assistant turn + tool results to history.
                # For both structured and inject paths, we keep the message
                # history simple (assistant text + tool role messages) so
                # the next chat call sees a coherent context. We don't try
                # to round-trip OpenAI's tool_calls/tool_call_id linkage —
                # local llama-server accepts plain tool-role messages.
                messages.append({"role": "assistant", "content": resp.text or ""})
                for cs, res in zip(call_strings, exec_results):
                    messages.append({
                        "role": "tool",
                        "content": f"[{cs}] -> {res}",
                    })

                step_call_lists.append(call_strings)
                trace.steps.append(StepTrace(
                    assistant_raw_text=resp.text or "",
                    parsed_calls=parsed, call_strings=call_strings,
                    execution_results=exec_results,
                    structured_tool_calls=used_structured,
                ))
            else:
                stop_reason = "max_steps"

            trace.stop_reason = stop_reason
            trace.prompt_tokens_at_turn_end = result.prompt_tokens
            trace.completion_tokens_this_turn = result.completion_tokens - ct_before
            result.per_turn_steps.append(step_call_lists)
            result.per_turn_trace.append(trace)
    except Exception as e:  # noqa: BLE001
        result.error = f"{type(e).__name__}: {e}"

    result.wall_s = time.monotonic() - t0
    return result
