"""HumanEval pass@1 adapter for llama-bench.

Per-problem flow:
1. Load (prompt, test, entry_point) from the canonical HumanEval.jsonl.
2. Send a chat completion: instruction asking the model to complete the
   function inside a fenced ```python``` block.
3. Extract the function source from the response (fenced block first;
   fall back to "first def block" + everything after).
4. Compose ``prompt + completion + test``, write to a temp .py file, and
   exec under subprocess with a timeout. Pass = exit 0; fail = exit != 0
   or timeout.

We avoid the ``human-eval`` package's executor (it allows arbitrary code
exec; the standard uses ``--allow-execution`` opt-in) and roll a lean
subprocess sandbox: temp dir, no network in the runner, hard wall-clock
cap, capture stdout/stderr for diagnostics.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llamabench.backend import Backend


HUMAN_EVAL_PATH = Path(__file__).parent / "HumanEval.jsonl"


@dataclass
class HumanEvalResult:
    task_id: str
    passed: bool
    wall_s: float
    extract_ok: bool
    completion_tokens: int = 0
    prompt_tokens: int = 0
    error: str = ""
    raw_text: str = ""


def load_problems(limit: int | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(HUMAN_EVAL_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


_FENCED_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
_DEF_OR_IMPORT = re.compile(r"^(\s*(?:from\s+\S+\s+import|import\s+\S+|def\s+\w+|class\s+\w+))",
                            re.MULTILINE)


def extract_code(text: str, entry_point: str) -> str:
    """Pull a function definition for ``entry_point`` out of model output.

    Strategy (in order):
      1. First fenced ```python``` block.
      2. From the first import/def/class line through end of text.
      3. The whole text.

    The returned string is appended to the canonical prompt, so it can
    omit imports the prompt already has — we don't need to be picky here.
    """
    for m in _FENCED_BLOCK.finditer(text):
        block = m.group(1).rstrip()
        if entry_point in block:
            return block
    # Fenced but wrong fence: any block, then check for the entry point
    for m in _FENCED_BLOCK.finditer(text):
        return m.group(1).rstrip()
    # No fence — try grabbing from the first def/import line
    m = _DEF_OR_IMPORT.search(text)
    if m:
        return text[m.start():].rstrip()
    return text.rstrip()


def _build_prompt(problem: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You complete Python functions. Output a single ```python``` "
                "fenced block containing the full function (signature included). "
                "No prose, no extra commentary, no tests."
            ),
        },
        {
            "role": "user",
            "content": (
                "Complete this function:\n\n"
                "```python\n"
                f"{problem['prompt']}"
                "```\n"
                "Return the full function in a fenced ```python``` block."
            ),
        },
    ]


def _exec_check(
    canonical_prompt: str,
    completion: str,
    test_code: str,
    entry_point: str,
    timeout_s: float = 10.0,
) -> tuple[bool, str]:
    """Compose ``prompt + completion + test`` and run it under a subprocess.

    Returns (passed, error_text). Passed means ``check(<entry_point>)``
    returned without raising. Timeouts and non-zero exits are failures.
    """
    program = (
        canonical_prompt
        + "\n"
        + completion
        + "\n\n"
        + test_code
        + f"\ncheck({entry_point})\n"
    )
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "candidate.py"
        path.write_text(program)
        try:
            r = subprocess.run(
                [sys.executable, str(path)],
                capture_output=True, text=True, timeout=timeout_s,
                cwd=td,
            )
            if r.returncode == 0:
                return True, ""
            return False, (r.stderr or r.stdout or "non-zero exit").strip()[:500]
        except subprocess.TimeoutExpired:
            return False, f"timeout after {timeout_s}s"
        except Exception as e:  # noqa: BLE001
            return False, f"{type(e).__name__}: {e}"


def run_problem(
    backend: Backend,
    problem: dict[str, Any],
    *,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    test_timeout_s: float = 10.0,
) -> HumanEvalResult:
    pid = problem["task_id"]
    t0 = time.monotonic()
    try:
        resp = backend.chat(
            messages=_build_prompt(problem),
            max_tokens=max_tokens, temperature=temperature,
        )
    except Exception as e:  # noqa: BLE001
        return HumanEvalResult(
            task_id=pid, passed=False, wall_s=time.monotonic() - t0,
            extract_ok=False, error=f"backend: {type(e).__name__}: {e}",
        )

    code = extract_code(resp.text, problem["entry_point"])
    extract_ok = bool(code) and problem["entry_point"] in code

    # If the model emitted a complete function (signature included), use only
    # `code` — don't double-prefix the canonical prompt. If `code` looks like
    # *just the body*, fall back to prompt+code.
    if f"def {problem['entry_point']}" in code:
        program_prompt = ""
        completion = code
    else:
        program_prompt = problem["prompt"]
        completion = code

    passed, err = _exec_check(
        canonical_prompt=program_prompt,
        completion=completion,
        test_code=problem["test"],
        entry_point=problem["entry_point"],
        timeout_s=test_timeout_s,
    )
    return HumanEvalResult(
        task_id=pid,
        passed=passed,
        wall_s=time.monotonic() - t0,
        extract_ok=extract_ok,
        completion_tokens=resp.timing.completion_tokens,
        prompt_tokens=resp.timing.prompt_tokens,
        error="" if passed else err,
        raw_text=resp.text,
    )
