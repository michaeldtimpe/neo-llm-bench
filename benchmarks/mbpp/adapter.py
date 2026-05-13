"""MBPP (sanitized) pass@1 adapter.

Same shape as the HumanEval adapter (`benchmarks/humaneval/adapter.py`) but
with two key differences MBPP requires:

1. **No `entry_point` field**: MBPP problems describe the spec in natural
   language and reveal the function name only inside the test assertions.
   We extract the function name from `test_list[0]` via `ast.parse` and
   show it (with the first assert) to the model as a function-signature
   hint — standard MBPP 1-shot framing.

2. **Aggressive completion normalization**: MBPP completions are noisier
   than HumanEval's (markdown prose, multiple candidate definitions,
   trailing test scaffolding). We strip fences, anchor to the first
   `def`/`import` line, and drop trailing `if __name__ == "__main__"`
   blocks. Both the raw model output AND the normalized completion are
   persisted so we can audit normalization impact.

Per-task execution is in a fresh subprocess (no shared interpreter state),
matching HumanEval. The 10s wall cap is the same.
"""

from __future__ import annotations

import ast
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


MBPP_PATH = Path(__file__).parent / "MBPP.jsonl"


@dataclass
class MbppResult:
    task_id: str
    passed: bool
    wall_s: float
    extract_ok: bool
    completion_tokens: int = 0
    prompt_tokens: int = 0
    error: str = ""
    raw_text: str = ""
    completion: str = ""  # post-normalization; what was actually executed


def load_problems(limit: int | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(MBPP_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


# Identifiers that are obvious wrappers in MBPP assert lines and aren't
# the actual function under test.
_TEST_WRAPPER_BUILTINS = frozenset({
    "set", "list", "tuple", "dict", "frozenset", "len", "abs", "min", "max",
    "sum", "any", "all", "int", "float", "str", "bool", "bytes", "round",
    "sorted", "reversed", "enumerate", "range", "type", "isinstance",
    "map", "filter", "zip", "iter", "next", "id", "repr", "ascii",
    "math", "round",
})


def _entry_point_from_test(assert_line: str) -> str | None:
    """Pull the function-under-test name out of a test assertion.

    `assert set(similar_elements((3,4,5,6),(5,7,4,10))) == set((4,5))`
    -> `similar_elements`. Skips builtin wrappers (`set`, `len`, etc.).
    """
    try:
        tree = ast.parse(assert_line, mode="exec")
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name not in _TEST_WRAPPER_BUILTINS:
                return name
    return None


_FENCED_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
_FIRST_CODE_LINE = re.compile(r"^[ \t]*(?:from\s+\S+\s+import|import\s+\S+|def\s+\w+|class\s+\w+)",
                              re.MULTILINE)
_MAIN_GUARD = re.compile(r"\n\s*if\s+__name__\s*==\s*['\"]__main__['\"]\s*:\s*$.*",
                         re.MULTILINE | re.DOTALL)


def normalize_completion(text: str, entry_point: str | None = None) -> str:
    """Trim model output to executable Python.

    1. Prefer the first fenced ```python``` block (matching entry_point if
       we have one — protects against blocks that show usage examples).
    2. Strip leading prose (anything before the first def/import/from/class).
    3. Drop trailing `if __name__ == "__main__":` blocks.

    Returns the normalized completion. Empty string if nothing salvageable.
    """
    # Step 1: fenced block (prefer the one that mentions entry_point).
    candidate = ""
    blocks = list(_FENCED_BLOCK.finditer(text))
    if blocks:
        if entry_point:
            for m in blocks:
                if entry_point in m.group(1):
                    candidate = m.group(1)
                    break
        if not candidate:
            candidate = blocks[0].group(1)
    else:
        candidate = text

    # Step 2: anchor to first code-like line.
    m = _FIRST_CODE_LINE.search(candidate)
    if m:
        candidate = candidate[m.start():]

    # Step 3: drop a trailing main-guard block (common scaffolding).
    candidate = _MAIN_GUARD.sub("", candidate)

    return candidate.rstrip()


def _build_prompt(problem: dict[str, Any], entry_point: str | None) -> list[dict[str, str]]:
    """1-shot MBPP framing: spec + one test as a signature hint."""
    first_test = problem["test_list"][0] if problem.get("test_list") else ""
    hint_block = (
        f"The function must satisfy this test:\n```python\n{first_test}\n```\n"
        if first_test else ""
    )
    name_hint = (
        f"The function name should be `{entry_point}`.\n"
        if entry_point else ""
    )
    return [
        {
            "role": "system",
            "content": (
                "You write Python functions to satisfy short natural-language "
                "specs. Output a single ```python``` fenced block containing "
                "the complete function definition (signature + body). No "
                "prose, no example usage, no tests."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{problem['prompt']}\n\n"
                f"{hint_block}"
                f"{name_hint}"
                "Return the function in a single ```python``` fenced block."
            ),
        },
    ]


def _exec_check(
    completion: str,
    test_imports: list[str],
    test_list: list[str],
    timeout_s: float = 10.0,
) -> tuple[bool, str]:
    """Run completion + test_imports + every assert in a fresh subprocess.

    Pass = all asserts run without raising; subprocess exit 0. Fail = any
    assert raises, subprocess non-zero exit, or wall timeout.
    """
    program = (
        "\n".join(test_imports)
        + "\n"
        + completion
        + "\n\n"
        + "\n".join(test_list)
        + "\n"
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
) -> MbppResult:
    pid = f"Mbpp/{problem['task_id']}"
    first_test = problem["test_list"][0] if problem.get("test_list") else ""
    entry_point = _entry_point_from_test(first_test)
    t0 = time.monotonic()
    try:
        resp = backend.chat(
            messages=_build_prompt(problem, entry_point),
            max_tokens=max_tokens, temperature=temperature,
        )
    except Exception as e:  # noqa: BLE001
        return MbppResult(
            task_id=pid, passed=False, wall_s=time.monotonic() - t0,
            extract_ok=False, error=f"backend: {type(e).__name__}: {e}",
        )

    completion = normalize_completion(resp.text, entry_point)
    extract_ok = bool(completion) and (
        entry_point is None or entry_point in completion
    )

    passed, err = _exec_check(
        completion=completion,
        test_imports=list(problem.get("test_imports") or []),
        test_list=list(problem.get("test_list") or []),
        timeout_s=test_timeout_s,
    )
    return MbppResult(
        task_id=pid,
        passed=passed,
        wall_s=time.monotonic() - t0,
        extract_ok=extract_ok,
        completion_tokens=resp.timing.completion_tokens,
        prompt_tokens=resp.timing.prompt_tokens,
        error="" if passed else err,
        raw_text=resp.text,
        completion=completion,
    )
