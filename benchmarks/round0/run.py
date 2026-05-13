"""Round 0 substrate-gate runner — bake-off candidate probes (0b/0c/0d).

Resume: each (candidate, probe) result lands at
`acceptance/round0/<candidate>__<probe>.json` immediately on completion;
re-running the script skips any tuple that already has a result file.
Delete a result file to force re-run.

Progress: prints start time, per-probe wall, running average, and ETA based
on completed-probe mean × remaining count.

Usage:
    .venv/bin/python -m benchmarks.round0.run
    .venv/bin/python -m benchmarks.round0.run --candidate Qwen3-32B-4bit
    .venv/bin/python -m benchmarks.round0.run --probe 0b
    .venv/bin/python -m benchmarks.round0.run --candidate Qwen3-32B-4bit --probe 0d

Failure semantics: ERROR (probe machinery broke) is distinct from FAIL
(model produced wrong behavior). ERROR results are written too — re-running
will skip them; remove the file to retry.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

from llamabench.backend import Backend, BackendError, ToolCallResponse
from llamabench.agents.loop import _parse_text_tool_calls


def _resolve_tool_calls(resp, known_names: set[str]) -> list[ToolCallResponse]:
    """Mirror loop.py: prefer structured tool_calls; fall back to text-channel parser.

    Without this fallback we'd reject capable models that emit valid tool-call
    JSON in the text channel because oMLX didn't promote it to structured form.
    """
    if resp.tool_calls:
        return list(resp.tool_calls)
    if resp.text:
        return _parse_text_tool_calls(resp.text, known_names)
    return []


# Bumped from 512 — thinking-mode models (Qwen3 reasoning, Llama-3.x with
# extended chain-of-thought) need headroom past the <think> block before they
# emit the tool call. 256 was clipping mid-think on Qwen3-32B-4bit.
PROBE_MAX_TOKENS_T1 = 4096
PROBE_MAX_TOKENS_FOLLOWUP = 1024


# --- Configuration --------------------------------------------------------

CANDIDATES: list[str] = [
    # starcoder2-15b-instruct-v0.1 dropped (both 4-bit and 8-bit) on 2026-05-05:
    # chat template has no tool slot (tools field never referenced in
    # tokenizer_config.json), system messages forbidden, model emits Python
    # code instead of tool calls. Not a fixable connector issue — the model
    # wasn't trained for tool semantics.
    # gemma-3-27b-it-4bit dropped: tool-call refusal is a known model property
    # ("I am programmed to be a harmless AI assistant" — observed all 3 probes
    # 2026-05-05); user confirms this is consistent prior behavior, not probe noise.
    "Yi-1.5-34B-Chat-4bit",
    "Codestral-22B-v0.1-8bit",
    "gemma-2-27b-it-4bit",
    "gemma-2-27b-it-8bit",
    "Qwen2.5-Coder-32B-Instruct-4bit",
    "Qwen2.5-32B-Instruct-4bit",
    "Qwen3-32B-4bit",
    "Llama-3.3-70B-Instruct-3bit",
    # Llama-3.3-70B-Instruct-4bit dropped: 38.81GB exceeds 36GB oMLX ceiling
    # (substrate-fail recorded in acceptance/round0/_dropped/ on 2026-05-05).
]

PROBES: list[str] = ["0b", "0c", "0d"]

OMLX_BASE_URL = "http://127.0.0.1:8000"
RESULTS_DIR = Path("acceptance/round0")
PROBE_TIMEOUT_S = 180.0  # any single probe call must complete inside this
PROBE_LONG_TIMEOUT_S = 95.0  # 0c soft pass threshold
KVPROBE_TARGET_TOKENS = 10_000


# --- Tool definitions (OpenAI tool-call schema) ---------------------------

TOOLS_LIST_AND_READ = [
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List the contents of a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute directory path."}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute file path."}
                },
                "required": ["path"],
            },
        },
    },
]

TOOLS_LIST_ONLY = [TOOLS_LIST_AND_READ[0]]


# --- Result schema --------------------------------------------------------

@dataclass
class ProbeResult:
    candidate: str
    probe: str
    started_at: str  # ISO-8601 UTC
    ended_at: str
    wall_s: float
    verdict: str  # PASS | FAIL | ERROR
    reason: str
    detail: dict  # probe-specific extra fields (truncated traces, tool args, etc.)


def result_path(candidate: str, probe: str) -> Path:
    safe = candidate.replace("/", "_")
    return RESULTS_DIR / f"{safe}__{probe}.json"


def save_result(r: ProbeResult) -> None:
    p = result_path(r.candidate, r.probe)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(r), indent=2))


def load_result(candidate: str, probe: str) -> ProbeResult | None:
    p = result_path(candidate, probe)
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    return ProbeResult(**data)


# --- Probe 0b: chained two-call -------------------------------------------

def probe_0b(model: str) -> tuple[str, str, dict]:
    """list_directory → read_file (smallest) → return first line.

    Returns (verdict, reason, detail).
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="probe0b_"))
    detail: dict = {"tmpdir": str(tmpdir)}
    try:
        # Three files; alpha is the smallest by byte count
        contents = {
            "alpha.txt": "alpha file is the smallest\nsecond line\nthird line\n",
            "bravo.txt": "bravo file is medium-sized\n" + "padding\n" * 5,
            "charlie.txt": "charlie file is the largest\n" + "x" * 800 + "\n",
        }
        for name, body in contents.items():
            (tmpdir / name).write_text(body)
        smallest = "alpha.txt"
        smallest_first_line = "alpha file is the smallest"

        backend = Backend(
            base_url=OMLX_BASE_URL, model=model,
            timeout_s=PROBE_TIMEOUT_S, max_attempts=2,
        )
        prompt = (
            f"List the contents of the directory {tmpdir}, then read the "
            f"smallest text file you find there. After you've read it, "
            f"tell me the first line of that file."
        )
        messages = [{"role": "user", "content": prompt}]

        # --- Turn 1: expect list_directory ---
        known = {"list_directory", "read_file"}
        r1 = backend.chat(messages=messages, tools=TOOLS_LIST_AND_READ,
                          temperature=0.0, max_tokens=PROBE_MAX_TOKENS_T1)
        tcs1 = _resolve_tool_calls(r1, known)
        detail["t1_text"] = (r1.text or "")[:200]
        detail["t1_tool_calls"] = [{"name": tc.name, "arguments": tc.arguments} for tc in tcs1]
        detail["t1_via_text_parser"] = bool(tcs1) and not r1.tool_calls
        if not tcs1:
            return "FAIL", "turn-1: no tool call (model returned prose)", detail
        if len(tcs1) > 1:
            return "FAIL", f"turn-1: emitted {len(tcs1)} parallel tool calls (expected 1)", detail
        tc1 = tcs1[0]
        if tc1.name != "list_directory":
            return "FAIL", f"turn-1: called {tc1.name} (expected list_directory)", detail
        listed_path = tc1.arguments.get("path", "")
        if not Path(listed_path).is_dir():
            return "FAIL", f"turn-1: list_directory path doesn't exist: {listed_path!r}", detail

        # Execute and feed back
        listing_lines = sorted(
            f"{p.name}\t{p.stat().st_size}B" for p in Path(listed_path).iterdir()
        )
        listing_blob = "\n".join(listing_lines)

        messages.append({
            "role": "assistant",
            "content": r1.text or "",
            "tool_calls": [{
                "id": tc1.id, "type": "function",
                "function": {"name": tc1.name, "arguments": json.dumps(tc1.arguments)},
            }],
        })
        messages.append({
            "role": "tool", "tool_call_id": tc1.id, "content": listing_blob,
        })

        # --- Turn 2: expect read_file on smallest ---
        r2 = backend.chat(messages=messages, tools=TOOLS_LIST_AND_READ,
                          temperature=0.0, max_tokens=PROBE_MAX_TOKENS_FOLLOWUP)
        tcs2 = _resolve_tool_calls(r2, known)
        detail["t2_text"] = (r2.text or "")[:200]
        detail["t2_tool_calls"] = [{"name": tc.name, "arguments": tc.arguments} for tc in tcs2]
        if not tcs2:
            return "FAIL", "turn-2: no tool call (failed to chain)", detail
        tc2 = tcs2[0]
        if tc2.name != "read_file":
            return "FAIL", f"turn-2: called {tc2.name} (expected read_file)", detail
        target_path = tc2.arguments.get("path", "")
        target_name = Path(target_path).name
        listed_names = {p.name for p in Path(listed_path).iterdir()}
        if target_name not in listed_names:
            return "FAIL", f"turn-2: hallucinated path {target_path!r} not in listing", detail
        is_smallest = target_name == smallest

        # Execute and feed back
        content = (Path(listed_path) / target_name).read_text()
        messages.append({
            "role": "assistant",
            "content": r2.text or "",
            "tool_calls": [{
                "id": tc2.id, "type": "function",
                "function": {"name": tc2.name, "arguments": json.dumps(tc2.arguments)},
            }],
        })
        messages.append({"role": "tool", "tool_call_id": tc2.id, "content": content})

        # --- Turn 3: expect final assistant text containing first line ---
        r3 = backend.chat(messages=messages, tools=TOOLS_LIST_AND_READ,
                          temperature=0.0, max_tokens=PROBE_MAX_TOKENS_FOLLOWUP)
        detail["t3_text"] = (r3.text or "")[:300]
        first_line = content.splitlines()[0] if content else ""
        contains_first = first_line in (r3.text or "")

        if is_smallest and contains_first:
            return "PASS", f"chained correctly; read smallest ({smallest}); reported first line", detail
        if not is_smallest and contains_first:
            return "FAIL", f"chained but read wrong file ({target_name} not {smallest})", detail
        return "FAIL", f"chained {target_name} but final response missing first line", detail
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# --- Probe 0c: long-prompt KV-pressure ------------------------------------

def _kv_pressure_corpus(target_tokens: int) -> str:
    """Synthesize ~target_tokens worth of structured text. ~4 chars/token."""
    char_budget = target_tokens * 4
    chunk = (
        "## Module summary\n\n"
        "This is a synthetic chunk of text designed to exercise the model's "
        "KV cache without depending on any specific repo content. It contains "
        "structured prose, code blocks, lists, and pseudo-code so a model that "
        "behaves differently on different content shapes still encounters a mix.\n\n"
        "```python\n"
        "def example_function(arg1, arg2, *, flag=False):\n"
        "    \"\"\"Compute something interesting.\"\"\"\n"
        "    if flag:\n"
        "        return arg1 + arg2\n"
        "    return arg1 * arg2\n"
        "```\n\n"
        "Notes:\n"
        "- The function above demonstrates keyword-only arguments.\n"
        "- The flag parameter modifies behavior.\n"
        "- Returning either sum or product based on flag is a toy pattern.\n\n"
        "Additional context: The system under test should ignore most of this "
        "content and only use it as a token-count weight. The actual instruction "
        "comes at the end and is unambiguous.\n\n"
    )
    body = ""
    while len(body) < char_budget:
        body += chunk
    return body[:char_budget]


def probe_0c(model: str) -> tuple[str, str, dict]:
    """Long-prompt KV-pressure: ~10k tokens context + 1 list_directory call.

    Pass = a valid list_directory tool call within PROBE_LONG_TIMEOUT_S; no OOM.
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="probe0c_"))
    detail: dict = {"tmpdir": str(tmpdir), "target_tokens": KVPROBE_TARGET_TOKENS}
    try:
        (tmpdir / "single.txt").write_text("only one file\n")

        corpus = _kv_pressure_corpus(KVPROBE_TARGET_TOKENS)
        prompt = (
            f"Below is some context. After reading, just call list_directory "
            f"on the path {tmpdir} and stop.\n\n"
            f"--- CONTEXT START ---\n{corpus}\n--- CONTEXT END ---\n\n"
            f"Now: list the contents of {tmpdir}."
        )
        messages = [{"role": "user", "content": prompt}]

        backend = Backend(
            base_url=OMLX_BASE_URL, model=model,
            timeout_s=PROBE_TIMEOUT_S, max_attempts=2,
        )
        t0 = time.monotonic()
        r = backend.chat(messages=messages, tools=TOOLS_LIST_ONLY,
                         temperature=0.0, max_tokens=PROBE_MAX_TOKENS_T1)
        wall = time.monotonic() - t0
        tcs = _resolve_tool_calls(r, {"list_directory"})
        detail["chat_wall_s"] = round(wall, 2)
        detail["prompt_tokens"] = r.timing.prompt_tokens
        detail["completion_tokens"] = r.timing.completion_tokens
        detail["text"] = (r.text or "")[:200]
        detail["tool_calls"] = [{"name": tc.name, "arguments": tc.arguments} for tc in tcs]
        detail["via_text_parser"] = bool(tcs) and not r.tool_calls

        if not tcs:
            return "FAIL", f"no tool call after {wall:.1f}s with {r.timing.prompt_tokens} prompt tokens", detail
        tc = tcs[0]
        if tc.name != "list_directory":
            return "FAIL", f"called {tc.name} (expected list_directory)", detail
        if wall > PROBE_LONG_TIMEOUT_S:
            return "FAIL", f"valid call but {wall:.1f}s > {PROBE_LONG_TIMEOUT_S}s threshold", detail
        return "PASS", f"valid tool call in {wall:.1f}s with {r.timing.prompt_tokens} prompt tokens", detail
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# --- Probe 0d: adversarial tool-recovery ----------------------------------

def probe_0d(model: str) -> tuple[str, str, dict]:
    """Force list_directory; return a SURPRISING faked result; observe turn 2.

    Real dir contains a.txt + b.txt. Faked listing returns b.txt + z.txt
    (a.txt missing, z.txt fictional). Three outcomes:
      PASS: turn 2 picks something from the received listing (b.txt or z.txt)
            or reasonably re-lists / asks for clarification.
      FAIL hallucinate: reads a.txt (not in listing).
      FAIL stuck-loop: re-calls list_directory with same args.
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="probe0d_"))
    detail: dict = {"tmpdir": str(tmpdir)}
    try:
        (tmpdir / "a.txt").write_text("alpha real content\n")
        (tmpdir / "b.txt").write_text("bravo real content\n")
        backend = Backend(
            base_url=OMLX_BASE_URL, model=model,
            timeout_s=PROBE_TIMEOUT_S, max_attempts=2,
        )

        prompt = (
            f"List the contents of {tmpdir} and then read each file you find. "
            f"Tell me what's in them."
        )
        messages = [{"role": "user", "content": prompt}]

        # Turn 1
        known = {"list_directory", "read_file"}
        r1 = backend.chat(messages=messages, tools=TOOLS_LIST_AND_READ,
                          temperature=0.0, max_tokens=PROBE_MAX_TOKENS_T1)
        tcs1 = _resolve_tool_calls(r1, known)
        detail["t1_tool_calls"] = [{"name": tc.name, "arguments": tc.arguments} for tc in tcs1]
        detail["t1_text"] = (r1.text or "")[:200]
        detail["t1_via_text_parser"] = bool(tcs1) and not r1.tool_calls
        if not tcs1:
            return "FAIL", "turn-1: no tool call", detail
        tc1 = tcs1[0]
        if tc1.name != "list_directory":
            return "FAIL", f"turn-1: called {tc1.name} (expected list_directory)", detail

        # FAKE the listing: omit a.txt, include z.txt
        faked_listing = "b.txt\t19B\nz.txt\t42B"
        detail["faked_listing"] = faked_listing

        messages.append({
            "role": "assistant",
            "content": r1.text or "",
            "tool_calls": [{
                "id": tc1.id, "type": "function",
                "function": {"name": tc1.name, "arguments": json.dumps(tc1.arguments)},
            }],
        })
        messages.append({"role": "tool", "tool_call_id": tc1.id, "content": faked_listing})

        # Turn 2 — observe
        r2 = backend.chat(messages=messages, tools=TOOLS_LIST_AND_READ,
                          temperature=0.0, max_tokens=PROBE_MAX_TOKENS_T1)
        tcs2 = _resolve_tool_calls(r2, known)
        detail["t2_tool_calls"] = [{"name": tc.name, "arguments": tc.arguments} for tc in tcs2]
        detail["t2_text"] = (r2.text or "")[:300]
        detail["t2_via_text_parser"] = bool(tcs2) and not r2.tool_calls

        if not tcs2:
            # Pure prose — could be asking for clarification (PASS) or refusing
            text = (r2.text or "").lower()
            if any(k in text for k in ("clarif", "unclear", "missing", "not see", "can't find", "where is")):
                return "PASS", "turn-2: prose response asks for clarification (adapted)", detail
            return "FAIL", "turn-2: prose with no clarification or tool call", detail

        tc2 = tcs2[0]
        if tc2.name == "list_directory":
            same_args = tc2.arguments == tc1.arguments
            if same_args:
                return "FAIL", "turn-2: re-called list_directory with same args (stuck-loop)", detail
            return "PASS", "turn-2: re-listed with different args (adapting)", detail

        if tc2.name == "read_file":
            target_name = Path(tc2.arguments.get("path", "")).name
            received_names = {"b.txt", "z.txt"}
            if target_name in received_names:
                return "PASS", f"turn-2: read {target_name} from received listing (adapted)", detail
            if target_name == "a.txt":
                return "FAIL", "turn-2: read a.txt — hallucinated from real-dir knowledge, ignored result", detail
            return "FAIL", f"turn-2: read fabricated file {target_name!r}", detail

        return "FAIL", f"turn-2: unexpected tool {tc2.name}", detail
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


PROBE_FNS = {"0b": probe_0b, "0c": probe_0c, "0d": probe_0d}
PROBE_DESC = {
    "0b": "chained two-call (list → read → first line)",
    "0c": "long-prompt KV-pressure (~10k tokens)",
    "0d": "adversarial tool-recovery",
}


# --- Runner ---------------------------------------------------------------

def fmt_dur(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        return f"{int(s // 60)}m {int(s % 60):02d}s"
    return f"{int(s // 3600)}h {int((s % 3600) // 60):02d}m"


def fmt_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


_SUBSTRATE_FAIL_MARKERS = (
    "exceeds max-model-memory",
    "out of memory",
    "metal command buffer",
    "model not found",  # candidate not registered — also a substrate-side decision
    "chat_template is not set",  # base model w/o chat template — can't even render a prompt
    "chat template error",
)


def _is_substrate_fail(msg: str) -> bool:
    low = msg.lower()
    return any(m in low for m in _SUBSTRATE_FAIL_MARKERS)


def run_one(candidate: str, probe: str) -> ProbeResult:
    fn = PROBE_FNS[probe]
    started = datetime.now(timezone.utc).astimezone()
    t0 = time.monotonic()
    try:
        verdict, reason, detail = fn(candidate)
    except BackendError as e:
        msg = str(e)
        if _is_substrate_fail(msg):
            # Substrate gate decision — record FAIL so resume skips it.
            verdict, reason, detail = "FAIL", f"substrate-fail: {msg[:200]}", {
                "exc_type": "BackendError", "substrate_fail": True,
            }
        else:
            verdict, reason, detail = "ERROR", f"BackendError: {msg}", {"exc_type": "BackendError"}
    except Exception as e:
        verdict, reason, detail = "ERROR", f"{type(e).__name__}: {e}", {
            "exc_type": type(e).__name__,
            "traceback": traceback.format_exc()[-1500:],
        }
    wall = time.monotonic() - t0
    ended = datetime.now(timezone.utc).astimezone()
    return ProbeResult(
        candidate=candidate, probe=probe,
        started_at=started.isoformat(),
        ended_at=ended.isoformat(),
        wall_s=round(wall, 2),
        verdict=verdict, reason=reason, detail=detail,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", action="append", default=None,
                    help="Run only specified candidate(s). Repeatable.")
    ap.add_argument("--probe", action="append", default=None, choices=PROBES,
                    help="Run only specified probe(s). Repeatable.")
    ap.add_argument("--force", action="store_true", help="Re-run even if a result file exists.")
    args = ap.parse_args()

    cands = args.candidate or CANDIDATES
    probes = args.probe or PROBES
    queue = [(c, p) for c in cands for p in probes]

    if not args.force:
        queue = [(c, p) for (c, p) in queue if not result_path(c, p).exists()]

    total = len(queue)
    already_done = (len(cands) * len(probes)) - total

    print(f"[round0] start: {fmt_iso()}")
    print(f"[round0] queue: {total} probes ({len(cands)} candidates × {len(probes)} probes; {already_done} already done — re-run with --force to override)")
    print(f"[round0] OMLX_API_KEY present: {bool(os.environ.get('OMLX_API_KEY'))}")
    print()

    if total == 0:
        print("[round0] nothing to do.")
        return 0

    suite_t0 = time.monotonic()
    completed_walls: list[float] = []
    pass_n = fail_n = error_n = 0
    prev_cand: str | None = None

    for i, (cand, probe) in enumerate(queue, 1):
        suite_elapsed = time.monotonic() - suite_t0
        avg = (sum(completed_walls) / len(completed_walls)) if completed_walls else 90.0
        eta = avg * (total - i + 1)

        print(f"[{i:2d}/{total}] {cand} | {probe} ({PROBE_DESC[probe]})")
        print(f"        elapsed={fmt_dur(suite_elapsed)}  avg/probe={fmt_dur(avg)}  eta={fmt_dur(eta)}", flush=True)

        # oMLX pins a model after serving a request and refuses to unload it
        # for the next candidate's first call (returns 507 "all loaded models
        # are pinned"). Unload everything when the candidate changes so the
        # next candidate has the full memory ceiling available cold.
        if prev_cand is not None and cand != prev_cand:
            try:
                _ub = Backend(base_url=OMLX_BASE_URL, model="(unload-between-candidates)")
                results = _ub.unload_all_loaded()
                if results:
                    freed = ", ".join(f"{m}={'ok' if ok else 'fail'}" for m, ok in results.items())
                    print(f"        · unload before {cand}: {freed}", flush=True)
            except Exception as e:
                print(f"        · unload skipped: {e}", flush=True)

        r = run_one(cand, probe)
        prev_cand = cand
        save_result(r)
        completed_walls.append(r.wall_s)
        if r.verdict == "PASS":
            pass_n += 1
            tag = "✓"
        elif r.verdict == "FAIL":
            fail_n += 1
            tag = "✗"
        else:
            error_n += 1
            tag = "!"
        print(f"        {tag} {r.verdict} in {fmt_dur(r.wall_s)} — {r.reason}")
        print()

    suite_elapsed = time.monotonic() - suite_t0
    print(f"[round0] done: {fmt_iso()}  total={fmt_dur(suite_elapsed)}")
    print(f"[round0] PASS={pass_n}  FAIL={fail_n}  ERROR={error_n}  out of {total}")
    return 0 if error_n == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
