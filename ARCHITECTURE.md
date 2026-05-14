# Architecture

## Repo layout

```
neo-llm-bench/
├── src/llamabench/                  # core package
│   ├── backend.py                   # OpenAI-compatible HTTP client → llama-server
│   ├── server.py                    # LlamaServer subprocess lifecycle (n_parallel-aware)
│   ├── runner.py                    # bench orchestrator: start server, run benches, stop
│   ├── metadata.py                  # metadata.json builder (GGUF SHA, llama.cpp commit, host)
│   ├── config.py                    # Pydantic ModelConfig + BenchProfile loaders
│   ├── agents/                      # 5-shape text-channel tool-call parser (lifted from deluxe)
│   └── tools/                       # filesystem/git/shell tool implementations
├── benchmarks/
│   ├── bfcl/
│   │   ├── adapter.py               # raw + agent mode (single-turn)
│   │   ├── multi_turn.py            # multi-turn driver — wraps bfcl_eval mock APIs
│   │   ├── grade.py                 # call-shape graders + grade_multi_turn (via bfcl_eval)
│   │   └── schemas.py               # ToolDef ↔ BFCL func-spec; make_stub_executor
│   ├── humaneval/                   # HumanEval pass@1 — fenced extraction + subprocess sandbox
│   └── mbpp/                        # MBPP sanitized — adapter + vendored MBPP.jsonl (n=427)
├── scripts/
│   ├── run_bakeoff.py               # CLI: orchestrate models × benches
│   ├── grade_bakeoff.py             # post-hoc BFCL leaderboard (handles multi-turn dispatch)
│   ├── failure_modes.py             # per-model failure-bucket breakdown
│   ├── audit_one_multi_turn.py      # subprocess-isolated multi-turn regrade
│   └── stub_ablation.py             # one-off Phase C0 stub-realism experiment
├── configs/
│   ├── profile_8gb.yaml             # original 8 GB-Mac profile
│   ├── profile_m5max.yaml           # 128 GB M5 Max profile (parallel_models=3)
│   └── models/*.yaml                # per-model config (GGUF path, sampling, tool template)
├── tests/                           # pytest; ~689 tests across the suite
└── acceptance/                      # benchmark outputs (mostly committed for finalist data)
    ├── bfcl/<model>/rep_N/<category>/<problem_id>.json
    ├── humaneval/<model>/rep_N/results.jsonl + summary.json + metadata.json
    ├── mbpp/<model>/rep_0/results.jsonl + summary.json + metadata.json
    └── _logs/                       # run logs (gitignored)
```

### Rep conventions

- `rep_0` = deterministic baseline (HumanEval t=0.0, MBPP t=0.0)
- `rep_1` = round-2 deep BFCL (curated 150/cat + live 100/cat, raw mode)
- `rep_2`, `rep_3` = HumanEval temperature sweep (t=0.3, t=0.7)
- `rep_4` = BFCL agent mode (same problems as rep_1, closed-loop dispatch)
- `rep_5` = BFCL multi-turn (the 4 `multi_turn_*` categories)
- `rep_6`+ reserved for round 3 (prompt-engineering experiments)

## Runtime flow

1. **`scripts/run_bakeoff.py`** parses CLI args, loads the profile (`--profile` flag; defaults to `profile_8gb.yaml`) and per-model YAMLs. Picks a port (`--auto-port` for parallel runs, `--port` for explicit override).
2. For each `(model, bench)` step:
   - **`src/llamabench/runner.py:run()`** spawns a `llama-server` subprocess via `LlamaServer.start()` with model-specific flags (n_ctx, gpu_layers, KV cache types, jinja) and the profile's `max_parallel_requests`.
   - Before the bench runs, `metadata.json` is written next to the eventual `summary.json` capturing GGUF SHA, llama.cpp commit, host info, and `mode` (bench-specific: `bfcl_mode`, `bfcl_run_mode`).
   - The benchmark's runner function (looked up in `_BENCH_RUNNERS: dict[str, BenchmarkSpec]`) iterates problems, calls `backend.chat()`, persists per-problem JSON.
   - `LlamaServer.stop()` reaps the subprocess. Next model starts a fresh server (llama.cpp has no hot model swap).
3. Resume semantics: each `(model, bench, rep)` checks for an existing `summary.json`. If present and `--force` not set, the step is skipped. `--force` calls `_clear_stale_for_force(out_dir, spec.force_clean_filenames)` to delete each registered bench's per-problem resume artifacts before re-running.
4. **`BenchmarkSpec`** (in `runner.py`) is the registration unit: `{name, runner_fn, supports_per_problem_resume, force_clean_filenames}`. New benchmarks extend `_BENCH_RUNNERS` without touching cleanup branches.

## BFCL adapter

`benchmarks/bfcl/adapter.py` loads problems from the installed `bfcl_eval` package (`data/BFCL_v4_<category>.json`) and converts BFCL's chat-shape into OpenAI-style messages.

### The v2 system prompt

`BFCL_SYSTEM_PROMPT` is prepended to every BFCL problem in raw mode. Three rules added empirically after round-1 failure-mode analysis:

1. **"Emit N separate tool calls for N inputs"** — targets the single-call collapse (qwen25-coder packed N inputs into one call with array args).
2. **"Don't call any tool when tools can't satisfy the user"** — targets irrelevance over-calling (every instruct model's top failure mode at round 1).
3. **"Use `**` not `^` for exponentiation"** — targets math-notation mismatches that the grader can't auto-fix.

Measured effects: granite33 irrelevance went 12/30 → 25/30 (rule 2); qwen25-coder parallel single-call rate dropped 29/30 → 16/30 (rule 1, partial fix). See `lessons.md` for the full story.

### Modes (`--bfcl-mode`)

- **`auto`** (default): try structured-tools first; on HTTP 500 from llama-server's jinja parser, fall back to prompt-inject mode for that problem.
- **`structured`**: structured-tools only; problems fail if jinja can't parse.
- **`inject`**: always prompt-inject (text-rendered tool specs). Required for models without a native tool template (phi-1.5, deepseek-coder-1.3b in round 1) — set `bfcl_mode: inject` in the model YAML to force this.

### Categories

- **Curated** (`SUPPORTED_CATEGORIES`): `simple_python`, `multiple`, `parallel`, `parallel_multiple`, `irrelevance` — 5 categories, problem counts 200–400 each in BFCL v4.
- **Live** (`SUPPORTED_LIVE_CATEGORIES`): `live_simple`, `live_multiple`, `live_parallel`, `live_parallel_multiple`, `live_irrelevance`, `live_relevance` — user-submitted, generally harder distributions.
- `live_irrelevance` reuses `grade_irrelevance` (pass = no call). `live_relevance` uses `grade_relevance` (pass = at least one call, no GT validation — BFCL v4 has no `possible_answer` file for these). Multi-turn categories are present in the data but deferred (grader needs state tracking).

### Grader

`benchmarks/bfcl/grade.py` implements a pragmatic subset of BFCL's official grader:

- `simple` / `multiple`: exactly one call; name matches GT; every GT arg's value is in its allowed-list.
- `parallel` / `parallel_multiple`: call-count equals GT; greedy match each emitted call to one unconsumed GT entry.
- `irrelevance` / `live_irrelevance`: zero calls.
- `live_relevance`: at least one call.

Two normalizations applied at value-match time:
- **Nested-dict allowed-lists** (`_dict_shape_matches`): BFCL v4 wraps every leaf inside a dict-typed arg in its own list. The grader recurses so `budget: {min: 300000, max: 400000}` matches GT shape `budget: [{min: [300000], max: [400000]}]`. Round-1 patch.
- **Implicit-multiplication normalizer** (`_normalize_math_expr`): `3*x**2` ↔ `3x**2` (collapses `<digit>*<ident>` on both sides). Does NOT rewrite `^` → `**` (XOR is a real error).

34 unit tests in `tests/test_bfcl_grade.py`.

### Per-problem persistence and `raw_text` semantics

Each BFCL problem produces a `<problem_id>.json` row in
`acceptance/bfcl/<model>/rep_N/<category>/`. Shared fields across raw,
agent, and multi-turn paths: `id`, `actual_calls`, `wall_s`,
`prompt_tokens`, `completion_tokens`, `error`, plus per-mode extras
(`n_turns`, `n_tool_calls_total`, `n_schema_rejects` for agent;
`per_turn_steps` for multi-turn). After grading, `passed` and `reason`
are written back by `scripts/grade_bakeoff.py --write-back`.

**`raw_text` field** (since 2026-05-14, after Phase J audit fix). The
field is optional (`raw_text: str | None`); legacy reps written before
this date deserialize with `raw_text` absent and remain readable.
Semantics depend on mode:

- **Raw mode** (`run_problem_raw`): `raw_text` = the assistant's
  `ChatResponse.text` from the single backend call. May be empty if
  the model emits only structured tool calls with no surrounding
  prose. Useful for offline re-extraction (e.g. text-channel tool-call
  parsing) and for diagnosing models that ignore the tool-spec.
- **Agent mode** (`run_problem_agent`): `raw_text` = the **concatenated
  assistant turns** across the agent loop, joined by a literal
  `"\n---\n"` separator. Each entry is one assistant message's
  `content`; tool-result turns are excluded. This is a trace, not a
  single completion — multi-turn loops may contribute several entries.
- **Multi-turn mode** (`run_problem_multi_turn`): per-turn assistant
  text is already captured inside `per_turn_steps`; `raw_text` is not
  populated separately.

Why this matters: Phase H asserted "smollm3 emits Python code blocks
in agent mode" without persisted evidence (the field didn't exist).
Any future agent-mode mechanism claim should cite `raw_text` directly,
not derive shape from `actual_calls` absence.

## HumanEval adapter

`benchmarks/humaneval/adapter.py` loads problems from the canonical `HumanEval.jsonl` (164 problems). Per problem:

1. Build a 2-message chat: system ("complete the function, fenced python block") + user (the prompt).
2. Call `backend.chat(temperature=...)`.
3. Extract code: fenced ```python``` block first; fall back to "first def/import line to end".
4. Compose `prompt + completion + test` (or just `completion + test` if the completion is a full function); write to a temp `.py`; `subprocess.run` with a 10s wall cap.
5. Pass = exit 0; fail = non-zero / timeout / exception.

The full model response is persisted in `results.jsonl` as `raw_text` (since 2026-05-13) — enables offline re-extraction and re-execution for independent verification.

Temperature is taken from `req.temperature_override` if set (CLI `--temperature` flag), otherwise from the model YAML's `sampling.temperature`. This lets the multi-temp sweep run without YAML edits.

## Memory-on-8GB constraints

The harness runs **one server per model**, not a long-lived multi-model server, because:

- llama.cpp has no in-process model swap.
- The 8 GB Mac doesn't have headroom to keep multiple models loaded.

The 2B Granite model at Q8_0 + n_ctx 8192 + Q8_0 KV uses ~3 GB RSS and pushes the system into ~6 GB swap under sustained load. Expect ~2.5× wall-time slowdown vs the 1.5B Qwen models for the same workload. See `lessons.md` "swap thrashing" entry.

## Reps and output paths

Each `(model, bench, rep)` produces a self-contained directory. Conventions in this repo:

| rep | content |
|---|---|
| `rep_0` | round-1 (10/cat) and round-1.5 (30/cat) BFCL + HumanEval baseline (t=0.0) |
| `rep_1` | round-2 deep BFCL — curated 150/cat + live ≤100/cat |
| `rep_2` | round-2 HumanEval t=0.3 |
| `rep_3` | round-2 HumanEval t=0.7 |

Historical snapshots are preserved (gitignored) for diffing:
- `rep_0_pre_v2_*` — pre-system-prompt run, 30/cat
- `rep_0_v2_partial_*` — partial v2 run that was interrupted

## BFCL multi-turn adapter

`benchmarks/bfcl/multi_turn.py` drives multi-turn conversations and hands the per-turn-per-step call lists to bfcl_eval's `multi_turn_checker` for end-state grading.

- **Tool specs**: loaded from `bfcl_eval/data/multi_turn_func_doc/<file>.json` for each `involved_classes` entry. Excluded methods per problem (`excluded_function`) are filtered out at load time — the model literally doesn't see them.
- **Conversation driver**: per turn, an inner step loop calls the model, parses tool calls, converts them to bfcl_eval-shaped call-strings via `repr()`, executes via `execute_multi_turn_func_call` against stateful mock APIs (GorillaFileSystem, MathAPI, TwitterAPI, etc.), appends tool-role messages with results, and repeats until the model stops calling tools or hits `max_steps_per_turn`.
- **Mode selection**: structured-with-fallback to inject; mode locked after step 0 to keep message history coherent.
- **Namespace isolation**: instance globals use `neollmbench_runtime_<problem_id>` prefix to avoid colliding with bfcl_eval's eval-time `_eval`-suffixed namespace.

`grade_multi_turn` is a thin wrapper around bfcl_eval's checker. Crashes are caught and surfaced as `grader_crash:*` reason strings — see `graded_failure_modes.md` for the 3-way model/infrastructure/execution failure split.

## MBPP adapter

`benchmarks/mbpp/adapter.py` follows the HumanEval shape with two MBPP-specific guardrails:

- **No `entry_point` field**: extracted from `test_list[0]` via `ast.parse`, skipping builtin wrappers (`set`, `len`, `list`, etc.).
- **Aggressive completion normalization**: prefer fenced block containing the entry_point, anchor to first `def`/`import`/`from`/`class` line, drop trailing `if __name__ == "__main__":` guards. Both raw model output and normalized completion are persisted in `results.jsonl` for audit.
- **Per-task subprocess isolation** (not just per-task try/except) — MBPP completions leak globals more aggressively than HumanEval.

## Run metadata (`metadata.json`)

`src/llamabench/metadata.py:build_run_metadata()` writes provenance per (model, bench, rep) step:

```json
{
  "ts_started": "...",
  "model_id": "...",
  "benchmark": "bfcl",
  "rep": 1,
  "mode": {"bfcl_mode": "auto", "bfcl_run_mode": "raw"},
  "model_config": {"gguf_path": "...", "gguf_sha256": "...", "quant": "Q8_0", ...},
  "server": {"bin": "...", "llama_cpp_commit": "...", "n_ctx": 8192, "n_parallel": 1, ...},
  "sampling": {"temperature": 0.0, ...},
  "host": {"arch": "arm64", "cpu": "Apple M5 Max", "mem_gb": 128, "profile": "..."},
  "tooling": {"python": "3.11.15", "neo_llm_bench_commit": "..."}
}
```

GGUF SHA is cached by `(path, mtime, size)` at `~/.llamabench/gguf-sha-cache.json` so re-runs don't re-hash 2 GB. llama.cpp commit is best-effort (walks up from `server_bin` looking for `.git`).

## What's deferred

- **Per-step `prompt_tokens` capture** for direct per-call truncation visibility. Current `TurnTrace.prompt_tokens_at_turn_end` is cumulative across steps. Clean follow-up if Round-3 branch C or future multi-turn work needs sharper signals.
- **n_ctx auto-bump** when `multi_turn_long_context` is selected. Currently uses the per-model YAML's `n_ctx=8192`, which busts for verbose models (qwen25-coder: 49/100 problems hit HTTP 400).
