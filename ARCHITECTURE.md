# Architecture

## Repo layout

```
neo-llm-bench/
├── src/llamabench/                  # core package
│   ├── backend.py                   # OpenAI-compatible HTTP client → llama-server
│   ├── server.py                    # LlamaServer subprocess lifecycle
│   ├── runner.py                    # bench orchestrator: start server, run benches, stop
│   ├── config.py                    # Pydantic ModelConfig + BenchProfile loaders
│   ├── agents/                      # 5-shape text-channel tool-call parser (lifted from deluxe)
│   └── tools/                       # filesystem/git/shell tool implementations
├── benchmarks/
│   ├── bfcl/                        # BFCL v4 adapter + grader (curated + live categories)
│   └── humaneval/                   # HumanEval pass@1 — fenced extraction + subprocess sandbox
├── scripts/
│   ├── run_bakeoff.py               # CLI: orchestrate models × benches
│   ├── grade_bakeoff.py             # post-hoc BFCL leaderboard from per-problem JSONs
│   ├── failure_modes.py             # per-model failure-bucket breakdown
│   ├── bench.sh / graded_bench.sh   # convenience wrappers
│   └── sample_resources.sh          # background CPU/mem sampler (CSV output)
├── configs/
│   ├── profile_8gb.yaml             # bench profile (memory budget, server binary path)
│   └── models/*.yaml                # per-model config (GGUF path, sampling, tool template)
├── tests/                           # pytest; 46 tests cover BFCL grader + adapter + others
└── acceptance/                      # benchmark outputs (excluded from VCS except finalist data)
    ├── bfcl/<model>/rep_N/<category>/<problem_id>.json
    └── humaneval/<model>/rep_N/results.jsonl + summary.json
```

## Runtime flow

1. **`scripts/run_bakeoff.py`** parses CLI args, loads `configs/profile_8gb.yaml` and `configs/models/<model>.yaml` for each requested model.
2. For each `(model, bench)` step:
   - **`src/llamabench/runner.py:run()`** spawns a `llama-server` subprocess via `LlamaServer.start()` with model-specific flags (n_ctx, gpu_layers, KV cache types, jinja).
   - The bench's runner function (`_run_bfcl` or `_run_humaneval`) iterates problems, calls `backend.chat()`, persists per-problem JSON.
   - `LlamaServer.stop()` reaps the subprocess. Next model starts a fresh server (llama.cpp has no hot model swap).
3. Resume semantics: each `(model, bench, rep)` checks for an existing `summary.json` at `acceptance/<bench>/<model>/rep_N/`. If present and `--force` not set, the step is skipped. HumanEval also resumes per-problem from `results.jsonl`.

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

## What's deferred

- **BFCL multi-turn categories** (`multi_turn_base`, `multi_turn_miss_func`, etc.) — 800 problems in the BFCL v4 data, grader needs state tracking.
- **Agent-mode BFCL** — raw mode is the comparable baseline; agent mode (full `run_agent()` loop with stub executor) is scaffolded but not wired into the bake-off.
- **MBPP** — mentioned in the original plan, not yet implemented. HumanEval is the only coding bench currently.
