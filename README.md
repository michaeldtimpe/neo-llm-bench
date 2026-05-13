# neo-llm-bench

A llama.cpp tool-use + coding bake-off harness for small models on Apple Silicon. Ports the evaluation methodology of [`michaeldtimpe/luxe`](https://github.com/michaeldtimpe/luxe) / [`michaeldtimpe/deluxe`](https://github.com/michaeldtimpe/deluxe) (MLX-only) to the llama.cpp ecosystem.

## What it measures

Five orthogonal benchmark signals (see [`BENCHMARKS.md`](BENCHMARKS.md) for the evaluation-dimensions taxonomy):

- **BFCL raw** (single-turn tool-call accuracy): 5 curated + 6 live categories
- **BFCL agent** (closed-loop orchestration): same problems, run-agent loop with stub feedback
- **BFCL multi-turn** (stateful conversation): 4 categories, graded via `bfcl_eval`'s state checker
- **HumanEval** pass@1 across t={0.0, 0.3, 0.7} + pass-any cross-product
- **MBPP** sanitized split (n=427): short-form coding priors

## Current results — round 2 complete (2026-05-13)

Finalists narrowed from 8 candidates after round 1. Round 2 is **a non-dominated triangle** — each finalist wins at least one axis:

| | BFCL raw (n=1106) | HumanEval t=0.0 | HE pass-any | MBPP (n=427) | live_irrelevance |
|---|---|---|---|---|---|
| **qwen25-1.5b-instruct** | **77.0% ±2.5pp** ⭐ | 58.5% | 64.0% | 62.4% ±4.6pp | 77% |
| **granite33-2b-instruct** | 69.4% ±2.7pp | 53.0% | 64.6% | 59.6% ±4.6pp | **97% ±3.7pp** ⭐ |
| **qwen25-coder-1.5b-instruct** | 58.6% ±2.9pp | **69.5%** ⭐ | **78.0%** ⭐ | **64.0% ±4.5pp** | 74% |

Multi-turn (rep_5) at this size class is floor-level for all three (0–1.5%) — see `graded_report.md` for the full breakdown including agent-mode lift, retry sensitivity, and infrastructure-failure analysis.

## Round 3 — scoped, awaiting execution

[`round_3_design.md`](round_3_design.md) defines three prompt-engineering experiments testing whether the prompt-mediated capability layer can move each finalist on its weakest axis. Combined wall ~30 min parallel. Prereq: ~30 LOC for a `--bfcl-system-prompt` CLI flag.

## Quick links

- [`BENCHMARKS.md`](BENCHMARKS.md) — taxonomy: what each signal measures (read first)
- [`QUICKSTART.md`](QUICKSTART.md) — set up on a new machine + reproduce the leaderboard
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — repo layout, runner lifecycle, grader internals
- [`CLAUDE.md`](CLAUDE.md) — guidance for AI agents working on this codebase
- [`lessons.md`](lessons.md) — hard-won lessons (system sleep, scope discipline, the v2 system prompt)
- [`graded_report.md`](graded_report.md) — round-2 leaderboard, all five benchmarks
- [`graded_failure_modes.md`](graded_failure_modes.md) — failure-mode breakdown per (model, benchmark)
- [`round_3_planning.md`](round_3_planning.md) — round-3 decision matrix (with provenance)
- [`round_3_design.md`](round_3_design.md) — executable round-3 scope

## Status

- ✅ BFCL v4 raw mode: curated + live categories, 34 grader unit tests
- ✅ BFCL agent mode: closed-loop dispatch via `run_agent` + stub executors
- ✅ BFCL multi-turn: state-based grading via bfcl_eval's `multi_turn_checker`
- ✅ HumanEval pass@1: subprocess sandbox, fenced extraction, temperature CLI override
- ✅ MBPP sanitized: per-task subprocess isolation, aggressive completion normalization
- ✅ Multi-rep / multi-temp runs, parallel multi-model runs (`--auto-port`)
- ✅ Run metadata (`metadata.json`) per (model, bench, rep): GGUF SHA, llama.cpp commit, host info
- ✅ Wilson CIs, head-to-head, retry sensitivity in reports
- ⏳ Round 3 (prompt-engineering): scoped in `round_3_design.md`, awaiting execution

## Hardware envelope

Two supported profiles via `--profile configs/profile_<name>.yaml`:

- **`profile_8gb.yaml`** — original target. 8 GB Apple Silicon. 2B models swap-thrash under sustained load.
- **`profile_m5max.yaml`** — current dev environment (128 GB M5 Max). Supports `parallel_models: 3` via `--auto-port`. Round-2 reruns + Phase A–E work was done here.

`n_ctx=8192` is the current per-model default; multi-turn `long_context` problems can bust this for verbose models (see `graded_report.md` multi-turn section).
