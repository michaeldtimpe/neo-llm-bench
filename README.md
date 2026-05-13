# neo-llm-bench

A llama.cpp tool-use + coding bake-off harness for small models on a memory-constrained Mac (~8 GB unified RAM). Ports the evaluation methodology of [`michaeldtimpe/luxe`](https://github.com/michaeldtimpe/luxe) / [`michaeldtimpe/deluxe`](https://github.com/michaeldtimpe/deluxe) (MLX-only) to the llama.cpp ecosystem.

## What it measures

- **BFCL v4** function-calling: 5 curated categories (simple_python, multiple, parallel, parallel_multiple, irrelevance) + 6 live (user-submitted) categories. Sample size adjustable per category.
- **HumanEval** pass@1 across configurable temperatures (deterministic and nucleus sampling).
- **Resource cost**: wall clock and completion tokens per model on the target hardware.

## Current results — round 2 (2026-05-13)

Finalists narrowed from 8 candidates after round 1. Multi-spectrum run at n=150/curated category, n≤100/live category, HumanEval × 3 temperatures.

| | BFCL (n=1106) | HumanEval pass@1 | HumanEval pass@3 (any temp) | live_irrelevance |
|---|---|---|---|---|
| **qwen25-1.5b-instruct** | **77% ±2.5pp** ⭐ | 56% | 66% | 81% |
| **granite33-2b-instruct** | 69% ±2.7pp | 52% | 63% | **100%** ⭐ |
| **qwen25-coder-1.5b-instruct** | 62% ±2.9pp | **70%** ⭐ | **75%** ⭐ | 82% |

Non-dominated triangle: each finalist wins at least one axis. See [`graded_report.md`](graded_report.md) for the full leaderboard with CIs and [`graded_failure_modes.md`](graded_failure_modes.md) for the per-finalist failure breakdown.

## Quick links

- [`QUICKSTART.md`](QUICKSTART.md) — set up on a new machine and reproduce the round-2 leaderboard
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — repo layout, runner lifecycle, grader internals
- [`CLAUDE.md`](CLAUDE.md) — guidance for AI agents working on this codebase
- [`lessons.md`](lessons.md) — hard-won lessons (system sleep, swap thrashing, scope discipline, the v2 system prompt)
- [`graded_report.md`](graded_report.md) — round-2 leaderboard
- [`graded_failure_modes.md`](graded_failure_modes.md) — failure-mode breakdown

## Status

- ✅ BFCL v4: curated + live category support (adapter, grader, 34 unit tests)
- ✅ HumanEval pass@1: subprocess sandbox, fenced extraction, temperature CLI override
- ✅ Multi-rep / multi-temp runs (per-rep dir under `acceptance/<bench>/<model>/rep_N/`)
- ✅ Wilson CIs and head-to-head analysis in reports
- ⏳ BFCL multi-turn categories (data present, grader deferred — needs state tracking)
- ⏳ MBPP (mentioned in plan, not yet wired)
- ⏳ Agent-mode BFCL (raw mode is the comparable baseline; agent mode planned)

## Hardware envelope

Designed for Apple Silicon Macs with **8 GB unified RAM**. Comfortably runs 0.5B–2B Q8_0 GGUFs. 2B models will swap-thrash under sustained load on this RAM tier; budget extra wall time. See `lessons.md` for the empirical degradation curve we observed.
