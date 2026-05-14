# neo-llm-bench

A llama.cpp tool-use + coding bake-off harness for small models on Apple Silicon. Ports the evaluation methodology of [`michaeldtimpe/luxe`](https://github.com/michaeldtimpe/luxe) / [`michaeldtimpe/deluxe`](https://github.com/michaeldtimpe/deluxe) (MLX-only) to the llama.cpp ecosystem.

## What it measures

Five orthogonal benchmark signals (see [`BENCHMARKS.md`](BENCHMARKS.md) for the evaluation-dimensions taxonomy):

- **BFCL raw** (single-turn tool-call accuracy): 5 curated + 6 live categories
- **BFCL agent** (closed-loop orchestration): same problems, run-agent loop with stub feedback
- **BFCL multi-turn** (stateful conversation): 4 categories, graded via `bfcl_eval`'s state checker
- **HumanEval** pass@1 across t={0.0, 0.3, 0.7} + pass-any cross-product
- **MBPP** sanitized split (n=427): short-form coding priors

## Current results — rounds 1-4 complete, audit-corrected 2026-05-14

Finalists narrowed from 8 candidates after round 1. Round 2 established the non-dominated triangle. Round 3 (prompt-engineering branches A+B+C) all falsified. Round 4 Branch D added smollm3-3b as a fourth model — competitive on tool-use + coding, axis-loser on decline. The 2026-05-14 audit (commits `e50fdce2`→`af56bb3c`) corrected a lex-sort slicing bug, added BFCL `raw_text` persistence, and re-ran finalists on full live cats (rep_7) for cross-model parity on the full distribution.

| | BFCL matched (n=1106) | BFCL active live (rep_7, n=1351) | HumanEval t=0.0 | HE pass-any | MBPP (n=427) | live_irrelevance (rep_7, n=884) |
|---|---|---|---|---|---|---|
| **qwen25-1.5b-instruct** | 77.1% [74.6, 79.5] | **67.6%** ⭐ | 58.5% | 64.0% | 62.4% ±4.6pp | 52.8% |
| **granite33-2b-instruct** | 69.4% [66.7, 72.1] | 53.7% | 53.0% | 64.6% | 59.6% ±4.6pp | **81.4% [78.8, 83.9]** ⭐ |
| **qwen25-coder-1.5b-instruct** | 58.7% [55.8, 61.5] | 63.7% | **69.5%** ⭐ | **78.0%** ⭐ | **64.0% ±4.5pp** | 30.7% |
| **smollm3-3b-instruct** | 77.8% [75.2, 80.1] | 63.6% | 64.0% | 75.0% | 61.1% | 49.1% |

Multi-turn (rep_5) at this size class is floor-level for all four (0–1.5%) — see `graded_report.md` for the full breakdown including agent-mode lift, retry sensitivity, and infrastructure-failure analysis.

**Non-dominated quadrilateral after audit:**
- **Active tool-use** (live single + multi + parallel, n=1351): qwen25-1.5b 67.6% (+4pp on smollm3, CI-overlap)
- **Decline-discipline** (live_irrelevance, n=884): granite33 81.4% (**+28.6pp CI-distinct** on next-best)
- **Coding** (HumanEval pass-any): qwen25-coder 78.0% (within CI of smollm3 75.0%)
- **Balanced generalist**: smollm3 tied within CI of qwen25-1.5b on BFCL AND qwen25-coder on every coding metric — axis-loser on decline

Cross-model BFCL claims route through [`scripts/compare_matched_slice.py`](scripts/compare_matched_slice.py) with matched-ID artifacts in [`acceptance/audits/`](acceptance/audits/). See the **Errata and methodology corrections** section in `graded_report.md` for the bug history and per-claim trail.

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
- ✅ Round 3 (prompt-engineering A+B+C): executed; all three primary gates falsified — see `graded_report.md` "Round 3 take-away"
- ✅ Round 4 / Branch D: smollm3 added as 4th model; gemma2 + llama32-3b strictly dominated
- ✅ Audit-correction cycle (2026-05-14, Phase I+J+K+L): lex-sort slicing bug fixed, `raw_text` persistence landed, finalists rerun on full live cats (rep_7) for cross-model parity

## Hardware envelope

Two supported profiles via `--profile configs/profile_<name>.yaml`:

- **`profile_8gb.yaml`** — original target. 8 GB Apple Silicon. 2B models swap-thrash under sustained load.
- **`profile_m5max.yaml`** — current dev environment (128 GB M5 Max). Supports `parallel_models: 3` via `--auto-port`. Round-2 reruns + Phase A–E work was done here.

`n_ctx=8192` is the current per-model default; multi-turn `long_context` problems can bust this for verbose models (see `graded_report.md` multi-turn section).
