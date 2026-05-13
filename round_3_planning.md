# Round-3 planning — primary-model decision matrix

The round-2 dataset is complete (5 phases shipped, commits `c3bca63` →
`d9a5f1c`). This doc surfaces the data for the round-3 primary-model
pick. **Preserved on purpose** — it's the frozen snapshot of round-2
state and the audit trail for whichever direction round-3 takes.

For benchmark vocabulary see `BENCHMARKS.md`. For per-rep numbers and
narrative, see `graded_report.md` and `graded_failure_modes.md`.

## Comparison matrix

Every cell cites its source rep + grading artifact. Bold = leader for
that axis.

| axis | qwen25-1.5b | qwen25-coder | granite33-2b | source |
|---|---|---|---|---|
| BFCL raw (n=1106) | **77.0%** | 58.6% | 69.4% | `rep_1` · `graded_report.md` TL;DR |
| HumanEval pass@1, t=0.0 | 58.5% | **69.5%** | 53.0% | `rep_0` · `acceptance/humaneval/<m>/rep_0/summary.json` |
| HumanEval pass-any (3 temps) | 64.0% | **78.0%** | 64.6% | `rep_0` + `rep_2` + `rep_3` cross-product · `graded_report.md` HE section |
| MBPP sanitized (n=427) | 62.4% | **64.0%** | 59.6% | `rep_0` · `acceptance/mbpp/<m>/rep_0/summary.json` |
| BFCL agent **lift** (vs raw) | +0.2pp (+1) | **+4.8pp (+52)** | +0.8pp (+8) | `rep_4` − `rep_1` · `graded_report.md` agent-mode section |
| live_irrelevance | 77% | 74% | **97%** (LB 89%) | `rep_1` · `graded_report.md` BFCL live table |
| BFCL multi-turn (n=400) | 0.0% (0) | 0.5% (2) | **1.5% (4)** | `rep_5` · `acceptance/bfcl/<m>/rep_5/*/passed` — see floor-level caveat |
| **retry sensitivity** (HE pass-any − pass@1) | +5.5pp | +8.5pp | **+11.6pp** | derived from `rep_0/2/3` cross-product |
| Wall (BFCL `rep_1`, sec) | **846s** | 1029s | 1339s | `acceptance/bfcl/<m>/rep_1/summary.json:wall_s` (aggregate) |
| Multi-turn `long_ctx` HTTP 400s / 100 | **11** | 49 | 30 | `rep_5` · `acceptance/_logs/multi_turn-<m>.log` |

The retry-sensitivity row matters because it disentangles two things:
**absolute capability** (HumanEval pass@1) from **how much capability
is unlocked by retries** (pass-any minus pass@1). granite33 gains the
most from retries in *percentage points* (+11.6pp); qwen25-coder gains
the most in *absolute count of recovered problems* (+14 problems
between t=0.0 and pass-any); qwen25-1.5b is the steadiest single-shot.

## Deployment profile snapshots

### Tool-use generalist (qwen25-1.5b)

- **Strengths**: BFCL raw 77% (+7.6pp on next). Lowest cost (846s
  wall, 65k tokens for the full rep_1). Cleanest multi-turn
  infrastructure (only 11 long_ctx HTTP 400s — 1/4 of coder's rate).
- **Weaknesses**: Loses HumanEval by 11pp. live_irrelevance is 77% —
  20pp behind granite, which is the difference between "occasionally
  over-calls" and "industrial-grade decline discipline".
- **Best for**: general-purpose assistants where tool-call accuracy +
  speed dominate. Worst for: deployments where over-action is costly.

### Coding specialist (qwen25-coder-1.5b)

- **Strengths**: Wins HumanEval by 11pp at t=0.0 and 14pp on pass-any.
  Biggest agent-loop recovery in the data (+52 problems on BFCL
  parallel categories — agent mode rescues the `under_called_1_of_N`
  failure pattern).
- **Weaknesses**: Worst BFCL raw of the three (58.6%, -18pp from
  qwen25-1.5b). Costs ~3× the tokens of qwen25-1.5b on BFCL. Worst
  multi-turn infrastructure: 49/100 long_ctx HTTP 400s. Per-turn
  verbosity drives cumulative prompt to p95=134k tokens, busting
  n_ctx=8192 repeatedly.
- **Best for**: technical agents with retry budget tolerance + ≥32k
  context. Worst for: deployments with strict latency or cost SLAs
  on tool-use workloads.

### Decline-discipline specialist (granite33-2b)

- **Strengths**: live_irrelevance 97% with CI lower bound 89%. Slight
  edge on multi-turn (4/400 vs 0–2/400). The cleanest "knows when not
  to call a tool" signal in the dataset by a wide margin.
- **Weaknesses**: Falls behind on coding (HumanEval 53%, MBPP 59.6%).
  Over-declines on `multi_turn_miss_func` (74/100 `empty_response`)
  when an alternative tool would have worked — the decline instinct
  is too aggressive at the boundary.
- **Best for**: high-stakes deployments where hallucinated tool calls
  are catastrophic. Worst for: coding-heavy or partial-tool-coverage
  deployments where graceful improvisation matters.

## Round-3 branches considered

Four directions, each from `~/.claude/plans/snappy-riding-pinwheel.md`:

| branch | primary model | hypothesis | wall | strength of case |
|---|---|---|---|---|
| **A** | qwen25-1.5b | v3 prompt closes 30–50% of irrelevance over-call gap | ~30 min | medium — narrow target, measurable, but irrelevance failure is concentrated in the model's prior |
| **B** | qwen25-coder | few-shot prompt matches agent-mode parallel-recovery at 1.2× tokens | ~15 min | **highest** — single mechanistic failure, cheap intervention, clean falsifiable gate |
| **C** | granite33 | v3 prompt shifts the decline boundary on `miss_func` without sacrificing irrelevance | ~20 min | medium — *trade-off measurement*, not one-sided improvement; valid "boundary won't move" outcome |
| **D** | new model | a fourth finalist (gemma-2-2b / llama-3.2-3b / smollm3-3b) breaks the triangle | ~3–4 h | medium — discovers the ceiling but doesn't move the existing axes |

### Why other branches would be set aside

A short note recording the rationale for whichever branches are
*not* picked — preserves auditability across rounds.

- Branches not picked because they don't isolate a clean intervention
  on a specific failure mode (typical for "do more eval" framing).
- Branch D specifically: justified only if you suspect the current
  triangle isn't a true ceiling at this scale. If round-2 data is
  internally consistent (which it is — the failure modes are
  mechanically explainable per model), D is exploratory rather than
  decisive.

## What this matrix doesn't tell you

- **The right pick depends on the *deployment target*, not just the
  numbers.** A coding-bot deployment cares about the HumanEval row;
  a customer-facing assistant cares about live_irrelevance + multi-
  turn infra; an open-ended agent loop cares about pass-any and
  retry sensitivity.
- **Multi-turn (rep_5) doesn't separate the models.** All within
  CI overlap, all at floor level. Don't pick on the multi-turn row.
- **Branches B and C produce data with similar wall cost (~15-30 min).
  Branch A is 30 min. Branch D is hours.** If you're undecided
  between A/B/C, latency isn't the deciding factor.

## Decision

Recorded for paper trail; flows into `round_3_design.md`.

| field | value |
|---|---|
| round-3 primary model | **no single primary — A + B + C run as parallel experiments** |
| chosen branches | A (qwen25-1.5b), B (qwen25-coder), C (granite33-2b) |
| not chosen | D (disruptor model) — no new models this round |
| decision date | 2026-05-13 |

### Rationale

Each of the three round-2 finalists has a distinct, mechanically-
explainable weakness. Rather than picking a single primary, round 3
asks the same question of all three: *can prompt engineering move
each model on its individual weakest axis without harming the others?*
The three branches together test the **prompt-mediated capability**
layer (per `BENCHMARKS.md` methodology invariants) — holding base
weights + orchestrator constant — across the full triangle. Branch D
(new model) is set aside because the goal here is to probe the
existing finalists' headroom, not to extend the field.

The combined-data answer is broader than any single branch: *which
finalists move under v3 prompts and by how much*. That informs the
deployment-profile decisions in a way picking one model wouldn't.
