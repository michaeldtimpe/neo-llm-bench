# Benchmarks — what each signal in this repo actually measures

By the end of round 2, `graded_report.md` cites five different
benchmarks across several reps. They look superficially similar (they
all produce pass rates over a problem set) but they probe *orthogonal*
capabilities. Treating them as five flavors of the same thing causes
ranking confusion — e.g., qwen25-1.5b wins one and loses another not
because the model "improved" between runs, but because the benchmarks
measure different layers of the stack.

This doc names the dimensions, defines the project vocabulary, and
pins down what each benchmark is for.

## Evaluation dimensions

Four orthogonal axes the five benchmarks span. Disagreement between
two benchmarks usually maps to one of these axes:

| dimension | examples | what disagreement means |
|---|---|---|
| closed-book vs closed-loop | BFCL raw vs BFCL agent | agent lift = recoverability *under orchestration*, **not** smarter reasoning |
| stateless vs stateful | BFCL raw vs BFCL multi-turn | multi-turn lift requires holding prior-turn state, not just better tool selection |
| syntax-match vs execution-based | BFCL raw vs HumanEval | execution-based grading is unforgiving on subtly-wrong-but-readable output |
| single-sample vs pass-any | HumanEval t=0.0 vs pass-any | pass-any rewards stochastic diversity (and exposes models with high variance) |

Worked example: qwen25-coder picks up +52 BFCL problems in agent mode
vs raw. That looks like a capability gain. It isn't — it's a
*closed-book → closed-loop* shift. The orchestrator recovers calls the
model got wrong on the first pass. Same model, different evaluation
posture.

## Project vocabulary

Terms that have become core but aren't self-explanatory:

- **`rep_N`** — a numbered experimental run. Conventions in this repo:
  - `rep_0` = deterministic baseline (HumanEval t=0.0, MBPP t=0.0)
  - `rep_1` = round-2 deep BFCL (curated 150/cat + live 100/cat, raw mode)
  - `rep_2`, `rep_3` = HumanEval temperature sweep (t=0.3, t=0.7)
  - `rep_4` = BFCL agent mode (same problems as rep_1, closed-loop dispatch)
  - `rep_5` = BFCL multi-turn (the 4 multi_turn_* categories)
- **pass@1** — fraction of problems passed at the chosen sampling
  temperature, one attempt each.
- **pass-any** (≈ pass@N across temps) — a HumanEval problem is
  "passed-any" if it passes at ≥1 of `rep_0/2/3`. Measures the model's
  ceiling with best-of-N sampling across `{t=0.0, 0.3, 0.7}`.
- **curated vs live BFCL** — curated = original BFCL test cases
  (n=150/cat). live = newer user-submitted problems (n=100/cat, except
  the parallel/relevance categories which cap at 16–24). They probe
  similar shapes; live tends to be harder on tool-disambiguation.
- **irrelevance vs relevance** — irrelevance = the expected behavior
  is *no tool call* (the toolset can't satisfy the user). relevance =
  the expected behavior is *at least one tool call*.

## Per-benchmark table

| benchmark | reps | what it measures | grading basis |
|---|---|---|---|
| BFCL raw (single-turn) | `rep_1` | first-pass tool-call accuracy | call-shape match vs GT |
| BFCL agent (single-turn) | `rep_4` | model + closed-loop orchestration | call-shape match (post-recovery) |
| BFCL multi-turn | `rep_5` | state-tracking across 4–5 turns | end-state of stateful mock APIs (bfcl_eval) |
| HumanEval | `rep_0` (t=0.0), `rep_2` (t=0.3), `rep_3` (t=0.7) | synthesis coding (longer dependencies) | subprocess unit test execution |
| MBPP (sanitized) | `rep_0` | short-form coding priors | subprocess unit test execution |

### What each one *uniquely* probes

- **BFCL raw**: the model's first-pass capability with no feedback.
  Public-leaderboard-comparable. If the model can't produce the right
  tool call on the first shot, raw catches it.
- **BFCL agent**: the closed-loop system's capability — model + the
  agent's recovery + a stub tool world. Diverges from raw on models
  whose first-pass mistakes are *recoverable* with another turn.
- **BFCL multi-turn**: persistent state across user-driven turns. The
  grader executes call sequences against bfcl_eval's stateful mock
  APIs (`GorillaFileSystem`, `MathAPI`, `TradingBot`, etc.) and
  compares end-state — strictly different from "did the model emit
  the right call shape."
- **HumanEval**: synthesis coding. The model gets a function signature
  + docstring and writes the body. Tests longer dependency chains,
  edge cases, and architectural coherence.
- **MBPP (sanitized)**: short-form coding from a natural-language
  spec, no signature given. Rewards memorized educational patterns
  and short-template completion — distinct from HumanEval's synthesis
  shape.

### Where to look

- Adapters: `benchmarks/bfcl/adapter.py` (raw + agent),
  `benchmarks/bfcl/multi_turn.py` (multi-turn driver),
  `benchmarks/humaneval/adapter.py`, `benchmarks/mbpp/adapter.py`
- Graders: `benchmarks/bfcl/grade.py` (call-shape primitives + the
  `grade_multi_turn` wrapper around bfcl_eval's state checker),
  subprocess executors live inside the coding adapters.

## Multi-turn caveat

Multi-turn `rep_5` pass rates (0–1.5% across the three finalists) are
**floor-level signals at this model scale**, not mature capability
rankings. Treat them as stress measurements for long-context handling
and state-tracking robustness. Public BFCL multi-turn shows GPT-4-
class models at 30–50%; small open-weight models reliably cluster near
zero, and the 0–1.5% spread in this report does not statistically
separate the three finalists. The useful comparative signal in `rep_5`
is in the *failure-mode mix* (`graded_failure_modes.md`) — who fails
on context overruns vs who over-declines vs who emits state-mismatches
— not the bare pass rate.

## Methodology invariants

The hierarchy that comparisons across rounds must respect:

1. **Base model capability** — what the weights know. Exposed by
   deterministic raw eval at t=0.0 (`rep_0`, `rep_1` raw).
2. **Prompt-mediated capability** — what a system prompt unlocks.
   The v2 `BFCL_SYSTEM_PROMPT` is the current default; future v3
   variants live here.
3. **Agent/orchestrator recovery** — what closed-loop dispatch
   recovers (`rep_4`).
4. **Infrastructure robustness** — what survives context limits,
   tokenizer quirks, server-side parsing edges (`rep_5` infra
   failures).

When comparing across rounds, hold the higher layers constant or
explicitly note that they moved. "Model X improved" claims should
specify which layer moved — otherwise you're mixing a prompt change
with a model change with an orchestration change and the comparison
is incoherent.

## See also

- `graded_report.md` — current per-rep leaderboard tables
- `graded_failure_modes.md` — failure-shape analysis per (model,
  benchmark) cell
- `ARCHITECTURE.md` — repo layout + runner internals
- `lessons.md` — operational lessons (scope discipline, system sleep,
  swap thrashing on the prior 8 GB hardware envelope)
