# Round 3 — scope, branches, and executable plan

**Decision** (recorded in `round_3_planning.md`): no single primary
model. Branches **A + B + C** run as parallel experiments; branch D
(new model) is deferred. The unifying question: *can prompt
engineering move each finalist on its individual weakest axis without
harming the others?*

Round-3 evaluation holds **base model + orchestrator constant** (per
the `BENCHMARKS.md` methodology invariants) and varies only the
**prompt-mediated capability** layer. Any movement attributable to
prompts alone is the round-3 finding; absence of movement is also a
valid finding ("the weakness is in the base model, not the prompt").

## Prerequisite — runner support for prompt variants (~30 LOC)

All three branches need a way to swap `BFCL_SYSTEM_PROMPT` per-run.
Required code change before launching anything:

1. Add a `--bfcl-system-prompt {v2,v3a,v3b,v3c,v2_fewshot_parallel}`
   flag to `scripts/run_bakeoff.py`.
2. Wire it through `RunRequest.bfcl_system_prompt: str = "v2"`.
3. In `benchmarks/bfcl/adapter.py`, a `_BFCL_SYSTEM_PROMPTS: dict[str,
   str]` registry. `_problem_messages` reads
   `req.bfcl_system_prompt` and selects the variant string.
4. `metadata.json` `mode` block records the variant.

Round 3 cannot execute without this. ~30 minutes of code + tests
before any compute.

## Branch A — qwen25-1.5b · irrelevance discipline

**Hypothesis**: prompt engineering can recover 30–50% of qwen25-1.5b's
irrelevance over-call gap (64/64 of curated irrelevance failures =
100% are `over_called_when_irrelevant`) without harming the other 10
categories.

**Variants**:

- **v3a — stronger imperative**:
  ```
  ... (existing rule 1) ...
  - You MUST NOT call any tool unless at least one available tool
    fully satisfies the user's request. If unsure, do not call.
    Answer in plain text.
  ... (existing rule 3) ...
  ```

- **v3b — decision tree**:
  ```
  ... (existing rule 1) ...
  - Before any tool call, ask yourself: does any single available
    tool fully satisfy the user's request? If yes, call it. If no,
    do not call any tool. Answer in plain text.
  ... (existing rule 3) ...
  ```

**Command**:
```bash
for variant in v3a v3b; do
  uv run python scripts/run_bakeoff.py --models qwen25-1.5b-instruct \
    --benchmarks bfcl --rep 6 --bfcl-limit 150 \
    --bfcl-categories simple_python multiple parallel parallel_multiple irrelevance \
                      live_simple live_multiple live_parallel live_parallel_multiple \
                      live_irrelevance live_relevance \
    --bfcl-system-prompt $variant --auto-port --force
done
```

Note: runs the full 11 BFCL categories (not just irrelevance) so the
regression gate is measurable. `--rep 6` for v3a; `--rep 7` for v3b.

**Gates**:

| metric | gate | source |
|---|---|---|
| irrelevance over-call reduction | **≥30%** (from 64 fails → ≤45) | failure-mode count on curated irrelevance |
| BFCL raw overall delta vs `rep_1` | **≥0pp** (no harm) | `graded_report.md` BFCL totals |
| any other category drop | **≤1pp** | per-cat table |
| token-cost vs `rep_1` | **≤10%** | summary.json completion_tokens |

**Abort condition**: any category drops ≥3pp at v3a → skip v3b.

**Wall**: ~30 min (1106 problems × 2 variants × 1 model).

## Branch B — qwen25-coder · parallel-collapse rescue

**Hypothesis**: few-shot prompting can match agent-mode's parallel-
recovery (+52 problems vs `rep_1`) at raw-mode token cost (1.2× vs
4.8×).

**Variant — `v2_fewshot_parallel`**:

Append 2 in-context examples to the v2 system prompt:

```
... (existing v2 rules 1, 2, 3) ...

Example 1 — parallel:
User: Get the weather in Paris, Tokyo, and New York.
Correct: Three separate tool calls:
  get_weather(city="Paris")
  get_weather(city="Tokyo")
  get_weather(city="New York")
NOT a single call with city=["Paris","Tokyo","New York"].

Example 2 — parallel_multiple:
User: Convert 100 USD to EUR and find me a hotel in Berlin.
Correct: Two separate tool calls, one per intent:
  currency_convert(amount=100, from="USD", to="EUR")
  hotel_search(city="Berlin")
```

**Command**:
```bash
uv run python scripts/run_bakeoff.py --models qwen25-coder-1.5b-instruct \
  --benchmarks bfcl --rep 6 --bfcl-limit 150 \
  --bfcl-categories simple_python multiple parallel parallel_multiple irrelevance \
                    live_simple live_multiple live_parallel live_parallel_multiple \
                    live_irrelevance live_relevance \
  --bfcl-system-prompt v2_fewshot_parallel --auto-port --force
```

Full 11 categories so regression on non-parallel can be measured.

**Gates**:

| metric | gate | source |
|---|---|---|
| `parallel` + `parallel_multiple` recovery | **≥25 problems vs `rep_1`** | per-cat pass counts (current: 32+42=74; target ≥99) |
| token multiplier vs `rep_1` | **≤1.2×** | summary.json |
| non-parallel regression (simple, multiple, irrelevance, live_*) | **≤1pp avg** | per-cat |
| compare to `rep_4` (agent mode) | report the delta vs agent's +52 | analytical, not a gate |

**Abort condition**: parallel pass rate worsens vs `rep_1`.

**Wall**: ~15 min (1106 problems × 1 variant × 1 model).

This is the cleanest experiment of the three — single mechanistic
failure, cheap intervention, falsifiable gate, well-understood
expected mechanism (in-context examples shift the next-token
distribution on the "parallel" pattern).

## Branch C — granite33 · decline-boundary calibration

**Hypothesis**: granite's decline boundary is over-tuned for
irrelevance, causing `multi_turn_miss_func` to fail 74/100 with
`empty_turn_model_response` when an alternative tool exists. A prompt
addendum can shift the boundary at a measurable irrelevance cost.

**Framing — this is a *trade-off measurement*, not a one-sided
improvement.** A valid round-3 outcome is "the prompt cannot move
the boundary without sacrificing irrelevance" — that finding closes
the question with data rather than speculation.

**Variant — v3c "use best available, decline only when off-surface"**:

```
... (existing rule 1) ...
- If at least one available tool reasonably satisfies the user's
  request, call it — prefer the best-matching tool even if no tool
  is a perfect fit. Only decline if the user's request is
  fundamentally outside the available tool surface (no tool
  approaches the right shape).
... (existing rule 3) ...
```

**Command**:
```bash
uv run python scripts/run_bakeoff.py --models granite33-2b-instruct \
  --benchmarks bfcl --rep 6 --bfcl-limit 100 \
  --bfcl-categories irrelevance live_irrelevance multi_turn_miss_func \
                    multi_turn_base multi_turn_miss_param \
  --bfcl-system-prompt v3c --auto-port --force
```

Runs irrelevance categories (regression target) + miss_func (movement
target) + base/miss_param (collateral effect on other multi-turn
categories — granite's slight multi-turn edge could shift).

**Gates** (intentionally a trade-off):

| metric | gate | source |
|---|---|---|
| `multi_turn_miss_func` `empty_response` reduction | **≥20%** (from 74 → ≤59) | failure-mode counts |
| irrelevance + live_irrelevance combined harm | **≤5pp drop** | per-cat |
| `live_irrelevance` CI lower bound | **stays ≥85%** (currently 89%) | Wilson CI |
| `multi_turn_base` / `miss_param` collateral | report; no gate | analytical |

**Abort condition**: `live_irrelevance` drops below 90% (loses the
big CI lower-bound advantage).

**Wall**: ~20 min (300 problems × 1 variant × 1 model — multi-turn
is slower per problem than single-turn).

## Combined-round expected wall

| step | wall |
|---|---|
| prereq runner change + tests | ~30 min code |
| Branch A (×2 variants, full 11 cats) | ~30 min compute |
| Branch B (×1 variant, full 11 cats) | ~15 min compute |
| Branch C (×1 variant, 5 cats including 3 multi-turn) | ~20 min compute |
| **total compute** if run sequentially | **~65 min** |
| **total compute** if parallelized across 3 models | **~30 min** |

Parallelize via `--auto-port` exactly as in rep_1/rep_4/rep_5
multi-model runs. Wall budget unchanged from the heaviest single
branch.

## Reporting

After round 3 executes:

1. Grade rep_6 (and rep_7 for branch A's second variant) with
   `--write-back` (multi-turn audit uses the subprocess-isolated
   regrade pattern from `scripts/audit_one_multi_turn.py`).
2. Add a **Round 3 — prompt-engineering experiments** section to
   `graded_report.md` between the existing rep_5 multi-turn section
   and the stochastic-notes section. Per branch: hypothesis +
   gate-pass/fail + delta-vs-rep_1.
3. Add a per-branch failure-mode breakdown to
   `graded_failure_modes.md` — specifically: did v3a/v3b move the
   `over_called_when_irrelevant` bucket? did v2_fewshot move the
   `under_called_1_of_N` bucket? did v3c move the `empty_response`
   bucket on `miss_func`?

## Verification (round-3 ship gates)

| check | target |
|---|---|
| runner accepts `--bfcl-system-prompt` and threads through to adapter | unit test + smoke |
| `metadata.json` records the prompt variant used | smoke |
| each branch's gates evaluated explicitly in the report (pass / fail) | report discipline |
| `rep_6` (and `rep_7` if applicable) data committed; audit clean | mirror rep_5 audit pattern |
| `689+/689+` tests still green after the runner change | regression |

## Appendix — branch D (considered, not pursued this round)

| field | value |
|---|---|
| Branch | D (model-comparison disruptor) |
| Candidates | `gemma-2-2b-it`, `llama-3.2-3b-instruct`, `smollm3-3b-instruct` |
| Rationale for deferral | Round 3 probes the existing finalists' headroom under prompt engineering. Adding a fourth model dilutes that question and consumes 3–4 h of compute that the prompt-engineering branches don't need. |
| When to revisit | Round 4 or later, if A+B+C show the existing triangle has limited prompt-mediated headroom. A truly different base model could shift the ceiling. |
