# Round 2 — champion bake-off (multi-spectrum, refreshed 2026-05-13)

Finalists from round 1:

- **qwen25-1.5b-instruct** — Alibaba instruct, 1.5B params, Q8_0
- **qwen25-coder-1.5b-instruct** — coder-tuned sibling
- **granite33-2b-instruct** — IBM Granite 3.3, 2B params, Q8_0

All BFCL data uses the v2 `BFCL_SYSTEM_PROMPT` (see
`benchmarks/bfcl/adapter.py`). Five round-1 models are cut; see appendix.

**Hardware note.** All `rep_*` data in this edition was re-run on a
128 GB M5 Max — the original 8 GB-Mac numbers are superseded. Wall
times are therefore not comparable across editions and are not
reported here.

## TL;DR

| model | BFCL deep (n=1106) | HumanEval pass@1 (t=0.0) | HE pass@3 (any temp) | MBPP pass@1 (t=0.0, n=427) |
|---|---|---|---|---|
| qwen25-1.5b-instruct | **77.0% (853/1106) ±2.5pp** | 58.5% (96/164) | 64.0% (105/164) | 62.4% (267/427) ±4.6pp |
| granite33-2b-instruct | 69.4% (768/1106) ±2.7pp | 53.0% (87/164) | 64.6% (106/164) | 59.6% (255/427) ±4.6pp |
| qwen25-coder-1.5b-instruct | 58.6% (649/1106) ±2.9pp | **69.5% (114/164)** | **78.0% (128/164)** | **64.0% (274/427) ±4.5pp** |

The three are in a non-dominated triangle: qwen25-1.5b wins tool-use,
qwen25-coder wins coding, granite33 wins irrelevance discipline. The
~8pp gap on BFCL between the top two is well outside CI overlap — that's
a real win, not noise. The coding gap between qwen25-coder and the rest
is even larger (14pp at t=0.0, outside the ~7.5pp CI half-width).

## BFCL — full leaderboard, with 95% Wilson CIs

n=150 per curated category (5 categories × 3 finalists × 150 = 2,250
problems); n=100 per live category where available (capped: live_parallel
16, live_parallel_multiple 24, live_relevance 16 → 356 per finalist
across 6 live categories). 1,106 total BFCL problems per finalist.

### Curated (BFCL v4 non-live, 750 / model)

| category | n | qwen25-1.5b | granite33 | qwen25-coder |
|---|---|---|---|---|
| simple_python | 150 | **91.0%** ±4.4 | 83.2% ±5.9 | **91.0%** ±4.4 |
| multiple | 150 | **87.7%** ±5.1 | 67.6% ±7.4 | 87.1% ±5.2 |
| parallel | 150 | **78.0%** ±6.5 | 61.7% ±7.7 | 22.0% ±6.5 |
| parallel_multiple | 150 | **72.8%** ±7.0 | 54.6% ±7.9 | 28.5% ±7.1 |
| irrelevance | 150 | 57.2% ±7.8 | **85.1%** ±5.6 | 52.0% ±7.9 |
| **curated total** | 750 | **78.0% (585/750) ±2.9** | 70.9% (532/750) ±3.3 | 56.3% (422/750) ±3.5 |

### Live (BFCL v4 user-submitted, 356 / model)

| category | n | qwen25-1.5b | granite33 | qwen25-coder |
|---|---|---|---|---|
| live_simple | 100 | **80.8%** ±7.5 | 67.3% ±9.0 | 69.3% ±8.8 |
| live_multiple | 100 | **70.2%** ±8.8 | 48.1% ±9.6 | 61.6% ±9.3 |
| live_parallel | 16 | **60.1%** ±21.4 | 29.8% ±19.7 | 24.8% ±18.2 |
| live_parallel_multiple | 24 | **50.0%** ±18.6 | 39.2% ±18.1 | 14.1% ±11.8 |
| live_irrelevance | 100 | 77.0% ±8.0 | **95.3%** ±3.7 | 74.1% ±8.4 |
| live_relevance | 16 | 85.3% ±13.6 | 60.1% ±21.4 | 85.3% ±13.6 |
| **live total** | 356 | **75.3% (268/356) ±4.4** | 66.3% (236/356) ±4.9 | 63.8% (227/356) ±5.0 |

Interpretation:
- **qwen25-1.5b** is the dominant generalist — top on 8 of 11 categories,
  comfortably ahead on the parallel categories (still 78% / 73% / 60% / 50%
  at n=150 / n=100 / n=16 / n=24).
- **granite33** dominates irrelevance: **97/100 on live_irrelevance**
  (95.3% ±3.7pp, lower bound 89%) plus 85% on curated irrelevance. Its
  decline-discipline is the strongest signal in the data and worth ~30pp
  vs the next finalist on that axis.
- **qwen25-coder** is no better than the others on tool-use overall; its
  parallel collapse is even more pronounced in this run (22% / 28.5% on
  curated parallel/parallel_multiple, vs the 38.7% / 29.3% in the prior
  edition — see "Stochastic notes" below). It does outperform granite
  on live_simple/live_multiple/live_relevance, indicating it's
  competitive on single-call workloads.

## HumanEval — temperature sweep (164 problems × 3 temps)

| model | t=0.0 | t=0.3 | t=0.7 | pass-all-3 | pass-any |
|---|---|---|---|---|---|
| qwen25-coder-1.5b | **69.5%** (114) | 66.5% (109) | 65.2% (107) | **56.1%** (92) | **78.0%** (128) |
| qwen25-1.5b | 58.5% (96) | 56.7% (93) | 48.8% (80) | 42.7% (70) | 64.0% (105) |
| granite33-2b | 53.0% (87) | **53.7%** (88) | 51.8% (85) | 41.5% (68) | 64.6% (106) |

CI half-width at n=164, p~0.6 is about ±7.5pp. So:
- The qwen25-coder lead over the other two on HumanEval is real
  (>10pp at every temperature). Outside CI overlap.
- qwen25-1.5b vs granite33 is within CI at every temperature — they're
  statistically tied on coding (the t=0.0 gap narrowed to 5.5pp in this
  edition, well inside the CI overlap).

Temperature shape:
- **qwen25-coder degrades monotonically** (69.5 → 66.5 → 65.2). Loses
  7 problems going t=0.0 → t=0.7 — most temperature-stable model in
  absolute terms.
- **qwen25-1.5b** peaks at t=0.0 (58.5%) in this edition rather than
  the inverted-V the prior edition showed. Still falls 10pp at t=0.7.
- **granite33** shows a tiny inverted-V (53.0 → 53.7 → 51.8), but it's
  effectively flat within noise. The most t=0.7-stable model.
- **pass-all-3 vs pass-any** spread (the model's coding uncertainty
  window): qwen25-coder = 36, qwen25-1.5b = 35, granite33 = 38. All
  similar at this draw — best-of-3 is worth ~22pp over pass@1 for each.

## MBPP — sanitized split, n=427 (rep_0 t=0.0)

Second coding signal beyond HumanEval. Sanitized split (427 problems)
vendored at `benchmarks/mbpp/MBPP.jsonl`. Per-problem subprocess
isolation; the adapter normalizes model output (markdown-fence + first-
def anchor + main-guard drop) before execution.

| model | pass@1 | passed | 95% Wilson CI | wall | comp tokens |
|---|---|---|---|---|---|
| qwen25-coder-1.5b-instruct | **64.0%** | 274/427 | ±4.5pp | 4:01 | 18,985 |
| qwen25-1.5b-instruct | 62.4% | 267/427 | ±4.6pp | 4:48 | 25,150 |
| granite33-2b-instruct | 59.6% | 255/427 | ±4.6pp | 6:44 | 33,275 |

### MBPP head-to-head matrix (PPP / FPF / etc.)

| outcome | qwen-1.5b / coder / granite | count |
|---|---|---|
| all pass | P P P | 205 (48%) |
| all fail | F F F | 106 (25%) |
| qwen + coder, not granite | P P F | 32 (7%) |
| only coder | F P F | 21 (5%) |
| qwen + granite, not coder | P F P | 17 (4%) |
| only granite | F F P | 17 (4%) |
| coder + granite, not qwen | F P P | 16 (4%) |
| only qwen | P F F | 13 (3%) |

### Interpretation (MBPP vs HumanEval)

MBPP and HumanEval measure overlapping but **non-equivalent** coding
skills. Treat the two numbers as orthogonal, not redundant:

- **HumanEval** rewards synthesis, longer dependency chains, and
  edge-case handling. Function signatures are given; the model fills
  in the body.
- **MBPP (sanitized)** rewards short-template completion, memorized
  educational patterns, and shallow reasoning. The model produces a
  fresh function from a natural-language spec + a single test hint.

What the numbers show this draw:
- **qwen25-coder** wins both, but its lead is much smaller on MBPP
  (+1.7pp over qwen-1.5b vs +11pp on HumanEval). Coder-tuned models
  saturate on MBPP-shaped problems sooner — the synthesis ceiling
  HumanEval probes is where the gap widens.
- **granite33 and qwen-1.5b each gain ~+6pp going HE → MBPP** (53 → 60,
  58 → 62 respectively). Both benefit from MBPP's shorter problem
  template. Granite specifically uniquely solves 17 MBPP rows the
  other two miss (vs only 5 on HumanEval at t=0.0) — a ~3.5× lift
  in its unique-solve count.
- All three sit within mutual CI overlap on MBPP (±4.5pp at n=427).
  The MBPP ranking is essentially tied; only HumanEval separates the
  coder model from the others.

If qwen-coder widens its lead on MBPP but stays tied on HumanEval —
that didn't happen in this draw, but if it ever does — it would not
be a signal contradiction. Coder models tend to over-fit short-form
coding priors. Read each number for the skill it measures.

## BFCL agent mode — closed-loop orchestration (rep_4)

**Different benchmark from raw BFCL.** Raw measures the model's first-
pass capability (single forward pass under no feedback). Agent measures
the *model + tool-call loop + stub world* — does the orchestration
layer recover (or break) what the model started? When an agent number
exceeds the raw number, the orchestrator recovered some mistakes;
when it falls below, the loop added new ones. Compare paired numbers
on identical inputs, never the two columns separately.

Stub realism ablation (10 problems × 3 stub variants on qwen25-1.5b,
"current"/"opaque-JSON"/"empty-ack") showed **no measurable behavior
shift** (mean tool calls per problem = 1.90 across raw and all three
agent variants). Stuck with the current `[stub:name] called with args=…`
format for parity with the existing scaffold.

### Agent overall — small lift on average, dominated by coder

| model | raw pass | agent pass | Δ | wall raw→agent | tokens raw→agent |
|---|---|---|---|---|---|
| qwen25-1.5b-instruct | 853/1106 (77.0%) | 854/1106 (77.2%) | **+1** | 846s → 1527s (1.8×) | 65k → 119k (1.8×) |
| granite33-2b-instruct | 768/1106 (69.4%) | 776/1106 (70.2%) | **+8** | 1339s → 2316s (1.7×) | 102k → 162k (1.6×) |
| qwen25-coder-1.5b-instruct | 649/1106 (58.6%) | 701/1106 (63.4%) | **+52** | 1029s → 2660s (2.6×) | 87k → 270k (**3.1×**) |

The aggregate disguises a much sharper per-category story.

### Where agent mode helps / hurts (qwen25-coder is the standout)

| model · category | raw → agent | Δ | mean turns | token mult |
|---|---|---|---|---|
| **qwen-coder · parallel** | 32/150 → 83/150 | **+51** | 5.51 | 4.83× |
| **qwen-coder · parallel_multiple** | 42/150 → 93/150 | **+51** | 4.83 | 4.19× |
| qwen-coder · multiple | 132/150 → 101/150 | **−31** | 3.84 | 3.75× |
| qwen-coder · live_multiple | 62/100 → 44/100 | **−18** | 4.22 | 4.50× |
| qwen-coder · live_simple | 70/100 → 65/100 | −5 | 3.46 | 2.93× |
| qwen-coder · live_parallel_multiple | 2/24 → 4/24 | +2 | 7.38 | **7.92×** |
| granite · parallel | 93/150 → 95/150 | +2 | 1.79 | 1.92× |
| granite · parallel_multiple | 82/150 → 84/150 | +2 | 1.79 | 1.90× |
| granite · live_irrelevance | 97/100 → 99/100 | +2 | 1.01 | 1.00× |
| qwen-1.5b · most categories | (flat) | 0 to ±1 | 1.2 – 2.5 | 1.3 – 2.5× |

What the pattern shows:

1. **qwen-coder's parallel collapse is partially recovered by the loop.**
   The raw mode's `under_called_1_of_N` failure (74% of parallel rows)
   is rescued when the loop gives the model another turn to emit the
   missing calls. +51 problems on `parallel` and `parallel_multiple` —
   essentially doubling its raw count.
2. **qwen-coder loses on multiple / live_multiple.** Agent mode pushes
   the model to over-call: when stub results come back, it interprets
   "thanks, here's stub output" as encouragement to try another tool.
   −31 on curated `multiple` (a category that requires *exactly one*
   call) and −18 on `live_multiple`. The orchestrator broke as much as
   it fixed.
3. **qwen25-1.5b doesn't move.** Its first-pass behavior is already
   close to ceiling on most categories, so there's nothing to recover.
   Agent mode pays 1.8× tokens for ±1 problem.
4. **granite33 moves a little, efficiently.** Mean turns 1.0–1.9 — it
   ends conversations quickly. +8 problems for 1.6× tokens. Of the
   three, the most cost-effective in agent mode.

### Overhead is real and asymmetric

If you read pass rate alone, agent looks like a free upgrade for
qwen-coder. With overhead beside it, the picture is different:

| model | raw `passes/M_tokens` | agent `passes/M_tokens` | efficiency Δ |
|---|---|---|---|
| qwen25-1.5b-instruct | 13.1k | 7.2k | **-45%** |
| granite33-2b-instruct | 7.5k | 4.8k | **-36%** |
| qwen25-coder-1.5b-instruct | 7.5k | 2.6k | **-65%** |

Read as: per-million-tokens-of-compute, agent mode is **less efficient**
for every model. Coder pays the steepest price for the biggest absolute
gain. Whether the trade is worth it is a deployment choice — agent mode
is the right model for a deployment with retry budget and latency
tolerance; raw is the right baseline if you're cost-sensitive.

### Why we measure both

Raw BFCL is the public-leaderboard-comparable number — it measures the
**model's** tool-selection capability. Agent BFCL measures the
**system's** capability (model + loop + stub feedback). Both matter
for different deployment patterns. Don't conflate them.

## Multi-turn BFCL — state-based grading (rep_5)

**Different benchmark from raw and agent BFCL.** Multi-turn problems are
4–5 turns of user-driven conversation; each turn the model emits tool
calls, the bfcl_eval mock APIs (stateful Python classes — filesystem,
trading, messaging, etc.) execute them, results feed back, and the
model decides what to do next. **The grader is end-state comparison**:
bfcl_eval runs the model's call sequence and the ground-truth sequence
against fresh mock-API instances and checks whether the resulting state
matches turn-by-turn. Strict aggregation — any turn mismatching fails
the whole problem. No partial credit.

n=100 / category × 4 categories = 400 problems per finalist.

### Overall (rep_5, n=400 each)

| model | passed | overall | model_behavior failures | infrastructure failures |
|---|---|---|---|---|
| granite33-2b-instruct | **4/400** | 1.5% ±1.1pp (Wilson 95%) | 362 | 34 |
| qwen25-coder-1.5b-instruct | 2/400 | 1.0% ±0.8pp | 344 | 54 |
| qwen25-1.5b-instruct | 0/400 | 0.5% ±0.5pp | 389 | 11 |

For context: the public BFCL multi-turn leaderboard caps around 30–50%
for GPT-4-class models. Open-weight 1.5B–2B models reliably land near 0–
2%. **These numbers are consistent with the leaderboard pattern**:
state-tracking across 4–5 conversational turns demands capability that
emerges at scales an order of magnitude above what we're testing.

The headline rate (≤1.5%) is mostly meaningful as a *bottom-of-curve*
measurement: it tells you which model marginally edges the others, and
where the infrastructure failure modes concentrate.

### Per-category — n=100 each, Wilson 95%

| category | qwen25-1.5b | qwen25-coder | granite33 |
|---|---|---|---|
| multi_turn_base | 0/100 (1.8% ±1.8) | 0/100 (1.8% ±1.8) | **1/100** (2.8% ±2.6) |
| multi_turn_long_context | 0/100 | 0/100 | **1/100** (2.8% ±2.6) |
| multi_turn_miss_func | 0/100 | **1/100** (2.8% ±2.6) | 0/100 |
| multi_turn_miss_param | 0/100 | 1/100 | **2/100** (3.8% ±3.2) |

All cells overlap within CI. There is no statistically significant
separation between the three models on multi-turn at this sample size.

### Context utilization on multi_turn_long_context (n_ctx=8192)

`prompt_tokens_at_turn_end` in the persisted trace is **cumulative
across all chat calls in the conversation**, not per-call peak. (A
follow-up runner change to record per-step prompt sizes would give
direct per-call truncation visibility; today we infer from cumulative
deltas + the explicit backend 400 logs.)

| model | cumulative tokens at final turn — p50 / p95 / max | backend 400s (context overruns) |
|---|---|---|
| qwen25-1.5b-instruct | 25,305 / 58,841 / 66,418 | 11 / 100 |
| granite33-2b-instruct | 22,845 / 64,254 / 117,520 | 30 / 100 |
| qwen25-coder-1.5b-instruct | **62,868 / 134,811 / 186,876** | **49 / 100** |

qwen25-coder is dramatically more verbose in multi-turn — its cumulative
prompt budget at p95 is 2×–3× what the other two models consume on the
same conversations. That verbosity is what generates 49 backend-rejected
calls on the long_context category — half of coder's long_context
problems hit 400 errors at least once during their conversations. Those
problems land in the `infrastructure` bucket of the failure breakdown,
not in the model-capability budget.

**Practical implication**: if you want multi-turn agent-loop deployment
on this hardware (n_ctx=8192), qwen25-coder is the worst pick of the
three. Its per-turn verbosity blows past the context window the fastest
even when the underlying behavior is on the right track.

## Round 3 — prompt-engineering experiments (rep_6)

Three system-prompt variants, one per finalist, each targeting that
model's worst axis from round 2. Base weights + orchestrator + sampler
held constant; only the BFCL system prompt changes. All three branches'
primary gates **failed** — the prompt-mediated capability layer cannot
shift the round-2 weaknesses on these finalists at this quant.

### Branch A — qwen25-1.5b · v3a (stronger imperative)

**Hypothesis**: a "MUST NOT call unless fully satisfies" imperative
tightens the decline boundary on `irrelevance` over-call without
harming legitimate live single-call categories.

> **2026-05-14 audit note** — Originally published with a
> lexicographic-sort slice. Corrected matched-ID numbers below;
> matched-ID artifact at `acceptance/audits/branch_a_matched_ids.json`.

**Verdict shift**: hypothesis A **still falsified** on the target gate
(irrelevance regressed -6pp same as published), but the magnitude on
two collateral cats was overstated by the buggy slice.

| metric | original (published) | corrected (matched-ID) | verdict |
|---|---|---|---|
| irrelevance | -6.0pp (57.3% → 51.3%) | -6.0pp (same, both reps at n=150) | unchanged |
| live_simple | **-15.0pp** (82.0% → 67.0%) | **-2.0pp** (82.0% → 80.0%) | **withdrawn — slicing artifact** |
| live_multiple | -4.0pp (71.0% → 67.0%) | -3.0pp (71.0% → 68.0%) | weakened (within noise) |
| live_irrelevance | **-23.0pp** (78.0% → 55.0%) | **-15.0pp** (78.0% → 63.0%) | weakened, still material |
| OVERALL | -4.8pp (77.1% → 72.3%) | -3.0pp (77.7% → 74.7%) | weakened |

Per-category (matched-ID, rep_1 ∩ rep_6 IDs):

| cat | rep_1 | rep_6 v3a | Δpp |
|---|---|---|---|
| simple_python | 92.0% | 91.3% | -0.7 |
| multiple | 88.7% | 89.3% | +0.7 |
| parallel | 78.7% | 77.3% | -1.3 |
| parallel_multiple | 73.3% | 72.7% | -0.7 |
| irrelevance | 57.3% | 51.3% | **-6.0** ← target regressed |
| live_simple | 82.0% | 80.0% | -2.0 (was -15.0; bug-corrected) |
| live_multiple | 71.0% | 68.0% | -3.0 |
| live_irrelevance | 78.0% | 63.0% | **-15.0** ← collateral, real |
| OVERALL (n=1050) | 77.7% | 74.7% | **-3.0** |

The imperative did move the decline boundary on `live_irrelevance` —
the model now over-declines on ambiguous user requests (-15pp). The
target gate also fires (irrelevance -6pp, over-call bucket 64 → 73).
But the "regressed across the board" framing in the original write-up
does not survive — `live_simple` was unaffected (-2pp is within noise),
not a -15pp collateral. **Hypothesis A is still falsified, with one
fewer piece of supporting evidence and overall magnitude shrunk by
~1.8pp.**

### Branch B — qwen25-coder · v2_fewshot_parallel

**Hypothesis**: two in-context parallel examples close qwen25-coder's
under_called_1_of_N gap on `parallel`/`parallel_multiple` — recovering
agent-mode's +52 parallel-recovery (rep_4) at raw-mode token cost.

> **2026-05-14 audit note** — Branch B's primary gate is on curated
> parallel cats where both rep_1 and rep_6 ran 150 problems with
> matching IDs, so the slicing bug did **not** affect this branch's
> verdict. Matched-ID recount confirmed identical numbers. Matched-ID
> artifact at `acceptance/audits/branch_b_matched_ids.json`.

**Verdict shift**: none. Hypothesis B remains falsified on identical
numbers.

| metric | original (published) | corrected (matched-ID) | verdict |
|---|---|---|---|
| parallel + parallel_multiple recovery | -13 problems (74 → 61) | -13 (same) | unchanged |
| parallel | 32/150 → 25/150 | same | unchanged |
| parallel_multiple | 42/150 → 36/150 | same | unchanged |

**Result**: combined parallel cats moved **-13 problems** vs the +25
required; few-shot transferred in the wrong direction.

| gate | required | observed | verdict |
|---|---|---|---|
| parallel + parallel_multiple recovery | ≥+25 problems | **-13** (74 → 61) | FAIL |
| completion-token mult on parallel cats | ≤1.2× | 1.12× / 1.13× | PASS |
| non-parallel regression (5-cat mean) | ≤-1pp | +1.80pp | PASS |

Per-category:

| cat | rep_1 | rep_6 v2_fewshot | Δ |
|---|---|---|---|
| parallel | 32/150 | 25/150 | -7 |
| parallel_multiple | 42/150 | 36/150 | -6 |

Failure-bucket movement on `parallel` is the diagnostic: `under_called_1_of_N`
fell 111 → 102 (-9, marginal), but `emitted_0_calls_expected_N` (model
emitted nothing) jumped 0 → 20 (+20). The few-shot examples didn't
teach the model to emit *more* calls — they pushed 20 problems into
outright silence. Likely a prompt-length / instruction-confusion
effect at this size. Hypothesis B falsified.

Note: matched-ID recount also surfaced an unrelated **+17pp jump on
live_irrelevance** (75/100 → 92/100) — the few-shot examples did
sharpen qwen-coder's decline discipline on user-submitted irrelevance
prompts. This wasn't a gate and doesn't change the verdict, but it's
a real side-effect worth knowing about for any future parallel-recovery
prompting attempt on this model.

### Branch C — granite33-2b · v3c (decline-boundary calibration)

**Hypothesis**: a prompt addendum loosens the decline boundary on
`multi_turn_miss_func` — converting `empty_turn_model_response`
problems (baseline 74/100) into alternative-tool attempts. Framed as a
trade-off measurement, not one-sided improvement; collateral on
irrelevance is the question.

> **2026-05-14 audit note** — Originally published with a
> lexicographic-sort slice on the irrelevance comparison (rep_1 had
> 150 problems, rep_6 had 100; the published "first 100" of rep_1 was
> the lex-mangled subset, not the proper rep_1 ∩ rep_6 intersection).
> Matched-ID artifact at `acceptance/audits/branch_c_matched_ids.json`.

**Verdict shift**: primary gate still fails (unchanged). The **collateral
gate that originally PASSED now FAILS on matched data** — the irrelevance
harm was understated by 3pp because of the slicing bug.

| metric | original (published) | corrected (matched-ID) | verdict |
|---|---|---|---|
| mt_miss_func empty_response reduction (primary) | -4.1% (74 → 71) | same — matched-ID multi-turn rep_5 ∩ rep_6 confirms 71/100 | unchanged — primary gate FAILS |
| irrelevance (rep_1 → rep_6) | -5pp (85/100 → 80/100) | **-8pp (88/100 → 80/100)** | corrected baseline; harm larger than published |
| live_irrelevance (rep_1 → rep_6) | +1pp (97/100 → 98/100) | +1pp (same — both reps 100 IDs, full match) | unchanged |
| **irrelevance + live_irrelevance combined harm gate** | -4pp combined; **PASS** (≤5pp budget) | **-7pp combined; FAIL** (exceeds 5pp budget) | **reversed — collateral gate now fails** |

Per-category (matched-ID intersection):

| cat | baseline | rep_6 v3c | Δ |
|---|---|---|---|
| mt_miss_func `empty_response` | 74/100 (rep_5) | 71/100 | -3 |
| mt_miss_func `state_mismatch` | 14/100 (rep_5) | 17/100 | +3 |
| mt_miss_func PASS | 0/100 | 0/100 | 0 |
| irrelevance (matched 100) | **88/100** (rep_1) | 80/100 | **-8** (was published -5) |
| live_irrelevance | 97/100 (rep_1) | 98/100 | +1 |
| multi_turn_base (collateral) | 1/100 (rep_5) | 2/100 | +1 |
| multi_turn_miss_param (collateral) | 2/100 (rep_5) | 2/100 | 0 |

The boundary did move — granite traded exactly 3 empty responses for 3
state-mismatch failures on the target — but no movement converted to
passes: the calls the model now emits on `miss_func` are the wrong tool
in the wrong order. The original write-up framed the irrelevance harm
as "within budget (combined -4pp), so the trade isn't catastrophic,
just inefficient." On matched data the combined harm is -7pp, **above
the -5pp budget the gate set in advance**. The trade is **catastrophic
by the pre-registered criterion** — not just inefficient.

Hypothesis C is now more thoroughly disconfirmed: the decline boundary
is movable, but moving it costs ~8pp on curated irrelevance for no
target-gate gain. The alternative-tool-selection ceiling is still the
deeper finding; the boundary-calibration trade-off **also** fails on
the harm dimension that the original write-up reported as acceptable.

### Round 3 take-away

Three orthogonal hypotheses, three falsified primary gates. The
prompt-mediated capability layer — at least via the variants tested —
cannot shift any of the round-2 weaknesses on the finalists' base
weights:

- qwen25-1.5b's irrelevance over-call is structural, not promptable
- qwen25-coder's parallel collapse is not a few-shot-recoverable
  pattern at this size
- granite33's `miss_func` decline is a capability ceiling
  (alternative-tool selection), not a boundary-calibration problem

The non-dominated round-2 triangle is unchanged. The data closes round
3 with negative findings rather than a new leader.

**Per-branch verdict summary after 2026-05-14 audit:**

| branch | primary gate | collateral gate(s) | original verdict | corrected verdict | shift class |
|---|---|---|---|---|---|
| A (qwen25-1.5b v3a) | FAIL (-6pp irrelevance) | live_simple -2pp (was published -15pp), live_irrelevance -15pp (was published -23pp) | falsified | falsified | **weakened** — magnitudes smaller; live_simple regression was slicing artifact |
| B (qwen-coder fewshot) | FAIL (-13 problems on parallel cats) | non-parallel mean PASS | falsified | falsified | **unchanged** — curated cats unaffected by bug |
| C (granite33 v3c) | FAIL (-4.1% mt_miss_func empty_response reduction) | combined irrelevance harm published as PASS (-4pp); matched-ID shows -7pp | partially disconfirmed; "trade isn't catastrophic" | more thoroughly disconfirmed; collateral gate also fails | **reversed** on collateral gate — trade exceeds the pre-registered 5pp harm budget |

Overall: the round-3 falsification holds in direction but on cleaner
evidence. Branch A's supporting "collateral damage on legitimate user
requests" softened (live_simple was unaffected); Branch C's verdict
hardened (the harm budget was exceeded, not within it).

## Round 4 — Branch D model-comparison disruptor

Round-3 design doc deferred Branch D ("a truly different base model
could shift the ceiling") pending the outcome of A+B+C. With all three
prompt branches falsified, the deferral condition fired and three new
≤3B disruptor candidates were run through the **headline triangle**
data only (BFCL curated rep_1 + HumanEval rep_0 + MBPP rep_0):
`gemma-2-2b-it`, `llama-3.2-3b-instruct`, `smollm3-3b-instruct`. Live
BFCL, agent mode, multi-turn, and the HumanEval temperature sweep are
deferred pending an interesting headline.

### Headline (apples-to-apples curated cats, 750 / model)

| model | BFCL curated | HumanEval | MBPP | BFCL mode | rank shift |
|---|---|---|---|---|---|
| **smollm3-3b-instruct** | **601/750 (80.1%)** | 105/164 (64.0%) | 261/427 (61.1%) | inject | new #1 on BFCL? |
| qwen25-1.5b-instruct | 585/750 (78.0%) | 96/164 (58.5%) | 267/427 (62.5%) | structured | prior BFCL #1 |
| granite33-2b-instruct | 532/750 (70.9%) | 87/164 (53.0%) | 255/427 (59.7%) | structured | — |
| qwen25-coder-1.5b-instruct | 422/750 (56.3%) | **114/164 (69.5%)** | **274/427 (64.2%)** | structured | HE/MBPP #1 |
| llama32-3b-instruct | 393/750 (52.4%) | 95/164 (57.9%) | 265/427 (62.1%) | auto (struct) | strictly dominated |
| gemma2-2b-it | 364/750 (48.5%) | 73/164 (44.5%) | 211/427 (49.4%) | inject | strictly dominated |

### Wilson 95% CIs on the smollm3 vs prior-champions comparison

| comparison | smollm3 | prior | overlap? |
|---|---|---|---|
| BFCL curated (n=750) | 80.1% [77.1, 82.8] | qwen25-1.5b 78.0% [74.9, 80.8] | **YES** |
| HumanEval pass@1 (n=164) | 64.0% [56.4, 71.0] | qwen25-coder 69.5% [62.1, 76.0] | **YES** |
| MBPP pass@1 (n=427) | 61.1% [56.4, 65.6] | qwen25-coder 64.2% [59.5, 68.6] | **YES** |
| parallel (n=150) | 83.3% [76.6, 88.4] | qwen25-1.5b 78.7% [71.4, 84.5] | **YES** |

CIs overlap on every comparison. smollm3 is **statistically tied** with
the prior champions on each headline axis, not a clean dethroner. The
+2.1pp BFCL curated edge and -5.5pp HumanEval gap are within the
multi-pp wobble band documented in "Stochastic notes."

### Per-model verdict

**gemma2-2b-it** (49% BFCL / 44.5% HE / 49.4% MBPP) — **strictly
dominated** on every axis by every existing finalist. Gemma 2's chat
template has no native function-calling support; in structured mode
the model emits Python-like pseudocalls (`area(10, 5)`) that no parser
recognizes. Inject mode rescues the call format but the underlying
weights still trail the field. Coding numbers (HE 44.5%, MBPP 49.4%)
are worst-on-board. Confirm dropped.

**llama-3.2-3b-instruct** (52% BFCL / 58% HE / 62% MBPP) — middling.
Strong on parallel/parallel_multiple curated cats (89/82) but
catastrophic on `irrelevance` (31/150 = 21% pass, 119/150 over-calls).
Coding is middle-of-pack between qwen25-1.5b and qwen25-coder, doesn't
beat either. The 3B parameter count buys nothing here over the 1.5B
finalists. Strictly dominated by qwen25-1.5b on every axis except
parallel/parallel_multiple, where it matches but doesn't exceed.
Confirm dropped.

**smollm3-3b-instruct** (80% BFCL / 64% HE / 61% MBPP) — **the only
candidate worth a follow-up**. Tied with qwen25-1.5b at the top of
BFCL curated within CIs, second on both HumanEval (behind qwen25-
coder) and MBPP. Notably it does *not* collapse on parallel cats the
way qwen25-coder does — parallel 125/150 (83%) and par_mul 116/150
(77%) are the highest non-qwen25-1.5b numbers on either axis. Its
irrelevance behavior (86/150, 57%) is identical to qwen25-1.5b — same
over-call bucket, same volume.

### Apples-to-apples caveat: BFCL mode

smollm3 and gemma2 ran in `--bfcl-mode inject` (no native tool
template, BFCL prompt-injected); the three existing finalists and
llama32-3b ran in structured mode. Each model used the best mode it
supports — methodology-consistent — but the comparison is "model +
adapter mode" not "model alone." If a future run wants to make smollm3
the new BFCL champion, the cleanest follow-up is to re-baseline the
existing finalists in inject mode too (or accept that "best mode each"
is the operational comparison that matters).

### Triangle status after Branch D

| corner | prior holder | Branch D challenger | verdict |
|---|---|---|---|
| **Tool-use generalist (BFCL)** | qwen25-1.5b (77%) | smollm3 (80% curated, no live data) | **tied within CI**, smollm3 untested on live cats |
| **Synthesis coding (HumanEval)** | qwen25-coder (69.5% / 78% pass@3) | smollm3 (64.0% pass@1, no temp sweep) | qwen25-coder holds; smollm3 inside CI but no pass@3 yet |
| **Decline discipline (live_irrelevance)** | granite33 (97%) | none (no Branch D live data) | granite33 unchallenged |

The non-dominated triangle is **not redrawn** by Branch D headline
data — but it's slightly *less stable* on the tool-use corner, where
smollm3 sits within the CI of qwen25-1.5b and dominates qwen25-1.5b on
the two parallel categories (+7 / +6 problems).

### Recommendation — expand-or-close

Two options for the next session, decision is yours:

- **Expand smollm3 to the full round-2 spectrum** (live BFCL + multi-
  turn + agent + HumanEval temp sweep). Wall ~2h. If smollm3 holds its
  CI parity on live cats and shows a non-floor multi-turn result, it
  earns a spot in a redrawn triangle. If it collapses on live cats
  (which is where qwen25-coder lost ~22pp from curated→live), the
  curated finding doesn't carry.
- **Close**: the headline data is enough. Branch D produced one
  partial near-tie (smollm3) and two strictly-dominated drops
  (gemma2, llama32-3b). The triangle stands; the gemma2/llama32-3b
  results close those candidates definitively.

### Phase H — smollm3 full-spectrum follow-up (rep_1 live + rep_4 + rep_5 + HE rep_2/rep_3)

The Branch D headline-pass section above flagged smollm3 as the only
candidate worth expanding. The expansion ran the four missing data
slices.

> **2026-05-14 audit note** — This section was originally published with
> a lexicographic-sort slicing bug in the cross-model "apples-to-apples"
> table; numbers are corrected below and the original tables are
> reproduced under "Errata and methodology corrections" near the end of
> this document. Source: `scripts/compare_matched_slice.py --rep 1
> --models smollm3-3b-instruct qwen25-1.5b-instruct granite33-2b-instruct
> qwen25-coder-1.5b-instruct`; matched-ID artifact at
> `acceptance/audits/phase_h_n1106_4way_matched_ids.json`.

The data separates into two methodologically distinct questions:

1. **Matched-ID comparison** (subsection 1) — head-to-head, same problem
   IDs across models; valid for "is X better than Y on the same tasks."
2. **Full-distribution comparison** (subsection 2) — smollm3 on the full
   live cats (258/1053/884) vs finalists on the smaller 100-per-cat slice
   they ran. This is an *asymmetric* comparison and tells us about
   distribution-robustness, not head-to-head superiority. Cleaner numbers
   pending Phase K (rep_7 finalist full-live runs).

#### Subsection 1 — Matched-ID comparison (n=1106 intersection)

Per-category overlap_n shown in the helper output; for live cats the
intersection equals the smaller dir's size (finalists' rep_1 ran live
cats with `--bfcl-limit 100`, smollm3 ran no limit). For curated cats
all four models have 150 IDs each (full overlap).

| model | overall | live_simple | live_multiple | live_irrelevance | parallel (curated) |
|---|---|---|---|---|---|
| smollm3-3b-instruct | **860/1106 (77.8%)** [75.2, 80.1] | 73/100 | 65/100 | 89/100 | **125/150** |
| qwen25-1.5b-instruct | 853/1106 (77.1%) [74.6, 79.5] | **82/100** | **71/100** | 78/100 | 118/150 |
| granite33-2b-instruct | 768/1106 (69.4%) [66.7, 72.1] | 68/100 | 48/100 | **97/100** | 93/150 |
| qwen25-coder-1.5b-instruct | 649/1106 (58.7%) [55.8, 61.5] | 70/100 | 62/100 | 75/100 | 32/150 |

smollm3 and qwen25-1.5b are statistically tied — CIs overlap from 75.2
to 79.5 (a ~4.3pp shared band). smollm3's mean is +0.7pp above qwen25-1.5b,
not below.

Per-cat matched deltas, smollm3 vs qwen25-1.5b:

| cat | smollm3 | qwen25-1.5b | Δ | CI overlap? |
|---|---|---|---|---|
| live_simple | 73/100 (73.0%) | 82/100 (82.0%) | -9.0pp | YES |
| live_multiple | 65/100 (65.0%) | 71/100 (71.0%) | -6.0pp | YES |
| live_parallel (n=16) | 5/16 (31.2%) | 10/16 (62.5%) | -31.3pp | YES (small N) |
| live_parallel_multiple (n=24) | 11/24 (45.8%) | 12/24 (50.0%) | -4.2pp | YES |
| live_irrelevance | **89/100 (89.0%)** | 78/100 (78.0%) | **+11.0pp** | YES (smollm3 above) |
| live_relevance (n=16) | 16/16 (100.0%) | 15/16 (93.8%) | +6.2pp | YES |

**No CI-distinct losses on matched data.** The previously-reported
live_irrelevance "-20pp CI-distinct collapse" was an artifact of
non-overlapping slices — on the same 100 problems, smollm3 actually
outscores qwen25-1.5b by 11pp.

The curated parallel-cat advantage holds (+7 / +6 / +5 on parallel /
parallel_multiple / simple_python) — the one durable mechanical Branch D
improvement.

#### Subsection 2 — Full-distribution comparison (rep_7, full live cats)

Phase K reran the three finalists on the full live distribution
(2251 problems per model, same as smollm3's rep_1). Source:
`scripts/compare_matched_slice.py --rep 7 --models ...`; matched-ID
artifact at `acceptance/audits/phase_k_rep7_4way_live_matched_ids.json`.

This is now a proper apples-to-apples comparison on the full live
distribution.

| model | overall (n=2251) | live_simple (n=258) | live_multiple (n=1053) | live_irrelevance (n=884) | live_parallel | live_par_mul | live_relevance |
|---|---|---|---|---|---|---|---|
| **granite33-2b-instruct** | **1456/2251 (64.7%)** [62.7, 66.6] | 156/258 (60.5%) | 557/1053 (52.9%) | **720/884 (81.4%)** | 4/16 | 9/24 | 10/16 |
| qwen25-1.5b-instruct | 1395/2251 (62.0%) [59.9, 64.0] | **192/258 (74.4%)** | **699/1053 (66.4%)** | 467/884 (52.8%) | **10/16** | **12/24** | 15/16 |
| smollm3-3b-instruct | 1309/2251 (58.2%) [56.1, 60.2] | 173/258 (67.1%) | 670/1053 (63.6%) | 434/884 (49.1%) | 5/16 | 11/24 | **16/16** |
| qwen25-coder-1.5b-instruct | 1147/2251 (51.0%) [48.9, 53.0] | 172/258 (66.7%) | 684/1053 (65.0%) | 271/884 (30.7%) | 3/16 | 2/24 | 15/16 |

The overall rankings shift on full live data — **granite33 leads
overall** because `live_irrelevance` is 39% of the slice (884/2251) and
granite33 dominates it (+28.6pp over the next-best). Strip irrelevance
out and look at active categories (live_simple + live_multiple +
live_parallel + live_par_mul = 1351 problems): qwen25-1.5b wins
(913/1351 = 67.6%), smollm3 second (859/1351 = 63.6%), qwen-coder
third (861/1351 = 63.7% — within smollm3 of CI), granite33 last
(726/1351 = 53.7%).

#### Subsection 3 — Distribution robustness (within-model: first-100 → full live)

Each model ran on the same first-100 problems per live cat in rep_1
(matched). rep_7 added the remaining 158 / 953 / 784 problems per cat.
The within-model drop on the remaining problems measures distribution
robustness — does performance hold across the full distribution, or
were the first-100 systematically easier?

Per-model drop on `live_irrelevance` (the most distribution-sensitive cat):

| model | first-100 rate | remaining-784 rate | Δ (within-model) |
|---|---|---|---|
| granite33-2b | 97/100 (97.0%) | 623/784 (79.5%) | -17.5pp |
| qwen25-1.5b | 78/100 (78.0%) | 389/784 (49.6%) | -28.4pp |
| smollm3-3b | 89/100 (89.0%) | 345/784 (44.0%) | **-45.0pp** |
| qwen25-coder | 75/100 (75.0%) | 196/784 (25.0%) | -50.0pp |

**All four models drop substantially** on the remaining live_irrelevance
distribution — the first 100 problems are systematically easier than
the rest (likely curated-at-source ordering effect). granite33 is the
most robust (-17.5pp). smollm3's drop (-45pp) is larger than the
finalists' but smaller than qwen-coder's (-50pp). The "smollm3
collapses on live_irrelevance" framing **is now properly evidenced**
as a within-model distribution claim: smollm3's irrelevance behavior
on the broader distribution is meaningfully weaker than its first-100
performance, and is bottom-tier among the four models (only qwen-coder
drops more).

For other live cats, drops are small to negligible (most models within
±3pp first-100 → full). The irrelevance category is the only one where
distribution-robustness is a separating signal.

#### Synthesis — which corner does each model own on full live data?

| corner | rep_7 leader | second | gap |
|---|---|---|---|
| Tool-use (live active cats, n=1351) | qwen25-1.5b 67.6% | smollm3 63.6% | +4.0pp, CIs overlap |
| Decline-discipline (live_irrelevance, n=884) | granite33 81.4% | qwen25-1.5b 52.8% | **+28.6pp, CI-distinct** |
| Distribution robustness (within-model live_irrelevance drop) | granite33 -17.5pp | qwen25-1.5b -28.4pp | granite33 -11pp better drop |
| Coding (HumanEval pass-any) | qwen25-coder 78.0% | smollm3 75.0% | +3.0pp, CIs overlap |

The non-dominated triangle holds **directionally**, but with cleaner
evidence and one substantive sharpening: granite33's decline-discipline
lead is now CI-distinct at +28.6pp on the full distribution (was a
CI-overlap 8pp on the matched-100 slice). smollm3 is competitive on
active tool-use cats (within CI of qwen25-1.5b) but bottom-tier on
distribution robustness for irrelevance.

#### HumanEval pass@1 / pass-any / pass-all — smollm3 vs qwen25-coder

| metric | smollm3 | qwen25-coder | Δ | CI overlap? |
|---|---|---|---|---|
| pass@1 (rep_0, n=164) | 64.0% [56.4, 71.0] | 69.5% [62.1, 76.0] | -5.5pp | YES |
| pass-any (rep_0∪2∪3) | 75.0% [67.9, 81.0] | 78.0% [71.1, 83.7] | -3.0pp | YES |
| pass-all (rep_0∩2∩3) | 54.3% [46.6, 61.7] | 56.1% [48.4, 63.5] | -1.8pp | YES |

Statistically tied across all three coding metrics. qwen25-coder's
mean is consistently above smollm3 by 2–6pp but never CI-distinct on
any metric. **MBPP**: smollm3 61.1% vs qwen25-coder 64.2%, also CI
overlap.

#### Multi-turn (rep_5) — floor everywhere

| model | mt_base | mt_long_context | mt_miss_func | mt_miss_param |
|---|---|---|---|---|
| qwen25-1.5b | 0/100 | 0/100 | 0/100 | 0/100 |
| qwen25-coder | 0/100 | 0/100 | 1/100 | 1/100 |
| granite33-2b | 1/100 | 1/100 | 0/100 | 2/100 |
| smollm3-3b | 0/100 | 0/100 | 0/100 | 0/100 |

smollm3 doesn't separate from the floor. No tiebreaker.

#### Agent mode (rep_4) — non-functional for smollm3

In agent mode, smollm3 emits **zero parseable tool calls** across all
1240 problems (verified: `n_with_calls=0` in every cat's summary.json;
per-problem `actual_calls` empty everywhere). The result reads as
240/1240 (19%) but is artifactual — the 240/240 on irrelevance is
"100%" only because the model emits no calls at all (correct by
accident on a no-call category).

**Verified mechanism** (Phase J, after `raw_text` persistence landed
2026-05-14; rep_4 re-run wall=59:57, same 240/1240 reproduced
deterministically at T=0). Reproducible mechanism taxonomy via
`scripts/sample_raw_text.py --seed 1337 --n 20`; full-population
sample artifact at
`acceptance/audits/phase_j_smollm3_mechanism_samples.json`. Bucket
distribution over 1229 problems with non-empty `raw_text`:

| bucket | count | % |
|---|---|---|
| `prose_only` (math/explanation, no call shape) | 747 | 60.8% |
| `code_block` (fenced ```python or bare `def`) | 464 | 37.8% |
| `pseudo_tool` (call shape outside JSON) | 15 | 1.2% |
| `partial_tool` (truncated mid-emission) | 3 | 0.2% |
| `malformed_json` | 0 | 0.0% |
| `empty` (sampled subset) | 0 | 0.0% |

The dominant mode is **prose** (~61%), not the previously asserted
"Python code blocks" (~38%). Both shapes are present and neither is
parseable as a structured tool call. The model behaves as a Python
tutor — it explains the task and sometimes provides a code block —
rather than as a tool-using agent. The original "emits Python code
blocks" framing captured the visually-striking minority shape but
missed the majority pattern.

Aggregate finding — smollm3 is **effectively unusable in raw agent
mode** at this size without an adapter change — is independent of
the dominant-bucket question and stands on the n_with_calls=0
signal alone. Existing finalists' agent advantages (qwen25-coder's
parallel recovery rep_4 → rep_4) remain unchallenged.

#### Triangle status after the full smollm3 spectrum

After the Phase K rep_7 cross-model run, the triangle picture stabilizes
on cleaner evidence:

| corner | prior claim (Phase H, 2026-05-13) | corrected verdict (2026-05-14, post-Phase K) | audit note |
|---|---|---|---|
| **Tool-use generalist (BFCL)** | qwen25-1.5b holds; 77.1% vs smollm3 72.8% (CIs barely overlap) | **qwen25-1.5b holds on active live cats** (67.6% vs smollm3 63.6% on 1351 active live problems, CIs overlap ~4pp). On the matched-1106 slice (curated + first-100 live), smollm3 is +0.7pp above qwen — statistical tie. | prior 72.8% smollm3 number was from a lex-sort non-matching slice; see errata |
| **Synthesis coding (HumanEval)** | qwen25-coder holds (69.5%/78% pass-any); smollm3 close (64.0%/75.0%) | unchanged — qwen25-coder holds, smollm3 within CI on all 3 metrics | full n=164 / n=427 each, no slicing involved |
| **Decline discipline (live_irrelevance)** | granite33 97% vs smollm3 58% (CI-distinct, -39pp) | **granite33 holds with cleaner, larger gap**: on full live_irrelevance (n=884), granite33 81.4% vs next-best qwen25-1.5b 52.8% (**+28.6pp, CI-distinct**). Distribution-robustness ranking on the cat: granite -17pp < qwen25-1.5b -28pp < smollm3 -45pp < qwen-coder -50pp. | prior 58% number was the lex-sort first-100 slice for smollm3; the -39pp magnitude was wrong but the granite-holds direction is verified at +28.6pp on full data |

**Triangle stands**, with two sharpenings vs the published Phase H:

1. **Tool-use corner is "statistically tied on matched, qwen leads on
   full active cats"** — published as "qwen holds, CIs barely overlap";
   the matched-slice tie was hidden by the bug. On the full active live
   distribution (1351 problems, excluding irrelevance), qwen25-1.5b
   leads by 4pp with CI overlap.
2. **Decline corner is on much firmer footing** — published with the
   wrong specific magnitude (-39pp from a bad slice). True full-data
   gap is **+28.6pp CI-distinct** between granite33 (81.4%) and the
   next-best (qwen25-1.5b at 52.8%). Granite's lead grew when the
   comparison became apples-to-apples on the full distribution.

The "smollm3 collapses on live" framing now resolves cleanly:
matched-quality shows smollm3 is competitive with qwen25-1.5b on active
cats and slightly above on the matched 100-problem irrelevance slice;
**distribution-robustness** shows smollm3's live_irrelevance behavior
drops -45pp going from first-100 to remaining-784 (worse than qwen
and granite, better than qwen-coder). Both claims are real and they
measure different things.

#### Smollm3's actual niche

Smollm3 is the only model in the project's data that's competitive on
**two corners simultaneously** (within CIs of qwen25-1.5b on BFCL
overall *and* qwen25-coder on every coding metric). The existing
finalists are each axis-specialists; smollm3 is the only generalist
near the frontier. For deployments that want one model that does
*both* tool-use and coding adequately — instead of two models for two
axes — smollm3 is the cleanest pick. For axis-specific deployments,
the round-2 champions still win.

The mechanical durable finding from Branch D: smollm3 doesn't suffer
the parallel-call collapse that crushes qwen25-coder. On curated
parallel/parallel_multiple smollm3 is +93/+74 problems vs qwen25-coder
and +7/+6 vs qwen25-1.5b. If your deployment needs reliable parallel
emission, smollm3 is the strongest of the four-model field at this
size.

## Stochastic notes (vs the prior edition's published numbers)

- All non-deterministic runs (BFCL with `seed: 42` but model sampling
  at varying temperatures; HumanEval rep_2 at t=0.3, rep_3 at t=0.7)
  produce stochastically-different draws on different hardware/runtime
  even at fixed seed, because Apple Metal kernels don't guarantee
  bit-identical reductions. Expect ±3–5pp wobble between draws on
  individual categories; aggregate totals are stable to ±1–2pp.
- **qwen25-coder parallel categories** showed the largest swing
  (22% / 28.5% here vs 38.7% / 29.3% prior). The category is dominated
  by the under_called_1_of_N collapse pattern, which is bistable
  around the temperature/seed boundary; a small shift in first-token
  logits flips the pattern on a row from "emit 1 of 2" to "emit 2 of 2"
  or vice versa. The aggregate failure mode is the same in both editions.
- **granite33 live_irrelevance** dropped from 100/100 to 97/100 — still
  the cleanest signal in the data, but no longer a perfect score. The
  3 misses (`live_irrelevance_{16,20,32}-*`) are subtle: tools were
  available that *could* technically apply, just not what the user
  asked for. Borderline cases by construction.
- **rep_0 (t=0.0 HumanEval)** is now also re-run on M5 (was preserved
  from the 8 GB-Mac in the prior edition). Per-model deltas are modest:
  qwen25-1.5b 92 → 96 (+4), qwen25-coder 115 → 114 (−1), granite33
  85 → 87 (+2). The head-to-head shape stays the same; PPP/FFF the
  largest buckets at 70/35.

## Champion-decision framework

Non-dominated triangle — no single model wins every axis:

| if you value… | pick |
|---|---|
| **Synthesis coding** (HumanEval) | qwen25-coder-1.5b (69.5% pass@1, +11pp on next; pass@3 78.0%) |
| **Short-form coding** (MBPP) | qwen25-coder-1.5b (64.2%), but all three are tied within CI overlap; pick on HumanEval signal |
| **Tool-use generalist + lowest cost** | qwen25-1.5b (BFCL 77%, +7.6pp on next, cheapest of the three) |
| **Decline discipline** (knows when *not* to act) | granite33-2b (95% live_irrelevance, 85% curated) |

Two tactical observations to feed into the decision:

1. **If the deployment runs an agent loop with retries, qwen25-coder's
   pass@3 ceiling (77.4%) is the more relevant number than pass@1.**
   The sibling pass@3 ceilings are 64.6% / 63.4%. The gap widens at the
   ceiling, not the floor.
2. **Granite33's live_irrelevance result is still the cleanest decline
   signal in the data** (95% ±3.7pp, lower bound 89%). If the
   deployment surfaces tool-irrelevant queries, granite handles those
   correctly far more often than the other two (which over-call
   ~23–26% of the time).

## Errata and methodology corrections

### 2026-05-14 — Lexicographic-sort slicing bug (commit `e50fdce2`)

**Bug**: `sorted(cat_dir.glob("*.json"))` returns paths in Python string
order, not natural-integer order. When used to compute "first-100"
slices on category directories with >100 files, the resulting subset
is not problems 0–99 but a chaotic mix
(`0, 1, 10, 100..149, 11, 110..119, ...`). Two such "first-100" slices
on different-sized directories overlap by 7–11/100, not 100/100. The
comparisons were measuring mostly-disjoint problem subsets while
claiming controlled head-to-head inference.

**Affected sections** (all corrected in `e50fdce2`):

- **Phase H BFCL "apples-to-apples (n=1106)"**: smollm3 reported as
  805/1106 (72.8%) [70.1, 75.3]; matched-ID recount = **860/1106
  (77.8%) [75.2, 80.1]**. smollm3 is +0.7pp ABOVE qwen25-1.5b on the
  same problems, not -4.3pp below as published.
- **Phase H per-cat collapse**: live_irrelevance reported -20pp
  CI-distinct; matched-ID = **+11pp** (direction reversed — smollm3
  outperforms qwen25-1.5b on the matched 100). live_simple reported
  -25pp CI-distinct; matched-ID = -9pp CI-overlap.
- **Round 3 Branch A v3a vs rep_1**: live_simple reported -15pp
  collateral regression; matched-ID = -2pp (within noise, withdrawn).
  live_irrelevance reported -23pp; matched-ID = -15pp (smaller, still
  real). Hypothesis A still falsified, **weakened**.
- **Round 3 Branch B (qwen-coder fewshot)**: unchanged — curated cats
  ran 150/150 IDs both reps; bug didn't reach this branch.
- **Round 3 Branch C v3c**: collateral irrelevance harm reported -4pp
  combined (PASS within 5pp budget); matched-ID = **-7pp combined
  (FAIL exceeds budget)**. Pre-registered collateral gate **reversed**
  from PASS to FAIL. The "trade isn't catastrophic, just inefficient"
  framing in the original write-up does not survive.

**Evidentiary status changes**:

- Tool-use corner ("qwen25-1.5b holds, CIs barely overlap") →
  "**tied within CI on matched slice**" (~4.3pp shared band).
- live_irrelevance "smollm3 collapse" (head-to-head, CI-distinct) →
  withdrawn on matched data; surviving claim is **within-smollm3
  distribution-robustness** (49.1% on full 884 vs 89% on matched 100),
  which is a different epistemological category from cross-model
  superiority.
- Granite33 decline corner: -39pp CI-distinct gap is replaced with a
  split: matched-100 Δ=-8pp **CIs overlap**; full-distribution Δ=-48pp
  pending Phase K rep_7 for proper cross-model evidence.

**Fix**: `scripts/compare_matched_slice.py` with explicit
`--policy {intersection,union}` arg, `<model>:<rep>` target syntax for
cross-rep comparisons, and `--write-ids` for reproducible matched-ID
artifacts in `acceptance/audits/`. New helper module
`scripts/_problem_ids.py` provides natural-integer sort if a future
analysis needs slice-by-position. Regression test in
`tests/test_matched_slice.py::test_intersection_does_not_pick_lex_mangled_first_100`
fails if the helper ever silently regresses to lex-sort. Memory entry
`feedback_slicing_methodology.md` codifies the operational rule.

### 2026-05-14 — BFCL `raw_text` persistence gap (commit `3905bd7f`)

**Bug**: `BfclInvocationResult` dataclass and the runner serializer
never wrote a `raw_text` field; rep_4 per-problem JSONs contained
`actual_calls`, `n_turns`, etc., but not the model's actual output
text. CLAUDE.md asserted persistence from 2026-05-13, but for BFCL
specifically the path was missing.

**Affected section**: Phase H agent-mode mechanism claim — "smollm3
emits Python code blocks instead of structured tool calls" — was not
verifiable from on-disk data. The 240/1240 artifact count was correct
on disk (n_with_calls=0 across all 1240 problems), but the *what does
it emit instead* question had no persisted answer.

**Fix**:
- `BfclInvocationResult.raw_text: str | None = None` (optional for
  back-compat with legacy reps).
- Raw mode captures `ChatResponse.text` from the single call.
- Agent mode captures concatenated assistant turns joined by literal
  `"\n---\n"` (schema in ARCHITECTURE.md).
- Runner serializer omits the field when None (legacy rows differ from
  new rows only when there's text to capture).
- 11 regression tests in `tests/test_raw_text_persistence.py` cover
  forward path, backward-compat for legacy rows, and the serializer
  shape contract.

**Verified mechanism** (re-ran smollm3 rep_4; wall=59:57; reproduced
240/1240 deterministically). Reproducible sampling via
`scripts/sample_raw_text.py --seed 1337 --n 20`; full-population
artifact at `acceptance/audits/phase_j_smollm3_mechanism_samples.json`.
Bucket distribution over 1229 problems with non-empty raw_text:

- prose_only: 747 (60.8%) — dominant mode
- code_block: 464 (37.8%) — the visually-striking mode the prior
  claim focused on
- pseudo_tool / partial_tool: 18 (1.5%) combined
- empty / malformed_json: 0

The prior "emits Python code blocks" claim captured the minority
shape. Majority is prose explanation. Both fail to produce parseable
tool calls; the aggregate finding (smollm3 unusable in raw agent
mode at this size) is unchanged.

## What changed from round 1

- **Per-category sample size**: 30 → 150 curated, 30 → 100 live (where
  available). CI half-widths shrunk from ±18pp to ±5–8pp.
- **Added BFCL "live" categories** (live_simple, live_multiple,
  live_parallel, live_parallel_multiple, live_irrelevance, live_relevance).
  Adapter + grader extended; 34 grader unit tests pass.
- **Added HumanEval temperature sweep** (t=0.0 / 0.3 / 0.7). Round 1
  was deterministic-only.
- **`--temperature`, `--port`, `--auto-port` CLI overrides** in
  `run_bakeoff.py` — let the temp sweep and multi-model parallel runs
  work without editing per-model YAMLs.

## Reproducing

```bash
# BFCL leaderboard (rep_1 = round 2 deep + live)
uv run python scripts/grade_bakeoff.py --rep 1 \
    --models qwen25-1.5b-instruct qwen25-coder-1.5b-instruct granite33-2b-instruct

# HumanEval per-temperature summaries
for r in 0 2 3; do
    for m in qwen25-1.5b-instruct qwen25-coder-1.5b-instruct granite33-2b-instruct; do
        cat acceptance/humaneval/$m/rep_$r/summary.json | python3 -m json.tool
    done
done
```

Failure-mode detail is in `graded_failure_modes.md`.

## What comes next — round 3

Round-2 work is complete. Forward-looking docs added at repo root:

- **[`BENCHMARKS.md`](BENCHMARKS.md)** — what each of the five
  benchmark signals in this report actually measures, and the four
  orthogonal evaluation dimensions they span. Read this first if
  you're new to the project.
- **[`round_3_planning.md`](round_3_planning.md)** — the decision
  matrix that informed the round-3 scope, preserved as paper trail.
  Comparison cells cite their source rep + grading artifact.
- **[`round_3_design.md`](round_3_design.md)** — round-3 scope:
  prompt-engineering experiments on all three finalists (branches A,
  B, C) probing whether the prompt-mediated capability layer can
  move each model on its individual weakest axis. Executable
  commands + falsifiable gates per branch.

## Appendix: round-1 cut models

Not eligible for round 2. BFCL numbers below are 30-problem rep_0
(pre-prompt for most; v2 prompt for granite/qwen25/qwen25-coder which
*are* finalists). HumanEval numbers are at t=0.0.

| model | BFCL (30/cat) | HumanEval | reason cut |
|---|---|---|---|
| smollm2-1.7b-instruct | 88/150 (59%) | 53/164 (32%) | over-action bias; SmolLM2 schema-leak quirk on `multiple_1` |
| llama32-1b-instruct | 83/150 (55%) | 48/164 (29%) | runaway loops (32–47 calls on some rows); behind on coding |
| deepseek-coder-1.3b-instruct | 50/150 (33%) | 96/164 (59%) | BFCL bimodal failure (no-call / loop); coding competitive but tool-use gap too large |
| deepseek-r1-distill-qwen-1.5b | 10/50 (20%) | 6/30 (20%) | `<think>` consumes 8k budget; zero parseable tool calls |
| phi-1.5 | 30/150 (20%) | 13/164 (8%) | base model, parser stress-test only |
