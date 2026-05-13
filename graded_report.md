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

**Result**: regressed across the board. v3b (decision-tree sibling)
aborted per the round-3 abort gate ("any cat drops ≥3pp at v3a → skip
v3b"). Apples-to-apples slice (live cats truncated to first-100 to
match rep_1's smaller live limit):

| cat | rep_1 | rep_6 v3a | Δpp |
|---|---|---|---|
| simple_python | 92.0% | 91.3% | -0.7 |
| multiple | 88.7% | 89.3% | +0.7 |
| parallel | 78.7% | 77.3% | -1.3 |
| parallel_multiple | 73.3% | 72.7% | -0.7 |
| irrelevance | 57.3% | 51.3% | **-6.0** ← target regressed |
| live_simple | 82.0% | 67.0% | **-15.0** ← collateral |
| live_multiple | 71.0% | 67.0% | -4.0 |
| live_irrelevance | 78.0% | 55.0% | **-23.0** ← collateral |
| OVERALL | 77.1% | 72.3% | **-4.8** |

The imperative shifted the decline boundary in the wrong direction —
the model declined on legitimate user requests (`live_simple`,
`live_irrelevance`) as severely as on irrelevant ones, while
*increasing* over-call on the curated `irrelevance` set (over_called
bucket 64 → 73, +9 problems). Hypothesis A falsified.

### Branch B — qwen25-coder · v2_fewshot_parallel

**Hypothesis**: two in-context parallel examples close qwen25-coder's
under_called_1_of_N gap on `parallel`/`parallel_multiple` — recovering
agent-mode's +52 parallel-recovery (rep_4) at raw-mode token cost.

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

### Branch C — granite33-2b · v3c (decline-boundary calibration)

**Hypothesis**: a prompt addendum loosens the decline boundary on
`multi_turn_miss_func` — converting `empty_turn_model_response`
problems (baseline 74/100) into alternative-tool attempts. Framed as a
trade-off measurement, not one-sided improvement; collateral on
irrelevance is the question.

**Result**: primary gate failed; collateral gates passed.

| gate | required | observed | verdict |
|---|---|---|---|
| mt_miss_func `empty_response` reduction | ≥20% (74 → ≤59) | **4.1%** (74 → 71) | FAIL |
| irrelevance + live_irrelevance combined harm | ≤5pp drop | -4pp combined | PASS |
| live_irrelevance Wilson 95% CI lower bound | ≥85% | 93.0% (98/100) | PASS |

Per-category (irrelevance sliced to first 100 to match rep_6's limit):

| cat | baseline | rep_6 v3c | Δ |
|---|---|---|---|
| mt_miss_func `empty_response` | 74/100 (rep_5) | 71/100 | -3 |
| mt_miss_func `state_mismatch` | 14/100 (rep_5) | 17/100 | +3 |
| mt_miss_func PASS | 0/100 | 0/100 | 0 |
| irrelevance (first 100) | 85/100 (rep_1) | 80/100 | -5 |
| live_irrelevance | 97/100 (rep_1) | 98/100 | +1 |
| multi_turn_base (collateral) | 1/100 (rep_5) | 2/100 | +1 |
| multi_turn_miss_param (collateral) | 2/100 (rep_5) | 2/100 | 0 |

The boundary did move — granite traded exactly 3 empty responses for 3
state-mismatch failures — but no movement converted to passes: the
calls the model now emits on `miss_func` are the wrong tool in the
wrong order. The irrelevance harm landed within budget (combined
-4pp), so the trade isn't catastrophic, just inefficient. Hypothesis C
partially disconfirmed: the decline boundary is movable, but the
alternative-tool-selection capability that would convert moved
problems into passes is absent at this size/quant.

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
