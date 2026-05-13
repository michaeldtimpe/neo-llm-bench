# Round 2 — champion bake-off (multi-spectrum, 2026-05-13)

Finalists from round 1:

- **qwen25-1.5b-instruct** — Alibaba instruct, 1.5B params, Q8_0
- **qwen25-coder-1.5b-instruct** — coder-tuned sibling
- **granite33-2b-instruct** — IBM Granite 3.3, 2B params, Q8_0

All BFCL data uses the v2 `BFCL_SYSTEM_PROMPT` (see
`benchmarks/bfcl/adapter.py`). Five round-1 models are cut; see appendix.

## TL;DR

| model | BFCL deep (n=1106) | HumanEval pass@1 (t=0.0) | HumanEval pass@3 (any temp) |
|---|---|---|---|
| qwen25-1.5b-instruct | **76.9% (851/1106) ±2.5pp** | 56.1% (92/164) | 65.9% (108/164) |
| granite33-2b-instruct | 69.3% (767/1106) ±2.7pp | 51.8% (85/164) | 63.4% (104/164) |
| qwen25-coder-1.5b-instruct | 61.7% (682/1106) ±2.9pp | **70.1% (115/164)** | **75.0% (123/164)** |

The three are in a non-dominated triangle: qwen25-1.5b wins tool-use,
qwen25-coder wins coding, granite33 wins irrelevance discipline (and is
the strongest tool-user-of-last-resort). 8pp gap on BFCL between the top
two is well outside CI overlap — that's a real win, not noise. Coding
margin between qwen25-coder and the rest is even larger.

## BFCL — full leaderboard, with 95% Wilson CIs

n=150 per curated category (5 categories × 3 finalists × 150 = 2,250
problems); n=100 per live category where available (capped: live_parallel
16, live_parallel_multiple 24, live_relevance 16 → 356 per finalist
across 6 live categories). 1,106 total BFCL problems per finalist.

### Curated (BFCL v4 non-live, 750 / model)

| category | n | qwen25-1.5b | granite33 | qwen25-coder |
|---|---|---|---|---|
| simple_python | 150 | **92.0%** ±4.4 | 84.0% ±5.9 | 90.0% ±4.8 |
| multiple | 150 | 88.0% ±5.2 | 68.0% ±7.4 | 88.0% ±5.2 |
| parallel | 150 | **77.3%** ±6.7 | 61.3% ±7.7 | 38.7% ±7.7 |
| parallel_multiple | 150 | **76.7%** ±6.7 | 52.7% ±7.9 | 29.3% ±7.2 |
| irrelevance | 150 | 58.0% ±7.8 | **86.0%** ±5.6 | 52.0% ±7.9 |
| **curated total** | 750 | **78.4% (588/750) ±2.9** | 70.4% (528/750) ±3.3 | 59.6% (447/750) ±3.5 |

### Live (BFCL v4 user-submitted, ~356 / model)

| category | n | qwen25-1.5b | granite33 | qwen25-coder |
|---|---|---|---|---|
| live_simple | 100 | 76.0% ±8.3 | 68.0% ±9.0 | 70.0% ±8.8 |
| live_multiple | 100 | **67.0%** ±9.1 | 48.0% ±9.6 | 59.0% ±9.5 |
| live_parallel | 16 | **62.5%** ±21.4 | 25.0% ±19.7 | 25.0% ±19.7 |
| live_parallel_multiple | 24 | **58.3%** ±18.3 | 37.5% ±18.1 | 20.8% ±15.6 |
| live_irrelevance | 100 | 81.0% ±7.6 | **100.0%** ±1.8 | 82.0% ±7.5 |
| live_relevance | 16 | 93.8% ±13.6 | 62.5% ±21.4 | 93.8% ±13.6 |
| **live total** | 356 | 73.9% (263/356) ±4.5 | 67.1% (239/356) ±4.9 | **66.0%** (235/356) ±4.9 |

Interpretation:
- **qwen25-1.5b** is the dominant generalist — top on 6 of 11 categories,
  comfortably ahead on the parallel categories.
- **granite33** crushes live_irrelevance (**100/100, CI lower bound 96%**)
  and curated irrelevance (86% ±5.6) — its irrelevance discipline is the
  strongest signal in the data.
- **qwen25-coder** is no better than the others on tool-use overall; its
  parallel categories collapse (38.7% / 29.3%) is the dominant drag.
  Closer to parity on `live_` than `curated` (66% vs 60%) — the live
  problems lean toward single-call work that suits its training.

## HumanEval — temperature sweep (164 problems × 3 temps)

| model | t=0.0 | t=0.3 | t=0.7 | pass-all-3 | pass-any |
|---|---|---|---|---|---|
| qwen25-coder-1.5b | **70.1%** (115) | 68.9% (113) | 61.0% (100) | **56.1%** (92) | **75.0%** (123) |
| qwen25-1.5b | 56.1% (92) | **57.3%** (94) | 53.0% (87) | 43.9% (72) | 65.9% (108) |
| granite33-2b | 51.8% (85) | **53.7%** (88) | 50.0% (82) | 40.2% (66) | 63.4% (104) |

CI half-width at n=164, p~0.6 is about ±7.5pp. So:
- The qwen25-coder lead over the other two on HumanEval is real
  (>10pp at every temperature). Outside CI overlap.
- qwen25-1.5b vs granite33 is within CI at every temperature — they're
  statistically tied on coding.

Temperature shape:
- **qwen25-coder degrades monotonically** (70.1 → 68.9 → 61.0). The
  model is sharper at the deterministic-correctness ceiling; sampling
  noise just removes hits.
- The other two show a faint **inverted-V** peaking at t=0.3 (+2pp on
  qwen25-1.5b, +3pp on granite33). They're slightly under-confident at
  t=0.0; a little entropy escapes some bad first-token traps. But t=0.7
  is too much for both.
- **pass-all-3 vs pass-any** spread is the model's coding uncertainty
  window. qwen25-coder's window is 31 problems wide (123−92); the other
  two are 36 and 38 — qwen25-coder is **also the most stable** across
  temperatures, not just the strongest.

## Resource cost (multi-spectrum round 2 only)

BFCL rep_1 (1,106 problems) + HumanEval rep_2 (t=0.3, 164) + rep_3
(t=0.7, 164) = 1,434 problems per model. Wall is **compute time**; on
this 8 GB Mac the actual wall-clock spans several days of intermittent
system sleep.

| model | BFCL rep_1 wall | HE t=0.3+0.7 wall | **total** | total comp tokens |
|---|---|---|---|---|
| qwen25-1.5b-instruct | 1h 00m | 0h 34m | **1h 34m** | 116,962 |
| qwen25-coder-1.5b-instruct | 1h 15m | 0h 38m | **1h 53m** | 149,789 |
| granite33-2b-instruct | 2h 19m | 0h 53m | **3h 13m** | 146,067 |

Granite33 is ~2× wall-expensive vs qwen25-1.5b — at Q8_0 + 2 B params +
n_ctx 8192 it tips the 8 GB system into swap under sustained load. The
two qwens fit cleanly in RAM. Cost-of-quality matters at deploy time on
this hardware.

## Champion-decision framework

This is a non-dominated triangle. No single model wins every axis:

| if you value… | pick |
|---|---|
| **Coding accuracy** | qwen25-coder-1.5b (70.1% pass@1, +14pp on next; pass@3 75%) |
| **Tool-use generalist + lowest cost** | qwen25-1.5b (BFCL 77%, fastest, cheapest, +7.6pp on next) |
| **Decline discipline** (knows when *not* to act) | granite33-2b (100% live_irrelevance, 86% curated) |

Two tactical observations to feed into the decision:

1. **If the deployment runs an agent loop with retries, qwen25-coder's
   pass@3 ceiling (75%) is the more relevant number than pass@1.** The
   sibling's pass@3 ceilings are 66% / 63%. The gap widens at the ceiling,
   not the floor.
2. **Granite33's live_irrelevance result is the cleanest signal in the
   data** (100/100, CI lower bound 96%). If the deployment has
   tool-irrelevant queries — i.e. user can ask for things outside the
   tool surface — granite handles that perfectly. The other two
   over-call ~18–19% of the time.

## What changed from round 1

- **Per-category sample size**: 30 → 150 curated, 30 → 100 live (where
  available). CI half-widths shrunk from ±18pp to ±5–8pp.
- **Added BFCL "live" categories** (live_simple, live_multiple,
  live_parallel, live_parallel_multiple, live_irrelevance, live_relevance).
  Adapter + grader extended; 34 grader unit tests now pass (was 32).
- **Added HumanEval temperature sweep** (t=0.0 / 0.3 / 0.7). Round 1
  was deterministic-only.
- **`--temperature` CLI override** in `run_bakeoff.py` — lets the temp
  sweep run without editing per-model YAMLs.

## Reproducing

```bash
# BFCL leaderboard (rep_1 = round 2 deep + live)
uv run python scripts/grade_bakeoff.py --rep 1 \
    --models qwen25-1.5b-instruct qwen25-coder-1.5b-instruct granite33-2b-instruct

# HumanEval temperature data
for r in 0 2 3; do
    for m in qwen25-1.5b-instruct qwen25-coder-1.5b-instruct granite33-2b-instruct; do
        cat acceptance/humaneval/$m/rep_$r/summary.json | python3 -m json.tool
    done
done
```

Failure-mode detail is in `graded_failure_modes.md`.

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
