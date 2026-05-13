# Round 2 — champion bake-off (multi-spectrum, refreshed 2026-05-13)

Finalists from round 1:

- **qwen25-1.5b-instruct** — Alibaba instruct, 1.5B params, Q8_0
- **qwen25-coder-1.5b-instruct** — coder-tuned sibling
- **granite33-2b-instruct** — IBM Granite 3.3, 2B params, Q8_0

All BFCL data uses the v2 `BFCL_SYSTEM_PROMPT` (see
`benchmarks/bfcl/adapter.py`). Five round-1 models are cut; see appendix.

**Hardware note.** `rep_1` BFCL and `rep_2`/`rep_3` HumanEval in this
edition were re-run on a 128 GB M5 Max (the original 8 GB-Mac numbers
were superseded). `rep_0` HumanEval (t=0.0) is preserved from the
original 8 GB-Mac run. Wall times are therefore not comparable across
reps and are not reported here.

## TL;DR

| model | BFCL deep (n=1106) | HumanEval pass@1 (t=0.0) | HumanEval pass@3 (any temp) |
|---|---|---|---|
| qwen25-1.5b-instruct | **77.0% (853/1106) ±2.5pp** | 56.1% (92/164) | 64.6% (106/164) |
| granite33-2b-instruct | 69.4% (768/1106) ±2.7pp | 51.8% (85/164) | 63.4% (104/164) |
| qwen25-coder-1.5b-instruct | 58.6% (649/1106) ±2.9pp | **70.1% (115/164)** | **77.4% (127/164)** |

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
| qwen25-coder-1.5b | **70.1%** (115) | 66.5% (109) | 65.2% (107) | **55.5%** (91) | **77.4%** (127) |
| qwen25-1.5b | 56.1% (92) | **56.7%** (93) | 48.8% (80) | 41.5% (68) | 64.6% (106) |
| granite33-2b | 51.8% (85) | **53.7%** (88) | 51.8% (85) | 41.5% (68) | 63.4% (104) |

CI half-width at n=164, p~0.6 is about ±7.5pp. So:
- The qwen25-coder lead over the other two on HumanEval is real
  (>10pp at every temperature). Outside CI overlap.
- qwen25-1.5b vs granite33 is within CI at every temperature — they're
  statistically tied on coding.

Temperature shape:
- **qwen25-coder degrades monotonically** in this edition too
  (70.1 → 66.5 → 65.2), though the t=0.3 → t=0.7 drop is much milder
  than the prior edition (68.9 → 61.0). Coder's sampling resilience
  varies between draws — interpret the absolute number with care.
- **qwen25-1.5b** and **granite33** show the same faint inverted-V
  peaking at t=0.3 (+0.6pp and +1.9pp). Slight under-confidence at
  t=0.0 → some first-token traps are escaped with a little entropy.
- **qwen25-1.5b** falls 8pp at t=0.7 (sampling noise overwhelms);
  granite33 holds (50→52→52).
- **pass-all-3 vs pass-any** spread is the model's coding uncertainty
  window: qwen25-coder = 36, qwen25-1.5b = 38, granite33 = 36. All
  similar at this draw — best-of-3 is worth ~22pp / ~23pp / ~22pp over
  pass@1, the bulk of which is t=0.0 → t=0.3 alone.

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
- **rep_0 (t=0.0 HumanEval)** is preserved from the prior edition;
  the head-to-head table below is byte-for-byte the same.

## Champion-decision framework

Non-dominated triangle — no single model wins every axis:

| if you value… | pick |
|---|---|
| **Coding accuracy** | qwen25-coder-1.5b (70.1% pass@1, +14pp on next; pass@3 77.4%) |
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
