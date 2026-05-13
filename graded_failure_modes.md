# Round 2 — failure-mode breakdown (multi-spectrum, refreshed 2026-05-13)

Companion to `graded_report.md`. Per-finalist failure patterns at the
deep sample (n=150 per curated BFCL category, n=100 per live category
where available, n=164 per HumanEval temperature).

Raw per-problem data:
- BFCL deep: `acceptance/bfcl/<model>/rep_1/<category>/*.json`
- HumanEval temperature sweep: `acceptance/humaneval/<model>/rep_{0,2,3}/results.jsonl`

## Per-finalist BFCL — what each one gets wrong at scale

### qwen25-1.5b-instruct — 77.0% (853/1106)

The clean generalist. At n=150 the pattern shape stays the same as
round 1; only the magnitudes sharpen.

- `simple_python` 91.0% — 12 misses, all name-swaps where the model picks
  a similar but wrong tool name (math/finance/restaurant domains).
- `multiple` 87.7% — 17 misses. Top buckets: `over_called_2_of_1` ×2
  (model emits the right call then a duplicate); rest are name-swap
  did-not-match-GT.
- `parallel` 78.0%, `parallel_multiple` 72.8% — **dominant failure is
  under_called**: `under_called_2_of_4` ×7 and `_2_of_3` ×3 on parallel,
  with smaller buckets at `_1_of_2`, `_2_of_6`, `_2_of_8`. The model
  commits to the first 1–2 calls and stops short. Not the qwen25-coder
  collapse pattern (which is always *exactly* 1 call) — qwen25-1.5b
  consistently emits 1–2 *less* than asked.
- `irrelevance` 57.2% — **64 of 64 failures are over-called**
  (`over_called_when_irrelevant`). User asks for something only
  partially served by the toolbox; model calls the closest tool.
  Largest single failure mode for this model.
- Live categories show the same shape: live_simple/live_multiple top
  buckets are name-swaps (`get_current_weather`, `ThinQ_Connect`,
  `HNA_WQA.search` are repeat offenders).

### granite33-2b-instruct — 69.4% (768/1106)

The disciplined model. **No-call is the dominant failure mode in every
call-emitting category** — granite declines when it shouldn't.

| category | passed | no-call fails | other fails (under/over/name-swap) |
|---|---|---|---|
| simple_python | 126/150 | 14 | 10 |
| multiple | 102/150 | **37** | 11 |
| parallel | 93/150 | 34 | 23 |
| parallel_multiple | 82/150 | 32 | 36 |
| live_simple | 68/100 | 17 | 15 |
| live_multiple | 48/100 | **35** | 17 |
| live_parallel | 4/16 | 7 | 5 |
| live_parallel_multiple | 9/24 | 8 | 7 |
| live_relevance | 10/16 | **6** | 0 |

- `irrelevance` 85.1% **+ live_irrelevance 97/100** — this is the
  trade-off. The discipline that produces 95–97% on live_irrelevance
  is the same discipline that makes the model decline when 2+
  candidate tools are in scope.
- Especially visible on `multiple` (37/48 fails are no-call) and
  `live_multiple` (35/52). When the toolbox has multiple candidates,
  granite's instinct is "I'm not sure which" → no call. The v2 system
  prompt rule 2 ("if tools cannot satisfy the user, don't call") seems
  to be over-firing on legitimate multi-tool cases.
- `live_relevance` 10/16 — **all 6 failures are no-call**. Granite
  declined when at least one tool would have served. The other two
  finalists score 15/16 here.
- The model rarely collapses (under-call by 1–2 is rare; only `live_*`
  shows a handful). Granite is qualitatively the most reliable when it
  *does* emit — its no-call rate is the bottleneck.

### qwen25-coder-1.5b-instruct — 58.6% (649/1106)

The coder. The parallel-collapse pattern named in round 1 reproduces
even harder in this edition.

| category | passed | dominant pattern |
|---|---|---|
| simple_python | 138/150 | 2 no-call + 10 misc — clean on singletons |
| multiple | 132/150 | 1 no-call + 17 name-swaps — clean |
| **parallel** | **32/150** | **`under_called_1_of_N` × 111 of 118 fails** |
| **parallel_multiple** | **42/150** | **`under_called_1_of_N` × 97 of 108 fails** |
| irrelevance | 78/150 | 72 over-calls (worst of the three) |
| live_parallel | 3/16 | name swaps + no-call |
| live_parallel_multiple | 2/24 | under_called_1_of_2 dominant |

The pattern: **the model commits one call when N are required, regardless
of how clearly the prompt asks for multiple.** Breakdown of the 111
`under_called_1_of_N` rows on curated `parallel`:

- `_1_of_2` ×72, `_1_of_3` ×24, `_1_of_4` ×14, `_1_of_6` ×1

And on `parallel_multiple` (97 of 108 fails):

- `_1_of_2` ×45, `_1_of_3` ×34, `_1_of_4` ×18

Rule 1 of the v2 system prompt ("emit N separate tool calls for N inputs")
moves the needle, but **74% of parallel rows (111/150) and 65% of
parallel_multiple rows (97/150) still emit exactly one call**. Single-
call emissions account for **94%** of parallel failures (111/118) and
**90%** of parallel_multiple failures (97/108). This is a training-
distribution issue the prompt can't reach.

Curiously, qwen25-coder's live_simple (69.3%) is competitive with the
other two — when the problem is "single call, realistic phrasing" it
holds its own; when the problem demands N emissions, it collapses.

## HumanEval — failure modes & temperature behavior

All three are 100% extraction-clean on rep_0/rep_2; rep_3 has 1
extraction miss for qwen25-1.5b (`n_extract_ok=163/164`). Across the
1,476 attempts (164 × 3 temps × 3 models) total extraction-fail count
is 1. Failures are correctness, not code-shaping.

### Head-to-head at t=0.0 (the comparable baseline — preserved from rep_0)

This table is byte-for-byte the same as the prior edition; rep_0 was
not re-run on the M5.

| outcome | qwen25 / coder / granite | count |
|---|---|---|
| all pass | P P P | 68 (41%) |
| only coder | F P F | 23 (14%) |
| qwen25 + coder, not granite | P P F | 16 (10%) |
| coder + granite, not qwen25 | F P P | 8 |
| only granite | F F P | 5 |
| qwen25 + granite, not coder | P F P | 4 |
| only qwen25 | P F F | 4 |
| all fail | F F F | 36 (22%) |

- **qwen25-coder uniquely solves 23 problems** the other two can't (14%
  of the set). These are mostly algorithmic with edge-case handling.
- **qwen25-coder uniquely misses 4** — simple list manipulation where
  it over-engineers. Real but small.
- **36 problems all 3 fail** — these are the genuinely hard 22% of
  HumanEval at this size class.

### Temperature stability

| model | t=0.0 | t=0.3 | t=0.7 | pass-all-3 | pass-any | swing |
|---|---|---|---|---|---|---|
| qwen25-coder | 115 (70.1%) | 109 (66.5%) | 107 (65.2%) | 91 (55.5%) | 127 (77.4%) | 36 |
| qwen25-1.5b | 92 (56.1%) | **93** (56.7%) | 80 (48.8%) | 68 (41.5%) | 106 (64.6%) | 38 |
| granite33 | 85 (51.8%) | **88** (53.7%) | 85 (51.8%) | 68 (41.5%) | 104 (63.4%) | 36 |

Shape:
- **qwen25-coder degrades monotonically** (115 → 109 → 107). Still the
  most temperature-stable in absolute terms (only 8 problems lost
  going t=0.0 → t=0.7), though the prior edition's much steeper
  t=0.3 → t=0.7 cliff (113 → 100) didn't reproduce here. Two
  stochastic draws sampled different points on this model's
  noise envelope.
- **qwen25-1.5b and granite33** show inverted-V at t=0.3 (+0.6pp,
  +1.9pp). Slightly under-confident at t=0.0; a little entropy
  escapes some first-token traps.
- **qwen25-1.5b** falls 8pp at t=0.7 (sampling noise overwhelms);
  **granite33** holds within noise across all three temperatures
  (51.8% → 53.7% → 51.8%). Granite is the most t=0.7-stable model
  by a clear margin in this edition.

The pass-any column is the model's effective ceiling with best-of-3
sampling. qwen25-coder's pass-any (127/164 = 77.4%) is still solidly
ahead of the others' pass@1 ceilings.

## Cross-bench observations

1. **The three are non-dominated.** qwen25-1.5b wins tool-use,
   qwen25-coder wins coding, granite33 wins decline-discipline. No
   model wins all three; ranking depends entirely on the deployment
   profile.

2. **Granite33's no-call instinct cuts both ways.** Same training that
   produces 97/100 on `live_irrelevance` produces 37 no-calls of 48
   fails on `multiple` and 35 no-calls of 52 on `live_multiple`. If
   you can engineer the deployment to surface single-tool contexts
   cleanly, granite shines; if the toolbox is broad and the right
   tool is one-of-many, granite under-emits.

3. **qwen25-coder's parallel collapse is the hard ceiling**, not a
   noise floor. 74% of curated parallel rows (111/150) and 65% of
   curated parallel_multiple rows (97/150) emit exactly one call when
   N≥2 are required; these single-call emissions account for 94%/90%
   of failures in those categories. The non-coder qwen25-1.5b sibling
   at the same n=150 emits the right count comfortably 73–78% of the
   time.

4. **Live BFCL is harder than curated except on irrelevance.** Each
   model loses 3–10pp going from curated to live on most active
   categories. The exception is irrelevance/relevance, where live is
   *easier* (the user-submitted irrelevance prompts are more clearly
   out of scope; curated includes "model has a partial-fit tool" traps).

5. **All three are essentially extraction-clean on HumanEval.** 1
   extraction failure across 1,476 attempts. The fenced-block + def-line
   extractor in `benchmarks/humaneval/adapter.py` is doing its job.

## Grader notes (still applicable)

`benchmarks/bfcl/grade.py` patches in effect:
- **Nested-dict allowed-lists** (`_dict_shape_matches`): BFCL v4 wraps
  every leaf inside a dict-typed arg in its own list; the grader
  recurses. Round 1 covered.
- **Implicit-multiplication normalizer** (`_normalize_math_expr`):
  `3*x**2` matches `3x**2`. Does NOT rewrite `^` → `**`.
- **Live BFCL support**: live_simple / live_multiple / live_parallel /
  live_parallel_multiple route to existing graders; live_irrelevance
  shares irrelevance criteria; live_relevance uses `grade_relevance`
  (pass = at least one call).

34/34 grader unit tests pass.

## Runtime notes

- **`rep_1` BFCL** and **`rep_2`/`rep_3` HumanEval** in this edition
  were re-run on a 128 GB M5 Max using the parallel-3 `--auto-port`
  runner path. Wall times are not directly comparable to the prior
  edition's 8 GB-Mac runs.
- **`rep_0` HumanEval (t=0.0)** is preserved from the original 8 GB
  Mac run.

## Reproducing

```bash
# Full failure-mode JSON
uv run python scripts/failure_modes.py --rep 1 --json > /tmp/fm.json

# Per-row inspection of a specific failure
cat acceptance/bfcl/<model>/rep_1/<category>/<id>.json | python3 -m json.tool

# HumanEval failure on a specific problem at a specific temperature
uv run python -c "
import json
for line in open('acceptance/humaneval/<model>/rep_<2 or 3>/results.jsonl'):
    r = json.loads(line)
    if r['task_id'] == 'HumanEval/<n>':
        print('passed:', r['passed'])
        print('error:', r['error'])
        print('---raw model output---')
        print(r['raw_text'])
"
```

## Appendix: round-1 cut models

Not part of round 2. Numbers from rep_0 / round 1 only.

- **smollm2-1.7b-instruct** — 88/150 BFCL, 53/164 HumanEval. SmolLM2
  schema-leak quirk on `multiple_1`; over-action bias.
- **llama32-1b-instruct** — 73/150 BFCL, 48/164 HumanEval. Looping
  (47-call, 36-call, 32-call rows).
- **deepseek-coder-1.3b-instruct** — 50/150 BFCL, 96/164 HumanEval.
  BFCL bimodal failure (no-call / loop); coding competitive but tool-
  use unworkable. Partial v2 BFCL data preserved at
  `acceptance/bfcl/deepseek-coder-1.3b-instruct/rep_0_v2_partial_20260512/`
  in case useful later.
- **deepseek-r1-distill-qwen-1.5b** — 10/50 BFCL, 6/30 HumanEval.
  Reasoning model, `<think>` consumes the budget; zero parseable tool
  calls. Not viable at this size without budget surgery.
- **phi-1.5** — 30/150 BFCL, 13/164 HumanEval. Base model, parser
  stress-test only.
