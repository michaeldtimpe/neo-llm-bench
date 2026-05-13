# Round 2 — failure-mode breakdown (multi-spectrum, 2026-05-13)

Companion to `graded_report.md`. Per-finalist failure patterns at the
deep sample (n=150 per curated BFCL category, n=100 per live category
where available, n=164 per HumanEval temperature).

Raw per-problem data:
- BFCL deep: `acceptance/bfcl/<model>/rep_1/<category>/*.json`
- HumanEval temperature sweep: `acceptance/humaneval/<model>/rep_{0,2,3}/results.jsonl`

## Per-finalist BFCL — what each one gets wrong at scale

### qwen25-1.5b-instruct — 76.9% (851/1106)

The clean generalist. At n=150 the pattern shape stays the same as
round 1; only the magnitudes sharpen.

- `simple_python` 92.0% — 12 misses scattered across arg-value mismatches
  (mostly math/finance functions where the model picks a similar but
  wrong tool name).
- `multiple` 88.0% — 18 misses. Top bucket: over_called_2_of_1 (×3) —
  the model emits one call and then a duplicate. Rest are
  did-not-match-GT name swaps.
- `parallel` 77.3%, `parallel_multiple` 76.7% — **dominant failure is
  under_called**: `under_called_2_of_4` ×6 and `_2_of_3` ×3 on parallel.
  The model commits to the first 1–2 calls and stops short. Not the
  qwen25-coder collapse pattern (which is always *exactly* 1 call) —
  qwen25-1.5b consistently emits 1–2 less than asked.
- `irrelevance` 58.0% — **63 of 63 failures are over-called** (rule 2
  trap). User asks for something only partially served by the toolbox;
  model calls the closest tool. Largest single failure mode for this
  model.
- Live curated categories show a domain-specific over-call pattern:
  `get_current_weather` and LG's `ThinQ_Connect` are name-shaped traps
  where the live problem has a similar-but-different tool and the model
  picks the wrong one.

### granite33-2b-instruct — 69.3% (767/1106)

The disciplined model. **No-call is the dominant failure mode in every
call-emitting category** — granite declines when it shouldn't.

| category | passed | no-call fails | other fails |
|---|---|---|---|
| simple_python | 126/150 | 14 | 10 |
| multiple | 102/150 | **37** | 11 |
| parallel | 92/150 | 33 | 25 |
| parallel_multiple | 79/150 | 33 | 38 |
| live_simple | 68/100 | 17 | 15 |
| live_multiple | 48/100 | **35** | 17 |
| live_parallel | 4/16 | 7 | 5 |
| live_parallel_multiple | 9/24 | 8 | 7 |
| live_relevance | 10/16 | **6** | 0 |

- `irrelevance` 86.0% **+ live_irrelevance 100/100** — this is the
  trade-off. The discipline that produces 100% on live_irrelevance is
  the same discipline that makes the model decline when 2+ candidate
  tools are in scope.
- Especially visible on `multiple` (37/48 fails are no-call) and
  `live_multiple` (35/52). When the toolbox has multiple candidates,
  granite's instinct is "I'm not sure which" → no call. The v2 system
  prompt rule 2 ("if tools cannot satisfy the user, don't call") seems
  to be over-firing on legitimate multi-tool cases.
- `live_relevance` 10/16 — **all 6 failures are no-call**. Granite
  declined when at least one tool would have served. The other two
  finalists score 15/16 here.
- The model never collapses (0 under-call patterns dominate; over-call
  patterns rare except on irrelevance). Granite is qualitatively the
  most reliable when it *does* emit — its no-call rate is the problem.

### qwen25-coder-1.5b-instruct — 61.7% (682/1106)

The coder. The parallel-collapse pattern named in round 1 reproduces
hard at n=150.

| category | passed | dominant pattern |
|---|---|---|
| simple_python | 135/150 | 4 no-call + 11 misc — clean on singletons |
| multiple | 132/150 | scattered name-swaps — clean |
| **parallel** | **58/150** | **`under_called_1_of_N` × 77 of 92 fails** |
| **parallel_multiple** | **44/150** | **`under_called_1_of_N` × 91 of 106 fails** |
| irrelevance | 78/150 | 72 over-calls (worst of the three) |
| live_parallel | 4/16 | under_called_1_of_2 × 6 |
| live_parallel_multiple | 5/24 | **under_called_1_of_2 × 15 of 19** |

The pattern: **the model commits one call when N are required, regardless
of how clearly the prompt asks for multiple.** Rule 1 of the v2 system
prompt fixes the worst 14pp of this at curated `parallel` (was 0/30
pre-prompt, now 39%), but **51% of parallel rows (77/150) and 61% of
parallel_multiple rows (92/150) still emit exactly one call**. Single-
call emissions account for **84%** of parallel failures (77 of 92) and
**86%** of parallel_multiple failures (91 of 106). This is a training-
distribution issue the prompt can't reach.

Curiously, qwen25-coder's live_simple (70%) is *higher* than its
curated `parallel` (38.7%) — when the problem is "single call,
realistic phrasing" it's competitive; when the problem demands N
emissions, it can't.

## HumanEval — failure modes & temperature behavior

All three are 100% extraction-clean (0 syntax / extraction failures
across 1,476 attempts: 164 × 3 temps × 3 models). Failures are
correctness, not code-shaping.

### Head-to-head at t=0.0 (the comparable baseline)

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
- **qwen25-coder uniquely misses 4** (`HumanEval/5`, `/102`, `/155`,
  `/157`) — simple list manipulation where it over-engineers. Real but
  small.
- **36 problems all 3 fail** — these are the genuinely hard 22% of
  HumanEval at this size class.

### Temperature stability

| model | t=0.0 | t=0.3 | t=0.7 | pass-all-3 | pass-any | swing |
|---|---|---|---|---|---|---|
| qwen25-coder | 115 (70.1%) | 113 (68.9%) | 100 (61.0%) | 92 (56.1%) | 123 (75.0%) | 31 |
| qwen25-1.5b | 92 (56.1%) | **94** (57.3%) | 87 (53.0%) | 72 (43.9%) | 108 (65.9%) | 36 |
| granite33 | 85 (51.8%) | **88** (53.7%) | 82 (50.0%) | 66 (40.2%) | 104 (63.4%) | 38 |

Shape:
- **qwen25-coder degrades monotonically**. At the deterministic-
  correctness ceiling, sampling can only remove hits; there's no slack
  to recover. **Most temperature-stable of the three** (swing = 31
  problems where any-temp ≠ all-temps; the other two have 36–38).
- **qwen25-1.5b and granite33 show inverted-V** at t=0.3 (+2pp / +3pp).
  Slightly under-confident at t=0.0; a little entropy escapes some
  first-token traps. Both fall below baseline at t=0.7.
- **All three lose 5–11pp going from t=0.3 → t=0.7**. Temperature 0.7
  is too much at this size for any of them.

The pass-any column (problems any temperature passes) is the model's
effective ceiling with best-of-N sampling. The gap to pass@1 is
modest — best-of-3 isn't a game-changer; ~9 extra problems for each
model. Of the three, qwen25-coder's pass-any (123/164 = 75%) is still
solidly ahead of the others' pass@1 ceilings.

## Cross-bench observations

1. **The three are non-dominated.** qwen25-1.5b wins tool-use,
   qwen25-coder wins coding, granite33 wins decline-discipline. No
   model wins all three; ranking depends entirely on the deployment
   profile.

2. **Granite33's no-call instinct cuts both ways.** Same training that
   produces 100/100 on `live_irrelevance` produces 37 no-calls of 48
   fails on `multiple` and 35 no-calls of 52 on `live_multiple`. If
   you can engineer the deployment to surface single-tool contexts
   cleanly, granite shines; if the toolbox is broad and the right
   tool is one-of-many, granite under-emits.

3. **qwen25-coder's parallel collapse is a hard ceiling**, not a
   noise floor. 51% of curated parallel rows (77/150) and 61% of
   curated parallel_multiple rows (92/150) emit exactly one call when
   N≥2 are required; these single-call emissions account for 84%/86%
   of failures in those categories. The non-coder qwen25-1.5b sibling
   at the same n=150 emits the right count comfortably 76–77% of the
   time.

4. **Live BFCL is harder than curated except on irrelevance.** Each
   model loses 3–10pp going from curated to live on most active
   categories. The exception is irrelevance, where live is *easier*
   (the user-submitted irrelevance prompts are more clearly out of
   scope; curated includes "model has a partial-fit tool" traps).

5. **All three are 100% extraction-clean on HumanEval.** No code-
   shape failures across 1,476 attempts. The fenced-block + def-line
   extractor in `benchmarks/humaneval/adapter.py` is doing its job.

## Grader notes (still applicable)

`benchmarks/bfcl/grade.py` patches in effect:
- **Nested-dict allowed-lists** (`_dict_shape_matches`): BFCL v4 wraps
  every leaf inside a dict-typed arg in its own list; the grader
  recurses. Round 1 covered.
- **Implicit-multiplication normalizer** (`_normalize_math_expr`):
  `3*x**2` matches `3x**2`. Does NOT rewrite `^` → `**`.
- **Live BFCL support** (added 2026-05-12): live_simple / live_multiple
  / live_parallel / live_parallel_multiple route to existing graders;
  live_irrelevance shares irrelevance criteria; live_relevance has a
  new `grade_relevance` (pass = at least one call).

34/34 grader unit tests pass.

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
