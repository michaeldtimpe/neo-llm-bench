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

## MBPP — sanitized split, n=427 (rep_0 t=0.0)

| model | pass@1 | uniquely solves | uniquely misses |
|---|---|---|---|
| qwen25-coder-1.5b-instruct | 64.2% (274/427) | 21 (5%) | 13 |
| qwen25-1.5b-instruct | 62.5% (267/427) | 13 (3%) | 21 |
| granite33-2b-instruct | 59.7% (255/427) | 17 (4%) | 28 |

48% of problems pass for all three (205/427); 25% fail for all three
(106/427). The remaining 27% is where the models differentiate, and the
"only X" buckets are evenly distributed (13/17/21) — no single model
runs away with the MBPP-unique territory.

Compare to the HumanEval head-to-head shape: there qwen-coder uniquely
solves 23 problems (14% of HE), and granite/qwen-1.5b uniquely solve 4–5
each. **MBPP equalizes the unique-solve counts** because short-form,
template-heavy problems are within reach of all three architectures.
HumanEval-shaped synthesis is what concentrates wins in the coder
model.

Normalization audit: all 427 model outputs contained a markdown fence
across all three models — the system prompt's "single fenced block"
instruction is followed 100% of the time. The normalizer's fence-strip
+ first-def-anchor + main-guard-drop converted every raw output into
executable Python without dropping any (`n_extract_ok = 427/427` for
all three models).

## BFCL agent mode (rep_4) — orchestration effects per finalist

Companion to the agent-mode section of `graded_report.md`. Per-finalist
per-category notes on *what the loop did* to each model's behavior.

### qwen25-coder-1.5b — the most agent-loop-sensitive model

- **Parallel rescue**: 74% of curated `parallel` rows fail in raw mode
  with `under_called_1_of_N`. Agent mode lets the model emit additional
  calls across turns; +51 problems each on `parallel` and
  `parallel_multiple`. Mean 5.5 turns on parallel (vs 2.0 elsewhere) —
  the loop is actively iterating until the model exhausts its call set.
- **Multi-call over-shoot**: −31 on curated `multiple` and −18 on
  `live_multiple`. These categories require *exactly one* call. Agent
  mode pushes the model to chain: stub results come back, the model
  reads them as "OK, what else?" and calls a second tool. The
  `over_called_2_of_1` pattern moves from 2 cases in raw to dozens in
  agent.
- **Live_parallel_multiple regression**: +2 raw→agent but mean turns
  hits **7.4** (i.e. close to `max_steps=12`). The model is iterating
  hard and rarely succeeding. Worst token efficiency in the matrix
  (7.92× over raw).
- Net: +52 problems aggregate, but at 3.1× token cost. Per-token
  efficiency drops 65%.

### qwen25-1.5b-instruct — agent mode is a no-op

- Pass counts unchanged on every curated category. ±1 on live (one
  problem moved between `live_simple`/`live_multiple` and `live_par_mul`/
  `live_irrelevance`/`live_relevance`).
- Mean turns 1.2–2.5 depending on category. Model wraps quickly.
- Token cost 1.3–2.5× across categories — pure overhead, zero pass-rate
  benefit. This model is already at its first-pass ceiling.

### granite33-2b — small consistent lift, efficient loop

- +8 problems aggregate, distributed: +2 on `parallel`, +2 on
  `parallel_multiple`, +1 on `multiple`, +1 on `live_parallel`, +2 on
  `live_irrelevance`.
- Mean turns 1.0–1.9 — shortest of the three. Granite's instinct to
  decline-when-unsure shows up here too: it doesn't iterate as
  aggressively as coder, so the loop's "extra emit" effect is mild.
- Token cost 1.0–1.95× — most efficient agent-mode token-multiplier
  of the three.
- `live_irrelevance` 97/100 → 99/100 in agent mode (the loop's stub
  results don't push granite to over-call on irrelevance — its decline-
  discipline survives the loop).

### Cross-model: turn-count distribution tells the story

Sorted by raw→agent pass delta:

```
delta=+52  qwen-coder    mean turns 4.0  ←  loop is rescuing & breaking
delta= +8  granite33     mean turns 1.6  ←  efficient short turns
delta= +1  qwen25-1.5b   mean turns 1.9  ←  already at ceiling
```

The model whose agent-loop turn count is *highest* is also the one
whose raw mode underperforms most relative to its sibling. The loop
extracts more value when the model has more first-pass mistakes to
recover — but pays more tokens to do it.

### Cross-model agent-mode `raw_text` mechanism taxonomy (2026-05-14)

Phase J added `raw_text` persistence to BFCL and re-ran agent-mode
rep_4 for all four models. The taxonomy is over the assistant text
captured each turn (joined by `\n---\n` across turns). Source artifacts:
`acceptance/audits/phase_j_<model>_mechanism_samples.json`.

Each model has a **distinctive mechanism signature** for what it
emits in agent mode — even when the loop ends up dispatching the
right structured tool calls, the text the model produces alongside
those calls is highly model-shape-specific.

| model | dominant bucket | distribution (n with raw_text) |
|---|---|---|
| qwen25-1.5b | **prose_only 99.5%** | 1233/1239 prose, 3 code_block, 3 pseudo_tool |
| qwen25-coder | **code_block 93.7%** | 1162/1240 code_block, 60 prose, 12 malformed_json, 4 pseudo_tool, 2 partial |
| granite33-2b | **pseudo_tool 69.8%** | 866/1240 pseudo_tool, 232 prose, 97 code_block, 39 malformed_json, 6 partial |
| smollm3-3b | mixed: **prose_only 60.8% + code_block 37.8%** | 747/1229 prose, 464 code_block, 15 pseudo_tool, 3 partial |

Plain-text descriptions of each model's mechanism:

- **qwen25-1.5b** — talks while it tools. Emits structured tool calls
  through its native template and adds a natural-language sidebar
  (e.g. "I'll look that up for you" or a summary of what it intends
  to do). Almost never emits anything that looks like code or
  function-call syntax in text. The structured tool channel and the
  prose channel are decoupled; tool calls go through the right path,
  the text is human-readable narration.
- **qwen25-coder** — writes the code. Its dominant text-channel
  output is fenced ```python blocks containing the function body or
  example usage. The structured tool channel is also active (78%
  agent-mode pass rate), but the text channel betrays the model's
  "complete the code I'm writing" bias documented in the curated
  parallel collapse — even in agent mode where the harness is dispatching
  real calls, the model is *also* writing Python code in the message.
- **granite33-2b** — emits tool calls as text. 70% of granite's
  agent-mode text matches the pseudo-tool-call shape — function-call
  syntax outside JSON (e.g. `funcname(arg=val)`) that the agent loop's
  text-channel parser picks up. This is granite's tool-call emission
  format: the model writes the call as text and lets the parser
  extract it, rather than using the OpenAI-style structured `tool_calls`
  field. 3.1% are `malformed_json` (attempted-but-invalid structured
  calls) — granite occasionally tries structured emission and trips
  on the syntax.
- **smollm3-3b** — split between prose explanation (61%) and Python
  code blocks (38%). Emits **zero** structured tool calls anywhere in
  the rep_4 run — the structured channel is non-functional for
  smollm3 at this size/quant. Both prose and code-block emissions are
  unparsable as tool calls; the agent loop terminates on every
  problem with `actual_calls=[]`. The 240/1240 (19%) agent-mode pass
  rate is entirely irrelevance pass-by-silence (240/240 on irrelevance
  + 0 elsewhere). See "Subsection 3 — Smollm3-unique failure shape"
  in the full-distribution section for the placeholder-hallucination
  pattern.

The taxonomy explains **why** the smollm3 agent-mode 19% number is
incomparable to the finalists' 66–78% — smollm3 isn't producing
parseable tool intent in any channel. The finalists all have a
working tool channel (either structured for the qwens, or
text-channel-with-parser for granite); their text-channel output
varies in shape but is **secondary signal**, not the primary
dispatch mechanism. For smollm3 there *is no* primary dispatch
mechanism, only the secondary text — and the text doesn't parse.

Deployment implication: agent-mode performance at this size is
gated on the model having a working tool-call channel that survives
the chat-template parser. qwen25-1.5b, qwen25-coder, and granite33
all do (via different mechanisms — qwen's structured channel,
granite's text-channel emission). Smollm3 does not. Picking smollm3
for a deployment that uses agent loops requires either an adapter
rewrite to surface the text channel as tool intent or accepting
that smollm3 will function as a coding-assistant prose generator
rather than a tool-using agent.

## Multi-turn BFCL (rep_5) — state-based failure-mode breakdown

n=100 per category × 4 categories per model. All cells <3% pass rate
(see `graded_report.md` for the headline table). The interesting
structure is in *why* models fail and *which kind* of failure dominates.

### 3-way failure-type split (per model, summed across all 4 cats, n=400)

Categorizes each non-passing problem by *whether the failure is the
model's fault or the infrastructure's*:

| failure type | qwen25-1.5b | qwen25-coder | granite33-2b |
|---|---|---|---|
| **pass** | 0 (0%) | 2 (1%) | 4 (1%) |
| **model_behavior** (wrong call/state, no clarification) | **389 (97%)** | 344 (86%) | 362 (91%) |
| **infrastructure** (grader IndexError on empty steps, backend 400) | 11 (3%) | **54 (14%)** | 34 (9%) |
| **execution** (mock-API runtime crash) | 0 | 0 | 0 |

The `infrastructure` bucket here is dominated by a single underlying
cause: **context overruns**. When the model emits enough verbose calls
that the cumulative prompt exceeds `n_ctx=8192`, llama-server rejects
with HTTP 400. The driver records `n_turns=0, per_turn_steps=[]` for
that problem; bfcl_eval's `multi_turn_checker` then IndexErrors on the
empty list (`grader_crash:IndexError:list index out of range`). Both
the 400 and the downstream IndexError trace back to the same root.
**Don't read the model-behavior pass rates as if these problems were
attempted** — they weren't.

Discounting infrastructure failures, the *attempted* pass rates are:
- granite33: 4/366 = **1.1%**
- qwen25-coder: 2/346 = **0.6%**
- qwen25-1.5b: 0/389 = **0%**

### Failure-reason distribution per category (top 3 per cell)

bfcl_eval's checker tags each failure with one of:
- `multi_turn:instance_state_mismatch` — at some turn, the model's mock-API state diverged from GT's
- `multi_turn:execution_response_mismatch` — same call shape but different return values (mock API responded differently)
- `multi_turn:empty_turn_model_response` — model emitted no calls when GT expected ≥1
- `grader_crash:IndexError:*` — checker errored on our empty/partial output (infrastructure, see above)

Dominant reason per (model, category) cell:

| model | mt_base | mt_long_context | mt_miss_func | mt_miss_param |
|---|---|---|---|---|
| qwen25-1.5b | state_mismatch 59 / response_mismatch 35 | state_mismatch 58 / response_mismatch 29 / **crash 7** | **empty_response 59** / state 23 | state 57 / response 35 |
| qwen25-coder | state 61 / response 35 | state 46 / **crash 30** / response 20 | state 59 / response 32 | state 59 / response 36 |
| granite33-2b | state 50 / empty 25 / response 23 | state 42 / empty 25 / **crash 17** | **empty_response 74** / state 14 | state 46 / empty 32 |

Key reads:

1. **`multi_turn_miss_func` brings out each model's structural personality**:
   - granite33: **74/100 `empty_turn_model_response`** — the decline-
     discipline we saw on single-turn `live_irrelevance` (97/100) is
     here too. When the right tool is excluded, granite says nothing
     rather than improvise with the wrong tool. *But miss_func sometimes
     requires the model to use an alternative — abstention isn't always
     correct here*, and that's exactly where granite leaves points on
     the table.
   - qwen25-1.5b: **59/100 empty_response** — same pattern, less
     extreme.
   - qwen25-coder: only 7/100 empty — coder *tries* something with
     whatever tool is available, which produces `state_mismatch`
     (59/100) rather than empty. Different failure mode, same outcome
     (both fail), but qualitatively a different model behavior.

2. **`multi_turn_long_context` is half-infrastructure for coder**: 30
   `grader_crash:IndexError` + 49 backend 400s out of 100 problems means
   roughly half of qwen25-coder's long_context problems never got fairly
   attempted on this n_ctx. Same root cause: coder's verbose
   tool-call style + cumulative conversation history quickly exceeds
   8k tokens.

3. **state_mismatch is the dominant model_behavior failure across the
   board** — for the problems that *do* execute, the model emits a
   plausible-looking call sequence whose end-state simply doesn't
   match GT. This is the "right area, wrong specifics" failure: the
   model picks the right filesystem method but with the wrong path,
   or grep's the wrong file. The qualitative right answer at this
   model size, but state-grading is unforgiving.

### Mean turns per problem (how far conversations get)

| category | qwen25-1.5b | qwen25-coder | granite33-2b |
|---|---|---|---|
| multi_turn_base (4 turns) | 3.30 | 3.29 | 3.27 |
| multi_turn_long_context (4 turns) | 2.98 | **1.92** | 2.47 |
| multi_turn_miss_func (5 turns) | 4.30 | 4.27 | 4.23 |
| multi_turn_miss_param (5 turns) | 4.30 | 4.27 | 4.26 |

Models complete most of the conversation on `_base`, `miss_func`, and
`miss_param`. **`long_context` is short for coder (1.92 mean turns)**
because backend 400s abort conversations partway through — the same
context-overrun pattern again. The conversation literally doesn't reach
turn 3 or 4 for half of coder's long_context problems.

### What this means for the deployment decision

On this hardware (`n_ctx=8192`), multi-turn agent-loop deployment is
*not viable* for any of the three models at the pass-rate ceiling
state-tracking demands. **None of them are within striking distance
of usable multi-turn performance.** That isn't a model-comparison
signal — it's a *capability-ceiling* signal at this size class.

Three secondary signals that *do* matter for picking among the three:

- **granite33 is the most disciplined** (highest pass rate, lowest
  infra-failure rate) but its decline-instinct over-fires on
  `miss_func` (74/100 empty responses). If your deployment surfaces
  partial-tool-coverage scenarios, granite under-emits.
- **qwen25-coder is verbose to a fault**: 49 backend overruns and
  context utilization at p95=134k tokens means coder fills up `n_ctx`
  faster than it solves problems. Avoid for any deployment with
  conversation-history accumulation unless `n_ctx` is at least 32k.
- **qwen25-1.5b is the cleanest infrastructure profile** (11 errors,
  lowest tokens) but solves zero — its first-pass strength on
  single-turn doesn't transfer to multi-turn at this size.

## HumanEval — failure modes & temperature behavior

All three are 100% extraction-clean on rep_0/rep_2; rep_3 has 1
extraction miss for qwen25-1.5b (`n_extract_ok=163/164`). Across the
1,476 attempts (164 × 3 temps × 3 models) total extraction-fail count
is 1. Failures are correctness, not code-shaping.

### Head-to-head at t=0.0 (the comparable baseline)

Recomputed on fresh M5 rep_0 data. Shape essentially identical to the
prior edition; bucket counts shift by 1–3 problems each due to
stochastic re-draw (Metal kernel reductions aren't bit-identical even
at fixed seed).

| outcome | qwen25 / coder / granite | count |
|---|---|---|
| all pass | P P P | 70 (43%) |
| all fail | F F F | 35 (21%) |
| only coder | F P F | 23 (14%) |
| qwen25 + coder, not granite | P P F | 15 (9%) |
| qwen25 + granite, not coder | P F P | 7 |
| coder + granite, not qwen25 | F P P | 6 |
| only granite | F F P | 4 |
| only qwen25 | P F F | 4 |

- **qwen25-coder uniquely solves 23 problems** the other two can't (14%
  of the set). Same count as the prior edition — these are mostly
  algorithmic with edge-case handling.
- **qwen25-coder uniquely misses 4** — simple list manipulation where
  it over-engineers. Real but small.
- **35 problems all 3 fail** — these are the genuinely hard 21% of
  HumanEval at this size class.

### Temperature stability

| model | t=0.0 | t=0.3 | t=0.7 | pass-all-3 | pass-any | swing |
|---|---|---|---|---|---|---|
| qwen25-coder | **114** (69.5%) | 109 (66.5%) | 107 (65.2%) | 92 (56.1%) | 128 (78.0%) | 36 |
| qwen25-1.5b | **96** (58.5%) | 93 (56.7%) | 80 (48.8%) | 70 (42.7%) | 105 (64.0%) | 35 |
| granite33 | 87 (53.0%) | **88** (53.7%) | 85 (51.8%) | 68 (41.5%) | 106 (64.6%) | 38 |

Shape:
- **qwen25-coder degrades monotonically** (114 → 109 → 107). Loses
  only 7 problems going t=0.0 → t=0.7 — most temperature-stable model
  in absolute terms (swing also tied lowest at 36).
- **qwen25-1.5b** peaks at t=0.0 in this edition (no inverted-V).
  Falls 16pp at t=0.7 — the most temperature-sensitive of the three.
- **granite33** is effectively flat across all three temperatures
  (53.0% → 53.7% → 51.8% is within noise). Most t=0.7-stable in
  *relative* terms (only loses 1.2pp) even though its absolute pass
  rate is lowest.

The pass-any column is the model's effective ceiling with best-of-3
sampling. qwen25-coder's pass-any (128/164 = 78.0%) is still solidly
ahead of the others' pass@1 ceilings, by ~20pp.

## Round 3 prompt-engineering experiments — failure-bucket movement (rep_6)

Per-branch, did the variant move the targeted bucket? Each table shows
the bucket count for non-passing problems, base→variant.

### Branch A — qwen25-1.5b · v3a · curated `irrelevance` (target = `called_tool_when_irrelevant`)

| bucket | rep_1 | rep_6 v3a | Δ |
|---|---|---|---|
| PASS | 86/150 | 77/150 | **-9** |
| over_called (`called_tool_when_irrelevant`) | 64/150 | 73/150 | **+9** ← wrong direction |

The imperative made the over-call bucket *bigger*. The "MUST NOT call
unless fully satisfies" prefix appears to act more like a salience cue
("calls are being discussed") than a restraint cue — the model emits
more, not fewer, when told not to over-emit. The collateral live-cat
damage (`live_simple` -15pp, `live_irrelevance` -23pp) confirms the
boundary moved system-wide, not just on the target slice.

### Branch B — qwen25-coder · v2_fewshot_parallel · `parallel` (target = `under_called_1_of_N`)

| bucket | rep_1 | rep_6 v2_fewshot | Δ |
|---|---|---|---|
| PASS | 32/150 | 25/150 | -7 |
| `under_called_1_of_N` (collapse to single call) | 111/150 | 102/150 | -9 |
| `emitted_0_calls_expected_N` (no calls at all) | 0/150 | 20/150 | **+20** ← new failure mode |
| `under_called_other` / wrong call | 7/150 | 3/150 | -4 |

Few-shot examples did move 9 problems out of the single-call collapse
bucket — but pushed 20 *new* problems into outright silence. The
model is interpreting the few-shot block as part of the user message
in some fraction of cases ("the parallel calls already happened, no
further action needed") rather than as a pattern to imitate. Net
movement on the target bucket is negative (-7 pass).

### Branch C — granite33-2b · v3c · `multi_turn_miss_func` (target = `empty_turn_model_response`)

| bucket | rep_5 baseline | rep_6 v3c | Δ |
|---|---|---|---|
| PASS | 0/100 | 0/100 | 0 |
| `empty_turn_model_response` | 74/100 | 71/100 | -3 |
| `instance_state_mismatch` (wrong tool, wrong state) | 14/100 | 17/100 | **+3** |
| `execution_response_mismatch` | 11/100 | 12/100 | +1 |
| `grader_crash` | 1/100 | 0/100 | -1 |

The 1:1 trade between `empty_response` and `state_mismatch` is the
cleanest signal: the prompt *can* shift the decline boundary, but the
problems that move all break state instead of completing. The
alternative-tool-selection competence required to convert these into
passes is not present at this size.

Collateral irrelevance buckets (per round-3 trade-off framing):

| cat | bucket | rep_1 baseline | rep_6 v3c | Δ |
|---|---|---|---|---|
| irrelevance (first 100) | `called_tool_when_irrelevant` | 15/100 | 20/100 | +5 |
| live_irrelevance | `called_tool_when_irrelevant` | 3/100 | 2/100 | -1 |

The over-call cost concentrated on the curated `irrelevance` set
(easier-to-trigger over-calls), with `live_irrelevance` essentially
flat — the boundary shift didn't reach the live distribution.

## Round 4 — Branch D model-comparison disruptor (rep_1 / rep_0)

Three new ≤3B candidates run through curated BFCL + HumanEval + MBPP.
Live cats / multi-turn / agent / temp sweep deferred. Per-model
failure shape on the curated 5-cat BFCL (n=750 each):

### gemma2-2b-it — strictly dominated

| cat | pass | dominant failure |
|---|---|---|
| simple_python | 126/150 | wrong-tool selection (16) + no_calls_emitted (7) |
| multiple | 124/150 | wrong-tool selection (15) + no_calls_emitted (1) |
| parallel | 49/150 | under-called (101 problems) — same parallel collapse pattern as qwen25-coder |
| parallel_multiple | 18/150 | under-called (132) — even worse than coder's 36/150 |
| irrelevance | 47/150 | over_called_when_irrelevant ×103 |

Gemma 2 has **no native function-calling chat template** in
llama.cpp's `--jinja` resolution; in structured mode it emits
Python-like pseudocalls that no parser recognises (e.g. `area(10, 5)`
as plain text). Inject mode rescues the format but the underlying
weights still trail. Strictly dominated on every BFCL category by
every existing finalist *except* parallel (vs qwen25-coder's 32/150).

### llama32-3b-instruct — middling, weak decline

| cat | pass | dominant failure |
|---|---|---|
| simple_python | 91/150 | wrong-tool selection (59) — verbose tool selection without grounding |
| multiple | 100/150 | wrong-tool selection (50) |
| parallel | 89/150 | under-called (61) — better than coder but worse than qwen25-1.5b |
| parallel_multiple | 82/150 | under-called (68) |
| irrelevance | **31/150** | over_called_when_irrelevant **×119** ← weakest decline discipline of the 6 models |

Llama-3.2-3B's irrelevance pass rate (21%) is the lowest BFCL
irrelevance number anywhere in the project's data. It will call a
tool with high confidence even when the toolbox is wrong for the
question. Coding numbers (HE 57.9%, MBPP 62.1%) are middle-of-pack
but don't beat any existing finalist — qwen25-1.5b matches MBPP and
qwen25-coder beats both.

### smollm3-3b-instruct — the credible challenger

| cat | pass | failure shape vs qwen25-1.5b |
|---|---|---|
| simple_python | 143/150 | +5 vs qwen25-1.5b (138/150) |
| multiple | 131/150 | -2 vs qwen25-1.5b (133/150) |
| parallel | 125/150 | **+7** vs qwen25-1.5b (118/150) |
| parallel_multiple | 116/150 | **+6** vs qwen25-1.5b (110/150) |
| irrelevance | 86/150 | tied with qwen25-1.5b (86/150); identical over_called bucket (64) |

SmolLM3's failure shape on parallel/parallel_multiple is materially
better than qwen25-1.5b: the under_called_1_of_N collapse that hit
qwen25-coder catastrophically (32/150 parallel) and qwen25-1.5b
moderately (118/150) is less severe in smollm3 (125/150). This is the
one place Branch D produced a *mechanical* improvement, not just a
within-CI numeric one. On irrelevance, smollm3 over-calls at exactly
the same volume as qwen25-1.5b — neither model has the granite33-style
decline instinct.

## Full-distribution failure breakdown (rep_7, 2026-05-14)

Phase K reran the three finalists on the full live cats (no
`--bfcl-limit`); smollm3 already had full-live data in rep_1. This
section is the per-cat failure-shape analysis on the full distribution
— the first-100 numbers in the earlier per-finalist sections were
sliced to the same baseline rep_1 used. Source artifacts:
`acceptance/bfcl/<model>/rep_7/` + `acceptance/audits/phase_k_rep7_4way_live_matched_ids.json`.

### live_irrelevance (n=884) — where the decline-corner gap actually lives

The headline `+28.6pp CI-distinct` granite33 lead on full
`live_irrelevance` decomposes into concrete per-problem behavior:

| model | pass | over-call (≥1 emitted) | within-model drop (first-100 → remaining-784) |
|---|---|---|---|
| granite33-2b | 720/884 (81.4%) | 164 (18.6%) | 98% → 79.3% (-18.7pp) |
| qwen25-1.5b | 467/884 (52.8%) | 417 (47.2%) | 77% → 49.7% (-27.3pp) |
| smollm3-3b | 434/884 (49.1%) | 450 (50.9%) | 89% → 44.0% (-45.0pp) |
| qwen25-coder | 271/884 (30.7%) | 613 (69.3%) | 75% → 25.0% (-50.0pp) |

Failure-overlap breakdown — for each smollm3 failure (450 problems),
how many of the other 3 finalists also fail on that same problem?

| smollm3 fails AND … | count | share |
|---|---|---|
| 0 of the other 3 also fail (smollm3-unique) | 20 | 4.4% |
| 1 of the other 3 also fail | 118 | 26.2% |
| 2 of the other 3 also fail | 200 | 44.4% |
| all 3 also fail (universal hard problems) | 112 | 24.9% |

Smollm3 has very few unique failures (20/884 = 2.3% of the
distribution). Most of its failures are shared with at least one
finalist; nearly a quarter (112) are universal — problems where every
model over-calls. The differentiator is the long tail of "smollm3 +
1 or 2 others fail, granite alone succeeds."

**Smollm3-unique failure shape — placeholder-value hallucination.**
Inspecting the 20 smollm3-unique failures reveals a coherent pattern:
smollm3 emits the call with *fabricated placeholder values* to satisfy
the user's stated intent, where the other three finalists recognize
"I don't have the information." Examples from rep_7 (raw_text omitted
for space; full data in `acceptance/bfcl/smollm3-3b-instruct/rep_7/live_irrelevance/`):

- `live_irrelevance_202-*`: emits `user_authentication.login` with
  `{"username": "your_username", "password": "your_password"}` —
  placeholder strings literally used as arg values.
- `live_irrelevance_300-*`: emits `regression_model_predict` with
  `features: [0.5, 0.8, 0.3, 0.7, 0.2]` — invented feature vector
  for an unanswerable user request.
- `live_irrelevance_312-*`: emits `requests.get` against
  `https://api.fixerio.com/v1/latest` with `{lat: 0, lon: 0}` — both
  the URL and the params are model-fabricated.
- `live_irrelevance_322-*`: routes a question about basketball through
  a `requests.get` to a reddit URL with `Accept: application/json`.

Operational implication: deployments that accept placeholder values
silently (e.g. tools that pass them through to downstream systems
without strict validation) inherit a smollm3-specific hallucination
risk on irrelevance traffic. Tools that error on `your_username` /
default-valued numeric features will catch these locally. The pattern
is small in absolute count (20/884 = 2.3%) but distinctive enough to
mark as a model-specific deployment hazard.

### Distribution-robustness only generalizes for `live_irrelevance`

The "first-100 are systematically easier than remaining" pattern
documented above for `live_irrelevance` (all four models drop -19pp
to -50pp) is **NOT a universal dataset shape phenomenon**. Re-running
the same first-100 vs remaining-N comparison on the other two
high-volume live cats reveals mixed-direction behavior:

**`live_multiple` (n=1053; first-100 vs remaining-953)**

| model | first-100 | remaining-953 | drop |
|---|---|---|---|
| smollm3-3b | 65.0% | 63.5% | +1.5pp (flat) |
| qwen25-1.5b | 71.0% | 65.9% | +5.1pp (mild) |
| granite33-2b | 48.0% | 53.4% | **-5.4pp (inverse)** |
| qwen25-coder | 62.0% | 65.3% | **-3.3pp (inverse)** |

Two of the four models do *better* on the remaining 953 problems than
on the first-100 — the easier-first ordering effect is absent on this
cat.

**`live_simple` (n=258; first-100 vs remaining-158)**

| model | first-100 | remaining-158 | drop |
|---|---|---|---|
| smollm3-3b | 73.0% | 63.3% | +9.7pp |
| qwen25-1.5b | 82.0% | 69.6% | +12.4pp |
| granite33-2b | 68.0% | 55.7% | +12.3pp |
| qwen25-coder | 70.0% | 64.6% | +5.4pp |

All four drop on `live_simple` but more modestly than on
`live_irrelevance` (5–12pp vs 19–50pp).

**What this means for the "smollm3 collapses on live cats" framing:**
- The collapse signal **was real** but **specific to `live_irrelevance`**, not a general distribution-robustness deficit.
- On `live_multiple` smollm3 is the most distribution-robust model (+1.5pp = flat).
- On `live_simple` smollm3 drops less than qwen25-1.5b and granite33.
- The original headline mixed `live_irrelevance` (where smollm3 is third-of-four robust) with a more general "live cats" claim. Decomposed: smollm3 is competitive on robustness for active categories and bottom-tier-of-finalists on irrelevance specifically.

For deployment: smollm3's distribution risk concentrates on
irrelevance traffic; for tool-using active categories, smollm3 is
roughly as robust as the qwen siblings.

**Granite33's unique decline advantage: 181 problems** (20.5% of the
distribution) where granite alone declines correctly while all three
other finalists over-call. This is the concrete operational evidence
behind the +28.6pp lead — there are 181 specific problems where, in a
deployment surfacing irrelevance prompts at scale, granite would
correctly refuse and the others would all attempt an unsuitable tool.

### Active live cats (live_simple + live_multiple + parallel cats, n=1351)

| model | pass | mean per-cat rate |
|---|---|---|
| qwen25-1.5b | 913/1351 (67.6%) | live_simple 74.4%, live_multiple 66.4%, live_par 62.5%, live_par_mul 50.0% |
| qwen25-coder | 861/1351 (63.7%) | live_simple 66.7%, live_multiple 65.0%, live_par 18.8%, live_par_mul 8.3% |
| smollm3-3b | 859/1351 (63.6%) | live_simple 67.1%, live_multiple 63.6%, live_par 31.2%, live_par_mul 45.8% |
| granite33-2b | 726/1351 (53.7%) | live_simple 60.5%, live_multiple 52.9%, live_par 25.0%, live_par_mul 37.5% |

Two patterns to read off the active cats:

1. **qwen25-coder's parallel collapse persists on live data** — 18.8%
   on `live_parallel` and 8.3% on `live_parallel_multiple`, both
   substantially below the next-worst (granite33). The same
   `under_called_1_of_N` pattern documented for curated parallel cats
   carries over to user-submitted prompts.
2. **granite33's decline-by-default hurts on active cats** — 53.7%
   overall vs the other three at 63–68%. Same trade documented in the
   curated section (97/100 on irrelevance comes with 37/48 no-call
   failures on `multiple`) plays out at scale.

## Cross-bench observations

1. **The four are non-dominated.** qwen25-1.5b wins active live
   tool-use (rep_7 1351-problem slice), qwen25-coder wins coding,
   granite33 wins decline-discipline (+28.6pp CI-distinct on rep_7),
   smollm3 is the balanced generalist (within CIs of qwen25-1.5b on
   matched BFCL and qwen25-coder on every coding metric). No model
   wins every axis; ranking depends on the deployment profile.

2. **Granite33's no-call instinct cuts both ways.** Same training that
   produces 81.4% on full `live_irrelevance` (n=884) and 181
   problems-where-only-granite-declines produces 37 no-calls of 48
   fails on curated `multiple` and 35 no-calls of 52 on
   `live_multiple` (first-100). The trade is durable across the full
   distribution — granite is bottom-on-active-cats (53.7% on the 1351
   active live problems) and top-on-decline by 28.6pp.

3. **qwen25-coder's parallel collapse is the hard ceiling on both
   curated and live data.** 74% of curated parallel rows (111/150)
   and 65% of curated parallel_multiple rows (97/150) emit exactly
   one call when N≥2 are required; live parallel rates are even lower
   (18.8% live_parallel, 8.3% live_parallel_multiple on rep_7 — both
   bottom-of-field). The non-coder qwen25-1.5b sibling at the same
   n=150 emits the right count comfortably 73–78% of the time.

4. **Live BFCL is harder than curated except on relevance.** Active
   live cats run 5–15pp below the curated equivalents per model. The
   exception is `live_relevance` (16 problems, easy for all but
   granite). Within the full live distribution, the first-100 are
   systematically easier than the remaining-784 — *every* model drops
   substantially on the broader irrelevance distribution (granite -19pp
   to qwen-coder -50pp). This is a dataset-shape phenomenon, not a
   per-model failure mode.

5. **Smollm3's distribution-robustness on irrelevance is third-of-four**
   (-45pp drop from first-100 to remaining-784, between qwen25-1.5b
   at -27pp and qwen-coder at -50pp). Most smollm3 failures (95.6%)
   are shared with at least one other finalist; only 20/884 (2.3%)
   are smollm3-unique. The drop magnitude is real but the failures
   aren't idiosyncratic — smollm3 fails on roughly the same hard
   problems the others fail on.

6. **All four are essentially extraction-clean on HumanEval.** 1
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

- All `rep_*` data in this edition was re-run on a 128 GB M5 Max using
  the parallel-3 `--auto-port` runner path (rep_0 was the final piece
  refreshed — completed alongside the other reps in <6 min wall each).
- Wall times are not directly comparable to the prior edition's 8 GB-Mac
  runs.

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
