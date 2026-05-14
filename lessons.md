# Lessons (mistakes & non-obvious findings)

Hard-won lessons from the bake-off. Read before starting fresh work.

## Process & scope

### 1. Confirm scope before launching long runs (2026-05-12)

**What happened**: After the 9 May 10-problem run, the on-disk `graded_report.md` was stale (some models had been re-run at 30/cat with a new system prompt; others hadn't). I extrapolated "the right thing to do is re-run the 5 no-prompt models for apples-to-apples comparison" and kicked off a multi-hour background run. After ~11 hours, the user clarified: round 2 had already narrowed to 3 finalists. The 5 re-runs were wasted compute.

**Lesson**: When the user says "the bake-off has run," ask what round / what scope before extrapolating from data alone. The narrative scope ("we're picking a champion between 3 finalists") doesn't live in the data files; it lives in the user's head.

**How to apply**: Before launching anything multi-hour, restate the intended scope and time budget back to the user. Wait for an OK.

### 2. Estimate wall time from real data, not theoretical rates (2026-05-12)

**What happened**: Estimated the 8h multi-spectrum run at 6.7h based on prior 30/cat wall times. The actual run took ~26h wall clock — partly because problems 30+ were harder on some categories, mostly because the Mac was sleeping between batches and the runner couldn't make progress when asleep.

**Lesson**: `time.monotonic()` pauses during macOS sleep, so the runner reports accurate *compute* time. Wall clock can be 4–10× longer. Communicate both numbers when estimating.

### 3. summary.json is not authoritative (2026-05-13)

**What happened**: While auditing the final reports, the resource cost table was wrong because `summary.json` was the last thing each step wrote — and step 2 (live BFCL) overwrote step 1's standard-BFCL summary, so it contained only live-category totals. The per-problem JSONs had the full data.

**Lesson**: For aggregate stats across multi-step runs of the same `(model, bench, rep)`, walk per-problem files. `scripts/grade_bakeoff.py` does this correctly; manual computations from `summary.json` will undercount.

## Hardware & system

### 4. 2B models swap on an 8 GB Mac (2026-05-12)

**What happened**: Granite33-2b-instruct's BFCL per-problem wall time degraded from 3.6 s/problem (simple_python) to 176 s/problem (parallel) over the first few hours of a deep run. Looked like a runaway loop. It wasn't — diagnostic showed 6042 MB swap used out of 7168 MB total, with the granite llama-server eating 3 GB RSS. Swap reads on every model forward pass.

**Lesson**: 2B Q8_0 + n_ctx 8192 + Q8_0 KV pushes 8 GB Macs into heavy swap under sustained load. The 1.5B Qwen models fit cleanly in RAM. The Granite finalist took ~2.5× wall time on this machine for the same workload. Budget accordingly when planning multi-hour runs.

**How to apply**: For long granite runs, expect compute-to-wall ratios of 1:3 or worse. Don't use early-category timing to extrapolate later categories.

### 5. System sleep is hidden in the wall clock (2026-05-12)

**What happened**: A run that took 10h 8m wall clock reported `wall=1:32:00` in the runner output. The python `time.monotonic()` paused while macOS slept (lid closed, idle, etc.).

**Lesson**: When wall-clock time and reported wall don't match, suspect sleep, not slowdown. Per-problem `wall_s` in the JSONs is accurate compute time; sum of those is the true runtime.

**How to apply**: If you need contiguous wall-clock time (e.g. for resource sampling), use `caffeinate -d` to inhibit display sleep during the run.

## BFCL grader specifics

### 6. Nested-dict allowed-lists need recursion (2026-05-09)

**What happened**: Five models were silently failing `multiple_8` (realestate.find_properties with `budget: {min, max}`) and `multiple_9` (calculate_average with `gradeDict`) on structurally-correct emissions. BFCL v4 wraps *every* leaf inside dict-typed args in its own allowed-list — including leaves inside dict-typed args. The grader's `_value_matches` was doing plain `==`.

**Lesson**: BFCL v4's allowed-list shape is recursive. The fix is `_dict_shape_matches` which treats dict allowed-entries as shape templates, not equality targets. Tests in `tests/test_bfcl_grade.py:test_value_matches_nested_dict_*`.

### 7. Math-notation false negatives (2026-05-11)

**What happened**: `parallel_multiple_4`-style problems had model emissions like `3*x**2 + 2*x - 1` versus BFCL GT `3x**2 + 2x - 1` — semantically identical, plain-`==` mismatch.

**Lesson**: Added `_normalize_math_expr` which collapses `<digit>*<ident>` → `<digit><ident>` on both sides before comparison. Conservative scope: it does NOT rewrite `^` → `**` (in Python `^` is XOR; emitting `x^3` is a real error). Tested.

### 8. live_irrelevance and live_relevance have no possible_answer file (2026-05-12)

**What happened**: When adding live BFCL support, the grader expected a `possible_answer/BFCL_v4_<category>.json` for every category. BFCL v4 ships none for `live_irrelevance` (pass criterion: zero calls) or `live_relevance` (pass criterion: at least one call).

**Lesson**: Both live categories are graded purely on call-count semantics, no GT match. `load_ground_truth` returns `{}` for these. Dispatcher in `grade()` routes them to `grade_irrelevance` and `grade_relevance` respectively.

## Round-1 model-specific findings (preserved for context)

### 9. Single-call collapse is partly system-prompt-tractable

**What happened**: qwen25-coder pre-prompt: 0/10 parallel, 1/10 parallel_multiple — every parallel problem collapsed to one tool call with array-packed args. With v2 system-prompt rule 1 ("emit N separate calls for N inputs"), the curated parallel single-call rate dropped from 29/30 → 16/30 (still 51% collapse at scale).

**Lesson**: Coder-tuned models bias toward "complete the code I'm writing for you" rather than "match the schema and stop." A system prompt rule can recover ~40% of the lost ground; the rest is a training-distribution wall.

### 10. Irrelevance over-calling is the universal instruct-model trap

**What happened**: All round-1 instruct models scored worse on `irrelevance` (3-4/10) than on the 4 active categories (27-31/40). The training signal "use the tool when given one" overrides the user's actual intent. The v2 system prompt rule 2 ("don't call if tools can't satisfy the user") moves granite33 from 12/30 → 25/30 on this category — biggest single-rule improvement in the field.

**Lesson**: For tool-use deployments where the user can ask for things outside the tool surface, the model's irrelevance behavior is the highest-leverage trait to evaluate. Granite33's 100/100 on `live_irrelevance` makes it the deploy-time pick if irrelevance traffic is meaningful.

### 11. Granite's no-call instinct cuts both ways

**What happened**: The same training signal that produces granite33's 100/100 on `live_irrelevance` produces 37/48 of its `multiple` failures and 35/52 of its `live_multiple` failures as "no calls emitted." When the toolbox has multiple candidates, granite tends to decline rather than guess.

**Lesson**: There isn't a single "best" calibration. Granite's discipline costs it ~12pp on tool-use categories where the right tool is one-of-many. qwen25-1.5b takes the opposite trade: emits aggressively, scores 16pp better on tool-use, costs 28pp on irrelevance.

### 12. Reasoning models don't fit at this size (2026-05-09)

**What happened**: deepseek-r1-distill-qwen-1.5b's `<think>` blocks consumed the entire 8k completion budget on every call-emitting BFCL row. Zero parseable tool calls in 40 in-scope rows. The 10/10 on irrelevance was mechanical (it can't call tools, so it can't over-call).

**Lesson**: Reasoning models need either a constrained `<think>` budget or a strip-and-re-prompt pass before they can be evaluated on tool-use at this size. Distilled R1 was cut from round 2 as a result.

### 13. Lex-sort over numeric problem IDs is silent corruption (2026-05-14)

**What happened**: Phase H's BFCL "apples-to-apples (n=1106)" table for smollm3 used `sorted(cat_dir.glob("*.json"))[:100]` to slice the full live cats down to 100 problems for comparison against the finalists' 100-per-cat baseline. Python lexicographic sort returns `live_simple_10` before `live_simple_2` and `live_simple_100` before `live_simple_11`. The "first 100" picked a chaotic subset that overlapped the finalists' actual reference 100 by only 7–11 problems out of 100. Same bug propagated through Round 3 Branch A and Branch C "first-100" comparisons.

Headline downstream impact: smollm3 published as 805/1106 (72.8%) on matched data is actually 860/1106 (77.8%); the `live_irrelevance` "-20pp CI-distinct collapse" is actually **+11pp** when measured on the same problems; Branch C's collateral irrelevance harm gate published as PASS (-4pp combined) actually FAILS on matched data (-7pp combined).

**Lesson**: Never lex-sort numeric problem IDs. Cross-model or cross-rep deltas must intersect problem-ID sets, not slice positionally. `[:N]` over a globbed list is itself a code smell on this kind of data.

**How to apply**: All cross-model BFCL deltas go through `scripts/compare_matched_slice.py` with explicit `--policy intersection` and matched-ID artifact persistence in `acceptance/audits/`. Regression test in `tests/test_matched_slice.py::test_intersection_does_not_pick_lex_mangled_first_100` fails if the helper ever regresses. Memory: `feedback_slicing_methodology.md`.

### 14. Mechanism claims need persisted evidence (2026-05-14)

**What happened**: Phase H asserted "smollm3 emits Python code blocks instead of structured tool calls" in agent mode. The 240/1240 artifact count was real (n_with_calls=0 across all problems), but the per-problem JSON shape for rep_4 had no `raw_text` field — only `actual_calls` (empty everywhere). The specific text shape was inferred without persisted data to check. After Phase J added the field and re-ran rep_4, the dominant bucket turned out to be **prose** (61%, math/explanation text), not code blocks (38%). The prior claim captured the visually-striking minority.

**Lesson**: When a report claims "model emits X instead of Y" in failure analysis, the underlying X must be in the artifact files at audit time. If the persistence path doesn't capture the relevant text, the claim is unverifiable and should be flagged as such, not asserted.

**How to apply**: Per-problem JSONs persist `raw_text` since Phase J (BFCL specifically; HumanEval/MBPP earlier). Schema contract in ARCHITECTURE.md "Per-problem persistence and `raw_text` semantics" — raw mode = single `ChatResponse.text`; agent mode = assistant turns joined by `\n---\n`. Mechanism classification via `scripts/sample_raw_text.py --seed <s> --n <n>` with the 6-bucket taxonomy. Persist samples to `acceptance/audits/` for reproducibility.

### 15. Matched-quality vs distribution-robustness are different claims (2026-05-14)

**What happened**: Phase H's "smollm3 collapses on live cats" headline conflated two separately measurable phenomena: (a) head-to-head pass-rate on the same problems (matched-quality) and (b) within-model performance change when the problem distribution broadens (distribution-robustness). The published evidence for the collapse claim was largely the matched-quality table with a slicing bug. After correction: smollm3's matched-quality numbers are statistically tied with qwen25-1.5b (and above on live_irrelevance). Smollm3's *within-model* live distribution drop (89% on first-100 → 49% on the full 884) is real, but is a different claim — and the finalists hadn't been measured on full-live data, so no cross-model collapse comparison was actually evidenced.

**Lesson**: Every cross-model claim must specify whether it's (a) head-to-head on the same tasks or (b) within-model behavioral drift. Mixing them produces conclusions that look strong but rest on incompatible evidence.

**How to apply**: Report structure now splits "Subsection 1: matched-ID" from "Subsection 2: full-distribution" for BFCL cross-model claims. Phase K rep_7 (finalists on full live cats) turns within-model distribution signals into proper cross-model comparisons.

## What's still unknown

- **BFCL multi-turn**: data exists (800 problems across 4 categories), grader is deferred. Need state-tracking; not trivially additive to the current single-turn grader.
- **MBPP**: planned as a second coding bench; not yet wired. HumanEval is the only coding signal in round 2.
- **Whether the live_irrelevance 100/100 result generalizes**: n=100 with CI lower bound 96%. A larger live_irrelevance sample (~500) would tighten this, but the pattern is so clean it would take a strong reversal to dislodge granite's lead on this axis.
- **Agent-mode (full run_agent loop) BFCL numbers**: scaffolded but not in the bake-off. The current numbers are raw mode (single-turn `backend.chat()`) — comparable to public BFCL but not measuring the harness's scaffolding.
