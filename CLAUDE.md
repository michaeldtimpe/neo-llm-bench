# Guidance for AI agents working on neo-llm-bench

This file is loaded into Claude Code's context automatically when the working directory is this repo. Keep it concise and current.

## First read for a new session

If you're new to the project, read `BENCHMARKS.md` first. It defines
the five benchmark signals + their orthogonal evaluation dimensions so
you don't conflate `rep_1` (BFCL raw) with `rep_4` (BFCL agent) with
`rep_5` (BFCL multi-turn) — they probe different layers.

## Project shape

llama.cpp-based bake-off harness for small (≤3B) GGUF models. Round 1
narrowed an 8-model roster to 3 finalists (qwen25-1.5b-instruct,
qwen25-coder-1.5b-instruct, granite33-2b-instruct); Round 4 / Branch D
added smollm3-3b-instruct as the 4th model (competitive on tool-use +
coding, axis-loser on decline).

**Rounds 1–4 are complete**, plus a 2026-05-14 audit-correction cycle
(Phase I+J+K+L) that:
- Fixed a lexicographic-sort slicing bug in cross-model BFCL deltas
  (commit `e50fdce2`). All cross-model claims now route through
  `scripts/compare_matched_slice.py` with matched-ID artifacts
  in `acceptance/audits/`.
- Added BFCL `raw_text` persistence (commit `3905bd7f`) and verified
  the smollm3 agent-mode mechanism (dominant bucket: prose_only at
  ~61%, not "Python code blocks" at ~38% as previously asserted).
- Reran finalists on full live cats (rep_7, commit `f7bbf35d`) so
  cross-model comparisons on the live distribution are properly
  apples-to-apples. ~56 min wall.
- Final consistency sweep + champion-framework update (commit
  `af56bb3c`).

Current non-dominated quadrilateral: qwen25-1.5b leads active tool-use
(rep_7 1351 problems), granite33 leads decline-discipline (rep_7
live_irrelevance n=884, +28.6pp CI-distinct), qwen25-coder leads
HumanEval, smollm3 is the balanced generalist (within CI of qwen25-1.5b
on BFCL and qwen25-coder on every coding metric).

See `graded_report.md` for the full leaderboard + the
"Errata and methodology corrections" section near the end for the
audit trail. `graded_failure_modes.md` for the per-model failure shape
(curated cats, pre-audit but unaffected by the bug).

**The user picks the champion from the data.** Do not recommend a
winner unprompted.

## Hardware envelope (changed during round 2)

The project was originally designed for 8 GB Macs (see `profile_8gb.yaml`).
Round 2 BFCL `rep_1` / HumanEval `rep_0` data was generated on that
hardware. Round-2 reruns + Phase A–E work was done on a 128 GB M5 Max
(see `profile_m5max.yaml`). Practical implications:

- The "2B models swap-thrash on this box" warning is **only true with
  `profile_8gb.yaml`** — not with `profile_m5max.yaml`. Pick the right
  profile via `--profile configs/profile_<name>.yaml`.
- Wall times across these two hardware environments are **not
  comparable**. Pass rates are comparable (modulo small stochastic
  variance from non-bit-identical Metal kernels).
- Multi-turn `long_context` regularly busts `n_ctx=8192` (49/100
  problems for qwen25-coder hit HTTP 400). Not a model bug; an
  infrastructure ceiling. Document if the report cell is affected.

## Operating principles

1. **Confirm scope before launching long runs.** The bake-off has
   cost real time before. Always restate the intended set of (models
   × benchmarks × reps × temperatures × prompt variants) and the
   expected wall time before kicking off anything multi-hour. See
   `lessons.md` for the precedent.
2. **Round-2 finalists are the only models worth re-running.** The 5
   round-1 cut models (smollm2, llama32, deepseek-coder,
   deepseek-r1-distill, phi-1.5) are appendix-only.
3. **Use `time.monotonic()`-derived wall times, not wall-clock.**
   macOS sleep pauses monotonic; the runner's reported wall is
   compute time.
4. **`raw_text` is persisted on all HumanEval/MBPP reps from 2026-05-13
   forward; BFCL persistence landed 2026-05-14 (Phase J).** For BFCL,
   `raw_text` is optional; legacy reps without the field remain
   readable. Semantics: raw mode = `ChatResponse.text` from the single
   call; agent mode = concatenated assistant turns joined by `"\n---\n"`.
   See `ARCHITECTURE.md` "Per-problem persistence and `raw_text`
   semantics" for the full schema contract.
5. **Provenance discipline.** When citing numbers across rounds, link
   to the source rep + grading artifact. `round_3_planning.md`'s
   matrix is the pattern.
6. **`audit_one_multi_turn.py` for multi-turn regrading.** bfcl_eval's
   `globals()` instance cache pollutes re-grade calls in the same
   process. Use subprocess isolation for any multi-turn audit.
7. **Cross-model BFCL deltas go through `scripts/compare_matched_slice.py`.**
   Lex-sort over numeric problem IDs is invalid (string order puts
   `live_simple_10` before `live_simple_2`). The helper takes
   `<model>:<rep>` targets and an explicit `--policy {intersection,union}`
   and persists matched-ID sets to `acceptance/audits/`. Regression
   test: `tests/test_matched_slice.py`. See
   `memory/feedback_slicing_methodology.md` for the operational rule.

## Code conventions

- `uv` for env management; `uv run python ...` to invoke scripts.
- Python 3.11 (pinned via `.python-version`).
- Use `Path.expanduser()` on user-facing paths containing `~`.
- Per-problem JSON file naming: `<problem_id>.json`. BFCL IDs use
  dashes; live IDs like `live_simple_0-0-0` are fine on Mac.
- `summary.json` gets **overwritten** each time a step runs. For
  per-category aggregates across multiple invocations, walk per-
  problem files directly — `scripts/grade_bakeoff.py` does this.
- `metadata.json` (since Phase A½) lands next to each `summary.json`
  with GGUF SHA, llama.cpp commit, host info, and run mode.
- New benchmarks register a `BenchmarkSpec` in
  `src/llamabench/runner.py:_BENCH_RUNNERS` with their own
  `force_clean_filenames` set.

## Common task patterns

### "Run the bake-off"

`scripts/run_bakeoff.py` — key flags:
- `--rep N` (always pass explicitly)
- `--auto-port` (mandatory for parallel multi-model runs)
- `--port N` (explicit override)
- `--force` (overwrite — also deletes per-bench resume files via the
  registered `BenchmarkSpec.force_clean_filenames`)
- `--profile configs/profile_m5max.yaml` (or `profile_8gb.yaml`)
- `--bfcl-mode {auto,structured,inject}` (raw-mode tool delivery)
- `--bfcl-run-mode {raw,agent}` (raw vs closed-loop agent dispatch)
- `--bfcl-categories <...>` (subset; `multi_turn_*` cats use the
  multi-turn driver automatically via `is_multi_turn(cat)`)
- `--bfcl-limit N` / `--humaneval-limit N` / `--mbpp-limit N`
- `--temperature F` (CLI override; doesn't modify per-model YAMLs)

Resume: steps with existing `summary.json` are skipped by default.
`--force` clears the per-bench resume files (currently `summary.json`
+ for HumanEval/MBPP `results.jsonl`) before re-running.

For multi-model parallel runs, **stagger launches ~60s apart** to
avoid synchronized tokenizer/model-load contention. Each pipeline gets
its own port via `--auto-port`.

### "Grade the data" / "Regenerate reports"

```bash
uv run python scripts/grade_bakeoff.py --rep N             # markdown leaderboard (includes mt_* cats)
uv run python scripts/grade_bakeoff.py --rep N --json      # machine-readable
uv run python scripts/grade_bakeoff.py --rep N --write-back  # stamp passed/reason into per-problem JSON
uv run python scripts/failure_modes.py --rep N             # per-model failure buckets
```

For multi-turn (`rep_5` shape) the grader calls bfcl_eval's
`multi_turn_checker` — see `benchmarks/bfcl/grade.py:grade_multi_turn`.

`graded_report.md` and `graded_failure_modes.md` are hand-written
narrative; regenerate from the script outputs above.

### "Verify the numbers" (audit pattern)

Mirror the Phase A–D audit shape:
1. Walk per-problem files independently and re-count `passed` totals;
   compare to the grader's printed table.
2. Sample 5 rows per (model, cat); re-grade independently.
3. For multi-turn, use `scripts/audit_one_multi_turn.py` invoked in
   fresh subprocesses (not just unique model_name — bfcl_eval has
   `globals()` cache + other module-level state that subprocess
   isolation cleanly defeats).
4. Re-compute Wilson 95% CIs from raw pass counts.
5. **Mismatch = halt, not auto-restamp.** Mismatches mean either
   grader non-determinism or audit isolation gaps; both require
   diagnosis before the report ships.

### "Background runs"

For runs >5 min, use `Bash` with `run_in_background: true`. Always:
- Tee output to `acceptance/_logs/<run-name>.log`
- Arm a `Monitor` filtering for `\[OK  \]` / `bake-off complete` /
  `=== ALL-DONE` + error patterns (`Traceback|FAILED|Error:|Killed|
  OOM|crashed|exited with code`)
- Pre-warn the user of the wall estimate before kicking off
- Stagger parallel launches by 60s — see "Run the bake-off"

## Things that have bitten us

See `lessons.md` for the canonical list. Highlights:

- **Scope mistakes**: don't extrapolate "X needs doing" without
  confirming with the user.
- **`summary.json` is overwritten**: walk per-problem files for
  aggregates.
- **`--force` used to not honor HumanEval/MBPP per-problem resume**
  → silent stale results. Fixed Phase A; `_clear_stale_for_force`
  helper + per-`BenchmarkSpec.force_clean_filenames` registry.
- **bfcl_eval's `globals()` instance cache** can replay multi-turn
  conversations against dirty state if you re-grade in the same
  process. Use `scripts/audit_one_multi_turn.py` subprocess.
- **Multi-turn `long_context` busts `n_ctx=8192`** for verbose models
  (qwen25-coder: 49/100 HTTP 400s). Per-step prompt-token capture is
  the cleanest fix; current trace is cumulative-across-steps.
- **System sleep distorts wall**: `time.monotonic()` pauses; report
  compute time, not wall-clock.

## Memory

User-level memory for THIS project lives in
`~/.claude/projects/-Users-mtimpe-Downloads-neo-llm-bench/memory/`.
(Older project location was `Documents/llama-bench`; not used for this
repo.) Read those at session start.
