# Guidance for AI agents working on neo-llm-bench

This file is loaded into Claude Code's context automatically when the working directory is this repo. Keep it concise and current.

## Project shape

This is a llama.cpp-based bake-off harness for small (≤2B) GGUF models on an 8 GB Mac. Round 1 narrowed an 8-model roster to 3 finalists (qwen25-1.5b-instruct, qwen25-coder-1.5b-instruct, granite33-2b-instruct). Round 2 is the multi-spectrum capability test — the data is in `acceptance/`, reports in `graded_report.md` and `graded_failure_modes.md`.

**The user picks the champion from the data.** Do not recommend a winner unprompted.

## Operating principles

1. **Confirm scope before launching long runs.** The bake-off has cost real time before due to scope mistakes — always restate the intended set of (models × benchmarks × reps × temperatures) and the expected wall time before kicking off anything multi-hour. See `lessons.md` for the precedent.
2. **Round-2 finalists are the only models worth re-running.** The 5 round-1 cut models (smollm2, llama32, deepseek-coder, deepseek-r1-distill, phi-1.5) are appendix-only. Do not re-run them for new comparisons unless the user explicitly asks.
3. **Use `time.monotonic()`-derived wall times, not wall-clock.** macOS sleep pauses monotonic. The runner's reported wall (`summary.json:wall_s`) is compute time. The wall clock can be 4–10× longer due to sleep gaps. Don't interpret wall-clock as runtime degradation.
4. **2B models swap on this machine.** Granite33-2b-instruct degrades per-problem wall by ~2.5× vs the 1.5B siblings due to swap pressure (~6 GB swap used at steady state). Budget accordingly.
5. **The HumanEval results.jsonl now persists `raw_text`** (since 2026-05-13). Earlier runs do not. If you need to re-execute, check that field exists; if not, you can only trust the recorded `passed` value.

## Code conventions

- `uv` for env management; `uv run python ...` to invoke scripts.
- Python 3.11 (pinned in `.python-version` and `pyproject.toml`).
- Use `Path.expanduser()` on any user-facing path that might contain `~`. The runner and DEFAULT_BIN do this; new code should too.
- Per-problem JSON file naming: `<problem_id>.json` (BFCL IDs use dashes; live IDs like `live_simple_0-0-0` are fine on Mac).
- summary.json gets *overwritten* each time a step runs (it reflects only the categories run in that invocation). For per-category aggregates across multiple invocations, walk per-problem files directly — `scripts/grade_bakeoff.py` does this.

## Common task patterns

### "Run the bake-off"

Use `scripts/run_bakeoff.py`. Always pass `--rep N` explicitly. For temperature sweeps, use `--temperature` (CLI override, doesn't modify per-model YAMLs). For category subsets, use `--bfcl-categories`. Use `--force` to overwrite an existing rep's summary.

Resume semantics: by default, steps with an existing `summary.json` are skipped. HumanEval per-problem resume reads existing `results.jsonl` and only re-runs missing task_ids.

### "Grade the data" / "Regenerate reports"

```bash
uv run python scripts/grade_bakeoff.py --rep N             # markdown leaderboard
uv run python scripts/grade_bakeoff.py --rep N --json      # machine-readable
uv run python scripts/grade_bakeoff.py --rep N --write-back  # stamp passed/reason into per-problem JSON
uv run python scripts/failure_modes.py --rep N             # per-model failure buckets
```

`graded_report.md` and `graded_failure_modes.md` are hand-written narrative; regenerate from the script outputs above.

### "Verify the numbers"

Audit pattern from session log: re-grade per-problem files from scratch (don't trust stamped `passed`), compare to summary.json, sample-execute HumanEval rows, compute Wilson CIs independently. Be alert to confusing "fail rate" with "specific-failure-mode rate" — they're different denominators.

### "Background runs"

Use `Bash` with `run_in_background: true` for runs that take >5 min. Always:
- Tee output to `acceptance/bench-*.log`
- Arm a `Monitor` filtering for step-completion markers (`\[OK  \]`, `bake-off complete`, `=== ALL DONE`, error patterns)
- Pre-warn the user of the time estimate before kicking off
- Don't poll — the monitor notifies on each event; just wait

## Things that have bitten us

See `lessons.md` for the canonical list. Highlights:

- **Scope mistakes**: don't extrapolate "X needs doing" without confirming with the user. The 4-models-overnight re-run from 2026-05-11 was wasted because round-2 framing wasn't captured.
- **Stale data**: the on-disk reports were stale (10/cat) when the new data was already 30/cat. Always check report file mtime vs raw data mtime.
- **summary.json is overwritten**: don't rely on it for aggregate wall/token totals across multi-step runs. Walk per-problem files.
- **2000-char raw_text truncation**: HumanEval `raw_text` was truncated to 2000 chars and not persisted. Both fixed 2026-05-13.
- **System sleep distorts wall**: described above. Document the discrepancy in any report that uses wall numbers.

## Memory

User-level memory lives in `~/.claude/projects/-Users-mtimpe-Documents-llama-bench/memory/`. Project context, round-2 framing, feedback patterns are all there. Read those at session start.
