#!/usr/bin/env bash
# Round 1 BFCL overnight batch — runs all 5 categories against the 4
# Round 0 survivors, in order, with explicit oMLX unload between
# candidates (so the pin cascade documented in lessons.md §11 doesn't
# strand the second model on a 507).
#
# Resumable: the BFCL runner skips per-problem JSON that already
# exists, so killing this and re-running picks up where it left off.
#
# Usage:
#   OMLX_API_KEY=... ./scripts/round1_overnight.sh
#   OMLX_API_KEY=... ./scripts/round1_overnight.sh --limit 5   # smoke
#
# Output:
#   acceptance/bfcl/<model>/rep_1/<category>/<problem_id>.json
#   acceptance/bfcl/<model>/rep_1/summary.json
#   /tmp/round1_<model>.log              (per-candidate stdout)
#   /tmp/round1_overnight_summary.txt    (final tally)
#
# Wall estimate: ~3-3.75h per candidate × 4 = 12-15h.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${OMLX_API_KEY:-}" ]]; then
    echo "ERROR: OMLX_API_KEY not set. Export it before running." >&2
    echo "  e.g. export OMLX_API_KEY=omlx-..." >&2
    exit 2
fi

PYTHON="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: $PYTHON not found or not executable." >&2
    echo "  Create the venv with: python3.11 -m venv .venv && .venv/bin/pip install -e ." >&2
    exit 2
fi

# The `llamabench` CLI is installed as a console_script (see pyproject.toml).
# `python -m llamabench` won't work because the package has no __main__.
LLAMABENCH_CLI="${LLAMABENCH_CLI:-.venv/bin/llamabench}"
if [[ ! -x "$LLAMABENCH_CLI" ]]; then
    echo "ERROR: $LLAMABENCH_CLI not found. Did you run pip install -e .?" >&2
    exit 2
fi

OMLX_BASE_URL="${OMLX_BASE_URL:-http://127.0.0.1:8000}"

# Sanity-check oMLX before burning 15h on a dead endpoint.
if ! curl -fsS -o /dev/null --max-time 5 "$OMLX_BASE_URL/v1/models" \
        -H "Authorization: Bearer $OMLX_API_KEY"; then
    echo "ERROR: oMLX unreachable at $OMLX_BASE_URL." >&2
    echo "  Start it (brew services start omlx) and retry." >&2
    exit 2
fi

# Round 0 survivors, in priority order. Smaller / faster first so a
# failure of the runner doesn't strand the cheap candidates.
SURVIVORS=(
    Qwen2.5-Coder-32B-Instruct-4bit
    Qwen2.5-32B-Instruct-4bit
    Qwen3-32B-4bit
    Llama-3.3-70B-Instruct-3bit
)

CATEGORIES=(simple_python multiple parallel parallel_multiple irrelevance)

EXTRA_ARGS=()  # forwarded to the BFCL runner (e.g. --limit 5)
# Drop blank/whitespace-only positional args. A stray `\ ` in a launch
# command line gets passed through as a literal space and argparse
# rejects it with `unrecognized arguments:` before any work runs.
for arg in "$@"; do
    [[ -n "${arg// /}" ]] && EXTRA_ARGS+=("$arg")
done

SUMMARY_FILE="/tmp/round1_overnight_summary.txt"
: > "$SUMMARY_FILE"

started_iso="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
# Per-candidate wall estimate (seconds): midpoint of 3-3.75h.
EST_PER_CAND=$((3 * 3600 + 1350))   # 12150s = ~3h22m
EST_TOTAL=$((EST_PER_CAND * ${#SURVIVORS[@]}))
eta_iso_initial="$(date -u -v +"$((EST_TOTAL / 60))M" +'%Y-%m-%dT%H:%M:%SZ' 2>/dev/null \
    || date -u -d "+$((EST_TOTAL / 60)) minutes" +'%Y-%m-%dT%H:%M:%SZ')"
echo "[round1] start: $started_iso"
echo "[round1] survivors: ${SURVIVORS[*]}"
echo "[round1] categories: ${CATEGORIES[*]}"
echo "[round1] extra args: ${EXTRA_ARGS[*]:-(none)}"
echo "[round1] est per-candidate: $((EST_PER_CAND / 60))m  est total: $((EST_TOTAL / 3600))h$(( (EST_TOTAL % 3600) / 60))m"
echo "[round1] initial ETA (all 4 done): $eta_iso_initial"
echo

t_suite_start=$(date +%s)

for i in "${!SURVIVORS[@]}"; do
    model="${SURVIVORS[$i]}"
    out_dir="acceptance/bfcl/${model}/rep_1"
    log_file="/tmp/round1_${model}.log"

    banner="[$((i + 1))/${#SURVIVORS[@]}] $model"
    echo "================================================================"
    echo "$banner"
    echo "  output: $out_dir"
    echo "  log:    $log_file"
    echo "================================================================"

    # Best-effort unload of every previously-loaded model. Skipped on
    # the first iteration (nothing loaded yet) and tolerated if it
    # fails — oMLX may have nothing to unload, which is fine.
    if [[ $i -gt 0 ]]; then
        echo "[$model] unloading previous models..."
        if ! "$LLAMABENCH_CLI" unload 2>&1 | sed 's/^/  /'; then
            echo "  (unload returned non-zero — continuing; oMLX state may already be clean)"
        fi
        # Settle: give oMLX a beat to release memory before the next load.
        sleep 5
    fi

    t_cand_start=$(date +%s)

    # Run BFCL. The runner is resumable per-problem, so a kill here
    # only loses in-flight wall, not committed verdicts.
    set +e
    "$PYTHON" -m benchmarks.bfcl.run \
        --model "$model" \
        --categories "${CATEGORIES[@]}" \
        --output "$out_dir" \
        --base-url "$OMLX_BASE_URL" \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee "$log_file"
    rc="${PIPESTATUS[0]}"
    set -e

    t_cand_end=$(date +%s)
    cand_wall=$((t_cand_end - t_cand_start))

    if [[ $rc -ne 0 ]]; then
        echo "[$model] FAILED rc=$rc after ${cand_wall}s — see $log_file" \
            | tee -a "$SUMMARY_FILE"
        # Don't abort the whole batch on one model's failure — record
        # and move on so the rest of the suite still produces data.
        continue
    fi

    # Pull the totals line out of the runner's last few lines for the
    # rollup (the runner writes summary.json too, so this is just a
    # human-friendly recap).
    totals=$(grep "BFCL .* mode — totals" "$log_file" | tail -1 || true)
    line="[$model] OK in ${cand_wall}s — ${totals:-(no totals line)}"
    echo "$line" | tee -a "$SUMMARY_FILE"

    # Live ETA: extrapolate remaining wall from this candidate's actual time.
    remaining=$(( ${#SURVIVORS[@]} - i - 1 ))
    if [[ $remaining -gt 0 ]]; then
        eta_secs=$(( cand_wall * remaining ))
        eta_iso="$(date -u -v +"$((eta_secs / 60))M" +'%Y-%m-%dT%H:%M:%SZ' 2>/dev/null \
            || date -u -d "+$((eta_secs / 60)) minutes" +'%Y-%m-%dT%H:%M:%SZ')"
        echo "[round1] $remaining candidate(s) remaining; ETA at last-cand pace: $eta_iso"
    fi
    echo
done

t_suite_end=$(date +%s)
suite_wall=$((t_suite_end - t_suite_start))
ended_iso="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"

echo "================================================================"
echo "[round1] done: $ended_iso  total=${suite_wall}s ($((suite_wall / 60))m)"
echo "[round1] per-candidate summary:"
sed 's/^/  /' "$SUMMARY_FILE"
echo "[round1] per-model JSON: acceptance/bfcl/<model>/rep_1/summary.json"
echo "[round1] gate check: irrelevance ≥80%, simple_python ≥65%"
