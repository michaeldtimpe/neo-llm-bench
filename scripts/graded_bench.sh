#!/usr/bin/env bash
# Graded bake-off: run BFCL + HumanEval (resumable), then print a graded
# leaderboard with full BFCL grading (function-name + arg-value match).
#
# Forwards all args to scripts/bench.sh (presets, --models, --bfcl-limit,
# --rep, --inject, etc.). After the bench finishes, runs the post-hoc
# grader and writes both a markdown leaderboard and a JSON dump.
#
# Output: ./graded_report.md  (markdown table, suitable for paste / PR)
#         ./graded_report.json (per-model per-category counts)
#
# Usage:
#   scripts/graded_bench.sh                       # default preliminary scope
#   scripts/graded_bench.sh --standard            # 30/cat BFCL + 164 HE
#   scripts/graded_bench.sh --models qwen25-coder-1.5b-instruct
#   scripts/graded_bench.sh --rep 1               # variance rep

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Detect --rep so the grader uses the same rep as the run.
REP=0
for ((i=1; i<=$#; i++)); do
    if [[ "${!i}" == "--rep" ]]; then
        next=$((i+1))
        REP="${!next}"
    fi
done

echo "[graded_bench] running underlying bench.sh ..."
bash scripts/bench.sh "$@"

echo ""
echo "[graded_bench] grading BFCL outputs (rep $REP) ..."
uv run python scripts/grade_bakeoff.py --rep "$REP" --write-back \
    | tee graded_report.md
uv run python scripts/grade_bakeoff.py --rep "$REP" --json > graded_report.json

echo ""
echo "[graded_bench] wrote: graded_report.md + graded_report.json"
