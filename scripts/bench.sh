#!/usr/bin/env bash
# llama-bench: run the BFCL + HumanEval bake-off across the model roster.
#
# Resumable: re-run with the same arguments to continue. Each (model, bench)
# step writes acceptance/<bench>/<model>/rep_<n>/summary.json on success;
# subsequent invocations skip steps whose summary already exists.
#
# Pause: Ctrl+C once → finishes the current step, then exits cleanly.
#        Ctrl+C twice → aborts the running step (loses its partial result).
#
# Usage:
#   scripts/bench.sh                           # default: BFCL 10/cat + HumanEval 30 (preliminary)
#   scripts/bench.sh --standard                # BFCL 30/cat + HumanEval all 164
#   scripts/bench.sh --full                    # BFCL all + HumanEval all 164
#   scripts/bench.sh --inject                  # set BFCL mode to inject (no structured tools)
#   scripts/bench.sh --models qwen25-1.5b-instruct phi-1.5
#   scripts/bench.sh --rep 1 --inject          # variance / methodology rep
#   scripts/bench.sh --resume                  # alias for default; re-running already resumes
#
# Pass-through to the python script with --raw:
#   scripts/bench.sh --raw --benchmarks bfcl --bfcl-limit 5 --models phi-1.5

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---- defaults (preliminary scope) ----
PRESET="preliminary"
MODELS=("all")
BFCL_LIMIT=10
HUMANEVAL_LIMIT=30
BFCL_MODE="auto"
REP=0
BENCHMARKS=("bfcl" "humaneval")
RAW_ARGS=()
RAW_MODE=0

# ---- arg parse ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --preliminary)  PRESET="preliminary";  BFCL_LIMIT=10; HUMANEVAL_LIMIT=30; shift ;;
        --standard)     PRESET="standard";     BFCL_LIMIT=30; HUMANEVAL_LIMIT=164; shift ;;
        --full)         PRESET="full";         BFCL_LIMIT=10000; HUMANEVAL_LIMIT=164; shift ;;
        --inject)       BFCL_MODE="inject"; shift ;;
        --auto)         BFCL_MODE="auto"; shift ;;
        --structured)   BFCL_MODE="structured"; shift ;;
        --rep)          REP="$2"; shift 2 ;;
        --models)
            MODELS=()
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do MODELS+=("$1"); shift; done
            ;;
        --benchmarks)
            BENCHMARKS=()
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do BENCHMARKS+=("$1"); shift; done
            ;;
        --bfcl-limit)        BFCL_LIMIT="$2"; shift 2 ;;
        --humaneval-limit)   HUMANEVAL_LIMIT="$2"; shift 2 ;;
        --resume)            shift ;;  # no-op — re-running already resumes
        --raw)               RAW_MODE=1; shift; RAW_ARGS=("$@"); break ;;
        -h|--help)
            sed -n '1,30p' "$0"; exit 0 ;;
        *)
            echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# ---- show plan ----
echo "============================================================"
echo "llama-bench bake-off"
echo "  preset      : $PRESET"
if (( RAW_MODE )); then
    echo "  raw args    : ${RAW_ARGS[*]}"
else
    echo "  models      : ${MODELS[*]}"
    echo "  benchmarks  : ${BENCHMARKS[*]}"
    echo "  bfcl-limit  : $BFCL_LIMIT (per category)"
    echo "  humaneval   : $HUMANEVAL_LIMIT problems"
    echo "  bfcl-mode   : $BFCL_MODE"
    echo "  rep         : $REP"
fi
echo "  output      : $REPO_ROOT/acceptance"
echo "  llama-server: $(which llama-server 2>/dev/null || echo /Users/mtimpe/code/llama.cpp/build/bin/llama-server)"
echo "  hostname    : $(hostname -s)"
echo "  started     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""
echo "Pause: Ctrl+C once (finishes current step). Twice = abort step."
echo "Resume: re-run this script with the same arguments."
echo ""

# ---- show pre-run memory snapshot so the operator can see if isolation is in effect ----
echo "--- memory snapshot before run ---"
echo "  swap : $(sysctl -n vm.swapusage 2>/dev/null | tr -d '\n')"
echo "  free%: $(memory_pressure 2>/dev/null | grep 'free percentage' || echo 'memory_pressure unavailable')"
echo "  top RSS:"
ps -axo rss,pid,comm | sort -rn | awk 'NR<=5 {printf "    %8d KB  %5d  %s\n", $1, $2, $3}'
echo ""

# ---- exec the python orchestrator ----
if (( RAW_MODE )); then
    exec uv run python scripts/run_bakeoff.py "${RAW_ARGS[@]}"
fi

PY_ARGS=(
    --models "${MODELS[@]}"
    --benchmarks "${BENCHMARKS[@]}"
    --bfcl-limit "$BFCL_LIMIT"
    --humaneval-limit "$HUMANEVAL_LIMIT"
    --bfcl-mode "$BFCL_MODE"
    --rep "$REP"
    --output acceptance/
)

exec uv run python scripts/run_bakeoff.py "${PY_ARGS[@]}"
