#!/usr/bin/env bash
# sample_resources.sh — sample macOS memory + swap to a CSV during a bake-off.
#
# Usage:
#   scripts/sample_resources.sh OUT_CSV [INTERVAL_SECONDS]
#
# Columns:
#   ts             unix epoch seconds
#   swap_used_mb   from sysctl vm.swapusage
#   swap_free_mb   from sysctl vm.swapusage
#   swap_total_mb  from sysctl vm.swapusage
#   page_free      from vm_stat
#   page_active    from vm_stat
#   page_inactive  from vm_stat
#   page_wired     from vm_stat
#   page_compress  from vm_stat ("Pages occupied by compressor")
#   swapins        from vm_stat (cumulative)
#   swapouts       from vm_stat (cumulative)
#   llama_rss_mb   summed RSS of all llama-server processes
#   bench_rss_mb   summed RSS of run_bakeoff.py / benchmarks.bfcl.run / benchmarks.humaneval workers
#
# Stops cleanly on SIGTERM/SIGINT. Page size on Apple silicon is 16 KiB.

set -u

OUT="${1:-resources.csv}"
INTERVAL="${2:-5}"

PAGE_SIZE_BYTES=$(vm_stat | head -1 | grep -oE 'page size of [0-9]+' | grep -oE '[0-9]+')
if [[ -z "${PAGE_SIZE_BYTES:-}" ]]; then PAGE_SIZE_BYTES=16384; fi

echo "ts,swap_used_mb,swap_free_mb,swap_total_mb,page_free,page_active,page_inactive,page_wired,page_compress,swapins,swapouts,llama_rss_mb,bench_rss_mb,page_size_bytes" > "$OUT"

cleanup() { exit 0; }
trap cleanup TERM INT

while true; do
    TS=$(date +%s)

    SWAP_LINE=$(sysctl -n vm.swapusage 2>/dev/null || echo "")
    SWAP_USED=$(  echo "$SWAP_LINE" | sed -nE 's/.*used = ([0-9.]+)M.*/\1/p')
    SWAP_FREE=$(  echo "$SWAP_LINE" | sed -nE 's/.*free = ([0-9.]+)M.*/\1/p')
    SWAP_TOTAL=$( echo "$SWAP_LINE" | sed -nE 's/.*total = ([0-9.]+)M.*/\1/p')

    # Parse vm_stat: each "Pages X: N." line. Use awk to grab the four we want.
    VM=$(vm_stat 2>/dev/null | awk '
        /Pages free:/        {gsub(/\./,"",$3); free=$3}
        /Pages active:/      {gsub(/\./,"",$3); active=$3}
        /Pages inactive:/    {gsub(/\./,"",$3); inactive=$3}
        /Pages wired down:/  {gsub(/\./,"",$4); wired=$4}
        /Pages occupied by compressor:/ {gsub(/\./,"",$5); compressor=$5}
        /Swapins:/           {gsub(/\./,"",$2); swapins=$2}
        /Swapouts:/          {gsub(/\./,"",$2); swapouts=$2}
        END {printf "%s,%s,%s,%s,%s,%s,%s", free,active,inactive,wired,compressor,swapins,swapouts}')

    # ps RSS is in KB. Sum across matching processes. `comm` is exec basename.
    LLAMA_RSS_KB=$(ps -axo rss=,comm= 2>/dev/null | awk '$2 ~ /llama-server$/ {sum+=$1} END {print sum+0}')
    BENCH_RSS_KB=$(ps -axo rss=,command= 2>/dev/null | awk '/run_bakeoff\.py|benchmarks\.bfcl|benchmarks\.humaneval/ {sum+=$1} END {print sum+0}')
    LLAMA_RSS_MB=$(awk -v kb="$LLAMA_RSS_KB" 'BEGIN{printf "%.1f", kb/1024}')
    BENCH_RSS_MB=$(awk -v kb="$BENCH_RSS_KB" 'BEGIN{printf "%.1f", kb/1024}')

    echo "$TS,${SWAP_USED:-},${SWAP_FREE:-},${SWAP_TOTAL:-},${VM},${LLAMA_RSS_MB},${BENCH_RSS_MB},${PAGE_SIZE_BYTES}" >> "$OUT"

    sleep "$INTERVAL"
done
