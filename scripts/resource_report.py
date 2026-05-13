"""Summarize bake-off resource samples per (model, bench) step.

Reads the CSV written by ``scripts/sample_resources.sh`` and the
``<csv>.steps.jsonl`` companion produced by ``scripts/run_bakeoff.py``,
joins them on timestamp, and prints a per-step breakdown:

- llama-server RSS (avg / peak, MB)
- bench runner RSS (avg / peak, MB)
- swap used at step start vs end (MB, delta MB)
- swapouts delta during step (any > 0 → we hit swap)
- compressor pages avg / peak (Apple silicon compressed-memory pressure)

Usage:
    uv run python scripts/resource_report.py acceptance/resources-<TS>.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import fmean


def _load_samples(csv_path: Path) -> list[dict]:
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        rows: list[dict] = []
        for r in reader:
            try:
                r["ts"] = float(r["ts"])
                for k in ("swap_used_mb", "swap_free_mb", "swap_total_mb",
                         "llama_rss_mb", "bench_rss_mb"):
                    r[k] = float(r[k]) if r[k] else 0.0
                for k in ("page_free", "page_active", "page_inactive",
                          "page_wired", "page_compress", "swapins", "swapouts"):
                    r[k] = int(r[k]) if r[k] else 0
            except (ValueError, KeyError):
                continue
            rows.append(r)
    rows.sort(key=lambda r: r["ts"])
    return rows


def _load_steps(jsonl_path: Path) -> list[dict]:
    out: list[dict] = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _samples_in_range(samples: list[dict], t_start: float, t_end: float) -> list[dict]:
    return [s for s in samples if t_start <= s["ts"] <= t_end]


def _summarize_step(window: list[dict]) -> dict:
    if not window:
        return {"n_samples": 0}
    llama = [s["llama_rss_mb"] for s in window]
    bench = [s["bench_rss_mb"] for s in window]
    compr_pages = [s["page_compress"] for s in window]
    page_size = int(window[0].get("page_size_bytes") or 16384)
    swap_used = [s["swap_used_mb"] for s in window]
    return {
        "n_samples": len(window),
        "llama_rss_avg_mb": round(fmean(llama), 1),
        "llama_rss_peak_mb": round(max(llama), 1),
        "bench_rss_avg_mb": round(fmean(bench), 1),
        "bench_rss_peak_mb": round(max(bench), 1),
        "compressor_avg_mb": round(fmean(compr_pages) * page_size / (1024 * 1024), 1),
        "compressor_peak_mb": round(max(compr_pages) * page_size / (1024 * 1024), 1),
        "swap_used_start_mb": round(swap_used[0], 1),
        "swap_used_end_mb": round(swap_used[-1], 1),
        "swap_used_peak_mb": round(max(swap_used), 1),
        "swap_used_delta_mb": round(swap_used[-1] - swap_used[0], 1),
        "swapouts_delta": int(window[-1]["swapouts"] - window[0]["swapouts"]),
        "swapins_delta": int(window[-1]["swapins"] - window[0]["swapins"]),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path)
    p.add_argument("--steps", type=Path, default=None,
                   help="Step-event JSONL (default: <csv>.steps.jsonl).")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    csv_path = args.csv
    steps_path = args.steps or csv_path.with_suffix(csv_path.suffix + ".steps.jsonl")
    if not csv_path.is_file():
        print(f"missing CSV: {csv_path}", file=sys.stderr); return 2
    if not steps_path.is_file():
        print(f"missing steps JSONL: {steps_path}", file=sys.stderr); return 2

    samples = _load_samples(csv_path)
    steps = _load_steps(steps_path)

    # Pair start+end events by (step_idx).
    pairs: dict[int, dict] = {}
    for ev in steps:
        slot = pairs.setdefault(ev["step_idx"], {
            "step_idx": ev["step_idx"],
            "model_id": ev.get("model_id"),
            "bench": ev.get("bench"),
        })
        slot[f"{ev['phase']}_ts"] = ev["ts"]
        if ev["phase"] == "end":
            slot["wall_s"] = ev.get("wall_s", 0.0)

    rows: list[dict] = []
    for idx in sorted(pairs):
        slot = pairs[idx]
        if "start_ts" not in slot or "end_ts" not in slot:
            continue
        win = _samples_in_range(samples, slot["start_ts"], slot["end_ts"])
        slot.update(_summarize_step(win))
        rows.append(slot)

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    print(f"# Resource report — {csv_path.name}")
    print()
    if rows:
        wall_total = sum(r.get("wall_s", 0) for r in rows)
        peak_llama = max((r.get("llama_rss_peak_mb", 0) for r in rows), default=0)
        peak_compr = max((r.get("compressor_peak_mb", 0) for r in rows), default=0)
        peak_swap = max((r.get("swap_used_peak_mb", 0) for r in rows), default=0)
        any_swapouts = any((r.get("swapouts_delta", 0) > 0) for r in rows)
        print(f"- {len(rows)} steps tracked, total wall={wall_total/60:.1f}m")
        print(f"- peak llama-server RSS across all steps: {peak_llama:.0f} MB")
        print(f"- peak compressor (compressed-memory) usage: {peak_compr:.0f} MB")
        print(f"- peak swap used: {peak_swap:.1f} MB")
        print(f"- any swapouts during a step? {'YES' if any_swapouts else 'no'}")
        print()

    header = ("| step | model | bench | wall | llama avg/peak (MB) | "
              "compressor avg/peak (MB) | swap start→end (Δ MB) | "
              "swapouts Δ |")
    sep = "|---" * 8 + "|"
    print(header)
    print(sep)
    for r in rows:
        wall = r.get("wall_s", 0)
        wall_str = f"{int(wall // 60):d}:{int(wall % 60):02d}"
        print(
            f"| {r['step_idx']} | {r['model_id']} | {r['bench']} | {wall_str} | "
            f"{r.get('llama_rss_avg_mb',0)}/{r.get('llama_rss_peak_mb',0)} | "
            f"{r.get('compressor_avg_mb',0)}/{r.get('compressor_peak_mb',0)} | "
            f"{r.get('swap_used_start_mb',0)}→{r.get('swap_used_end_mb',0)} "
            f"({r.get('swap_used_delta_mb',0):+}) | "
            f"{r.get('swapouts_delta',0)} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
