"""Drive the BFCL + HumanEval bake-off across the model roster.

Usage:
    uv run python scripts/run_bakeoff.py \
        --models all \
        --benchmarks bfcl humaneval \
        --bfcl-limit 30 --humaneval-limit 164 \
        --output acceptance/

For each (model, bench), if ``acceptance/<bench>/<model>/rep_<n>/summary.json``
already exists, the bench is skipped (resumable). Ctrl+C is safe: the current
step runs to completion (or the next iteration), then exits cleanly. Re-run the
same command to resume — already-done steps are skipped.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from llamabench.config import load_model_config, load_profile  # noqa: E402
from llamabench.runner import RunRequest, run  # noqa: E402


def _all_model_configs() -> list[Path]:
    return sorted((ROOT / "configs" / "models").glob("*.yaml"))


def _bench_already_done(out: Path, bench: str, model_id: str, rep: int) -> bool:
    return (out / bench / model_id / f"rep_{rep}" / "summary.json").is_file()


def _summary_completion_tokens(out: Path, bench: str, model_id: str, rep: int) -> int:
    """Read completion_tokens from a previously-written summary.json (for resume mode)."""
    sf = out / bench / model_id / f"rep_{rep}" / "summary.json"
    if not sf.is_file():
        return 0
    try:
        s = json.loads(sf.read_text())
    except Exception:
        return 0
    if "completion_tokens" in s:
        return int(s["completion_tokens"])
    return sum(int(c.get("completion_tokens", 0)) for c in s.get("categories", {}).values())


def _hms(secs: float) -> str:
    secs = max(0, int(secs))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def _ts() -> str:
    return time.strftime("%H:%M:%S")


_INTERRUPTED = False


def _on_sigint(signum, frame):  # noqa: ARG001
    global _INTERRUPTED
    if _INTERRUPTED:
        # Second Ctrl+C — bail out hard, will likely interrupt the running step.
        print("\n[!] Second Ctrl+C — exiting immediately. Re-run the same command to resume.",
              file=sys.stderr)
        sys.exit(130)
    _INTERRUPTED = True
    print("\n[*] Pause requested — finishing the current step, then exiting cleanly. "
          "Press Ctrl+C again to abort the current step (will lose its results).",
          file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["all"],
                   help="Model ids to run (or 'all'). Match basename of configs/models/*.yaml.")
    p.add_argument("--benchmarks", nargs="+", default=["bfcl", "humaneval"])
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--bfcl-limit", type=int, default=None,
                   help="Per-category problem cap for BFCL.")
    p.add_argument("--humaneval-limit", type=int, default=None,
                   help="Total problem cap for HumanEval (max 164).")
    p.add_argument("--bfcl-categories", nargs="+", default=None)
    p.add_argument("--bfcl-mode", choices=("auto", "structured", "inject"), default="auto",
                   help="auto: structured-tools first, prompt-inject on 500 (default). "
                        "structured: only structured. inject: always prompt-inject.")
    p.add_argument("--temperature", type=float, default=None,
                   help="Override the per-model sampling.temperature for this run. "
                        "Useful for the multi-temp HumanEval sweep without touching "
                        "the model YAMLs. Default: use the model config's value.")
    p.add_argument("--profile", type=Path,
                   default=ROOT / "configs" / "profile_8gb.yaml")
    p.add_argument("--output", type=Path, default=ROOT / "acceptance")
    p.add_argument("--force", action="store_true",
                   help="Re-run benches even if summary.json exists.")
    p.add_argument("--resource-log", type=Path, default=None,
                   help="If set, spawn scripts/sample_resources.sh writing to "
                        "this CSV every --resource-interval seconds, and also "
                        "write step-boundary records to "
                        "<resource-log>.steps.jsonl for post-hoc joining.")
    p.add_argument("--resource-interval", type=int, default=5,
                   help="Sampler interval in seconds (default 5).")
    args = p.parse_args()

    level_name = os.environ.get("LLAMABENCH_LOG_LEVEL", "WARNING").upper()
    level = getattr(logging, level_name, logging.WARNING)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    cfgs = _all_model_configs()
    if args.models != ["all"]:
        wanted = set(args.models)
        cfgs = [c for c in cfgs if c.stem in wanted or c.stem.replace(".", "_") in wanted]
        unknown = wanted - {c.stem for c in cfgs}
        if unknown:
            print(f"unknown model ids: {sorted(unknown)}", file=sys.stderr)
            return 2

    profile = load_profile(args.profile)

    # Plan: build the full list of (model, bench) steps, mark which are
    # already done, and use that to compute progress + ETA.
    signal.signal(signal.SIGINT, _on_sigint)
    steps: list[tuple] = []  # (model_config, bench, already_done)
    for cfg_path in cfgs:
        mc = load_model_config(cfg_path)
        for bench in args.benchmarks:
            already = (not args.force) and _bench_already_done(args.output, bench, mc.id, args.rep)
            steps.append((mc, bench, already))

    n_total = len(steps)
    n_done_pre = sum(1 for _, _, d in steps if d)
    n_to_run = n_total - n_done_pre

    print(f"bake-off plan: {n_total} steps total ({n_to_run} to run, {n_done_pre} already done)",
          file=sys.stderr)
    print(f"  models   : {[c.stem for c in cfgs]}", file=sys.stderr)
    print(f"  benches  : {args.benchmarks}", file=sys.stderr)
    print(f"  bfcl mode: {args.bfcl_mode}  rep: {args.rep}", file=sys.stderr)
    print(f"  output   : {args.output}", file=sys.stderr)
    print("", file=sys.stderr)

    overall_t0 = time.monotonic()
    step_walls: list[float] = []  # wall seconds for each step we actually ran
    cumulative_tokens = 0
    n_run = 0

    # --- spawn resource sampler if requested ---
    sampler_proc: subprocess.Popen | None = None
    steps_jsonl: Path | None = None
    if args.resource_log:
        args.resource_log.parent.mkdir(parents=True, exist_ok=True)
        sampler_script = ROOT / "scripts" / "sample_resources.sh"
        try:
            sampler_proc = subprocess.Popen(
                [str(sampler_script), str(args.resource_log), str(args.resource_interval)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            steps_jsonl = args.resource_log.with_suffix(args.resource_log.suffix + ".steps.jsonl")
            steps_jsonl.write_text("")
            print(f"  resource sampler started (pid={sampler_proc.pid}) -> "
                  f"{args.resource_log} (interval={args.resource_interval}s)",
                  file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            print(f"  WARN: failed to start resource sampler: {e}", file=sys.stderr)
            sampler_proc = None

    def _record_step_event(idx: int, mc_id: str, bench: str, phase: str,
                           extra: dict | None = None) -> None:
        if not steps_jsonl:
            return
        rec = {
            "ts": time.time(),
            "step_idx": idx,
            "step_total": n_total,
            "model_id": mc_id,
            "bench": bench,
            "phase": phase,
        }
        if extra:
            rec.update(extra)
        with steps_jsonl.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    def _stop_sampler() -> None:
        if sampler_proc is not None and sampler_proc.poll() is None:
            sampler_proc.terminate()
            try:
                sampler_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                sampler_proc.kill()

    for idx, (mc, bench, already) in enumerate(steps, start=1):
        if _INTERRUPTED:
            print(f"[*] Pause: stopped before step {idx}/{n_total}. Re-run to resume.",
                  file=sys.stderr)
            _stop_sampler()
            return 130
        prefix = f"[{idx:>2}/{n_total}]"

        if already:
            tok = _summary_completion_tokens(args.output, bench, mc.id, args.rep)
            cumulative_tokens += tok
            print(f"{prefix} [{_ts()}] [skip] {mc.id}/{bench}  (cached, {tok} comp-tok)",
                  file=sys.stderr)
            continue

        limit = args.bfcl_limit if bench == "bfcl" else args.humaneval_limit
        req = RunRequest(
            model=mc, benchmarks=[bench], output_dir=args.output,
            rep=args.rep, limit=limit,
            bfcl_categories=tuple(args.bfcl_categories) if args.bfcl_categories else None,
            bfcl_mode=args.bfcl_mode,
            temperature_override=args.temperature,
        )

        # ETA based on average wall of steps we've completed THIS invocation.
        if step_walls:
            avg = sum(step_walls) / len(step_walls)
            remaining = (n_to_run - n_run) * avg
            eta_str = f"  est-remaining ~{_hms(remaining)}"
        else:
            eta_str = ""

        print(f"{prefix} [{_ts()}] [run ] {mc.id}/{bench}  "
              f"(limit={limit}, mode={args.bfcl_mode if bench=='bfcl' else 'n/a'}){eta_str}",
              file=sys.stderr)

        _record_step_event(idx, mc.id, bench, "start")
        t0 = time.monotonic()
        r = run(req, profile)
        wall = time.monotonic() - t0
        step_walls.append(wall)
        n_run += 1
        _record_step_event(idx, mc.id, bench, "end", extra={"wall_s": wall})

        # Pull completion-tokens from the bench's own summary.
        step_tokens = 0
        bench_summary = r.ran.get(bench, {})
        if "completion_tokens" in bench_summary:
            step_tokens = int(bench_summary["completion_tokens"])
        else:
            step_tokens = sum(int(c.get("completion_tokens", 0))
                              for c in bench_summary.get("categories", {}).values())
        cumulative_tokens += step_tokens

        tag = "ERR " if r.error or "error" in bench_summary else "OK  "
        rate = step_tokens / max(1.0, wall)
        elapsed_total = time.monotonic() - overall_t0
        print(f"{prefix} [{_ts()}] [{tag}] {mc.id}/{bench}  "
              f"wall={_hms(wall)}  tok={step_tokens} ({rate:.0f} tok/s)  "
              f"cum={cumulative_tokens}  elapsed={_hms(elapsed_total)}  "
              f"err={r.error or bench_summary.get('error', '-')[:60]}",
              file=sys.stderr)

    total_wall = time.monotonic() - overall_t0
    print(f"\nbake-off complete: ran {n_run} steps in {_hms(total_wall)}, "
          f"{cumulative_tokens} cumulative completion tokens", file=sys.stderr)

    _stop_sampler()
    if sampler_proc is not None:
        print(f"  resource sampler stopped (wrote {args.resource_log})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
