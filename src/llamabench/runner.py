"""Bench orchestrator — spawns llama-server, runs benchmarks, tears down.

One run = (one model) × (one or more benchmarks). Server lifecycle is
owned here: ``LlamaServer.start`` before benchmarks, ``stop`` after.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from llamabench.backend import Backend
from llamabench.config import BenchProfile, ModelConfig, ServerConfig as _ServerCfg
from llamabench.server import DEFAULT_BIN, LlamaServer, ServerSpec


logger = logging.getLogger(__name__)


def model_to_server_spec(mc: ModelConfig) -> ServerSpec:
    s = mc.server
    return ServerSpec(
        model_path=Path(mc.gguf_path).expanduser(),
        n_ctx=s.n_ctx,
        n_gpu_layers=s.n_gpu_layers,
        n_threads=s.n_threads,
        n_batch=s.n_batch,
        n_ubatch=s.n_ubatch,
        flash_attn=s.flash_attn,
        mmap=s.mmap,
        mlock=s.mlock,
        cache_type_k=s.cache_type_k,
        cache_type_v=s.cache_type_v,
        chat_template=s.chat_template,
        chat_template_file=Path(s.chat_template_file).expanduser() if s.chat_template_file else None,
        jinja=s.jinja,
        alias=mc.alias or mc.id,
    )


@dataclass
class RunRequest:
    model: ModelConfig
    benchmarks: list[str]
    output_dir: Path
    rep: int = 0
    limit: int | None = None
    bfcl_categories: tuple[str, ...] | None = None  # None = use SUPPORTED_CATEGORIES
    bfcl_mode: str = "auto"  # auto | structured | inject — see bfcl/adapter.run_problem_raw
    # Override the model's configured sampling temperature for this run.
    # Used by the multi-temperature HumanEval sweep — leaves the YAML alone.
    temperature_override: float | None = None


@dataclass
class RunResult:
    model_id: str
    rep: int
    ran: dict[str, dict]  # benchmark name -> summary dict
    server_log: Path | None
    wall_s: float
    error: str = ""


def run(req: RunRequest, profile: BenchProfile) -> RunResult:
    spec = model_to_server_spec(req.model)
    if not spec.model_path.is_file():
        return RunResult(
            model_id=req.model.id, rep=req.rep, ran={},
            server_log=None, wall_s=0.0,
            error=f"GGUF not found at {spec.model_path}",
        )

    bin_path = Path(profile.server_bin).expanduser() if profile.server_bin else DEFAULT_BIN
    server = LlamaServer(
        spec=spec, host=profile.server_host, port=profile.server_port, bin_path=bin_path,
    )
    t0 = time.monotonic()
    ran: dict[str, dict] = {}
    err = ""
    try:
        server.start()
        backend = Backend(base_url=server.base_url, model=spec.alias or req.model.id)
        for bench in req.benchmarks:
            runner_fn = _BENCH_RUNNERS.get(bench)
            if runner_fn is None:
                ran[bench] = {"error": f"unknown benchmark: {bench}"}
                continue
            try:
                ran[bench] = runner_fn(backend, req)
            except Exception as e:  # noqa: BLE001
                logger.exception("bench %s crashed for %s", bench, req.model.id)
                ran[bench] = {"error": f"{type(e).__name__}: {e}"}
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
        logger.exception("run failed for %s", req.model.id)
    finally:
        try:
            server.stop()
        except Exception:  # noqa: BLE001
            pass
    return RunResult(
        model_id=req.model.id, rep=req.rep, ran=ran,
        server_log=server.log_dir, wall_s=time.monotonic() - t0,
        error=err,
    )


# --- Per-benchmark runners --------------------------------------------------


def _run_bfcl(backend: Backend, req: RunRequest) -> dict:
    """Run BFCL categories in raw mode against `backend`. Stores per-problem
    JSON under <output>/bfcl/<model>/rep_<n>/<category>/<id>.json plus summary.json.
    """
    from benchmarks.bfcl.adapter import (
        SUPPORTED_CATEGORIES, load_problems, run_problem_raw,
    )

    out_dir = req.output_dir / "bfcl" / req.model.id / f"rep_{req.rep}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cats = req.bfcl_categories or SUPPORTED_CATEGORIES

    # Per-model override beats the run-level mode. Models whose chat
    # template lacks a tools branch (deepseek-coder-1.3b, smollm2,
    # phi-1.5) must use "inject" or tools are silently dropped.
    effective_mode = req.model.bfcl_mode or req.bfcl_mode

    summary: dict[str, dict] = {
        "model": req.model.id, "rep": req.rep, "mode": effective_mode, "categories": {},
    }
    for cat in cats:
        cat_dir = out_dir / cat
        cat_dir.mkdir(exist_ok=True)
        problems = load_problems(cat, limit=req.limit)
        cat_summary = {
            "n_problems": len(problems), "n_with_calls": 0,
            "n_errors": 0, "wall_s": 0.0, "completion_tokens": 0,
        }
        temp = (req.temperature_override
                if req.temperature_override is not None
                else req.model.sampling.temperature)
        for p in problems:
            r = run_problem_raw(
                backend, p,
                max_tokens=req.model.sampling.max_tokens,
                temperature=temp,
                mode=effective_mode,
            )
            (cat_dir / f"{r.problem_id}.json").write_text(json.dumps({
                "id": r.problem_id, "actual_calls": r.actual_calls,
                "wall_s": r.wall_s, "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens, "error": r.error,
            }))
            cat_summary["wall_s"] += r.wall_s
            cat_summary["completion_tokens"] += r.completion_tokens
            if r.error:
                cat_summary["n_errors"] += 1
            elif r.actual_calls:
                cat_summary["n_with_calls"] += 1
        summary["categories"][cat] = cat_summary

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _run_humaneval(backend: Backend, req: RunRequest) -> dict:
    from benchmarks.humaneval.adapter import load_problems, run_problem

    out_dir = req.output_dir / "humaneval" / req.model.id / f"rep_{req.rep}"
    out_dir.mkdir(parents=True, exist_ok=True)
    problems = load_problems(limit=req.limit)
    results_path = out_dir / "results.jsonl"

    # Per-problem resume: rows already written are kept; we only run task_ids
    # missing from results.jsonl. A torn last line (kill mid-write) is dropped
    # and the file is rewritten cleanly before we append.
    done: dict[str, dict] = {}
    torn = False
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                torn = True
                continue
            tid = row.get("task_id")
            if tid:
                done[tid] = row
        if torn:
            results_path.write_text("".join(json.dumps(r) + "\n" for r in done.values()))

    summary = {
        "model": req.model.id, "rep": req.rep,
        "n_problems": len(problems), "n_passed": 0, "n_extract_ok": 0,
        "wall_s": 0.0, "completion_tokens": 0,
    }
    wanted = {p["task_id"] for p in problems}
    for tid, r in done.items():
        if tid not in wanted:
            continue
        summary["wall_s"] += float(r.get("wall_s", 0.0))
        summary["completion_tokens"] += int(r.get("completion_tokens", 0))
        if r.get("passed"):
            summary["n_passed"] += 1
        if r.get("extract_ok"):
            summary["n_extract_ok"] += 1

    temp = (req.temperature_override
            if req.temperature_override is not None
            else req.model.sampling.temperature)
    with results_path.open("a") as fp:
        for p in problems:
            if p["task_id"] in done:
                continue
            r = run_problem(
                backend, p,
                max_tokens=req.model.sampling.max_tokens,
                temperature=temp,
            )
            fp.write(json.dumps({
                "task_id": r.task_id, "passed": r.passed,
                "extract_ok": r.extract_ok, "wall_s": r.wall_s,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "error": r.error[:500] if r.error else "",
                "raw_text": r.raw_text,
            }) + "\n")
            fp.flush()
            summary["wall_s"] += r.wall_s
            summary["completion_tokens"] += r.completion_tokens
            if r.passed:
                summary["n_passed"] += 1
            if r.extract_ok:
                summary["n_extract_ok"] += 1
    summary["pass_at_1"] = summary["n_passed"] / max(1, summary["n_problems"])
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


_BENCH_RUNNERS: dict[str, Callable[[Backend, RunRequest], dict]] = {
    "bfcl": _run_bfcl,
    "humaneval": _run_humaneval,
}
