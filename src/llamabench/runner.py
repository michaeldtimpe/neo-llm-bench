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
from llamabench.metadata import build_run_metadata, write_run_metadata
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
    # Orthogonal to bfcl_mode: "raw" = single-pass capability measurement
    # (the prior default); "agent" = closed-loop run_agent dispatch with
    # stub-tool feedback. Different benchmarks; see graded_report.md
    # framing note.
    bfcl_run_mode: str = "raw"
    # Override the model's configured sampling temperature for this run.
    # Used by the multi-temperature HumanEval sweep — leaves the YAML alone.
    temperature_override: float | None = None
    # Path of the loaded BenchProfile yaml — surfaced in metadata.json so
    # cross-run comparisons can pin which profile was used. None if not set.
    profile_path: Path | None = None


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
        n_parallel=profile.max_parallel_requests,
    )
    t0 = time.monotonic()
    ran: dict[str, dict] = {}
    err = ""
    try:
        server.start()
        backend = Backend(base_url=server.base_url, model=spec.alias or req.model.id)
        for bench in req.benchmarks:
            spec_entry = _BENCH_RUNNERS.get(bench)
            if spec_entry is None:
                ran[bench] = {"error": f"unknown benchmark: {bench}"}
                continue
            # Persist provenance before the bench writes its own outputs.
            # Best-effort; metadata failures must not block the run.
            try:
                step_dir = req.output_dir / bench / req.model.id / f"rep_{req.rep}"
                meta = build_run_metadata(
                    model=req.model, benchmark=bench, rep=req.rep,
                    profile=profile, server_bin=bin_path,
                    profile_path=req.profile_path,
                    mode=({"bfcl_mode": req.bfcl_mode,
                           "bfcl_run_mode": req.bfcl_run_mode}
                          if bench == "bfcl" else None),
                    temperature_override=req.temperature_override,
                )
                write_run_metadata(step_dir, meta)
            except Exception:  # noqa: BLE001
                logger.exception("metadata write failed for %s/%s", req.model.id, bench)
            try:
                ran[bench] = spec_entry.runner(backend, req)
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
    """Run BFCL categories against `backend`. Stores per-problem JSON
    under <output>/bfcl/<model>/rep_<n>/<category>/<id>.json + summary.json.

    Two orthogonal axes:
    - `bfcl_mode` ("auto"/"structured"/"inject"): how raw mode delivers
      tool specs to the model. Per-model `model.bfcl_mode` overrides.
    - `bfcl_run_mode` ("raw"/"agent"): single-pass vs closed-loop. Agent
      mode uses the existing run_agent loop with stub executors.
    """
    from benchmarks.bfcl.adapter import (
        SUPPORTED_CATEGORIES, load_problems, run_problem_agent, run_problem_raw,
    )
    from benchmarks.bfcl.multi_turn import is_multi_turn, run_problem_multi_turn
    from dataclasses import asdict

    out_dir = req.output_dir / "bfcl" / req.model.id / f"rep_{req.rep}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cats = req.bfcl_categories or SUPPORTED_CATEGORIES

    effective_mode = req.model.bfcl_mode or req.bfcl_mode
    run_mode = req.bfcl_run_mode or "raw"

    # Agent mode builds one RoleConfig from the per-model sampling/server
    # so temperature + n_ctx match raw mode. max_steps hardcoded so the
    # bake-off comparison isn't confounded by per-model agent tuning.
    role_cfg = None
    if run_mode == "agent":
        from llamabench.config import RoleConfig
        temp_for_role = (req.temperature_override
                         if req.temperature_override is not None
                         else req.model.sampling.temperature)
        role_cfg = RoleConfig(
            model_key=req.model.id,
            num_ctx=req.model.server.n_ctx,
            max_steps=12,
            max_tokens_per_turn=req.model.sampling.max_tokens,
            temperature=temp_for_role,
        )

    summary: dict[str, Any] = {
        "model": req.model.id, "rep": req.rep,
        "mode": effective_mode, "run_mode": run_mode,
        "categories": {},
    }
    for cat in cats:
        cat_dir = out_dir / cat
        cat_dir.mkdir(exist_ok=True)
        problems = load_problems(cat, limit=req.limit)
        cat_summary: dict[str, Any] = {
            "n_problems": len(problems), "n_with_calls": 0,
            "n_errors": 0, "wall_s": 0.0, "completion_tokens": 0,
        }
        if run_mode == "agent":
            cat_summary["n_turns_total"] = 0
            cat_summary["n_tool_calls_total"] = 0
            cat_summary["n_schema_rejects_total"] = 0
        temp = (req.temperature_override
                if req.temperature_override is not None
                else req.model.sampling.temperature)
        for p in problems:
            if is_multi_turn(cat):
                # Multi-turn problems use their own driver + grader path.
                # The runner here just persists the per-turn-per-step call
                # strings; grade_bakeoff.py loads them later and dispatches
                # to grade_multi_turn against the bfcl_eval state checker.
                mt = run_problem_multi_turn(
                    backend, p,
                    max_tokens=req.model.sampling.max_tokens,
                    temperature=temp,
                    category=cat,
                )
                row = {
                    "id": mt.problem_id,
                    "per_turn_steps": mt.per_turn_steps,
                    "wall_s": mt.wall_s,
                    "prompt_tokens": mt.prompt_tokens,
                    "completion_tokens": mt.completion_tokens,
                    "error": mt.error,
                    "n_turns": len(mt.per_turn_steps),
                    "trace": [asdict(t) for t in mt.per_turn_trace],
                }
                (cat_dir / f"{mt.problem_id}.json").write_text(json.dumps(row))
                cat_summary["wall_s"] += mt.wall_s
                cat_summary["completion_tokens"] += mt.completion_tokens
                if mt.error:
                    cat_summary["n_errors"] += 1
                elif any(mt.per_turn_steps):
                    cat_summary["n_with_calls"] += 1
                continue

            if run_mode == "agent":
                r = run_problem_agent(backend, role_cfg, p)
            else:
                r = run_problem_raw(
                    backend, p,
                    max_tokens=req.model.sampling.max_tokens,
                    temperature=temp,
                    mode=effective_mode,
                )
            row: dict[str, Any] = {
                "id": r.problem_id, "actual_calls": r.actual_calls,
                "wall_s": r.wall_s, "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens, "error": r.error,
            }
            if run_mode == "agent":
                row["n_turns"] = r.n_turns
                row["n_tool_calls_total"] = r.n_tool_calls_total
                row["n_schema_rejects"] = r.n_schema_rejects
            (cat_dir / f"{r.problem_id}.json").write_text(json.dumps(row))
            cat_summary["wall_s"] += r.wall_s
            cat_summary["completion_tokens"] += r.completion_tokens
            if r.error:
                cat_summary["n_errors"] += 1
            elif r.actual_calls:
                cat_summary["n_with_calls"] += 1
            if run_mode == "agent":
                cat_summary["n_turns_total"] += r.n_turns
                cat_summary["n_tool_calls_total"] += r.n_tool_calls_total
                cat_summary["n_schema_rejects_total"] += r.n_schema_rejects
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


def _run_mbpp(backend: Backend, req: RunRequest) -> dict:
    from benchmarks.mbpp.adapter import load_problems, run_problem

    out_dir = req.output_dir / "mbpp" / req.model.id / f"rep_{req.rep}"
    out_dir.mkdir(parents=True, exist_ok=True)
    problems = load_problems(limit=req.limit)
    results_path = out_dir / "results.jsonl"

    # Per-problem resume — same shape as HumanEval; task_ids are
    # `Mbpp/<int>` strings (the adapter prefixes them).
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
    wanted = {f"Mbpp/{p['task_id']}" for p in problems}
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
            tid = f"Mbpp/{p['task_id']}"
            if tid in done:
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
                "completion": r.completion,  # post-normalization, for audit
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


@dataclass(frozen=True)
class BenchmarkSpec:
    """Static facts about one registered benchmark.

    `force_clean_filenames` enumerates the artifacts that `--force` must
    delete before the step re-runs (so per-problem resume paths can't
    silently reuse stale data). New benchmarks extend the registry; the
    cleanup helper in `run_bakeoff.py` consults this set, not a global
    constant — which means MBPP / multi-turn / agent-mode can each
    declare their own cleanup surface without cross-bench coupling.
    """

    name: str
    runner: Callable[[Backend, RunRequest], dict]
    supports_per_problem_resume: bool = False
    force_clean_filenames: frozenset[str] = frozenset({"summary.json"})


_BENCH_RUNNERS: dict[str, BenchmarkSpec] = {
    "bfcl": BenchmarkSpec(
        name="bfcl",
        runner=_run_bfcl,
        supports_per_problem_resume=False,
        force_clean_filenames=frozenset({"summary.json"}),
    ),
    "humaneval": BenchmarkSpec(
        name="humaneval",
        runner=_run_humaneval,
        supports_per_problem_resume=True,
        force_clean_filenames=frozenset({"summary.json", "results.jsonl"}),
    ),
    "mbpp": BenchmarkSpec(
        name="mbpp",
        runner=_run_mbpp,
        supports_per_problem_resume=True,
        force_clean_filenames=frozenset({"summary.json", "results.jsonl"}),
    ),
}
