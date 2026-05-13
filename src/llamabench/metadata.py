"""Run-metadata capture for each (model, benchmark, rep) step.

A sibling ``metadata.json`` lands next to each step's ``summary.json``,
recording the provenance information needed to compare results across
hardware, runtime, and time — most importantly the GGUF SHA256 and the
llama.cpp commit, which silently shift HumanEval pass rates by several
points if anything in the build moves.

Everything here is best-effort: any field that can't be resolved cheaply
is omitted (not faked). Subprocess calls are bounded with short timeouts
so a hung git or sysctl can't block a bench step.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llamabench.config import BenchProfile, ModelConfig


_SHA_CACHE_PATH = Path.home() / ".llamabench" / "gguf-sha-cache.json"
_CMD_TIMEOUT_S = 5.0
_HASH_CHUNK_BYTES = 8 * 1024 * 1024  # 8 MB — large enough to amortize syscall cost


def _gguf_sha256(path: Path) -> str | None:
    """SHA256 of the GGUF, cached by (path, mtime, size).

    A 2 GB hash takes ~5s on this hardware; cache invalidates only when
    the file is rewritten (mtime/size change), so re-runs are free.
    Returns None if the file is missing or unreadable.
    """
    if not path.is_file():
        return None
    st = path.stat()
    cache_key = f"{path}|{int(st.st_mtime)}|{st.st_size}"
    cache: dict[str, str] = {}
    if _SHA_CACHE_PATH.is_file():
        try:
            cache = json.loads(_SHA_CACHE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            cache = {}
    if cache_key in cache:
        return cache[cache_key]
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(_HASH_CHUNK_BYTES), b""):
                h.update(chunk)
    except OSError:
        return None
    digest = h.hexdigest()
    cache[cache_key] = digest
    try:
        _SHA_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SHA_CACHE_PATH.write_text(json.dumps(cache, indent=2))
    except OSError:
        pass  # cache write failure shouldn't block the run
    return digest


def _git_rev_parse(repo_dir: Path) -> str | None:
    if not (repo_dir / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=_CMD_TIMEOUT_S,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _llama_cpp_commit(server_bin: Path) -> str | None:
    """Walk up from `server_bin` looking for the llama.cpp checkout's .git."""
    try:
        p = server_bin.expanduser().resolve()
    except (OSError, RuntimeError):
        return None
    for parent in p.parents:
        if (parent / ".git").exists():
            return _git_rev_parse(parent)
    return None


def _project_commit() -> str | None:
    # src/llamabench/metadata.py -> repo root is parents[2]
    repo_root = Path(__file__).resolve().parents[2]
    return _git_rev_parse(repo_root)


def _sysctl(key: str) -> str | None:
    try:
        result = subprocess.run(
            ["sysctl", "-n", key],
            capture_output=True, text=True, timeout=_CMD_TIMEOUT_S,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _host_info(profile_path: Path | None = None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "arch": platform.machine(),
        "os": f"{platform.system()} {platform.release()}",
    }
    cpu = _sysctl("machdep.cpu.brand_string")
    if cpu:
        info["cpu"] = cpu
    mem_raw = _sysctl("hw.memsize")
    if mem_raw and mem_raw.isdigit():
        info["mem_gb"] = round(int(mem_raw) / 1024 / 1024 / 1024, 1)
    if profile_path is not None:
        info["profile"] = str(profile_path)
    return info


def _tooling() -> dict[str, Any]:
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "neo_llm_bench_commit": _project_commit(),
    }


def build_run_metadata(
    *,
    model: ModelConfig,
    benchmark: str,
    rep: int,
    profile: BenchProfile,
    server_bin: Path,
    profile_path: Path | None = None,
    mode: dict[str, Any] | None = None,
    temperature_override: float | None = None,
) -> dict[str, Any]:
    """Build the metadata payload for one (model, benchmark, rep) step.

    `mode` is bench-specific freeform (e.g. `{"bfcl_mode": "auto"}`).
    `temperature_override` is recorded under sampling.temperature_effective
    if the runner overrode the per-model YAML default.
    """
    gguf_path = Path(model.gguf_path).expanduser()
    sampling: dict[str, Any] = {
        "temperature": model.sampling.temperature,
        "top_p": model.sampling.top_p,
        "max_tokens": model.sampling.max_tokens,
        "seed": model.sampling.seed,
    }
    if temperature_override is not None and temperature_override != model.sampling.temperature:
        sampling["temperature_effective"] = temperature_override

    return {
        "ts_started": datetime.now(timezone.utc).isoformat(),
        "model_id": model.id,
        "benchmark": benchmark,
        "rep": rep,
        "mode": mode or {},
        "model_config": {
            "gguf_path": str(gguf_path),
            "gguf_sha256": _gguf_sha256(gguf_path),
            "quant": model.quant,
            "params_b": model.params_b,
            "family": model.family,
            "alias": model.alias,
        },
        "server": {
            "bin": str(server_bin),
            "llama_cpp_commit": _llama_cpp_commit(server_bin),
            "n_ctx": model.server.n_ctx,
            "cache_type_k": model.server.cache_type_k,
            "cache_type_v": model.server.cache_type_v,
            "jinja": model.server.jinja,
            "n_parallel": profile.max_parallel_requests,
        },
        "sampling": sampling,
        "host": _host_info(profile_path=profile_path),
        "tooling": _tooling(),
    }


def write_run_metadata(out_dir: Path, meta: dict[str, Any]) -> None:
    """Persist metadata.json into the step's output directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
