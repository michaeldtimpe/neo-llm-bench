"""Tests for the run-metadata builder.

Subprocess + filesystem touchpoints are mocked so the test doesn't depend
on the real environment (git, sysctl, GGUF presence).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from llamabench import metadata as md
from llamabench.config import BenchProfile, ModelConfig


@pytest.fixture
def fake_model(tmp_path: Path) -> ModelConfig:
    # Write a tiny "GGUF" with known content so SHA256 is deterministic.
    gguf = tmp_path / "fake.gguf"
    gguf.write_bytes(b"GGUF\x00FAKE-PAYLOAD-1234567890")
    return ModelConfig(
        id="fake-1.5b-instruct",
        gguf_path=str(gguf),
        family="qwen",
        params_b=1.5,
        quant="Q8_0",
        alias="fake-1.5b-instruct",
    )


@pytest.fixture
def profile() -> BenchProfile:
    return BenchProfile(
        memory_budget_gb=96,
        parallel_models=3,
        max_parallel_requests=1,
        server_bin="~/code/llama.cpp/build/bin/llama-server",
    )


@pytest.fixture
def isolated_cache(tmp_path: Path, monkeypatch):
    """Point the SHA cache at a tmp file so tests don't touch ~/.llamabench."""
    cache = tmp_path / "gguf-sha-cache.json"
    monkeypatch.setattr(md, "_SHA_CACHE_PATH", cache)
    return cache


def _stub_git_and_sysctl(monkeypatch, *,
                        llama_commit: str = "abc1234",
                        project_commit: str = "def5678",
                        cpu: str = "Apple M5 Max",
                        mem_bytes: int = 137438953472):  # 128 GB
    """Replace _git_rev_parse and _sysctl so we don't touch the real env."""

    def fake_git_rev_parse(repo_dir: Path) -> str | None:
        # Distinguish llama.cpp vs project commits by path content.
        s = str(repo_dir)
        if "llama.cpp" in s:
            return llama_commit
        return project_commit

    def fake_sysctl(key: str) -> str | None:
        if key == "machdep.cpu.brand_string":
            return cpu
        if key == "hw.memsize":
            return str(mem_bytes)
        return None

    monkeypatch.setattr(md, "_git_rev_parse", fake_git_rev_parse)
    monkeypatch.setattr(md, "_sysctl", fake_sysctl)
    # _llama_cpp_commit walks parents looking for .git — stub it directly
    # so we don't have to create a fake repo on disk.
    monkeypatch.setattr(md, "_llama_cpp_commit", lambda p: llama_commit)
    monkeypatch.setattr(md, "_project_commit", lambda: project_commit)


def test_build_metadata_top_level_shape(
    fake_model, profile, isolated_cache, monkeypatch, tmp_path
):
    _stub_git_and_sysctl(monkeypatch)
    meta = md.build_run_metadata(
        model=fake_model, benchmark="bfcl", rep=1, profile=profile,
        server_bin=Path("~/code/llama.cpp/build/bin/llama-server"),
        profile_path=Path("configs/profile_m5max.yaml"),
        mode={"bfcl_mode": "auto"},
    )
    # Top-level keys present.
    assert set(meta) >= {"ts_started", "model_id", "benchmark", "rep", "mode",
                         "model_config", "server", "sampling", "host", "tooling"}
    assert meta["model_id"] == "fake-1.5b-instruct"
    assert meta["benchmark"] == "bfcl"
    assert meta["rep"] == 1
    assert meta["mode"] == {"bfcl_mode": "auto"}
    # ts is ISO-with-tz.
    assert meta["ts_started"].endswith("+00:00") or meta["ts_started"].endswith("Z")


def test_build_metadata_gguf_sha256_deterministic(
    fake_model, profile, isolated_cache, monkeypatch
):
    _stub_git_and_sysctl(monkeypatch)
    meta = md.build_run_metadata(
        model=fake_model, benchmark="humaneval", rep=0, profile=profile,
        server_bin=Path("/bin/true"),
    )
    expected = hashlib.sha256(b"GGUF\x00FAKE-PAYLOAD-1234567890").hexdigest()
    assert meta["model_config"]["gguf_sha256"] == expected


def test_gguf_sha256_cache_is_used(
    fake_model, isolated_cache, monkeypatch
):
    """Second call doesn't re-read the file if cache hits."""
    gguf = Path(fake_model.gguf_path)
    first = md._gguf_sha256(gguf)
    assert first is not None
    # Now poison the file but keep mtime/size — cache key won't change.
    # (Re-writing identical bytes preserves size; we have to also reset mtime.)
    st = gguf.stat()
    poisoned_bytes = b"x" * st.st_size
    gguf.write_bytes(poisoned_bytes)
    import os
    os.utime(gguf, (st.st_atime, st.st_mtime))
    second = md._gguf_sha256(gguf)
    assert second == first  # cache hit, not recomputed


def test_gguf_sha256_missing_file_returns_none(tmp_path, isolated_cache):
    assert md._gguf_sha256(tmp_path / "nope.gguf") is None


def test_build_metadata_includes_commits_and_host(
    fake_model, profile, isolated_cache, monkeypatch
):
    _stub_git_and_sysctl(
        monkeypatch, llama_commit="LLAMA_C", project_commit="PROJ_C",
        cpu="Apple M5 Max", mem_bytes=128 * 1024**3,
    )
    meta = md.build_run_metadata(
        model=fake_model, benchmark="bfcl", rep=1, profile=profile,
        server_bin=Path("~/code/llama.cpp/build/bin/llama-server"),
    )
    assert meta["server"]["llama_cpp_commit"] == "LLAMA_C"
    assert meta["tooling"]["neo_llm_bench_commit"] == "PROJ_C"
    assert meta["host"]["cpu"] == "Apple M5 Max"
    assert meta["host"]["mem_gb"] == 128.0


def test_build_metadata_temperature_override_recorded(
    fake_model, profile, isolated_cache, monkeypatch
):
    _stub_git_and_sysctl(monkeypatch)
    # Per-model temperature defaults to 0.0; override to 0.3.
    assert fake_model.sampling.temperature == 0.0
    meta = md.build_run_metadata(
        model=fake_model, benchmark="humaneval", rep=2, profile=profile,
        server_bin=Path("/bin/true"), temperature_override=0.3,
    )
    assert meta["sampling"]["temperature"] == 0.0
    assert meta["sampling"]["temperature_effective"] == 0.3


def test_build_metadata_no_override_no_effective_key(
    fake_model, profile, isolated_cache, monkeypatch
):
    _stub_git_and_sysctl(monkeypatch)
    meta = md.build_run_metadata(
        model=fake_model, benchmark="bfcl", rep=1, profile=profile,
        server_bin=Path("/bin/true"), temperature_override=None,
    )
    assert "temperature_effective" not in meta["sampling"]


def test_write_run_metadata_creates_dir_and_writes_json(
    fake_model, profile, isolated_cache, monkeypatch, tmp_path
):
    _stub_git_and_sysctl(monkeypatch)
    out_dir = tmp_path / "deep" / "nested" / "path"  # doesn't exist
    meta = md.build_run_metadata(
        model=fake_model, benchmark="bfcl", rep=1, profile=profile,
        server_bin=Path("/bin/true"),
    )
    md.write_run_metadata(out_dir, meta)
    assert (out_dir / "metadata.json").is_file()
    parsed = json.loads((out_dir / "metadata.json").read_text())
    assert parsed["model_id"] == "fake-1.5b-instruct"


def test_build_metadata_unresolvable_fields_omitted(
    fake_model, profile, isolated_cache, monkeypatch
):
    """When git / sysctl fail, the relevant fields should be None or absent,
    not faked. The schema must remain valid JSON regardless."""
    monkeypatch.setattr(md, "_llama_cpp_commit", lambda p: None)
    monkeypatch.setattr(md, "_project_commit", lambda: None)
    monkeypatch.setattr(md, "_sysctl", lambda k: None)
    meta = md.build_run_metadata(
        model=fake_model, benchmark="bfcl", rep=1, profile=profile,
        server_bin=Path("/bin/true"),
    )
    # llama_cpp_commit and neo_llm_bench_commit can be None
    assert meta["server"]["llama_cpp_commit"] is None
    assert meta["tooling"]["neo_llm_bench_commit"] is None
    # Host info is still structurally there even when sysctl unavailable
    assert "arch" in meta["host"] and "os" in meta["host"]
    assert "cpu" not in meta["host"]
    assert "mem_gb" not in meta["host"]
    # JSON-serializable end-to-end.
    json.dumps(meta)
