"""Pipeline & per-model configuration.

Two distinct config shapes live here:

* ``PipelineConfig`` — agent-loop / role / task-type config lifted from
  deluxe. The lifted tests pin its shape; don't change field names.
* ``ModelConfig`` — per-model llama-server invocation config (NEW). Used
  by the bench harness to start a ``LlamaServer`` with the right flags.
  Independent of PipelineConfig so the agent-loop legacy doesn't bleed
  into single-shot benchmarks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class RoleConfig(BaseModel):
    model_key: str
    num_ctx: int = 8192
    max_steps: int = 12
    max_tokens_per_turn: int = 2048
    temperature: float = 0.2
    tools: list[str] = Field(default_factory=list)
    # Prompt-shaping bake-off levers (default to baseline-equivalent).
    # See src/llamabench/agents/prompts.py for the registry.
    system_prompt_id: str = "baseline"
    task_prompt_id: str = "baseline"
    # Per-task-type overlay (Branch B). Empty string = no overlay; use
    # the role-level prompt ids above for every task type. When set,
    # the overlay's by_task mapping wins for matching task types.
    # See ~/.claude/plans/task-type-overlays.md.
    task_overlay_id: str = ""
    # Sampling penalty forwarded as oMLX extra_body. None = omit (current
    # behaviour). Small values (1.02-1.10) discourage repeated tokens; too
    # aggressive corrupts code-gen by forcing identifier divergence.
    repeat_penalty: float | None = None


class TaskTypeConfig(BaseModel):
    description: str = ""
    pipeline: list[str] = Field(default_factory=list)
    architect_prompt: str = ""


class ProfileConfig(BaseModel):
    name: str = ""
    description: str = ""
    memory_budget_gb: int = 64
    peak_model_gb: float = 0.0


class PipelineConfig(BaseModel):
    omlx_base_url: str = "http://127.0.0.1:8000"
    profile: ProfileConfig = Field(default_factory=ProfileConfig)
    models: dict[str, str] = Field(default_factory=dict)
    roles: dict[str, RoleConfig] = Field(default_factory=dict)
    task_types: dict[str, TaskTypeConfig] = Field(default_factory=dict)

    def role(self, name: str) -> RoleConfig:
        if name not in self.roles:
            raise KeyError(f"Unknown pipeline role: {name}")
        return self.roles[name]

    def model_for_role(self, role_name: str) -> str:
        role_cfg = self.role(role_name)
        return self.models[role_cfg.model_key]

    def task_type(self, name: str) -> TaskTypeConfig:
        if name not in self.task_types:
            raise KeyError(f"Unknown task type: {name}. Available: {list(self.task_types)}")
        return self.task_types[name]


def load_config(path: str | Path | None = None) -> PipelineConfig:
    if path is None:
        path = Path(__file__).parent.parent.parent / "configs" / "single_64gb.yaml"
    path = Path(path)
    raw: dict[str, Any] = yaml.safe_load(path.read_text())
    return PipelineConfig.model_validate(raw)


# --- Per-model llama-server config -----------------------------------------


class ServerConfig(BaseModel):
    """llama-server invocation flags. Maps onto ``ServerSpec`` in server.py."""

    n_ctx: int = 8192
    n_gpu_layers: int = 99
    n_threads: int = 6
    n_batch: int = 512
    n_ubatch: int = 512
    flash_attn: str = "auto"
    mmap: bool = True
    mlock: bool = False
    cache_type_k: str = "q8_0"
    cache_type_v: str = "q8_0"
    chat_template: str | None = None
    chat_template_file: str | None = None
    jinja: bool = True


class SamplingConfig(BaseModel):
    """Sampling parameters sent on each /v1/chat/completions request."""

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    repeat_penalty: float = 1.0
    seed: int | None = 42
    max_tokens: int = 2048
    stop: list[str] = Field(default_factory=list)


class ModelConfig(BaseModel):
    """One entry in ``configs/models/<id>.yaml`` — describes how to launch
    a single GGUF under llama-server and how to sample from it.
    """

    id: str
    gguf_path: str  # absolute or ~ — expanded at use site
    alias: str | None = None  # passed to llama-server --alias for /v1/models id
    family: str = ""  # informational only — qwen / llama / granite / smollm / phi / deepseek
    params_b: float = 0.0  # informational
    quant: str = "Q8_0"  # informational
    has_native_tool_template: bool = True
    # Optional per-model BFCL mode override. None = honor the global
    # `--bfcl-mode` (typically "auto"). Set to "inject" for models whose
    # chat template lacks a tools branch — auto-mode never falls back
    # there (no parse error fires) so tools are silently dropped.
    bfcl_mode: str | None = None
    server: ServerConfig = Field(default_factory=ServerConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    notes: str = ""


class BenchProfile(BaseModel):
    """Top-level runner profile (``configs/profile_<name>.yaml``)."""

    memory_budget_gb: int = 8
    peak_model_gb: float = 2.0
    benchmarks: list[str] = Field(default_factory=lambda: ["bfcl"])
    parallel_models: int = 1
    work_dir: str = "~/.llamabench/bench-workspace"
    keep_loaded: bool = False
    server_host: str = "127.0.0.1"
    server_port: int = 8080
    server_bin: str = "/Users/mtimpe/code/llama.cpp/build/bin/llama-server"


def load_model_config(path: str | Path) -> ModelConfig:
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
    return ModelConfig.model_validate(raw)


def load_profile(path: str | Path) -> BenchProfile:
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
    return BenchProfile.model_validate(raw)
