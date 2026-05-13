"""C0 ablation: how much does the stub-executor return-text shift
BFCL agent-mode behavior?

10 BFCL problems × 3 stub-text variants on qwen25-1.5b-instruct.
Variants:
  current      — f"[stub:{name}] called with args={args}"  (today's default)
  opaque       — '{"status":"success"}'                     (minimal plausible API)
  empty        — ''                                         (just acknowledges resolution)

We also run RAW mode on the same 10 problems as a baseline so the
"closest-to-raw" decision criterion has a reference.

Per-variant we record:
  total_tool_calls per problem
  steps (run_agent loop iterations) per problem
  schema_rejects per problem

If the distributions are within ±10% of each other across variants AND
close to raw-mode call counts, the current stub doesn't materially
distort behavior and we keep it (parity with the existing scaffold).

This script is a one-off experiment; not wired into the bake-off and
intended for deletion after the report has captured its finding.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from llamabench.agents.loop import run_agent  # noqa: E402
from llamabench.backend import Backend  # noqa: E402
from llamabench.config import RoleConfig, load_model_config, load_profile  # noqa: E402
from llamabench.runner import model_to_server_spec  # noqa: E402
from llamabench.server import LlamaServer  # noqa: E402

from benchmarks.bfcl.adapter import (  # noqa: E402
    BFCL_SYSTEM_PROMPT, _problem_messages, _problem_tools, load_problems,
    run_problem_raw,
)


VARIANTS = {
    "current": lambda name, args: (f"[stub:{name}] called with args={args}", None),
    "opaque":  lambda name, args: ('{"status":"success"}', None),
    "empty":   lambda name, args: ("", None),
}


def _build_tool_fns(tool_defs, variant_fn):
    def wrap(name):
        def _fn(args: dict[str, Any]) -> tuple[str, str | None]:
            return variant_fn(name, args)
        return _fn
    return {td.name: wrap(td.name) for td in tool_defs}


def _agent_one(backend: Backend, role_cfg: RoleConfig, problem: dict, variant_fn):
    messages_seed = _problem_messages(problem)
    user_text = "\n\n".join(m["content"] for m in messages_seed if m.get("role") == "user")
    tool_defs = _problem_tools(problem)
    tool_fns = _build_tool_fns(tool_defs, variant_fn)
    r = run_agent(
        backend=backend, role_cfg=role_cfg,
        system_prompt=BFCL_SYSTEM_PROMPT, task_prompt=user_text,
        tool_defs=tool_defs, tool_fns=tool_fns,
    )
    return {
        "n_tool_calls": r.tool_calls_total,
        "steps": r.steps,
        "schema_rejects": r.schema_rejects,
        "aborted": r.aborted,
        "completion_tokens": r.completion_tokens,
    }


def _raw_one(backend: Backend, problem: dict, max_tokens: int, temperature: float):
    r = run_problem_raw(
        backend, problem, max_tokens=max_tokens, temperature=temperature, mode="auto",
    )
    return {
        "n_tool_calls": len(r.actual_calls),
        "completion_tokens": r.completion_tokens,
        "error": r.error,
    }


def _pick_problems() -> list[dict]:
    """4 parallel + 4 parallel_multiple + 2 irrelevance — small but balanced."""
    out = []
    out += load_problems("parallel", limit=4)
    out += load_problems("parallel_multiple", limit=4)
    out += load_problems("irrelevance", limit=2)
    return out


def main() -> int:
    profile = load_profile(ROOT / "configs" / "profile_m5max.yaml")
    model = load_model_config(ROOT / "configs" / "models" / "qwen25-1.5b-instruct.yaml")
    spec = model_to_server_spec(model)
    if not spec.model_path.is_file():
        print(f"GGUF missing at {spec.model_path}", file=sys.stderr)
        return 2

    # Match the agent-mode RoleConfig the runner will build later (C2).
    role_cfg = RoleConfig(
        model_key="ablation",
        max_steps=12,
        max_tokens_per_turn=model.sampling.max_tokens,
        num_ctx=model.server.n_ctx,
        temperature=model.sampling.temperature,
    )

    problems = _pick_problems()
    print(f"# stub ablation — qwen25-1.5b, {len(problems)} problems "
          f"(4 parallel + 4 parallel_multiple + 2 irrelevance)", file=sys.stderr)

    # Auto-port
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((profile.server_host, 0))
        port = int(s.getsockname()[1])

    bin_path = Path(profile.server_bin).expanduser()
    server = LlamaServer(spec=spec, host=profile.server_host, port=port, bin_path=bin_path,
                        n_parallel=profile.max_parallel_requests)
    server.start()
    backend = Backend(base_url=server.base_url, model=spec.alias or model.id)

    rows: list[dict] = []
    t0 = time.monotonic()
    try:
        # RAW baseline first.
        print(f"[{time.strftime('%H:%M:%S')}] raw baseline...", file=sys.stderr)
        for p in problems:
            pid = p.get("id", "?")
            res = _raw_one(backend, p, max_tokens=model.sampling.max_tokens,
                           temperature=model.sampling.temperature)
            rows.append({"variant": "raw", "problem_id": pid, **res})

        # Each agent variant.
        for variant_name, variant_fn in VARIANTS.items():
            print(f"[{time.strftime('%H:%M:%S')}] agent variant={variant_name}...",
                  file=sys.stderr)
            for p in problems:
                pid = p.get("id", "?")
                res = _agent_one(backend, role_cfg, p, variant_fn)
                rows.append({"variant": variant_name, "problem_id": pid, **res})
    finally:
        server.stop()
    wall = time.monotonic() - t0
    print(f"\n[{time.strftime('%H:%M:%S')}] ablation done in {wall:.0f}s", file=sys.stderr)

    out = ROOT / "acceptance" / "_logs" / "stub_ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"raw rows written to {out}")

    # Summary table.
    print("\n# Summary (averaged across {} problems per variant)".format(len(problems)))
    print(f"{'variant':<10}  {'mean_calls':>10}  {'mean_steps':>10}  {'mean_rejects':>12}  {'mean_comp_tok':>13}")
    for v in ["raw", "current", "opaque", "empty"]:
        rs = [r for r in rows if r["variant"] == v]
        n = len(rs)
        m_calls = sum(r["n_tool_calls"] for r in rs) / n
        m_steps = sum(r.get("steps", 0) for r in rs) / n
        m_rej = sum(r.get("schema_rejects", 0) for r in rs) / n
        m_tok = sum(r.get("completion_tokens", 0) for r in rs) / n
        print(f"{v:<10}  {m_calls:>10.2f}  {m_steps:>10.2f}  {m_rej:>12.2f}  {m_tok:>13.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
