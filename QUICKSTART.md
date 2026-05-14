# Quickstart

Get neo-llm-bench running on a new Mac and reproduce the round-2 leaderboard.

## Prerequisites

- macOS on Apple Silicon (tested on A18 Pro / 8 GB RAM)
- `uv` (`brew install uv`)
- Python 3.11 (`uv` will pin it from `.python-version`)
- `git`, `git-lfs`, `cmake`, `ccache` (`brew install cmake ccache git-lfs`)
- HuggingFace CLI: `pip install 'huggingface_hub[cli]'` or `uv tool install huggingface_hub[cli]`
- GitHub CLI if you'll push results: `brew install gh`

## 1. Clone & install dependencies

```bash
git clone git@github.com:michaeldtimpe/neo-llm-bench.git
cd neo-llm-bench
uv sync                            # installs runtime + dev deps incl. bfcl-eval
uv run pytest tests/test_bfcl_grade.py tests/test_bfcl_adapter.py  # 46 tests should pass
```

## 2. Build llama.cpp with Metal

```bash
mkdir -p ~/code && cd ~/code
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DLLAMA_CURL=ON
cmake --build build -j --config Release
```

This produces `~/code/llama.cpp/build/bin/llama-server`, which is the default path in `configs/profile_8gb.yaml` and `src/llamabench/server.py`. If you build elsewhere, edit `profile_8gb.yaml`'s `server_bin`.

## 3. Download model GGUFs

The four finalists (3 from Round 2 + smollm3 added in Round 4/Branch D):

| model | HuggingFace repo | quant | size |
|---|---|---|---|
| qwen25-1.5b-instruct | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` | Q8_0 | ~1.6 GB |
| qwen25-coder-1.5b-instruct | `Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF` | Q8_0 | ~1.6 GB |
| granite33-2b-instruct | `ibm-granite/granite-3.3-2b-instruct-GGUF` | Q8_0 | ~2.4 GB |
| smollm3-3b-instruct | `ggml-org/SmolLM3-3B-GGUF` | Q8_0 | ~3.1 GB |

Download to `~/models/`:

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p ~/models
hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF             qwen2.5-1.5b-instruct-q8_0.gguf       --local-dir ~/models/
hf download Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF       qwen2.5-coder-1.5b-instruct-q8_0.gguf --local-dir ~/models/
hf download ibm-granite/granite-3.3-2b-instruct-GGUF    granite-3.3-2b-instruct-Q8_0.gguf     --local-dir ~/models/
hf download ggml-org/SmolLM3-3B-GGUF                    SmolLM3-Q8_0.gguf                     --local-dir ~/models/

# Rename to match configs/models/*.yaml gguf_path (Qwen repos use lowercase
# `q8_0`; smollm3 file ships without the model name prefix).
mv ~/models/qwen2.5-1.5b-instruct-q8_0.gguf        ~/models/Qwen2.5-1.5B-Instruct-Q8_0.gguf
mv ~/models/qwen2.5-coder-1.5b-instruct-q8_0.gguf  ~/models/Qwen2.5-Coder-1.5B-Instruct-Q8_0.gguf
mv ~/models/SmolLM3-Q8_0.gguf                      ~/models/SmolLM3-3B-Q8_0.gguf
# granite file already matches its YAML — no rename needed.
```

Check filenames match the per-model YAMLs (`configs/models/<id>.yaml`). The configs use tildes (`~/models/...`) so they're portable.

## 4. Smoke test — single problem

```bash
uv run python scripts/run_bakeoff.py \
    --models qwen25-1.5b-instruct \
    --benchmarks bfcl \
    --bfcl-limit 3 \
    --bfcl-categories simple_python \
    --rep 99 \
    --force
```

You should see 3 problems run, results land at `acceptance/bfcl/qwen25-1.5b-instruct/rep_99/simple_python/*.json`, and a `summary.json` written. The first call will be slow (model loading); subsequent are fast.

## 5. Reproduce the round-2 leaderboard

Three commands (chain with `&&` to run sequentially in the background; the full multi-spectrum run takes ~5h compute, but spreads over more wall-clock time on a constrained Mac that puts itself to sleep).

```bash
# 5a. BFCL deep (curated 150/cat + live up-to-100/cat)
uv run python scripts/run_bakeoff.py \
    --models qwen25-1.5b-instruct qwen25-coder-1.5b-instruct granite33-2b-instruct \
    --benchmarks bfcl --rep 1 --bfcl-limit 150 --bfcl-mode auto --force

uv run python scripts/run_bakeoff.py \
    --models qwen25-1.5b-instruct qwen25-coder-1.5b-instruct granite33-2b-instruct \
    --benchmarks bfcl --rep 1 --bfcl-limit 100 \
    --bfcl-categories live_simple live_multiple live_parallel live_parallel_multiple live_irrelevance live_relevance \
    --bfcl-mode auto --force

# 5b. HumanEval at t=0.3
uv run python scripts/run_bakeoff.py \
    --models qwen25-1.5b-instruct qwen25-coder-1.5b-instruct granite33-2b-instruct \
    --benchmarks humaneval --rep 2 --humaneval-limit 164 --temperature 0.3 --force

# 5c. HumanEval at t=0.7
uv run python scripts/run_bakeoff.py \
    --models qwen25-1.5b-instruct qwen25-coder-1.5b-instruct granite33-2b-instruct \
    --benchmarks humaneval --rep 3 --humaneval-limit 164 --temperature 0.7 --force
```

For deterministic baseline (t=0.0), use `--rep 0` and omit `--temperature` — it picks up the per-model YAML's `sampling.temperature` (which is 0.0 by default).

## 6. Generate the reports

```bash
# BFCL leaderboard from rep_1 data
uv run python scripts/grade_bakeoff.py --rep 1 \
    --models qwen25-1.5b-instruct qwen25-coder-1.5b-instruct granite33-2b-instruct

# Failure-mode breakdown
uv run python scripts/failure_modes.py --rep 1
```

HumanEval pass@1 is in each `acceptance/humaneval/<model>/rep_N/summary.json`.

The hand-written `graded_report.md` / `graded_failure_modes.md` files contain the dashboard with CIs, head-to-head analysis, decision framework. Regenerate them by hand from the script outputs above.

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `llama-server: command not found` | not in PATH or not built | check `~/code/llama.cpp/build/bin/llama-server` exists; or edit `profile_8gb.yaml` `server_bin` |
| HTTP 500 from llama-server on BFCL | jinja can't parse model's tool shape | the adapter auto-falls-back to inject mode; if it doesn't, set `bfcl_mode: inject` in the model YAML |
| Wall times degrade over hours | the Mac is putting itself to sleep | normal; `time.monotonic()` pauses during sleep so the runner's reported wall is accurate compute time |
| `ModuleNotFoundError: bfcl_eval` | dev deps not installed | `uv sync` (includes dev group), or `uv pip install bfcl-eval` |
| Per-problem wall keeps growing as run continues | swap pressure (granite-style) | check `sysctl vm.swapusage`; the 2B granite model swaps on 8 GB Macs; expect 2.5× slowdown |
