# Environment Setup for Official M2A

This file documents the working setup for the official agentic `m2a` path in this repo.

The current `m2a` runtime is a two-process setup:

1. `memorybench` conda env runs the benchmark, local `all-MiniLM-L6-v2`, and the agentic M2A loop.
2. `vllm` conda env serves `google/siglip2-base-patch16-384` at `http://127.0.0.1:8050/v1`.

## Canonical Configs

Use these configs together:

- Method: `Benchmark_Pipeline/config/methods/m2a.yaml`
- Model metadata: `Benchmark_Pipeline/config/models/gpt_4o_mini.yaml`
- Example task: `Benchmark_Pipeline/config/tasks_external/home_renovation_interior_design.yaml`

Important runtime fact:

- The actual LLM used by agentic `m2a` is pinned inside `config/methods/m2a.yaml`.
- In this repo that value is intentionally fixed to `gpt-4o-mini`.
- `--model-config` still affects run metadata and output file naming, so it should match the same model to avoid misleading artifacts.

## Verified Working Environments

### `memorybench`

Use:

```bash
conda run -n memorybench ...
```

Verified package versions in the working setup:

| Package | Version |
|---------|---------|
| PyTorch | `2.9.0+cu128` |
| sentence-transformers | `5.3.0` |
| transformers | `5.2.0` |
| PyYAML | `6.0.3` |

Expected check:

```bash
conda run -n memorybench python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

This should print `2.9.0+cu128` and `True`.

### `vllm`

Use:

```bash
conda run -n vllm ...
```

Verified package versions in the working setup:

| Package | Version |
|---------|---------|
| vLLM | `0.11.2` |
| PyTorch | `2.9.0+cu128` |

## Start the SigLIP2 Service

```bash
/common/home/mg1998/MemoryBench/Benchmark_Pipeline/scripts/start_siglip2_vllm.sh
```

Health check:

```bash
curl -sS http://127.0.0.1:8050/v1/models
```

The returned model list must include `siglip2-base-patch16-384`.

## Run Official M2A

Generic wrapper:

```bash
/common/home/mg1998/MemoryBench/Benchmark_Pipeline/scripts/run_official_m2a.sh
```

The wrapper:

- loads `/common/home/mg1998/MemoryBench/.env.local` if present
- unsets `HUGGING_FACE_HUB_TOKEN`
- checks that `OPENAI_API_KEY` is present
- checks that `http://127.0.0.1:8050/v1/models` exposes `siglip2-base-patch16-384`
- runs `Benchmark_Pipeline.run_benchmark` from the repo root

Default target:

- task: `home_renovation_interior_design`
- model metadata: `gpt_4o_mini`
- method: `m2a`

Useful overrides:

```bash
MAX_QUESTIONS=1 /common/home/mg1998/MemoryBench/Benchmark_Pipeline/scripts/run_official_m2a.sh
OUTPUT_ROOT=/tmp/m2a_runs /common/home/mg1998/MemoryBench/Benchmark_Pipeline/scripts/run_official_m2a.sh
TASK_CONFIG=/abs/path/to/task.yaml /common/home/mg1998/MemoryBench/Benchmark_Pipeline/scripts/run_official_m2a.sh
```

## Known Pitfalls

### Base conda is broken for this repo

Symptom:

```text
libtorch_cuda.so: undefined symbol: ncclGroupSimulateEnd
```

Cause:

- Using `/common/users/mg1998/miniforge3/bin/python`
- This base environment has an incompatible NCCL/PyTorch combination for the repo

Fix:

- Do not use base conda
- Use `conda run -n memorybench ...` for the benchmark
- Use `conda run -n vllm ...` for the SigLIP2 service

### `HUGGING_FACE_HUB_TOKEN` can break HF and httpx

Symptom:

- ASCII/header encoding errors during model loading or HTTP requests

Cause:

- A local shell placeholder value such as `你的token`

Fix:

```bash
unset HUGGING_FACE_HUB_TOKEN
```

The provided scripts already do this.

### Legacy setup documentation is obsolete

The active official path in this repo is:

- `config/methods/m2a.yaml`
- local `all-MiniLM-L6-v2`
- remote local-service `SigLIP2` via `vLLM`

Do not follow older `CLIP` or `cu130` instructions for current official M2A runs.
