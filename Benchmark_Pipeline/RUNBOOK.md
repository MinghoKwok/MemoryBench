# RUNBOOK

Short runbook for the official agentic `m2a` path in this repo.

## Canonical Path

- Benchmark env: `memorybench`
- Embedding service env: `vllm`
- Method config: `Benchmark_Pipeline/config/methods/m2a.yaml`
- Model metadata config: `Benchmark_Pipeline/config/models/gpt_4o_mini.yaml`
- Example task: `Benchmark_Pipeline/config/tasks_external/home_renovation_interior_design.yaml`

Important:

- The real LLM for agentic `m2a` is pinned inside `m2a.yaml`.
- In this repo, that is intentionally `gpt-4o-mini`.
- `--model-config` still affects artifact naming. Keep it aligned.

## One-Time Checks

- Do not use base conda.
- `memorybench` must have working CUDA PyTorch.
- `vllm` must serve `siglip2-base-patch16-384` on `127.0.0.1:8050`.

Quick checks:

```bash
conda run -n memorybench python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
curl -sS http://127.0.0.1:8050/v1/models
```

Expected:

- `torch 2.9.0+cu128`
- `True`
- model list contains `siglip2-base-patch16-384`

## Start SigLIP2

```bash
/common/home/mg1998/MemoryBench/Benchmark_Pipeline/scripts/start_siglip2_vllm.sh
```

## Run Official M2A

```bash
/common/home/mg1998/MemoryBench/Benchmark_Pipeline/scripts/run_official_m2a.sh
```

Default target:

- task: `home_renovation_interior_design`
- model metadata: `gpt_4o_mini`
- method: `m2a`

Useful overrides:

```bash
MAX_QUESTIONS=1 /common/home/mg1998/MemoryBench/Benchmark_Pipeline/scripts/run_official_m2a.sh
TASK_CONFIG=/abs/path/to/task.yaml /common/home/mg1998/MemoryBench/Benchmark_Pipeline/scripts/run_official_m2a.sh
OUTPUT_ROOT=/tmp/m2a_runs /common/home/mg1998/MemoryBench/Benchmark_Pipeline/scripts/run_official_m2a.sh
```

## What The Wrapper Already Does

- loads `/common/home/mg1998/MemoryBench/.env.local`
- unsets `HUGGING_FACE_HUB_TOKEN`
- checks `OPENAI_API_KEY`
- checks `http://127.0.0.1:8050/v1/models`
- runs unbuffered Python with persistent logs

## Watch A Live Run

The wrapper prints the log file path at startup.

Key files for a live run:

- `Benchmark_Pipeline/logs/*.log`
- `Benchmark_Pipeline/runs/<task>/<timestamp>_<model>_<method>/progress.json`
- `Benchmark_Pipeline/runs/<task>/<timestamp>_<model>_<method>/predictions.jsonl`
- `Benchmark_Pipeline/runs/<task>/<timestamp>_<model>_<method>/metrics.json`

Typical checks:

```bash
tail -f /common/home/mg1998/MemoryBench/Benchmark_Pipeline/logs/<logfile>.log
cat /common/home/mg1998/MemoryBench/Benchmark_Pipeline/runs/<task>/<run_dir>/progress.json
wc -l /common/home/mg1998/MemoryBench/Benchmark_Pipeline/runs/<task>/<run_dir>/predictions.jsonl
```

## Known Pitfalls

### Base conda fails

Symptom:

```text
libtorch_cuda.so: undefined symbol: ncclGroupSimulateEnd
```

Fix:

- use `conda run -n memorybench ...`
- use `conda run -n vllm ...`

### Bad HF token placeholder breaks requests

Symptom:

- header / ASCII encoding failures during HF or HTTP calls

Fix:

```bash
unset HUGGING_FACE_HUB_TOKEN
```

### No `8050` service

Symptom:

- `curl http://127.0.0.1:8050/v1/models` fails

Fix:

```bash
/common/home/mg1998/MemoryBench/Benchmark_Pipeline/scripts/start_siglip2_vllm.sh
```

### Old docs mention `m2a_full + CLIP`

Ignore old `m2a_full`, `CLIP`, and `cu130` guidance for the official path.

Use:

- `config/methods/m2a.yaml`
- local `all-MiniLM-L6-v2`
- local-service `SigLIP2` via `vLLM`

### Agentic `m2a` hits OpenAI TPM

Symptoms:

- `429`
- long retries during memory build

Current behavior:

- retries are already hardened
- timeout is explicit
- progress and logs are persisted

This is expected on longer runs. Do not kill the job just because QA output has not started yet. `m2a` must first build memory from all sessions.

## Current Reference Outputs

Completed full run:

- run dir: `/common/home/mg1998/MemoryBench/Benchmark_Pipeline/runs/home_renovation_interior_design/20260326_214923_gpt_4o_mini_m2a`
- summary json: `/common/home/mg1998/MemoryBench/Benchmark_Pipeline/output/home_renovation_interior_design/results_home_renovation_interior_design__gpt_4o_mini__m2a.json`
