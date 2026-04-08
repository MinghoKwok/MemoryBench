# Benchmark Pipeline

This directory is a small benchmark scaffold for multimodal memory experiments, not just a single inference script.

The benchmark separates:
- task data and evaluation
- model routing
- memory method selection
- run artifact storage

The current sample task is `brand_memory_test`. The benchmark supports local and API-backed model routers, plus multiple memory methods:
- `full_context_multimodal`
- `full_context_text_only`
- `semantic_rag_multimodal`
- `semantic_rag_text_only`
- `m2a`
- `mma`

For MemEye task design and `point` annotation rules, read:

- `Benchmark_Pipeline/MemEye_Annotation_Guide.md`

## Layout

- `run_benchmark.py`: modular benchmark entrypoint
- `run_matrix.py`: model x method matrix runner
- `run_legacy_benchmark.py`: generic legacy-compatible single-config entrypoint
- `run_pittads.py`: compatibility shim for older commands
- `benchmark/`: dataset loading, method selection, evaluation, run orchestration
- `router/`: model router implementations
- `config/tasks/`: task configs
- `config/models/`: model configs
- `config/methods/`: method configs
- `data/`: example dialogue and images
- `runs/`: per-run artifacts
- `output/results_brand_memory_test.json`: optional legacy output file for the sample task

## Requirements

Recommended:
- Python 3.10+
- Either a local vision-language checkpoint or API credentials for an online model provider

Python packages are listed in `requirements.txt`.

## Environment Setup

Create and activate a Python 3.10 environment with Conda or another environment manager, then install dependencies:

```bash
pip install -r requirements.txt
```

For API-backed routers, load credentials before running:

```bash
set -a
source .env.local
set +a
```

If you maintain a machine-specific setup, keep those details in `README.local.md`, which is intended to stay untracked.

## Benchmark Model

This repo is moving toward three independent benchmark axes:
- task: what dataset and evaluation rules to use
- model: what router/checkpoint/provider to use
- method: how to construct memory context for each QA

That lets you compare, for example, the same task under:
- different models
- the same model with different memory strategies
- the same method across tasks

## Quick Start

Activate your project environment first.

Run the modular benchmark from the repo root:

```bash
python -m Benchmark_Pipeline.run_benchmark \
  --task-config Benchmark_Pipeline/config/tasks/brand_memory_test.yaml \
  --model-config Benchmark_Pipeline/config/models/qwen_local_default.yaml \
  --method-config Benchmark_Pipeline/config/methods/full_context_multimodal.yaml
```

Script execution still works if you prefer it:

```bash
python Benchmark_Pipeline/run_benchmark.py \
  --task-config Benchmark_Pipeline/config/tasks/brand_memory_test.yaml \
  --model-config Benchmark_Pipeline/config/models/qwen_local_default.yaml \
  --method-config Benchmark_Pipeline/config/methods/full_context_multimodal.yaml
```

The generic legacy command still works:

```bash
python -m Benchmark_Pipeline.run_legacy_benchmark --config Benchmark_Pipeline/config/default.yaml
```

The older task-specific entrypoint still exists for compatibility:

```bash
python Benchmark_Pipeline/run_pittads.py --config Benchmark_Pipeline/config/default.yaml
```

## Common Experiments

Switch method:

```bash
python -m Benchmark_Pipeline.run_benchmark \
  --task-config Benchmark_Pipeline/config/tasks/brand_memory_test.yaml \
  --model-config Benchmark_Pipeline/config/models/qwen_local_default.yaml \
  --method-config Benchmark_Pipeline/config/methods/clue_only.yaml
```

Run the dense retrieval baseline:

```bash
python -m Benchmark_Pipeline.run_benchmark \
  --task-config Benchmark_Pipeline/config/tasks/brand_memory_test.yaml \
  --model-config Benchmark_Pipeline/config/models/gpt_4_1_nano.yaml \
  --method-config Benchmark_Pipeline/config/methods/semantic_rag_text_only.yaml
```

Run the lightweight M2A-style baseline:

```bash
python -m Benchmark_Pipeline.run_benchmark \
  --task-config Benchmark_Pipeline/config/tasks/brand_memory_test.yaml \
  --model-config Benchmark_Pipeline/config/models/gpt_4_1_nano.yaml \
  --method-config Benchmark_Pipeline/config/methods/m2a_lite.yaml
```

Limit to a quick smoke test:

```bash
python -m Benchmark_Pipeline.run_benchmark \
  --task-config Benchmark_Pipeline/config/tasks/brand_memory_test.yaml \
  --model-config Benchmark_Pipeline/config/models/qwen_local_default.yaml \
  --method-config Benchmark_Pipeline/config/methods/full_context_multimodal.yaml \
  --max-questions 1
```

Override evaluation mode:

```bash
python -m Benchmark_Pipeline.run_benchmark \
  --task-config Benchmark_Pipeline/config/tasks/brand_memory_test.yaml \
  --model-config Benchmark_Pipeline/config/models/qwen_local_default.yaml \
  --method-config Benchmark_Pipeline/config/methods/full_context_multimodal.yaml \
  --mode both
```

Enable rich evaluation metrics (F1, BLEU, BERTScore, LLM judge):

```bash
python -m Benchmark_Pipeline.run_benchmark \
  --task-config Benchmark_Pipeline/config/tasks/brand_memory_test.yaml \
  --model-config Benchmark_Pipeline/config/models/gpt_4_1_nano.yaml \
  --method-config Benchmark_Pipeline/config/methods/full_context_multimodal.yaml \
  --enable-bert-score \
  --enable-llm-judge \
  --judge-model gpt-4.1-nano \
  --judge-max-retries 3 \
  --judge-timeout 60
```

Run a model x method matrix:

```bash
python -m Benchmark_Pipeline.run_matrix \
  --task-config Benchmark_Pipeline/config/tasks/brand_memory_test.yaml \
  --model-config Benchmark_Pipeline/config/models/gpt_4_1_nano.yaml \
  --method-config Benchmark_Pipeline/config/methods/full_context_multimodal.yaml \
  --method-config Benchmark_Pipeline/config/methods/clue_only.yaml \
  --method-config Benchmark_Pipeline/config/methods/semantic_rag_text_only.yaml \
  --method-config Benchmark_Pipeline/config/methods/m2a_lite.yaml
```

## Config Structure

Example task config:

```yaml
name: brand_memory_test

dataset:
  dialog_json: data/dialog/Brand_Memory_Test.json
  image_root: data/image

eval:
  mode: open
  output_json: output/results_brand_memory_test.json
  max_questions: 0
```

Example model config:

```yaml
name: qwen2_vl_local
provider: qwen_local
model_path: /path/to/local/Qwen2-VL
max_new_tokens: 128
```

Example method config:

```yaml
name: full_context_multimodal
```

`semantic_rag_text_only` accepts retrieval-specific parameters:

```yaml
name: semantic_rag_text_only
retrieval_backend: dense_text
top_k: 1
neighbor_window: 0
text_embedding_model: all-MiniLM-L6-v2
```

`m2a_lite` uses a simple two-stage memory lookup:

```yaml
name: m2a_lite
semantic_top_k: 3
raw_top_k: 2
neighbor_window: 1
semantic_lexical_weight: 0.25
semantic_dense_weight: 0.75
raw_lexical_weight: 0.4
raw_dense_weight: 0.6
```

The default `config/default.yaml` composes the same pieces in one file. The older `config/tasks/pittads.yaml` remains as a legacy alias for backward compatibility.

## Memory Methods

- `full_context_multimodal`: feeds all rounds (with images) from all sessions, truncated to context window.
- `full_context_text_only`: feeds all rounds as text only (image captions replace images); requires `image_caption` in dataset.
- `semantic_rag_multimodal`: dense retrieval fusing text and image embeddings (text via all-MiniLM-L6-v2, image via SigLIP2 or local CLIP fallback).
- `semantic_rag_text_only`: dense text-only retrieval using all-MiniLM-L6-v2; captions included in index if present.
- `m2a`: two-phase agentic memory — builds semantic memory store from sessions, then answers via ReAct loop.
- `mma`: confidence-aware multimodal memory agent with source/temporal/consensus scoring.

## M2A Note

`m2a_lite` is not a full reproduction of the M2A paper/system.

In this repo, `m2a_lite` should be understood as an M2A-inspired benchmark baseline that keeps only the core high-level idea:

- a lightweight semantic memory layer
- a raw evidence layer
- retrieval from semantic memory first, then back-linking to raw rounds

It does not currently implement the full M2A stack such as:

- a complete memory manager / agent loop
- continuous memory writing and update policies
- a full multimodal retrieval pipeline with separate external embedding services
- the original system's full orchestration and component design

So the intended comparison in this benchmark is:

- `semantic_rag_text_only` / `semantic_rag_multimodal`: direct retrieval over raw rounds
- `m2a`: full agentic memory with ReAct loop

## Supported Data Format

The expected top-level structure is:

```json
{
  "character_profile": { "...": "..." },
  "multi_session_dialogues": [
    {
      "session_id": "D1",
      "date": "2024-03-10",
      "dialogues": [
        {
          "round": "D1:1",
          "user": "...",
          "assistant": "...",
          "input_image": ["../image/<scenario>/<file>.jpg"]
        }
      ]
    }
  ],
  "human-annotated QAs": [
    {
      "point": [["X2"], ["Y1"]],
      "question": "...",
      "answer": "...",
      "session_id": ["D1"],
      "clue": ["D1:1"]
    }
  ]
}
```

Also accepted for the QA list:
- `human_annotated_qas`
- `qas`

- `point` should use MemEye binocular coordinates, for example `[['X2'], ['Y1']]`
- multiple labels are allowed when justified, for example `[['X0', 'X4'], ['Y3']]`
- `input_image` may use relative paths such as `../image/...`, `./image/...`, `image/...`, or `data/image/...`
- absolute image paths are supported
- `file://...` image paths are supported
- `image_root` is used as an explicit override when image paths need rebasing
- config and dataset paths resolve correctly whether you run from repo root or from inside `Benchmark_Pipeline`

## Adding A New Task

If another partner wants to add a new dataset with the same format, the recommended workflow is:

1. Put the dialogue JSON under `data/dialog/`, for example:

```bash
Benchmark_Pipeline/data/dialog/My_Task.json
```

2. Put the corresponding images under `data/image/<task_name>/`, for example:

```bash
Benchmark_Pipeline/data/image/My_Task/...
```

3. Add a task config under `config/tasks/`, for example:

```yaml
name: my_task

dataset:
  dialog_json: data/dialog/My_Task.json
  image_root: data/image

eval:
  mode: open
  output_json: output/results_my_task.json
  max_questions: 0
```

## Using An External Data Repo

If your MemEye data lives outside this repo, for example in a separate local clone of a Hugging Face dataset repo, you do not need to copy it into `Benchmark_Pipeline/data`.

Expected external layout:

```bash
<external_data_root>/
  dialog/
    *.json
  image/
    ...
```

Generate task configs that point to the external absolute paths:

```bash
python Benchmark_Pipeline/register_external_data.py \
  --data-root /path/to/external/data \
  --overwrite
```

This writes generated task configs under:

```bash
Benchmark_Pipeline/config/tasks_external/
```

Then run any generated task config normally, for example:

```bash
python -m Benchmark_Pipeline.run_benchmark \
  --task-config Benchmark_Pipeline/config/tasks_external/chat_ui_memory_test.yaml \
  --model-config Benchmark_Pipeline/config/models/gpt_4_1_nano.yaml \
  --method-config Benchmark_Pipeline/config/methods/clue_only.yaml
```

## Recommended HF Dataset Workflow

Recommended separation of concerns:

- keep code, benchmark logic, generators, configs, and docs in this GitHub repo
- keep benchmark `data/` as the canonical dataset payload in the Hugging Face dataset repo
- keep images in the Hugging Face dataset repo rather than the GitHub code repo when the files are large
- treat local `Benchmark_Pipeline/data/` as a synced working copy from the HF dataset repo, not as the long-term source of truth

This is the cleaner workflow for ongoing benchmark development:

1. Pull the latest dataset from Hugging Face into your local working copy.
2. Make dataset edits locally.
3. Validate with `run_benchmark`.
4. Push dataset changes back to the Hugging Face dataset repo.

The sync helper script uses `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` from your environment:

```bash
set -a
source .env.local
set +a
```

Check sync status:

```bash
python Benchmark_Pipeline/sync_hf_data.py status
```

Pull HF dataset `data/` into local `Benchmark_Pipeline/data`:

```bash
python Benchmark_Pipeline/sync_hf_data.py pull
```

Commit local `Benchmark_Pipeline/data` changes into the HF dataset repo working copy:

```bash
python Benchmark_Pipeline/sync_hf_data.py push \
  --commit-message "Update MemEye benchmark data"
```

Push those committed changes back to Hugging Face:

```bash
python Benchmark_Pipeline/sync_hf_data.py push \
  --commit-message "Update MemEye benchmark data" \
  --git-user-name "Your Name" \
  --git-user-email "you@example.com" \
  --push
```

By default the HF repo working copy lives at:

```bash
~/.cache/memeye_hf/MemEye
```

4. Run the benchmark against the new task config:

```bash
python -m Benchmark_Pipeline.run_benchmark \
  --task-config Benchmark_Pipeline/config/tasks/my_task.yaml \
  --model-config Benchmark_Pipeline/config/models/gpt_4_1_nano.yaml \
  --method-config Benchmark_Pipeline/config/methods/full_context_multimodal.yaml
```

This lets each partner add a task instance without changing framework code.

## Package Usage

The pipeline now works both as a script collection and as a standard Python package.

Recommended from the repo root:

```bash
python -m Benchmark_Pipeline.run_benchmark ...
python -m Benchmark_Pipeline.run_matrix ...
python -m Benchmark_Pipeline.run_legacy_benchmark ...
```

Compatibility script entrypoints still work:

```bash
python Benchmark_Pipeline/run_benchmark.py ...
python Benchmark_Pipeline/run_matrix.py ...
python Benchmark_Pipeline/run_pittads.py ...
```

## Output

Each benchmark run now writes a dedicated directory under `runs/`, typically containing:
- `config.json`: resolved run config
- `metrics.json`: run metadata and aggregate metrics
- `predictions.jsonl`: one row per prediction

Matrix runs also write a timestamped directory under `runs/<task>/matrices/`, typically containing:
- `summary.json`
- `summary.csv`
- `summary.md`

If `eval.output_json` is set, the benchmark also writes a legacy summary JSON for backward compatibility.

For `open` mode, each result includes:
- predicted answer
- ground truth
- exact-match (`em`) and contains-GT flags
- token-level F1 (`f1`) with Porter stemming
- BLEU-4 / BLEU-1 / BLEU-2 (`bleu`, `bleu_1`, `bleu_2`) via NLTK smoothing
- BERTScore F1 (`bert`) — enabled with `--enable-bert-score`; downloads roberta-large (~440 MB) on first use
- LLM-as-a-judge score (`judge`) ∈ {0, 0.25, 0.5, 0.75, 1.0} — enabled with `--enable-llm-judge`
- latency

For `mcq` mode, each result includes:
- raw model output
- extracted choice (`A`, `B`, or `C`)
- valid-choice flag
- latency

Each row also records benchmark metadata such as:
- `method_name`
- `history_turns`
- `source_sessions`
- `clue_rounds`

## Rich Evaluation Metrics

By default the pipeline computes exact-match, contains-GT, F1, and BLEU on every open-ended QA. Two additional metric families can be enabled at runtime:

### BERTScore

Enable with `--enable-bert-score`. Uses `roberta-large` (rescaled). Downloads ~440 MB on first use; subsequent runs use the cached model.

```bash
python -m Benchmark_Pipeline.run_benchmark \
  --task-config ... --model-config ... --method-config ... \
  --enable-bert-score
```

### LLM-as-a-judge

Enable with `--enable-llm-judge`. Scores each prediction on a 5-point scale (0 / 0.25 / 0.5 / 0.75 / 1.0) using a judge model. Defaults to `gpt-4.1-nano` via the `OPENAI_API_KEY` environment variable.

```bash
python -m Benchmark_Pipeline.run_benchmark \
  --task-config ... --model-config ... --method-config ... \
  --enable-llm-judge \
  --judge-model gpt-4.1-nano \
  --judge-api-key <key>        # optional; falls back to OPENAI_API_KEY
  --judge-base-url <url>       # optional; defaults to OpenAI endpoint
  --judge-max-retries 3        # optional; default 3
  --judge-timeout 60           # optional; default 60 seconds
```

### Aggregation in `metrics.json`

When either flag is used, `metrics.json` includes a `summary` block with per-metric averages broken down along the MemEye Matrix axes:

```json
"summary": {
  "overall":  { "em": 0.10, "f1": 0.42, "bleu": 0.18, "bert": 0.61, "judge": 0.55 },
  "by_x":     { "X1": { "f1": 0.50 }, "X3": { "f1": 0.30 } },
  "by_y":     { "Y1": { "f1": 0.55 }, "Y2": { "f1": 0.38 } },
  "by_cell":  { "X1_Y1": { "f1": 0.60 }, "X3_Y2": { "f1": 0.28 } }
}
```

## Notes

- The current methods are intentionally lightweight baselines, intended to make cross-model and cross-method comparison easy.
- The current scoring is still lightweight and heuristic, not a final benchmark spec.
- `mode=mcq` still checks output validity rather than gold-option accuracy.
- Large models and long histories can be slow or memory-intensive.
- `semantic_rag_text_only` and `semantic_rag_multimodal` require sentence-transformers; `semantic_rag_multimodal` additionally requires a vLLM SigLIP2 server or local CLIP.
