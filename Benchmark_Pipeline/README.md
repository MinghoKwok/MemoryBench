# Benchmark Pipeline

This directory is a small benchmark scaffold for multimodal memory experiments, not just a single inference script.

The benchmark separates:
- task data and evaluation
- model routing
- memory method selection
- run artifact storage

The benchmark supports local and API-backed model routers, plus multiple memory methods:
- `full_context`
- `hybrid_rag`
- `m2a_lite`
- `m2a_full`

Current representative tasks in active use include:
- `brand_memory_test`
- `chat_ui_memory_test`
- `comicscene_alley_oop_draft`
- `home_renovation_interior_design`

The main open-answer metrics now emphasized in this repo are:
- `EM`
- `F1`
- `BLEU-1`
- `BLEU-2`

For MemEye task design and `point` annotation rules, read:

- `Benchmark_Pipeline/MemEye_Annotation_Guide.md`
- `Benchmark_Pipeline/benchmark/prompt/README.md`

## Layout

- `run_benchmark.py`: modular benchmark entrypoint
- `run_matrix.py`: model x method matrix runner
- `run_legacy_benchmark.py`: generic legacy-compatible single-config entrypoint
- `run_pittads.py`: legacy compatibility shim for older commands
- `benchmark/`: dataset loading, method selection, evaluation, run orchestration
- `benchmark/prompt/`: MemEye benchmark prompt templates and task-family prompt building blocks
- `router/`: model router implementations
- `config/tasks/`: task configs
- `config/tasks_external/`: generated task configs that point to external or HF-synced datasets
- `config/models/`: model configs
- `config/methods/`: method configs
- `data/`: local synced working copy of benchmark dialogue/images from the HF dataset repo
- `runs/`: per-run artifacts
- `output/`: optional legacy-style summary JSON outputs

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
  --task-config Benchmark_Pipeline/config/tasks_external/brand_memory_test.yaml \
  --model-config Benchmark_Pipeline/config/models/qwen_local_default.yaml \
  --method-config Benchmark_Pipeline/config/methods/full_context.yaml
```

Script execution still works if you prefer it:

```bash
python Benchmark_Pipeline/run_benchmark.py \
  --task-config Benchmark_Pipeline/config/tasks_external/brand_memory_test.yaml \
  --model-config Benchmark_Pipeline/config/models/qwen_local_default.yaml \
  --method-config Benchmark_Pipeline/config/methods/full_context.yaml
```

Representative current task configs:

- `Benchmark_Pipeline/config/tasks_external/brand_memory_test.yaml`
- `Benchmark_Pipeline/config/tasks_external/chat_ui_memory_test.yaml`
- `Benchmark_Pipeline/config/tasks_external/comicscene_alley_oop_draft.yaml`
- `Benchmark_Pipeline/config/tasks_external/home_renovation_interior_design.yaml`

## Common Experiments

Run the lightweight retrieval baseline:

```bash
python -m Benchmark_Pipeline.run_benchmark \
  --task-config Benchmark_Pipeline/config/tasks_external/home_renovation_interior_design.yaml \
  --model-config Benchmark_Pipeline/config/models/gpt_4_1_nano.yaml \
  --method-config Benchmark_Pipeline/config/methods/hybrid_rag.yaml
```

Run the lightweight M2A-style baseline:

```bash
python -m Benchmark_Pipeline.run_benchmark \
  --task-config Benchmark_Pipeline/config/tasks_external/comicscene_alley_oop_draft.yaml \
  --model-config Benchmark_Pipeline/config/models/gpt_4_1_nano.yaml \
  --method-config Benchmark_Pipeline/config/methods/m2a_lite.yaml
```

Limit to a quick smoke test:

```bash
python -m Benchmark_Pipeline.run_benchmark \
  --task-config Benchmark_Pipeline/config/tasks_external/chat_ui_memory_test.yaml \
  --model-config Benchmark_Pipeline/config/models/qwen_local_default.yaml \
  --method-config Benchmark_Pipeline/config/methods/full_context.yaml \
  --max-questions 1
```

Override evaluation mode:

```bash
python -m Benchmark_Pipeline.run_benchmark \
  --task-config Benchmark_Pipeline/config/tasks_external/brand_memory_test.yaml \
  --model-config Benchmark_Pipeline/config/models/qwen_local_default.yaml \
  --method-config Benchmark_Pipeline/config/methods/full_context.yaml \
  --mode both
```

Run a model x method matrix:

```bash
python -m Benchmark_Pipeline.run_matrix \
  --task-config Benchmark_Pipeline/config/tasks_external/chat_ui_memory_test.yaml \
  --model-config Benchmark_Pipeline/config/models/gpt_4_1_nano.yaml \
  --method-config Benchmark_Pipeline/config/methods/full_context.yaml \
  --method-config Benchmark_Pipeline/config/methods/hybrid_rag.yaml \
  --method-config Benchmark_Pipeline/config/methods/m2a_lite.yaml \
  --method-config Benchmark_Pipeline/config/methods/m2a_full.yaml
```

Representative `gpt-4.1-nano` results on the four active MemEye tasks:

| Task | Full Context | Hybrid RAG | M2A Lite |
| --- | --- | --- | --- |
| `brand_memory_test` | EM `1.000`, F1 `1.000`, BLEU-1 `1.000`, BLEU-2 `0.829` | EM `1.000`, F1 `1.000`, BLEU-1 `1.000`, BLEU-2 `0.829` | EM `1.000`, F1 `1.000`, BLEU-1 `1.000`, BLEU-2 `0.829` |
| `chat_ui_memory_test` | EM `0.600`, F1 `0.744`, BLEU-1 `0.709`, BLEU-2 `0.238` | EM `0.600`, F1 `0.745`, BLEU-1 `0.708`, BLEU-2 `0.237` | EM `0.600`, F1 `0.745`, BLEU-1 `0.708`, BLEU-2 `0.237` |
| `comicscene_alley_oop_draft` | EM `0.933`, F1 `0.933`, BLEU-1 `0.933`, BLEU-2 `0.295` | EM `0.933`, F1 `0.933`, BLEU-1 `0.933`, BLEU-2 `0.295` | EM `0.933`, F1 `0.933`, BLEU-1 `0.933`, BLEU-2 `0.295` |
| `home_renovation_interior_design` | EM `0.200`, F1 `0.520`, BLEU-1 `0.386`, BLEU-2 `0.298` | EM `0.167`, F1 `0.481`, BLEU-1 `0.360`, BLEU-2 `0.297` | EM `0.200`, F1 `0.523`, BLEU-1 `0.388`, BLEU-2 `0.301` |

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
name: full_context
```

`hybrid_rag` accepts retrieval-specific parameters:

```yaml
name: hybrid_rag
top_k: 2
neighbor_window: 1
lexical_weight: 0.35
semantic_weight: 0.65
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

`m2a_full` uses a fuller M2A-style retrieval pipeline:

```yaml
name: m2a_full
semantic_top_k: 8
raw_top_k: 6
neighbor_window: 1
max_iterations: 2
rrf_k: 60
min_round_words_for_memory: 3
semantic_lexical_weight: 0.35
semantic_dense_weight: 0.65
raw_lexical_weight: 0.35
raw_dense_weight: 0.65
```

The default `config/default.yaml` composes the same pieces in one file. Legacy single-file configs still exist for compatibility, but new work should prefer task/model/method configs plus `config/tasks_external/`.

## Memory Methods

- `full_context`: feeds all rounds from the target session set.
- `hybrid_rag`: retrieves round-level evidence with a lightweight lexical + TF-IDF scoring pass, then expands local neighbors.
- `m2a_lite`: builds a simple two-layer memory over session summaries and round summaries, retrieves semantic memory first, then resolves back to raw rounds.
- `m2a_full`: builds multi-granularity semantic memories (turn/round/session), fuses dense and lexical ranks with RRF, then iteratively refines semantic-to-raw evidence retrieval.

When `hybrid_rag` retrieves no evidence, it falls back internally to the QA's annotated clue rounds. This fallback is an implementation detail, not a separately exposed benchmark method.

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

- `hybrid_rag`: direct retrieval over raw rounds
- `m2a_lite`: lightweight two-stage memory retrieval inspired by M2A
- `m2a_full`: richer M2A-style iterative semantic-to-raw retrieval baseline (benchmark-oriented)

It should not be described as "the full M2A method" in reports or comparisons.

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

For new MemEye datasets, the recommended workflow is:

1. Add or update the dataset under the HF dataset repo `data/` tree.
2. Pull that dataset into your local synced working copy with `sync_hf_data.py pull`, or work directly from an external local clone.
3. Generate a task config that points to the dataset JSON and image root.
4. Validate with `run_benchmark`.

For a task that already lives under local `Benchmark_Pipeline/data/`, a minimal task config still looks like:

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

Use this only for a synced local working copy. The long-term source of truth should remain the HF dataset repo.

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
  --method-config Benchmark_Pipeline/config/methods/full_context.yaml
```

Current generated external task configs commonly used in this repo are:
- `Benchmark_Pipeline/config/tasks_external/brand_memory_test.yaml`
- `Benchmark_Pipeline/config/tasks_external/chat_ui_memory_test.yaml`
- `Benchmark_Pipeline/config/tasks_external/comicscene_alley_oop_draft.yaml`
- `Benchmark_Pipeline/config/tasks_external/home_renovation_interior_design.yaml`

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

To benchmark against a synced local task config, run for example:

```bash
python -m Benchmark_Pipeline.run_benchmark \
  --task-config Benchmark_Pipeline/config/tasks_external/chat_ui_memory_test.yaml \
  --model-config Benchmark_Pipeline/config/models/gpt_4_1_nano.yaml \
  --method-config Benchmark_Pipeline/config/methods/full_context.yaml
```

This lets each partner add a task instance without changing framework code while keeping benchmark data canonical in the HF dataset repo.

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
python -m Benchmark_Pipeline.run_legacy_benchmark ...
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
- exact-match / contains-GT flags
- `EM`
- `F1`
- `BLEU-1`
- `BLEU-2`
- latency

For `mcq` mode, each result includes:
- raw model output
- extracted choice
- valid-choice flag
- latency

Each row also records benchmark metadata such as:
- `method_name`
- `history_turns`
- `source_sessions`
- `clue_rounds`

## Notes

- The code repository is still named `MemoryBench`, but the current benchmark identity and taxonomy are `MemEye`.
- The current methods are intentionally lightweight baselines, intended to make cross-model and cross-method comparison easy.
- The current scoring emphasis for open-answer tasks is `EM/F1/BLEU-1/BLEU-2`.
- `mode=mcq` still checks output validity rather than gold-option accuracy.
- Large models and long histories can be slow or memory-intensive.
- `hybrid_rag`, `m2a_lite`, and `m2a_full` currently use local token-based retrieval only; they do not require an external embedding service.
