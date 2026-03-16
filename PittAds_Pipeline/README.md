# PittAds Benchmark

This directory is now a small benchmark scaffold for multimodal memory experiments, not just a single inference script.

The benchmark separates:
- task data and evaluation
- model routing
- memory method selection
- run artifact storage

The current task is PittAds, with a local Qwen-VL router and two baseline methods: `full_context` and `clue_only`.

## Layout

- `run_benchmark.py`: modular benchmark entrypoint
- `run_pittads.py`: legacy-compatible entrypoint
- `benchmark/`: dataset loading, method selection, evaluation, run orchestration
- `router/`: model router implementations
- `config/tasks/`: task configs
- `config/models/`: model configs
- `config/methods/`: method configs
- `data/`: example dialogue and images
- `runs/`: per-run artifacts
- `output/results_pittads.json`: optional legacy output file

## Requirements

Recommended:
- Python 3.10+
- A local vision-language checkpoint

Python packages are listed in `requirements.txt`.

## Environment Setup

Create and activate a Python 3.10 environment with Conda or another environment manager, then install dependencies:

```bash
pip install -r requirements.txt
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
python PittAds_Pipeline/run_benchmark.py \
  --task-config PittAds_Pipeline/config/tasks/pittads.yaml \
  --model-config PittAds_Pipeline/config/models/qwen_local_default.yaml \
  --method-config PittAds_Pipeline/config/methods/full_context.yaml
```

Or from inside `PittAds_Pipeline`:

```bash
python run_benchmark.py \
  --task-config config/tasks/pittads.yaml \
  --model-config config/models/qwen_local_default.yaml \
  --method-config config/methods/full_context.yaml
```

The legacy command still works:

```bash
python PittAds_Pipeline/run_pittads.py --config PittAds_Pipeline/config/default.yaml
```

## Common Experiments

Switch method:

```bash
python PittAds_Pipeline/run_benchmark.py \
  --task-config PittAds_Pipeline/config/tasks/pittads.yaml \
  --model-config PittAds_Pipeline/config/models/qwen_local_default.yaml \
  --method-config PittAds_Pipeline/config/methods/clue_only.yaml
```

Limit to a quick smoke test:

```bash
python PittAds_Pipeline/run_benchmark.py \
  --task-config PittAds_Pipeline/config/tasks/pittads.yaml \
  --model-config PittAds_Pipeline/config/models/qwen_local_default.yaml \
  --method-config PittAds_Pipeline/config/methods/full_context.yaml \
  --max-questions 1
```

Override evaluation mode:

```bash
python PittAds_Pipeline/run_benchmark.py \
  --task-config PittAds_Pipeline/config/tasks/pittads.yaml \
  --model-config PittAds_Pipeline/config/models/qwen_local_default.yaml \
  --method-config PittAds_Pipeline/config/methods/full_context.yaml \
  --mode both
```

## Config Structure

Example task config:

```yaml
name: pittads

dataset:
  dialog_json: data/dialog/Brand_Memory_Test.json
  image_root: data/image

eval:
  mode: open
  output_json: output/results_pittads.json
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

The legacy `config/default.yaml` still works and composes the same pieces in one file.

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
      "point": "FR",
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

- `input_image` may use relative paths such as `../image/...`, `./image/...`, `image/...`, or `data/image/...`
- absolute image paths are supported
- `file://...` image paths are supported
- `image_root` is used as an explicit override when image paths need rebasing
- config and dataset paths resolve correctly whether you run from repo root or from inside `PittAds_Pipeline`

## Output

Each benchmark run now writes a dedicated directory under `runs/`, typically containing:
- `config.json`: resolved run config
- `metrics.json`: run metadata and aggregate metrics
- `predictions.jsonl`: one row per prediction

If `eval.output_json` is set, the benchmark also writes a legacy summary JSON for backward compatibility.

For `open` mode, each result includes:
- predicted answer
- ground truth
- exact-match / contains-GT flags
- keyword hits
- soft score
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

- The current methods are simple baselines, intended to make cross-model and cross-method comparison easy.
- The current scoring is still lightweight and heuristic, not a final benchmark spec.
- `mode=mcq` still checks output validity rather than gold-option accuracy.
- Large models and long histories can be slow or memory-intensive.
