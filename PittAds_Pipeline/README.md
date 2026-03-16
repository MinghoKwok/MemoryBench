# PittAds Pipeline

This pipeline runs a multimodal memory benchmark on top of a local Qwen2-VL model.

It reads:
- a dialogue JSON with multi-session conversations and human-annotated QAs
- the referenced images for each dialogue round
- a local vision-language model checkpoint

It outputs per-QA predictions plus a small summary JSON.

## Files

- `run_pittads.py`: main benchmark entry point
- `config/default.yaml`: default config
- `router/qwen_local.py`: local Qwen2-VL inference wrapper
- `data/dialog/Brand_Memory_Test.json`: example dialogue file
- `data/image/`: example image root
- `output/results_pittads.json`: example results file

## Requirements

Recommended:
- Python 3.10+
- A local Qwen2-VL or Qwen2.5-VL checkpoint

Python packages are listed in `requirements.txt`.

## Environment Setup

Create and activate a Python 3.10 environment with Conda or another environment manager, then install dependencies:

```bash
pip install -r requirements.txt
```

If you maintain a machine-specific setup, keep those details in `README.local.md`, which is intended to stay untracked.

## Quick Start

Activate your project environment first.

From repo root:

```bash
python PittAds_Pipeline/run_pittads.py --config PittAds_Pipeline/config/default.yaml
```

From inside `PittAds_Pipeline`:

```bash
python run_pittads.py --config config/default.yaml
```

## Common Options

Override the dialogue file:

```bash
python PittAds_Pipeline/run_pittads.py \
  --config PittAds_Pipeline/config/default.yaml \
  --dialog-json /path/to/data/dialog/example.json \
  --image-root /path/to/data/image
```

Switch evaluation mode:

```bash
python PittAds_Pipeline/run_pittads.py \
  --config PittAds_Pipeline/config/default.yaml \
  --mode both
```

Change model/output path:

```bash
python PittAds_Pipeline/run_pittads.py \
  --config PittAds_Pipeline/config/default.yaml \
  --model-path /path/to/Qwen2-VL \
  --output-json /tmp/pittads_results.json
```

## Config

Default config:

```yaml
model:
  provider: qwen_local
  model_path: /path/to/local/Qwen2-VL
  max_new_tokens: 128

dataset:
  dialog_json: data/dialog/Brand_Memory_Test.json
  image_root: data/image

eval:
  mode: open
  output_json: output/results_pittads.json
```

CLI flags override config values.

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

The output JSON contains:
- resolved `dialog_json`
- `model_path`
- evaluation `mode`
- `num_qas`
- aggregate `summary`
- per-question `results`

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

## Notes

- The current scoring is lightweight and heuristic, not a full benchmark reproduction.
- `mode=mcq` does not compare against a gold option label; it only checks whether the model returned a valid choice.
- Large models and long histories can be slow or memory-intensive.
