# Comic Memory Benchmark Demo

This demo runs an end-to-end pipeline for comic memory testing:

1. Crop panels from a comic page using panel `bbox` from JSON.
2. Extract a ground-truth story summary from annotations.
3. Run Qwen2-VL visual QA on a sequence of panel images.

## Files

- `run_pipeline.py`: main pipeline.
- `run_demo.sh`: one-command runner script.
- `output/panels/`: generated panel crops and low-res images.

## Requirements

- Python 3.10+ (recommended in conda env `memorybench`)
- Dependencies:
  - `torch`
  - `transformers`
  - `qwen-vl-utils`
  - `pillow`

Install:

```bash
pip install torch transformers qwen-vl-utils pillow
```

## Local Model

Default local model path in this project:

`/common/users/mg1998/models/Qwen2-VL-2B-Instruct`

You can also use an HF model name, but local path is recommended for stability and speed.

## Quick Start

From project root:

```bash
bash run_demo.sh
```

## Useful Run Modes

- Full multi-image memory test (default):

```bash
bash run_demo.sh
```

- Single image debug:

```bash
bash run_demo.sh --single-image-test
```

- Only data processing and context extraction (skip VLM):

```bash
bash run_demo.sh --skip-vlm
```

- Tune generation/image budget:

```bash
MAX_NEW_TOKENS=160 MAX_IMAGE_PIXELS=102400 bash run_demo.sh
```

## Direct Python Command

```bash
python run_pipeline.py \
  --model-path /common/users/mg1998/models/Qwen2-VL-2B-Instruct \
  --max-new-tokens 120 \
  --max-image-pixels 129600
```

## Notes

- `--max-image-pixels` controls per-panel downsampling before VLM inference.
- Lower `max_image_pixels` helps avoid multi-image context overflow on small models.
- Output answer quality may vary; compare `[VLM Answer]` with `[Step 2] Ground Truth Story`.
