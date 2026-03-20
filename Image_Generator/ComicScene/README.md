# ComicScene Generator

This directory is the active home for MemEye comic-task generation work.

The current focus is not story-summary evaluation. The current focus is turning comic pages into MemEye-style benchmark items with:

1. visible answers only
2. controlled answer formats
3. MemEye binocular `point` labels such as `[['X2'], ['Y2']]`
4. intentionally retained hard items when the difficulty reflects real visual-memory failure rather than annotation noise

For the MemEye-oriented assessment and candidate task directions, see:

- `Image_Generator/ComicScene/MemEye_ComicScene_Assessment.md`

Cross-generator task and image design rules are documented in:

- `Image_Generator/Generation_Guidelines.md`

## Files

- `drafts/`: current MemEye task drafts such as `Alley_Oop_MemEye_Draft.json`
- `Data/`: source comic pages and metadata
- `run_pipeline.py`: legacy demo pipeline
- `run_demo.sh`: legacy one-command demo runner
- `output/panels/`: generated panel crops and low-res images from the legacy demo

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

## Legacy Demo Model

Default local model path in this project:

`/common/users/mg1998/models/Qwen2-VL-2B-Instruct`

You can also use an HF model name, but local path is recommended for stability and speed.

## Current Workflow

For current MemEye work in this directory:

1. start from source pages under `Data/`
2. design questions using `Image_Generator/Generation_Guidelines.md`
3. assign MemEye `point` labels using `Benchmark_Pipeline/MemEye_Annotation_Guide.md`
4. write draft benchmark JSON under `drafts/`
5. sync finalized dialog/image assets into the HF dataset repo `data/` tree
6. benchmark the synced task through `Benchmark_Pipeline`

The current Alley Oop draft already follows this pattern:

- `Image_Generator/ComicScene/drafts/Alley_Oop_MemEye_Draft.json`

## Legacy Demo Quick Start

From project root:

```bash
bash run_demo.sh
```

## Legacy Demo Run Modes

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

## Legacy Demo Direct Python Command

```bash
python run_pipeline.py \
  --model-path /common/users/mg1998/models/Qwen2-VL-2B-Instruct \
  --max-new-tokens 120 \
  --max-image-pixels 129600
```

## Notes

- The demo scripts in this directory are legacy helpers, not the MemEye benchmark interface.
- `--max-image-pixels` controls per-panel downsampling before VLM inference.
- Lower `max_image_pixels` helps avoid multi-image context overflow on small models.
- Do not use hidden metadata or free-form story summaries as MemEye ground truth for new tasks.
