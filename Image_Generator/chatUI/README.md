# Synthetic Chat UI Generator

This folder contains a minimal synthetic chat UI generator for MemEye-style identity binding tests.

## What It Produces

- procedurally rendered multi-screenshot chat episodes
- episode-level JSON metadata under `Image_Generator/chatUI/outputs`
- benchmark-ready dialog JSON that can be synced into the HF dataset repo `data/dialog/`
- rendered screenshots that can be synced into the HF dataset repo `data/image/`
- optional local working-copy outputs under `Benchmark_Pipeline/data/...` when you want to benchmark immediately on this machine

## Current MVP Scope

- 4 speakers with persistent avatars
- 3 screenshots per episode
- left/right bubble placement is tied to speaker identity
- generated queries for:
  - speaker binding
  - cross-screenshot consistency
  - visual structure
  - hallucination resistance

The generator is intentionally simple. It is designed to test whether a model keeps avatar-message bindings over multiple turns, not just whether it can read chat text.

The generated benchmark JSON uses the MemEye binocular `point` format documented in:

- `Benchmark_Pipeline/MemEye_Annotation_Guide.md`

Cross-generator task and image design rules are documented in:

- `Image_Generator/Generation_Guidelines.md`

When integrating generated data into the benchmark, follow the HF-first workflow documented in:

- `Benchmark_Pipeline/README.md`

## Run

```bash
python Image_Generator/chatUI/generate_chat_ui_dataset.py --num-episodes 8
```

## Dependency

```bash
pip install pillow
```

## Next Upgrades

- add harder avatar similarity cases
- add explicit negative topics per episode
- vary theme/layout while keeping identity consistent
- generate train/dev/test splits
- keep syncing the dataset through the HF dataset repo rather than treating local `Benchmark_Pipeline/data` as canonical
