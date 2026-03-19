# Synthetic Chat UI Generator

This folder contains a minimal synthetic chat UI generator for MemEye-style identity binding tests.

## What It Produces

- procedurally rendered multi-screenshot chat episodes
- episode-level JSON metadata under `Image_Generator/chatUI/outputs`
- benchmark-ready dialog JSON at `Benchmark_Pipeline/data/dialog/Chat_UI_Memory_Test.json`
- rendered screenshots at `Benchmark_Pipeline/data/image/Chat_UI_Memory_Test`

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
- add a task config so this dataset can be benchmarked directly with `run_benchmark.py`
