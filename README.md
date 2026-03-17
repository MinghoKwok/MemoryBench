# MemoryBench

This repository currently contains two separate multimodal memory experiment directories:

- `Benchmark_Pipeline`
- `ComicScene_Pipeline`

## What To Focus On

For current benchmark work, treat `Benchmark_Pipeline` as the active project.

If you are a collaborator joining this repo for the first time, start with:

```bash
cd Benchmark_Pipeline
```

and read:

- `Benchmark_Pipeline/README.md`

The `Benchmark_Pipeline` directory is the main benchmark scaffold in active use. It supports:

- modular task / model / method configs
- benchmark runs and matrix runs
- current memory methods including `full_context`, `clue_only`, `hybrid_rag`, and `m2a_lite`
- partner-added tasks that reuse the same dialogue/image format

## What To Ignore For Now

`ComicScene_Pipeline` is an older comic-memory demo and is not the current focus of benchmark development.

Unless you are explicitly working on the comic demo, you can ignore `ComicScene_Pipeline` for now.

## Recommended Entry Points

Single benchmark run:

```bash
python -m Benchmark_Pipeline.run_benchmark \
  --task-config Benchmark_Pipeline/config/tasks/brand_memory_test.yaml \
  --model-config Benchmark_Pipeline/config/models/gpt_4_1_nano.yaml \
  --method-config Benchmark_Pipeline/config/methods/full_context.yaml
```

Method comparison matrix:

```bash
python -m Benchmark_Pipeline.run_matrix \
  --task-config Benchmark_Pipeline/config/tasks/brand_memory_test.yaml \
  --model-config Benchmark_Pipeline/config/models/gpt_4_1_nano.yaml \
  --method-config Benchmark_Pipeline/config/methods/full_context.yaml \
  --method-config Benchmark_Pipeline/config/methods/clue_only.yaml \
  --method-config Benchmark_Pipeline/config/methods/hybrid_rag.yaml \
  --method-config Benchmark_Pipeline/config/methods/m2a_lite.yaml
```

If you use API-backed models, load local credentials first:

```bash
set -a
source .env.local
set +a
```
