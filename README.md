# MemoryBench

Code repository name: `MemoryBench`
Current benchmark name: `MemEye`

This repository currently contains several multimodal memory workstreams:

- `Benchmark_Pipeline`
- `Image_Generator`
- `overleaf_paper`

## What To Focus On

For current MemEye benchmark work, treat `Benchmark_Pipeline` as the active benchmark scaffold.

If you are a collaborator joining this repo for the first time, start with:

```bash
cd Benchmark_Pipeline
```

and read:

- `Benchmark_Pipeline/README.md`
- `Benchmark_Pipeline/MemEye_Annotation_Guide.md`
- `Image_Generator/Generation_Guidelines.md`

The `Benchmark_Pipeline` directory is the main benchmark scaffold in active use. It supports:

- modular task / model / method configs
- benchmark runs and matrix runs
- current memory methods including `full_context`, `hybrid_rag`, `m2a_lite`, and `m2a_full`
- partner-added tasks that reuse the same dialogue/image format
- task data that is canonically stored in the Hugging Face dataset repo and synced locally when needed
- representative active tasks including `brand_memory_test`, `chat_ui_memory_test`, `comicscene_alley_oop_draft`, and `home_renovation_interior_design`
- open-answer evaluation centered on `EM/F1/BLEU-1/BLEU-2`

For current generator work, the active entry points are:

- `Image_Generator/chatUI`
- `Image_Generator/ComicScene`

## What To Ignore For Now

`ComicScene_Pipeline` is an older comic-memory demo and is not the current focus of benchmark development.

Unless you are explicitly working on legacy comic demo code, you can ignore `ComicScene_Pipeline` for now.
Current comic-task generation work lives under `Image_Generator/ComicScene`.

## Recommended Entry Points

Current representative benchmark tasks:

- `Benchmark_Pipeline/config/tasks_external/brand_memory_test.yaml`
- `Benchmark_Pipeline/config/tasks_external/chat_ui_memory_test.yaml`
- `Benchmark_Pipeline/config/tasks_external/comicscene_alley_oop_draft.yaml`
- `Benchmark_Pipeline/config/tasks_external/home_renovation_interior_design.yaml`

Single benchmark run:

```bash
python -m Benchmark_Pipeline.run_benchmark \
  --task-config Benchmark_Pipeline/config/tasks_external/brand_memory_test.yaml \
  --model-config Benchmark_Pipeline/config/models/gpt_4_1_nano.yaml \
  --method-config Benchmark_Pipeline/config/methods/full_context.yaml
```

Method comparison matrix:

```bash
python -m Benchmark_Pipeline.run_matrix \
  --task-config Benchmark_Pipeline/config/tasks_external/chat_ui_memory_test.yaml \
  --model-config Benchmark_Pipeline/config/models/gpt_4_1_nano.yaml \
  --method-config Benchmark_Pipeline/config/methods/full_context.yaml \
  --method-config Benchmark_Pipeline/config/methods/hybrid_rag.yaml \
  --method-config Benchmark_Pipeline/config/methods/m2a_lite.yaml \
  --method-config Benchmark_Pipeline/config/methods/m2a_full.yaml
```

Current `gpt-4.1-nano` comparison snapshot on the four representative MemEye tasks:

| Task | Full Context | Hybrid RAG | M2A Lite |
| --- | --- | --- | --- |
| `brand_memory_test` | EM `1.000`, F1 `1.000`, B1 `1.000`, B2 `0.829` | EM `1.000`, F1 `1.000`, B1 `1.000`, B2 `0.829` | EM `1.000`, F1 `1.000`, B1 `1.000`, B2 `0.829` |
| `chat_ui_memory_test` | EM `0.600`, F1 `0.744`, B1 `0.709`, B2 `0.238` | EM `0.600`, F1 `0.745`, B1 `0.708`, B2 `0.237` | EM `0.600`, F1 `0.745`, B1 `0.708`, B2 `0.237` |
| `comicscene_alley_oop_draft` | EM `0.933`, F1 `0.933`, B1 `0.933`, B2 `0.295` | EM `0.933`, F1 `0.933`, B1 `0.933`, B2 `0.295` | EM `0.933`, F1 `0.933`, B1 `0.933`, B2 `0.295` |
| `home_renovation_interior_design` | EM `0.200`, F1 `0.520`, B1 `0.386`, B2 `0.298` | EM `0.167`, F1 `0.481`, B1 `0.360`, B2 `0.297` | EM `0.200`, F1 `0.523`, B1 `0.388`, B2 `0.301` |

If you use API-backed models, load local credentials first:

```bash
set -a
source .env.local
set +a
```
