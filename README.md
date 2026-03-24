# MemoryBench

Code repository name: `MemoryBench`
Current benchmark name: `MemEye`

This repository currently contains several multimodal memory workstreams:

- `Benchmark_Pipeline`
- `Image_Generator`
- `overleaf_paper`

## Unified Persona & Life-Scenario Design

MemEye tasks are designed to form a coherent life scenario for a single persona, so that different tasks naturally represent different facets of one person's daily life. This ensures cross-task ecological validity: a memory system must handle the same user across social, domestic, professional, and leisure contexts.

### Persona: Hannah Brooks

- **Age:** 32
- **Occupation:** Freelance graphic designer / remote worker
- **Personality:** Visually perceptive, meticulous, occasionally indecisive, budget-conscious
- **Living situation:** Recently purchased a 970 sq ft two-bedroom apartment; undergoing full renovation

### Life-Scenario Task Mapping

| Life Facet | Task | What Hannah Is Doing |
|-----------|------|---------------------|
| **Home** | `home_renovation_interior_design` | Documenting and planning her apartment renovation with an AI assistant — choosing furniture, paint colors, layouts |
| **Work (Screen)** | `work_screen_memory` *(planned)* | Sharing computer screenshots during design work — IDE, documents, dashboards, emails — and discussing them with AI |
| **Social** | `chat_ui_memory_test` | Reviewing chat conversations with friends; the AI helps her recall who said what |
| **Organization** | `visual_case_archive_assistant` | Tracking objects and room states in her apartment/office as she reorganizes during renovation |
| **Leisure** | `comicscene_alley_oop_draft` | Reading comics in her downtime; the AI helps her remember plot and character details |
| **Work (Analysis)** | `brand_memory_test` | Analyzing brand advertisements for client projects as a graphic designer |

### Planned: Work Screen Memory

Inspired by ScreenshotVQA (MIRIX, Wang & Chen 2025), this task captures Hannah's computer work sessions. Key design principles:

- **Screenshots with memory value:** IDE with code review, document version diffs, data dashboards, email threads — not random browsing
- **Queries require image grounding:** File paths, variable names, chart values, UI element positions are only visible in screenshots
- **Cross-session state tracking:** The same document or project evolves across sessions, requiring the agent to track changes
- **MemEye coverage:** Fills X4 (OCR/micro-attribute) + Y2/Y3 (cross-session version tracking) coordinates that are under-represented in current tasks

### Design Principles

1. **Visual necessity:** Every query must require seeing the original image — text dialogue alone never leaks the answer
2. **Cross-task coherence:** Sessions across tasks share a consistent timeline and persona context
3. **Complementary memory demands:** Each task stresses a different combination of visual granularity (X-axis) and reasoning complexity (Y-axis)

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

Current `gpt-4.1-nano` comparison snapshot on five representative MemEye tasks:

| Task | Full Context | Hybrid RAG | M2A Lite | M2A Full |
| --- | --- | --- | --- | --- |
| `brand_memory_test` | EM `1.000`, F1 `1.000`, B1 `1.000`, B2 `0.829` | EM `1.000`, F1 `1.000`, B1 `1.000`, B2 `0.829` | EM `1.000`, F1 `1.000`, B1 `1.000`, B2 `0.829` | EM `0.500`, F1 `0.500`, B1 `0.500`, B2 `0.329` |
| `chat_ui_memory_test` | EM `0.600`, F1 `0.744`, B1 `0.709`, B2 `0.238` | EM `0.600`, F1 `0.745`, B1 `0.708`, B2 `0.237` | EM `0.600`, F1 `0.745`, B1 `0.708`, B2 `0.237` | EM `0.600`, F1 `0.754`, B1 `0.715`, B2 `0.240` |
| `comicscene_alley_oop_draft` | EM `0.933`, F1 `0.933`, B1 `0.933`, B2 `0.295` | EM `0.933`, F1 `0.933`, B1 `0.933`, B2 `0.295` | EM `0.933`, F1 `0.933`, B1 `0.933`, B2 `0.295` | EM `0.867`, F1 `0.867`, B1 `0.867`, B2 `0.274` |
| `home_renovation_interior_design` | EM `0.080`, F1 `0.428`, B1 `0.296`, B2 `0.195` | EM `0.167`, F1 `0.481`, B1 `0.360`, B2 `0.297` | EM `0.200`, F1 `0.523`, B1 `0.388`, B2 `0.301` | EM `0.080`, F1 `0.438`, B1 `0.288`, B2 `0.200` |
| `visual_case_archive_assistant` | EM `0.059`, F1 `0.251`, B1 `0.204`, B2 `0.102` | EM `0.059`, F1 `0.188`, B1 `0.176`, B2 `0.125` | EM `0.118`, F1 `0.289`, B1 `0.263`, B2 `0.132` | EM `0.059`, F1 `0.235`, B1 `0.192`, B2 `0.097` |

If you use API-backed models, load local credentials first:

```bash
set -a
source .env.local
set +a
```
