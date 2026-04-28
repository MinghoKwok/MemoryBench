# MemoryBench

Code repository name: `MemoryBench`
Current benchmark name: `MemEye`

This repository contains:

- `Benchmark_Pipeline` — Main benchmark scaffold (tasks, methods, evaluation)
- `Image_Generator` — Image/data generation tools
- `overleaf_paper` — LaTeX paper source

## Quick Start

```bash
# 1. Activate environment
conda activate memorybench

# 2. Load API keys
source .env.local
unset HUGGING_FACE_HUB_TOKEN

# 3. Run a benchmark
cd Benchmark_Pipeline
bash run_all.sh --method full_context_multimodal --model gemini_2_5_flash_lite
```

## Running Experiments

### `run_all.sh` — Run a method × model across all datasets

```bash
# Full sweep: one method × one model × all 8 datasets
bash run_all.sh --method full_context_multimodal --model gemini_2_5_flash_lite

# Specific datasets only
bash run_all.sh --method m2a_gemini --model gemini_2_5_flash_lite \
  --datasets "card_playlog_test,brand_memory_test"

# Quick smoke test (5 questions per dataset)
bash run_all.sh --method mma --model gpt_4_1_nano --max-questions 5

# List all available methods, models, and datasets
bash run_all.sh --list
```

### Single benchmark run

```bash
cd Benchmark_Pipeline
python run_benchmark.py \
  --task-config config/tasks_external/brand_memory_test.yaml \
  --model-config config/models/gemini_2_5_flash_lite.yaml \
  --method-config config/methods/full_context_multimodal.yaml
```

### Available Models

| Model | Config | API Key Env |
|-------|--------|-------------|
| Gemini 2.5 Flash Lite | `gemini_2_5_flash_lite` | `GEMINI_API_KEY` |
| GPT-4.1 Nano | `gpt_4_1_nano` | `OPENAI_API_KEY` |
| GPT-4o Mini | `gpt_4o_mini` | `OPENAI_API_KEY` |
| Qwen3-VL-8B (OpenRouter) | `qwen3_vl_8b_openrouter` | `OPENROUTER_API_KEY` |

### Available Methods

| Method | Config | Type | Notes |
|--------|--------|------|-------|
| Full Context (V) | `full_context_multimodal` | Baseline | Full history with images |
| Full Context (T) | `full_context_text_only` | Baseline | Full history, captions replace images |
| Semantic RAG (V) | `semantic_rag_multimodal` | RAG | Dense retrieval + images |
| Semantic RAG (T) | `semantic_rag_text_only` | RAG | Dense retrieval, text-only |
| A-MEM | `a_mem` / `a_mem_gemini` | Agentic | Autonomous memory agent |
| Gen. Agents | `gen_agents` / `gen_agents_gemini` | Agentic | Generative Agents reflection |
| Reflexion | `reflexion` / `reflexion_gemini` | Agentic | Self-reflection loop |
| SimpleMem | `simplemem` / `simplemem_gemini` | Summarize | Omni-SimpleMem (text) |
| SimpleMem (V) | `simplemem_multimodal` / `simplemem_multimodal_gemini` | Summarize | Omni-SimpleMem + images |
| MemoryOS | `memoryos` / `memoryos_gemini` | Agentic | MemoryOS baseline |
| EverMemOS | `evermemos` / `evermemos_openrouter` | Agentic | EverMemOS memory system |
| MemGPT | `memgpt` | Agentic | MemGPT/Letta baseline |
| M2A | `m2a` / `m2a_gemini` | Agentic | Multimodal agentic; uses SigLIP2 |
| MMA | `mma` / `mma_gemini` | Agentic | Confidence-aware; uses SigLIP v1 |

**`*_gemini` variants** automatically redirect the OpenAI SDK to Gemini's OpenAI-compatible endpoint. Use these on machines without OpenAI API access.

### Available Datasets (8 tasks, 371 QAs)

| Dataset | Config | QAs | Domain |
|---------|--------|-----|--------|
| Brand Memory Test | `brand_memory_test` | 29 | Ad campaign comparison |
| Card Playlog Test | `card_playlog_test` | 48 | Card game state tracking |
| Cartoon Entertainment Companion | `cartoon_entertainment_companion` | 76 | Animation + comic visual narrative |
| Home Renovation Interior Design | `home_renovation_interior_design` | 52 | Renovation planning |
| Multi-Scene VCAA | `multi_scene_visual_case_archive_assistant` | 50 | Multi-room object tracking |
| Outdoor Navigation | `outdoor_navigation_route_memory_assistant` | 28 | Route memory |
| Personal Health Dashboard | `personal_health_dashboard_assistant` | 51 | Health metric tracking |
| Social Chat Memory Test | `social_chat_memory_test` | 37 | Chat screenshot memory |

### Output

Results are saved to `Benchmark_Pipeline/output/<dataset>/results_<dataset>__<model>__<method>.json`. Each result file contains:

- `summary.overall` — aggregate EM / F1 / BLEU
- `summary.by_x` — accuracy broken down by X-axis level
- `summary.by_y` — accuracy broken down by Y-axis level
- `summary.by_cell` — accuracy per (X, Y) matrix cell

## MemEye Taxonomy

MemEye evaluates multimodal agent memory along two orthogonal axes:

**X-axis (Visual Granularity):**
- `X1` Global Scene — scene-level understanding
- `X2` Region Scene — localized scene regions and grouped visual context
- `X3` Instance Identity — specific object/person instance discrimination
- `X4` Fine-Grained Attributes — color, OCR, texture, and small visual details

**Y-axis (Reasoning Complexity):**
- `Y1` Atomic Retrieval — single-session fact lookup
- `Y2` Composite Retrieval — cross-session monotonic integration
- `Y3` State Update Reasoning — temporal logic required, later evidence overrides earlier

Each QA is assigned a single `(X_i, Y_j)` coordinate following the **Highest-Bottleneck Rule**: the label reflects the highest level whose absence would prevent a correct answer.

## Data

Dataset files are stored on HuggingFace (`MemEyeBench/MemEye`) and synced locally:

```bash
cd Benchmark_Pipeline

# Pull from HuggingFace
export HF_TOKEN=<token>
python sync_hf_data.py pull

# Push local changes
python sync_hf_data.py push --push
```

## Environment Setup

```bash
conda activate memorybench
```

See `CLAUDE.md` for detailed environment setup, common issues, and package versions.
