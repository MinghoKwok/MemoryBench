# MemoryBench Project Instructions

## Critical: Environment Setup

**ALWAYS use the `memorybench` conda environment.** The base conda environment has a PyTorch NCCL library conflict that breaks `sentence-transformers` and `transformers`.

```bash
conda activate memorybench
```

Or use `conda run`:
```bash
conda run -n memorybench python <script.py>
```

### Environment Verification

Correct interpreter: `/common/users/mg1998/miniforge3/envs/memorybench/bin/python`

Wrong interpreter (will fail): `/common/users/mg1998/miniforge3/bin/python`

To verify the environment works:
```bash
conda run -n memorybench python -c "
import torch; print('PyTorch OK:', torch.__version__)
from sentence_transformers import SentenceTransformer; print('sentence-transformers OK')
from transformers import CLIPModel; print('CLIP OK')
"
```

## Environment Variables

Load API keys before running benchmarks:

```bash
source .env.local
unset HUGGING_FACE_HUB_TOKEN  # May contain invalid characters
```

The `.env.local` file contains:
- `OPENAI_API_KEY` - Required for GPT models
- `GEMINI_API_KEY` - Required for Gemini models
- `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` - For HuggingFace dataset access
- `M2A_*` variables - For M2A embedding services

### CRITICAL: Never Hardcode API Keys

**NEVER write API key values directly into scripts, configs, or any file that could be committed to git.** Always reference them via environment variables loaded from `.env.local`.

**Incident (2026-04-17):** A Gemini API key was hardcoded in `scripts/run_phase2_gemini.sh`, committed and pushed to GitHub. GitHub's automated secret scanning detected it, reported it to Google, and the key was permanently revoked within hours. This caused all Gemini API calls in a 72-run overnight batch to fail, wasting an entire night of compute.

**Correct pattern:**
```bash
# In scripts:
source ../.env.local
export OPENAI_API_KEY="${OPENAI_API_KEY}"  # reference, never literal

# In Python:
api_key = os.environ["GEMINI_API_KEY"]  # never a string literal
```

**Wrong pattern:**
```bash
# NEVER DO THIS:
export GEMINI_API_KEY="AIzaSy..."  # will be detected and revoked
```

## M2A Method Configuration

| Method | Dense Embeddings | Image Retrieval | Notes |
|--------|-----------------|-----------------|-------|
| `full_context_multimodal` | N/A | Yes | Full history with images |
| `full_context_text_only` | N/A | No | Full history, captions as text; needs `image_caption` in dataset |
| `semantic_rag_multimodal` | Yes (all-MiniLM-L6-v2) | Yes (SigLIP2) | Requires memorybench env |
| `semantic_rag_text_only` | Yes (all-MiniLM-L6-v2) | No | Requires memorybench env |
| `m2a` | Yes (all-MiniLM-L6-v2) | Yes (SigLIP2) | Agentic; requires memorybench env |
| `mma` | Yes (all-MiniLM-L6-v2) | Yes (SigLIP v1 so400m) | Confidence-aware agentic; requires memorybench env |

**Important:** If you see these warnings, you're in the wrong environment:
```
WARNING - Dense embeddings will be disabled
WARNING - Dense embeddings requested but unavailable. Falling back to TF-IDF based retrieval.
WARNING - Image retrieval requested but unavailable. Disabling cross-modal search.
```

## Common Commands

### Run benchmark
```bash
cd Benchmark_Pipeline
conda run -n memorybench python run_benchmark.py \
  --task <task_name> \
  --model <model_name> \
  --method <method_name>
```

Example:
```bash
source ../.env.local
unset HUGGING_FACE_HUB_TOKEN
conda run -n memorybench python run_benchmark.py \
  --task visual_case_archive_assistant \
  --model gpt-4.1-nano \
  --method m2a_full
```

### Using config files
```bash
python run_benchmark.py \
  --task-config config/tasks_external/<task>.yaml \
  --model-config config/models/<model>.yaml \
  --method-config config/methods/<method>.yaml
```

### Sync data to HuggingFace
```bash
export HF_TOKEN=<token from .env.local>
python sync_hf_data.py push --push
```

## Key Directories

- `Benchmark_Pipeline/` - Main benchmark code
- `Benchmark_Pipeline/benchmark/` - Core benchmark modules
  - `embeddings.py` - Dense embedding implementations
  - `retrieval.py` - M2A retrieval logic
  - `methods.py` - History construction methods
- `Benchmark_Pipeline/config/` - Configuration files
- `Benchmark_Pipeline/runs/` - Experiment outputs
- `Benchmark_Pipeline/output/` - Result JSON files
- `~/.cache/memeye_hf/MemEye/` - HuggingFace dataset cache
- `overleaf_paper/` - LaTeX paper source

## Common Issues

### PyTorch NCCL Error
```
ImportError: libtorch_cuda.so: undefined symbol: ncclGroupSimulateEnd
```
**Fix:** Use `conda activate memorybench` instead of base environment.

### ASCII Encoding Error
```
'ascii' codec can't encode characters in position 7-8
```
**Fix:** `unset HUGGING_FACE_HUB_TOKEN`

### Missing API Key
```
RuntimeError: Missing API key in environment variable: OPENAI_API_KEY
```
**Fix:** `source .env.local`

## Data Format Pitfalls

### Image field MUST be `input_image`, not `images`

`dataset.py` line 62 reads `d.get("input_image", [])` — any other field name (e.g. `images`) will be silently ignored. The benchmark will run without error but with **zero images**, producing misleadingly low scores.

When adding or converting a new dataset, always verify images are loaded:
```python
ds = MemoryBenchmarkDataset(dialog_json_path, image_root)
found = sum(1 for r in ds.rounds.values() for _ in r.get('images', []))
print(f'Images loaded: {found}')  # must be > 0 for visual tasks
```

### Image paths must resolve relative to dialog JSON

`input_image` paths like `"../image/Task_Name/IMG.png"` are resolved relative to the dialog JSON file location. If using absolute `image_root` in the task config, paths like `"Task_Name/IMG.png"` (without `../image/`) may also work via fallback resolution. Always verify with the snippet above.

### Qwen2.5-VL local: MUST use single GPU

`device_map="auto"` with multiple GPUs causes silent degenerate output (garbage tokens, ~100s per question). Always set `CUDA_VISIBLE_DEVICES=<single_gpu>` when running Qwen local. See memory notes for details.

### MultimodalEmbedder text truncation

SigLIP2's text encoder has a 64-token context limit. `MultimodalEmbedder.embed_text()` automatically truncates long text to fit (character heuristic, word-boundary aware). `LocalCLIPEmbedder` already handles this via the HuggingFace `truncation=True` flag. `TextEmbedder` (sentence-transformers) also truncates internally. No action needed from callers.

### M2A with weaker models (< 70B)

M2A's ReAct loop requires strict tool calling compliance. Models like Qwen2.5-VL-7B hallucinate tool names (`CREATE` instead of `add_memory`) and parameter names (`content` instead of `text`). The `memory_manager.py` has fuzzy matching for both, but memory quality will still be low. For reliable M2A results, use GPT-4 class models or 70B+.

## Package Versions (memorybench env)

| Package | Version |
|---------|---------|
| Python | 3.10.x |
| PyTorch | 2.11.0+cu130 |
| sentence-transformers | 5.3.0 |
| transformers | 5.2.0 |

Note: CUDA may show as unavailable due to driver version, but CPU inference works fine for embeddings.
