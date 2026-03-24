# Environment Setup for M2A Full (Dense Embeddings)

## Required Conda Environment

The `m2a_full` method requires `sentence-transformers` and `transformers` with working PyTorch to enable dense embeddings and CLIP image retrieval. The correct environment is:

```bash
conda activate memorybench
```

### Environment Details

| Package | Version |
|---------|---------|
| Python | 3.10.x |
| PyTorch | 2.11.0+cu130 |
| sentence-transformers | 5.3.0 |
| transformers | 5.2.0 |

### Interpreter Path

**Correct:**
```
/common/users/mg1998/miniforge3/envs/memorybench/bin/python
```

**Incorrect (will fail):**
```
/common/users/mg1998/miniforge3/bin/python  # Base conda - has NCCL library conflict
```

## Running Benchmarks

### Method 1: Direct conda run (Recommended)

```bash
cd /common/home/mg1998/MemoryBench/Benchmark_Pipeline

# Load API keys
source ../.env.local

# Unset problematic HF token if set incorrectly
unset HUGGING_FACE_HUB_TOKEN

# Run with conda environment
conda run -n memorybench python run_benchmark.py \
    --task visual_case_archive_assistant \
    --model gpt-4.1-nano \
    --method m2a_full
```

### Method 2: Activate environment first

```bash
conda activate memorybench
cd /common/home/mg1998/MemoryBench/Benchmark_Pipeline
source ../.env.local
unset HUGGING_FACE_HUB_TOKEN
python run_benchmark.py --task visual_case_archive_assistant --model gpt-4.1-nano --method m2a_full
```

## Verifying Dense Embeddings Work

Run this test to confirm the environment supports dense embeddings:

```bash
conda run -n memorybench python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
emb = model.encode('test sentence')
print('Text embedding dim:', len(emb))

from transformers import CLIPModel, CLIPProcessor
clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
print('CLIP loaded OK')
print('All checks passed!')
"
```

## Fallback Behavior

If dense embeddings are unavailable (wrong environment, missing packages), `m2a_full` will automatically fallback to TF-IDF based retrieval. You will see these warnings:

```
WARNING - Dense embeddings will be disabled
WARNING - Dense embeddings requested but unavailable. Falling back to TF-IDF based retrieval.
WARNING - Image retrieval requested but unavailable. Disabling cross-modal search.
```

If you see these warnings, switch to the `memorybench` conda environment.

## Common Issues

### Issue: `libtorch_cuda.so: undefined symbol: ncclGroupSimulateEnd`

**Cause:** Using base conda environment instead of memorybench

**Fix:** `conda activate memorybench` or use `conda run -n memorybench`

### Issue: `'ascii' codec can't encode characters`

**Cause:** `HUGGING_FACE_HUB_TOKEN` environment variable contains non-ASCII characters

**Fix:** `unset HUGGING_FACE_HUB_TOKEN`

### Issue: Missing OpenAI API key

**Cause:** API keys not loaded

**Fix:** `source /common/home/mg1998/MemoryBench/.env.local`

## Method Configurations

| Method | Dense Embeddings | Image Retrieval | Config File |
|--------|-----------------|-----------------|-------------|
| `m2a_full` | Yes (all-MiniLM-L6-v2) | Yes (CLIP) | `config/methods/m2a_full.yaml` |
| `m2a_lite` | Yes | No | `config/methods/m2a_lite.yaml` |
| `m2a_tfidf` | No (baseline) | No | `config/methods/m2a_tfidf.yaml` |
| `hybrid_rag` | No | No | `config/methods/hybrid_rag.yaml` |
| `full_context` | N/A | N/A | `config/methods/full_context.yaml` |
