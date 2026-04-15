# AGENTS.md

## Scope
This file contains EverMemOS-specific instructions for adapting official EverMemOS to this benchmark’s Mem-Gallery-style dataset.

Follow this file for all work in this subtree.

## Project Goal
Implement EverMemOS on this project’s dataset, which follows the same general format as Mem-Gallery.

The target is a faithful benchmark adaptation of official EverMemOS, not a redesign of EverMemOS and not a reimplementation of Mem-Gallery’s internal baseline framework.

## Source of Truth

### Official EverMemOS code is the source of truth for:
- memory architecture
- storage behavior
- updating behavior
- retrieval behavior
- generation behavior
- EverMemOS-specific configs and internal logic

Use official EverMemOS code as the primary reference for core system behavior.

Official EverMemOS references:
- main repo: `EverMind-AI/EverMemOS`
- official evaluation reference: `EverMind-AI/EverMemOS/evaluation/README.md`

### Benchmark-side adaptation logic is the source of truth for:
- dataset parsing
- dialogue-to-memory conversion
- how image captions are injected into text-only memory
- how question images are represented at recall time
- the final QA loop structure
- benchmark-style prompt flow
- local output and runner conventions

Use:
- external reference: Mem-Gallery `benchmark/run/run_bench.py`
- local reference: `Benchmark_Pipeline/benchmark/a_mem.py`
- local reference: `Benchmark_Pipeline/benchmark/runner.py`
- local reference: `Benchmark_Pipeline/benchmark/methods.py`
- `Benchmark_Pipeline/benchmark/prompt/`

Notes:
- There is no local `Benchmark_Pipeline/benchmark/run/run_bench.py` in this branch.
- Use Mem-Gallery’s official `run_bench.py` as an external behavioral reference only.
- In this branch, there are no local `memory_os.py` or `memgpt` adapter files. Do not assume they exist.
- Use `Benchmark_Pipeline/benchmark/a_mem.py` as the main local adapter reference for code structure, benchmark integration pattern, and logging style.
- Reuse `Benchmark_Pipeline/benchmark/methods.py` and `Benchmark_Pipeline/benchmark/runner.py` if they are useful.
- Do not duplicate existing benchmark utilities if they already solve the needed problem cleanly.

## Non-Negotiable Rules
- Keep the adaptation thin.
- Build a benchmark adapter around official EverMemOS.
- Do not rewrite EverMemOS into MemEngine form unless explicitly required.
- Do not reimplement unrelated Mem-Gallery baselines.
- Do not redesign EverMemOS core behavior.
- Do not silently substitute models, embedders, APIs, or retrieval components.
- If a required API, model, or config is missing, stop and ask.

## Critical Rule: Treat EverMemOS as Text-Only for This Adaptation
For this benchmark adaptation, treat EverMemOS as a text-only memory system unless the user explicitly asks for a new multimodal EverMemOS extension.

Therefore:
- Do not add a native visual memory module.
- Do not add a separate visual embedding store.
- Do not add cross-modal retrieval.
- Do not redesign EverMemOS into a new multimodal memory architecture.

For this benchmark, images must be incorporated using the Mem-Gallery text-only strategy:
- use the provided image caption from the dataset
- append image information into the stored text
- append question-image caption text into the recall query when needed

This is a benchmark adaptation layer. Do not present EverMemOS itself as natively multimodal unless explicitly implementing a new multimodal extension.

## How to Represent Images for EverMemOS
When a dialogue turn contains an image, store it as text in the adapter layer, not as a native image memory object.

Append:
- `image_caption`

Recommended stored-text pattern:

    <dialogue text>
    image:
    image_caption: <caption>

At recall time, if a question includes an image, append the question image caption to the recall query in the same textual style.

Do this in the benchmark adapter layer. Do not push this logic deep into EverMemOS core internals unless there is no clean wrapper alternative.

## Wrapper Boundary
The wrapper is responsible for:
- loading Mem-Gallery-format dialogue data
- converting each dialogue turn into the text form expected by EverMemOS
- injecting image captions into stored text
- injecting question-image captions into recall queries
- passing retrieved EverMemOS context into the benchmark QA loop
- reusing local benchmark utilities when useful
- saving outputs in the local benchmark format

Official EverMemOS code remains responsible for:
- storage internals
- updating internals
- retrieval internals
- generation internals
- EverMemOS-specific data flow and configuration

## Code Organization
Prefer clean separation between:
- official EverMemOS code
- benchmark adapter code
- local experiment utilities

Preferred approach:
- keep official EverMemOS code minimally modified
- place dataset parsing and benchmark glue outside EverMemOS core files
- use wrapper or adapter code whenever possible
- reuse `Benchmark_Pipeline/benchmark/methods.py` and `Benchmark_Pipeline/benchmark/runner.py` where practical
- use `Benchmark_Pipeline/benchmark/a_mem.py` as the main local reference for the expected adapter pattern
- make it easy to trace which files are original and which are local additions

## Dataset
Dataset root:
- `Benchmark_Pipeline/data`

Current priority files:
- `Benchmark_Pipeline/data/dialog/Home_Renovation_Interior_Design.json`
- `Benchmark_Pipeline/data/dialog/Multi-Scene_Visual_Case_Archive_Assistant.json`

Image root:
- `Benchmark_Pipeline/data/image`

Assume Mem-Gallery-style format unless local files show otherwise.

## Prompts
Prompts are stored in:
- `Benchmark_Pipeline/benchmark/prompt`

Use existing benchmark prompts when possible.
Only add new prompts if necessary.
Do not silently replace existing benchmark prompts.

## Models
- Always prefer the model specified in the official EverMemOS code, paper, or benchmark protocol.
- If a required API or model is unavailable, ask the user.
- Do not switch to alternative models without notification.

## Outputs
Write outputs and run artifacts to:
- `Benchmark_Pipeline/output/{task_name}`
- `Benchmark_Pipeline/runs/{task_name}`

Save enough information to debug and reproduce runs, including when available:
- config used
- dataset path
- model names
- prompt names
- stored memory text
- recall query text
- retrieved memories
- final predictions
- run logs
- error logs

## Modification Policy
Reuse official EverMemOS code whenever possible.

Before making substantive changes to official EverMemOS behavior, ask for permission.

Substantive changes include:
- changing memory schema
- changing storage logic
- changing update logic
- changing retrieval logic
- changing generation logic
- changing embeddings
- changing core prompts inside EverMemOS
- changing EverMemOS-specific configs or module interfaces

When in doubt:
- keep EverMemOS core unchanged
- implement the behavior in the adapter layer
- ask before making deeper changes

## Practical Implementation Priority
Implement in this order:
1. inspect local dataset files directly
2. inspect official EverMemOS evaluation or example code
3. inspect external Mem-Gallery `run_bench.py` for benchmark-side behavior
4. inspect `Benchmark_Pipeline/benchmark/methods.py` and `Benchmark_Pipeline/benchmark/runner.py` for reusable utilities
5. inspect `Benchmark_Pipeline/benchmark/a_mem.py` for local adapter structure reference
6. load Mem-Gallery-format dialogue data
7. convert each dialogue turn into EverMemOS-storable text
8. inject image captions into stored text
9. inject question-image captions into recall queries
10. connect EverMemOS retrieval to the benchmark QA loop
11. save outputs in the project’s output structure
12. only then do cleanup or refactoring

## Files to Read First
Read these files before coding:
1. official EverMemOS evaluation or example code
2. official EverMemOS evaluation README: `EverMind-AI/EverMemOS/evaluation/README.md`
3. external Mem-Gallery `benchmark/run/run_bench.py`
4. `Benchmark_Pipeline/benchmark/methods.py`
5. `Benchmark_Pipeline/benchmark/runner.py`
6. `Benchmark_Pipeline/benchmark/a_mem.py`
7. `Benchmark_Pipeline/data/dialog/Home_Renovation_Interior_Design.json`
8. `Benchmark_Pipeline/data/dialog/Multi-Scene_Visual_Case_Archive_Assistant.json`

## Human Approval Required
Do not do any of the following without explicit permission:
- change official EverMemOS core behavior
- add a new multimodal retrieval architecture
- install major new dependencies
- substantially restructure the repository
- switch models or APIs from the paper-faithful choice
- push to GitHub
- push to Hugging Face