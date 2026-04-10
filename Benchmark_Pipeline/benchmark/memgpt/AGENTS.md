# AGENTS.md

## Scope
This file contains MemGPT-specific instructions for adapting official MemGPT to this benchmark’s Mem-Gallery-style dataset.

Follow this file for all work in this subtree.

## Project Goal
Implement MemGPT on this project’s dataset, which follows the same general format as Mem-Gallery.

The target is a faithful benchmark adaptation of official MemGPT, not a redesign of MemGPT and not a reimplementation of Mem-Gallery’s internal baseline framework.

## Source of Truth

### Official MemGPT code is the source of truth for:
- memory architecture
- memory tier / context management behavior
- archival or long-term memory behavior
- retrieval behavior
- agent-side memory handling
- MemGPT-specific configs and internal logic

Use official MemGPT code as the primary reference for core system behavior. The official github has been downloaded in the local benchmark subtree for MemGPT.

### Local benchmark code is the source of truth for:
- dataset parsing
- dialogue-to-memory conversion
- how image captions are injected into text-only memory
- how question images are represented at recall time
- the final QA loop structure
- benchmark-style prompt flow
- local output and runner conventions

Primary files to reference:
- `Benchmark_Pipeline/benchmark/methods.py`
- `Benchmark_Pipeline/benchmark/runner.py`
- `Benchmark_Pipeline/benchmark/a_mem.py`
- `Benchmark_Pipeline/benchmark/memory_os.py`
- `Benchmark_Pipeline/benchmark/prompt/`

Reuse `Benchmark_Pipeline/benchmark/methods.py` and `Benchmark_Pipeline/benchmark/runner.py` if they are useful.
Use `Benchmark_Pipeline/benchmark/a_mem.py` and `Benchmark_Pipeline/benchmark/memory_os.py` as adapter references for code structure, benchmark integration pattern, and logging style.
Do not duplicate existing benchmark utilities if they already solve the needed problem cleanly.

## Non-Negotiable Rules
- Keep the adaptation thin.
- Build a benchmark adapter around official MemGPT.
- Do not rewrite MemGPT into MemEngine form unless explicitly required.
- Do not reimplement unrelated Mem-Gallery baselines.
- Do not redesign MemGPT core behavior.
- Do not silently substitute models, embedders, APIs, or retrieval components.
- If a required API, model, or config is missing, stop and ask.

## Critical Rule: Treat MemGPT as Text-Only for This Adaptation
For this benchmark adaptation, treat MemGPT as a text-only memory system unless the user explicitly asks for a new multimodal MemGPT extension.

Therefore:
- Do not add a native visual memory module.
- Do not add a separate visual embedding store.
- Do not add cross-modal retrieval.
- Do not redesign MemGPT into a new multimodal memory architecture.

For this benchmark, images must be incorporated using the Mem-Gallery text-only strategy:
- use the provided image caption from the dataset
- append image information into the stored text
- append question-image caption text into the recall query when needed

## How to Represent Images for MemGPT
When a dialogue turn contains an image, store it as text in the adapter layer, not as a native image memory object.

Append at least:
- `image_id`
- `image_caption`

Recommended stored-text pattern:

    <dialogue text>
    image:
    image_id: <id>
    image_caption: <caption>

At recall time, if a question includes an image, append the question image caption to the recall query in the same textual style.

Do this in the benchmark adapter layer. Do not push this logic deep into MemGPT core internals unless there is no clean wrapper alternative.

## Wrapper Boundary
The wrapper is responsible for:
- loading Mem-Gallery-format dialogue data
- converting each dialogue turn into the text form expected by MemGPT
- injecting image captions into stored text
- injecting question-image captions into recall queries
- passing retrieved MemGPT context into the benchmark QA loop
- reusing local benchmark utilities when useful
- saving outputs in the local benchmark format

Official MemGPT code remains responsible for:
- core memory and context-management behavior
- long-term / archival memory logic
- retrieval internals
- agent-side memory handling
- MemGPT-specific data flow and configuration

## Code Organization
Prefer clean separation between:
- official MemGPT code
- benchmark adapter code
- local experiment utilities

Preferred approach:
- keep official MemGPT code minimally modified
- place dataset parsing and benchmark glue outside MemGPT core files
- use wrapper or adapter code whenever possible
- reuse `Benchmark_Pipeline/benchmark/methods.py` and `Benchmark_Pipeline/benchmark/runner.py` where practical
- use `a_mem.py` and `memory_os.py` as references for the expected adapter pattern
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
- Always prefer the model specified in the official MemGPT code, paper, or benchmark protocol.
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
Reuse official MemGPT code whenever possible.

Before making substantive changes to official MemGPT behavior, ask for permission.

Substantive changes include:
- changing memory schema
- changing context-management logic
- changing retrieval logic
- changing archival or long-term memory behavior
- changing embeddings
- changing core prompts inside MemGPT
- changing MemGPT-specific configs or module interfaces

When in doubt:
- keep MemGPT core unchanged
- implement the behavior in the adapter layer
- ask before making deeper changes

## Practical Implementation Priority
Implement in this order:
1. inspect local dataset files directly
2. inspect official MemGPT evaluation or example code
3. inspect `Benchmark_Pipeline/benchmark/methods.py` and `Benchmark_Pipeline/benchmark/runner.py` for reusable utilities
4. inspect `Benchmark_Pipeline/benchmark/a_mem.py` and `Benchmark_Pipeline/benchmark/memory_os.py` for adapter structure reference
5. load Mem-Gallery-format dialogue data
6. convert each dialogue turn into MemGPT-storable text
7. inject image captions into stored text
8. inject question-image captions into recall queries
9. connect MemGPT retrieval to the benchmark QA loop
10. save outputs in the project’s output structure
11. only then do cleanup or refactoring

## Files to Read First
Read these files before coding:
1. official MemGPT evaluation or example code
3. `Benchmark_Pipeline/benchmark/methods.py`
4. `Benchmark_Pipeline/benchmark/runner.py`
5. `Benchmark_Pipeline/benchmark/a_mem.py`
6. `Benchmark_Pipeline/benchmark/memory_os.py`
8. `Benchmark_Pipeline/data/dialog/Home_Renovation_Interior_Design.json`

## Human Approval Required
Do not do any of the following without explicit permission:
- change official MemGPT core behavior
- add a new multimodal retrieval architecture
- install major new dependencies
- substantially restructure the repository
- switch models or APIs from the paper-faithful choice
- push to GitHub
- push to Hugging Face