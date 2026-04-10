# AGENTS.md

## Scope
This file contains MemoryOS-specific instructions for adapting official MemoryOS to this benchmark’s Mem-Gallery-style dataset.

Follow this file for all work in this subtree.

## Project Goal
Implement MemoryOS on this project’s dataset, which follows the same general format as Mem-Gallery.

The target is a faithful benchmark adaptation of official MemoryOS, not a redesign of MemoryOS and not a reimplementation of Mem-Gallery’s internal baseline framework.

## Source of Truth

### Official MemoryOS code is the source of truth for:
- memory architecture
- storage behavior
- updating behavior
- retrieval behavior
- generation behavior
- MemoryOS-specific configs and internal logic

Use official MemoryOS code as the primary reference for core system behavior. The official github has been downloaded at Benchmark_Pipeline/benchmark/memoryos/MemoryOS. 

### Mem-Gallery code is the source of truth for:
- dataset parsing
- dialogue-to-memory conversion
- how image captions are injected into text-only memory

- how question images are represented at recall time
- the final QA loop structure
- benchmark-style prompt flow

Reference to run_bench.py in Mem-gallery's official code.
This is the a_mem adaptation for reference: Benchmark_Pipeline/benchmark/a_mem.py

Use:
- `Benchmark_Pipeline/benchmark/prompt/`

## Non-Negotiable Rules
- Keep the adaptation thin.
- Build a Mem-Gallery-style adapter around official MemoryOS.
- Do not rewrite MemoryOS into MemEngine form unless explicitly required.
- Do not reimplement unrelated Mem-Gallery baselines.
- Do not redesign MemoryOS core behavior.
- Do not silently substitute models, embedders, APIs, or retrieval components.
- If a required API, model, or config is missing, stop and ask.

## Critical Rule: Treat MemoryOS as Text-Only for This Adaptation
For this benchmark adaptation, treat MemoryOS as a text-only memory system unless the user explicitly asks for a new multimodal MemoryOS extension.

Therefore:
- Do not add a native visual memory module.
- Do not add a separate visual embedding store.
- Do not add cross-modal retrieval.
- Do not redesign MemoryOS into a new multimodal memory architecture.

For this benchmark, images must be incorporated using the Mem-Gallery text-only strategy:
- use the provided image caption from the dataset
- append image information into the stored text
- append question-image caption text into the recall query when needed

## How to Represent Images for MemoryOS
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

Do this in the benchmark adapter layer. Do not push this logic deep into MemoryOS core internals unless there is no clean wrapper alternative.

## Wrapper Boundary
The wrapper is responsible for:
- loading Mem-Gallery-format dialogue data
- converting each dialogue turn into the text form expected by MemoryOS
- injecting image captions into stored text
- injecting question-image captions into recall queries
- passing retrieved MemoryOS context into the benchmark QA loop
- saving outputs in the local benchmark format

Official MemoryOS code remains responsible for:
- storage internals
- updating internals
- retrieval internals
- generation internals
- MemoryOS-specific data flow and configuration

## Code Organization
Prefer clean separation between:
- official MemoryOS code
- benchmark adapter code
- local experiment utilities

Preferred approach:
- keep official MemoryOS code minimally modified
- place dataset parsing and benchmark glue outside MemoryOS core files
- use wrapper or adapter code whenever possible
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
- Always prefer the model specified in the official MemoryOS code, paper, or benchmark protocol.
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
Reuse official MemoryOS code whenever possible.

Before making substantive changes to official MemoryOS behavior, ask for permission.

Substantive changes include:
- changing memory schema
- changing storage logic
- changing update logic
- changing retrieval logic
- changing generation logic
- changing embeddings
- changing core prompts inside MemoryOS
- changing MemoryOS-specific configs or module interfaces

When in doubt:
- keep MemoryOS core unchanged
- implement the behavior in the adapter layer
- ask before making deeper changes

## Practical Implementation Priority
Implement in this order:
1. inspect local dataset files directly
2. load Mem-Gallery-format dialogue data
3. convert each dialogue turn into MemoryOS-storable text
4. inject image captions into stored text
5. inject question-image captions into recall queries
6. connect MemoryOS retrieval to the benchmark QA loop
7. save outputs in the project’s output structure
8. only then do cleanup or refactoring

## Files to Read First
Read these files before coding:
1. official MemoryOS evaluation or example code
2. `Benchmark_Pipeline/benchmark/run/run_bench.py`
3. `Benchmark_Pipeline/MemEye_Annotation_Guide.md`
4. `Benchmark_Pipeline/data/dialog/Home_Renovation_Interior_Design.json`
5. `Benchmark_Pipeline/data/dialog/Multi-Scene_Visual_Case_Archive_Assistant.json`

## Human Approval Required
Do not do any of the following without explicit permission:
- change official MemoryOS core behavior
- add a new multimodal retrieval architecture
- install major new dependencies
- substantially restructure the repository
- switch models or APIs from the paper-faithful choice
- push to GitHub
- push to Hugging Face