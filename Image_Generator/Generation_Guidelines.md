# Image Generator Guidelines

This document records the current best practices for creating MemEye-ready image datasets and associated question sets.

It applies across generators, not only to a single task family.

## Goal

Generated data should support benchmark items that are:

- visually grounded
- resistant to text-only bypass
- easy to score robustly
- aligned with the MemEye `point` taxonomy

## Core Rules

### 1. Never require hidden answers

The answer must be recoverable from the user-visible input.

Allowed:

- names visible in the image or dialogue
- visible object categories
- explicit option labels
- left/right or other visible positional labels
- yes/no over visible conditions

Not allowed unless explicitly exposed:

- internal object ids
- latent speaker ids
- generator-only metadata
- backend labels like `avatar_1`, `spk_3`, or private scene ids

### 2. Prefer controlled answer spaces

For benchmark stability, prefer:

- `yes` / `no`
- one visible name
- one option letter
- one position label
- one short phrase with explicit output formatting

If the answer must contain multiple tokens, constrain the exact format in the question.

Example:

- `Reply with only yes or no in lowercase, with no punctuation.`
- `Reply with only the character name, with no punctuation.`
- `Answer with exactly two color words separated by 'and'.`

### 3. Keep the answer space visible and local

A question is better when the model knows what kind of answer is expected.

Good:

- binary decisions
- visible choice sets
- fixed role names
- fixed location labels

Weak:

- open-ended summaries
- free-form explanations unless explanation quality itself is the benchmark target

### 4. Preserve true visual necessity

Do not ask questions whose answer can be recovered almost entirely from dialogue text or a simple summary.

Prefer items that depend on:

- identity persistence
- location change
- panel-to-panel continuity
- subtle visible state differences
- small but still reliable local visual evidence

### 5. Use hard items on purpose

MemEye should contain clean hard items.

A good hard item:

- has a visible answer
- uses a controlled output format
- still causes model errors

Do not remove a question only because the model misses it.
Remove it only if the failure is caused by:

- hidden answers
- unclear annotation
- ambiguous task wording
- unreliable evidence

## Task Design Heuristics

### Good question families

- character tracking across images or panels
- spatial continuity across time
- visual state change
- constrained visual comparison
- panel-level event ordering when the event is visually explicit

### Weak question families

- broad plot summaries
- questions that reward retelling a caption
- questions whose answer is mostly supplied by narration
- tasks with free-form answers when a controlled alternative exists

## Difficulty Strategy

Each dataset should include a mix of:

- baseline items
- medium items
- hard items

Recommended interpretation:

- `baseline`: likely solvable if the model preserves a single visible fact
- `hard`: answer remains visible and well-defined, but requires multi-step visual memory or careful comparison

Hard items are desirable when they are clean.

## Writing Questions

Before finalizing a question, check:

1. Is the answer visible somewhere in the actual benchmark input?
2. Can the answer be expressed in a controlled format?
3. Is the question testing visual memory rather than text paraphrase?
4. Is the MemEye `point` label the minimum sufficient coordinate?
5. Would the question remain valid if another partner read only the benchmark input and not the generator code?

## Writing Images

When generating images, optimize for:

- stable visual identifiers
- repeated entities under varied context
- clear but non-trivial location changes
- enough detail to support hard items
- consistency across sessions or panels

Avoid:

- clutter that makes annotation unreliable
- details that are too tiny to be read consistently even by strong MLLMs
- generator artifacts that force the answer to rely on private metadata

## Relationship To MemEye

All generated tasks should eventually map cleanly into:

- `point = [[X...], [Y...]]`

Use the MemEye annotation guide as the final source of truth for taxonomy:

- `Benchmark_Pipeline/MemEye_Annotation_Guide.md`
