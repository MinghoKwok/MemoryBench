# MemEye Annotation Guide

This document defines how partners should assign `point` labels for MemEye tasks and how to interpret the current benchmark items already in this repo.

## Goal

MemEye does not use legacy one-dimensional tags such as `FR`, `VS`, `MR`, `CD`, `VB`, `CS`, or `HR` as the primary benchmark taxonomy.

Instead, every QA item should be annotated with a binocular coordinate:

```json
"point": [["X2"], ["Y1"]]
```

The first list is the visual axis.
The second list is the reasoning axis.

Multiple labels are allowed on either axis when a question genuinely requires more than one kind of visual demand or reasoning demand:

```json
"point": [["X0", "X4"], ["Y3"]]
```

Use multiple labels sparingly. Do not add a second label unless the item would be mischaracterized by a single label.

## Formal Schema

Each QA item should follow this structure:

```json
{
  "question": "...",
  "answer": "...",
  "point": [["X_i", "..."], ["Y_j", "..."]],
  "session_id": ["D1", "D7"],
  "clue": ["D1:9", "D7:3"]
}
```

Rules:

- `point[0]` is the `X` axis list.
- `point[1]` is the `Y` axis list.
- Every `X` label must be one of `X1`, `X2`, `X3`, `X4`.
- Every `Y` label must be one of `Y1`, `Y2`, `Y3`.
- Prefer one `X` label and one `Y` label.
- Use multiple `X` labels only when the question truly combines multiple visual demands.
- Use multiple `Y` labels only in rare cases. In most cases, choose the highest reasoning level that best characterizes the item.

## X-Axis Rules

### `X1`: Global Scene Gist

Use `X1` when only coarse visual gist is required.

Use `X1` for:

- room type, scene type, broad style, broad environment
- high-level visual theme that does not require precise object binding or spatial precision

Do not use `X1` if the item depends on:

- exact instance identity
- fine spatial arrangement
- small attributes, subtle color, texture, or OCR

### `X2`: Entity Instance Retrieval

Use `X2` when the model must bind a reference to a specific person, object, item, or persistent visual identity.

Use `X2` for:

- “Who was the person/object/item from before?”
- speaker or object identity tracking
- cross-turn or cross-session reference resolution to a particular instance

Examples:

- “Who asked about dinner?”
- “Which chair was the one discussed earlier?”

### `X3`: Spatial Grounding

Use `X3` when the answer depends on location, side, order, adjacency, layout, or structural arrangement in the image.

Use `X3` for:

- left/right or top/bottom placement
- relative location
- layout structure
- visually ordered attachments or adjacency, such as a message followed by a photo card

### `X4`: Micro-Attribute Reasoning

Use `X4` when the decisive evidence is fine-grained and likely to be lost in captioning.

Use `X4` for:

- subtle color distinctions
- texture or material
- small OCR
- tiny marks, logos, inscriptions, shape differences

## Y-Axis Rules

### `Y1`: Direct Retrieval

This is the lowest reasoning level. The model only needs to retrieve a single fact from a single session, without linking information across sessions or resolving ambiguity. It mainly tests whether the model can access stored memory at all.

Use `Y1` for:

- one-session lookup
- one-image lookup
- one fact tied to one clearly relevant clue span
- no cross-session linking needed

### `Y2`: Compositional Linking

At this level, the model must connect information across sessions, modalities, or references. The answer is not contained in one isolated memory fragment, but can still be obtained by combining consistent evidence. Importantly, this level remains **monotonic**: later information does not overwrite or invalidate earlier information.

Use `Y2` for:

- “the person who said A later said what?”
- linking a name or instance to later evidence
- verifying whether an entity did or did not satisfy some condition after checking multiple pieces of evidence
- cross-session or cross-modal reference resolution where evidence is consistent

### `Y3`: State-Evolving Synthesis

This is the highest reasoning level. The model must reason over evolving states, where later evidence may update, override, or conflict with earlier memory. Solving these tasks requires **non-monotonic reasoning**, conflict detection, and coherent world-model revision rather than simple retrieval or linking.

Use `Y3` for:

- inferring the **current** state after several updates (e.g., a decision is made, then reversed)
- planning under changing constraints
- reconciling conflicting evidence
- questions where the naive (first-seen) answer is wrong because it was later overridden

## Partner Rules

Partners should follow these rules when creating new tasks.

### 1. Annotate the minimum sufficient coordinate

Choose the lowest-complexity coordinate that still accurately describes the task.

Good:

- `[['X2'], ['Y1']]` for “Who asked about dinner?”

Bad:

- `[['X2', 'X3', 'X4'], ['Y1', 'Y2']]` when only identity retrieval is needed

### 2. Do not encode hidden generator metadata in the answer

The answer must be recoverable from user-visible inputs.

Allowed:

- speaker names
- visible object identity
- left/right
- yes/no
- visible design choice

Not allowed unless explicitly exposed in the prompt or image:

- internal ids like `avatar_1`
- latent speaker ids like `spk_3`
- backend object keys

### 3. Prefer answer spaces that are easy to evaluate

For benchmark stability:

- use `yes` or `no` explicitly for binary questions
- ask for a speaker name when the image shows names
- ask for `left` or `right` when the structure is spatial
- ask for a concrete option label when a choice set is visible
- prefer short controlled answers such as one word, one name, one option id, or a fixed phrase pattern when the task does not require free-form explanation
- when multiple tokens are necessary, constrain the format explicitly, for example “answer with exactly two color words separated by 'and'”

Avoid vague free-form questions when a constrained answer space is available.

### 4. Ensure QAs require image memory (critical)

Every QA must require viewing the actual images to answer correctly. Text-only answerable QAs defeat the purpose of visual memory benchmarking.

#### 4.1 Prohibited patterns in dialogue

**Do NOT include image descriptions in assistant responses:**

Bad:
```json
{
  "assistant": "This advertisement features a classic Coca-Cola glass contour bottle held by a person. The product is set against the brand's signature solid red background."
}
```

Good:
```json
{
  "assistant": "Stored for later memory questions."
}
```

**Do NOT include answer-revealing details in user or assistant text:**

Bad:
```json
{
  "user": "Look at those gray cabinets. That color is so depressing."
}
```
(If a QA asks "What color are the cabinets?", the answer "gray" is in the text)

Good:
```json
{
  "user": "Look at those cabinets. That color is so depressing."
}
```

#### 4.2 Keep image_caption minimal

The `image_caption` field should be generic and not reveal visual details that could answer QAs.

Bad:
```json
{
  "image_caption": ["Alley Oop page 1 with five panels, including a crowned character in water and later panels showing the same crowned character again."]
}
```

Good:
```json
{
  "image_caption": ["Alley Oop comic page 1."]
}
```

#### 4.3 Prohibited QA types

The following QA patterns should be excluded or rewritten:

| Pattern | Example | Problem |
|---------|---------|---------|
| Answer is "Not mentioned" | "What brand is the refrigerator?" → "Not mentioned" | No image viewing required |
| Answer is in dialogue | "What color did she choose?" when dialogue says "I'll go with Cavern Clay" | Text memory sufficient |
| Synthesis of text-only facts | "Trace the evolution of her color choice" when all colors are named in dialogue | Text memory sufficient |

#### 4.4 Verification checklist

Before finalizing a QA, verify:

- [ ] The answer cannot be found by searching dialogue text
- [ ] The assistant never describes the image content
- [ ] The image_caption does not reveal the answer
- [ ] Removing the image would make the question unanswerable
- [ ] The answer is NOT "Not mentioned" or similar negative responses

### 5. Use `X0` carefully

`X0` is not “cross-image” by itself.
It is specifically inter-image pattern induction.

Do not assign `X0` just because a question references multiple sessions.
Use `X0` only when the answer depends on inferring a latent visual family or rule across examples.

## Current Task Audit

This section records the current recommended mapping for the benchmark tasks already in the repo.

### `Brand_Memory_Test`

Recommended mappings:

- “In the first advertisement, was the Coca-Cola shown in a bottle or a can? Answer with one word only.”
  - `[['X2'], ['Y1']]`
  - rationale: single-session instance-level visual recall

- “What were the two main colors of the logo on the iced coffee cup in the Dunkin' ad? Answer with exactly two color words separated by 'and'.”
  - `[['X4'], ['Y1']]`
  - rationale: fine-grained visual attribute recall

- “Across the two advertisements, which brand relied on a solid red background as its visual anchor? Answer with the brand name only.”
  - `[['X0'], ['Y2']]`
  - rationale: cross-session visual pattern comparison with relational association

- “Which brand showed its featured drink next to a donut and sandwich rather than in a person's hand?”
  - `[['X2', 'X3'], ['Y2']]`
  - rationale: cross-session visual comparison over entity presentation and local spatial arrangement

### `Chat_UI_Memory_Test`

Recommended mappings:

- “Who asked about dinner?”
  - `[['X2'], ['Y1']]`

- “The person who mentioned Friday later said what in another screenshot?”
  - `[['X2'], ['Y2']]`

- “Was the message '...' shown on the left or right side?”
  - `[['X3'], ['Y1']]`

- “Did NAME mention airport? Answer yes or no.”
  - `[['X2'], ['Y2']]`

- “Who sent a message followed by a photo card?”
  - `[['X2', 'X3'], ['Y1']]`
  - rationale: identity binding plus local structural layout

## Review Checklist

Before merging a new task, verify:

**Annotation quality:**
- the answer is recoverable from visible inputs
- the question does not depend on hidden metadata
- the `X` label describes the minimal necessary visual demand
- the `Y` label describes the minimal necessary reasoning demand
- multi-label `point` is used only when justified
- the clue list actually supports the annotated reasoning path
- the answer format is as controlled as possible without weakening the underlying visual or reasoning demand

**Image memory requirement (critical):**
- the answer is NOT findable in dialogue text alone
- assistant responses do NOT describe image contents
- image_caption fields are generic (e.g., "Comic page 1") not descriptive
- the answer is NOT "Not mentioned" or similar negative responses
- removing the image would make the question unanswerable

**Common failures to check:**
- grep the dialogue for keywords in the answer
- verify assistant responses are minimal (e.g., "Stored for later memory questions")
- confirm image_caption does not contain answer-relevant details

**Method implementation — `clue` field usage (critical):**

The `clue` field in each QA item is **annotation-level gold evidence** — it records which dialogue rounds contain the visual evidence needed to answer the question. It exists for evaluation analysis (e.g., retrieval recall), NOT for use at inference time.

- **NEVER** pass `clue` round IDs or their images to a method's `answer()` function. Doing so leaks oracle visual evidence and inflates scores.
- Agentic methods (M2A, MMA, etc.) must retrieve their own evidence from the memory store. The only images an agentic method may receive at inference time are those attached to the **question itself** (via a `question_images` field), not clue-derived images.
- If a method needs query-side images, use `qa.get("question_images")`, never `qa.get("clue")`.
- This was a real bug in an earlier MMA implementation that inflated EM by ~27% before being caught and fixed.

Example of **wrong** implementation:
```python
# BAD — leaks gold evidence
clue_rounds = qa.get("clue", [])
images = [dataset.rounds[r]["images"] for r in clue_rounds]
system.answer_question(question, image_paths=images)
```

Example of **correct** implementation:
```python
# GOOD — no oracle information
qa_images = qa.get("question_images") or None
system.answer_question(question, image_paths=qa_images)
```
