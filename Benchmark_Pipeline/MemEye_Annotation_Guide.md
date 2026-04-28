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

Although `point` is stored as a two-list JSON field for schema stability, the current benchmark policy is **single-label on each axis**: one `X` label and one `Y` label per question.

## Highest-Bottleneck Rule

MemEye now uses a single-label **Highest-Bottleneck Rule**.

Visual perception and reasoning are inclusive: a question that depends on a tiny OCR token (`X4`) also depends on broader scene understanding (`X1`) and usually instance binding (`X2`). If we assigned every involved level, per-cell counts in the `(X, Y)` matrix would become uninterpretable and errors would be hard to attribute. Therefore, each question receives exactly one `X_i` and one `Y_j`: the highest level whose absence would prevent a correct answer.

Operationally:

- Walk up the `X` axis and ask whether the question would still be answerable using only that level of visual perception.
- Assign the highest `X` level that is strictly required, equivalently the lowest level at which the answer becomes recoverable.
- Apply the same rule on the `Y` axis and assign the highest reasoning level without which the answer would fail.
- If one drafted question mixes two separable demands, split it into two questions rather than attaching multiple labels.

## Formal Schema

Each QA item should follow this structure:

```json
{
  "question": "...",
  "answer": "...",
  "point": [["X_i"], ["Y_j"]],
  "session_id": ["D1", "D7"],
  "clue": ["D1:9", "D7:3"]
}
```

Rules:

- `point[0]` is the `X` axis list and must contain exactly one label.
- `point[1]` is the `Y` axis list and must contain exactly one label.
- The only valid `X` labels are `X1`, `X2`, `X3`, `X4`.
- The only valid `Y` labels are `Y1`, `Y2`, `Y3`.
- Do not use `X0`.
- Do not attach multiple labels to represent everything a question "involves"; assign only the bottleneck coordinate.

## X-Axis Rules

### `X1`: Global Scene

Use `X1` when only coarse visual gist is required.

Use `X1` for:

- room type, scene type, broad style, broad environment
- high-level visual theme that does not require precise object binding or spatial precision

Do not use `X1` if the item depends on:

- exact instance identity
- fine spatial arrangement
- small attributes, subtle color, texture, or OCR

### `X2`: Region Scene

Use `X2` when the decisive evidence is a semantically coherent local region, functional area, grouped visual context, or region-level layout within a scene.

Use `X2` for:

- local room, shelf, table, dashboard, road, or UI regions
- grouped entities within a localized scene region
- relative placement, ordering, or proximity when the bottleneck is the local region layout
- state changes in a local region, such as where an object currently sits in a shelf or tabletop layout

Examples:

- “Which backsplash sample sits between the white subway tile and the gray herringbone sample?”
- “Where is the key bowl on the entry-console tabletop?”
- “Which dashboard card shows the metric closest to the warning banner?”

### `X3`: Instance Identity

Use `X3` when the model must bind a reference to a specific person, object, item, character, vehicle, or persistent visual identity.

Use `X3` for:

- “Who was the person/object/item from before?”
- speaker, character, vehicle, or object identity tracking
- cross-turn or cross-session reference resolution to a particular instance
- distinguishing visually similar instances or categories across images

Examples:

- “Who asked about dinner?”
- “Which chair was the one discussed earlier?”
- “On which run does the delivery tricycle cross in front of the stopped SUV?”

### `X4`: Fine-Grained Attributes

Use `X4` when the decisive evidence is fine-grained and likely to be lost in captioning.

Use `X4` for:

- subtle color distinctions
- texture or material
- small OCR
- tiny marks, logos, inscriptions, shape differences

## Y-Axis Rules

### `Y1`: Atomic Retrieval

This is the lowest reasoning level. The model only needs to retrieve a single fact from a single session, without linking information across sessions or resolving ambiguity. It mainly tests whether the model can access stored memory at all.

Use `Y1` for:

- one-session lookup
- one-image lookup
- one fact tied to one clearly relevant clue span
- no cross-session linking needed

### `Y2`: Composite Retrieval

At this level, the model must connect information across sessions, modalities, or references. The answer is not contained in one isolated memory fragment, but can still be obtained by combining consistent evidence. Importantly, this level remains **monotonic**: later information does not overwrite or invalidate earlier information.

Use `Y2` for:

- “the person who said A later said what?”
- linking a name or instance to later evidence
- verifying whether an entity did or did not satisfy some condition after checking multiple pieces of evidence
- cross-session or cross-modal reference resolution where evidence is consistent

### `Y3`: State Update Reasoning

This is the highest reasoning level. The model must reason over evolving states, where later evidence may update, override, or conflict with earlier memory. Solving these tasks requires **non-monotonic reasoning**, conflict detection, and coherent world-model revision rather than simple retrieval or linking.

Use `Y3` for:

- inferring the **current** state after several updates (e.g., a decision is made, then reversed)
- planning under changing constraints
- reconciling conflicting evidence
- questions where the naive (first-seen) answer is wrong because it was later overridden

## Partner Rules

Partners should follow these rules when creating new tasks.

### 1. Annotate the minimum sufficient coordinate

Choose the single bottleneck coordinate that still accurately describes the task.

Good:

- `[['X2'], ['Y1']]` for “Who asked about dinner?”

Bad:

- `[['X2', 'X3', 'X4'], ['Y1', 'Y2']]` when only identity retrieval is needed
- `[['X1'], ['Y2']]` when the question actually fails without instance binding and should be `[['X2'], ['Y2']]`

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
  - `[['X1'], ['Y2']]`
  - rationale: cross-session comparison of coarse visual gist under monotonic evidence

- “Which brand showed its featured drink next to a donut and sandwich rather than in a person's hand?”
  - `[['X3'], ['Y2']]`
  - rationale: the decisive evidence is local spatial arrangement, compared across sessions under monotonic evidence

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
  - `[['X3'], ['Y1']]`
  - rationale: the bottleneck is recognizing the ordered layout pattern in a single screenshot

## Review Checklist

Before merging a new task, verify:

**Annotation quality:**
- the answer is recoverable from visible inputs
- the question does not depend on hidden metadata
- the `X` label describes the highest visual bottleneck that is strictly required
- the `Y` label describes the highest reasoning bottleneck that is strictly required
- `point` contains exactly one `X` label and one `Y` label
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
