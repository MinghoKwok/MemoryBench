# MemEye ComicScene Assessment

This note records the first-pass assessment of `Image_Generator/ComicScene` as a candidate MemEye task family.

## Bottom Line

`ComicScene` is promising as a MemEye sub-benchmark, but not in its current form.

The strongest reusable assets are:

- raw comic page images
- panel bounding boxes
- page ordering

The weakest assets are the old summary-heavy scene annotations and story outputs, which are too close to text-compressed narrative supervision.

## Keep / Rewrite / Drop

### Keep

These are good MemEye building blocks:

- `Data/Alley_Oop/images/*.jpg`
- `Data/Alley_Oop/Alley_Oop.json`

Why:

- they preserve native visual evidence
- they support identity tracking, spatial continuity, and cross-panel state change
- they do not force the benchmark into a summary-style format

### Rewrite

These can help author questions, but should not define the benchmark format:

- `Data/Alley_Oop/benchmark_scenes/*.json`
- `Data/Alley_Oop/benchmark_refined_scenes/**/*.json`
- `output/results.json`

Why:

- they are useful as weak scene references
- but they over-compress the comic into textual narrative arcs
- this increases the risk of visual bypassability

### Drop From Core Evaluation

These old task styles should not be the MemEye target format:

- free-form story-summary supervision
- summary-first “what happened in the comic?” tasks
- broad narrative comprehension without image-locked answer spaces

Why:

- they reward text-style retelling more than visual memory
- they are hard to score robustly
- they do not align well with controlled MemEye answer formats

## Best Initial Task Families

### 1. Character Tracking

Recommended coordinate region:

- `X2 + Y1`
- `X2 + Y2`

Target ability:

- track a visually persistent character across panels or pages
- answer with `yes/no`, a fixed role name, or a fixed option label

Why this is strong:

- identity persistence is central to multimodal memory
- comics naturally produce repeated entities under changing pose and context
- answer spaces can be tightly controlled

### 2. Spatial Continuity

Recommended coordinate region:

- `X3 + Y2`
- sometimes `X2 + X3 + Y2`

Target ability:

- preserve where a character or object is relative to the scene
- track movement or position changes across panels

Why this is strong:

- caption surrogates often lose exact panel-level spatial detail
- comics make location changes visually explicit

### 3. Visual State Change

Recommended coordinate region:

- `X2 + Y2`
- `X2 + X3 + Y2`

Target ability:

- compare the same entity across time
- detect whether its state changed, and how

Why this is strong:

- supports visually grounded memory without requiring open-ended summarization

### 4. Fine Detail Tasks

Recommended coordinate region:

- `X4 + Y2`

Status:

- promising, but lower priority for the first batch

Why:

- old comic print quality and OCR noise may make this expensive to curate
- better as a second-phase task family

## First Candidate Questions

These are draft candidates for a first `Alley_Oop` MemEye subset. They are not yet exported into benchmark JSON. They are written to validate task shape.

### Candidate 1

- Source: `Alley_Oop_Page_1`, panel 1 and panel 5
- Question: `In panel 1 and panel 5 of page 1, is the crowned character in the water both times? Answer yes or no.`
- Answer: `yes`
- Point: `[['X2', 'X3'], ['Y2']]`
- Why: requires identifying the crowned character and comparing his location across two panels

### Candidate 2

- Source: `Alley_Oop_Page_1`, panel 1
- Question: `Who is in the water in panel 1 of page 1? Answer with the character name only.`
- Answer: `King Guz`
- Point: `[['X2'], ['Y1']]`
- Why: simple controlled identity retrieval

### Candidate 3

- Source: `Alley_Oop_Page_1`, panel 1 and panel 5
- Question: `For the crowned character on page 1, which location pattern is correct? A) land-to-land B) water-to-land C) water-to-water. Answer with A, B, or C only.`
- Answer: `C`
- Point: `[['X3'], ['Y2']]`
- Why: controlled spatial continuity question

### Candidate 4

- Source: `Alley_Oop_Page_2`, panel 2 and panel 3
- Question: `On page 2, does Dinny run into a tree before the conversation about his clumsiness? Answer yes or no.`
- Answer: `yes`
- Point: `[['X2', 'X3'], ['Y2']]`
- Why: visually grounded event ordering over adjacent panels

### Candidate 5

- Source: `Alley_Oop_Page_2` and `Alley_Oop_Page_3`
- Question: `Across pages 2 and 3, is the dinosaur being considered for trade-in larger than the newer dinosaur shown at the dealership? Answer yes or no.`
- Answer: `yes`
- Point: `[['X2'], ['Y2']]`
- Why: cross-page entity comparison with controlled output

### Candidate 6

- Source: `Alley_Oop_Page_3`, dealership panels
- Question: `At the dealership, is the dinosaur labeled as today's special smaller than the dinosaur marked cheap? Answer yes or no.`
- Answer: `yes`
- Point: `[['X3', 'X4'], ['Y1']]`
- Why: local visual comparison with mild fine-detail reading

### Candidate 7

- Source: `Alley_Oop_Page_27`
- Question: `On page 27, does the crowned character appear in more than one panel? Answer yes or no.`
- Answer: `yes`
- Point: `[['X2'], ['Y1']]`
- Why: repeated-entity tracking within one page

### Candidate 8

- Source: `Alley_Oop_Page_27`, panel 1 and panel 5
- Question: `On page 27, is the crowned character physically confronted by the shirtless man by the final panel? Answer yes or no.`
- Answer: `yes`
- Point: `[['X2', 'X3'], ['Y2']]`
- Why: cross-panel state transition and interaction grounding

## What To Build First

The best first MemEye-ready subset from `ComicScene` should:

- use only one comic series first, preferably `Alley_Oop`
- stay in controlled answer spaces
- avoid free-form plot summarization
- focus on `Character Tracking` and `Spatial Continuity`

Recommended first batch size:

- 15 to 30 curated QA items

Recommended answer formats:

- `yes` / `no`
- one character name
- one option letter
- one controlled position label

## Caption-Proof Expectation

Most promising for caption-proof evaluation:

- identity tracking across repeated panels
- fine-grained spatial continuity
- cross-panel interaction changes

Least promising:

- broad “what happened?” narrative questions
- scene-summary tasks whose answer can be restated from a caption or summary

## Next Step

If this direction is adopted, the next implementation step should be:

1. select 15 to 30 candidate panel groups
2. author controlled-answer QAs
3. assign `point` using the MemEye annotation guide
4. export to benchmark JSON under the same format used by `Benchmark_Pipeline/data/dialog`
