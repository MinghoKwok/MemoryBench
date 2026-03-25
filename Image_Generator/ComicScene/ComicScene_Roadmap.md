# ComicScene Roadmap

This document describes how the current `ComicScene` work should evolve from a small MemEye draft into a fuller benchmark family closer in completeness to `Home_Renovation_Interior_Design` and `Visual_Case_Archive_Assistant`.

## Goal

The goal is not to revive the old story-summary pipeline. The goal is to turn comic pages into a benchmark family that supports:

- clear story state
- persistent character and object identity
- visually grounded event change
- cross-page reasoning
- harder `Y3` synthesis without relying on hidden metadata or free-form summaries

## Phase 1

Stabilize ComicScene as a clean MemEye task family.

Targets:

- standardize everything to the current MemEye schema
- keep answers visible and controllable
- remove dependence on legacy narrative summary outputs
- keep the first set small but high quality

What to build:

- 20 to 40 benchmark-quality QA items
- primary focus on:
  - `X2 + Y1`
  - `X2 + Y2`
  - `X3 + Y2`

Success criteria:

- every item has a valid `point=[[X...],[Y...]]`
- every answer is directly observable or inferable from observable evidence
- answer formats are controlled
- hard items are retained only when the failure is real, not annotation noise

## Phase 2

Expand from continuity to explicit visual state tracking.

Core task families:

- Character Tracking
- Spatial Continuity
- Object Persistence
- Action-State Change

Typical questions:

- whether the same character reappears later
- how a character's position or situation changes across pages
- whether a salient object persists or moves
- whether a visually grounded event has already happened

Main emphasis:

- grow `Y2`
- make cross-panel and cross-page links explicit
- keep question formats stable enough for exact evaluation

## Phase 3

Introduce goals, conflict, and evolving world state.

At this stage, a comic sample should expose more than isolated continuity. It should define:

- current character goal
- current conflict or obstacle
- current state of the scene or relationship
- whether later evidence overrides an earlier interpretation

Typical questions:

- whether a later page invalidates an earlier assumption
- what the character's current problem is
- whether the new visual evidence changes the interpretation of the event

Main emphasis:

- expand `Y3`
- keep the decisive evidence visual rather than purely textual

## Phase 4

Build a small but high-value hard set.

This phase should not optimize for quantity. It should optimize for analysis value.

Hard items should be:

- visually grounded
- answerable with controlled output
- resistant to caption-only replacement
- likely to expose real model failure modes

Recommended hard-item directions:

- similar-looking character confusion
- cross-page spatial remapping
- small visual details that change event interpretation
- later evidence that forces non-monotonic revision

## Recommended Order

If work needs to be staged conservatively, prioritize in this order:

1. Character Tracking
2. Spatial Continuity
3. State Change
4. Story Synthesis

This order gives the best tradeoff between annotation stability and benchmark value.

## Practical Output Plan

The most realistic near-term plan is:

1. make `Alley_Oop` a stable 20 to 40 item benchmark subset
2. center it on character tracking and spatial continuity
3. add a smaller layer of state-change questions
4. add a limited number of `Y3` synthesis items only after the lower layers are clean

## Annotation Rules That Still Apply

All roadmap phases should continue to follow the existing MemEye rules:

- visible answers only
- no hidden generator or source metadata
- controlled short answers whenever possible
- explicit `point` labels
- selective retention of clean hard items
- future caption-proof checks for the strongest items

## Intended End State

The intended end state is a ComicScene benchmark family with a role similar to:

- `Chat_UI_Memory_Test` for identity and reference stress
- `Visual_Case_Archive_Assistant` for evidence-chain tracking
- `Home_Renovation_Interior_Design` for long-horizon multimodal synthesis

ComicScene should eventually occupy the region where visual continuity, narrative state, and evolving story memory meet.
