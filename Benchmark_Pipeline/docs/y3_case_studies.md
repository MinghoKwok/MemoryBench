# Y3 Case Studies — State-Evolving Synthesis Examples

This document collects curated Y3 (non-monotonic, state-evolving) question examples
across MemEye datasets. Each case demonstrates a clear **assertion → override → final
state** chain where a model must track belief revision across multiple sessions.

These cases are intended for use in the paper's Qualitative Analysis / Case Study section.

---

## Case 1: Paint Color Reversal (Home Renovation)

**Dataset**: Home_Renovation_Interior_Design
**Question ID**: Q64
**Cell**: [X4, Y3]

### State Evolution Chain

| Session | Event | Visual Evidence |
|---------|-------|-----------------|
| D3 (2025-03-20) | Hannah tests **sage green** paint — multiple swatch images, close-ups on wall | D3_IMG_002 (green swatch card), D3_IMG_004 (green paint on wall) |
| D6 (2025-04-28) | **Color pivot**: Hannah says "New inspiration after the color pivot", tests **terracotta** paint on wall | D6_IMG_002 (warm paint options), D6_IMG_004 (terracotta swatch on wall) |
| D9 (2025-07-18) | Terracotta accent wall visible in living room with furniture placed | D9_IMG_005 (tan leather sofa against terracotta wall) |
| D10 (2025-08-10) | Final living room — terracotta accent wall confirmed | D10_IMG_001 (final living room, terracotta wall) |

### Non-Monotonic Structure

```
D3:  belief = "sage green is the chosen wall color"
         ↓  (D6 pivot)
D6:  belief overridden → "terracotta is the new direction"
         ↓  (D9-D10 confirmation)
D10: final state = terracotta accent wall
```

A model that only attends to the earliest paint-testing sessions (D3) would conclude
sage green. It must detect the D6 pivot and track through D9-D10 to reach the correct
final state.

### Question

> During the renovation, Hannah tested multiple wall paint colors across different
> sessions. Which tested color was ultimately NOT used in the finished living room?
>
> A. **Sage green** ← correct
> B. Terracotta
> C. Navy blue
> D. No painting was done in the living room

### Why Y3

- **Early commitment**: D3 dedicates an entire session (7 rounds, 6 images) to sage green testing
- **Explicit override**: D6:1 user says "after the color pivot" — non-monotonic signal
- **Question forces final-state output**: asks "which was NOT used" — requires knowing both the abandoned color and the adopted color
- **Single-image insufficient**: looking at D10:1 alone shows terracotta walls, but doesn't reveal sage green was ever tested

### Caption-Proof Assessment

- Text-only methods can infer terracotta from D6/D9/D10 captions ("terracotta paint swatch", "warm terracotta wall")
- But the question asks which color was **abandoned**, requiring knowledge of D3's sage green — which captions describe only as "gradient of green shades" and "sage green paint tested on wall"
- Expected Δ: medium (text-only can partially reconstruct, but the override reasoning adds difficulty)

---

## Case 2: Doctor's Guidance A→B→A Reversal (Personal Health)

**Dataset**: Personal_Health_Dashboard_Assistant
**Question ID**: Q61
**Cell**: [X1, Y3]

### State Evolution Chain

| Session | Event | Visual Evidence |
|---------|-------|-----------------|
| D3 (2025-04-05) | Dr. Ramirez portal note #1: **"pair carbs with protein or fiber"** | D3_IMG_001, D3_IMG_002 (portal message screenshots) |
| D3:3 | Maya in dialog: "Dr. Elena Ramirez said to be more deliberate with breakfast and to watch the pattern" | (text) |
| D6 (2025-04-12) | Dr. Ramirez portal note #2: **"managing small carb snacks around workouts"** — different focus | D6_IMG_001 (updated portal note screenshot) |
| D6:2 | Maya in dialog: **"The note feels a little different, so I want to be careful with it"** | (text — explicit acknowledgment of change) |
| D11 (2025-04-24) | Dr. Ramirez portal note #3: **"continue pairing carbs with protein or fiber"** — returns to D3 theme | D11_IMG_001, D11_IMG_002 (follow-up portal screenshots) |

### Non-Monotonic Structure (Double Reversal)

```
D3:   belief = "guidance is: pair carbs with protein/fiber"
          ↓  (D6 update)
D6:   belief overridden → "guidance changed to: workout carb management"
          ↓  (D11 reaffirmation)
D11:  belief overridden again → "guidance returned to: pair carbs with protein/fiber"
```

This is an **A → B → A** pattern — the hardest Y3 variant because:
- A model anchoring on D3 alone gets the right answer (A) but for the wrong reason
- A model that correctly tracks D6's update but misses D11 answers B (wrong)
- Only a model tracking all three notes answers A for the right reason

### Question

> Across all doctor portal messages Maya received during the month, what is the focus
> of the MOST RECENT guidance from Dr. Ramirez?
>
> A. **Continue pairing carbs with protein or fiber** ← correct
> B. Manage carb snacks specifically around workouts
> C. Reduce overall carbohydrate intake
> D. All three messages communicated the exact same advice

### Why Y3

- **Two explicit overrides**: D3→D6 (topic shift) and D6→D11 (topic return)
- **Maya acknowledges the shift** in D6:2 dialog ("feels a little different")
- **A→B→A trap**: models that learn "always pick the latest update" still fail if they stop at D6

### Caption-Proof Assessment

- D3 captions: "doctor advises the patient to keep pairing carbohydrates with protein or fiber" — leaks D3 content
- D6 caption: "doctor advising a patient on managing small carb snacks around workouts" — leaks D6 content
- D11 captions: **"A patient views a follow-up nutrition message from their doctor"** — does NOT leak D11's actual content
- Text-only methods know D3=A and D6=B, but cannot determine D11's content → likely answer B (wrong)
- Multimodal methods can read D11 portal screenshot → answer A (correct)
- **Expected Δ: large** — this is an ideal caption-proof Y3 specimen

---

## Case 3: Snack Pattern Break and Return (Personal Health)

**Dataset**: Personal_Health_Dashboard_Assistant
**Question ID**: Q62
**Cell**: [X1, Y3]

### State Evolution Chain

| Session | Event | Visual Evidence |
|---------|-------|-----------------|
| D4 (2025-04-08) | Deliberate pre-workout snack: banana + crackers + peanut butter | D4_IMG_004 (packed snack photo) |
| D6 (2025-04-12) | Deliberate paired snack: banana + yogurt | D6_IMG_004 (banana and yogurt photo) |
| D8 (2025-04-17) | **Pattern break**: granola bar + fruit cup; Maya says "I was being a little looser today" | D8_IMG_001 (granola bar + fruit cup) |
| D9 (2025-04-19) | Maya revises: "Yesterday's read may have been a little loose" | (text — explicit revision of D8 assessment) |
| D11 (2025-04-24) | **Return to pattern**: yogurt + fresh berries (bedtime snack) | D11_IMG_005 (yogurt bowl with berries) |

### Non-Monotonic Structure

```
D4-D6:  pattern = "deliberate, paired snacks (yogurt, banana, protein)"
            ↓  (D8 break)
D8:     pattern broken → "looser packaged snack"
            ↓  (D9 acknowledgment + D11 return)
D11:    pattern restored → "yogurt with fresh berries"
```

### Question

> Maya's snack choices showed a deliberate pattern early on, were disrupted once
> mid-month, and then evolved further by month-end. What type of snack appears in
> her MOST RECENT upload?
>
> A. Packaged bar snack (granola bar or nut bar)
> B. **Yogurt with fresh berries** ← correct
> C. Banana with peanut butter and crackers
> D. No snack was logged after mid-month

### Why Y3

- **Established pattern** (D4/D6): deliberate pairings
- **Explicit break** (D8): Maya self-reports "looser" — creates a belief that snacking discipline may have shifted
- **Return** (D11): visual evidence of yogurt + berries, back in line with early pattern
- A model anchoring on the D8 disruption would answer A; must track through D11 to answer B

### Caption-Proof Assessment

- D8 caption: "Granola bar and fruit cup" — text-only knows D8 was a packaged snack
- D11 caption: "A bowl of creamy yogurt topped with fresh berries" — text-only CAN infer B
- Expected Δ: medium (text-only can reconstruct the return from captions, but the full pattern-break-return chain adds reasoning difficulty)

---

## Case 4: Fossil Room Tag Silent Override (Multi-Scene VCAA)

**Dataset**: Multi-Scene_Visual_Case_Archive_Assistant
**Question ID**: Q55
**Cell**: [X4, Y3]

### State Evolution Chain

| Session | Event | Visual Evidence |
|---------|-------|-----------------|
| S7:R2 | Fossil room display case shows tag **"C-1127"** | S7_IMG (fossil room) |
| S7:R5 | Same room, tag still "C-1127" | S7_IMG |
| S7:R7 | Third image, tag still "C-1127" | S7_IMG |
| S9:R1 | Same fossil room — tag now reads **"A-209"** | S9_IMG |

### Why Y3 (not RAG-solvable)

RAG retrieval for "fossil room tag" hits 4 images: 3× "C-1127" and 1× "A-209". A frequency-voting RAG system answers "C-1127" (3:1 majority). Only temporal logic — understanding S9 is later than S7 — yields the correct current answer "A-209". This is the **classic RAG failure mode for Y3**: old evidence outnumbers new evidence.

---

## Case 5: Object Migration — Brass Compass (Multi-Scene VCAA)

**Dataset**: Multi-Scene_Visual_Case_Archive_Assistant
**Question ID**: Q51
**Cell**: [X2, Y3]

### State Evolution Chain

| Session | Event |
|---------|-------|
| S7:R2 | Brass compass visible on **left side of fossil display case** |
| S7:R5 | Compass **absent** from fossil case |
| S8:R4/R6 | Compass appears on **restoration table** |
| S8:R8 | Dialog explicitly confirms restoration table as strongest candidate location |

### Why Y3

Retrieval for "brass compass" returns S7 (fossil case) and S8 (restoration table). Without temporal ordering, the system cannot determine which is "current." The dialog in S8:R8 helps but is itself session-bound — a RAG system must still rank S8 above S7 temporally.

---

## Case 6: Oop's Status Arc — Commoner → King → Commoner (ComicScene)

**Dataset**: ComicScene_Alley_Oop_V3_Dev
**Question ID**: Q51
**Cell**: [X1, Y3]

### State Evolution Chain

| Session | Event |
|---------|-------|
| D1 | Oop is an **ordinary citizen/rescuer** — saves drowning king |
| D27 | King offers Oop the throne — Oop **becomes temporary king** |
| D28-D32 | 5 sessions of Oop suffering as king (pelted, labor, ulcers) |
| D33 | Oop **returns the crown**, goes back to being ordinary citizen |

### Why Y3 (A→B→A double reversal)

Retrieval for "Oop king" hits 6 sessions (D27-D32) where Oop IS king, vs 1 session (D33) where he returns the crown. RAG by evidence volume says "Oop is king." Only tracking the full temporal arc reveals the final reversion. Same A→B→A pattern as Case 2 (doctor's guidance).

---

## Case 7: Paul Bunyan's Circus Career Reversal (ComicScene)

**Dataset**: ComicScene_Alley_Oop_V3_Dev
**Question ID**: Q50
**Cell**: [X1, Y3]

### State Evolution Chain

| Session | Event |
|---------|-------|
| TC3 | Bunyan defeats Matto Grasso, **becomes the circus strongman** |
| TC4-TC7 | Bunyan performs, fights villains, saves circus — all as strongman |
| TC8 | Bunyan **refuses the permanent offer and leaves** for the North Woods |

### Why Y3

5 sessions establish "Bunyan = circus strongman." 1 session (TC8) overrides with departure. RAG sees 5:1 evidence ratio favoring "still working at circus." Temporal logic is the only way to reach "he left."

---

## Summary: Y3 Design Principles Extracted from Cases

1. **Assertion before override**: True Y3 requires an explicit or strongly implicit early commitment (not just "showing options")
2. **Force final-state output**: Questions should ask "what IS the current state" — never "did it change" (which scaffolds the revision)
3. **A→B→A is harder than A→B**: Double reversals penalize models that simply learn "pick the latest update"
4. **Caption-Proof sweet spot**: The override evidence should be in images whose captions don't fully describe the content (e.g., portal screenshots with vague captions)
5. **Dialog acknowledgment helps**: When the user explicitly notes a change ("the note feels a little different", "I was being looser"), it strengthens the Y3 signal without requiring dialog augmentation
