# MemEye Paper Outline v2 (NeurIPS Datasets and Benchmarks Track)

**Title**: MemEye: A Vision-Centric Benchmark for Multimodal Agent Memory

**Principle**: Benchmark section proves the dataset is diagnostically valid; Experiments section proves what memory systems fail at.

## Abstract

Agent memory requires preserved visual evidence + reasoning over memory state. Current benchmarks test neither reliably. MemEye: binocular taxonomy (X: visual granularity, Y: reasoning depth). Ablation-driven construction validates both axes. 13 methods × 5 backbones expose architectural failure modes by (X, Y) coordinate.

## 1. Introduction (~1 page)

Central claim: agent memory needs both **preserved visual evidence** and **reasoning over memory state**.

- Agent memory is not just "remembering text" — real agents must retain visual details across sessions.
- Visual evidence has granularity: scene gist survives captioning, but instance identity and pixel-level attributes do not.
- Memory is not just retrieval: agents must associate distributed evidence and synthesize evolving state.
- MemEye makes both dimensions explicit via a binocular taxonomy: X-axis (visual granularity) and Y-axis (reasoning depth).

**Contributions**:
1. A vision-centric long-term memory benchmark (371 QAs, 8 tasks, MCQ + open mirrored).
2. A binocular (X, Y) taxonomy that localizes failures along visual and reasoning bottlenecks.
3. Ablation-driven benchmark construction: every question passes caption-proof, option-bias, text-leak, and oracle-evidence gates.
4. Systematic evaluation of 13 memory architectures × 5 backbones, exposing architectural failure modes by (X, Y) coordinate.

## 2. Related Work (~0.5 page)

Three gaps:
- **Memory benchmarks lack visual evidence granularity.** Existing benchmarks cluster in X1–X2, Y1–Y2.
- **Multimodal dialogue benchmarks lack long-term evolving memory.** Single-turn VQA doesn't test cross-session state tracking.
- **Memory systems lack diagnostic evaluation.** No benchmark separately measures visual preservation vs. temporal reasoning. MemEye fills this via the binocular taxonomy.

## 3. The MemEye Benchmark (~3 pages) — core section

### 3.1 A Binocular Taxonomy of Visual Memory

Why "MemEye": one eye sees visual granularity, the other sees memory reasoning depth. The (X, Y) coordinate is an observation lens, not a task category. Highest-Bottleneck Rule: each question gets one (Xi, Yj).

### 3.2 X-axis: Granularity of Visual Evidence — "what must survive"

| Level | Name | What must be preserved |
|-------|------|------------------------|
| X1 | Scene-level | Holistic scene gist; caption-recoverable |
| X2 | Region-level | Localized regions, functional areas, grouped context |
| X3 | Instance-level | Specific object/person identity binding |
| X4 | Pixel-level | Color, texture, OCR, micro-attributes |

### 3.3 Y-axis: Reasoning over Memory — "how evidence is used"

| Level | Name | Core demand |
|-------|------|-------------|
| Y1 | Atomic Retrieval | Retrieve one directly relevant fact |
| Y2 | Relational Association | Associate distributed but consistent evidence |
| Y3 | Evolutionary Synthesis | Synthesize valid state under updates/conflicts/overrides |

Emphasize: Y measures how retrieved evidence is used, not retrieval difficulty.

### 3.4 Dataset Construction

371 questions, 8 tasks, 6 life-scenario domains. MCQ/open mirrored. Clue rounds annotated. 4-rotation MCQ debiasing.

| Task | Domain | n |
|------|--------|---|
| Home Renovation | Domestic | 52 |
| Brand Memory | Work | 29 |
| Card Playlog | Leisure | 48 |
| Cartoon Ent. | Leisure | 76 |
| Multi-Scene VCAA | Organization | 50 |
| Outdoor Nav. | Domestic | 28 |
| Personal Health | Health | 51 |
| Social Chat | Social | 37 |
| **Total** | | **371** |

### 3.5 Benchmark Validation — ablation-driven data quality gates

**This is the key methodological contribution. Each gate proves a property of the dataset.**

**Gate 1: Text-Leak Audit.**
Grep answer keywords against dialogue text; verify assistant responses are generic. No question answerable from dialogue text alone.

**Gate 2: Option-Bias Rejection (MCQ).**
Question-only ablation with 4-rotation MCQ under two model families (GPT-4.1-nano + Gemini-2.5-FL). Remove questions with debiased EM ≥ 0.75 under both. Result: 63/434 removed → 371 remain; QO baseline ≈ 0.30.

Case study: Outdoor Navigation R1 route-type labels.

**Gate 3: Caption-Proof Validation (X-axis validity test).**
Replace images with dense captions; compare FC(V) vs FC(T). Δ increases from X1 to X4, confirming X-axis captures visual irreplaceability.

**Gate 4: Oracle-Evidence Validation (Y-axis validity test).**
Provide gold clue rounds + original images (perfect retrieval). Open-ended LLM-Judge still decreases: Y1 → Y2 → Y3 (0.673 → 0.601 → 0.558). Since retrieval is solved, the drop is purely reasoning difficulty.

Additional evidence: method ranking changes across Y levels (SRAG wins Y1; FC(T) overtakes FC(V) at Y3).

## 4. Experiments (~2.5 pages) — focused on memory systems

Setup: 13 methods × 5 backbones. MCQ (debiased EM) + Open (LLM-Judge with gpt-5.2).

### 4.1 Main Results: Reading the MemEye Matrix

Table: GPT-5.4-mini full (X, Y) matrix.

Five findings:
1. **Overall performance far from saturation.** Best ≈ 0.63 EM, ≈ 0.49 Judge.
2. **Native visual memory helps when X increases.** Δ(V-T) near zero at X1/X2, positive at X3/X4.
3. **Y-axis changes the bottleneck.** Y1/Y2: retrieval wins. Y3: full-context and text memory catch up.
4. **Lower-right (X4, Y3) is the strongest stress test.** Requires both pixel-level preservation and state-evolving synthesis.
5. **Different architectures fail in different regions.** FC: attention dilution. SRAG: retrieval miss at Y2, stale evidence at Y3. MMA: strong at Y1 fine-grained, degrades at Y3. M2A: state-aware at Y3 but lossy memory writing.

### 4.2 Mixed-Memory Scaling

1-task → 2-task pairs → 4-task quads.
- FC(V) degrades (MCQ −21%, Open −36%).
- SRAG(V) and MMA are stable.
- Conclusion: full context is not a scalable memory strategy.

### 4.3 Architectural Patterns

Brief per-method analysis. Key: no single method dominates; best method varies by (X, Y) coordinate.

## 5. Conclusion (~0.3 page)

- MemEye shows visual memory is not solved by captions (X-axis).
- Memory is not solved by retrieval alone (Y-axis).
- Future systems need to preserve visual evidence AND reason over evolving state.
- The binocular taxonomy provides a diagnostic lens for measuring progress.

## Terminology

- Do NOT call benchmark validation "ablation study." Use: **Benchmark Validation** / **Diagnostic Construction Protocol**.
- X levels: Scene-level, Region-level, Instance-level, Pixel-level.
- Y levels: Atomic Retrieval, Relational Association, Evolutionary Synthesis.
- Caption-Proof = X-axis validity test.
- Oracle-Evidence = Y-axis validity test.

## Status (2026-04-28)

- [x] 371 QAs, 8 tasks, MCQ/open mirrored — done
- [x] MCQ rotation + question-only filtering — done
- [x] Locked results: gpt-4.1-nano, gemini-2.5-flash-lite, gpt-5.4-mini — done
- [x] 13 methods × 3 models — done (nano/gemini/5.4-mini)
- [x] LLM-as-judge (gpt-5.2) for all open results — done
- [x] Context-scale ablation (2-task pairs + 4-task quads) — done
- [x] Caption-proof validation — done
- [x] Oracle-evidence Y-axis validation (clue-only) — done
- [x] Outdoor Navigation R1 fix — done
- [ ] Qwen3-VL-8B / Qwen2.5-VL-7B results — need re-run on new dataset
- [ ] Paper writing: introduction, conclusion
