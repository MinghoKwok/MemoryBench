# MemEye Paper Outline (NeurIPS Datasets and Benchmarks Track)

**Title**: MemEye: A Vision-Centric Benchmark for Multimodal Agent Memory

## Abstract

Retrieval-centric limitation of existing benchmarks -> Visual Bypassability + Reasoning Shallowness -> Binocular taxonomy (X: visual granularity, Y: reasoning complexity) -> Caption-Proof protocol -> Upper-right quadrant performance collapse finding.

## 1. Introduction (~1-1.5 pages)

- **Context**: MLLMs evolving toward long-horizon interactive agents; memory is the bottleneck.
- **The Problem**: Two recurring weaknesses in current evaluation:
  - *Visual Bypassability*: tasks solvable after replacing images with captions.
  - *Reasoning Shallowness*: tasks reducible to single-hop retrieval without state tracking or non-monotonic revision.
- **Our Solution**: MemEye — a binocular evaluation framework.
  - X-axis: visual granularity ($X_1$--$X_4$, from scene gist to micro-attribute).
  - Y-axis: reasoning complexity ($Y_1$--$Y_3$, from direct retrieval to state-evolving synthesis).
- **Key Contributions**:
  1. A 4x3 binocular taxonomy with the Highest-Bottleneck annotation rule.
  2. Caption-Proof validation protocol ($\Delta = \text{Acc}_V - \text{Acc}_T$).
  3. 8 tasks under a unified life-scenario persona (Hannah Brooks), covering 464 questions.
  4. Large-scale experiments (14 methods x 4 backbones) exposing upper-right quadrant collapse.

## 2. Related Work (~0.5-1 page)

- **Multimodal Memory Benchmarks**: Compare with MemGallery, etc. Point out they cluster in $X_{1\text{-}2}$, $Y_{1\text{-}2}$ quadrant.
- **Visual Grounding & Reasoning**: We test grounding *preservation* over long horizons, not just single-turn grounding.
- **Long-Context Agents & Memory Architectures**: Paradigm shift from monotonic retrieval to non-monotonic state tracking. Covers full-context, RAG, agentic (A-Mem, MemGPT, MMA, M2A), and summarization (SimpleMem, Generative Agents) families.

## 3. The MemEye Taxonomy (~1.5 pages)

*Core figure: Figure 1 — the 4x3 binocular matrix with task placements.*

### 3.1 Visual Granularity (X-Axis) — "How deeply the agent must see"

| Level | Name | Description |
|-------|------|-------------|
| $X_1$ | Global Scene Gist | Holistic scene-level recognition; tolerant to captioning |
| $X_2$ | Entity Instance Retrieval | Instance-level identity binding across sessions |
| $X_3$ | Spatial Grounding | Topological/coordinate-sensitive layout memory |
| $X_4$ | Micro-attribute Reasoning | Subtle color, texture, OCR, tiny inscriptions |

Caption-Proof interpretation: $X_1$ (textual anchor) -> $X_2$ (cross-modal link) -> $X_3$ (spatial grounding) -> $X_4$ (fine-grained fixation). Higher X = more caption-resistant.

### 3.2 Reasoning Complexity (Y-Axis) — "How far the agent must think"

| Level | Name | Description |
|-------|------|-------------|
| $Y_1$ | Direct Retrieval | Single fact from single session |
| $Y_2$ | Compositional Linking | Cross-session/cross-modal evidence combination (monotonic) |
| $Y_3$ | State-Evolving Synthesis | Non-monotonic reasoning with belief revision |

### 3.3 The Highest-Bottleneck Annotation Rule

Each question gets a single $(X_i, Y_j)$ label — the highest level whose absence would prevent a correct answer. Justification: statistical interpretability, error attribution, empirical validation via Caption-Proof.

## 4. The MemEye Benchmark (~2 pages)

### 4.1 Unified Life Scenario: Hannah Brooks

Persona: 32-year-old freelance graphic designer, apartment renovation, remote work. Each task maps to a life facet:

| Task | Life Facet | Primary $(X, Y)$ Region | $n$ |
|------|-----------|-------------------------|-----|
| Home Renovation | Domestic | $X_3$-$X_4$, $Y_2$-$Y_3$ | 65 |
| Brand Memory | Work (Analysis) | $X_2$-$X_4$, $Y_1$-$Y_2$ | 33 |
| Card Playlog | Leisure | $X_4$, $Y_2$-$Y_3$ | 48 |
| Cartoon Entertainment | Leisure | $X_1$-$X_2$, $Y_1$-$Y_3$ | 112 |
| Multi-Scene VCAA | Organization | $X_2$-$X_4$, $Y_2$-$Y_3$ | 59 |
| Outdoor Navigation | Domestic | $X_3$, $Y_1$-$Y_2$ | 40 |
| Personal Health | Health | $X_1$-$X_4$, $Y_1$-$Y_3$ | 63 |
| Social Chat | Social | $X_2$-$X_3$, $Y_1$-$Y_2$ | 44 |

Total: 8 tasks, 464 questions (MCQ + open-ended).

### 4.2 Design Principles

- **Visual Necessity**: Assistant never describes image content; captions are minimal labels.
- **Cross-Task Coherence**: Consistent timeline and persona across tasks.
- **Complementary Memory Demands**: Tasks span different regions of the matrix.

### 4.3 Data Construction Pipeline (brief; details in Appendix)

Five stages: Visual Grounding -> Dialogue Construction -> QA Authoring -> Text-Leak Audit -> Caption-Proof Verification.

### 4.4 The Caption-Proof Protocol

- Generate dense captions for every image (using GPT-4o-class VLM).
- Replace all images with captions, re-run benchmark.
- $\Delta = \text{Acc}_V - \text{Acc}_T$: an item is visually robust only if $\Delta$ exceeds threshold.
- Protocol predicts monotonic relationship between $X$ level and $\Delta$ — empirically testable.

## 5. Experiments (~2 pages)

### 5.1 Setup

**Backbone models (4)**:
- Qwen3-VL-8B (open, via OpenRouter)
- Qwen2.5-VL-7B (open, local)
- GPT-4.1-nano (closed)
- GPT-5-mini (closed)

**Memory methods (14)**:
- Text-only (8): Full Context (T), Semantic RAG (T), A-Mem, MemGPT, MemoryOS, Reflexion, Generative Agents, EverMemOS
- Multimodal (6): Full Context (V), Semantic RAG (V), MIRIX, MMA, M2A, SimpleMem

**Metrics**: MCQ exact-match (EM); Open-ended: normalized EM, F1, BLEU-1, BLEU-2, LLM-as-judge.

### 5.2 Main Results

- Per-task accuracy table (14 methods x 8 tasks).
- Key findings:
  - No single method dominates — best method varies by task.
  - Agentic memory shines on high-density tasks (Multi-Scene: SimpleMem 0.729 vs FC 0.271).
  - Full context remains competitive on moderate-length tasks (<200 turns).

### 5.3 The "Upper-Right" Collapse

- Accuracy heatmap by $(X_i, Y_j)$ cell.
- Caption-Proof gap $\Delta$ increases monotonically with $X$ level ($+0.07$ at $X_1$ to $+0.28$ at $X_4$).
- $(X_4, Y_2)$ and $(X_4, Y_3)$ are the hardest cells — combining fine-grained visual detail with complex reasoning.

### 5.4 V-Stream vs. T-Stream (Caption-Proof Validation)

- Matched comparison: FC(V) vs FC(T), SRAG(V) vs SRAG(T).
- Confirms visual input consistently helps ($\Delta = +0.12$ average).
- Two anomalies: Multi-Scene (negative $\Delta$ due to exceptionally detailed captions) and Outdoor Navigation ($\Delta = 0$).

### 5.5 Context-Scale Stress Test

Three conditions: clue-only (oracle) -> per-dataset -> all-concat (~850 rounds). Measures cost of context noise at increasing scales.

## 6. Analysis & Case Studies (~1 page)

### 6.1 Y3 Error Patterns

- **Stale Memory Failure**: RAG frequency-voting returns outdated majority evidence (e.g., fossil room tag C-1127 appears 3x but was updated to A-209).
- **A->B->A Reversal Blindness**: Models that learn "pick latest update" fail when final state reverts to initial (e.g., Oop's king->commoner arc, doctor guidance reversal).
- **Attribute Drift**: Fine-grained visual attributes (paint colors, card details) get confused across sessions.

### 6.2 Representative Cases

- Paint-Color Reversal (Home Reno, $X_4$/$Y_3$): sage green -> terracotta pivot.
- Doctor's Guidance A->B->A (Health, $X_1$/$Y_3$): portal screenshot content only readable from image.
- Fossil Room Tag Override (Multi-Scene, $X_4$/$Y_3$): 3:1 old-vs-new evidence ratio defeats RAG.

## 7. Conclusion & Limitations (~0.5 page)

- MemEye redefines multimodal memory evaluation via binocular taxonomy + Caption-Proof.
- Upper-right quadrant remains unsolved by all current methods.
- **Limitations**:
  - Human annotation cost is high (Text-Leak Audit + Caption-Proof are manual gates).
  - Single persona (Hannah Brooks) — may not generalize to all agent deployment contexts.
  - $Y_3$ questions are inherently fewer, limiting per-cell statistical power.
  - Current caption baseline uses a single VLM; stronger future captioners may narrow $\Delta$.

---

## Writing Priority

1. **Section 3 (Taxonomy)** + **Section 4.4 (Caption-Proof)** — the formal backbone; write these first.
2. **Figure 1** — the 4x3 matrix visualization; first impression for reviewers.
3. **Section 5 (Experiments)** — fill the main results table once runs complete.
4. **Section 1 (Introduction)** — write last; every claim must have experimental backing.

## Open Items

- [ ] GPT-5-mini: no results yet — decide whether to include or drop from claims.
- [ ] MIRIX: no results in locked_results — run experiments or remove from method list.
- [ ] Qwen2.5-VL-7B: tables in main.tex have data but not in locked_results — verify provenance.
- [ ] LLM-as-judge metric: implementation TBD (`[TO-BE-FIXED]` in experiments.tex).
- [ ] EverMemOS / MemGPT: only 8 results each — need full coverage or acknowledge partial.
