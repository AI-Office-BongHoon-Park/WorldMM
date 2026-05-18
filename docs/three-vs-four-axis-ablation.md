# 3-axis vs 4-axis Ablation Report — WorldMM on EgoLife A1_JAKE DAY1

**Date:** 2026-05-18 KST
**Subject:** A1_JAKE, DAY1 (10 hours of egocentric footage, 11:00–22:00)
**Question source:** EgoLifeQA DAY1 subset, first **n = 25** questions in published order
**Reasoning model:** `chatgpt/gpt-5.4` via local LiteLLM proxy (streaming, JSON-via-prompt)
**Text embedding (for semantic + spatial indexing):** `sentence-transformers/all-MiniLM-L6-v2` (CPU, swapped in to fit the 5.6 GB-free GPU)
**Sister docs:** [`docs/spatial-memory-ablation.md`](spatial-memory-ablation.md) (synthetic + EgoLifeQA-WHERE only) · [`docs/spatial-encoding-sensor-based.md`](spatial-encoding-sensor-based.md) (architectural design for the future 5th axis / 4th-axis extension)

This report measures, for the first time on this codebase, the *real* added value of the new 4th (spatial) memory axis when the reasoning agent has the existing 3 axes (episodic + semantic + visual) available simultaneously. Earlier ablations (`docs/spatial-memory-ablation.md`) compared **episodic-only** vs **episodic+spatial**, which was a strict lower bound on the baseline.

> ⚠️ **Visual axis is a no-op in this run.** The text-only LiteLLM proxy cannot consume images, and we did not pre-compute video clip embeddings (the original EgoLife pipeline requires a vision-capable LLM + heavy VLM2Vec). The "Visual" memory is initialised empty in both configurations, so the agent sees it in the prompt but every `retrieve_from_visual` call returns nothing. This caveat is symmetrical — both runs suffer identically — but it does mean the 3-axis baseline here is in practice an **E + S** baseline, not an E + S + V baseline. See [§5](#5-limitations).

---

## 1. TL;DR

| Configuration | Correct | % |
|---|---:|---:|
| **3-axis** (episodic + semantic + visual-no-op)               | **15 / 25** | **60.0 %** |
| **4-axis** (episodic + semantic + visual-no-op + **spatial**) | **14 / 25** | **56.0 %** |
| **Δ** (4-axis − 3-axis)                                       | **−1**     | **−4.0 %p** |
| Answers differ between configs | 6 / 25 | 24 % |
| Of those: **UP** (4-axis fixes 3-axis wrong) | 2 | |
| Of those: **DN** (4-axis breaks 3-axis right) | 3 | |
| Of those: both wrong, different letters | 1 | |

**Headline:** on real EgoLifeQA DAY1 questions, adding the spatial axis is a **mild net negative** (−1 / 25). The spatial signal **is** real — it fixes 2 questions the 3-axis baseline gets wrong — but the reasoning agent also picks the spatial memory in 3 cases where it should not, and those *cost* a correct answer.

This is not a contradiction with the earlier `docs/spatial-memory-ablation.md` (which showed **+15 %p on hand-curated WHERE questions**). The earlier report deliberately filtered to questions whose answer lives in the spatial layer. This report uses the **unfiltered first 25 questions of DAY1**, which include `EntityLog`, `RelationMap`, `EventRecall`, `TaskMaster`, and `HabitInsight` types. Most of those are not WHERE questions, so the spatial layer adds noise as often as it adds signal.

---

## 2. Setup

### 2.1 Memory build artifacts

All four memories were built on the same data (A1_JAKE DAY1, 808 fine-grained captions). Stats:

| Memory | Build pipeline | Output size | Notes |
|---|---|---:|---|
| **Episodic** | `extract_episodic_triples.py` (OpenIE via proxy) + multiscale summarisation (30s / 3min / 10min / 1h) | 808 captions / 9 631 triples | full multi-scale (was 30s only in earlier report) |
| **Semantic** | `extract_semantic_triples.py` then `consolidate_semantic_memory.py` (CPU MiniLM for cosine joins) | 826 consolidated triples | embedding model swapped from Qwen3-Embedding-4B to all-MiniLM-L6-v2 to fit 5.6 GB free VRAM (see [§5](#5-limitations)) |
| **Visual** | not built (no GPU room for VLM2Vec, no proxy multimodal model) | 0 clips | always returns empty in both configs |
| **Spatial (4th axis, shipped)** | `extract_spatial_triples.py` + custom dedup (GPU-OOM-safe) | 431 unique triples | from previous report |

### 2.2 Two configurations under test

The only variable between runs is the **`memory_reasoning` prompt template** the agent sees:

| Variant | Template | Memory types in agent prompt |
|---|---|---|
| 3-axis | [`memory_reasoning_3axis.py`](../src/worldmm/llm/templates/memory_reasoning_3axis.py) (new) | `episodic` / `semantic` / `visual` |
| 4-axis | [`memory_reasoning.py`](../src/worldmm/llm/templates/memory_reasoning.py) (existing, shipped 4th axis) | `episodic` / `semantic` / `visual` / `spatial` |

Both variants use the same `WorldMemory` object architecture, identical retrieval code, identical embedding model, identical reasoning + QA LLM (`chatgpt/gpt-5.4` via proxy), and identical `max_rounds = 3`. The 3-axis variant **never sees the spatial memory exists** — its agent prompt does not list it.

To run the ablation, `WorldMemory.__init__` was extended with a `reasoning_template_name` parameter (cherry-picked into [`src/worldmm/memory/memory.py`](../src/worldmm/memory/memory.py)) so the two templates can be selected at construction.

### 2.3 Question set

The first 25 questions of `EgoLifeQA_A1_JAKE.json` that have `query_time.date == "DAY1"`, kept in published order. Type distribution:

| Type | n |
|---|---:|
| EntityLog | 8 |
| RelationMap | 7 |
| EventRecall | 4 |
| HabitInsight | 3 |
| TaskMaster | 3 |
| **Total** | **25** |

No filter on whether the question is "spatial-flavored". This is the **unfiltered general-purpose** result — what a paper-table number should look like for a fair 3-vs-4-axis comparison.

---

## 3. Results

### 3.1 Overall

```
3-axis (E+S+V no-op):       15/25 (60.0%)
4-axis (E+S+V no-op + Sp):  14/25 (56.0%)
delta:                       -1  (-4.0%p)
answers differ:              6/25  (UP=2 DN=3 both-wrong=1)
```

### 3.2 By question type

| Type | n | 3-axis | 4-axis | Δ | differs | UP | DN |
|---|---:|---:|---:|---:|---:|---:|---:|
| **EntityLog** | 8 | 6 | 5 | **−1** | 2 | 0 | 1 |
| **EventRecall** | 4 | 2 | 2 |  0 | 2 | 1 | 1 |
| **HabitInsight** | 3 | 1 | 2 | **+1** | 1 | 1 | 0 |
| **RelationMap** | 7 | 5 | 4 | **−1** | 1 | 0 | 1 |
| **TaskMaster** | 3 | 1 | 1 |  0 | 0 | 0 | 0 |

Observations:
- **HabitInsight is the only category where spatial helps** (+1 / 3). Habits are often "always at place X" / "every morning in place Y" — the spatial layer's place lookups directly fit.
- **EntityLog and RelationMap lose 1 each**. These are *who-did-what* questions; the spatial layer's place tokens occasionally bait the reasoning agent into a place-based answer even when the question is about agency.
- **EventRecall is a wash**: 1 UP, 1 DN.
- **TaskMaster** (planning / what-next) shows no differences; the spatial layer is simply irrelevant here.

### 3.3 Question-level differences (6 / 25)

#### UP (spatial fixes a wrong 3-axis answer)

**Q20 — `EventRecall`** "We are currently discussing the details of the puzzle. When was this discussed before?"
- gold = D, 3a = A (wrong), **4a = D ✅**
- Why spatial helped: the place tokens for the puzzle-table scene tied the *previous* discussion's spatial context to a specific earlier timestamp the spatial PPR could surface.

**Q21 — `HabitInsight`** "Who always sits on that pony hair this morning"
- gold = A, 3a = C (wrong), **4a = A ✅**
- Why spatial helped: `(<person>, on, pony_hair)`-type triples are explicitly stored in the spatial layer; episodic captions describe the *action* but rarely state the seat assignment as a *fact*.

#### DN (spatial breaks a correct 3-axis answer)

**Q1 — `EntityLog`** "Who used the screwdriver first?"
- gold = B, 3a = B ✅, **4a = C ❌**
- Why spatial hurt: the spatial layer surfaced `(screwdriver, on, table)` and `(table, contains, screwdriver)` triples, which the agent over-weighted at the expense of the episodic event "Alice tightens screws" that pins the gold answer.

**Q12 — `RelationMap`** "When I handed out the charging cables just now, who had already started charging?"
- gold = D, 3a = D ✅, **4a = A ❌**
- Why spatial hurt: spatial returns `(<person>, near, charging_cable)` and `(charging_cable, on, table)` for *several* people — the reasoning agent picked the spatially closest, but the actual answer is about action sequence, not proximity.

**Q14 — `EventRecall`** "Who is missing compared to when we first started the puzzle?"
- gold = A, 3a = A ✅, **4a = B ❌**
- Why spatial hurt: comparing two scenes by *who was present* is an episodic-temporal task; the spatial layer's snapshot view of "who is at the puzzle table now" actively misleads.

#### Both wrong, different letters (1 / 25)

**Q16 — `EntityLog`** "Who was the first to move the puzzle board on the table?"
- gold = B, 3a = C, 4a = A — different answers, both wrong. Retrieval (keyword overlap) failed to surface the decisive moment in either configuration.

---

## 4. Methodology — details for reviewer

### 4.1 What is identical between the two runs

1. The 4 memory objects (episodic, semantic, visual, spatial) are constructed with identical data files in both runs. The 3-axis run is **not** denied access to the spatial data — it simply has no prompt vocabulary to choose it. (This matters for one edge case: if the agent's `selected_memory.memory_type` accidentally said `spatial` in the 3-axis run because of prompt leakage, the dispatch would still execute; in practice 25 / 25 questions stayed within the documented 3 types.)
2. Same retrieval top-K per memory: `episodic_top_k = 3`, `semantic_top_k = 10`, `visual_top_k = 3`, `spatial_top_k = 8`.
3. Same `max_rounds = 3`, `max_errors = 3`.
4. Same QA template (`qa_egolife`), same reasoning + respond LLM (`chatgpt/gpt-5.4`).
5. Same embedding model (`all-MiniLM-L6-v2`, CPU) and same disk caches *for the LLM call layer*. **HippoRAG cache directories are isolated per variant** (`.cache/episodic_memory_3axis` vs `.cache/episodic_memory_4axis`) to avoid cross-variant cache pollution — without this isolation the first attempt of the run failed with `len(chunk_to_rows): 265, len(ner_results_dict): 274` mid-loop.
6. Same question ordering (questions processed in published EgoLifeQA order, which is monotonic in `query_time`).

### 4.2 What is different

1. **Reasoning template only**: `memory_reasoning_3axis` (no Spatial entry) vs the shipped `memory_reasoning` (4 entries).

### 4.3 Question dispatch

`WorldMemory.answer(query, choices, until_time)` is called per question. The reasoning loop runs up to `max_rounds = 3`; the agent decides each round to either *search* (and which memory) or *answer*. When it answers, the QA template renders all accumulated context and produces a single-letter prediction. We compare that letter against the gold.

### 4.4 Fairness against the previous `docs/spatial-memory-ablation.md`

| Comparison | Previous report | This report |
|---|---|---|
| Baseline | episodic-only, top-50 by keyword overlap | episodic + semantic + visual (empty), full iterative reasoning |
| Treatment | + spatial-text top-25 | + spatial-text via the shipped 4th axis |
| Retrieval | keyword overlap | HippoRAG (episodic), PPR (semantic + spatial), shipped retrievers |
| Question set | (a) 20 synthetic WHERE-only (b) 22 EgoLifeQA WHERE-filtered | first 25 unfiltered DAY1 EgoLifeQA |
| Δ measured | (a) +15.0 %p (b) +4.5 %p | **−4.0 %p** |

The change of sign is **not** evidence that the previous report was wrong — it is evidence that **filtering questions changes the conclusion**. On WHERE-grounded questions, spatial helps. On a representative slice of EgoLifeQA, it slightly hurts. Both can be true at once.

---

## 5. Limitations

1. **n = 25 is small.** A wider follow-up (n = 60+) is running in the background; this document will be updated when those numbers land. Treat the −4.0 %p as directional.
2. **Visual axis is a no-op.** The text-only LiteLLM proxy plus absent visual-clip embeddings means both configurations effectively use only **E + S** (and S only on the 4-axis run). This is fair *between* the two configurations being compared, but it is **not** a comparison against the WorldMM paper's full E + S + V baseline.
3. **Embedding model is smaller.** We swapped Qwen3-Embedding-4B for `all-MiniLM-L6-v2` (384-d) because the larger model OOMs on the 5.6 GB-free dev box. This affects retrieval quality on both sides equally but lowers absolute scores.
4. **`max_rounds = 3` instead of the shipped default 5.** Lower budget hurts both configurations equally but caps how deeply the agent can iterate; spatial may benefit more than baseline from more rounds.
5. **Run-to-run variance.** A first-pass smoke run (n = 3) had Q1 as a `DN` flip; the n = 25 run has Q1 as a `DN` flip too — but the n = 25 v1 first attempt had Q1 as `UP`. The proxy is stochastic and the iterative reasoning context evolves across rounds; ≥ 2-seed averaging would tighten the numbers.
6. **No EgoLifeQA filtering.** Most DAY1 EgoLifeQA questions are *not* WHERE questions; this is a strength (fair) but it also means spatial cannot help by design on many questions.
7. **Visual axis no-op may flatter both sides.** With visual properly built, the reasoning agent would have one more memory to choose from, possibly drawing reasoning effort *away* from spatial (helping baseline relatively) or *toward* it via richer prompt context. Cannot be measured here.

---

## 6. Reproduction

```bash
# 1. Build the prerequisite memories (multiscale episodic, semantic, spatial)
bash script/3_build_memory.sh --step all --person A1_JAKE --model chatgpt-gpt-5.4

# 2. Run the 3-vs-4-axis ablation harness
WORLDMM_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
WORLDMM_EMBED_DEVICE=cpu \
uv run python eval/three_vs_four_axis_ablation.py \
    --max-n 25 --max-rounds 3 \
    --output output/three_vs_four_n25.json
```

Output:
- `output/three_vs_four_n25.json` — per-question predictions for both configs.
- Console log includes per-question summary lines and the totals block.

The harness is at [`eval/three_vs_four_axis_ablation.py`](../eval/three_vs_four_axis_ablation.py); the 3-axis prompt is at [`src/worldmm/llm/templates/memory_reasoning_3axis.py`](../src/worldmm/llm/templates/memory_reasoning_3axis.py).

---

## 7. Bottom line for the write-up

> On a representative unfiltered slice of EgoLifeQA DAY1 (n = 25), adding the closed-vocabulary spatial memory as a 4th axis is a **mild net negative (−4.0 %p)**: it recovers 2 questions the 3-axis baseline gets wrong, but the reasoning agent also picks the spatial branch in 3 cases where it should not have, and those *cost* a correct answer. The signal is sharply category-dependent: **+1 / 3 on `HabitInsight`**, **−1 / 7 on `RelationMap`**, **−1 / 8 on `EntityLog`**, neutral on `EventRecall` and `TaskMaster`. This is not in conflict with the earlier synthetic-WHERE finding of +15 %p — that one filtered to questions whose answers live in the spatial layer; this one does not. **The honest paper-level number for spatial-as-shipped is therefore: positive on WHERE-grounded queries, negative on relation / agency queries, and the choice of question filter dominates the headline.**
>
> The natural next move is a **routing fix in the reasoning prompt**: the agent currently treats the spatial axis as freely selectable; constraining it to questions whose surface form contains WHERE / location cues (or letting the agent choose spatial only as a *second-round* refinement after episodic) should preserve the +UP wins while collapsing the −DN losses. This is captured as the v2 extension plan in [`docs/spatial-encoding-sensor-based.md`](spatial-encoding-sensor-based.md) §7.
