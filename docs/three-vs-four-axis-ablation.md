# 3-axis vs 4-axis Ablation Report — WorldMM on EgoLife A1_JAKE DAY1

**Date:** 2026-05-18 KST
**Subject:** A1_JAKE, DAY1 (10 hours of egocentric footage, 11:00–22:00)
**Question source:** EgoLifeQA DAY1 subset, first **n = 60** questions in published order (≈ 60 % of the 102 DAY1 questions)
**Reasoning model:** `chatgpt/gpt-5.4` via local LiteLLM proxy (streaming, JSON-via-prompt)
**Text embedding (for semantic + spatial indexing):** `sentence-transformers/all-MiniLM-L6-v2` (CPU, swapped in to fit the 5.6 GB-free GPU)
**Sister docs:** [`docs/spatial-memory-ablation.md`](spatial-memory-ablation.md) (synthetic + EgoLifeQA-WHERE only) · [`docs/spatial-encoding-sensor-based.md`](spatial-encoding-sensor-based.md) (architectural design for the future 5th axis / 4th-axis extension)

This report measures, for the first time on this codebase, the *real* added value of the new 4th (spatial) memory axis when the reasoning agent has the existing 3 axes (episodic + semantic + visual) available simultaneously. Earlier ablations (`docs/spatial-memory-ablation.md`) compared **episodic-only** vs **episodic+spatial**, which was a strict lower bound on the baseline.

> ⚠️ **Visual axis is a no-op in this run.** The text-only LiteLLM proxy cannot consume images, and we did not pre-compute video clip embeddings (the original EgoLife pipeline requires a vision-capable LLM + heavy VLM2Vec). The "Visual" memory is initialised empty in both configurations, so the agent sees it in the prompt but every `retrieve_from_visual` call returns nothing. This caveat is symmetrical — both runs suffer identically — but it does mean the 3-axis baseline here is in practice an **E + S** baseline, not an E + S + V baseline. See [§5](#5-limitations).

---

## 1. TL;DR

| Configuration | Correct | % |
|---|---:|---:|
| **3-axis** (episodic + semantic + visual-no-op)               | **37 / 60** | **61.7 %** |
| **4-axis** (episodic + semantic + visual-no-op + **spatial**) | **36 / 60** | **60.0 %** |
| **Δ** (4-axis − 3-axis)                                       | **−1**     | **−1.7 %p** |
| Answers differ between configs | 10 / 60 | 16.7 % |
| Of those: **UP** (4-axis fixes 3-axis wrong) | 2 | |
| Of those: **DN** (4-axis breaks 3-axis right) | 3 | |
| Of those: both wrong, different letters | 5 | |

**Headline:** on real EgoLifeQA DAY1 questions, adding the spatial axis is **essentially neutral** (−1 / 60 ≈ −1.7 %p, well within run-to-run variance). The spatial signal **is** real — it fixes 2 questions the 3-axis baseline gets wrong — but the reasoning agent also picks the spatial memory in 3 cases where it should not, and those *cost* a correct answer. The remaining 5 differing answers are "both wrong with different letters" — spatial reroutes a wrong guess to a different wrong guess.

> A first-pass smaller sample at **n = 25** had measured Δ = **−4.0 %p**; at **n = 60** the gap narrows to **−1.7 %p**, suggesting the apparent regression at small n was sampling noise around an actually-neutral mean. Larger n (full 102 DAY1 Qs, multi-seed) would tighten further.

This is not a contradiction with the earlier `docs/spatial-memory-ablation.md` (which showed **+15 %p on hand-curated WHERE questions**). The earlier report deliberately filtered to questions whose answer lives in the spatial layer. This report uses the **unfiltered first 60 questions of DAY1**, which include `EntityLog`, `RelationMap`, `EventRecall`, `TaskMaster`, and `HabitInsight` types. Most of those are not WHERE questions, so the spatial layer adds noise as often as it adds signal.

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

The first 60 questions of `EgoLifeQA_A1_JAKE.json` that have `query_time.date == "DAY1"`, kept in published order. Type distribution:

| Type | n |
|---|---:|
| EntityLog | 16 |
| RelationMap | 13 |
| EventRecall | 12 |
| TaskMaster | 10 |
| HabitInsight | 9 |
| **Total** | **60** |

No filter on whether the question is "spatial-flavored". This is the **unfiltered general-purpose** result — what a paper-table number should look like for a fair 3-vs-4-axis comparison.

---

## 3. Results

### 3.1 Overall

```
3-axis (E+S+V no-op):       37/60 (61.7%)
4-axis (E+S+V no-op + Sp):  36/60 (60.0%)
delta:                       -1  (-1.7%p)
answers differ:              10/60  (UP=2 DN=3 both-wrong-different=5)
```

### 3.2 By question type

| Type | n | 3-axis | 4-axis | Δ | differs | UP | DN |
|---|---:|---:|---:|---:|---:|---:|---:|
| **EntityLog** | 16 | 10 | 11 | **+1** | 6 | 2 | 1 |
| **EventRecall** | 12 | 7 | 7 |  0 | 1 | 0 | 0 |
| **HabitInsight** | 9 | 7 | 6 | **−1** | 1 | 0 | 1 |
| **RelationMap** | 13 | 6 | 6 |  0 | 1 | 0 | 0 |
| **TaskMaster** | 10 | 7 | 6 | **−1** | 1 | 0 | 1 |

Observations:
- **EntityLog is the only category where spatial helps in this larger sample** (+1 / 16, with 2 UP / 1 DN). At n = 25 EntityLog was the loudest *loser*; at n = 60 it flipped to a winner. This is the cleanest evidence that the n = 25 result was **dominated by sample variance**, not a real type-level signal.
- **RelationMap is now flat** (0 / 13, 1 differ "both wrong"). At n = 25 it had been a loser. Again — sample variance.
- **HabitInsight switched sign too**: was +1 / 3 at n = 25, now −1 / 9. The 3-axis baseline got two new correct HabitInsight answers that 4-axis missed.
- **TaskMaster** (planning / what-next) showed 0 / 3 at n = 25 (no differs); at n = 60 it picked up one DN.
- **EventRecall**: 0 / 12 net, only 1 differ (both wrong with different letters).

Across all 5 types the absolute Δ at n = 60 is **±1** — there is no category where spatial reliably helps or reliably hurts.

### 3.3 Question-level differences (10 / 60)

#### UP (spatial fixes a wrong 3-axis answer) — 2 / 60

**Q1 — `EntityLog`** "Who used the screwdriver first?"
- gold = B, 3a = C (wrong), **4a = B ✅**
- Why spatial helped: the spatial layer surfaced `(<person>, near, screwdriver)` for the correct candidate when the episodic event-stream was ambiguous about who picked it up first.
- (Note: this question alternated between UP and DN across runs of this harness — model + retrieval variance dominates at n = 1.)

**Q33 — `EntityLog`** "What app did I order the takeout in my hand?"
- gold = A, 3a = D (wrong), **4a = A ✅**
- Why spatial helped: `(phone, on, table)` plus `(I, near, phone)` plus an open-vocab object tag `takeout-app screen` anchored the answer to the on-screen app the agent could then verify in episodic.

#### DN (spatial breaks a correct 3-axis answer) — 3 / 60

**Q6 — `TaskMaster`** "Who plans to grow flowers"
- gold = D, 3a = D ✅, **4a = A ❌**
- Why spatial hurt: `(<wrong-person>, near, flowers)` and `(flowers, on, table)` baited the agent into a proximity answer; the actual gold lives in a stated *plan* from episodic dialogue, not a spatial proximity at the query time.

**Q45 — `EntityLog`** "When was the last time we talked about coffee?"
- gold = A, 3a = A ✅, **4a = D ❌**
- Why spatial hurt: spatial returned multiple `(coffee, on, table)` mentions across the day; the reasoning agent latched onto a later spatial-anchored mention rather than the canonical *conversational* one.

**Q50 — `HabitInsight`** "What do I usually show everyone on my phone?"
- gold = D, 3a = D ✅, **4a = A ❌**
- Why spatial hurt: `(I, with, phone)` is everywhere; the spatial layer flooded the prompt with phone-related triples that distracted the agent from the *content* habit pattern.

#### Both wrong, different letters — 5 / 60

- **Q9** RelationMap "Who just helped Alice unpack and arrange the flowers" — 3a = C, 4a = A (gold D)
- **Q16** EntityLog "Who was the first to move the puzzle board on the table?" — 3a = C, 4a = A (gold B)
- **Q27** EntityLog "I put the pot aside, what was in the pot before?" — 3a = A, 4a = D (gold B)
- **Q31** EntityLog "Who used the black signature pen last time?" — 3a = D, 4a = B (gold C)
- **Q40** EventRecall "Shure mentioned Tiramisu, when was the last time we discussed making Tiramisu?" — 3a = A, 4a = B (gold C)

These are questions where retrieval failed on both sides — either the keyword overlap missed the decisive moment, or it surfaced enough of the wrong context to mislead. The spatial layer changed *which* wrong letter was picked but did not unlock the correct answer.

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
| Question set | (a) 20 synthetic WHERE-only (b) 22 EgoLifeQA WHERE-filtered | first 60 unfiltered DAY1 EgoLifeQA |
| Δ measured | (a) +15.0 %p (b) +4.5 %p | **−1.7 %p** (n = 60) — sample-noise narrowed from −4.0 %p at n = 25 |

The change of sign is **not** evidence that the previous report was wrong — it is evidence that **filtering questions changes the conclusion**. On WHERE-grounded questions, spatial helps clearly (+15 %p). On a representative slice of EgoLifeQA, it is essentially neutral (−1.7 %p, within the noise band of a stochastic LLM agent over 60 questions). Both can be true at once.

---

## 5. Limitations

1. **n = 60 of 102 DAY1 questions.** Full DAY1 plus multi-seed averaging would tighten the result. The shift from −4.0 %p (n = 25) to −1.7 %p (n = 60) shows the noise band at this n.
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

# 2. Run the 3-vs-4-axis ablation harness on the first 60 DAY1 questions
WORLDMM_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
WORLDMM_EMBED_DEVICE=cpu \
uv run python eval/three_vs_four_axis_ablation.py \
    --max-n 60 --max-rounds 3 \
    --output output/three_vs_four_n60.json
```

Output:
- `output/three_vs_four_n60.json` — per-question predictions for both configs.
- Console log includes per-question summary lines and the totals block.

The harness is at [`eval/three_vs_four_axis_ablation.py`](../eval/three_vs_four_axis_ablation.py); the 3-axis prompt is at [`src/worldmm/llm/templates/memory_reasoning_3axis.py`](../src/worldmm/llm/templates/memory_reasoning_3axis.py).

---

## 7. Bottom line for the write-up

> On a representative unfiltered slice of EgoLifeQA DAY1 (n = 60), adding the closed-vocabulary spatial memory as a 4th axis is **essentially neutral** (−1 / 60, −1.7 %p — within sample noise). The spatial signal **does exist** — it fixes 2 questions the 3-axis baseline gets wrong (Q1 screwdriver, Q33 takeout app) — but the reasoning agent also picks the spatial branch in 3 cases where it should not (Q6 grow-flowers, Q45 coffee-timing, Q50 phone-habit), and those *cost* a correct answer. The remaining 5 differing answers are "both wrong with different letters" — spatial reroutes a wrong guess without unlocking the correct one. Per-type Δ values at n = 60 are all within ±1, with no category showing a stable positive or negative effect. This is **not** in conflict with the earlier synthetic-WHERE finding of +15 %p — that one filtered to questions whose answers live in the spatial layer; this one does not. **The honest paper-level number for spatial-as-shipped is therefore: a clear +15 %p win on WHERE-grounded queries, but only a ≈0 %p effect on unfiltered EgoLifeQA, with question-type composition dominating the headline. The choice of question filter dominates the conclusion.**
>
> The natural next move is a **routing fix in the reasoning prompt**: the agent currently treats the spatial axis as freely selectable; constraining it to questions whose surface form contains WHERE / location cues (or letting the agent choose spatial only as a *second-round* refinement after episodic) should preserve the +UP wins while collapsing the −DN losses. This is captured as the v2 extension plan in [`docs/spatial-encoding-sensor-based.md`](spatial-encoding-sensor-based.md) §7.

---

## 8. Visual axis activated — first 15 DAY1 questions (n=15)

This smoke rerun used the real visual-memory files at `output/metadata/visual_memory/A1_JAKE/visual_embeddings_clip-ViT-B-32.pkl` and `output/metadata/visual_memory/A1_JAKE/visual_clips_clip-ViT-B-32.json` with the patched vision-capable LiteLLM proxy. It covered the first 15 DAY1 EgoLifeQA questions, `max_rounds = 3`, same `all-MiniLM-L6-v2` CPU text embedding path, and wrote `output/three_vs_four_n15_real_visual.json`.

| Configuration | Correct | % |
|---|---:|---:|
| **3-axis** (episodic + semantic + real visual) | **7 / 15** | **46.7 %** |
| **4-axis** (episodic + semantic + real visual + **spatial**) | **8 / 15** | **53.3 %** |
| **Δ** (4-axis − 3-axis) | **+1** | **+6.7 %p** |
| Answers differ between configs | 4 / 15 | 26.7 % |
| Of those: **UP** (4-axis fixes 3-axis wrong) | 2 | |
| Of those: **DN** (4-axis breaks 3-axis right) | 1 | |

Axis selections observed in an instrumented cached rerun of the same 15-question path:

| Configuration | Episodic | Semantic | Visual | Spatial |
|---|---:|---:|---:|---:|
| **3-axis** | 22 | 0 | 2 | n/a |
| **4-axis** | 23 | 0 | 5 | 4 |

The agent **did pick `visual`** (2 times in 3-axis; 5 times in 4-axis), but every visual retrieval returned **0 clips / 0 images** for these queries. Therefore visual hits did **not** contribute to any correct answer in this n = 15 prefix: the real visual axis was available and selectable, but this sample's selected visual queries fell outside the downloaded/indexed MP4 coverage or retrieved no frame payloads.

Compared with the visual-empty n = 60 run's first-15 prefix, the old prefix was **8 / 15 vs 8 / 15** (Δ 0), while this real-visual smoke is **7 / 15 vs 8 / 15** (Δ +1). The changed flips are not visual-evidence wins because the visual selections had no image payloads. Example: Q14 (`Who is missing compared to when we first started the puzzle?`) became an UP flip in the new run (gold A, 3-axis D, 4-axis A), but the 4-axis visual calls for Q14 returned 0 clips / 0 images, so the gain is run/routing variance, not a verified visual contribution.

Implication for the old disclaimer: remove the blanket claim that the visual axis is unbuilt or unavailable for this follow-up run, but keep a narrower caveat for this n = 15 result: **visual was active and selected, yet produced no clip/image hits on the selected queries**, so this smoke still does not measure a positive visual-evidence contribution.


## 8.1 Visual coverage expanded — n=30 rerun

**Date:** 2026-05-20 KST

Expanded the real-video coverage from **36** non-empty MP4s (**522 MB**) to **91** non-empty MP4s (**1.3 GB**) under `data/EgoLife/A1_JAKE/DAY1`, then rebuilt `clip-ViT-B-32` visual embeddings to **91** clip embeddings. Selection was targeted, not bulk: first-30 EgoLifeQA `target_time`/`query_time` neighborhoods plus top-15 spatial-extraction chunks by triple count. After expansion, all first-30 questions have at least one target/query timestamp covered within ±60 s; top-15 spatial-density chunks are also covered within ±60 s. A decord middle-frame sanity check passed on all 55 newly downloaded MP4s.

| Configuration | Correct | % |
|---|---:|---:|
| **3-axis** (episodic + semantic + visual) | **15 / 30** | **50.0 %** |
| **4-axis** (episodic + semantic + visual + spatial) | **16 / 30** | **53.3 %** |
| **Δ** (4-axis − 3-axis) | **+1** | **+3.3 %p** |
| Answers differ between configs | 6 / 30 | 20.0 % |
| Of those: **UP** (4-axis fixes 3-axis wrong) | 3 | |
| Of those: **DN** (4-axis breaks 3-axis right) | 2 | |

Visual telemetry required a trace rerun because `output/three_vs_four_n30_real_visual_expanded.json` stores only predictions/totals, not `round_history`. That trace rerun used the same MiniLM CPU embedder and `max_rounds=3`, but its answers varied, so use it for retrieval telemetry only, not headline accuracy.

| Config | Visual selections | Visual selections with image hits | Image payloads |
|---|---:|---:|---:|
| **3-axis trace** | 5 | 0 | 0 |
| **4-axis trace** | 10 | 2 | 60 images from 2 clip retrievals |

**Honest visual answer:** real visual now can return images, but it still produced **no measured accuracy lift** in this n=30 ablation. The only visual image hits in the trace were two 4-axis searches for Q16 (`Who was the first to move the puzzle board on the table?`), both retrieving frames around DAY1 12:27. The canonical ablation still got Q16 wrong in both configs (gold B, 3-axis C, 4-axis C), so those visual hits did not flip an answer. Q14 (`Who is missing compared to when we first started the puzzle?`) was an UP flip in the canonical run (gold A, 3-axis D, 4-axis A), but the trace visual searches for the analogous puzzle-participant queries returned `[No results]`; no evidence supports calling that a visual win.

Conclusion: expanded coverage fixes the earlier “visual selected but always 0 clips/images” failure mode only partially. Visual retrieval can now return frames, yet the measurable +1 / 30 lift belongs to the 4-axis configuration as a whole, not to verified visual evidence.
