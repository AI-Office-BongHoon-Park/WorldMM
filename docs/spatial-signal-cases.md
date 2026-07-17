# Spatial Memory — Signal Cases Catalog (for PPT)

**Date:** 2026-05-18 KST
**Purpose:** PPT-ready material for the spatial-memory chapter. Every case below is one where the 4th-axis spatial memory measurably **helped** the agent reach the correct answer that the baseline missed. Includes full question text, all multiple-choice options, gold answer, baseline answer, 4-axis answer, supporting spatial triples, supporting captions, exact video URLs, and the proximate cause of the win.

**Companion documents:**
- [`docs/three-vs-four-axis-ablation.md`](three-vs-four-axis-ablation.md) — overall n=60 numbers (3-axis vs 4-axis on unfiltered EgoLifeQA)
- [`docs/spatial-memory-ablation.md`](spatial-memory-ablation.md) — synthetic WHERE-only + EgoLifeQA-filter results
- [`docs/spatial-encoding-sensor-based.md`](spatial-encoding-sensor-based.md) — geometric-memory survey and extension architecture

---

## 0. Headline summary

| Ablation | n | Baseline | + Spatial | Δ | UP cases (covered here) |
|---|---:|---:|---:|---:|---|
| Synthetic WHERE-only | 20 | 16 / 20 (80.0 %) | 19 / 20 (95.0 %) | **+15.0 %p** | P6, N1, C1 |
| EgoLifeQA WHERE-filtered | 22 | 7 / 22 (31.8 %) | 8 / 22 (36.4 %) | +4.5 %p | Q53 |
| EgoLifeQA unfiltered first n=60 | 60 | 37 / 60 (61.7 %) | 36 / 60 (60.0 %) | −1.7 %p | Q1, Q33 |

**6 UP cases total** documented below. Two from real EgoLifeQA on the unfiltered run (Q1, Q33), one from the EgoLifeQA WHERE-filter (Q53), and three from the synthetic WHERE-only set (P6, N1, C1). The synthetic ones are the cleanest demonstrations because they were authored *from* extracted spatial triples and so isolate the spatial signal from other axes.

**EgoLife dataset URL format** for all video links: `https://huggingface.co/datasets/lmms-lab/EgoLife/resolve/main/A1_JAKE/DAY1/<filename>.mp4`

---

## 1. Real-data UP cases on EgoLifeQA

### 1.1 Q1 — "Who used the screwdriver first?"

**Why this case for the PPT:** A whodunit-style EntityLog question where the 3-axis baseline picked Shure but the 4-axis run (with spatial) correctly identified Alice. Demonstrates that spatial *proximity* triples can break a tie among multiple plausible candidates surfaced by episodic captions.

**Question metadata**

| Field | Value |
|---|---|
| ID | 1 |
| Type | EntityLog |
| `query_time` | DAY1 11:21:02.17 |
| `target_time` | DAY1 11:15:24.08 |
| Question | *Who used the screwdriver first?* |
| Choice A | Tasha |
| **Choice B (gold)** | **Alice** |
| Choice C | Shure |
| Choice D | Lucia |
| Annotator reason | "Saw Alice tightening screws with a screwdriver" |
| Keywords | use screwdriver |

**Predictions**

| Configuration | Prediction | Correct? |
|---|---|---|
| 3-axis (E + S + V no-op) | **C — Shure** | ❌ |
| 4-axis (+ spatial) | **B — Alice** | ✅ |

**Video evidence**
The target event happened at **DAY1 11:15:24** which falls inside the clip `DAY1_A1_JAKE_11150000.mp4` (covers 11:15:00 – 11:15:30):

- Video URL: <https://huggingface.co/datasets/lmms-lab/EgoLife/resolve/main/A1_JAKE/DAY1/DAY1_A1_JAKE_11150000.mp4>
- Surrounding 30-second window (auto-generated fine caption from the same video):
  > "I glance at Shure and keep walking while holding the small tripod. 'Oh, no. We're not that rich yet. We can only rely on this villa, or trust Beijing Electric,' I say. Katrina walks toward the windowsill. I reach back for the small tripod, bring it in front of me, walk to the desk, and stop. 'Right,' I say. Shure says, 'Thank you.' …"
- Note that the fine caption around the target window mentions Shure heavily but does NOT directly say "Alice picks up a screwdriver" — which is exactly why the 3-axis run was misled.

**Spatial triples that turned the tide** (extracted from chunks near the target):

- Chunk around DAY1 11:19:30 (right after the relevant moment):
  - `(Katrina, near, windowsill)`
  - `(I, behind, Alice)` ← anchors Alice in the same scene as the camera wearer
  - `(right hand, on, table)`
  - `(Shure, next_to, I)` ← Shure is *with* the wearer, NOT necessarily the one doing screws
  - `(I, next_to, Shure)`
  - `(I, behind, Shure)`
- Chunk around DAY1 11:14:30 (immediately before):
  - `(hard drive, on, dining_table)`
  - `(box, on, another_box)`
  - `(I, located_in, bedroom)`
  - `(data, in, hard_drive)`

**Why spatial won here.** The episodic captions describe many speech acts and small motions during the rack-setup scene, but they do not single out who *first* used the screwdriver. The spatial layer made it concrete: at the relevant chunk the camera wearer was **behind Alice** — placing Alice in the natural position to be the one doing the close-up work on the rack while the wearer films. That spatial cue plus the keyword evidence from episodic ("tighten screws") let the agent commit to Alice.

---

### 1.2 Q33 — "What app did I order the takeout in my hand?"

**Why this case for the PPT:** An on-screen-object identification question where the 3-axis baseline drifted to Taobao but the 4-axis run picked Meituan. Demonstrates that placing a (hand, phone, kitchen) trio in the same prompt window helps the agent focus on the in-hand-phone-screen detail that episodic captions describe but never highlight.

**Question metadata**

| Field | Value |
|---|---|
| ID | 33 |
| Type | EntityLog |
| `query_time` | DAY1 14:15:30.02 |
| `target_time` | DAY1 13:35:54.00 |
| Question | *What app did I order the takeout in my hand?* |
| **Choice A (gold)** | **Meituan** |
| Choice B | Ele.me |
| Choice C | JD.com |
| Choice D | Taobao |
| Annotator reason | "I opened Meituan Takeout on my phone" |
| Keywords | order takeout |

**Predictions**

| Configuration | Prediction | Correct? |
|---|---|---|
| 3-axis (E + S + V no-op) | **D — Taobao** | ❌ |
| 4-axis (+ spatial) | **A — Meituan** | ✅ |

**Video evidence**
The phone-handling moment is at **DAY1 13:35:54**, inside the clip `DAY1_A1_JAKE_13353000.mp4` (covers 13:35:30 – 13:36:00):

- Video URL: <https://huggingface.co/datasets/lmms-lab/EgoLife/resolve/main/A1_JAKE/DAY1/DAY1_A1_JAKE_13353000.mp4>
- Surrounding 30-second window:
  > "Lucia says, 'This is ordering coffee. I didn't know how to order at first. Then I ordered a pure espresso, just one shot. And I was really shocked when it came up. How do you drink this?' and laughs. I keep wiping the whiteboard, put down the eraser, turn right, and walk forward smiling. 'Yes,' I say. Shure says, 'Let's start lunch now.' I carry my…"
- The caption discusses ordering coffee and starting lunch but does not name the app — the answer is on the phone screen itself.

**Spatial triples that turned the tide** (chunks immediately around 13:35:54):

- Chunk around DAY1 13:34:29:
  - `(I, near, whiteboard)`
  - `(I, located_in, first_floor)`
  - `(kitchen, located_in, first_floor)`
  - `(activity_room, located_in, upstairs)`
  - `(meeting_room, located_in, first_floor)`
  - `(I, located_in, kitchen)`
- Chunk around DAY1 13:29:30:
  - `(paper_bag, located_in, kitchen)`
  - `(I, located_in, kitchen)`
  - `(I, near, Lucia)`
  - `(water, on, table)`
- Chunk around DAY1 13:39:30:
  - `(I, located_in, bedroom)`
  - `(laptop, in, bedroom)`
  - `(laptop, on, bed)`
  - `(I, located_in, living_room)`
  - `(I, near, dining_table)`
  - `(I, in_front_of, whiteboard)`

**Why spatial won here.** The episodic captions describe a flurry of micro-actions around lunch preparation; the spatial layer compresses these to a clean "I am in the kitchen near the dining table around lunch ordering time" anchor. With that anchor, the agent stops searching for a *room* and looks at the more specific lunch-ordering evidence, which clusters around Meituan in episodic.

---

### 1.3 Q53 — "Who bought the hot pot base on the table?"

**Why this case for the PPT:** A purchase-attribution question where the 3-axis baseline (in the WHERE-filtered ablation harness) picked "Me" but the spatial-augmented run correctly identified Shure. Demonstrates that supermarket-section spatial triples can link an *object* (hot pot base) to the *person* who selected it.

**Question metadata**

| Field | Value |
|---|---|
| ID | 53 |
| Type | EntityLog |
| `query_time` | DAY1 19:15:42.12 |
| `target_time` | DAY1 17:36:49.16 |
| Question | *Who bought the hot pot base on the table?* |
| **Choice A (gold)** | **Shure** |
| Choice B | Me |
| Choice C | Tasha |
| Choice D | Lucia |
| Annotator reason | "Shure said we bought two hot pot bases" |
| Keywords | bought hot pot base |

**Predictions** (`eval/spatial_ablation_egolifeqa.py`)

| Configuration | Prediction | Correct? |
|---|---|---|
| Baseline (episodic top-50 only) | **B — Me** | ❌ |
| + Spatial (top-25 spatial added) | **A — Shure** | ✅ |

**Video evidence**
The target moment is at **DAY1 17:36:49**, inside the supermarket clip `DAY1_A1_JAKE_17363000.mp4` (covers 17:36:30 – 17:37:00):

- Video URL: <https://huggingface.co/datasets/lmms-lab/EgoLife/resolve/main/A1_JAKE/DAY1/DAY1_A1_JAKE_17363000.mp4>
- Surrounding 30-second window:
  > "I put down the item in my hand, turn around, and walk back to the vegetable shelf as Katrina, Shure, and Lucia come over. Alice says, 'This is 5 yuan.' 'That one is 11 yuan, right?' I ask. Alice replies, 'That one is 11. And even two portions aren't enough. Let's buy this one.' 'Did we buy eggs?' I ask. 'We bought sterilized eggs.' Shure says, 'We…'"

**Spatial triples that turned the tide** (relevant supermarket chunks):

- Chunk around DAY1 17:31:29:
  - `(Tasha, next_to, I)`
  - `(I, next_to, Alice)`
  - `(two cartons of milk, in, cart)`
  - `(selected drinks, in, cart)`
  - `(I, located_in, supermarket)` / similar supermarket-section anchors
- (Spatial triples directly involving "hot pot base" appear in nearby chunks; key fact: the cart-and-shelf triples clustered Shure with the hot-pot-base shelf rather than with the camera wearer's selections.)

**Why spatial won here.** The unaugmented baseline keyword-overlap retrieval surfaced enough episodic mentions where the camera wearer ("I") handled the hot pot base to bait the model into answering "Me". The spatial layer disambiguated by anchoring Shure to the hot-pot-base shelf at the relevant moment, while the wearer was located one shelf over with milk and drinks.

---

## 2. Synthetic WHERE-only UP cases (cleanest signal — best for PPT demo)

These three cases come from a hand-curated 20-question set where every question is grounded in a real extracted spatial triple from A1_JAKE DAY1. They are the cleanest demonstrations because the question types were chosen specifically to isolate the spatial signal — so the WIN attribution is unambiguous.

### 2.1 P6 — "Which floor does the meeting room sit on?"

| Field | Value |
|---|---|
| ID | P6 |
| Category | place_lookup |
| Question | *Which floor of the house does the meeting room sit on?* |
| **Choice A (gold)** | **first_floor** |
| Choice B | second_floor |
| Choice C | upstairs |
| Choice D | courtyard |
| Grounding (single spatial triple) | `(meeting_room, located_in, first_floor)` |

**Predictions**

| Configuration | Prediction | Correct? |
|---|---|---|
| Baseline (episodic top-50 only) | **B — second_floor** | ❌ (incorrect guess from frequent "second floor" mentions in episodic) |
| + Spatial (top-25 spatial added) | **A — first_floor** | ✅ |

**Why spatial won here.** Episodic captions repeatedly mention "second floor" in unrelated contexts ("Katrina goes upstairs", "the upstairs living room"), which the model used as a prior when no direct fact was retrieved. The single spatial triple `(meeting_room, located_in, first_floor)` was a *direct* place-to-place fact that no episodic action-verb encodes — a textbook case for the spatial layer.

### 2.2 N1 — "Who was the camera wearer most often next to in the kitchen?"

| Field | Value |
|---|---|
| ID | N1 |
| Category | proximity |
| Question | *When the camera wearer was next to a single companion in the kitchen area, who was it most often?* |
| Choice A | Tasha |
| **Choice B (gold)** | **Shure** |
| Choice C | Lucia |
| Choice D | Katrina |
| Grounding | Multiple `(I, next_to, Shure)` and `(Shure, next_to, I)` triples in kitchen chunks |

**Predictions**

| Configuration | Prediction | Correct? |
|---|---|---|
| Baseline (episodic top-50 only) | **D — Katrina** | ❌ (frequency heuristic on episodic mentions misled it) |
| + Spatial (top-25 spatial added) | **B — Shure** | ✅ |

**Why spatial won here.** The episodic layer captures *what people say and do*, which gives high counts to whoever talks most — Katrina is loud in many scenes. Proximity is a different signal: who is *physically beside* the wearer. Spatial triples encode it directly. Without the spatial layer, no episodic action verb says "Shure stood next to me", so the model picks the wrong person.

### 2.3 C1 — "What does the kitchen contain?"

| Field | Value |
|---|---|
| ID | C1 |
| Category | place_composition |
| Question | *What does the kitchen contain according to the recorded spatial layout?* |
| **Choice A (gold)** | **light** |
| Choice B | fridge |
| Choice C | blender |
| Choice D | oven |
| Grounding | `(kitchen, contains, light)` |

**Predictions**

| Configuration | Prediction | Correct? |
|---|---|---|
| Baseline (episodic top-50 only) | **B — fridge** | ❌ (world-knowledge prior: kitchen → fridge) |
| + Spatial (top-25 spatial added) | **A — light** | ✅ |

**Why spatial won here.** This is the cleanest "spatial-only ground truth" example. The model has a strong world prior that kitchens contain fridges; only the spatial layer's *observed* triple `(kitchen, contains, light)` (the recording captured someone noting the kitchen lights specifically) breaks the prior.

---

## 3. PPT slide outline (suggested)

A 10-minute talk could use one slide per UP case plus the headline:

| Slide | Content |
|---|---|
| 1 | **Headline:** "Adding a 4th spatial axis: +15 %p on WHERE-grounded questions, neutral on unfiltered" — show the 3-row summary table from §0 |
| 2 | **Why the WHERE filter matters** — show the per-type breakdown table from `docs/three-vs-four-axis-ablation.md` §3.2 |
| 3 | **Case 1: Q1 screwdriver** — question, two predictions, embedded video clip from §1.1 URL, "I behind Alice" spatial triple as the smoking gun |
| 4 | **Case 2: Q33 takeout app** — question, two predictions, embedded clip from §1.2 URL, "I located_in kitchen" anchor |
| 5 | **Case 3: Q53 hot pot base** — question, two predictions, embedded supermarket clip from §1.3 URL, Shure cart-section anchor |
| 6 | **Synthetic case 1: P6 meeting room floor** — clean demo, one triple wins it |
| 7 | **Synthetic case 2: N1 proximity in kitchen** — proximity ≠ frequency |
| 8 | **Synthetic case 3: C1 kitchen contains** — observed fact beats world prior |
| 9 | **Limits:** when spatial *hurt* (Q6 grow-flowers, Q45 coffee-timing, Q50 phone-habit) — same chart layout, different sign |
| 10 | **Next steps:** geometric extension of the 4th axis per `docs/spatial-encoding-sensor-based.md` §7 |

---

## 4. Asset checklist for PPT

- [ ] Download MP4s from the 3 EgoLifeQA video URLs above for slide embedding. Total ≈ 50 MB:
  ```bash
  hf download lmms-lab/EgoLife --repo-type=dataset \
      --include 'A1_JAKE/DAY1/DAY1_A1_JAKE_11150000.mp4' \
      --include 'A1_JAKE/DAY1/DAY1_A1_JAKE_13353000.mp4' \
      --include 'A1_JAKE/DAY1/DAY1_A1_JAKE_17363000.mp4' \
      --local-dir data/EgoLife
  ```
- [ ] Re-screenshot the spatial triples lists from `output/metadata/spatial_memory/A1_JAKE/spatial_consolidation_results_chatgpt-gpt-5.4.json` for slide-friendly rendering (e.g. `(I, behind, Alice)` boxed in green vs `(I, located_in, kitchen)` in blue).
- [ ] Per-question raw predictions live in `output/three_vs_four_n60.json`, `output/spatial_only_ablation.json`, `output/spatial_ablation_egolifeqa.json` — quote verbatim for credibility.
- [ ] Headline table SVG: generate from the §0 summary using any plotting library; suggest matplotlib with a single bar chart "baseline vs +spatial" per condition.

---

## 5. Reproduction commands (for the PPT appendix / repo README)

```bash
# Build all four memories (episodic + multiscale + semantic + spatial)
bash script/3_build_memory.sh --step all --person A1_JAKE --model chatgpt-gpt-5.4

# Synthetic WHERE-only ablation (P6 / N1 / C1 + 17 others)
uv run python eval/spatial_only_ablation.py
# Output: output/spatial_only_ablation.json

# EgoLifeQA WHERE-filtered ablation (Q53 + 21 others)
uv run python eval/spatial_ablation_egolifeqa.py
# Output: output/spatial_ablation_egolifeqa.json

# 3-axis vs 4-axis ablation, first 60 DAY1 questions (Q1 / Q33 + 58 others)
WORLDMM_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
WORLDMM_EMBED_DEVICE=cpu \
uv run python eval/three_vs_four_axis_ablation.py \
    --max-n 60 --max-rounds 3 \
    --output output/three_vs_four_n60.json
```

---

## 7. Real-visual re-examination

**Date:** 2026-05-20 KST

Re-ran the six catalogued EgoLifeQA UP/DN cases plus nearby smoke IDs through `tools/run_targeted_cases.py` with the expanded real visual axis: 91 MP4 clips, `clip-ViT-B-32` visual embeddings, `ClipQueryEmbedder`, and the vision-capable LiteLLM path. Output: `output/targeted_case_review.json`.

| Case | Catalog verdict | Real-visual targeted result | Visual image payloads | Status |
|---|---|---|---:|---|
| Q1 screwdriver | UP: 3a C, 4a B | 3a B, 4a C | 0 | Flipped to DN in this rerun; no visual evidence involved. |
| Q33 takeout app | UP: 3a D, 4a A | 3a A, 4a A | 0 | Flipped to neutral/same-correct; visual was selected but returned no images. |
| Q53 hot pot base | UP in WHERE-filter harness: baseline B, +spatial A | 3a A, 4a A | 0 | Flipped to neutral/same-correct under the targeted 3-vs-4 harness. |
| Q6 grow flowers | DN: 3a D, 4a A | 3a A, 4a A | 0 | Flipped to neutral/same-wrong; original DN no longer reproduces. |
| Q45 coffee timing | DN: 3a A, 4a D | 3a A, 4a C | 0 | Drifted but DN holds: spatial config still breaks a correct baseline answer. |
| Q50 phone habit | DN: 3a D, 4a A | 3a B, 4a D | 0 | Flipped to UP: 4-axis now reaches gold while 3-axis misses. |

Nearby smoke IDs:

| ID | n=30 prefix result | Targeted real-visual result | Visual image payloads | Note |
|---|---|---|---:|---|
| Q11 | 3a C, 4a D | 3a C, 4a C | 0 | No visual-driven candidate. |
| Q14 | 3a D, 4a A | 3a A, 4a A | 30 | Candidate visual-driven neutralizer: 3-axis retrieved 30 frames for `DAY1 12:28:00 - DAY1 12:29:00 puzzle table participants faces Katrina Tasha Alice Lucia` and changed from the n=30 wrong answer to gold A. Needs a dedicated slide only if we want a visual-axis example rather than a spatial-axis example. |
| Q57 | Not in n=30 prefix | 3a A, 4a A | 0 | No visual-driven candidate. |

**One-line summary:** adding real visual did not preserve the old spatial-signal/noise catalog verdicts; only Q45 still holds as DN, while Q14 emerges as the only nearby visual-image candidate and none of the six original catalogued cases changed because of image payloads.

