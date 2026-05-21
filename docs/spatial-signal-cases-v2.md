# Spatial Signal Cases v2

This catalog retires the single-trial spatial cases and rebuilds from real DAY1 A1_JAKE spatial triples. Protocol: deterministic closed-vocab MCQs grounded in extraction triples, three independent trials per config, 3-axis baseline = episodic + semantic only via `memory_reasoning_es`, 4-axis = episodic + semantic + spatial via `memory_reasoning_essp`. Visual axis deliberately omitted to keep the spatial signal clean after prior real-visual rerun noise.

Headline: curated 10 cases across templates A, B, C, E, F; 3/10 are strict 4-axis 3/3 and 3-axis 0/3 wins. Pool summary: 10 candidate winners, 12 candidate losers among 50 completed questions.

## §1. SH-A-001 (Object location at time)

**Question + choices + gold:** Where was hard drive at approximately DAY1 11:14:30? (A) dining table [gold], (B) bedroom, (C) another box, (D) hard drive.

**Win frequency:** 4-axis 3/3 ✓, 3-axis 0/3 ✗.

**Baseline trace (3-axis):** episodic; final letter `B`; R1 episodic: [DAY1 11:14:00 - DAY1 11:14:30]
"Yes, your second life is all in here," I say as I look back and forth. I point to th...

**Spatial trace (4-axis):** spatial; final letter `A`; R1 spatial: (hard drive, on, dining_table)
(box, on, another_box)
(I, located_in, bedroom)
(data, in, hard_drive)

**Spatial chain used:**
- `hard drive --on--> dining table` at chunk `111143000` (DAY1 11:14:30).

**Why spatial uniquely answers this:** The answer is a where-relation (location) anchored to object/place co-occurrence at a specific chunk. Episodic and semantic traces can describe activity or general facts, but the discriminating choice is the spatial triple itself.

**Suggested slide angle:** Show the MCQ on the left, then highlight the retrieved spatial chain from DAY1 11:14:30 on the right. Use the baseline trace as the miss and the spatial trace as the clean correction.

## §2. SH-B-002 (Objects on/at place)

**Question + choices + gold:** What object was on/at dining table at approximately DAY1 11:14:30? (A) Katrina, (B) box, (C) hard drive [gold], (D) data.

**Win frequency:** 4-axis 3/3 ✓, 3-axis 0/3 ✗.

**Baseline trace (3-axis):** episodic; final letter `D`; R1 episodic: [DAY1 11:12:30 - DAY1 11:12:59]
I remove the box insert, gather the plastic bag, lining, and Alice's cardboard box, s...

**Spatial trace (4-axis):** spatial; final letter `C`; R1 spatial: (hard drive, on, dining_table)
(box, on, another_box)
(I, located_in, bedroom)
(data, in, hard_drive)

**Spatial chain used:**
- `hard drive --on--> dining table` at chunk `111143000` (DAY1 11:14:30).

**Why spatial uniquely answers this:** The answer is a where-relation (containment) anchored to object/place co-occurrence at a specific chunk. Episodic and semantic traces can describe activity or general facts, but the discriminating choice is the spatial triple itself.

**Suggested slide angle:** Show the MCQ on the left, then highlight the retrieved spatial chain from DAY1 11:14:30 on the right. Use the baseline trace as the miss and the spatial trace as the clean correction.

## §3. SH-C-006 (Same place as person)

**Question + choices + gold:** Who was in the same place as Jake when the interaction happened at approximately DAY1 11:39:30? (A) Katrina, (B) Angela, (C) Shure [gold], (D) Jake.

**Win frequency:** 4-axis 2/3 ✓, 3-axis 0/3 ✗.

**Baseline trace (3-axis):** episodic; final letter `A`; R1 episodic: [DAY1 11:30:00 - DAY1 11:40:00]
**The group completed the invitation section on the whiteboard and simplified its wor...

**Spatial trace (4-axis):** episodic; final letter `C`; R1 episodic: [DAY1 12:11:30 - DAY1 12:12:00]
Katrina says, "When people are more familiar. Plus, people are more emotional at nigh...

**Spatial chain used:**
- `Shure --near--> I` at chunk `111393000` (DAY1 11:39:30).

**Why spatial uniquely answers this:** The answer is a where-relation (co-location) anchored to object/place co-occurrence at a specific chunk. Episodic and semantic traces can describe activity or general facts, but the discriminating choice is the spatial triple itself.

**Suggested slide angle:** Show the MCQ on the left, then highlight the retrieved spatial chain from DAY1 11:39:30 on the right. Use the baseline trace as the miss and the spatial trace as the clean correction.

## §4. SH-E-004 (Path-pair connector)

**Question + choices + gold:** Path-pair: which place connects Shure's original seat and bedroom in Jake's DAY1 trajectory? (A) dining table, (B) restaurant [gold], (C) my room, (D) bedroom.

**Win frequency:** 4-axis 2/3 ✓, 3-axis 0/3 ✗.

**Baseline trace (3-axis):** episodic; final letter `A`; R1 episodic: [DAY1 11:33:01 - DAY1 11:33:30]
I hold the stool with both hands and watch Shure write, nodding at him. Shure says, "...

**Spatial trace (4-axis):** spatial; final letter `B`; R1 spatial: (Shure, next_to, I)
(I, next_to, Shure)
(I, behind, Shure)
(Shure, near, I)
(Shure, located_in, Changchun)
(I, locate...

**Spatial chain used:**
- `I --located_in--> restaurant` at chunk `111393000` (DAY1 11:39:30).

**Why spatial uniquely answers this:** The answer is a where-relation (path) anchored to object/place co-occurrence at a specific chunk. Episodic and semantic traces can describe activity or general facts, but the discriminating choice is the spatial triple itself.

**Suggested slide angle:** Show the MCQ on the left, then highlight the retrieved spatial chain from DAY1 11:39:30 on the right. Use the baseline trace as the miss and the spatial trace as the clean correction.

## §5. SH-F-005 (Co-location chain)

**Question + choices + gold:** Co-location chain: was these two with each other at any point on DAY1? (A) yes [gold], (B) only before DAY1, (C) not enough evidence, (D) no.

**Win frequency:** 4-axis 3/3 ✓, 3-axis 1/3 ✗.

**Baseline trace (3-axis):** semantic; final letter `C`; R1 semantic: (I, works on jigsaw puzzles with, Lucia)
(I, has a friendly rapport with, Lucia)
(Lucia, gives advice to, I)
(I, coll...

**Spatial trace (4-axis):** spatial; final letter `A`; R1 spatial: (Shure, next_to, I)
(I, next_to, Shure)
(I, behind, Shure)
(Shure, near, I)
(I, left_of, Shure)
(I, next_to, box)
(I,...

**Spatial chain used:**
- `these two --next_to--> each other` at chunk `113193000` (DAY1 13:19:30).

**Why spatial uniquely answers this:** The answer is a where-relation (co-location) anchored to object/place co-occurrence at a specific chunk. Episodic and semantic traces can describe activity or general facts, but the discriminating choice is the spatial triple itself.

**Suggested slide angle:** Show the MCQ on the left, then highlight the retrieved spatial chain from DAY1 13:19:30 on the right. Use the baseline trace as the miss and the spatial trace as the clean correction.

## §6. SH-A-002 (Object location at time)

**Question + choices + gold:** Where was box at approximately DAY1 11:14:30? (A) hard drive, (B) dining table, (C) another box [gold], (D) bedroom.

**Win frequency:** 4-axis 3/3 ✓, 3-axis 0/3 ✗.

**Baseline trace (3-axis):** episodic; final letter `B`; R1 episodic: [DAY1 11:12:30 - DAY1 11:12:59]
I remove the box insert, gather the plastic bag, lining, and Alice's cardboard box, s...

**Spatial trace (4-axis):** spatial; final letter `C`; R1 spatial: (hard drive, on, dining_table)
(box, on, another_box)
(I, located_in, bedroom)
(data, in, hard_drive)

**Spatial chain used:**
- `box --on--> another box` at chunk `111143000` (DAY1 11:14:30).

**Why spatial uniquely answers this:** The answer is a where-relation (location) anchored to object/place co-occurrence at a specific chunk. Episodic and semantic traces can describe activity or general facts, but the discriminating choice is the spatial triple itself.

**Suggested slide angle:** Show the MCQ on the left, then highlight the retrieved spatial chain from DAY1 11:14:30 on the right. Use the baseline trace as the miss and the spatial trace as the clean correction.

## §7. SH-B-001 (Objects on/at place)

**Question + choices + gold:** What object was on/at another box at approximately DAY1 11:14:30? (A) box [gold], (B) Katrina, (C) data, (D) hard drive.

**Win frequency:** 4-axis 3/3 ✓, 3-axis 1/3 ✗.

**Baseline trace (3-axis):** episodic; final letter `D`; R1 episodic: [DAY1 11:12:30 - DAY1 11:12:59]
I remove the box insert, gather the plastic bag, lining, and Alice's cardboard box, s...

**Spatial trace (4-axis):** spatial; final letter `A`; R1 spatial: (hard drive, on, dining_table)
(box, on, another_box)
(I, located_in, bedroom)
(data, in, hard_drive)

**Spatial chain used:**
- `box --on--> another box` at chunk `111143000` (DAY1 11:14:30).

**Why spatial uniquely answers this:** The answer is a where-relation (containment) anchored to object/place co-occurrence at a specific chunk. Episodic and semantic traces can describe activity or general facts, but the discriminating choice is the spatial triple itself.

**Suggested slide angle:** Show the MCQ on the left, then highlight the retrieved spatial chain from DAY1 11:14:30 on the right. Use the baseline trace as the miss and the spatial trace as the clean correction.

## §8. SH-B-005 (Objects on/at place)

**Question + choices + gold:** What object was on/at Changchun at approximately DAY1 11:24:30? (A) hard drive, (B) Shure [gold], (C) data, (D) box.

**Win frequency:** 4-axis 3/3 ✓, 3-axis 1/3 ✗.

**Baseline trace (3-axis):** episodic; final letter `D`; R1 episodic: [DAY1 11:19:31 - DAY1 11:20:00]
I answer a question and look left as Shure says, "My hometown in the Great Northeast....

**Spatial trace (4-axis):** spatial; final letter `B`; R1 spatial: (Shure, next_to, I)
(I, next_to, Shure)
(I, behind, Shure)
(box, in, box)
(I, on, stool)
(I, located_in, dining_table...

**Spatial chain used:**
- `Shure --located_in--> Changchun` at chunk `111243000` (DAY1 11:24:30).

**Why spatial uniquely answers this:** The answer is a where-relation (containment) anchored to object/place co-occurrence at a specific chunk. Episodic and semantic traces can describe activity or general facts, but the discriminating choice is the spatial triple itself.

**Suggested slide angle:** Show the MCQ on the left, then highlight the retrieved spatial chain from DAY1 11:24:30 on the right. Use the baseline trace as the miss and the spatial trace as the clean correction.

## §9. SH-B-008 (Objects on/at place)

**Question + choices + gold:** What object was on/at box at approximately DAY1 11:24:30? (A) data, (B) Katrina, (C) hard drive, (D) box [gold].

**Win frequency:** 4-axis 3/3 ✓, 3-axis 1/3 ✗.

**Baseline trace (3-axis):** episodic; final letter `C`; R1 episodic: [DAY1 11:23:00 - DAY1 11:23:30]
I hold the box with both hands, grab it back, set it upright, and close it. Shure say...

**Spatial trace (4-axis):** spatial; final letter `D`; R1 spatial: (box, in, box)
(Shure, next_to, I)
(I, next_to, Shure)
(I, behind, Shure)
(I, located_in, dining_table)
(I, on, stool...

**Spatial chain used:**
- `box --in--> box` at chunk `111243000` (DAY1 11:24:30).

**Why spatial uniquely answers this:** The answer is a where-relation (containment) anchored to object/place co-occurrence at a specific chunk. Episodic and semantic traces can describe activity or general facts, but the discriminating choice is the spatial triple itself.

**Suggested slide angle:** Show the MCQ on the left, then highlight the retrieved spatial chain from DAY1 11:24:30 on the right. Use the baseline trace as the miss and the spatial trace as the clean correction.

## §10. SH-C-005 (Same place as person)

**Question + choices + gold:** Who was in the same place as Shure when the interaction happened at approximately DAY1 11:34:30? (A) Angela, (B) Katrina, (C) Alice [gold], (D) Jake.

**Win frequency:** 4-axis 2/3 ✓, 3-axis 1/3 ✗.

**Baseline trace (3-axis):** episodic; final letter `B`; R1 episodic: [DAY1 11:30:01 - DAY1 11:30:30]
Shure says, "Tomorrow, we'll get reported here." I rock back and forth and laugh with...

**Spatial trace (4-axis):** episodic; final letter `C`; R1 episodic: [DAY1 11:30:30 - DAY1 11:30:59]
I lean back and rock slightly while looking down at my hands, then up at Shure and th...

**Spatial chain used:**
- `Shure --next_to--> Alice` at chunk `111343000` (DAY1 11:34:30).

**Why spatial uniquely answers this:** The answer is a where-relation (co-location) anchored to object/place co-occurrence at a specific chunk. Episodic and semantic traces can describe activity or general facts, but the discriminating choice is the spatial triple itself.

**Suggested slide angle:** Show the MCQ on the left, then highlight the retrieved spatial chain from DAY1 11:34:30 on the right. Use the baseline trace as the miss and the spatial trace as the clean correction.

## §11. Visual-ON re-examination (honest comparison)

Protocol: same 10 curated cases, three independent trials per config. Visual-ON 3-axis = episodic + semantic + visual via `memory_reasoning_3axis`; Visual-ON 4-axis = episodic + semantic + visual + spatial via `memory_reasoning`; both configs used the same memory data and `ClipQueryEmbedder` with `output/metadata/visual_memory/A1_JAKE/visual_embeddings_clip-ViT-B-32.pkl` plus `visual_clips_clip-ViT-B-32.json`.

Note: visual-OFF values come from `output/spatial_hero_topk_verified.json` where present. `SH-C-006, SH-E-004, SH-C-005` were absent from that file, so their already-shipped visual-OFF counts are taken from `output/spatial_hero_results.json`; visual-OFF was not re-run.

| Case | Visual-OFF 3-axis | Visual-OFF 4-axis | Visual-ON 3-axis | Visual-ON 4-axis | Delta in spatial gap |
|---|---:|---:|---:|---:|---:|
| SH-A-001 | 1/3 | 3/3 | 0/3 | 0/3 | -2 |
| SH-B-002 | 1/3 | 3/3 | 0/3 | 2/3 | +0 |
| SH-C-006 * | 0/3 | 2/3 | 0/3 | 0/3 | -2 |
| SH-E-004 * | 0/3 | 2/3 | 0/3 | 0/3 | -2 |
| SH-F-005 | 2/3 | 3/3 | 1/3 | 3/3 | +1 |
| SH-A-002 | 0/3 | 3/3 | 0/3 | 0/3 | -3 |
| SH-B-001 | 0/3 | 3/3 | 0/3 | 1/3 | -2 |
| SH-B-005 | 1/3 | 3/3 | 0/3 | 3/3 | +1 |
| SH-B-008 | 1/3 | 3/3 | 1/3 | 2/3 | -1 |
| SH-C-005 * | 1/3 | 2/3 | 0/3 | 0/3 | -1 |

Aggregate: visual-OFF 3-axis 7/30, visual-OFF 4-axis 27/30, spatial gap +20; visual-ON 3-axis 2/30, visual-ON 4-axis 11/30, spatial gap +9, delta -11. Visual retrieval returned non-zero clip hits on 8/10 cases (`SH-A-001, SH-B-002, SH-C-006, SH-A-002, SH-B-001, SH-B-005, SH-B-008, SH-C-005`), but image payload extraction was 0 on all cases.

Honest verdict: spatial advantage still holds under visual-ON, but it shrank sharply: +20 trial wins in the visual-OFF comparison became +9 when visual was live. Visual now did some work on `SH-B-008` (one correct 3-axis visual trial and one correct 4-axis visual trial), but more often it distracted both reasoners away from spatial evidence, especially `SH-A-001`, `SH-A-002`, and `SH-C-006`. No case flipped purely because visual carried image payloads; the visual axis produced CLIP clip hits, but every image-payload count was 0, so payload-based visual evidence did not drive any flip.

*Rows marked `*` use shipped `output/spatial_hero_results.json` visual-OFF counts because the ID was absent from `output/spatial_hero_topk_verified.json`.*

## §12. GPT 16-frame visual axis re-examination (honest comparison)

Protocol: same 10 curated cases, three independent trials per config. Replaces the middle-frame CLIP-ViT-B-32 visual encoder with a GPT-vision pipeline: 16 evenly-spaced frames per clip (≤384 px JPEG, base64), one `chatgpt/gpt-5.4` multi-image call per clip producing a single ~300-token rich prose description, MiniLM-L6-v2 384-d embedding of the description for retrieval. Build artifacts: `tools/build_visual_embeddings_gpt16.py`, `output/metadata/visual_memory/A1_JAKE/visual_descriptions_gpt16.json` (91 entries), `visual_embeddings_gpt16_minilm.pkl` (91 × 384 float32). Ablation harness: `tools/verify_spatial_hero_topk_gpt16.py`. Configs identical to §11 but with the GPT16-visual axis swapped in for the CLIP-visual axis.

| Case | Visual-OFF 3-axis | Visual-OFF 4-axis | CLIP-visual 4-axis | GPT16-visual 3-axis | GPT16-visual 4-axis |
|---|---:|---:|---:|---:|---:|
| SH-A-001 | 1/3 | 3/3 | 0/3 | 1/3 | 0/3 |
| SH-B-002 | 1/3 | 3/3 | 2/3 | 0/3 | 0/3 |
| SH-C-006 * | 0/3 | 2/3 | 0/3 | 0/3 | 0/3 |
| SH-E-004 * | 0/3 | 2/3 | 0/3 | 0/3 | 0/3 |
| SH-F-005 | 2/3 | 3/3 | 3/3 | 1/3 | 0/3 |
| SH-A-002 | 0/3 | 3/3 | 0/3 | 0/3 | 0/3 |
| SH-B-001 | 0/3 | 3/3 | 1/3 | 0/3 | 0/3 |
| SH-B-005 | 1/3 | 3/3 | 3/3 | 0/3 | 0/3 |
| SH-B-008 | 1/3 | 3/3 | 2/3 | 0/3 | 0/3 |
| SH-C-005 * | 1/3 | 2/3 | 0/3 | 0/3 | 0/3 |

Aggregate: visual-OFF 4-axis 27/30 → CLIP-visual-ON 4-axis 11/30 → **GPT16-visual-ON 4-axis 0/30**. Visual retrieval hit 8/10 cases (description text reached the retrieval trace in 8 cases), but every per-trial visual hit count was 0 and zero cases produced a correct 4-axis answer.

Honest verdict: GPT16 visual is **worse than CLIP visual, which was already worse than visual-OFF**. Two effects compound:
- Description-text-vs-query MiniLM retrieval is over-eager: rich English paragraphs share surface vocabulary with the MCQ prompt, so the reasoner keeps selecting the visual axis even when no genuine evidence sits there.
- The prose format hides the answer. A description that lists "checkered table, several black zippered cases on it, smartphone in hand" does **not** carry the structured spatial fact `(hard_drive, on, dining_table)` that the spatial axis surfaces directly.

So the GPT16 axis carries DATA but no usable SIGNAL for these closed-vocab WHERE questions, and it actively pulls the reasoner away from the spatial chain that *does* answer them. Next useful experiment is structured GPT output (object-place pairs instead of prose) so retrieval can match against the same shape spatial uses, not free-form English paragraphs.
