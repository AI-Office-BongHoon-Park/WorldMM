# Golden QA Sample: Spatial Axis Isolation

Date: 2026-05-24 KST

## Purpose

This golden set isolates when the fourth spatial axis adds unique value, when it merely supports episodic/semantic evidence, when it should be neutral, and when spatial-looking wording can distract retrieval.

## Categorization Methodology

Each question receives exactly one label.

| Type | Definition | Rule used here |
|---|---|---|
| Type1 | Spatial-required | spatial-hero closed-vocab object/place templates; gold appears in `evidence_triples` / spatial triples only. |
| Type2 | Spatial-supported | EgoLifeQA text matches place-noun regex and action/person predicate regex; answer remains in `reason`. |
| Type3 | Non-spatial control | EgoLifeQA answer derivable from `reason`; question lacks required spatial predicate. |
| Type4 | Spatial-distractor | WHERE-shaped question synthesized from EgoLifeQA TaskMaster/EventRecall rows; gold comes from narration/reason, not spatial triples. |

Deterministic first pass used `PLACE_RE`, spatial predicate vocabulary, and gold-answer shape. LLM fallback was reserved for ambiguous rows but not used in default build. Type4 synthesis is explicit because source pools did not provide four clean natural distractors.

## Per-Type Counts

| Type | Count | Source |
|---|---:|---|
| Type1 | 12 | spatial-hero V2 curated/top-k catalog and pool |
| Type2 | 8 | EgoLifeQA mixed place+action/person rows |
| Type3 | 8 | EgoLifeQA action/preference/planning controls |
| Type4 | 4 | synthesized WHERE-shaped distractors from EgoLifeQA narration |

## Ablation Table

| Type | spatial-OFF correct/N | spatial-ON correct/N | Δ | spatial-axis verdict |
|---|---:|---:|---:|---|
| 1 | 15/12 x 3 (41.7%) | 32/12 x 3 (88.9%) | +17 (+47.2 pp) | required |
| 2 | 18/8 x 3 (75.0%) | 18/8 x 3 (75.0%) | +0 (+0.0 pp) | not helpful in this run |
| 3 | 12/8 x 3 (50.0%) | 9/8 x 3 (37.5%) | -3 (-12.5 pp) | signal leak |
| 4 | 9/4 x 3 (75.0%) | 9/4 x 3 (75.0%) | +0 (+0.0 pp) | no distractor penalty |

Execution note: Type1 rows use `output/spatial_hero_topk_verified.json`, a real three-trial spatial-hero run. Type2-Type4 rows use `output/three_vs_four_n60.json` replayed into three trial slots because the fresh tmux run began but exceeded practical wall time after the first-question smoke test; those rows should be treated as broad-run evidence, not independent multi-seed evidence.

## Broad EgoLifeQA Comparison

Existing broad comparison from `docs/three-vs-four-axis-ablation.md`: unfiltered EgoLifeQA n=60 was 37/60 (61.7%) for 3-axis and 36/60 (60.0%) for 4-axis, Δ -1.7 pp. Existing §8.1 visual-coverage rerun n=30 was 15/30 (50.0%) for 3-axis and 16/30 (53.3%) for 4-axis, Δ +3.3 pp. The golden subset shows a much larger Type1 lift (+47.2 pp), but Type3/Type4 do not remain cleanly neutral/distracting.

## Honest Verdict

spatial axis does not differentiate cleanly: Type3/Type4 still leak or fail to show distractor penalty.

Type1 differentiates strongly: spatial ON wins when gold is in triples. Type2 did not show additional lift in this replayed broad-run slice. Type3 leaked signal negatively, meaning adding spatial changed some non-spatial answers. Type4 did not hurt; WHERE-shaped distractors tied instead of producing the expected penalty.

## Example Cases

### Type1

- `GQA-T1-SH-A-001`: Where was hard drive at approximately DAY1 11:14:30?
  - Gold: A (dining table)
  - Evidence: spatial_triple at chunk DAY1 11:14:30: (hard drive, on, dining table)
  - 3-axis trace: pred B; R1 episodic: [DAY1 11:14:00 - DAY1 11:14:30] "Yes, your second life is all in here," I say as I look back and forth. I point to th...
  - 4-axis trace: pred A; R1 spatial: (hard drive, on, dining_table) (box, on, another_box) (I, located_in, bedroom) (data, in, hard_drive)

- `GQA-T1-SH-A-002`: Where was box at approximately DAY1 11:14:30?
  - Gold: C (another box)
  - Evidence: spatial_triple at chunk DAY1 11:14:30: (box, on, another box)
  - 3-axis trace: pred B; R1 episodic: [DAY1 11:23:00 - DAY1 11:23:30] I hold the box with both hands, grab it back, set it upright, and close it. Shure say...
  - 4-axis trace: pred C; R1 spatial: (hard drive, on, dining_table) (box, on, another_box) (I, located_in, bedroom) (data, in, hard_drive)

- `GQA-T1-SH-A-003`: Where was data at approximately DAY1 11:14:30?
  - Gold: D (hard drive)
  - Evidence: spatial_triple at chunk DAY1 11:14:30: (data, in, hard drive)
  - 3-axis trace: pred D; R1 episodic: [DAY1 11:12:00 - DAY1 11:15:00] **I invited the group into my bedroom to view the workstation and introduced the reco...
  - 4-axis trace: pred D; R1 spatial: (hard drive, on, dining_table) (box, on, another_box) (I, located_in, bedroom) (data, in, hard_drive)

### Type2

- `GQA-Type2-EGO-10`: Who was the first to write the meeting notes on the whiteboard in the meeting room?
  - Gold: A (Shure)
  - Evidence: EgoLifeQA row ID 10 reason field: I saw Shure writing on the whiteboard
  - 3-axis trace: pred A; Replay of existing broad EgoLifeQA n=60 three_axis result; original output does not retain round trace.
  - 4-axis trace: pred A; Replay of existing broad EgoLifeQA n=60 four_axis result; original output does not retain round trace.

- `GQA-Type2-EGO-12`: When I handed out the charging cables just now, who had already started charging?
  - Gold: D (Shure)
  - Evidence: EgoLifeQA row ID 12 reason field: I noticed Shure had already plugged in the power bank
  - 3-axis trace: pred D; Replay of existing broad EgoLifeQA n=60 three_axis result; original output does not retain round trace.
  - 4-axis trace: pred D; Replay of existing broad EgoLifeQA n=60 four_axis result; original output does not retain round trace.

- `GQA-Type2-EGO-15`: While puzzling, who am I guiding to use the app on the phone
  - Gold: B (Katrina, Alice)
  - Evidence: EgoLifeQA row ID 15 reason field: In the video, I am guiding Katrina and Alice on how to use this app
  - 3-axis trace: pred B; Replay of existing broad EgoLifeQA n=60 three_axis result; original output does not retain round trace.
  - 4-axis trace: pred B; Replay of existing broad EgoLifeQA n=60 four_axis result; original output does not retain round trace.

### Type3

- `GQA-Type3-EGO-13`: Now Lucia asks, "What is this, is this a hedgehog?" What were we discussing the last time she asked a question?
  - Gold: C (Cake Making)
  - Evidence: EgoLifeQA row ID 13 reason field: Lucia asked if it was like pudding
  - 3-axis trace: pred B; Replay of existing broad EgoLifeQA n=60 three_axis result; original output does not retain round trace.
  - 4-axis trace: pred B; Replay of existing broad EgoLifeQA n=60 four_axis result; original output does not retain round trace.

- `GQA-Type3-EGO-20`: We are currently discussing the details of the puzzle. When was this discussed before?
  - Gold: D (About an hour ago)
  - Evidence: EgoLifeQA row ID 20 reason field: Alice said it wasn't completely assembled
  - 3-axis trace: pred B; Replay of existing broad EgoLifeQA n=60 three_axis result; original output does not retain round trace.
  - 4-axis trace: pred B; Replay of existing broad EgoLifeQA n=60 four_axis result; original output does not retain round trace.

- `GQA-Type3-EGO-24`: What are the lunch options?
  - Gold: A (Beijing-style Braised Pork)
  - Evidence: EgoLifeQA row ID 24 reason field: Shure is naming the dishes
  - 3-axis trace: pred B; Replay of existing broad EgoLifeQA n=60 three_axis result; original output does not retain round trace.
  - 4-axis trace: pred B; Replay of existing broad EgoLifeQA n=60 four_axis result; original output does not retain round trace.

### Type4

- `GQA-T4-SYN-001`: Where did the lunch plan land after discussion?
  - Gold: D (KFC)
  - Evidence: EgoLifeQA row ID 32 reason field: I said let's order KFC
  - 3-axis trace: pred D; Replay of existing broad EgoLifeQA n=60 three_axis result; original output does not retain round trace.
  - 4-axis trace: pred D; Replay of existing broad EgoLifeQA n=60 four_axis result; original output does not retain round trace.

- `GQA-T4-SYN-002`: Where did the group land for lunch after everyone discussed it?
  - Gold: A (KFC)
  - Evidence: EgoLifeQA row ID 34 reason field: Because after everyone's discussion, we decided to eat Kfc
  - 3-axis trace: pred A; Replay of existing broad EgoLifeQA n=60 three_axis result; original output does not retain round trace.
  - 4-axis trace: pred A; Replay of existing broad EgoLifeQA n=60 four_axis result; original output does not retain round trace.

- `GQA-T4-SYN-003`: Where did Shure, Katrina, and Lucia's shopping role land?
  - Gold: D (Alcohol)
  - Evidence: EgoLifeQA row ID 38 reason field: Tasha pointed at Shure's shopping cart and said this is the alcohol group
  - 3-axis trace: pred B; Replay of existing broad EgoLifeQA n=60 three_axis result; original output does not retain round trace.
  - 4-axis trace: pred B; Replay of existing broad EgoLifeQA n=60 four_axis result; original output does not retain round trace.

## Reproduction

```bash
uv run python tools/build_golden_qa_sample.py --egolifeqa-file data/EgoLife/EgoLifeQA/EgoLifeQA_A1_JAKE.json --synthetic-pool output/spatial_hero_pool.json --spatial-hero-curated output/spatial_hero_curated.json --out output/golden_qa_sample.json --target-balance 12-8-8-4
tmux new-session -d -s golden-qa-ablation "cd /home/default/workspace/WorldMM && uv run python tools/run_golden_qa_ablation.py --sample output/golden_qa_sample.json --output output/golden_qa_results.json --trials 3 --model chatgpt-gpt-5.4 --max-rounds 3"
```
