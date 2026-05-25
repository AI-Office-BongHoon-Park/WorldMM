# Golden QA Sample: Spatial Axis Isolation

Date: 2026-05-25 KST

## Purpose

This golden set isolates when the fourth spatial axis adds unique value, when it merely supports episodic/semantic evidence, when it should be neutral, and when spatial-looking wording can distract retrieval.

## Categorization Methodology

| Type | Definition | Rule used here |
|---|---|---|
| Type1 | Spatial-required | full-DAY1 `spatial_extraction_results_chatgpt-gpt-5.4.json` real object/place triples converted to closed-vocab WHERE questions. |
| Type2 | Spatial-supported | EgoLifeQA text matches place-noun regex and action/person predicate regex; answer remains in `reason`. |
| Type3 | Non-spatial control | EgoLifeQA answer derivable from `reason`; question lacks required spatial predicate. |
| Type4 | Spatial-distractor | WHERE-shaped question synthesized from EgoLifeQA TaskMaster/EventRecall rows; gold comes from narration/reason, not spatial triples. |

## Per-Type Counts

| Type | Count | Source |
|---|---:|---|
| Type1 | 24 | full-DAY1 spatial extraction triples, 828 chunks / 1,262 triples |
| Type2 | 12 | EgoLifeQA mixed place+action/person rows |
| Type3 | 12 | EgoLifeQA action/preference/planning controls |
| Type4 | 6 | synthesized WHERE-shaped distractors from EgoLifeQA narration |

## Ablation Table

| Type | spatial-OFF correct/N | spatial-ON correct/N | Δ | spatial-axis verdict |
|---|---:|---:|---:|---|
| 1 | 53/24 x 3 (73.6%) | 50/24 x 3 (69.4%) | -3 (-4.2 pp) | not cleanly required |
| 2 | 16/12 x 3 (44.4%) | 16/12 x 3 (44.4%) | +0 (+0.0 pp) | not helpful in this run |
| 3 | 14/12 x 3 (38.9%) | 20/12 x 3 (55.6%) | +6 (+16.7 pp) | signal leak |
| 4 | 5/6 x 3 (27.8%) | 8/6 x 3 (44.4%) | +3 (+16.7 pp) | no distractor penalty |

Execution note: `output/golden_qa_results.json` contains 3 independent trials per config per question, 324 direct LLM calls with 4 workers. Full memory-stack retrieval was attempted first but exceeded the 2-hour cap; completed results use the direct golden-evidence protocol recorded in JSON metadata.

## Honest Verdict

Type1 lift did **not** improve versus n=91: it changed from +47.2 pp to -4.2 pp. Type3 signal leak did not shrink; it changed from -12.5 pp to +16.7 pp. Type2 stayed neutral. Type4 still showed no distractor penalty.

## Example Cases

### Top Type1 Cases By ON-OFF Margin

- `GQA-T1-SP-111143000-03`: Where was box at approximately DAY1 11:14:30? Gold `C` (shelf); OFF 0/3, ON 3/3, Δ +3. Evidence: spatial_triple at chunk DAY1 11:14:30: (box, on, shelf)
- `GQA-T1-SP-112093000-07`: Where was director's board at approximately DAY1 12:09:30? Gold `C` (left-hand side); OFF 0/3, ON 2/3, Δ +2. Evidence: spatial_triple at chunk DAY1 12:09:30: (director's board, on, left-hand side)
- `GQA-T1-SP-112080000-01`: Where was mixing at approximately DAY1 12:08:00? Gold `C` (counter); OFF 3/3, ON 3/3, Δ +0. Evidence: spatial_triple at chunk DAY1 12:08:00: (mixing, on, counter)
- `GQA-T1-SP-112073000-01`: Where was director's slate at approximately DAY1 12:07:30? Gold `B` (floor); OFF 3/3, ON 3/3, Δ +0. Evidence: spatial_triple at chunk DAY1 12:07:30: (director's slate, on, floor)
- `GQA-T1-SP-112043000-04`: Where was data cable at approximately DAY1 12:04:30? Gold `B` (table); OFF 3/3, ON 3/3, Δ +0. Evidence: spatial_triple at chunk DAY1 12:04:30: (data cable, on, table)
