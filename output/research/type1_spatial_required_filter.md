# Type 1 Spatial-Required Filter

## Methodology
- Audited only rows with `type_label == Type1` from `output/golden_qa_sample.json`.
- For episodic memory, checked the `evidence_chunk` plus one previous and one next episodic chunk.
- For semantic memory, streamed the 131 MB consolidated file and checked the same +/- one chunk window in semantic chunk order.
- For DenseCaption and Transcript SRT, checked segments overlapping `evidence_chunk_label` +/- 30 seconds.
- Matching was deterministic: case-insensitive normalized substring plus basic singularization of the gold answer. No LLM and no synonym expansion beyond text that appears verbatim.

## Per-Q Audit
| id | gold_answer | found-in-episodic | found-in-semantic | found-in-densecaption | found-in-transcript | truly_spatial_required |
|---|---|---|---|---|---|---|
| GQA-T1-SP-111110000-01 | dining table | yes | yes | no | no | no |
| GQA-T1-SP-111113000-01 | table | yes | yes | no | no | no |
| GQA-T1-SP-111120000-02 | another box | yes | no | no | no | no |
| GQA-T1-SP-111123000-01 | shelf | yes | no | no | no | no |
| GQA-T1-SP-111143000-02 | table | no | yes | no | no | no |
| GQA-T1-SP-111143000-03 | shelf | no | no | no | no | yes |
| GQA-T1-SP-111170000-02 | left-hand side | yes | yes | no | no | no |
| GQA-T1-SP-111213000-01 | side | yes | yes | no | no | no |
| GQA-T1-SP-111233000-01 | stool | yes | yes | no | no | no |
| GQA-T1-SP-111243000-10 | stool | no | yes | no | no | no |
| GQA-T1-SP-111293000-02 | board | yes | yes | no | no | no |
| GQA-T1-SP-111343000-02 | whiteboard | yes | yes | no | no | no |
| GQA-T1-SP-111343000-08 | table | yes | yes | no | no | no |
| GQA-T1-SP-111363000-03 | camera | yes | yes | no | no | no |
| GQA-T1-SP-111363000-07 | camera | yes | yes | no | no | no |
| GQA-T1-SP-111443000-03 | dining table | yes | yes | no | no | no |
| GQA-T1-SP-111470000-01 | Earth | yes | yes | no | yes | no |
| GQA-T1-SP-111493000-01 | Earth | no | yes | no | no | no |
| GQA-T1-SP-112043000-02 | whiteboard | no | yes | no | no | no |
| GQA-T1-SP-112043000-04 | table | yes | yes | no | no | no |
| GQA-T1-SP-112073000-01 | floor | yes | no | no | no | no |
| GQA-T1-SP-112080000-01 | counter | yes | no | no | yes | no |
| GQA-T1-SP-112093000-06 | floor | no | no | no | no | yes |
| GQA-T1-SP-112093000-07 | left-hand side | no | yes | no | no | no |

## Summary
- Truly spatial-required: 2/24 (8.3%). Leaked/recoverable from non-spatial text: 22/24 (91.7%).
- Truly spatial-required dominant patterns: subjects box (1), director s slate (1); predicates on (2); subject/predicate box/on (1), director s slate/on (1).
- Leaked/recoverable dominant patterns: subjects box (3), shure (3), hard drives (2), we (2), hard drive (1); predicates on (22); subject/predicate box/on (3), shure/on (3), hard drives/on (2), we/on (2), hard drive/on (1).

## Recommendation
- Keep for the deck: GQA-T1-SP-111143000-03, GQA-T1-SP-112093000-06.
- Drop or relabel before claiming a spatial-axis-only effect: GQA-T1-SP-111110000-01, GQA-T1-SP-111113000-01, GQA-T1-SP-111120000-02, GQA-T1-SP-111123000-01, GQA-T1-SP-111143000-02, GQA-T1-SP-111170000-02, GQA-T1-SP-111213000-01, GQA-T1-SP-111233000-01, GQA-T1-SP-111243000-10, GQA-T1-SP-111293000-02, GQA-T1-SP-111343000-02, GQA-T1-SP-111343000-08, GQA-T1-SP-111363000-03, GQA-T1-SP-111363000-07, GQA-T1-SP-111443000-03, GQA-T1-SP-111470000-01, GQA-T1-SP-111493000-01, GQA-T1-SP-112043000-02, GQA-T1-SP-112043000-04, GQA-T1-SP-112073000-01, GQA-T1-SP-112080000-01, GQA-T1-SP-112093000-07.
- Re-running ablation on this subset would change the Type1 pool by removing 22/24 (91.7%) cases and keeping 2/24 (8.3%). The Phase 3b -4.2pp result should be treated as diluted by leaked Type1 cases until the subset is re-run; direction or magnitude cannot be derived from filtering alone.
