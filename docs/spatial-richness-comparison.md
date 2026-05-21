# Spatial Richness Comparison: EgoLife DAY1 vs AEA Spatial Sidecar

**Date:** 2026-05-21 KST  
**Scope:** Density comparison only. No ablation, no accuracy rerun.  
**Verdict:** AEA adds continuous metric-space signals that WorldMM's current EgoLife MP4 pipeline cannot recover from captions alone. EgoLife remains richer in symbolic predicates because the current pipeline is LLM-triple-first; AEA is pose/gaze-first and not yet clustered into places or converted into symbolic triples.

## Measurement Sources

| Source | File |
|---|---|
| EgoLife spatial triples | `output/metadata/spatial_memory/A1_JAKE/spatial_extraction_results_chatgpt-gpt-5.4.json` |
| EgoLife consolidated triples | `output/metadata/spatial_memory/A1_JAKE/spatial_consolidation_results_chatgpt-gpt-5.4.json` |
| EgoLife place graph | `output/spatial_trajectory_A1_JAKE_DAY1.json` |
| EgoLife DenseCaption SRT sample | `data/EgoLife/EgoLifeCap/DenseCaption/A1_JAKE/DAY1/A1_JAKE_DAY1_11000000.srt` |
| AEA sidecar chunks | `output/metadata/spatial_memory/AEA_loc5_script4_seq6_rec1/` |
| AEA pose stream | `data/AEA/loc5_script4_seq6_rec1/closed_loop_trajectory.csv` |
| AEA gaze stream | `data/AEA/loc5_script4_seq6_rec1/general_eye_gaze.csv` |
| AEA speech stream | `data/AEA/loc5_script4_seq6_rec1/speech.csv` |
| AEA MPS summary | `data/AEA/loc5_script4_seq6_rec1/summary.json` |
| AEA semidense points | `data/AEA/loc5_script4_seq6_rec1/semidense_points.csv.gz` |

## Normalization

EgoLife DAY1 duration measured from the task's effective chunk count: `91 chunks * 30 s = 2,730 s = 45.50 min`.

AEA duration measured from `summary.json`: `Recording total time: 0:03:33.167518`, so `213.167518 s = 3.552792 min`. The sidecar has 8 chunk JSON files because it uses 30 s windows over this 3:33 recording.

## Measured Density Table

| Signal | EgoLife DAY1 (45.50 min effective) | AEA loc5 seq6 rec1 (3.552792 min) | Notes |
|---|---:|---:|---|
| spatial triples / minute | 9.47 unique triples/min (#measured: 431 unique triples / 45.50 min; 598 raw emitted triples = 13.14/min) | 0.00 WorldMM triples/min (#measured: no `spatial_triples` field in 8 sidecar chunks; 8 pose-sidecar chunks = 2.25 chunks/min) | EgoLife: LLM-from-caption symbolic triples. AEA sidecar is pose/gaze/speech raw records at this stage, not converted to `(subject, predicate, object)` triples. |
| unique predicates used | 11 (#measured: `behind`, `contains`, `in`, `in_front_of`, `left_of`, `located_in`, `near`, `next_to`, `on`, `right_of`, `under`) | 0 (#measured: 8 chunk JSON files have pose/gaze/speech fields, no predicate field) | EgoLife rich symbolic vocab; AEA pose-only sidecar has no symbolic predicate layer yet. |
| place graph nodes | 33 places (#measured: unique `place` values in `output/spatial_trajectory_A1_JAKE_DAY1.json`) | 1 world frame (#measured: one `graph_uid` frame in chunk pose fields, treated as `aea_world_frame`) | AEA frame is metric, not semantic. Place labels need clustering or semantic naming. |
| place graph edges / transitions | 48 stops (#measured: `stops` length in `output/spatial_trajectory_A1_JAKE_DAY1.json`) | n/a | AEA pose stream has implicit dwell regions, but explicit DBSCAN place graph not built. |
| 6-DoF pose samples / minute | 0 (#measured: no pose stream in EgoLife MP4/SRT outputs audited for this pipeline) | 60,245.86 samples/min (#measured: 214,041 `closed_loop_trajectory.csv` rows / 3.552792 min) | AEA closed-loop trajectory is ~1 kHz 6-DoF pose; EgoLife current pipeline has none. |
| gaze samples / minute | 0 (#measured: no gaze stream in EgoLife MP4/SRT outputs audited for this pipeline) | 599.81 samples/min (#measured: 2,131 `general_eye_gaze.csv` rows / 3.552792 min) | AEA gaze is ~10 Hz. Sidecar emits 40 sampled gaze records across 8 chunk JSONs, but density here uses raw CSV rows. |
| speech segments / minute | 2,956.00 DenseCaption entries/min for one 30 s SRT sample (#measured: 1,478 numbered entries in `A1_JAKE_DAY1_11000000.srt` / 0.5 min) | 0.28 segments/min (#measured: 1 `speech.csv` row / 3.552792 min) | EgoLife value is DenseCaption SRT-entry density from one local file as required, not human speech turn density. AEA sequence is quiet: one low-confidence utterance, `you.`, confidence 0.008. |
| semidense world points | n/a | 1,115,096 points (#measured: rows in `semidense_points.csv.gz`) | Optional AEA sidecar; not normalized because pointcloud is a map artifact, not per-time sample stream. |
| trajectory length / minute | n/a | 17.45 m/min (#measured: 62.00 m from `summary.json` / 3.552792 min; sidecar integrated length is 62.33 m) | AEA only. EgoLife MP4 pipeline has no metric trajectory. |

## Axis-by-Axis Qualitative Difference

**Spatial triples.** EgoLife current pipeline converts captions/visual descriptions into closed-vocab symbolic triples. This is why it has 431 unique symbolic relations and 11 predicates. AEA sidecar, as built, does not yet emit symbolic triples; it adds raw metric substrate. Net: EgoLife answers object-relation questions now; AEA makes those relations place- and pose-groundable once a mapper or extractor attaches objects to the pose stream.

**Pose ground truth.** EgoLife MP4 alone could support visual SLAM in principle, but WorldMM did not build SLAM, and the released pipeline has no pose rows. AEA provides closed-loop 6-DoF poses directly: 214,041 measured rows over 213.17 s. That turns every chunk into metric position/orientation instead of inferred scene text like `bedroom` or `dining_table`.

**Gaze salience.** EgoLife MP4 treats the visible frame as the evidence field; attention must be inferred from captions, pointing language, or object salience. AEA provides 2,131 gaze samples, about 599.81/min, aligned to the same tracking timestamp domain as pose. This distinguishes what was likely attended from what merely appeared in the camera view.

**Speech timing.** EgoLife has local DenseCaption SRT files, but the measured SRT row density is caption-event density, not a sensor-aligned speech stream. AEA `speech.csv` has millisecond-scale start/end timestamps and confidence, although this sequence has only one low-confidence segment. For speech-rich sequences, that timing can answer “what was the wearer looking at when speaking?” without aligning a transcript post hoc.

**Trajectory and place inference.** EgoLife place graph is scene-text-driven: 48 stops and 33 place strings extracted from spatial triples. AEA has no semantic graph yet, but it has continuous trajectory length and 6-DoF pose. DBSCAN or dwell clustering can produce place anchors from physical movement rather than from caption labels.

## Where This Matters for Spatial Memory Accuracy

The existing spatial-hero wins would not automatically become correct from pose alone. Example curated case `SH-A-001` asks: “Where was hard drive at approximately DAY1 11:14:30?” The winning spatial chain is `(hard drive, on, dining table)`, while the three-axis baseline chose `bedroom`. AEA-style pose would not directly say “hard drive on dining table”; object grounding still needs vision or LLM extraction. But AEA would improve the surrounding index: the `DAY1 11:14:30` evidence could be anchored to a stable dwell region, gaze could indicate whether the hard drive/table was attended, and repeated visits to the same physical table could be grouped even if captions alternate between `bedroom`, `dining_table`, and generic `room`. Honest projection: the same 9/10 spatial-hero cases would benefit most in retrieval and disambiguation of place/trajectory context, not in replacing object-relation extraction.

## Recommended Next Experiments

- Build AEA-style spatial-hero questions on the loc5 sequence, especially gaze-anchored questions such as “what was the wearer looking at when speaking?”
- Implement DBSCAN place clustering on the pose stream and visualize per-chunk `place_anchor`.
- Process one outdoor and speech-rich AEA sequence to validate GPS and speech coverage; current loc5 sequence is indoor and quiet.

## Reproducibility Notes

Metrics were computed with `uv run python` using `json`, `pandas`, `gzip`, and `re`; no packages installed and no ablation run. Consolidated EgoLife output contains the same 431 unique triples as extraction but repeated across consolidation timestamps, so the table uses extraction unique triples for density. AEA sidecar chunk files are measured as raw sidecars because they contain `pose_6dof_mean`, `pose_6dof_median`, `gaze_samples`, and `speech_segments`, not symbolic predicates.
