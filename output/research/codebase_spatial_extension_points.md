# WorldMM spatial extension points for AI Glass signals

## 1) Spatial extraction → ingestion path
- `/home/default/workspace/WorldMM/preprocess/spatial_memory/extract_spatial_triples.py:35-61` groups caption chunks by `period`, pulls OpenIE triples by chunk hash, and builds the per-chunk batch passed into spatial extraction.
- `/home/default/workspace/WorldMM/preprocess/spatial_memory/extract_spatial_triples.py:64-111` runs `SpatialExtraction.batch_spatial_extraction(...)` and writes `spatial_extraction_results_<model>.json`.
- `/home/default/workspace/WorldMM/src/worldmm/memory/spatial/spatial_extraction.py:21-43` is the actual LLM extraction entrypoint per chunk.
- `/home/default/workspace/WorldMM/src/worldmm/memory/spatial/utils.py:6-18` and `:21-39` fix the closed predicate set and reject any triple whose predicate is not in vocab.
- `/home/default/workspace/WorldMM/src/worldmm/llm/templates/spatial_extraction.py:5-37` hard-codes the same closed set in the prompt.
- `/home/default/workspace/WorldMM/preprocess/spatial_memory/consolidate_spatial_memory.py:27-103` consolidates chunk outputs over time and emits `consolidated_spatial_triples`.
- `/home/default/workspace/WorldMM/src/worldmm/memory/spatial/spatial_consolidation.py:50-82` and `/home/default/workspace/WorldMM/src/worldmm/llm/templates/spatial_consolidation.py:6-23` enforce 3-item triples + closed-vocab predicates during merge.

How to add new triple shape:
- Today shape is fixed to `[subject, predicate, object]` in `SpatialRawOutput` and `SpatialConsolidationRawOutput` (`utils.py:21-58`).
- For `(object, at_3d_pose, x_y_z)` or `(person, gaze_at, object)`, first widen the validators in `utils.py`, then update the prompts in `spatial_extraction.py` / `spatial_consolidation.py`, then update `SpatialTripleEntry` and any display/retrieval code that assumes `subject/predicate/object` only.

## 2) Storage layer
- `/home/default/workspace/WorldMM/src/worldmm/memory/spatial/memory.py:25-35` defines `SpatialTripleEntry` fields: `id, subject, predicate, object, timestamp, place, subject_grounding, object_grounding`.
- `/home/default/workspace/WorldMM/src/worldmm/memory/spatial/memory.py:117-180` loads triples, filters by `SPATIAL_PREDICATE_VOCAB`, and optionally attaches `GeometricGrounding` sidecars.
- `/home/default/workspace/WorldMM/src/worldmm/memory/spatial/memory.py:209-271` indexes a single timestamp slice.
- `/home/default/workspace/WorldMM/src/worldmm/memory/spatial/memory.py:278-346` retrieves and renders entries.

Clean slot for `gaze_target_id` / `head_yaw_at_emit`:
- `SpatialTripleEntry` has no sensor fields today.
- Best additive spot: new optional dataclass fields on `SpatialTripleEntry` plus matching JSON sidecar keys in `load_triples_from_data()` (`memory.py:163-175`).
- If the data is per-event, not just per-side geometry, add fields at entry level, not inside `GeometricGrounding`.

## 3) Grounding sidecar
- `/home/default/workspace/WorldMM/src/worldmm/memory/spatial/grounding.py:8-37` defines `GeometricGrounding`.
- It is bbox-centric: `bbox_center`, `bbox_extent`, `units`, `source`, `confidence`, `keyframe_ts`, `instance_disambiguation`.
- `model_config = ConfigDict(extra="forbid")` blocks unknown sensor fields.

Extension recommendation:
- For IMU/GPS/magnetometer/head pose, a new sibling model is cleaner than stuffing them into `GeometricGrounding`.
- Reason: current validators and names are 3D box semantics, not wearable pose semantics.
- If you must keep one model, use optional fields and relax `extra`, but that turns bbox grounding into a mixed sensor envelope.

## 4) Builders
### Geometric grounding builder
- `/home/default/workspace/WorldMM/tools/build_geometric_grounding.py:70-80` extracts one keyframe from one chunk.
- `/home/default/workspace/WorldMM/tools/build_geometric_grounding.py:141-161` lifts a bbox into `GeometricGrounding`.
- `/home/default/workspace/WorldMM/tools/build_geometric_grounding.py:168-215` is the main per-chunk loop: load triples, pick `target_indices`, get one frame, compute depth, then build the sidecar per triple.

AI Glass plug-in point:
- Insert IMU→pose fusion before or beside `extract_frame()` and `lift_bbox()`.
- The current loop is chunk-based, not stream-based, so continuous wearable data needs a new pre-alignment step that maps sensor samples to `chunk_id` / `keyframe_ts`.

### Visual triples builder
- `/home/default/workspace/WorldMM/tools/build_visual_triples_gpt_structured.py:42-58` is the prompt with the closed predicate list.
- `/home/default/workspace/WorldMM/tools/build_visual_triples_gpt_structured.py:131-156` validates output triples.
- `/home/default/workspace/WorldMM/tools/build_visual_triples_gpt_structured.py:159-169` is the per-clip inference call over 16 sampled frames.
- `/home/default/workspace/WorldMM/tools/build_visual_triples_gpt_structured.py:207-239` builds the index and stores `frame_timestamp_s`.

AI Glass plug-in point:
- Replace/augment `frames` with sensor-aligned multimodal inputs, but keep `build_index()` as the place where timestamps get attached to each triple.

## 5) Retrieval
- `/home/default/workspace/WorldMM/src/worldmm/memory/spatial/memory.py:278-343` is core retrieval.
- It takes only `query`, `top_k`, `as_context`; no location-radius, gaze-target, or sensor filter argument.
- Time only enters through `index(until_time)` at `memory.py:209-271`.
- Retrieval score = embedding similarity + PPR over entity graph.

Wrapper attachment points:
- `/home/default/workspace/WorldMM/tools/grounded_spatial_retrieve.py:71-112` consumes `memory.retrieve(...)`, then resolves `first_seen_ts` from extraction JSON and adds thumbnails.
- `/home/default/workspace/WorldMM/tools/grounded_spatial_retrieve.py:130-193` is the CLI entrypoint that can grow new query args like `--radius`, `--lat`, `--lng`, `--gaze-target`.
- `/home/default/workspace/WorldMM/tools/grounded_visual_retrieve.py:56-89` is the visual analogue; retrieval is pure similarity over an index with `frame_timestamp_s` payload.

Direct answer:
- No native spatial-radius retrieval exists.
- Best place: add optional filter/pre-filter in the wrapper tools, then add a dedicated method on `SpatialMemory` only if you need the filter to affect ranking.

## 6) Reasoning prompt
- `/home/default/workspace/WorldMM/src/worldmm/llm/templates/memory_reasoning.py:16-21` defines the 4 memory types and the spatial axis description.
- `/home/default/workspace/WorldMM/src/worldmm/llm/templates/memory_reasoning.py:40-45` includes `spatial` in the output enum.
- `/home/default/workspace/WorldMM/src/worldmm/llm/templates/memory_reasoning.py:121-152` and `:154-181` show spatial examples.
- `/home/default/workspace/WorldMM/src/worldmm/llm/templates/memory_reasoning_3axis.py:18-23` removes spatial entirely.
- `/home/default/workspace/WorldMM/src/worldmm/llm/templates/memory_reasoning_essp.py:18-22` keeps spatial in the ablation prompt.

Closed-vocab predicate list:
- Canonical set lives in `/home/default/workspace/WorldMM/src/worldmm/memory/spatial/utils.py:6-18`.
- Prompt mirrors live in `/home/default/workspace/WorldMM/src/worldmm/llm/templates/spatial_extraction.py:16-18` and `/home/default/workspace/WorldMM/src/worldmm/llm/templates/spatial_consolidation.py:16-18`.
- Visual builder also mirrors it in `/home/default/workspace/WorldMM/tools/build_visual_triples_gpt_structured.py:29-39`.

New predicates like `gaze_at`, `facing`, `dwelling_at`, `held_in_hand`:
- Add to `SPATIAL_PREDICATE_VOCAB` first.
- Then update the extraction/consolidation prompts and any validator using that vocab.
- `memory_reasoning.py` itself only needs wording/examples if you want the LLM to query by those new concepts.

## 7) Data schema
- `/home/default/workspace/WorldMM/preprocess/build_memory.py:28-36` discovers videos by `caption_dir/<video_id>/10sec.json`.
- `/home/default/workspace/WorldMM/preprocess/build_memory.py:60-74` episodic build starts from that caption chunk.
- `/home/default/workspace/WorldMM/preprocess/build_memory.py:77-115` semantic build consumes episodic OpenIE output.
- `/home/default/workspace/WorldMM/preprocess/build_memory.py:118-158` spatial build consumes the same caption/OpenIE pair and writes `output/metadata/spatial_memory/<video_id>/...`.
- `/home/default/workspace/WorldMM/preprocess/build_memory.py:207-272` is the top-level orchestrator; `spatial` is already a first-class step.
- `/home/default/workspace/WorldMM/script/3_build_memory.sh:72-84` is the shell pipeline for spatial extraction + consolidation.
- `/home/default/workspace/WorldMM/preprocess/spatial_memory/consolidate_spatial_memory.py:88-102` shows the canonical consolidated output schema: timestamp-keyed map with `consolidated_spatial_triples`.

Where AI Glass sensor-stream chunks attach:
- Best fit today: same per-video build tree, new sibling inputs next to `10sec.json` and new outputs next to `spatial_memory/<video_id>/...`.
- No native sensor-stream schema exists in `build_memory.py`; you would add a new build step and keep chunk IDs aligned with the caption/timestamp keys.

## 8) Display strings
- `/home/default/workspace/WorldMM/src/worldmm/memory/spatial/memory.py:40-44` defines raw text rendering.
- `/home/default/workspace/WorldMM/src/worldmm/memory/spatial/memory.py:46-74` renders triples for prompts.
- Current format is either `(subject, predicate, object)` or `(subject) [predicate] (object)` plus optional place and bbox center/extent.

IMU/GPS-augmented display:
- Add sensor suffixes in `to_display_str()`.
- Example shape: `(... ) [gaze=..., yaw=..., gps=..., heading=...]`.
- If the new signal lives in a separate model, render it in a second bracket block rather than overloading bbox text.
