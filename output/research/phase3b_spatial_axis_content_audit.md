# Phase 3b spatial axis content audit

## Scope
Audit target: determine what the Phase 3b spatial axis actually loaded at runtime. Focus: whether the run carried only text triples or also grounding / pose / point cloud / scene latent sidecars.

## 1) Orchestrator wiring

### `/tmp/opencode/phase3b_ablation_orchestrator.py`
- The subprocess passes only `--spatial-file` into the worker invocation, no grounding sidecar path.
- Evidence: lines 229-251 show `--spatial-file`, but no `--grounding-file`, `--scene-latent-ref`, `--point-cloud`, or `--pose-6dof` args.
- Default spatial file: lines 312-320 set `--spatial-file` to `output/metadata/spatial_memory/A1_JAKE/spatial_consolidation_results_chatgpt-gpt-5.4.json`.

Citations:
- `/tmp/opencode/phase3b_ablation_orchestrator.py:229-251`
- `/tmp/opencode/phase3b_ablation_orchestrator.py:312-320`

### `/home/default/workspace/WorldMM/tools/run_golden_qa_ablation.py`
- `build_memory(...)` call passes `semantic_file=args.semantic_file` and `spatial_file=args.spatial_file` only.
- No grounding sidecar passed into `build_memory`.
- Default spatial file matches the same JSON path.

Citations:
- `/home/default/workspace/WorldMM/tools/run_golden_qa_ablation.py:103-115`
- `/home/default/workspace/WorldMM/tools/run_golden_qa_ablation.py:166-177`

### Shared memory construction
- `WorldMemory.load_spatial_triples(...)` delegates to `self.spatial_memory.load_triples_from_file(file_path)` with no grounding arg.
- So the orchestrators can only load geometry if the downstream `SpatialMemory.load_triples_from_file(...)` somehow synthesizes it from the triples JSON itself. It does not.

Citation:
- `/home/default/workspace/WorldMM/src/worldmm/memory/memory.py:167-182`

## 2) SpatialMemory load path

### `/home/default/workspace/WorldMM/src/worldmm/memory/spatial/memory.py`
- `load_triples_from_file(self, file_path, grounding_file=None)` loads `grounding_data = None` unless a grounding file path is explicitly supplied.
- In `load_triples_from_data(...)`, geometry fields are read only from `grounding_record = (grounding_data or {}).get(triple_id, {})`.
- `subject_grounding`, `object_grounding`, `pose_6dof`, `scene_latent_ref`, and `point_cloud` are all populated only from that grounding record.
- If `grounding_file=None`, all those fields stay `None`.
- `to_display_str()` renders geometry only when those fields exist; otherwise it returns plain text triple shape `(subject, predicate, object)`.

Citations:
- `/home/default/workspace/WorldMM/src/worldmm/memory/spatial/memory.py:146-153`
- `/home/default/workspace/WorldMM/src/worldmm/memory/spatial/memory.py:190-214`
- `/home/default/workspace/WorldMM/src/worldmm/memory/spatial/memory.py:52-103`

## 3) Spatial consolidation JSON schema

### `/home/default/workspace/WorldMM/output/metadata/spatial_memory/A1_JAKE/spatial_consolidation_results_chatgpt-gpt-5.4.json`
First two entries:

- `111101208` → `consolidated_spatial_triples: [["I", "near", "dining_table"]]`
- `111103000` → `consolidated_spatial_triples: [["I", "near", "dining_table"]]`

Observed shape: per timestamp, only `consolidated_spatial_triples` exists. No `grounding`, `depth`, `pose_6dof`, `point_cloud`, `scene_latent_ref`, or `bbox` keys inside the consolidation JSON.

Citations:
- `/home/default/workspace/WorldMM/output/metadata/spatial_memory/A1_JAKE/spatial_consolidation_results_chatgpt-gpt-5.4.json:1-19`

## 4) Sidecar directory cross-check

### Present under `A1_JAKE`
- `/home/default/workspace/WorldMM/output/metadata/spatial_memory/A1_JAKE/grounding/120255900.json` exists.
- `/home/default/workspace/WorldMM/output/metadata/spatial_memory/A1_JAKE/scene_latents_triposr/` exists.
- `/home/default/workspace/WorldMM/output/metadata/spatial_memory/A1_JAKE/scene_latents_mast3r/` exists.

### But not loaded by Phase 3b
- No orchestrator argument or call site references the grounding directory.
- Search for `output/metadata/spatial_memory/A1_JAKE/grounding` in repo found only demo / other tooling, not the Phase 3b orchestrators.
- `with_scene/` content in this workspace exists only under `/home/default/workspace/WorldMM/output/metadata/spatial_memory/AEA_loc5_script4_seq6_rec1/with_scene/...`; no `A1_JAKE/with_scene/` match found.

Citations:
- `/home/default/workspace/WorldMM/output/metadata/spatial_memory/A1_JAKE/grounding/120255900.json:1-37`
- `/home/default/workspace/WorldMM/tools/demo_grounded_spatial_retrieve.py:42-44` (example of explicit grounding-file usage elsewhere)
- `/home/default/workspace/WorldMM/tools/spatial_hero_grounded_ablation.py:176-176` and `:286-286` (example of explicit grounding-file usage elsewhere)

## 5) Concrete sample: first loaded spatial entry

Given the runtime path above, the first spatial entry loads with:
- `subject_grounding=None`
- `object_grounding=None`
- `pose_6dof=None`
- `scene_latent_ref=None`
- `point_cloud=None`

So `to_display_str()` returns exactly:

```text
(I, near, dining_table)
```

That is the exact string the spatial-axis reasoner sees for this first entry in this run, unless a separate grounding file is explicitly supplied (it was not). Direct package import in this shell hit a missing `tenacity` dependency, so this string is confirmed from the source path + JSON shape rather than a live import.

## Verdict
Phase 3b spatial axis carried text triples only. The run loaded `output/metadata/spatial_memory/A1_JAKE/spatial_consolidation_results_chatgpt-gpt-5.4.json` through `WorldMemory.load_spatial_triples(...)` → `SpatialMemory.load_triples_from_file(...)` with `grounding_file=None`, and the consolidation JSON itself has only `consolidated_spatial_triples`. The geometry-rich sidecars exist on disk, but this run did not reference them, so depth / pose / point cloud / bbox / scene-latent did not flow into the spatial axis at runtime.
