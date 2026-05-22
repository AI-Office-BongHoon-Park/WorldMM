# Spatial Memory Schema — what gets embedded vs what stays as structured fields

**Date:** 2026-05-21 KST
**Scope:** Inventory of every field that lives on a `SpatialTripleEntry` (and its optional sidecars) plus a clear split between fields that become similarity vectors and fields that are kept as raw structured data carried alongside.

This is a reference document. It describes the current schema in `src/worldmm/memory/spatial/memory.py`, `grounding.py`, and `utils.py` exactly as shipped on the `feat/spatial-memory` branch.

---

## 1. The one and only embedded field

```python
# src/worldmm/memory/spatial/memory.py:289
all_texts = [entry.text for entry in entries_to_index]
all_embeddings = self.embedding_model.encode_text(all_texts)
```

`SpatialMemory.index()` embeds **a single composed string per triple**, nothing else.

| Field | Composition | Vector dim |
|---|---|---:|
| `entry.text` | `f"{subject} {predicate} {object} @ {place}"` (or without `@ {place}` when `place is None`) | 384 (MiniLM-L6) or 4096 (Qwen3-Embedding-4B) |

Everything below in this document is **NOT vectorised**. It rides along with the entry as structured numeric or symbolic data, and surfaces into the reasoning prompt either through retrieval grouping (PPR over the graph) or through `SpatialTripleEntry.to_display_str()`.

---

## 2. Graph vertices (used by Personalised PageRank, not embedded)

`SpatialMemory.index()` builds an `igraph` over entities derived from the triples currently being indexed.

| Vertex type | Source | Notes |
|---|---|---|
| Subject entity | `entry.subject` | one vertex per unique subject across the indexed slice |
| Object entity | `entry.object` | one vertex per unique object |
| Edge | `(subject_vertex, object_vertex)` per triple | undirected, no self-loops |

PPR runs over this graph after a coarse cosine-similarity top-k pass over the text embeddings; the top-k seeds `personalization_entities` for the reset distribution. Predicate labels are NOT attached to edges in the current implementation.

---

## 3. Symbolic fields on `SpatialTripleEntry` (text/int, not embedded)

`src/worldmm/memory/spatial/memory.py:25-37`:

| Field | Type | Example value |
|---|---|---|
| `id` | `str` | per-triple unique id |
| `subject` | `str` | `"hard_drive"` |
| `predicate` | `str` | one of the 11 closed-vocab predicates listed below |
| `object` | `str` | `"dining_table"` |
| `timestamp` | `int` | day-encoded chunk timestamp (HHMMSSFF) |
| `place` | `Optional[str]` | `"kitchen"`, etc. — may be `None` |

### Closed predicate vocabulary

`src/worldmm/memory/spatial/utils.py:6-18`:

```python
SPATIAL_PREDICATE_VOCAB = frozenset({
    "in", "on", "under", "next_to", "near",
    "left_of", "right_of", "behind", "in_front_of",
    "contains", "located_in",
})
```

11 predicates. Any triple emitted by extraction whose predicate is not in this set is dropped by `SpatialRawOutput._validate_triples()`.

---

## 4. Geometric grounding sidecars (depth + detection, not embedded)

`src/worldmm/memory/spatial/grounding.py:9` defines `GeometricGrounding` with `extra='forbid'`. Each `SpatialTripleEntry` carries TWO independent slots — `subject_grounding` and `object_grounding` — because distance / containment queries need both endpoints.

| Field | Type | Notes |
|---|---|---|
| `bbox_center` | `[x, y, z]` float | 3D centroid of the detected instance |
| `bbox_extent` | `[w, h, d]` float | 3D bbox size |
| `units` | `Literal["meters", "relative"]` | metric only when calibrated; relative when monocular depth |
| `source` | `str` | e.g. `"depth_anything_v2"` |
| `confidence` | `float` in [-1, 1] | detector / matcher confidence; `-1` means grounding was dropped |
| `keyframe_ts` | `int` | timestamp of the keyframe the grounding was lifted from |
| `instance_disambiguation` | `Optional[str]` | one of `"closest match"`, `"first occurrence"`, `"highest confidence"`, etc. |

Where it comes from: `tools/build_geometric_grounding.py` (Depth Anything V2 small + 2D detector + lift via per-bbox depth median).

Where it surfaces: `to_display_str()` inlines `[subj@ (x,y,z) rel]` and the analogous object side.

---

## 5. AEA 6-DoF wearer pose sidecar (`Pose6DoF`, not embedded)

`src/worldmm/memory/spatial/grounding.py:41`:

| Field | Type | Notes |
|---|---|---|
| `tracking_timestamp_us` | `int` | microseconds, shared with gaze stream |
| `tx, ty, tz` | `float` | wearer position in world frame, meters |
| `qw, qx, qy, qz` | `float` | wearer orientation quaternion; validator enforces unit norm within 1e-2 |
| `quality_score` | `float` in [0, 1] | from AEA closed-loop trajectory |
| `graph_uid` | `str` | per-frame coordinate-frame id |
| `units` | `Literal["meters"]` | fixed |
| `frame` | `str` | defaults to `"world"` |

Source: AEA `closed_loop_trajectory.csv` rows, aggregated into chunk-level mean / median in `tools/build_aea_spatial_sidecar.py`.

---

## 6. AEA gaze sidecar (`GazeTarget`, not embedded)

`src/worldmm/memory/spatial/grounding.py:67`:

| Field | Type | Notes |
|---|---|---|
| `tracking_timestamp_us` | `int` | microseconds, same domain as pose |
| `yaw_rads_cpf` | `float` | yaw in Central Pupil Frame |
| `pitch_rads_cpf` | `float` | pitch in CPF |
| `point_cpf` | `Optional[(x, y, z)]` | finite 3D gaze point only when `depth_m` is present on the AEA row (older model: always NaN) |
| `confidence_interval` | `Optional[dict[str, float]]` | yaw/pitch low/high bounds |
| `session_uid` | `Optional[str]` | gaze session identifier |

Source: AEA `general_eye_gaze.csv` rows, downsampled to ≤ 5 samples per 30-second chunk.

---

## 7. Place anchor (`PlaceAnchor`, not embedded)

`src/worldmm/memory/spatial/grounding.py:80`. Derived from DBSCAN clustering of the AEA pose stream in `tools/build_aea_place_anchors.py`.

| Field | Type | Notes |
|---|---|---|
| `coordinate_frame_id` | `str` | the pose stream's `graph_uid` |
| `centroid_world_m` | `(x, y, z)` | cluster mean in world frame, meters |
| `time_range_us` | `(start, end)` | dwell window in microseconds |
| `label` | `Optional[str]` | e.g. `"place_0"`; semantic naming is a follow-up |
| `evidence` | `List[str]` | nearby visits / merge trace |

---

## 8. Speech segment (`SpeechSegment`, used by builders, not currently attached to entries)

`src/worldmm/memory/spatial/grounding.py:92`. Carried by AEA chunk sidecars under `speech_segments` but not yet merged into individual `SpatialTripleEntry` instances.

| Field | Type | Notes |
|---|---|---|
| `start_time_ns` | `int` | nanoseconds |
| `end_time_ns` | `int` | nanoseconds |
| `text` | `str` | transcript |
| `confidence` | `float` | ASR confidence |
| `language` | `Optional[str]` | language code |

Source: AEA `speech.csv` passthrough.

---

## 9. Visual structured triples (parallel store, embedded separately)

This is NOT on `SpatialTripleEntry` — it lives in `tools/visual_triples_memory.py` (the `VisualTriplesMemory` shim) and `output/metadata/visual_memory/A1_JAKE/visual_triples_gpt_index.pkl`. It is included here because it deliberately mirrors the spatial axis's text+vector shape so visual and spatial speak ONE language.

| Field | Type | Notes |
|---|---|---|
| `triple_id` | `str` | `f"{clip_id}#f{frame_idx}#{seq}"` |
| `clip_id` | `str` | source 30-second clip id |
| `triple_text` | `str` | `"(s, p, o)"` — same closed-vocab shape as spatial |
| `embedding` | `np.ndarray(384,)` | MiniLM L6 — same model as spatial query embedder |
| `frame_idx` | `int` in [0, 15] | GPT-emitted supporting frame index |
| `frame_timestamp_s` | `float` | clip_start + f × (30 / 16) |
| `f_provenance` | `"gpt_anchored"` or `"fallback_middle"` | tracks measurement quality |

---

## 10. Compact summary — embedding vs raw

| Category | Embedded? | Source |
|---|:---:|---|
| Triple text (`subject predicate object @ place`) | ✅ **vector** | LLM triple extraction + closed vocab |
| Subject/object entities | ⊘ graph vertices only | derived from triples |
| Predicate (11 closed labels) | ⊘ string label | fixed vocabulary |
| Geometric bbox (depth-derived) | ⊘ raw 3D centre + extent | Depth Anything V2 + 2D detector |
| 6-DoF pose | ⊘ raw 7-vector + quality | AEA `closed_loop_trajectory.csv` |
| Gaze CPF ray | ⊘ raw 2 angles + optional 3D point | AEA `general_eye_gaze.csv` |
| Place anchor centroid | ⊘ raw 3D + time range | DBSCAN on pose stream |
| Speech text | ⊘ raw transcript | AEA `speech.csv` |
| Visual structured triples | ✅ **vector** (parallel store) | GPT vision call → MiniLM |

**Take-away.** The spatial axis embeds exactly ONE string per triple. Every geometric, kinematic, attentional, and acoustic signal we have is carried alongside as structured numeric data and reaches the reasoner through `to_display_str()`, not through cosine similarity in vector space. The visual axis (when used as structured triples) reuses the same text+MiniLM space so both axes contribute homogeneous evidence to the QA prompt.
