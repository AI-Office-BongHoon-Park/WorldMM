# WorldMM Memory Schema — what gets embedded vs what stays as structured fields

**Date:** 2026-05-21 KST
**Scope:** Inventory of every field across all FOUR memory axes (episodic / semantic / spatial / visual) with a clear split between fields that become similarity vectors and fields that are kept as raw structured data carried alongside.

This is a reference document. It describes the current schema across `src/worldmm/memory/{episodic,semantic,spatial,visual}/` exactly as shipped on the `feat/spatial-memory` branch.

## TL;DR — what every axis embeds

| Axis | Embedded content | Embedder | Vector dim | Storage shape |
|---|---|---|---:|---|
| **Episodic** | caption text per granularity | HippoRAG-internal text embedder (same `EmbeddingModel`) | 384 / 4096 | per-caption row in a HippoRAG index, 5 parallel indices (one per granularity) |
| **Semantic** | `f"{subject} {predicate} {object}"` triple text | `EmbeddingModel.encode_text` | 384 / 4096 | per-triple row in a torch tensor + igraph |
| **Spatial** | `f"{subject} {predicate} {object} @ {place}"` triple text | `EmbeddingModel.encode_text` | 384 / 4096 | per-triple row in a torch tensor + igraph |
| **Visual (default CLIP path)** | middle frame of each 30-s clip → CLIP image embedding | `clip-ViT-B-32` via `sentence-transformers`; query side uses `EmbeddingModel.encode_vis_query` for cross-modal text→image | 512 | per-clip row in a torch tensor |
| **Visual (GPT triples path)** | `f"({s}, {p}, {o})"` triple text per GPT-extracted visual triple | MiniLM-L6 via `VisualTripleQueryEmbedder` | 384 | per-triple row in a separate pkl index |

Active embedder model is configurable via `WORLDMM_EMBED_MODEL` env var. Default in `EmbeddingModel.__init__` is `"Qwen/Qwen3-Embedding-4B"` for text (4096-d) and `"VLM2Vec/VLM2Vec-V2.0"` for vision; ablations on this branch swapped to `sentence-transformers/all-MiniLM-L6-v2` (384-d) on CPU and to `clip-ViT-B-32` for the visual path. Dim depends on which embedder is loaded.

---

# Part A — Spatial axis

(The original schema doc starts here.)

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

---

# Part B — Episodic axis

## E1. The embedded field

```python
# src/worldmm/memory/episodic/memory.py — index() calls
hipporag.index(...)  # HippoRAG handles its own NER + triple extraction + embedding internally
```

`EpisodicMemory` does NOT call `encode_text` directly. It delegates to a `HippoRAG` instance per granularity (constructed at `episodic/memory.py:120-126`); HippoRAG internally tokenises captions, extracts NER + triples via OpenIE, and embeds them with the same `EmbeddingModel` passed in.

| Granularity | What gets fed to HippoRAG.index() |
|---|---|
| `10sec` | per-10s caption text |
| `30sec` | per-30s caption text (matches MP4 chunk boundary) |
| `3min` | per-3min caption text |
| `10min` | per-10min caption text |
| `1h` | per-1h caption text |

5 parallel HippoRAG indices, one per granularity (`GRANULARITY_ORDER = ["10sec", "30sec", "3min", "10min", "1h"]`).

## E2. CaptionEntry symbolic fields (NOT embedded)

`src/worldmm/memory/episodic/memory.py:18`:

| Field | Type | Example |
|---|---|---|
| `id` | `str` | `mdhash(caption_text)` |
| `text` | `str` | the actual caption sentence |
| `start_time` | `str` | HHMMSSFF |
| `end_time` | `str` | HHMMSSFF |
| `date` | `str` | `"DAY1"` |
| `granularity` | `str` | one of `10sec / 30sec / 3min / 10min / 1h` |
| `video_path` | `Optional[str]` | `data/EgoLife/.../DAY1_A1_JAKE_HHMMSSFF.mp4` |

## E3. HippoRAG-internal artefacts (per granularity, all NOT directly accessed by EpisodicMemory)

| Internal file | Content |
|---|---|
| `openie_results_<model>.json` | per-chunk OpenIE output: `{chunk_id, named_entities, triples}` |
| `episodic_triple_results_<model>.json` | filtered triple cache |
| HippoRAG vector store | dense embedding of every passage + every extracted phrase node |
| HippoRAG entity graph | per-granularity igraph for PPR |

On DAY1 A1_JAKE these surface on disk under `output/metadata/episodic_memory/A1_JAKE/` (1.7 MB total for the 91-chunk slice).

## E4. Retrieval (multiscale filter, NOT vector-only)

```python
# episodic/memory.py retrieve flow
1. For each granularity, hipporag[g].retrieve(query) -> top-k passages
2. Concatenate candidates across granularities
3. LLM call with multiscale_filter prompt -> select / rank the most relevant captions
4. Return CaptionEntry list in ranked order
```

So episodic retrieval is **HippoRAG dense + PPR + an LLM filter step**, not pure cosine. The LLM filter sees the candidate captions as text and emits a ranked subset.

---

# Part C — Semantic axis

## S1. The embedded field

```python
# src/worldmm/memory/semantic/memory.py — entry.text is "subject predicate object"
all_texts = [entry.text for entry in entries_to_index]
all_embeddings = self.embedding_model.encode_text(all_texts)
```

| Field | Composition | Vector dim |
|---|---|---:|
| `entry.text` | `f"{subject} {predicate} {object}"` (no `@ place` — semantic is place-agnostic) | 384 (MiniLM) / 4096 (Qwen3) |

## S2. SemanticTripleEntry symbolic fields (NOT embedded)

`src/worldmm/memory/semantic/memory.py:18`:

| Field | Type | Example |
|---|---|---|
| `id` | `str` | per-triple unique id |
| `subject` | `str` | `"Jake"` |
| `predicate` | `str` | open-vocab predicate from LLM (NOT closed like spatial) |
| `object` | `str` | `"likes coffee"` |
| `timestamp` | `int` | consolidation timestamp |

Note: semantic predicates are NOT restricted to a closed vocabulary; LLM emits whatever predicate fits. Validators in `src/worldmm/memory/semantic/utils.py` strip whitespace and reject empties but do not enforce a vocab list.

## S3. Graph vertices for PPR (not embedded)

Same shape as spatial: subject + object entities form igraph vertices, undirected edges per triple, no predicate labels on edges.

## S4. Retrieval

Identical pattern to spatial:
1. Cosine top-k over triple-text embeddings.
2. Seed PPR with subjects + objects of those top-k.
3. Sum PPR scores of (subject, object) per triple in the full pool to rerank.

---

# Part D — Visual axis

The visual axis has TWO concrete loadable shapes on this branch — the default CLIP image-embedding path used by `VisualMemory`, and the structured-GPT-triples path used by `VisualTriplesMemory` (the visual-axis pivot that lifted 4-axis accuracy from 11/30 back to 26/30).

## V1. Default CLIP path — `VisualMemory`

### V1.a. The embedded field

```python
# src/worldmm/memory/visual/memory.py:285-289
self.embeddings = torch.tensor(
    np.array(all_embeddings),  # one per VideoClipEntry
    dtype=torch.float32, device=...
)
```

Each clip's `embedding` field is precomputed offline by `tools/build_visual_embeddings.py`:
1. decord-read the middle frame of the 30-s MP4.
2. `sentence_transformers.SentenceTransformer("clip-ViT-B-32").encode([frame])`.
3. Save to `output/metadata/visual_memory/A1_JAKE/visual_embeddings_clip-ViT-B-32.pkl` as `Dict[clip_id, np.ndarray(512,)]`.

| Embedded content | Encoder | Dim | Storage |
|---|---|---:|---|
| Middle frame of clip (1 frame per 30-s clip) | CLIP ViT-B/32 (image side) | 512 | per-clip ndarray in pkl |

Query side uses `EmbeddingModel.encode_vis_query(text_query)` — cross-modal text→image in the same CLIP space.

### V1.b. VideoClipEntry fields (mostly NOT embedded)

`src/worldmm/memory/visual/memory.py:20-31`:

| Field | Type | Embedded? |
|---|---|:---:|
| `id` | `str` | ⊘ symbolic |
| `video_path` | `str` | ⊘ symbolic |
| `start_time` | `str` HHMMSSFF | ⊘ symbolic |
| `end_time` | `str` HHMMSSFF | ⊘ symbolic |
| `date` | `str` `"DAY1"` | ⊘ symbolic |
| `clip_start_sec` | `Optional[float]` | ⊘ symbolic |
| `clip_end_sec` | `Optional[float]` | ⊘ symbolic |
| `embedding` | `Optional[np.ndarray]` (512,) | ✅ **CLIP image vector** |
| `description` | `Optional[str]` | ⊘ raw text returned in retrieve when present |

### V1.c. FrameEntry (lazy frame extraction, not embedded)

`src/worldmm/memory/visual/memory.py:48-53`. Used only for time-range queries that fall back to 1-fps frame extraction; not part of the index.

| Field | Type | Notes |
|---|---|---|
| `video_path` | `str` | source MP4 |
| `frame_index` | `int` | decord frame idx |
| `timestamp_sec` | `float` | absolute time |
| `frame` | `Optional[PIL.Image]` | populated lazily on retrieve |

### V1.d. Retrieval

Two branches in `VisualMemory.retrieve()`:
- If `query` matches a time-range pattern (`"DAY1 11:09:43 - DAY1 11:09:58"`), call `_retrieve_by_time_range` and return decoded frames at fps.
- Otherwise, `_retrieve_by_similarity`: cosine `encode_vis_query(query)` vs precomputed `embeddings` tensor, top-k. If retrieved clips have non-empty `description`, return the descriptions as text context; otherwise extract frames at fps and return image dict.

## V2. Structured GPT triples path — `VisualTriplesMemory`

Lives in `tools/visual_triples_memory.py` (not under `src/worldmm/memory/visual/`). This is the visual axis that ACTUALLY contributed in the §14 / §15 ablations on this branch.

### V2.a. The embedded field

```python
# tools/visual_triples_memory.py + tools/visual_triple_query_embedder.py
all_embeddings = MiniLM.encode([entry.triple_text for entry in entries])
```

| Embedded content | Encoder | Dim | Storage |
|---|---|---:|---|
| `f"({s}, {p}, {o})"` per GPT-emitted visual triple | `sentence-transformers/all-MiniLM-L6-v2` (CPU) | 384 | per-triple row in `visual_triples_gpt_index.pkl` |

This is INTENTIONALLY the same embedder space and text shape as the spatial axis. That is the "speak one language" property: when the visual axis retrieves a triple, the reasoner sees the same shape as a spatial triple and treats them as complementary evidence, not as competing modalities.

### V2.b. Per-triple entry fields

| Field | Type | Notes |
|---|---|---|
| `triple_id` | `str` | `f"{clip_id}#f{frame_idx}#{seq}"` |
| `clip_id` | `str` | source 30-s clip id |
| `triple_text` | `str` | `"(s, p, o)"` |
| `embedding` | `np.ndarray(384,)` | ✅ **MiniLM text vector** |
| `frame_idx` | `int` in [0, 15] | GPT-emitted supporting frame within the 16-frame call |
| `frame_timestamp_s` | `float` | `clip_start_s + frame_idx * (30 / 16)` |
| `f_provenance` | `"gpt_anchored"` or `"fallback_middle"` | tracks measurement quality |
| `video_path / start_time / end_time / date` | str | cross-reference to MP4 |

### V2.c. Retrieval

`VisualTriplesMemory.retrieve(query)`: MiniLM-embed query, cosine top-k over the 1031 visual triples, return triple text lines plus optional thumbnail extracted from the source MP4 at the exact `frame_idx` (via `tools/grounded_visual_retrieve.py`).

---

# Part E — Compact all-axis summary

| Axis | Embedded content | Encoder | Dim | NOT-embedded structured carries |
|---|---|---|---:|---|
| Episodic | caption text per granularity | HippoRAG-internal | 384 / 4096 | time range, granularity, video path, OpenIE NER/triples |
| Semantic | `"{s} {p} {o}"` triple text | EmbeddingModel.encode_text | 384 / 4096 | timestamp, graph vertices, open-vocab predicate |
| Spatial | `"{s} {p} {o} @ {place}"` triple text | EmbeddingModel.encode_text | 384 / 4096 | place, closed-vocab predicate (11), grounding 3D bbox, 6-DoF pose, gaze CPF ray, place anchor, speech segment |
| Visual (CLIP path) | middle frame image | CLIP ViT-B/32 | 512 | clip metadata, optional description text, lazily extracted frames |
| Visual (GPT triples) | `"(s, p, o)"` text | MiniLM-L6 (CPU) | 384 | clip_id, frame_idx, ts_s, f_provenance |

**Cross-axis design pattern.** All four axes embed exactly ONE string (or one image, in the CLIP case) per indexable unit. Every other signal — temporal range, graph topology, 3D geometry, kinematics, attention, audio, GPT supporting-frame anchor — rides along as structured numeric data and reaches the reasoning prompt through each entry's `to_display_str()` or as part of the LLM filter context, NOT through cosine similarity.

This is what makes the "speak one language" trick work: every axis's retrievable unit is a SHORT text string, the reasoner consumes them in the same prompt slot, and the heavy spatial / kinematic / visual-anchor data is auxiliary metadata, not a competing retrieval channel.
