# Spatial Memory with 3D Reconstruction — Extension Plan

**Date:** 2026-05-22 KST
**Scope:** Extend WorldMM's spatial axis so it stores not just closed-vocab triples but also a **reconstructable 3D latent** per scene chunk. Goal: at recall time, retrieve both (a) the symbolic triple AND (b) enough state to regenerate the original 3D space (point cloud / mesh / Gaussian splat).

**Sister reports:**
- [`output/research/geometry_embedding_for_reconstruction.md`](../output/research/geometry_embedding_for_reconstruction.md) — 12+ 3D embedding methods catalogued.
- [`output/research/retrievable_reconstructable_spatial_memory.md`](../output/research/retrievable_reconstructable_spatial_memory.md) — 10+ end-to-end retrieve+reconstruct systems.
- [`docs/spatial-memory-schema.md`](spatial-memory-schema.md) — current schema doc, what's embedded vs structured.

---

## 1. The architectural insight: "key + payload", not "one vector for everything"

The geometry survey's strongest conclusion:

> *"Best fit is not 'append one 384-d vector and expect full 3D back.' Best fit is 'append one retrieval vector + sidecar pointer/hash.' The sidecar holds reconstructable geometry state. Retrieval returns triple + sidecar; decoder reconstructs 3D."*

→ WorldMM's existing `(s, p, o) @ place` triple text + MiniLM vector remains the **index**. New scene latents live as **sidecars** referenced from the entry, not as additional ANN keys.

## 2. Static vs Dynamic — clean axis split

User insight on VGGT-class methods: static background and dynamic elements can be decomposed separately.

| Decomposition | Maps to WorldMM axis | Latent shape |
|---|---|---|
| **Static scene geometry** (room walls, furniture, stable objects) | **Spatial axis** — new `scene_latent` sidecar | embed ONCE per place, reuse across visits |
| **Dynamic elements** (moving people, transient objects, events) | **Episodic axis** — caption-level, optional `track_latent` | embed per-moment, not per-place |

The methods that can produce this decomposition from monocular video:
- **MonST3R** (dynamic MASt3R) — separates static/dynamic
- **TracksTo4D** (Meta) — 4D tracking
- **CUT3R / EmerNeRF** — auto-segmenting dynamic NeRF
- **Dynamic 3DGS** — explicit static + dynamic Gaussians

VGGT vanilla assumes mostly static; MonST3R / TracksTo4D / dynamic 3DGS extend to dynamic scenes.

## 3. Recommended sidecar backends (from geometry survey)

Three viable backend shapes for the `scene_latent` sidecar:

### Backend A — **Compressed 3D Gaussian Splat** (recommended for full-scene)
- **Best fit**: scene-scale recall ("what did the kitchen look like at DAY1 11:14:30?")
- **Storage**: `.ply` of Gaussians (xyz, opacity, scale, rotation, color SH); LightGaussian / Compact3D reduce 5-10×
- **Decoder**: 3DGS rasterizer → photoreal render from any camera pose
- **Hardware**: 6 GB GPU can render small compressed scenes; training a new splat per chunk is heavier (skip, use prebuilt or off-device)
- **License**: original 3DGS research-only; CC variants exist
- **Add-on for retrieval**: **LangSplat** binds CLIP features per Gaussian → enables "show me kitchens with red appliances"

### Backend B — **DUSt3R / MASt3R pointmap cache** (recommended for incremental capture)
- **Best fit**: per-keyframe geometry from any 2 frames
- **Storage**: per-pair pointmap `.pt` + confidence map
- **Decoder**: pointmap → unproject to point cloud / mesh
- **Hardware**: MASt3R-ViTLarge 2.75 GB checkpoint fits on 6 GB
- **License**: CC BY-NC-SA non-commercial ⚠️
- **Coverage**: NOT a learned latent — direct geometry prediction

### Backend C — **Triplane (TripoSR / OpenLRM)** (recommended for object-level)
- **Best fit**: single egocentric frame → object proxy mesh
- **Storage**: 3 × `(C, H, W)` tensors per object
- **Decoder**: triplane → density/color MLP → mesh
- **Hardware**: TripoSR-base inference fits on 6 GB
- **License**: MIT (TripoSR)
- **Use case**: per-object affordance recall, not full room

**Recommendation**: ship a backend-agnostic `SceneLatentRef` model and let `decoder_backend` field dispatch. Start with Backend B (DUSt3R pointmaps) since AEA `semidense_points.csv.gz` already gives us point clouds, but architect for Backend A (compressed 3DGS) as the long-term destination.

## 4. WorldMM uniqueness — what no surveyed system gives us

From the retrievable+reconstructable survey:

> *"None of the surveyed systems gives all requirements together: (1) Temporal versioned 3D latent store, (2) Dual symbolic + reconstructable recall, (3) Unified retrieval unit, (4) Change-aware consolidation, (5) Recall contract."*

WorldMM's contribution = **unifying the 4 axes' retrieval unit** AND **versioning 3D state over time** within the existing closed-vocab triple framework.

## 5. Concrete schema extension

### 5.1 New Pydantic models in `src/worldmm/memory/spatial/grounding.py`

```python
class SceneLatentRef(BaseModel):
    """Reference to a reconstructable 3D latent stored out-of-band."""
    model_config = ConfigDict(extra="forbid")

    backend: Literal["compact_3dgs", "dust3r_pointmap", "mast3r_pointmap",
                     "triplane_triposr", "shap_e_sdf", "semidense_points"]
    storage_uri: str               # relative path under output/scene_latents/
    coord_frame_id: str            # joins with pose_6dof.graph_uid
    time_us: int                   # capture timestamp
    state_kind: Literal["static_scene", "dynamic_event", "object_instance"]
    decoder_version: str           # encoder+decoder pair id
    confidence: float = Field(..., ge=0.0, le=1.0)
    provenance_frames: List[str] = []   # source frame timestamps / clip ids
    storage_bytes: int             # rough size on disk

class PointCloudSidecar(BaseModel):
    """Inline point-cloud sidecar (small enough to stream)."""
    model_config = ConfigDict(extra="forbid")

    points_world_m: List[List[float]]    # (N, 3); for N ≤ 1000, else use storage_uri
    uncertainty: Optional[List[float]] = None
    graph_uid: str
    time_range_us: Tuple[int, int]
    source: str                          # e.g., "aea_semidense", "dust3r_pair"
```

### 5.2 `SpatialTripleEntry` extension (additive, backward compatible)

```python
@dataclass
class SpatialTripleEntry:
    # ... existing fields ...
    pose_6dof: Optional[Pose6DoF] = None
    gaze_target: Optional[GazeTarget] = None
    place_anchor: Optional[PlaceAnchor] = None

    # NEW (this proposal):
    scene_latent_ref: Optional[SceneLatentRef] = None        # heavy 3D
    point_cloud: Optional[PointCloudSidecar] = None          # light 3D
```

### 5.3 Display string

```python
def to_display_str(self) -> str:
    # ... existing ...
    if self.scene_latent_ref:
        parts.append(f"[scene_latent={self.scene_latent_ref.backend}#{hash(self.scene_latent_ref.storage_uri) & 0xFFFFFF:06x}]")
    if self.point_cloud:
        parts.append(f"[points={len(self.point_cloud.points_world_m)} pts]")
```

Latent itself NOT serialized into the prompt — only a placeholder hash so the reasoner knows recallable 3D evidence exists.

## 6. Builders (new tools)

| Tool | Backend | Input | Output |
|---|---|---|---|
| `tools/build_scene_latent_dust3r.py` | `dust3r_pointmap` | 2 frames per keyframe pair | `.pt` pointmap + sidecar JSON |
| `tools/build_scene_latent_mast3r.py` | `mast3r_pointmap` | image set per chunk | global aligned point cloud + scene token |
| `tools/build_scene_latent_compact3dgs.py` | `compact_3dgs` | image set per place | compressed `.ply` Gaussian set |
| `tools/build_scene_latent_aea_semidense.py` | `semidense_points` | AEA `semidense_points.csv.gz` | direct passthrough wrapped as `PointCloudSidecar` |

The AEA path is the cheapest first delivery — we already have the data on disk.

## 7. Decoder interface (out-of-process)

Decoders run as separate CLI tools / services, NOT in the memory module:

```
tools/decode_scene.py
  --ref output/metadata/spatial_memory/AEA_<seq>/with_place/chunk_001.json
  --camera "DAY1 11:14:30"
  --output /tmp/scene_recall.ply
```

Memory module returns `SceneLatentRef` only; user (or downstream agent) invokes decoder. Keeps the memory hot-path light.

## 8. Retrieval API change

`SpatialMemory.retrieve()` currently returns `List[SpatialTripleEntry] | str`. No change needed — `SceneLatentRef` rides along with each entry. New helper:

```python
def retrieve_with_scenes(self, query: str, top_k: int = 5) -> List[Tuple[SpatialTripleEntry, Optional[Path]]]:
    """Returns entries + locally-resolved scene_latent file paths if present."""
```

## 9. Out-of-scope (explicit)

1. **Cross-sequence coordinate alignment.** Different AEA sequences / EgoLife days have different `graph_uid`. Stitching needs re-localization (PostgreSQL + Aria SDK / SuperPoint / NetVLAD); not in scope.
2. **Latent version drift.** When a decoder model is upgraded, old latents may decode poorly. `decoder_version` field tracks this; migration tooling not in scope.
3. **Storage scaling.** A single LightGaussian splat for a kitchen chunk ≈ 5-50 MB. Day-long capture × 33 places × 7 days × 6 subjects ≈ thousands of latents = 10-100 GB. Need a content-addressable store before scaling.
4. **Privacy.** A 3D reconstruction of someone's home is more sensitive than text triples. Need redaction policy, retention rules, encryption-at-rest. Not in scope; flag as gate before any user-facing deployment.
5. **Generative hallucination.** Generative backends (Shap-E, Point-E) can fabricate plausible-but-wrong geometry. Use ONLY observed-state backends (3DGS, pointmaps, AEA semidense) for memory; reserve generative backends for "imagined future state" queries (separate module).

## 10. Recommended first delivery (smallest end-to-end)

**Backend C-mini: AEA semidense as `PointCloudSidecar`** — 0 new ML models needed.

1. New Pydantic models in `grounding.py` (Section 5.1) — ~80 LOC.
2. `SpatialTripleEntry` field additions — ~10 LOC.
3. `tools/build_scene_latent_aea_semidense.py` — read `data/AEA/<seq>/semidense_points.csv.gz`, downsample to N ≤ 1000 points per chunk, write `PointCloudSidecar` JSON per chunk under `output/metadata/spatial_memory/AEA_<seq>/with_scene/`.
4. `tools/decode_scene.py --ref ... --format ply` — write a tiny PLY directly from the inline `points_world_m` list. No ML required.
5. End-to-end smoke: load a chunk, render its PLY in MeshLab/Open3D, screenshot for proof.

Wall-clock estimate: **~1 day of code work**.

After that landed, the same `SceneLatentRef` schema gracefully accommodates Backend B (DUSt3R/MASt3R pointmap) and Backend A (compressed 3DGS) as additional builders behind the `backend` discriminator.

## 11. VGGT static/dynamic split — where it slots in

If we add a VGGT or MonST3R / TracksTo4D builder later:

- Output `state_kind="static_scene"` → attach to spatial axis as `SceneLatentRef` (this proposal).
- Output `state_kind="dynamic_event"` → attach to episodic axis caption entries (separate extension, not in this plan).

The single `SceneLatentRef` model handles both via the `state_kind` discriminator without changing the spatial axis schema.

## 12. Open questions

1. **PLY vs glTF vs USD as canonical decoded format?** PLY is simplest; glTF supports animation; USD is industry standard for time-varying scenes.
2. **Inline `PointCloudSidecar` size cap?** Current 1000-point cap is arbitrary; might be 500 for streaming, 5000 for offline.
3. **Decoder model checkpoints — where do they live?** HF Hub by `decoder_version`? Local cache? Bundled with the codebase?
4. **Does the spatial axis embedding need to change?** Current: only `triple.text`. Proposal: same. The latent never enters the cosine-similarity space.

## 13. Bottom line

**The shortest path to "spatial memory that reconstructs at recall":**

```
SpatialTripleEntry + SceneLatentRef sidecar
       ↓                       ↓
   (s, p, o, @ place)    PLY / Gaussians / triplane
       ↓                       ↓
   text retrieval         decoder on demand
       ↓                       ↓
   reasoning prompt       reconstructed 3D view
```

Spatial-axis embedding stays text-only (one string per triple). Reconstructable 3D lives as a sidecar pointer keyed off the same triple. Each decoder backend is pluggable behind a single discriminator. First delivery is AEA-semidense → PLY with zero new ML dependencies; later upgrades to 3DGS / pointmap / triplane swap the backend without changing the schema.
