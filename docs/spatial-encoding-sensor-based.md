# Sensor-Based Spatial Encoding for WorldMM — Survey & Verdict

**Date:** 2026-05-18 KST
**Status:** Comparison document — feeds the implementation plan at [`.sisyphus/plans/spatial-sensor-memory.md`](../.sisyphus/plans/spatial-sensor-memory.md)
**Sister document:** [`docs/spatial-memory-ablation.md`](spatial-memory-ablation.md) (the WHERE-triple memory we already shipped)

WorldMM currently encodes spatial information as **closed-vocabulary text triples** extracted by an LLM from captions (`(subject, predicate, object)` with predicates like `located_in`, `on`, `next_to`). That works because everything in the agent's mental model is text the reasoning LLM can read.

The natural next question is: **could we encode spatial information from a geometric / sensor representation — point clouds, Gaussian splatting, mmWave radar — and would that meaningfully improve long-video reasoning?**

This document is the survey + opinionated verdict. It is intentionally one level higher than a reading list: every entry has a 1-line "would you use this today" answer.

---

## 0. TL;DR

| Question | Verdict |
|---|---|
| Should WorldMM add a geometric memory layer beyond text triples? | **Yes, but as an EXTENSION of the existing 4th axis, not a new 5th axis.** |
| Which method? | **Depth Anything V2 + open-vocab detector → per-triple geometric grounding attached to `SpatialTripleEntry.grounding`.** |
| Should we add non-RGB sensors (LiDAR, mmWave, WiFi-CSI, RGB-D, IMU)? | **No, not for the EgoLife pipeline.** All require re-capture; the project becomes a sensor-engineering project, not a memory-reasoning project. |
| Should we use NeRF or Gaussian Splatting as memory? | **No.** Beautiful renders, but the training cost, pose-drift, and dynamic-scene weakness make them a poor fit for 7-day egocentric memory. |
| What is the right agent-query interface? | **The same `retrieve(query: str) -> str` the 4th axis already exposes**, augmented to render geometric grounding inline when present. Reasoning prompt still picks `"spatial"`; no new memory type. |

If you read no further: **extend `SpatialTripleEntry` with an optional `grounding: Optional[GeometricGrounding]` field populated by a new offline pipeline (Depth Anything V2 → posed point cloud → object centroids → CLIP-match to existing triple subjects/objects). `SpatialMemory.retrieve()` keeps its current shape and just appends `[place=..., center=(x,y,z), extent=(...)]` to each line when grounding exists.** Treat sensors as a future-work section in the paper.

> **Architectural choice locked in [§7](#7-extension-over-new-axis-architectural-revision):** geometric data is an *extension* of the existing 4th axis, not a parallel 5th axis. The previous draft of this document recommended a new `SpatialEncodingMemory` class — that has been superseded.

---

## 1. RGB-only geometric memory — candidate methods

The EgoLife dataset ships **RGB only**. So the first cut is: what can we extract from RGB alone in 2026?

### 1.1 Monocular dense reconstruction

| Method | Output | License | VRAM (≈) | GT pose? | 5.6 GB dev-box? |
|---|---|---|---|---|---|
| **DUSt3R** (Naver, [arxiv 2312.14132](https://arxiv.org/abs/2312.14132)) | pairwise 3D point maps + recovered poses | CC BY-NC-SA 4.0 | ~8 GB at ViT-L 512 | No | △ marginal |
| **MASt3R** (Naver, [arxiv 2406.09756](https://arxiv.org/abs/2406.09756)) | metric pointmaps + matches + sparse global alignment | CC BY-NC-SA 4.0 | ~8 GB | No | △ |
| **MASt3R-SLAM** | full SLAM front-end over RGB | CC BY-NC-SA 4.0 | ~10 GB | No | ✗ |
| **VGGT** (Facebook Research, [arxiv 2503.11651](https://arxiv.org/abs/2503.11651)) | depth + pointmaps + camera intrinsics/extrinsics + tracks + COLMAP export | custom; commercial checkpoint separate | 1.9–5.6 GB for 1–20 frames, balloons past 50 frames | No | ○ for short clips |
| **MonST3R** (Junyi42, [paper](https://monst3r-project.github.io/files/monst3r_paper.pdf)) | time-varying dynamic point cloud + per-frame pose | CC BY-NC-SA 4.0 | ~23 GB for 65 16:9 frames | No | ✗ |
| **Spann3R** (HengyiWang, [arxiv 2408.16061](https://arxiv.org/abs/2408.16061)) | incremental reconstruction with built-in "spatial memory" | CC BY-NC-SA 4.0 | DUSt3R-class | No | △ |

**Honest read on these for WorldMM:**

- The whole `*ust3r/Mast3r/VGGT` family is the right *technology* but is **non-commercial-licensed** (CC BY-NC-SA), which is a real concern for any downstream productization but fine for a research paper.
- They are all designed for **short clips (~50 frames)**. A 10-hour day at 1 fps = 36 000 frames. The natural pattern is "reconstruct per chunk, then stitch", not "reconstruct the day in one pass".
- **VGGT** is the most modern and is the first one that natively exports COLMAP-style outputs — easiest to plug into downstream code.

### 1.2 Lightweight per-frame depth (no full SLAM)

| Method | Output | License | VRAM (≈) |
|---|---|---|---|
| **Depth Anything V2** (Small/Base/Large) | per-frame relative or metric depth | Apache-2.0 (Small/Base/Large), CC BY-NC for some metric variants | 1.5 GB Small, 3 GB Base, 6 GB Large |
| **ZoeDepth** | metric depth | MIT | 2 GB |
| **Metric3D v2** | metric depth + normals | Apache-2.0 | 4–6 GB |

**This row is the sweet spot for our dev box.** Depth Anything V2 Small/Base is the only thing that *certainly* fits in 5.6 GB free VRAM and gives you a useful depth map per frame, which is enough to lift the existing 2D scene into a per-frame oriented point cloud given known intrinsics (or estimated ones).

### 1.3 Gaussian splatting and NeRF

| Family | When it shines | When it fails |
|---|---|---|
| **3DGS** (original Kerbl et al. 2023) | one room, ~100 calibrated images, static scene | Long egocentric video: pose drift, dynamic occupants (people!), training cost per chunk |
| **4DGS** / Spacetime Gaussians | short dynamic clips | Hour-scale memory: not designed for it |
| **gsplat** (NerfStudio library) | nice query API | Still requires training per scene |
| **Block-NeRF** / **Mega-NeRF** | city-scale outdoor static driving | Indoor 7-day life: scene state changes constantly |
| **LERF** (language-embedded radiance fields) | natural-language 3D queries on a NeRF | NeRF prerequisite kills it for long video |

**Honest verdict for long egocentric video memory:** these are excellent at "render this room from a new viewpoint" but the assumption of a *static* scene that justifies a per-scene training pass falls apart in a 7-day life where chairs move, people enter and leave, and lighting changes. Use them in single-room single-time studies; do not treat them as a hour-scale memory.

### 1.4 Agent-queryable spatial representations

The hardest decision is not "how do we make a point cloud" but "**how does the LLM ask a question of it**". Three strong patterns:

| Pattern | Representative | Query interface | PoC cost |
|---|---|---|---|
| **JSON of objects** (CLIP-tagged) | **ConceptGraphs** ([Gu et al. 2024](https://github.com/concept-graphs/concept-graphs)) | retrieve top-K objects matching a text query, serialize `{id, tag, caption, bbox_center, bbox_extent}` to JSON | **Lowest** — fits `retrieve(str)->str` directly |
| **Collapsed scene graph with expand/contract** | **SayPlan** ([sayplan.github.io](https://sayplan.github.io)) | LLM tool-loops: `expand(node)`, `contract(node)` to navigate a 3D scene graph | **Medium** — needs agent tool support |
| **Inline 3D tokens** | **LLaVA-3D**, **3D-LLM**, **SceneVerse**, **Cube-LLM** | feed point / box / loc tokens directly into a 3D-finetuned LLM | **High** — requires a 3D-finetuned LLM, not our reasoning model |
| **Render-then-caption** | LERF, gsplat-as-memory | render a scene from a query pose, run a VLM caption | **Highest** — needs renderable model + VLM |

**Recommendation:** ConceptGraphs for v1, SayPlan-style upgrade for v2 once `WorldMemory.answer()` can multi-round on the spatial layer. Skip inline 3D tokens (different model class) and render-then-caption (way too expensive per query).

---

## 2. Non-RGB sensor encoding — honest assessment

This is the section users ask for ("can we use LiDAR / mmWave / WiFi"). The TL;DR is: **no, not for this project**. Reasons follow.

| Modality | 2024–2026 SOTA | Capture requirement | Verdict |
|---|---|---|---|
| **LiDAR / point cloud** | KISS-ICP, NKSR, ConceptGraphs+RGB-D | Real LiDAR scan or posed RGB-D capture. **Project Aria Gen1/Gen2 has NO LiDAR.** | **Paper future work.** Cannot retrofit to existing RGB-only EgoLife data. |
| **mmWave radar** | mmPose, RF-Pose, OpenRadar DSP | TI IWR1843BOOST + DCA1000 capture board, or Walabot / Vayyar. Sync + calibration is hell. | **Not recommended.** Would turn the project into a sensor-fusion paper, not a memory-reasoning paper. |
| **WiFi-CSI / BLE** | DensePose-from-WiFi, WiFlow, GraphPose-Fi | CSI-capable NIC/AP, environment-specific fingerprinting | **Not recommended.** Mismatched with continuous egocentric memory; spatial resolution is too coarse for object-level reasoning. |
| **Depth cameras / RGB-D** | ConceptGraphs, OK-Robot, Open3D RGB-D SLAM/TSDF | Intel RealSense, Azure Kinect, iPhone Pro LiDAR, ARKit depth. **Re-capture required.** | **Most realistic non-RGB option** but only if you collect new data. Not retrofittable to EgoLife. |
| **IMU + Visual-Inertial SLAM** | ORB-SLAM3, OpenVINS, OKVIS2 | Camera–IMU time-sync and calibration. **Aria does ship IMU.** | **Useful if you have Aria raw data.** Useless for stand-alone RGB video. |
| **Project Aria sensor stack** | RGB + stereo SLAM cams + eye cams + IMU + MPS SLAM | Aria VRS files | **Strong if available.** Cannot be retrofitted from RGB-only EgoLife. |

**The headline argument:** every non-RGB modality requires you to either (a) re-capture the dataset with new hardware or (b) accept Project Aria as a hard prerequisite. EgoLife is published as RGB video. Adding a sensor modality to this pipeline is therefore not a software change — it is a *new data-collection campaign*.

For a 6-month research arc on long-video memory, that trade is a clear *no*. Treat sensors as future work, list them in a one-paragraph paper section, and move on.

---

## 3. Recommended path for WorldMM (concrete, extension-style)

### 3.1 Cheapest viable prototype — wired as an EXTENSION of the 4th axis

```
RGB frames (1 fps from existing EgoLife mp4s)
        │
        ▼
Depth Anything V2 (Base or Small)  ─── per-frame relative depth, ~3 GB VRAM
        │
        ▼
Camera intrinsic estimation (VGGT or hard-coded EgoLife default) ─── per-clip, ~5 GB VRAM
        │
        ▼
Per-frame oriented point cloud (Open3D)
        │
        ▼
Open-vocabulary detection (GroundingDINO + SAM2) ─── object proposals per keyframe → 3D centroid
        │
        ▼
Object index: { detected_label, caption, bbox_center, bbox_extent, observed_at, clip_emb }
        │
        ▼     ◄── new joinerStep: CLIP-match each existing SpatialTripleEntry.subject / .object
        │         to the detected-object index. Where matched, populate
        │         SpatialTripleEntry.grounding = GeometricGrounding(...)
        ▼
SpatialMemory.retrieve(query: str) -> str    ◄── UNCHANGED INTERFACE
        │  - existing PPR retrieval over text triples
        │  - to_display_str() now optionally appends [place=..., center=(x,y,z), extent=(...)]
        ▼
WorldMemory.iterative_reasoning() still picks "spatial"  ◄── prompt unchanged, no 5th type
```

### 3.2 Why this exact path

1. **Fits 5.6 GB VRAM.** Depth Anything V2 Base + Open3D + GroundingDINO Small + SAM2 Small each comfortably under 4 GB; only one model loaded at a time.
2. **Reuses what we have.** Existing fine captions and episodic OpenIE supply the per-object captions that ConceptGraphs needs; no new captioner.
3. **Returns text.** `retrieve(query: str) -> str` is unchanged; the reasoning LLM never sees raw points.
4. **No license trap.** Depth Anything V2 (Apache-2.0), GroundingDINO (Apache-2.0), SAM2 (Apache-2.0), Open3D (MIT). DUSt3R-family is avoided for the v1 (CC BY-NC) — VGGT is only invoked optionally for intrinsics.
5. **No sensor capture.** Everything runs on the RGB we already downloaded.

### 3.3 Where this gets us — sample query behaviour

A query like *"What was on the kitchen counter around 11:30?"* should return JSON like:

```json
{
  "query": "kitchen counter at 11:30",
  "objects": [
    {"id": 142, "tag": "kettle",   "caption": "stainless steel electric kettle",
     "bbox_center": [1.32, 0.81, 0.55], "place": "kitchen", "observed_at": ["112700", "113200"]},
    {"id": 158, "tag": "knife",    "caption": "chef's knife",
     "bbox_center": [1.18, 0.92, 0.61], "place": "kitchen", "observed_at": ["113000"]},
    {"id": 161, "tag": "cutting_board", "caption": "wooden cutting board",
     "bbox_center": [1.40, 0.87, 0.58], "place": "kitchen", "observed_at": ["112900", "113300"]}
  ]
}
```

The reasoning LLM ingests this exactly as it ingests the existing spatial-triple memory, and can chain it with episodic memory for "who used the knife".

### 3.4 What it does NOT promise

- It does **not** give you renderable novel views. If you want "render the kitchen from the doorway at 11:30 as a sanity check", you still need 3DGS or NeRF separately.
- It does **not** model dynamic occupants well. The point cloud will smear moving people; the object graph will list them as transient detections. For 7-day life this is fine, but it is a known weakness.
- It does **not** give you metric distances reliably without intrinsics estimation. Plan to run VGGT (or COLMAP) on short clips to estimate intrinsics + poses for the chunks where distance matters.

---

## 4. Comparison — current 4th axis vs. extended 4th axis vs. (rejected) 5th axis

| Dimension | 4th axis as shipped (text triples only) | **Extended 4th axis (recommended)** | Rejected: separate 5th axis |
|---|---|---|---|
| Memory types exposed to the reasoning agent | 4 | **4** (same) | 5 |
| `retrieve(query)` return shape | text triples, line per entry | **text triples; grounded lines also carry `[place=..., center=(...), extent=(...)]`** | additional JSON-of-objects payload from the new memory |
| Triple ↔ geometric grounding join | n/a | **done offline at build time, persisted in the triple JSON** | done at query time across two memories |
| Reasoning prompt complexity | 4 memory types + 3 spatial few-shots | **4 memory types + 1 added few-shot showing a grounded triple** | 5 memory types + new few-shots → more agent confusion |
| Backward compat for existing builds | n/a | **full — `grounding` is `Optional[...]`; old JSON loads with `grounding=None`** | n/a — separate JSON, separate index |
| Distance queries (e.g. "how far is sofa from kitchen") | not supported | **supported via grounding fields** | supported |
| "Render the kitchen" | not supported | not supported | not supported (out of scope) |
| Compute footprint | LLM only | **LLM + 3 GB VRAM (depth) + 3 GB (detector) intermittently, only for the grounding step** | same VRAM, but adds always-on object index |
| Per-day build cost | ~10 min via proxy | **~10 min triples + 30–60 min one-time grounding pass on dev box** | similar; two pipelines instead of one |
| Failure mode | drops out-of-vocab predicates | drops out-of-vocab predicates; grounding silently omitted on detect-miss | mis-detection; agent must combine partial answers across memories |
| Code surface | `memory/spatial/*.py` | **same files + `grounding` field + 1 new builder script** | new package `memory/spatial_encoding/*` (~7 files) |

**Net:** the extension reuses one memory's PPR retrieval and one prompt slot for the same observable behaviour, at strictly lower code and prompt budget than a sibling 5th axis. The 5th-axis variant is preserved here only as a comparison point — see [§7](#7-extension-over-new-axis-architectural-revision) for the architectural argument.

---

## 5. References (selected — full bibliography in the plan document)

- ConceptGraphs — Gu et al. 2024, [GitHub](https://github.com/concept-graphs/concept-graphs)
- SayPlan — [project page](https://sayplan.github.io)
- DUSt3R — [arxiv](https://arxiv.org/abs/2312.14132), [GitHub](https://github.com/naver/dust3r)
- MASt3R — [arxiv](https://arxiv.org/abs/2406.09756), [GitHub](https://github.com/naver/mast3r)
- VGGT — [arxiv](https://arxiv.org/abs/2503.11651), [GitHub](https://github.com/facebookresearch/vggt)
- Depth Anything V2 — [GitHub](https://github.com/DepthAnything/Depth-Anything-V2)
- ConceptGraphs Planner Prompt — [code](https://github.com/concept-graphs/concept-graphs/blob/main/conceptgraph/scenegraph/prompts/concept_graphs_planner.txt)
- LLaVA-3D — [GitHub](https://github.com/ZCMax/LLaVA-3D)
- LERF — [GitHub](https://github.com/kerrj/lerf)
- Project Aria MPS — [Meta docs](https://www.projectaria.com/datasets/aea/)
- KISS-ICP — [GitHub](https://github.com/PRBonn/kiss-icp)
- NKSR — [GitHub](https://github.com/nv-tlabs/NKSR)

---

## 6. Bottom line for the write-up

> WorldMM's current 4th spatial-memory axis encodes symbolic WHERE-relations as closed-vocabulary text triples. We **extend** that same axis with optional geometric grounding derived from RGB video (Depth Anything V2 for monocular depth, GroundingDINO+SAM2 for open-vocab detection, Open3D for the per-frame point cloud, CLIP for joining detected objects back to the existing triple subjects and objects). The augmented triples flow through the same Personalized-PageRank retrieval the 4th axis already uses, and the reasoning agent continues to see exactly four memory types — no prompt-vocabulary expansion. Non-RGB sensors (LiDAR, mmWave radar, WiFi-CSI, RGB-D, IMU) require dataset re-capture and would turn a memory-reasoning project into a sensor-engineering project; they belong in a one-paragraph future-work section, not in the v1 pipeline. NeRF / Gaussian Splatting deliver beautiful renders but are mismatched with hour-scale dynamic egocentric memory.

---

## 7. Extension over new axis — architectural revision

The previous draft of this document recommended adding a **new 5th memory axis** (`SpatialEncodingMemory`) for geometric data. After implementing the 4th axis and seeing it land in the reasoning loop, the cleaner design is an **extension of the 4th axis**, not a sibling memory.

### 7.1 Why the 5th-axis idea was tempting

- Strict separation of concerns: text-symbolic vs. geometric WHERE knowledge.
- Mirrors how Semantic and Episodic are separate axes despite both encoding text.
- Lets the reasoning agent explicitly request `memory_type: "spatial_encoding"` for distance / 3D queries.

### 7.2 Why extension is better in practice

1. **Same conceptual entity, two representations.** A triple `(sofa, located_in, living_room)` and an object detection `sofa at (2.3, 0.5, 1.8)` are the same fact in two formats. Storing them in two memories forces the reasoning agent to *join* them at query time, every time. Extension performs the join *once*, offline.
2. **Reasoning prompt budget.** Each new memory type the agent can choose between adds branches to the system prompt and dilutes the few-shot examples. 4 types is already at the edge of what one-shot prompting handles well; 5 makes wrong-memory selection a measurable failure mode.
3. **Code reuse.** `SpatialMemory` already runs igraph + PPR + embedding similarity. A 5th axis would either duplicate that or invent a new retrieval path. Extension reuses 100 % of the existing retrieval code.
4. **Backward compatibility.** Existing `spatial_consolidation_results_*.json` builds continue to load — the new `grounding` field is `Optional` and absent in old files.
5. **Ablation cleanliness.** The next ablation can compare:
   - 4th axis OFF
   - 4th axis ON, grounding OFF (i.e. the version shipped in `docs/spatial-memory-ablation.md`)
   - 4th axis ON, grounding ON (the extension)
   That is a clean three-way comparison of the *same* memory at three augmentation levels, not a comparison across two distinct architectures.

### 7.3 Concrete schema diff

```python
# src/worldmm/memory/spatial/memory.py  (existing)
@dataclass
class SpatialTripleEntry:
    id: str
    subject: str
    predicate: str
    object: str
    timestamp: int
    place: Optional[str] = None

# After extension (drop-in):
@dataclass
class GeometricGrounding:
    bbox_center: List[float]       # (x, y, z) — metric if intrinsics known, relative otherwise
    bbox_extent: List[float]       # (dx, dy, dz)
    units: str                     # "meters" | "relative"
    source: str                    # e.g. "depth_anything_v2+grounding_dino+sam2"
    confidence: float              # detector confidence, [0, 1]
    matched_token: str             # which side of the triple the grounding refers to: "subject" | "object"
    keyframe_ts: int               # the video timestamp where the centroid was sampled

@dataclass
class SpatialTripleEntry:
    id: str
    subject: str
    predicate: str
    object: str
    timestamp: int
    place: Optional[str] = None
    grounding: Optional[GeometricGrounding] = None    # ← new, fully optional

    def to_display_str(self) -> str:
        base = f"({self.subject}, {self.predicate}, {self.object})"
        if self.place:
            base += f" [place={self.place}]"
        if self.grounding:
            c = self.grounding.bbox_center
            e = self.grounding.bbox_extent
            base += f" [center=({c[0]:.2f},{c[1]:.2f},{c[2]:.2f}) extent=({e[0]:.2f},{e[1]:.2f},{e[2]:.2f}) units={self.grounding.units}]"
        return base
```

### 7.4 Concrete file plan

**No new package.** Instead:
- `src/worldmm/memory/spatial/utils.py` — add `GeometricGrounding` dataclass + Pydantic schema for grounding JSON.
- `src/worldmm/memory/spatial/memory.py` — extend `SpatialTripleEntry` with `grounding: Optional[GeometricGrounding]` and update `to_display_str()`.
- `src/worldmm/memory/spatial/grounding_builder.py` *(new file)* — runs the depth + detect + CLIP-match offline pipeline and writes a sidecar JSON keyed by triple `id`.
- `src/worldmm/memory/spatial/memory.py` — extend `load_triples_from_*` to read the sidecar if present and populate the `grounding` field.
- `preprocess/spatial_memory/ground_spatial_triples.py` *(new CLI)* — wraps the builder.
- `preprocess/build_memory.py` — extend the `run_spatial` flow with an optional `--with-grounding` flag (default off so existing builds stay identical).
- `script/3_build_memory.sh` — add `--with-grounding` passthrough.

**Out of scope intentionally:**
- New memory type in the reasoning prompt.
- Renderable representations.
- Multi-day or per-day cross-day consolidation of grounding.

### 7.5 What the agent sees after the extension

Before:
```
(Lucia, located_in, living_room) [place=living_room]
```

After (grounding present):
```
(Lucia, located_in, living_room) [place=living_room] [center=(1.32,0.81,2.10) extent=(0.42,1.65,0.30) units=meters]
```

After (grounding absent — old data or detect-miss):
```
(Lucia, located_in, living_room) [place=living_room]
```

The reasoning LLM reads this without any new prompt vocabulary; existing distance questions ("how far is X from Y") become answerable when both ends are grounded, and degrade gracefully to the current behaviour when they are not.
