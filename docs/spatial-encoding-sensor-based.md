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
| Should WorldMM add a geometric memory layer beyond text triples? | **Yes, but RGB-only and as a text-summarising layer.** |
| Which method? | **Depth Anything V2 + ConceptGraphs-style JSON scene graph** (cheapest path that fits the existing `retrieve(query: str) -> str` contract). |
| Should we add non-RGB sensors (LiDAR, mmWave, WiFi-CSI, RGB-D, IMU)? | **No, not for the EgoLife pipeline.** All require re-capture; the project becomes a sensor-engineering project, not a memory-reasoning project. |
| Should we use NeRF or Gaussian Splatting as memory? | **No.** Beautiful renders, but the training cost, pose-drift, and dynamic-scene weakness make them a poor fit for 7-day egocentric memory. |
| What is the right agent-query interface? | **A JSON-of-objects summary** (ConceptGraphs pattern), with a `collapse / expand` upgrade path (SayPlan pattern) once the LLM can call tools mid-reasoning. |

If you read no further: **target Depth Anything V2 → posed point cloud → object-centric scene graph → ConceptGraphs JSON returned from `retrieve(query: str)`**. Treat sensors as a future-work section in the paper.

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

## 3. Recommended path for WorldMM (concrete)

### 3.1 Cheapest viable prototype

```
RGB frames (1 fps from existing EgoLife mp4s)
        │
        ▼
Depth Anything V2 (Base or Small)  ─── per-frame relative depth, ~3 GB VRAM
        │
        ▼
Camera intrinsic estimation (VGGT or COLMAP) ─── per-clip, ~5 GB VRAM
        │
        ▼
Per-frame oriented point cloud (Open3D)
        │
        ▼
Open-vocabulary detection (GroundingDINO + SAM2) ─── object proposals per keyframe
        │
        ▼
Object-centric scene graph (ConceptGraphs style)  ─── { id, tag, caption, bbox_center, bbox_extent, observed_at, place_name }
        │
        ▼
SpatialEncodingMemory.retrieve(query: str) -> str
        │  - top-K objects by CLIP similarity to query
        │  - JSON serialised exactly like ConceptGraphs planner prompt
        ▼
WorldMemory.iterative_reasoning() picks "spatial_encoding" alongside the existing 4 axes
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

## 4. Comparison vs. the shipped spatial-triple memory

| Dimension | Shipped spatial-triple memory (4th axis) | Proposed spatial-encoding memory (5th axis) |
|---|---|---|
| Input | Episodic triples + captions | RGB frames |
| Predicate vocabulary | 11 closed tokens | open-vocabulary object tags |
| Geometric grounding | None — purely linguistic | metric or relative 3D centres per object |
| Distance queries | not supported | supported |
| "Render the kitchen" | not supported | not supported (out of scope) |
| Compute footprint | LLM-only | adds 3 GB VRAM (depth) + 3 GB (detector) intermittently |
| Per-day build cost | ~10 min via proxy | estimated 30–60 min on dev box for full DAY1 |
| Failure mode | drops out-of-vocab predicates | mis-detection / drift in long sequences |

**They are complementary, not substitutes.** The shipped layer answers *symbolic* WHERE questions; the proposed layer answers *metric* WHERE questions and grounds objects in 3D space.

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

> WorldMM's current spatial memory encodes symbolic WHERE-relations as text triples. A natural 5th axis is a **geometric** memory derived from RGB video using Depth Anything V2 + object-centric scene graphs in the ConceptGraphs style, returning a JSON summary that drops cleanly into the existing `retrieve(query: str) -> str` interface. Non-RGB sensors (LiDAR, mmWave radar, WiFi-CSI, RGB-D, IMU) require dataset re-capture and turn a memory-reasoning project into a sensor-engineering project; they belong in a one-paragraph future-work section, not in the v1 pipeline. NeRF / Gaussian Splatting deliver beautiful renders but are mismatched with hour-scale dynamic egocentric memory.
