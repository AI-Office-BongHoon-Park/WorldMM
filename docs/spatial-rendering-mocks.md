# Spatial Rendering Mocks — From Frame Thumbnails to Gaussian Splatting

**Date:** 2026-05-19 KST
**Sister docs:** [`docs/spatial-encoding-sensor-based.md`](spatial-encoding-sensor-based.md) §3 (recommended path: extension of 4th axis with Depth Anything V2 + ConceptGraphs-style JSON) · [`docs/three-vs-four-axis-ablation.md`](three-vs-four-axis-ablation.md) (live numbers)

This document specifies what *spatial retrieval with imagery* looks like in WorldMM, today and tomorrow. **Today** is real video-frame thumbnails attached to retrieved spatial triples — the [`tools/thumbnail_extractor.py`](../tools/thumbnail_extractor.py) and [`tools/spatial_retrieve_with_thumbnails.py`](../tools/spatial_retrieve_with_thumbnails.py) tools produce this for any spatial query. **Tomorrow** is Gaussian-splatting-rendered novel views of the same places, pulled from a pre-trained per-scene splat model — this document specifies that interface so the PPT can show what the upgrade path costs and looks like.

---

## 1. What works today — frame thumbnails on spatial retrieval

Given a natural-language query, `tools/spatial_retrieve_with_thumbnails.py`:

1. Calls `SpatialMemory.retrieve(query, top_k, as_context=False)` for the top-K triples by PPR.
2. For each retrieved triple, looks up the **first-observed** chunk timestamp from the per-chunk extraction JSON (not the consolidation file, which only carries the latest accumulated timestamp).
3. Resolves that timestamp to the MP4 covering it via `find_clip` (30-second strict window with a 180-second fallback).
4. Decodes the closest frame with `decord` and writes a 480-px-wide JPEG thumbnail.
5. Emits a self-contained HTML page with one card per retrieved triple, each showing the `(subject, predicate, object)` line plus the frame thumbnail and timestamp.

### 1.1 Example output

```bash
uv run python tools/spatial_retrieve_with_thumbnails.py \
    --query "Alice screwdriver rack near table" \
    --top-k 12 \
    --out-html output/demo_spatial_q1.html
# -> wrote output/demo_spatial_q1.html  (12 triples, 4 thumbnails)
```

The output HTML lists every retrieved triple even when no MP4 covers its timestamp; missing thumbnails are rendered as a clearly-labelled "no clip for this timestamp" placeholder so the audience can see exactly how data availability gates the visual layer.

### 1.2 Why "frame thumbnail" is the right v1

- The agent's retrieval is already grounded in a *real moment* of egocentric video; rendering the actual frame at that moment is the most faithful possible visual.
- No new models to train; no per-scene optimisation.
- Storage is cheap: a 480-px JPEG is ~25–35 KB. A full DAY1 with 81 spatial chunks × 1 thumbnail = ~3 MB.
- The same machinery feeds the day-summary timeline (`tools/day_summary.py`).

---

## 2. What tomorrow looks like — Gaussian-splatting-rendered novel views

The natural upgrade is to allow the spatial retriever to return not just *the captured frame at the time the fact was observed* but **a synthesised view of the same place from an arbitrary camera pose** — for example, "what does the kitchen look like with the camera positioned to also show the dining table?" That requires a per-scene Gaussian splatting model.

This section specifies the integration so a follow-up PR can land it without further design discussion.

### 2.1 Conceptual flow

```
spatial query (text)
       │
       ▼
SpatialMemory.retrieve(query, top_k) -> [SpatialTripleEntry...]
       │
       ▼
for each entry:
   1. place = entry.place  OR  derive from (subject|object) if either matches a known place name
   2. if place has a trained Gaussian-splat model (per §2.2):
        rendered_image = gs_render(place, pose=heuristic_pose_for(triple))
      else:
        rendered_image = extract_thumbnail(entry.first_seen_ts)   # today's behaviour
   3. attach rendered_image as a base64 PNG to the triple's HTML card
       │
       ▼
output: HTML with the same card layout, but the image source is the rendered view
        when available and the live frame otherwise.
```

### 2.2 What we need to pre-compute per place

For each distinct place in the spatial vocabulary (`kitchen`, `living_room`, `bedroom`, `restaurant`, `supermarket`, `Hema Fresh`, `courtyard`, …; A1_JAKE DAY1 has ≈ 15 distinct places after dedup), the offline pipeline trains one Gaussian splatting model:

```
data/EgoLife/A1_JAKE/DAY1/*.mp4   # the RGB clips
        │
        ▼
sample 1 frame / 1 s from clips matching the place               # 30-300 frames per place
        │
        ▼
COLMAP (or VGGT) -> camera poses + intrinsics
        │
        ▼
gsplat / nerfstudio splatfacto -> per-place .ply Gaussian model
                                  + per-place pose distribution
        │
        ▼
output/gs_models/A1_JAKE/DAY1/<place>.ply
```

Per the [`docs/spatial-encoding-sensor-based.md`](spatial-encoding-sensor-based.md) §1.3 verdict, this is the most expensive piece of the whole stack and is only worth it if novel-view rendering buys something the live-frame approach does not. The PPT slide that makes the cost-benefit case lives in §4 below.

### 2.3 `gs_render(place, pose)` contract

```python
from pathlib import Path
from PIL import Image

def gs_render(place: str, pose: dict | None = None,
              model_dir: Path = Path("output/gs_models/A1_JAKE/DAY1"),
              image_size_px: int = 480) -> Image.Image | None:
    """Render a novel view of `place` from `pose` using its Gaussian-splat model.

    pose: optional dict with {"position": [x,y,z], "lookat": [x,y,z], "fov_deg": 60.0}.
          If None, picks a representative pose from the model's per-place pose distribution
          (typically: median camera + slight elevation to show context).

    Returns a PIL.Image (RGB, downscaled to image_size_px on the longest side), or None
    if the place has no trained model (caller falls back to extract_thumbnail).
    """
```

### 2.4 Pose-from-triple heuristic

For triple `(subject, predicate, object)`:

| Predicate | Heuristic pose |
|---|---|
| `located_in`, `in` | place's median camera pose (overview shot) |
| `on`, `under` | look at `subject` from the place's median pose |
| `next_to`, `near` | pose such that both `subject` and `object` fall in frame |
| `left_of`, `right_of` | pose that emphasises the lateral relation |
| `behind`, `in_front_of` | over-the-shoulder pose with both endpoints visible |
| `contains` | top-down or wide pose of `subject` (the container) |

Implementation: a per-place JSON sidecar `output/gs_models/A1_JAKE/DAY1/<place>.poses.json` listing a few preset poses; the heuristic picks one by predicate.

### 2.5 Storage budget

| Asset | per place | ×15 places | All-day |
|---|---:|---:|---:|
| `.ply` Gaussian model | 80–250 MB | ~1.5–4 GB | one-time |
| `.poses.json` | 1 KB | 15 KB | trivial |
| Rendered cached image (480-px PNG) | 60 KB | 0.9 MB | trivial |

≈ 2–4 GB per day is the dominant cost. Acceptable for a research artefact, prohibitive for live deployment.

### 2.6 Render speed (target)

`gsplat` benches at ~50 ms per 480-px render on a consumer GPU once the model is loaded. With ≤ 15 distinct places kept in CPU-pinned memory and swapped on demand, end-to-end query → 12 rendered cards ≈ 1 s. **No worse than the current frame-extraction path.**

---

## 3. Mock UI — what a slide of "spatial retrieval with GS rendering" looks like

The HTML at [`docs/slides/spatial-retrieval-gs-mock.html`](slides/spatial-retrieval-gs-mock.html) is a single 1280×720 slide showing what the live demo would look like once the GS layer is wired in. The mock uses the actual frame thumbnails from `output/thumbnails/` as placeholders — the only "fake" part is the label saying "GS-rendered view" instead of "captured frame".

The mock makes one point clearly: **the UI is unchanged**; only the *source* of the image swaps from "live frame at first-seen timestamp" to "GS-rendered novel view of the place". The retrieval logic, prompt, and JSON contract stay identical.

This is the strongest argument for the [`docs/spatial-encoding-sensor-based.md`](spatial-encoding-sensor-based.md) §7 extension architecture: the same `SpatialMemory.retrieve()` interface absorbs the upgrade, no new memory axis, no agent prompt change.

---

## 4. Cost-benefit (one slide)

| Aspect | Live frame thumbnail (today) | GS-rendered novel view (tomorrow) |
|---|---|---|
| **Pre-compute time** | 0 | hours to days per day-of-footage (COLMAP + gsplat training) |
| **Storage** | ~3 MB/day for cached thumbnails | 2–4 GB/day for trained .ply models |
| **Query latency** | ~50 ms (decord frame decode) | ~50 ms (gsplat render) once loaded |
| **Faithfulness to actual scene** | maximum (it IS the actual frame) | high but synthesised; lighting/shadow approximated |
| **Visual on questions whose triple has no covering MP4** | "no clip" placeholder | rendered view of the place is still available |
| **Visual on never-observed angle** | impossible | possible |
| **Dynamic people / occupants** | rendered correctly (live capture) | smeared (3DGS is static-scene assumption) |
| **License / data risk** | re-rendering existing frames | introduces a new derived artefact |

**The honest read for the PPT.** The GS upgrade pays off when (a) the audience cares about "show me an angle the camera wearer never used", or (b) when too many retrieved chunks lack covering MP4s on the deployment target machine. For pure paper-figures the live-frame approach is already sufficient and cheaper.

---

## 5. Files of record

| Path | What it is | Status |
|---|---|---|
| [`tools/thumbnail_extractor.py`](../tools/thumbnail_extractor.py) | timestamp → MP4 frame → JPEG | **shipped** |
| [`tools/spatial_retrieve_with_thumbnails.py`](../tools/spatial_retrieve_with_thumbnails.py) | spatial query → HTML with thumbnails | **shipped** |
| [`tools/day_summary.py`](../tools/day_summary.py) | hourly captions + semantic + spatial → fused day-narrative HTML with hourly thumbnails | **shipped** |
| [`output/demo_spatial_q1.html`](../output/demo_spatial_q1.html) | sample retrieval for "Alice screwdriver rack" | **shipped** (4/12 thumbnails) |
| [`output/demo_spatial_kitchen.html`](../output/demo_spatial_kitchen.html) | sample retrieval for "I in kitchen near whiteboard" | **shipped** (4/12 thumbnails) |
| [`output/day_summary_A1_JAKE_DAY1.html`](../output/day_summary_A1_JAKE_DAY1.html) | DAY1 fused day-narrative report | **shipped** (8/10 hourly thumbnails) |
| [`docs/slides/spatial-retrieval-gs-mock.html`](slides/spatial-retrieval-gs-mock.html) | PPT mock — same layout, "GS-rendered" labels | **mock** |
| `output/gs_models/A1_JAKE/DAY1/<place>.ply` | per-place Gaussian splat models | **not built** (out of scope; see §2.5 cost) |

---

## 6. Reproduction

```bash
# 1. Download the MP4s needed for thumbnails (≈ 50 MB for the 7 signal-case clips,
#    +200 MB if you want broad hour-boundary coverage for the day summary):
hf download lmms-lab/EgoLife --repo-type=dataset \
    --include 'A1_JAKE/DAY1/DAY1_A1_JAKE_11094208.mp4' \
    --include 'A1_JAKE/DAY1/DAY1_A1_JAKE_12000000.mp4' \
    --include 'A1_JAKE/DAY1/DAY1_A1_JAKE_13000000.mp4' \
    --include 'A1_JAKE/DAY1/DAY1_A1_JAKE_18000000.mp4' \
    --include 'A1_JAKE/DAY1/DAY1_A1_JAKE_22000000.mp4' \
    --local-dir data/EgoLife
# (loop over remaining hour-start filenames; `hf download` only honours the LAST --include flag,
#  so use a `for f in ...; do hf download --include "...${f}.mp4" ...; done` loop.)

# 2. Sanity-check thumbnail extractor on the Q1 / Q33 / Q53 cases:
uv run python tools/thumbnail_extractor.py \
    --timestamps 111152408,113355400,117364916

# 3. Spatial retrieval HTML with thumbnails:
uv run python tools/spatial_retrieve_with_thumbnails.py \
    --query "Alice screwdriver rack near table" \
    --top-k 12 \
    --out-html output/demo_spatial_q1.html

# 4. Day summary HTML (LLM fuses episodic + semantic + spatial + hourly thumbnails):
uv run python tools/day_summary.py
# -> output/day_summary_A1_JAKE_DAY1.html
# -> output/day_summary_A1_JAKE_DAY1.json   (the LLM context + structured response)
```
