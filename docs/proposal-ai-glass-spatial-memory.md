# AI Glass × WorldMM Spatial Memory — Proposal

**Date:** 2026-05-21 KST
**Scope:** Extending the WorldMM 4th spatial axis from EgoLife videos to live AI Glass capture. What spatial elements are needed, where they slot into our codebase, and which beyond-QA scenarios are most impactful.

**Sister documents:**
- [`output/research/ai_glass_sensors.md`](../output/research/ai_glass_sensors.md) — AI Glass sensor inventory (8 platforms)
- [`output/research/spatial_memory_beyond_qa.md`](../output/research/spatial_memory_beyond_qa.md) — 12 non-QA scenarios
- [`output/research/codebase_spatial_extension_points.md`](../output/research/codebase_spatial_extension_points.md) — Concrete code attachment points

---

## 1. Vision: "Space as the index"

> When space becomes the primary index, every recorded moment carries its own context — what was around it, what came before it in that place, what the wearer was looking at when it happened. The retrieval query is no longer "what did I see?" but "what happened at this exact spot, this object, this gaze target."

Today our spatial axis stores closed-vocab triples `(s, p, o)` per chunk timestamp. With AI Glass continuous capture, the same axis can carry **persistent identity**, **6-DoF pose**, **gaze**, **hand interaction**, and **place graph** — turning chunk timestamps into a place-anchored memory map.

---

## 2. Required Spatial Elements (Sensors → Signals → Memory Shape)

### 2.1 Sensor inventory matrix

Cross-platform spec from [`ai_glass_sensors.md`](../output/research/ai_glass_sensors.md):

| Signal | Aria Gen 2 | Ray-Ban Meta | Apple Vision Pro | Snap Spec 4 | EgoLife (= Aria) | Notes |
|---|:---:|:---:|:---:|:---:|:---:|---|
| RGB POV cam | ✅ rolling 110°FOV 2880² | ✅ 12MP photo / 1080p video | ✅ multi-cam passthrough | ✅ stereo cams | ✅ Aria-class | Foundation everyone has |
| **6-DoF pose / SLAM** | ✅ MPS/VIO raw + closed-loop | ⊘ no public stream | ✅ world tracking | ✅ 6DoF | ✅ MPS | **Critical for "space as index"** |
| **Depth** | ⊘ (stereo SLAM only) | ⊘ | ✅ LiDAR + TrueDepth | ⊘ | ⊘ | Apple class-leading |
| **Eye / gaze tracking** | ✅ ET-cams | ⊘ | ✅ continuous gaze | ⊘ | ✅ | High-value for salience |
| **Hand pose** | partial via vision | ⊘ | ✅ skeleton + pinch | ✅ marker/surface | partial | |
| **IMU 6-axis** | ✅ raw stream | hidden | hidden | exposed | ✅ | Pose + activity |
| **Magnetometer / GPS** | ✅ raw | partial | ⊘ outdoor | ⊘ | ✅ | Anchors place graph globally |
| **Audio mic array** | ✅ 7-mic spatial | ✅ 5-mic | ✅ | ✅ stereo | ✅ | Source direction = social context |
| **On-device chip** | client offload + tools | Qualcomm AR1 | M2 + R1 | Spectacles SoC | offload | |
| **Continuous capture battery** | ≈100 min Gen 2 | ≈30 min video | ≈2 h | ≈30 min | varies | All are short — events not frames |

> **Best target**: Meta Project Aria Gen 2. Research-grade raw streams + closed-loop SLAM/MPS + eye/audio mic array. EgoLife already uses Aria → our pipeline already speaks this data format.

### 2.2 What's available ONLY on AI Glass (not phone, not third-person camera)

1. **Continuous gaze-anchored salience** — what the wearer actually attended to (not just what was in frame)
2. **Hand-object interaction events** — pick-up / put-down moments with object identity
3. **Head-pose-anchored room mapping** — first-person FoV scene graph
4. **Audio source direction** — who was speaking, where
5. **Persistent 6-DoF pose stream** — same place re-encountered = retrieval anchor
6. **GPS-tagged dwelling** — semantic place inference from time spent

### 2.3 Sensor → Spatial Memory mapping (proposed)

| AI Glass signal | New WorldMM artefact | Maps to existing axis |
|---|---|---|
| 6-DoF pose (VIO/SLAM) | `pose_6dof: (x,y,z,qw,qx,qy,qz)` per chunk + cumulative trajectory | Extends `GeometricGrounding` |
| GPS / magnetometer | `place_anchor: {lat, lon, heading, place_label}` per chunk | New optional sidecar |
| Eye gaze + scene graph | Triple `(viewer, gaze_at, object)` + dwell duration | **New predicate** in `SPATIAL_PREDICATE_VOCAB` |
| Hand pose + object | Triple `(hand_L, holding, object)` + event `(object, placed_at, surface)` | **New predicates** + event log |
| Audio source direction | Triple `(person_X, speaking_from, place)` + speaker turns | **New predicate**, episodic adjacent |
| IMU activity classifier | `activity_state: {walking, sitting, cooking, driving}` per chunk | New axis or chunk metadata |
| Place graph + trajectory | `place_graph: nodes + edges` (already shipped on DAY1 in `spatial_trajectory_summary.json`) | Extends today's `place_graph-A1_JAKE-DAY1.html` |

---

## 3. Codebase Extension Points

From [`codebase_spatial_extension_points.md`](../output/research/codebase_spatial_extension_points.md):

### 3.1 Where each new signal slots in

```
6-DoF pose          → src/worldmm/memory/spatial/grounding.py
                      extend `GeometricGrounding` with `pose_6dof: Optional[Tuple[float, ...]]`
                      builder: tools/build_geometric_grounding.py per-frame loop already exists

GPS / place anchor  → src/worldmm/memory/spatial/memory.py:25-35
                      add `place_anchor: Optional[PlaceAnchor]` to `SpatialTripleEntry`
                      new Pydantic model in grounding.py

Gaze triples        → src/worldmm/memory/spatial/utils.py:6-18
                      extend `SPATIAL_PREDICATE_VOCAB` with `gaze_at`, `dwelling_on`
                      update prompt at src/worldmm/llm/templates/spatial_extraction.py:5-37

Hand interactions   → same vocab extension: `holding`, `placed_at`, `picked_up`
                      new event-style triple shape (carries timestamp + actor)

Activity state      → preprocess/build_memory.py
                      new chunk-level metadata field; rendered via SpatialTripleEntry.to_display_str()

Place graph         → already have docs/slides/place-graph-A1_JAKE-DAY1.html
                      extend to multi-day + GPS-anchored, drop in src/worldmm/memory/spatial/
```

### 3.2 Cleanest extension shape

- **Predicate vocabulary expansion**: lowest cost, highest leverage. Adding 5-7 predicates (`gaze_at`, `holding`, `placed_at`, `picked_up`, `speaking_from`, `dwelling_on`, `facing`) gives most beyond-QA scenarios their answer shape.
- **Optional sidecar models**: pose_6dof, place_anchor, gaze_target as `Optional[...]` fields on `SpatialTripleEntry`. Backward compatible.
- **Event log layer**: pickup/putdown events have temporal + actor structure that doesn't fit a static triple. Recommend NEW lightweight event store (5-field schema) parallel to the triple store.

---

## 4. Beyond-QA Scenarios — Ranked for AI Glass Impact

From [`spatial_memory_beyond_qa.md`](../output/research/spatial_memory_beyond_qa.md), reranked by (a) AI Glass uniqueness (b) need for WorldMM spatial primitives (c) demonstrable impact.

### Tier 1 — Killer AI Glass × WorldMM applications

**S1. Lost-and-found + proactive departure check** ⭐⭐⭐
- "You are leaving home, wallet still on kitchen island" — not retrospective QA, **threshold warning**
- Needs: object identity persistence, last-seen pose, exit zone detection, routine model
- Why glass: put-down moment happens hands-busy; phone absent; warn at door no explicit query

**S2. AR overlay / re-encounter memory** ⭐⭐⭐
- Return to a place → see what you did there last time, overlaid on the real world
- Needs: place re-localization (6-DoF), event log per place, AR rendering surface
- Why glass: only form factor that overlays at gaze

**S3. Cooking / DIY in-task assistance** ⭐⭐⭐
- "Where did I put the salt 5 min ago?" "Which step did I skip?" — temporal + spatial recall during the task
- Needs: hand-object events, object permanence model, step recognition, task graph
- Why glass: hands-free, gaze-aware, in the flow

### Tier 2 — Strong but later-stage

**S4. Memory aid for elderly / MCI** — spatial cues for object recall + routine deviation alerts
**S5. Daily spatial summary** — already prototyped via `spatial_trajectory_summary.json`; extend to AI Glass multi-day
**S6. Workplace ergonomics / industrial logbook** — tool usage tracking by station, zone entry/exit, OSHA replacement
**S7. Health / habit tracking by location** — fridge openings, screen time per room, sedentary zones

### Tier 3 — Visionary but heavy

**S8. Social / face-place binding** — privacy-governed; high complexity
**S9. Physical-world Google (lifelog search engine)** — ultimate vision but heaviest privacy + scale problem
**S10. "Monitor this place for me" (handover to AI agent)** — embodied agent gets spatial context handoff
**S11. Navigation / route memory** — different from GPS turn-by-turn
**S12. "Where am I" disambiguation for first-time visitors** — accessibility angle

---

## 5. Proposed Roadmap — 3 phases

### Phase 1: AI Glass adapter (2-3 weeks)
1. Aria SDK reader: ingest Aria VRS streams → WorldMM chunk schema
2. Predicate vocab v2: add `gaze_at`, `holding`, `placed_at`, `picked_up`, `facing` to `SPATIAL_PREDICATE_VOCAB`
3. `PlaceAnchor` + `Pose6DoF` models in `grounding.py`
4. Spatial extraction prompt v2: include the new predicates with examples

### Phase 2: Two killer demos (3-4 weeks each)
**Demo A — Proactive departure check (S1)**
- Trigger: GPS exit-zone OR door detection event
- Query spatial memory for "critical objects last seen far from current path"
- Surface as glass notification

**Demo B — AR re-encounter overlay (S2)**
- On re-localization to known place, query "events at this place" sorted by recency
- Render 3 most recent events as text-anchor overlays in user's field of view

### Phase 3: Spatial search engine kernel (continuous)
- Index all spatial triples + events + place anchors + gaze targets
- Query interface: text + place + time-range filters
- Foundation for S9 (physical-world Google) and S5 (daily summary)

---

## 6. Open Questions Before Implementation

1. **Hardware access**: Project Aria Gen 2 dev kits are gated. Apply via [Aria research program](https://www.projectaria.com/) or use existing EgoLife dataset (Aria recordings)?
2. **Privacy model**: face recognition, place semantics, audio retention — opt-in/local-only/redacted vs cloud?
3. **Predicate vocab scope**: closed (paper-friendly, retrieval-fast) vs open (richer, harder to embed)? Current spatial axis chose closed — extend or branch?
4. **Real-time vs batch**: Phase 1 batch-process Aria recordings end-to-end (proven path). Phase 2+ needs streaming. Architecture decision pending.
5. **Multi-day place graph**: how to merge place identity across capture sessions in different lighting/seasons? GPS+VIO+visual descriptor + manual labels?

---

## 7. Recommended Next Action

Pick **ONE Tier-1 scenario** (S1, S2, or S3) and build the end-to-end demo against a multi-hour Aria recording from the EgoLife corpus. This both validates the codebase extension path and produces a publishable artefact distinct from current EgoLifeQA work.

**My recommendation**: **S1 (proactive departure check)** — simplest sensor requirements (GPS + object-place triples + exit detection), strongest narrative impact ("AI Glass that catches you before you forget"), least privacy heat, falls out of the existing spatial-triple memory with one new predicate + one threshold rule.
