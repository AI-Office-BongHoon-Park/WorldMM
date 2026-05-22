# Next Dataset Recommendation

**Date:** 2026-05-22 KST
**Trigger:** EgoLife sparse (78 bursts, 69 isolated chunks) and AEA stationary (1 place per session). We need multi-room AND long-duration AND portable spatial signals AND zero SDK install.

**Sister reports:**
- [`output/research/multiroom_egocentric_datasets.md`](../output/research/multiroom_egocentric_datasets.md)
- [`output/research/longduration_egocentric_datasets.md`](../output/research/longduration_egocentric_datasets.md)
- [`output/research/simulated_egocentric_datasets.md`](../output/research/simulated_egocentric_datasets.md)

---

## 1. Three top candidates by lens

| Lens | Winner | Why |
|---|---|---|
| **Multi-room mobile** | **Nymeria** (Meta) | Real Aria, multi-room daily activity, MPS portable CSV |
| **Long-duration / longitudinal** | **EgoLife A1_JAKE DAY2-DAY7** | Same pipeline, real wearable, 6 more days of the same subject |
| **Synthetic full GT** | **Habitat-Sim HM3D-Semantic** | Multi-room by construction, full ground truth, MIT-style |

## 2. Honest comparison

| Dataset | Multi-room | Duration | Portable signals | License | Size | Acquisition friction |
|---|:---:|---:|---|---|---:|---|
| **Nymeria** | ✅ (multi-room daily activity) | 15-min recordings | Aria MPS CSV (pose, gaze, points, audio) | gated research | ~hundreds of GB | email-gated, same as AEA |
| **EgoLife A1_JAKE DAY2-DAY7** | sparse (similar to DAY1) | day-long × 6 days | MP4 + SRT only | MIT-style | **~93 GB** | direct HF download |
| **Habitat-Sim HM3D-Semantic** | ✅ (by construction) | scriptable | full GT (RGB/depth/semantic/pose/instance) | mostly open | ~10s GB | install Habitat-Sim (Python pkg, no special SDK) |
| **EPIC-KITCHENS-100** | single-room (kitchen) | multi-day | MP4 + narrations | research | ~1 TB | direct HF / web |
| **HoloAssist** | partial | task-length | RGB-D + 6DoF + hand pose + gaze | research | ~10s GB | gated |
| **TartanAir V2** | ✅ synth | scriptable | pose + depth + segmentation | open | varies | direct download |
| **ScanNet** | ✅ (whole apartments) | per-scan | RGB-D + pose + semantic mesh | research | 100s GB | gated |

## 3. Verdict — three concrete next-pull paths

### Path A — **Nymeria** (real Aria multi-room) ⭐ first choice if multi-room is the priority
- **Get**: real wearable + multi-room + portable MPS spatial signals.
- **Existing infrastructure**: identical to AEA (already verified — `tools/build_aea_*` works on Aria MPS CSV).
- **Cost**: email-gate (same flow as AEA), then download 1 multi-room sequence (~5-10 GB MPS-only).
- **Risk**: not yet confirmed Aria-licensed; need to submit email at https://www.projectaria.com/datasets/nymeria/ (Playwright walkthrough same as AEA).

### Path B — **EgoLife A1_JAKE DAY2–DAY7** (longitudinal) ⭐ first choice if longitudinal is the priority
- **Get**: 6 more days × ~12 GB MP4 each = **~75 GB additional** (DAY1 already on disk for 1.2 GB / 91 chunks).
- **Existing infrastructure**: same `preprocess/build_memory.py` pipeline. All 4 axes (episodic / semantic / spatial / visual) already shipped on DAY1.
- **Cost**: HF download time + storage. Then re-run extraction across DAY2-7.
- **Risk**: same sparsity pattern likely; A1_JAKE may also be single-home stationary.

### Path C — **Habitat-Sim HM3D-Semantic 1-hour scripted walk** (synthetic GT) ⭐ first choice if "clean experiment" matters
- **Get**: 8-15 rooms, 3600 s walk, full GT (pose / depth / instance segmentation / room labels).
- **Existing infrastructure**: need new `tools/build_habitat_spatial_sidecar.py` (uses `habitat-sim` Python pkg).
- **Cost**: install habitat-sim (~500 MB), download 1 HM3D building scene (~hundreds of MB), script walk.
- **Risk**: not real wearable; gain experimental clarity but lose realism.

## 4. My recommendation

**Hybrid: Path A primary + Path C secondary.**

1. **Apply for Nymeria via Playwright** (same flow as AEA). If access granted within ~minutes, download ONE multi-room daily-activity sequence (~5 GB MPS-only). Re-run `tools/build_aea_spatial_sidecar.py` + `tools/build_aea_place_anchors.py` — the schema is identical to AEA, so it should run with NO code changes.
2. **In parallel, set up Habitat-Sim recipe** as the "clean experiment" baseline. Even if Nymeria gives us real data, Habitat gives us full GT to validate the place-graph algorithm.

**Why not Path B alone**: EgoLife DAY2-DAY7 = same sparsity pattern as DAY1, same single-subject single-home. Multi-day adds longitudinal but does NOT add multi-room.

**Why not EPIC-KITCHENS / HoloAssist alone**: short / single-room / weak spatial signals.

## 5. If only one pull, then…

**Nymeria.** Because:
- Reuses AEA pipeline (zero new code if MPS-equivalent files).
- Multi-room daily activity is exactly the missing axis.
- Same Meta Aria hardware → comparable to EgoLife.
- License gate is identical to AEA which we already crossed.

If Nymeria license is more restrictive than AEA, fall back to **Habitat-Sim HM3D-Semantic recipe**.

## 6. Storage budget for Path A

Per Nymeria sequence MPS-only (estimate, similar to AEA):
- closed_loop_trajectory.csv ≈ 20-60 MB
- general_eye_gaze.csv ≈ 50-200 KB
- semidense_points.csv.gz ≈ 25-200 MB
- semidense_observations.csv.gz ≈ 100-200 MB
- speech.csv ≈ KB-MB depending on noise
- online_calibration.jsonl ≈ 5-10 MB

**Total**: ~150-500 MB per sequence MPS-only (skipping VRS ~2-7 GB).

## 7. Next action

1. Open https://www.projectaria.com/datasets/nymeria/ via Playwright.
2. Submit email + accept license.
3. Pull CDN URLs JSON.
4. Pick the SMALLEST multi-room sequence.
5. Download MPS only.
6. Run existing AEA pipeline; verify multi-place DBSCAN finally produces ≥ 3 places.
7. Document as `§17. Nymeria spatial sidecar (multi-room verification)` in `docs/spatial-memory-ablation.md`.

Estimated wall: ~30-60 min for everything if Nymeria license is open via web form (like AEA).
