# Multi-room egocentric / first-person datasets with portable spatial signals

Date: 2026-05-22 KST  
Request type: TYPE D comprehensive survey.  
Decision target: replace EgoLife/AEA for testing multi-place graph traversal.

## Executive summary

EgoLife and AEA fail because they do not force a single recording to traverse multiple places. Best candidates split into two groups:

1. **Real egocentric Aria data with portable MPS files**: **Nymeria**, **Ego-Exo4D**, **Aria Gen 2 Pilot**, partly **ADT**. These ship 6-DoF trajectory / point cloud / gaze as CSV/JSON-style MPS outputs; raw video/audio may still be VRS, but pose/point-cloud/gaze need not be decoded from VRS.
2. **Simulator / RGB-D scan data with plain files**: **TartanAir V2**, **ScanNet**, **HM3D/Replica via Habitat**. These are better for zero-friction spatial graph stress, but less “wearer daily-life” than Aria datasets.

**Top 3 shortlist** at end: **Nymeria**, **TartanAir V2**, **ScanNet**. Ego-Exo4D is close, but activity captures may be room-local even when captures are long.

## Evidence anchors

- Project Aria MPS outputs are portable: `closed_loop_trajectory.csv`, `open_loop_trajectory.csv`, `semidense_points.csv.gz`, `semidense_observations.csv.gz`, `general_eye_gaze.csv`, `summary.json` ([MPS Basics](https://facebookresearch.github.io/projectaria_tools/docs/data_formats/mps/mps_summary), [Trajectory](https://facebookresearch.github.io/projectaria_tools/docs/data_formats/mps/slam/mps_trajectory), [Multi-SLAM](https://facebookresearch.github.io/projectaria_tools/docs/data_formats/mps/slam/mps_multi_slam)).
- Nymeria official page: 300 hours, 1200 sequences, 50 locations, 400 km trajectory; each recording 15 min; 47 houses, 201 rooms, 37 multi-story houses; Aria MPS gives 6-DoF trajectory, semi-dense point clouds, eye gaze depth ([official](https://www.projectaria.com/datasets/nymeria/), [arXiv 2406.09905](https://arxiv.org/abs/2406.09905)).
- ADT official page: 200 sequences / ~400 min, 2 real indoor scenes; 6-DoF device trajectory, 3D object pose, human skeleton, eye gaze, depth map, instance segmentation; scene geometry aligned to poses ([official](https://www.projectaria.com/datasets/adt/), [docs data format](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset/data_format), [GitHub tools](https://github.com/facebookresearch/projectaria_tools/tree/main/projects/AriaDigitalTwinDatasetTools)).
- Aria Gen 2 Pilot official page: cleaning/cooking/eating/playing/outdoor walking; on-device SLAM + offline MPS; 8-channel audio, GPS/Wi-Fi/Bluetooth, depth estimation, 3D object detection; CC BY-NC 4.0 ([official](https://www.projectaria.com/datasets/gen2pilot/), [arXiv 2510.16134](https://arxiv.org/abs/2510.16134), [GitHub tools](https://github.com/facebookresearch/projectaria_tools)).
- Ego4D docs: v2.1 metadata has durations/scenarios; IMU/gaze are CSV; unprocessed components include huge video/IMU/gaze/audio, but no general 6-DoF/depth release ([start](https://ego4d-data.org/docs/start-here/), [metadata](https://ego4d-data.org/docs/data/metadata/), [IMU](https://ego4d-data.org/docs/data/imu/), [gaze](https://ego4d-data.org/docs/data/gaze/), [unprocessed](https://ego4d-data.org/docs/data/unprocessed_data/), [GitHub](https://github.com/facebookresearch/Ego4d)).
- Ego-Exo4D docs: V2 has 1286.30 video hours / 221.26 ego-hours / 5035 takes; takes 1–42 min; Aria + GoPro; MPS trajectory sampled at 1 kHz; point clouds and eye gaze available as files ([official](https://ego-exo4d-data.org/), [docs overview](https://docs.ego-exo4d-data.org/overview/), [MPS docs](https://docs.ego-exo4d-data.org/data/mps/), [arXiv 2311.18259](https://arxiv.org/abs/2311.18259), [GitHub](https://github.com/facebookresearch/Ego4d)).
- HM3D: 1000 building-scale scenes, academic/non-commercial, Habitat simulator ([official](https://aihabitat.org/datasets/hm3d/), [arXiv 2109.08238](https://arxiv.org/abs/2109.08238), [GitHub](https://github.com/facebookresearch/habitat-matterport3d-dataset)).
- ScanNet: 2.5M RGB-D views, >1500 scans, 3D camera poses, reconstructions, instance semantic segmentation ([official](http://www.scan-net.org/), [arXiv 1702.04405](https://arxiv.org/abs/1702.04405), [GitHub](https://github.com/ScanNet/ScanNet)).
- Matterport3D / Habitat-Web source data: 90 building-scale scenes, 10,800 panoramas, 194,400 RGB-D images, camera poses, 2D/3D semantic segmentation ([official](https://niessner.github.io/Matterport/), [arXiv 1709.06158](https://arxiv.org/abs/1709.06158), [GitHub](https://github.com/niessner/Matterport)).
- iGibson: 15 interactive high-quality scenes, hundreds of large 3D scenes, 12k+ compatible scenes; simulation, not pre-recorded wearer data ([official](https://svl.stanford.edu/igibson/), [GitHub](https://github.com/StanfordVL/iGibson)).
- TartanAir V2: synthetic trajectories with RGB, depth, segmentation, optical flow, camera poses, LiDAR; challenging recorded trajectories ([docs](https://www.tartanair.org/), [project page](https://theairlab.org/tartanair-dataset/), [arXiv 2003.14338](https://arxiv.org/abs/2003.14338), [GitHub tools](https://github.com/castacks/tartanair_tools)).
- LaMAria 2025: Project Aria city-scale egocentric SLAM, 63 sequences, indoor/outdoor transitions, km trajectories; likely raw Aria/MPS plus control-point benchmark, not a room dataset ([arXiv 2509.26639](https://arxiv.org/abs/2509.26639), [project](https://www.lamaria.ethz.ch/)).

## Candidate table

| Candidate | Multi-room? | Duration per recording | Total recordings | Pre-extracted spatial signals | Format | SDK required? | License | Download URL | Size for ONE multi-room recording | Spatial richness |
|---|---:|---:|---:|---|---|---|---|---|---:|---:|
| **Nymeria** | **yes** | 15 min | 1200 sequences | 6-DoF device trajectory, semi-dense point cloud, eye gaze w/depth, body motion, narrations, audio in raw streams | MPS CSV/CSV.GZ/JSON plus Aria VRS for raw sensors | **No for pose/point cloud/gaze**; yes/partial for raw VRS | CC BY-NC 4.0 | [official](https://www.projectaria.com/datasets/nymeria/), [explorer](https://explorer.projectaria.com/nymeria) | Unclear / not published per recording | Strong: houses total 201 rooms, 37 multi-story; per-recording room count not published |
| **Aria Digital Twin (ADT)** | mixed / likely limited | ~2 min avg (400 min / 200 seq) | 200 sequences | 6-DoF device trajectory, object pose, human skeleton, gaze, RGB/depth/segmentation, mesh/object models | VRS + ADT annotation files + MPS CSV; docs/tool loader | Partial; portable annotations exist, raw sensor via tools/VRS | research / Meta terms; exact public license unclear in official excerpt | [official](https://www.projectaria.com/datasets/adt/), [docs](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset/data_format) | Unclear / not published | Low-moderate: only 2 real indoor office scenes; “each room” scanned, but per-sequence cross-room unclear |
| **Aria Pilot Dataset** | mixed | ~3 min avg (7.5 h / 159 rec), varied | 143 Everyday + 16 Desktop | MPS-derived eye tracking, 3D trajectories, shared space-time, speech-to-text | older Aria Data Tools layout; raw VRS plus derived files | Partial; older tooling | research / Meta access terms | [docs](https://facebookresearch.github.io/Aria_data_tools/docs/pilotdata/pilotdata-index/) | Unclear / not published | Low for current need: Everyday updated/re-released as AEA; likely same single-location issue |
| **Aria Gen 2 Pilot (A2PD)** | mixed / likely yes for walking, maybe indoor scenes | unclear; initial release batches | unclear; initial release | on-device SLAM, MPS SLAM trajectory/point clouds, hand tracking, eye gaze, depth maps, 3D object detection, directional ASR, 8-ch audio, GPS/Wi-Fi/Bluetooth | MPS CSV/JSONL, extra algorithm outputs; raw VRS | **No for exported MPS/algorithm outputs**, partial for raw VRS | CC BY-NC 4.0 | [official](https://www.projectaria.com/datasets/gen2pilot/), [tools](https://facebookresearch.github.io/projectaria_tools/gen2/research-tools/dataset/pilot/content) | Unclear / not published | Medium: cleaning/cooking/eating/playing/outdoor walking; exact room transitions not published |
| **Ego4D v2/v2.1** | mixed | long videos; many >10 min in metadata | thousands of videos (exact v2.1 count in metadata, not cited here) | IMU CSV, gaze CSV subset, audio/video, annotations; no general 6-DoF/depth | MP4 + CSV + JSON | No for CSV/MP4; no 6-DoF file to use | research license | [official docs](https://ego4d-data.org/docs/start-here/), [CLI](https://github.com/facebookresearch/Ego4d) | Video components huge; per-recording size not published here | Medium for visual location transitions, weak for spatial graph because no pose/depth |
| **Ego-Exo4D** | mixed | 1–42 min takes | 5035 takes, 1286.30 video hours, 221.26 ego-hours | Aria 6-DoF localization, GoPro 6-DoF localization, point clouds, eye gaze, audio/IMU | MPS CSV/CSV.GZ; MP4 for GoPro; VRS for Aria raw | **No for trajectory/point cloud**, partial for VRS raw | research license / data agreement | [official](https://ego-exo4d-data.org/), [docs](https://docs.ego-exo4d-data.org/) | Unclear / not published | Medium: procedural tasks may move between stations; not guaranteed multi-room |
| **HM3D-ABO / Habitat-Web simulated egocentric** | yes by construction | episodes arbitrary | HM3D 1000 scenes; episode count generated | pose, RGB, depth, semantic/instance possible via simulator; not generally pre-recorded | Habitat JSON episode specs + rendered PNG/NPY if generated | **Yes to generate**; no if someone pre-renders | academic non-commercial | [HM3D](https://aihabitat.org/datasets/hm3d/), [GitHub](https://github.com/facebookresearch/habitat-matterport3d-dataset) | Not applicable unless generated | Strong: building-scale scenes, many rooms/floors |
| **ScanNet egocentric RGB-D scans** | **yes / mixed** | minutes per scan, sequence-dependent | >1500 scans | RGB-D frames, per-frame 3D camera poses, mesh/reconstruction, instance/semantic labels | image/depth files + pose text + PLY/JSON-style metadata | No after download; download script/account required | ScanNet Terms of Use | [official](http://www.scan-net.org/), [GitHub](https://github.com/ScanNet/ScanNet) | Unclear / sequence-dependent | Strong: scan trajectories often cover whole apartments/rooms; exact per-scan room count available from scene metadata |
| **Replica + scripted Habitat trajectories** | yes by construction | arbitrary scripted | 18 high-quality indoor scenes in common Replica release; trajectories generated | RGB/depth/semantic/pose if rendered; mesh/semantic scene | generated PNG/NPY/JSON possible | Yes to generate; no if pre-rendered | Replica terms / research | [Habitat-Sim](https://github.com/facebookresearch/habitat-sim), [Replica via Habitat](https://aihabitat.org/) | Not applicable unless generated | Strong for apartments/offices; simulator-dependent |
| **REASSEMBLE / WALT / Hephzibah / DPV-Real** | unclear / mostly not found as stable open datasets | unclear | unclear | likely SLAM/video pose if dataset exists; DPVO repo evaluates TartanAir/EuRoC/TUM/ICL, not DPV-Real dataset | unclear | unclear | unclear | Search found DPVO repo, not a clear DPV-Real dataset page | unclear | Not recommended until source identified |
| **Apartment / home-tour egocentric datasets** | unclear | likely >10 min | scattered videos | usually video only; trajectory annotations rare | MP4 mostly | no SDK, but no pose/depth | varied / often platform terms | No robust open benchmark found in this pass | unclear | Visually rich, spatially weak without pose |
| **iGibson / AI2-THOR procedural traversal recordings** | yes by construction | arbitrary | generated episodes | pose, depth, segmentation, object metadata, scene graph | simulator events/JSON + rendered frames | **Yes to generate** | iGibson/AI2-THOR licenses | [iGibson](https://svl.stanford.edu/igibson/), [AI2-THOR GitHub](https://github.com/allenai/ai2thor) | Not applicable unless generated | Strong, but not pre-recorded wearable data |
| **Habitat 3.0 evaluation episodes** | yes by construction | arbitrary | generated episodes | humanoid/robot poses, RGB/depth/semantics from sensors, articulated object states | simulator dataset JSON + rendered files if exported | **Yes to generate** | Habitat/HSSD/HM3D terms | [Habitat-Lab](https://github.com/facebookresearch/habitat-lab), [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) | Not applicable unless generated | Strong for multi-room navigation/social rearrangement; generated not shipped recordings |
| **First-person VR walkthrough corpora / Monado SLAM Dataset** | mixed | up to ~40 min | 64 non-calibration recordings; 5h15 total | visual-inertial tracking with ground truth; headset poses | likely raw/calibrated files; exact file format needs source confirmation | likely no SDK? unclear | CC BY 4.0 stated in arXiv excerpt | [arXiv 2508.00088](http://arxiv.org/pdf/2508.00088) | Unclear / not published | Medium: VR/gameplay/headset motions; room count unclear |
| **TartanAir V2** | **yes, simulated** | trajectory-dependent | many trajectories across environments | stereo/RGB, depth, segmentation, optical flow, camera poses, LiDAR | images + depth/seg/pose text/NPY-style files via tools | No for downloaded files; Python tool optional | research/open dataset terms | [docs](https://www.tartanair.org/), [project](https://theairlab.org/tartanair-dataset/) | Unclear / environment-dependent | Strong: synthetic homes/caves/offices; controllable multi-room paths |
| **LaMAria** | city-scale, indoor/outdoor not room-centric | short/long; some km | 63 sequences | Aria multi-sensor streams; sparse centimeter-accurate control point annotations | likely Aria/MPS plus benchmark files; exact package unclear | partial | unclear | [project](https://www.lamaria.ethz.ch/), [arXiv](https://arxiv.org/abs/2509.26639) | unclear | Strong for place transitions, weak for semantic rooms |

## Candidate notes

### 1. Nymeria

Best real-world match. Multi-room is **demonstrated at corpus level**: 47 houses, 201 rooms, 37 multi-story houses, 50 locations, 400 km trajectory. Official page says each recording is 15 min, which is long enough for room transitions. The missing piece: **per-recording room counts are not published on the official page**. Need sample metadata from Dataset Explorer or download manifest to select recordings whose trajectory crosses room clusters.

Spatial files: Aria MPS gives 6-DoF trajectory and semi-dense point cloud as CSV/CSV.GZ; gaze also CSV. Good fit for WorldMM place graph. Raw RGB/audio still VRS, but the requested spatial signals are portable.

Verdict: **shortlist #1**.

### 2. ADT

Excellent spatial ground truth: aligned pose, mesh, depth, segmentation, object pose, human skeleton. Weakness: only **2 real indoor scenes** and ~2 min average sequence length. It may exercise object-level 3D reasoning better than multi-room place graph. Multi-room per recording unclear. Good for geometry correctness, less good for place transitions.

Verdict: use as geometry sanity dataset, not main multi-room benchmark.

### 3. Aria Pilot

Older release with MPS-derived trajectories and eye tracking. Everyday Activities subset got updated into AEA, which already fails single-room/multi-place test. Desktop subset is object/tabletop. Outdoor subset was “planned” in old docs. Not a strong candidate unless specific Pilot recordings show room transitions.

Verdict: deprioritize.

### 4. Aria Gen 2 Pilot

Technically attractive: Gen2 on-device SLAM, offline MPS, depth maps, object detection, spatial audio, ASR, GPS/Wi-Fi/Bluetooth. Activities include walking outside, cleaning/cooking/eating/playing. Current uncertainty: official page does not publish count, duration, or per-recording room transitions in the excerpt. If Dataset Explorer exposes a “walking outside” or cleaning route through several rooms, it becomes strong.

Verdict: promising, but verify manifest before use.

### 5. Ego4D v2/v2.1

Good long egocentric videos and likely many location transitions. Bad for this goal: no general released 6-DoF pose/depth/semantic-place files. IMU and gaze are CSV, but no metric spatial trajectory. It can test visual memory and place-language, not metric multi-place graph.

Verdict: reject for spatial graph unless you run SLAM yourself.

### 6. Ego-Exo4D

Much better than Ego4D for spatial data: MPS trajectory, point cloud, eye gaze, Aria/GoPro 6-DoF localization. Takes are 1–42 min. Cooking/bike repair/health tasks can involve counter/sink/pantry/workbench transitions, but docs do not guarantee **multi-room** within a take. Still, it has zero-friction MPS CSV for pose and point cloud.

Verdict: good alternate if Nymeria access/metadata hard. Need filter takes by trajectory length + spatial clustering.

### 7. HM3D-ABO / Habitat-Web

Great for synthetic multi-room by construction. Weakness: not shipped as pre-extracted egocentric recordings; you generate episodes/renders through Habitat. This violates “zero SDK install friction” unless someone provides pre-rendered trajectories. Useful for controlled stress tests after generation.

Verdict: not shortlist for current constraint.

### 8. ScanNet egocentric subset

Not wearable glasses, but true first-person RGB-D scanning. Strong spatial package: RGB-D frames, camera poses, reconstructions, semantic/instance labels. Many scans traverse multiple rooms / entire apartments. Portable files after download. No VRS. Good for testing place graph with real indoor geometry.

Verdict: **shortlist #3** if “egocentric scanner” acceptable.

### 9. Replica + scripted Habitat trajectories

Good clean simulation, but not pre-extracted unless you render/export trajectories. More controlled than ScanNet, less “dataset already has recordings”. Useful as a generated benchmark.

Verdict: backup simulator route.

### 10. REASSEMBLE / WALT / Hephzibah / DPV-Real

Could not verify stable official open dataset pages from this pass. Search mostly surfaced DPVO/DPV-SLAM code and standard benchmarks. Treat as **unknown** until exact URLs/papers provided.

Verdict: do not spend integration time yet.

### 11. Apartment / home-tour egocentric datasets

Likely visually ideal, but public home-tour videos rarely include metric trajectory/depth/semantic labels. Without spatial files they reproduce EgoLife-like weakness.

Verdict: reject unless paired with SLAM output.

### 12. iGibson / AI2-THOR procedural traversal recordings

Excellent procedural multi-room control, but recordings are generated on demand. Formats can be JSON/images/depth/segmentation, no VRS. Requires simulator install or pre-render job.

Verdict: not zero-friction unless pre-rendered.

### 13. Habitat 3.0 evaluation episodes

Strong for multi-room humanoid/robot tasks, but again simulator episodes, not shipped egocentric recordings with all frames. Useful if WorldMM wants generated exhaustive place graphs.

Verdict: not current primary.

### 14. First-person VR walkthrough corpora

Monado SLAM Dataset looks relevant: real XR headset, 64 recordings, 5h15, up to ~40 min, dense ground truth, CC BY 4.0 per arXiv excerpt. But room/place labels and exact file formats need source confirmation. VR gameplay may not map to physical rooms.

Verdict: promising for SLAM stress, uncertain for semantic multi-room place graph.

## Shortlist: 3 datasets that beat EgoLife and AEA

### 1) Nymeria — best real egocentric daily-life multi-room candidate

Why it beats EgoLife/AEA:

- 15-min continuous recordings vs EgoLife sparse bursts.
- Houses with many rooms and multi-story layouts vs AEA single-room activities.
- MPS portable spatial files: trajectory CSV, point cloud CSV.GZ, gaze CSV.
- Audio present in Aria raw stream; spatial signals do not require raw VRS decoding.

Integration plan:

1. Download manifest/sample from Dataset Explorer.
2. Pick recordings with largest trajectory path length and multiple spatial clusters.
3. Build room/place nodes by clustering `closed_loop_trajectory.csv` plus `semidense_points.csv.gz`; optionally use narration for place labels.

Risk: published docs prove corpus-level rooms, not per-recording room count. Need one manifest/sample validation.

### 2) TartanAir V2 — best zero-SDK synthetic stress dataset

Why it beats EgoLife/AEA:

- Predefined simulated trajectories with camera pose, depth, segmentation, optical flow.
- Multi-room/large environments by construction.
- Portable files; no VRS; no Aria SDK.
- Easy to select trajectories by path length and semantic diversity.

Integration plan:

1. Use downloaded trajectory folders only; skip Python package if direct files accessible.
2. Convert pose + segmentation/depth into WorldMM place graph.
3. Use as regression benchmark for multi-room graph traversal before real Aria data.

Risk: not real wearable; no natural audio/gaze.

### 3) ScanNet — best real indoor portable RGB-D/pose baseline

Why it beats EgoLife/AEA:

- Real first-person RGB-D scan trajectories often traverse rooms/whole apartments.
- Per-frame camera poses + depth + semantic/instance reconstructions are portable.
- No VRS/Aria SDK.
- Dense geometry and labels make place graph validation easier.

Integration plan:

1. Pick scenes with multiple room labels / large trajectory extents.
2. Use `pose`, `depth`, RGB, semantic mesh/labels.
3. Derive room transitions from pose path crossing semantic/room partitions.

Risk: not wearable daily-life; scanner motion differs from egocentric AR glasses.

## Final recommendation

Start with **Nymeria** if access is available. It is closest to requested real egocentric multi-room daily activity and ships MPS spatial CSVs. In parallel, add **TartanAir V2** or **ScanNet** as zero-SDK baselines: TartanAir for controlled simulated multi-room paths; ScanNet for real indoor RGB-D/pose geometry. Use **Ego-Exo4D** as fourth candidate if Nymeria metadata cannot identify per-recording room transitions.
