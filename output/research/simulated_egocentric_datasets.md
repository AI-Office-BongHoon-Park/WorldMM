# Simulated egocentric datasets for WorldMM stress tests

**Request type:** Type D / comprehensive research. Date context: 2026-05-22 KST. 2025-only claims filtered unless still current.

## Executive take

Best fit is **Habitat-Sim + HM3D-Semantic/Replica/MP3D** for reproducible long, multi-room egocentric walks. It gives building-scale meshes, navmeshes, RGB-D, egomotion, semantic sensors, and fast replay/rerender. Main weakness: many scene assets are research-license / gated, not fully permissive. Second best is **AI2-THOR + ProcTHOR**: procedural multi-room houses, Apache-2.0 code, rich object metadata and segmentation/depth, easier dataset synthesis. Main weakness: Unity install + less realistic scan geometry than HM3D/Replica.

For WorldMM, choose **Habitat** when spatial memory is the target axis; choose **ProcTHOR** when semantic/event coverage and many generated houses matter more than scan realism.

## Evidence base

Key citations used below:

- Habitat-Sim repo says it supports 3D scans including HM3D, Matterport3D, Gibson, Replica; RGB-D cameras; egomotion; Bullet physics; and reports very high simulator throughput on a Titan Xp-class GPU ([GitHub permalink](https://github.com/facebookresearch/habitat-sim/blob/57ee4941dc4765240f0f91f70b2c97a919bf9038/README.md#L16-L23)). Habitat paper: [arXiv/ICCV 2019](https://arxiv.org/abs/1904.01201).
- HM3D repository states 1,000 high-resolution Matterport digital twins, GLB/OBJ meshes, textures, and academic non-commercial access ([GitHub permalink](https://github.com/matterport/habitat-matterport-3dresearch/blob/9b74d5aa58e371c9bc16b2ccb64748cddf9bf1c6/README.md#L1-L18)); HM3D paper: [OpenReview](https://openreview.net/forum?id=-v4OuqNs5P).
- HM3D-Semantic includes `*.semantic.glb`, `*.semantic.txt`, semantic configs, and Habitat loads them by enabling a semantic sensor ([GitHub permalink](https://github.com/facebookresearch/habitat-sim/blob/57ee4941dc4765240f0f91f70b2c97a919bf9038/DATASETS.md#L86-L113)).
- AI2-THOR README lists 200+ high-quality scenes, 2600+ objects, multi-agent support, 200+ actions, ego-centric RGB, instance/semantic segmentation, depth, normals, top-down and third-person camera frames, plus metadata after each step ([GitHub permalink](https://github.com/allenai/ai2thor/blob/24f79883b4889e3f0e6f4ae301808b9025872dfc/README.md#L66-L78)); paper: [arXiv:1712.05474](https://arxiv.org/abs/1712.05474).
- ProcTHOR README says it procedurally generates interactive, diverse, semantically plausible houses compatible with AI2-THOR, and installs via PyPI ([GitHub permalink](https://github.com/allenai/procthor/blob/53d5bd4c8c96a699e6a615dc390abb670cc9d353/README.md#L8-L22)); paper: [NeurIPS 2022](https://procthor.allenai.org/).
- iGibson docs list RGB, normals, segmentation, point cloud, depth, optical flow, scene flow, and 1/16-beam LiDAR sensors ([GitHub permalink](https://github.com/StanfordVL/iGibson/blob/3ad2aefabcf1f370ef3dabf704d0ac3becce0df9/docs/environments.md#L11-L16)); iGibson 2.0 paper: [arXiv:2108.03272](https://arxiv.org/abs/2108.03272).
- OmniGibson README advertises photorealistic visuals, physics, large-scale scenes/objects, dynamic kinematic/semantic object states, and mobile manipulators ([GitHub permalink](https://github.com/StanfordVL/OmniGibson/blob/dac15ed283a03deba88cf808bb7097b6918d7ccc/OmniGibson/README.md#L28-L35)); its vision sensor exposes RGB, depth, normals, semantic/instance segmentation, optical flow, 2D/3D boxes, camera state, point cloud ([GitHub permalink](https://github.com/StanfordVL/OmniGibson/blob/dac15ed283a03deba88cf808bb7097b6918d7ccc/OmniGibson/omnigibson/sensors/vision_sensor.py#L26-L76)).
- Aria Digital Twin tools document 2D/3D boxes, eye gaze direction/depth, depth, segmentation, and a full synthetic twin recording per raw dataset ([GitHub permalink](https://github.com/facebookresearch/projectaria_tools/blob/1bfe7e50c52698583b2ee3bbb63c5c177d962311/projects/AriaDigitalTwinDatasetTools/ReadMe.md#L1-L10)). Project Aria Tools list ASE as an open dataset entry and Apache-2.0 tools ([GitHub permalink](https://github.com/facebookresearch/projectaria_tools/blob/1bfe7e50c52698583b2ee3bbb63c5c177d962311/README.md#L131-L150)); ADT paper: [arXiv:2306.06362](https://arxiv.org/abs/2306.06362).
- TartanAir tools state data is organized in trajectory folders with RGB, depth, segmentation, camera pose, and flow; files are PNG/NPY/TXT ([GitHub permalink](https://github.com/castacks/tartanair_tools/blob/841584a8a1ebbdcc832430212609d48b6b624443/README.md#L12-L40)); paper: [arXiv:2011.00359](https://arxiv.org/abs/2011.00359).
- SAPIEN README describes a realistic physics-rich articulated-object environment and GPU requirement; depth sensor citation appears in the README ([GitHub permalink](https://github.com/haosulab/SAPIEN/blob/731622eac5b140b320076c8a1b6eb4b553c3ccd4/readme.md#L1-L16), [depth citation](https://github.com/haosulab/SAPIEN/blob/731622eac5b140b320076c8a1b6eb4b553c3ccd4/readme.md#L234-L245)); SAPIEN paper: [arXiv:2003.08515](https://arxiv.org/abs/2003.08515).
- GibsonEnv repo documents 572 spaces / 1440 floors and semantic masks when annotated ([GitHub permalink](https://github.com/StanfordVL/GibsonEnv/blob/f474d9efd5b5ef703e3bf630a6f7448b54875d0c/README.md#L30-L32), [semantic masks](https://github.com/StanfordVL/GibsonEnv/blob/f474d9efd5b5ef703e3bf630a6f7448b54875d0c/README.md#L331-L337)); paper: [CVPR 2018](http://gibson.vision/).
- EmbodiedScan repo says it is egocentric, multi-modal, with 5k scans, 1M RGB-D views, 1M language prompts, 160k 3D boxes over 760 categories, and dense semantic occupancy over 80 categories ([GitHub permalink](https://github.com/InternRobotics/EmbodiedScan/blob/fe26e4bc3f3fb706fd7e33788766f61f8857fc3c/README.md#L40-L44)); data prep requires raw datasets and a form ([GitHub permalink](https://github.com/InternRobotics/EmbodiedScan/blob/fe26e4bc3f3fb706fd7e33788766f61f8857fc3c/data/README.md#L1-L15)); paper: [arXiv:2312.16170](https://arxiv.org/abs/2312.16170).
- Habitat-Lab supports single/multi-agent tasks, humanoids/robots, sensors, and human-in-the-loop data collection ([GitHub permalink](https://github.com/facebookresearch/habitat-lab/blob/0fb6f43ffe806a8088a171b036336c093bcf604e/README.md#L17-L26), [HITL](https://github.com/facebookresearch/habitat-lab/blob/0fb6f43ffe806a8088a171b036336c093bcf604e/habitat-hitl/README.md#L1-L8)); Habitat 3.0 paper: [arXiv:2310.13724](https://arxiv.org/abs/2310.13724).

## Catalog

| Corpus / simulator | Render | Auto spatial signals | Scale | Format | Trajectory scripting / replay | License / friction | Compute | WorldMM value |
|---|---:|---|---|---|---|---|---|---|
| **Habitat-Sim + HM3D/Replica/MP3D** | high to photoreal for scans | RGB-D, egomotion, pose from agent state, semantic sensor where semantic assets exist; navmesh. Evidence: RGB-D/egomotion support ([link](https://github.com/facebookresearch/habitat-sim/blob/57ee4941dc4765240f0f91f70b2c97a919bf9038/README.md#L16-L23)); HM3D semantic assets/sensor ([link](https://github.com/facebookresearch/habitat-sim/blob/57ee4941dc4765240f0f91f70b2c97a919bf9038/DATASETS.md#L86-L113)). | multi-room to building-scale; HM3D has 1,000 Matterport spaces ([link](https://github.com/matterport/habitat-matterport-3dresearch/blob/9b74d5aa58e371c9bc16b2ccb64748cddf9bf1c6/README.md#L1-L18)). | GLB/OBJ scene meshes, navmesh, semantic GLB/TXT; user can export JSON/NPY/HDF5. | Script agent on navmesh, save `(timestamp, position, rotation, action)`; later reset agent state and rerender with any camera. | Code MIT; HM3D/MP3D gated/non-commercial; Replica license separate. Not zero-friction for full datasets. | GPU renderer preferred; published speed on Titan Xp, no RTX 3050 VRAM claim. | Best spatial/visual stressor. Strong for “where is X?” if semantic annotations exist; “what did I see in kitchen?” if room/place labels are added from semantic scene graph or manual region map. |
| **HM3D-Semantic** | high | semantic class IDs via semantic sensor; RGB-D/pose via Habitat. | building-scale. | `.semantic.glb`, `.semantic.txt`, scene dataset config ([link](https://github.com/facebookresearch/habitat-sim/blob/57ee4941dc4765240f0f91f70b2c97a919bf9038/DATASETS.md#L86-L90)). | same as Habitat. | academic/non-commercial HM3D terms ([link](https://github.com/matterport/habitat-matterport-3dresearch/blob/9b74d5aa58e371c9bc16b2ccb64748cddf9bf1c6/README.md#L3-L18)). | GPU preferred. | Best semantic+spatial corpus inside Habitat. |
| **AI2-THOR** | medium-high, near photoreal synthetic | ego RGB, instance/semantic segmentation, depth, normals, top-down/third-person camera frames, metadata after each step ([link](https://github.com/allenai/ai2thor/blob/24f79883b4889e3f0e6f4ae301808b9025872dfc/README.md#L66-L78)). | mostly room / apartment; multi-agent; 200+ scenes. | Event frames as NumPy arrays in Python; metadata as JSON-like dict. | `Controller.step()` action scripts; save events/metadata; replay action sequence deterministically enough if build/seed fixed. | Code Apache-2.0 ([license](https://github.com/allenai/ai2thor/blob/24f79883b4889e3f0e6f4ae301808b9025872dfc/LICENSE#L1-L10)); assets governed by AI2 terms. | Unity GPU/CPU; easier than Omniverse. | Excellent episodic+semantic. Object metadata makes QA generation easy. Spatial scale weaker than HM3D. |
| **ProcTHOR** | medium-high synthetic | inherits AI2-THOR signals; procedural house topology and objects. | multi-room procedural houses. | house JSON + AI2-THOR event frames. | Generate house, load in AI2-THOR, script walk. | Code Apache-2.0 ([license](https://github.com/allenai/procthor/blob/53d5bd4c8c96a699e6a615dc390abb670cc9d353/LICENSE#L1-L8)); PyPI install and example script ([link](https://github.com/allenai/procthor/blob/53d5bd4c8c96a699e6a615dc390abb670cc9d353/README.md#L13-L22)). | Unity; GPU useful. | Best scalable synthetic house generator. Strong for long “day” by stitching houses/tasks; weaker real-world texture fidelity. |
| **iGibson 2.0** | high-ish PBR, interactive | RGB, normals, segmentation, point cloud, depth, optical flow, scene flow, LiDAR ([link](https://github.com/StanfordVL/iGibson/blob/3ad2aefabcf1f370ef3dabf704d0ac3becce0df9/docs/environments.md#L11-L16)); object states and VR demo collection in README ([link](https://github.com/StanfordVL/iGibson/blob/3ad2aefabcf1f370ef3dabf704d0ac3becce0df9/README.md#L1-L22)). | homes/offices; 15 fully interactive scenes + imported scenes. | YAML configs, NumPy observations, scene-specific assets. | Gym-like env step; record robot state + observations; replay depends on physics determinism. | Code MIT ([license](https://github.com/StanfordVL/iGibson/blob/3ad2aefabcf1f370ef3dabf704d0ac3becce0df9/LICENSE#L1-L8)); data terms separate. | GPU renderer/Bullet. | Good for manipulation/object-state episodic memory. More install friction than Habitat/AI2. |
| **OmniGibson / BEHAVIOR-1K** | photoreal / RTX | RGB, depth, normals, semantic/instance seg, optical flow, 2D/3D boxes, camera state, pointcloud ([link](https://github.com/StanfordVL/OmniGibson/blob/dac15ed283a03deba88cf808bb7097b6918d7ccc/OmniGibson/omnigibson/sensors/vision_sensor.py#L26-L76)); object states and large scenes ([link](https://github.com/StanfordVL/OmniGibson/blob/dac15ed283a03deba88cf808bb7097b6918d7ccc/OmniGibson/README.md#L28-L35)). | household scenes + tasks; 1,000 activities in BEHAVIOR-1K root README ([link](https://github.com/StanfordVL/OmniGibson/blob/dac15ed283a03deba88cf808bb7097b6918d7ccc/README.md#L1-L13)). | USD assets, Gym observations; user-export JSON/NPY/HDF5. | Script robot/task controllers; replay via saved states/actions, but physics/Isaac version can affect exactness. | OmniGibson code MIT ([license](https://github.com/StanfordVL/OmniGibson/blob/dac15ed283a03deba88cf808bb7097b6918d7ccc/OmniGibson/LICENSE#L1-L12)); Isaac Sim install heavy. | RTX GPU/Isaac runtime; high friction. | Best for physics-rich visual+semantic episodes, not best “zero friction.” |
| **Aria Synthetic Environments / ADT synthetic twins** | high; Aria sensor realism | ADT tooling: 2D/3D boxes, gaze direction/depth, depth images, segmentation images, full synthetic twin ([link](https://github.com/facebookresearch/projectaria_tools/blob/1bfe7e50c52698583b2ee3bbb63c5c177d962311/projects/AriaDigitalTwinDatasetTools/ReadMe.md#L1-L10)); project tools list ASE ([link](https://github.com/facebookresearch/projectaria_tools/blob/1bfe7e50c52698583b2ee3bbb63c5c177d962311/README.md#L131-L150)). | room/building snippets depending dataset. | VRS + MPS CSV/JSON-style outputs; dataset-specific. | Replay real/synthetic Aria streams with tools; not a general house-walk simulator. | Tools Apache-2.0; dataset license must be accepted. | CPU for reading; GPU for model use. | Best for Aria-like egocentric gaze/depth/schema compatibility. Weak for scripted arbitrary day-long houses. |
| **TartanAir / SLAM synthetic** | high synthetic outdoor/indoor varied | RGB, depth, segmentation, camera pose, optical flow; organized as trajectory folders ([link](https://github.com/castacks/tartanair_tools/blob/841584a8a1ebbdcc832430212609d48b6b624443/README.md#L12-L40)). | large trajectories, often drone-like not human house. | PNG RGB; NPY depth/seg/flow; TXT poses. | Pre-recorded trajectories; rerender only if AirSim scene/control recreated, not from dataset alone. | Dataset large; tools available; license check per website. | No sim needed for data use; download heavy up to TB scale ([link](https://github.com/castacks/tartanair_tools/blob/841584a8a1ebbdcc832430212609d48b6b624443/README.md#L17-L19)). | Great SLAM baseline for pose/depth robustness. Poor kitchen/room episodic semantics. |
| **Replica RGB-D scans** | photoreal scan | scene mesh/texture; via Habitat can render RGB-D and semantics if annotations available. | apartment/room-scale, multi-room some scenes. | mesh assets; Habitat observations. | Same as Habitat. | Replica dataset terms; Habitat points to download instructions ([link](https://github.com/facebookresearch/habitat-sim/blob/57ee4941dc4765240f0f91f70b2c97a919bf9038/DATASETS.md#L191-L194)). | GPU preferred. | High visual realism; less scale than HM3D. Good calibration set. |
| **SAPIEN / PartNet-Mobility** | high object/physics | articulated object pose/part structure, cameras, realistic depth sensor support. | object/tabletop/manipulation, not house-scale. | URDF-like assets, simulator observations; user-export. | Script robot/camera around articulated objects. | SAPIEN code/data terms; PyPI install, GPU required ([link](https://github.com/haosulab/SAPIEN/blob/731622eac5b140b320076c8a1b6eb4b553c3ccd4/readme.md#L1-L16)). | GPU required; no 3050 VRAM quote found. | Adds articulated-object memory (“which cabinet door opened?”). Not primary multi-room corpus. |
| **Gibson Env** | medium by 2026 standards | RGB-D, semantic masks when annotated ([link](https://github.com/StanfordVL/GibsonEnv/blob/f474d9efd5b5ef703e3bf630a6f7448b54875d0c/README.md#L331-L337)). | 572 spaces / 1440 floors ([link](https://github.com/StanfordVL/GibsonEnv/blob/f474d9efd5b5ef703e3bf630a6f7448b54875d0c/README.md#L30-L32)). | legacy environment assets. | Gym-like demos; old stack. | Older, cited; more maintenance risk. | GPU. | Useful historical baseline; prefer Habitat/iGibson now. |
| **EmbodiedScan / EmbodiedScan-Sim** | real-scan RGB-D, not simulator-first | 1M egocentric RGB-D, 3D boxes, dense semantic occupancy, language prompts ([link](https://github.com/InternRobotics/EmbodiedScan/blob/fe26e4bc3f3fb706fd7e33788766f61f8857fc3c/README.md#L40-L44)). | 5k scans. | PKL infos, JSON visual grounding, occupancy folders ([link](https://github.com/InternRobotics/EmbodiedScan/blob/fe26e4bc3f3fb706fd7e33788766f61f8857fc3c/data/README.md#L16-L48)). | Mostly dataset replay, not arbitrary rerender. | Raw datasets require licenses/forms ([link](https://github.com/InternRobotics/EmbodiedScan/blob/fe26e4bc3f3fb706fd7e33788766f61f8857fc3c/data/README.md#L1-L15)); code Apache-2.0. | Data processing only unless rendering raw scans. | Excellent benchmark for spatial language grounding. Not zero-friction, not scripted synthetic day. |
| **Habitat 3.0 / SocialAI-style multi-agent** | inherits Habitat | multi-agent tasks, humanoids, robots, sensors, HITL/VR collection ([link](https://github.com/facebookresearch/habitat-lab/blob/0fb6f43ffe806a8088a171b036336c093bcf604e/README.md#L17-L26), [HITL](https://github.com/facebookresearch/habitat-lab/blob/0fb6f43ffe806a8088a171b036336c093bcf604e/habitat-hitl/README.md#L1-L8)). | multi-room, human+robot. | Habitat episode JSON + observations. | Script avatars/agents; record per-agent trajectories. | Code MIT; scene assets same Habitat constraints. | GPU preferred. | Adds social/spatial behavior: “where was person X relative to me?” Useful extension after single-agent corpus. |
| **Isaac Sim / NVIDIA Replicator / Mateusz3D-like releases** | photoreal RTX | Replicator can generate synthetic perception labels, but no single permissive egocentric multi-room corpus was verified here. | depends on USD scene. | USD + writer outputs. | Script camera/robot in Isaac; rerender flexible. | Heavy install; terms not a simple OSS dataset. | RTX GPU; likely too heavy for zero-friction 3050 workflow unless small scenes. | Tooling candidate, not corpus candidate. Do not rank as “best corpus” without a verified dataset package. |

## Best 2 for WorldMM

### 1) Habitat-Sim + HM3D-Semantic / Replica

Why best: multi-room/building-scale spatial layout is the stressor WorldMM lacks. Habitat has navmesh control, repeatable camera trajectories, RGB-D/egomotion, semantic sensors, and fast offline rendering. HM3D-Semantic adds labels without inventing them. It directly tests:

- **Spatial:** pose graph, room transitions, object/place localization.
- **Visual:** high-fidelity scan views, occlusion, revisits.
- **Semantic:** class labels per pixel when annotations exist.
- **Episodic:** scripted walks can revisit rooms with time gaps.

Caveat: HM3D/MP3D are not permissive open assets. Use for internal academic stress tests; do not redistribute derived frames unless license allows.

### 2) AI2-THOR + ProcTHOR

Why second: procedural houses plus AI2-THOR metadata make synthetic QA and episode generation easy. It may be less photoreal than scans, but it is much better for controlled semantic/event coverage: object states, interactions, multi-agent support, many actions, and house generation. It directly tests:

- **Semantic:** object inventories, receptacles, affordances.
- **Episodic:** action scripts, object interactions, revisits.
- **Spatial:** multi-room house layouts, though less scan-real.
- **Visual:** RGB/depth/segmentation/normals from consistent simulator.

Caveat: Unity runtime friction; “day-long” should be synthesized as continuous scripts across generated houses or one large house, not literal natural day capture.

## One concrete recipe: Habitat-Sim HM3D-Semantic 1-hour walk

**Goal:** one WorldMM-compatible egocentric “day-long-equivalent” sample from one multi-room HM3D building.

**Scene selection**

- Pick one HM3D-Semantic v0.2 building with 8-15 navigable rooms / regions.
- Load `*.basis.glb`, navmesh, `*.semantic.glb`, `*.semantic.txt`, and scene dataset config.
- Define a `room_regions.json` manually or from scene annotations: room id, room type, polygon/volume bounds. Habitat alone does not guarantee human room names for every scene; do not fabricate them.

**Trajectory**

- Duration: 3600 s.
- Control frequency: 5 Hz saved observations = **18,000 frames**.
- Agent height: adult egocentric camera, e.g. 1.5-1.7 m; keep fixed in metadata.
- Path: sample 60 navigation goals across 10 rooms; each goal uses shortest path on navmesh; add dwell segments of 20-90 s in kitchen/living/bedroom; revisit at least 3 rooms after >20 min gap.
- Motion: max speed ~1.0 m/s, yaw-rate capped; insert head-turn sweeps at room entrances.

**Sensors per frame**

- `rgb`: 640x480 uint8, shape `(18000, 480, 640, 3)`; stored as chunked JPEG/PNG or HDF5 `uint8`.
- `depth_m`: 640x480 float16 or float32, shape `(18000, 480, 640)`; meters.
- `semantic_id`: 640x480 uint16/int32, shape `(18000, 480, 640)`; raw Habitat semantic ids mapped by `semantic.txt`.
- `pose_world`: `(18000, 7)` float32 = x,y,z,qx,qy,qz,qw.
- `camera_intrinsics`: `(3,3)` float32 plus width/height/FOV.
- `room_id`: `(18000,)` int16 from region lookup.
- `events.jsonl`: one row per high-level segment: enter room, inspect object class, revisit room, turn-around.
- `nav_path.json`: sampled goals, shortest-path geodesic distance, collision flags if any.

**WorldMM schema mapping**

- Existing video frames: `rgb` -> visual frame store / thumbnails.
- Spatial memory: `pose_world` + `room_id` -> trajectory graph nodes every 1 s and keyframes at room transitions.
- Semantic memory: `semantic_id` histograms + visible class masks -> per-place object inventory.
- Episodic memory: `events.jsonl` + timestamps -> “visited kitchen at t=420s; saw chair/table/sink.”
- Retrieval QA templates:
  - “Where did I last see class X?” uses last frame/room where semantic mask class X area > threshold.
  - “What did I see in the kitchen?” uses room-filtered semantic histograms + keyframes.
  - “How did I get from bedroom to kitchen?” uses pose path between room transition events.

**Hardware cost: RTX 3050 6 GB**

No published Habitat-Sim RTX 3050 6 GB VRAM benchmark was found. Therefore no VRAM number quoted. Evidence only supports that Habitat-Sim is designed for speed and has published Matterport rendering throughput on a Titan Xp-class GPU ([GitHub permalink](https://github.com/facebookresearch/habitat-sim/blob/57ee4941dc4765240f0f91f70b2c97a919bf9038/README.md#L23-L23)). Practical conclusion: the recipe should be attempted on the existing RTX 3050 at 640x480, one simulator process, one scene loaded, sequential frame writing. If memory fails, drop to 320x240 or render RGB/depth/semantic in passes. Do not use OmniGibson/Isaac for the first ablation on this GPU; install/runtime cost and RTX memory pressure are higher.

## Recommendation

Start with **Habitat-Sim HM3D-Semantic** for the main WorldMM stress-test. Produce 20 one-hour walks over 10 buildings, 5 Hz, 640x480, RGB/depth/semantic/pose/room ids. Add **ProcTHOR** as controlled procedural ablation: same 1-hour script, but vary house seed/object layout to isolate semantic vs spatial generalization. Keep OmniGibson/SAPIEN as later manipulation extensions, ASE/ADT as Aria-format validation, and TartanAir as pose/depth SLAM baseline only.
