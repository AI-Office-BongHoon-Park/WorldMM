# Pre-extracted spatial egocentric datasets for WorldMM

Date: 2026-05-21 KST

Scope: datasets where spatial signals are already shipped as portable text/numeric files (CSV/JSON/NPY/NPZ/etc.) or where they are clearly not. "SDK required" below means required to download or decode the relevant signal, not merely useful for visualization.

## Key verdicts

- Best zero-SLAM candidates: **Aria Everyday Activities**, **Ego-Exo4D**, **HOT3D-Clips**.
- Best object/hand-pose candidate with plain files: **ARCTIC** and **HOI4D**.
- Best HoloLens-style multimodal dump: **HoloAssist**.
- Weak for spatial indexing: **EPIC-KITCHENS-100** (excellent narrations/detections; no native 3D pose/depth trajectory package found in EK-100 release).
- Static scaffolds only: **Replica / HM3D / ScanNet++**. Useful scene priors, not egocentric memory logs.

## Dataset catalog

### 1. Aria Everyday Activities (AEA)

| Field | Value |
|---|---|
| Download URL | https://www.projectaria.com/datasets/aea/ ; data-format doc: https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_everyday_activities_dataset/aea_data_format ; download doc: https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_everyday_activities_dataset/aea_download_dataset ; HF card: https://huggingface.co/datasets/projectaria/aria-everyday-activities |
| Total size | Official download doc says **~353 GB** for 143 sequences. HF dataset card metadata-only page reported **859 kB total file size** in search result; that is not the full data payload. |
| Egocentric? | Yes, Project Aria glasses; some simultaneous two-wearer sequences. |
| Modalities pre-extracted as portable files | RGB/audio/IMU in VRS; **6DoF pose** via MPS SLAM `closed_loop_trajectory.csv` and `open_loop_trajectory.csv`; **eye gaze** `general_eye_gaze.csv`; **semidense point cloud** `semidense_points.csv.gz`; semidense observations `semidense_observations.csv.gz`; speech `speech.csv`; calibration `online_calibration.csv`; metadata JSON. No hand pose or object 3D pose documented for AEA. |
| Pre-extracted file formats | VRS, JSON, CSV, CSV.GZ. Example sequence tree includes `recording.vrs`, `metadata.json`, `speech.csv`, `MPS/eye_gaze/general_eye_gaze.csv`, `MPS/slam/closed_loop_trajectory.csv`, `open_loop_trajectory.csv`, `online_calibration.csv`, `semidense_observations.csv.gz`, `semidense_points.csv.gz`. |
| SDK required to read? | **No for MPS CSV/JSON with pandas/numpy** after download. **Yes/partial for download** because official docs route through `aria_dataset_downloader` in `projectaria_tools`; VRS streams need Project Aria/VRS tooling if you decode raw sensors. |
| License | Project Aria dataset license; gated acceptance required. HF card links license but not a standard SPDX license. Treat as restricted/research license until reviewed. |
| Why interesting for "space as index" | AEA is closest to WorldMM’s desired "AI Glass but no SLAM" target: high-frequency global-ish wearer trajectories, eye-gaze rays, semidense points, speech, and multi-person time sync are already emitted as CSV/JSON/GZ files. It is weaker than ADT/HOT3D on objects/hands but strong for place trajectory + gaze + transcript indexing. |

Evidence: AEA docs show the exact portable MPS tree and HF card says AEA includes MPS outputs: per-frame eye tracking, accurate 3D trajectories, semidense point cloud, shared global coordinate frame, calibration, and speech-to-text.

### 2. Aria Digital Twin (ADT)

| Field | Value |
|---|---|
| Download URL | https://www.projectaria.com/datasets/adt/ ; data format: https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset/data_format ; download: https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset/dataset_download ; object models: https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset/object_models |
| Total size | Official docs: **236 sequences**; **~3.5 TB without MPS**; object model library **~5.5 GB**. |
| Egocentric? | Yes, Aria recordings in instrumented scanned spaces; synthetic twin variants included. |
| Modalities pre-extracted as portable files | RGB/sensor streams in VRS; **Aria 6DoF trajectory** `aria_trajectory.csv`; **object 6DoF pose** `scene_objects.csv`; 2D boxes `2d_bounding_box.csv`; 3D AABB `3d_bounding_box.csv`; **eye gaze** `eyegaze.csv`; optional skeleton `Skeleton_*.json`; instance metadata `instances.json`; MPS eye gaze and SLAM CSV/GZ files; object models as GLB. Depth/segmentation are VRS containers, not simple PNG/NPY in the documented sequence tree. |
| Pre-extracted file formats | CSV, JSON, GLB, VRS, CSV.GZ. Key files: `aria_trajectory.csv`, `scene_objects.csv`, `eyegaze.csv`, `instances.json`, `Skeleton_*.json`, `3d-asset.glb`. |
| SDK required to read? | **No for CSV/JSON/GLB** after download; **partial for download** via Project Aria downloader; **yes for VRS depth/segmentation/video decoding** unless converted externally. |
| License | Aria Digital Twin Dataset License Agreement; gated/restricted research terms. GLB object folders include `CC_BY-SA.txt` according to object-model docs, but sequence data license remains ADT license. |
| Why interesting for "space as index" | ADT is spatially richest among real Aria releases: wearer 6DoF, object 6DoF, gaze, 2D/3D boxes, skeleton, and digital object models share a scene frame. For WorldMM, it enables object-centric memory keys: "where was the mug relative to gaze and wearer pose" using CSV/JSON only, if VRS decoding is avoided. |

Evidence: ADT data-format docs list `aria_trajectory.csv`, `scene_objects.csv`, `eyegaze.csv`, `instances.json`, optional `Skeleton_*.json`, MPS CSV/GZ outputs; scene_objects schema includes translation/quaternion columns for object-to-world pose.

### 3. Aria Pilot Dataset

| Field | Value |
|---|---|
| Download URL | https://www.projectaria.com/datasets/apd ; archived tooling: https://facebookresearch.github.io/Aria_data_tools/docs/use-vrs/ |
| Total size | Not confirmed from accessible official page during this pass. Official page says 159 sequences. |
| Egocentric? | Yes, Project Aria glasses; includes everyday and desktop activities. |
| Modalities pre-extracted as portable files | Official page claims per-frame trajectory for every recording, shared reference-frame alignment for same environments, full calibration, calibrated eye-gaze, IMU/audio/GPS in sensors. Portable release layout was not clearly documented in accessible pages. VRS is central. |
| Pre-extracted file formats | VRS confirmed. VRS tooling can extract JPEG/WAV and `metadata.jsons`; trajectory/gaze file format in APD release unclear from accessible official page. |
| SDK required to read? | **Yes/partial**. VRS sensor data needs Aria tooling. Portable trajectory/gaze existence is likely but not proven with file tree. |
| License | Project Aria Pilot Dataset license; gated Meta access. |
| Why interesting for "space as index" | APD is older and has Aria pose/gaze, but AEA supersedes it for our purpose because AEA clearly documents MPS CSV/GZ outputs and shared-location closed-loop point clouds. Use APD only if already downloaded and if trajectory/gaze files are present. |

Evidence: APD official page states per-frame trajectory, aligned environments, full calibration, and calibrated eye-gaze; archived VRS docs show raw data extraction requires VRS commands and yields `metadata.jsons` for VRS records.

### 4. Aria Synthetic Environments (ASE)

| Field | Value |
|---|---|
| Download URL | https://www.projectaria.com/datasets/ase/ ; data format: https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_synthetic_environments_dataset/ase_data_format ; download: https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_synthetic_environments_dataset/ase_download_dataset |
| Total size | Official docs: **~23 TB**, 100K synthetic apartment sequences split into chunks. |
| Egocentric? | Synthetic egocentric Aria-like camera trajectories. |
| Modalities pre-extracted as portable files | RGB fisheye JPEGs; depth PNGs; instance segmentation PNGs; **ground-truth trajectory** `trajectory.txt`; scene language `ase_scene_language.txt`; semidense MPS point files `semidense_points.csv.gz`, `semidense_observations.csv.gz`; object instance-to-class map JSON. |
| Pre-extracted file formats | JPG, PNG, TXT, JSON, CSV.GZ, ZIP chunks. |
| SDK required to read? | **No for files after download** using stdlib/PIL-equivalent/pandas/numpy; **partial for download** because official downloader script is in `projectaria_tools`. No Aria SDK needed for CSV/TXT/PNG/JPG parsing. |
| License | Aria Synthetic Environments Dataset License Agreement; gated acceptance. |
| Why interesting for "space as index" | ASE is ideal for stress-testing spatial indices because every frame has ground-truth depth/instance/trajectory plus a symbolic scene-language floor plan. It is synthetic and huge, so best used by sampling a small scene range rather than ingesting all 23 TB. |

Evidence: ASE data-format docs show per-scene `rgb/`, `depth/`, `instances/`, `ase_scene_language.txt`, `trajectory.txt`, `semidense_points.csv.gz`, `semidense_observations.csv.gz`, `object_instances_to_classes.json`; download docs state ~23 TB.

### 5. EgoLife

| Field | Value |
|---|---|
| Download URL | Main HF: https://huggingface.co/datasets/lmms-lab/EgoLife ; project: https://egolife-ai.github.io/ ; code: https://github.com/egolife-ai/EgoLife ; auxiliary IMU: https://huggingface.co/datasets/Wangtwohappy/EgoLife_IMU ; auxiliary gaze/tracking: https://huggingface.co/datasets/Wangtwohappy/EgoLife_EyeTracking_EyeGaze |
| Total size | Main HF tree is split into many MP4 chunks; exact total GB not computed. Project page says ~50 hours per participant over six volunteers. Main HF API shows MP4 files under participant/day folders; auxiliary IMU/gaze repos exist separately. |
| Egocentric? | Yes, six participants wearing glasses; synchronized with third-person GoPros according to project page. |
| Modalities pre-extracted as portable files | Main `lmms-lab/EgoLife`: MP4 video chunks plus JSON QA/captions/transcripts. Auxiliary `Wangtwohappy/EgoLife_IMU`: per-chunk **NPZ IMU** files named `*_left.npz`, `*_right.npz`. Auxiliary `Wangtwohappy/EgoLife_EyeTracking_EyeGaze`: **gaze CSV** files and eye-tracking MP4 files. Project page claims video, gaze, IMU, 3D scans, and mmWave, but main HF release does not expose obvious 6DoF pose/body pose/depth/object pose files. |
| Pre-extracted file formats | MP4, JSON, CSV, NPZ. Main HF API for A1/DAY1 showed 828 `.mp4` files; gaze API showed 828 `.csv` files for A1/DAY1; IMU API showed `.npz` files per chunk; eye-tracking API showed MP4. |
| SDK required to read? | **No** for MP4/JSON/CSV/NPZ. No SDK required for auxiliary numeric files. |
| License | Main HF card: MIT. Auxiliary Wangtwohappy repos did not expose clear license in API output; treat as unclear. |
| Why interesting for "space as index" | EgoLife is already in WorldMM’s pipeline and now appears to have separate community/auxiliary gaze CSV and IMU NPZ releases. It lacks published 6DoF trajectories, so spatial indexing must be inferred or fused with place/text/visual grounding unless more official pose/depth/scan data becomes available. |

Evidence: HF API for `lmms-lab/EgoLife` lists participant/day MP4 directories and JSON QA/caption/transcript data; HF API for `Wangtwohappy/EgoLife_IMU` lists per-chunk `.npz`; HF API for `Wangtwohappy/EgoLife_EyeTracking_EyeGaze` lists `EyeGaze/.../*.csv` and `EyeTracking/.../*.mp4`.

### 6. Ego-Exo4D

| Field | Value |
|---|---|
| Download URL | https://docs.ego-exo4d-data.org/getting-started/ ; CLI/download parts: https://docs.ego-exo4d-data.org/download/ ; MPS docs: https://docs.ego-exo4d-data.org/data/mps/ ; EgoPose docs: https://docs.ego-exo4d-data.org/annotations/ego_pose/ |
| Total size | Download page parts: metadata 0.046 GB; annotations 10.533 GB; takes 10553.486 GB; take_trajectory 509.503 GB; take_eye_gaze 3.265 GB; take_point_cloud 6164.615 GB; take_vrs 12301.458 GB; capture_trajectory 851.691 GB; capture_eye_gaze 5.619 GB; capture_point_cloud 4750.039 GB. |
| Egocentric? | Mixed: first-person Aria plus 4-5 exocentric GoPros; dataset is explicitly ego-exo. |
| Modalities pre-extracted as portable files | MP4 frame-aligned videos; **trajectory CSV** (`closed_loop_trajectory.csv` via take/capture trajectory parts); **eye gaze** part; **semidense point cloud** CSV.GZ; metadata JSON (`takes.json`, `captures.json`, etc.); EgoPose 2D/3D hand/body keypoint **JSON**; timesync CSV; features. IMU/audio live in VRS/MP4, not always portable as plain files. |
| Pre-extracted file formats | CSV, CSV.GZ, JSON, MP4, VRS, feature files (format depends feature part). EgoPose annotations are JSON keyed by frame number. |
| SDK required to read? | **No for downloaded CSV/JSON/CSV.GZ**; **partial for download** via Ego4D CLI and AWS credentials after license; **yes for VRS sensor internals**. |
| License | Ego-Exo4D license agreement; gated credentials, research/commercial terms with redistribution restrictions. |
| Why interesting for "space as index" | Ego-Exo4D is large but perfectly aligned with WorldMM’s target: take-trimmed trajectory, gaze, point cloud, metadata, and 3D keypoints are separately downloadable, so we can skip VRS and ingest CSV/JSON. It also supplies exocentric views and tasks, giving strong context for validating spatial retrieval. |

Evidence: download docs list separate `take_trajectory`, `take_eye_gaze`, `take_point_cloud` parts and their sizes; MPS docs say trajectory provides 3D camera positions at 1 kHz and points to `closed_loop_trajectory.csv`; EgoPose docs show 2D/3D keypoints in per-take JSON and camera metadata JSON.

### 7. Ego4D v2

| Field | Value |
|---|---|
| Download URL | https://ego4d-data.org/docs/start-here/ ; CLI: https://ego4d-data.org/docs/CLI/ ; gaze: https://ego4d-data.org/docs/data/gaze/ ; IMU: https://ego4d-data.org/docs/data/imu/ ; metadata: https://ego4d-data.org/docs/data/metadata/ ; features: https://ego4d-data.org/docs/data/features/ |
| Total size | Overall full dataset very large; exact total not pulled here. Docs require license/AWS credentials. |
| Egocentric? | Yes, first-person videos. |
| Modalities pre-extracted as portable files | Metadata `ego4d.json`; annotations JSON; **gaze CSV** for subset of videos; **IMU CSV** per video; pre-extracted feature vectors; no general 6DoF camera trajectory/depth/3D object pose found in v2 docs. |
| Pre-extracted file formats | JSON, CSV, feature vectors, MP4. |
| SDK required to read? | **No for downloaded JSON/CSV/features**; **partial for download** via Ego4D CLI and AWS credentials. No ffmpeg needed for JSON/CSV metadata ingestion. |
| License | Ego4D license agreement; gated AWS credentials, restricted redistribution. |
| Why interesting for "space as index" | Ego4D v2 can enrich WorldMM with large-scale egocentric semantic/time metadata, gaze, and IMU but lacks explicit metric 6DoF/depth in the public machine-readable release. Treat it as temporal/action/narration memory data, not spatial-ground-truth data. |

Evidence: Ego4D gaze docs show flat CSV fields including normalized 2D gaze and optional 3D gaze fields; IMU docs show normalized flat CSV with gyro/accel columns; annotation schemas show JSON files after CLI download.

### 8. HOI4D

| Field | Value |
|---|---|
| Download URL | Project: https://hoi4d.github.io/ ; instructions: https://github.com/leolyliu/HOI4D-Instructions ; README raw: https://raw.githubusercontent.com/leolyliu/HOI4D-Instructions/main/README.md |
| Total size | Not confirmed from official page here. Public page says 2.4M RGB-D egocentric frames, 4000 sequences, 800 object instances, 610 rooms. |
| Egocentric? | Yes, RGB-D egocentric human-object interaction. |
| Modalities pre-extracted as portable files | RGB video `image.mp4`; depth video `depth_video.avi`; **object pose** JSON under `objpose/*.json`; **hand pose** MANO pickle files; action JSON; 3D scene segmentation PCD/log; 2D segmentation masks; CAD OBJ models; camera parameters. |
| Pre-extracted file formats | MP4, AVI, JSON, PICKLE, PCD, OBJ, CSV definitions, masks/images. |
| SDK required to read? | **No for JSON/PCD/OBJ and basic pickle with Python stdlib**; MANO mesh/joint reconstruction requires MANO/manopth if you want vertices/joints, but stored parameters are portable. FFmpeg suggested for RGB/depth frame extraction. |
| License | GitHub instruction repo has LICENSE; dataset terms not fully confirmed here. Treat as research dataset terms from project page. |
| Why interesting for "space as index" | HOI4D gives dense egocentric hand-object-scene interaction with object pose and room point clouds. It is strong for object-centric indexing but uses pickle/MANO and depth videos, so ingestion is less clean than CSV/JSON-only Aria MPS data. |

Evidence: project page says frame-wise 3D hand pose, category-level object pose, reconstructed object meshes and scene point clouds; instruction README lists directory tree with `objpose/*.json`, `action/color.json`, `3Dseg/*.pcd`, and hand pose `.pickle` fields (`poseCoeff`, `beta`, `trans`, `kps2D`).

### 9. HOT3D (Meta)

| Field | Value |
|---|---|
| Download URL | Official docs: https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/hot3d ; toolkit: https://github.com/facebookresearch/hot3d ; full dataset: https://www.projectaria.com/datasets/hot3D/ ; HOT3D-Clips HF: https://huggingface.co/datasets/bop-benchmark/hot3d ; clips README: https://raw.githubusercontent.com/facebookresearch/hot3d/main/hot3d/clips/README.md |
| Total size | Full dataset size not confirmed here; docs say >800 minutes, 33 objects, >1M multi-view frames. HOT3D-Clips: 3832 clips, 150 frames each. HF total size not fetched. |
| Egocentric? | Yes, Project Aria and Quest 3 multi-view egocentric recordings. |
| Modalities pre-extracted as portable files | Full HOT3D: VRS-based format with annotations via toolkit; docs say high-quality 3D pose annotations of hands/objects, 3D object models, 2D boxes, eye-gaze MPS and semidense point-cloud MPS for Aria. **HOT3D-Clips**: tar per clip with per-frame JPEGs, `cameras.json`, `hands.json`, `hand_crops.json`, `objects.json`, `info.json`, `__hand_shapes.json__`, object GLB models. |
| Pre-extracted file formats | Full: VRS-based plus annotation files accessed by toolkit. Clips: TAR/WebDataset, JPG, JSON, GLB. |
| SDK required to read? | **Full HOT3D: yes/partial** (toolkit/downloader and VRS-based format). **HOT3D-Clips: no for JSON/JPG/GLB inside tar**, can read with Python stdlib `tarfile` + `json`; MANO/UmeTrack visualization optional. |
| License | HOT3D dataset license via Project Aria; clips use same license. Toolkit Apache-style code license separate. |
| Why interesting for "space as index" | HOT3D-Clips are excellent for zero-SDK ingestion: per-frame camera pose, object 6DoF pose, hand pose, crops, and images are JSON/JPG in tar files. Full HOT3D is richer but VRS/toolkit-heavy. |

Evidence: HOT3D docs list hand/object 3D pose annotations, 3D object models, eye gaze MPS, semidense point cloud; clips README documents per-frame `<FRAME-ID>.objects.json` with `T_world_from_object`, `<FRAME-ID>.hands.json` with MANO/UmeTrack poses, and `<FRAME-ID>.cameras.json` with `T_world_from_camera`.

### 10. HoloAssist

| Field | Value |
|---|---|
| Download URL | https://holoassist.github.io/ ; dataset instructions mirrored at https://holoassist.github.io/data_links/README.html |
| Total size | Website lists: videos 184.20 GB; compressed videos 144.62 GB; Ahat depth 560.46 GB; eye gaze 2.45 GB; hand pose 219.24 GB; head pose 4.67 GB; IMU 4.63 GB; labels 111 MB; camera calibration 10.07 GB. |
| Egocentric? | Yes, HoloLens 2 / mixed-reality headset task performer; instructor collaboration. |
| Modalities pre-extracted as portable files | RGB MP4; depth PNG folders + sync text; **head pose** text; **hand pose** left/right sync text; **eye gaze** sync text; IMU accelerometer/gyro/magnetometer sync text; RGB/depth calibration text; labels/annotations JSON. |
| Pre-extracted file formats | MP4, PNG, TXT, JSON. Example tree includes `AhatDepth/*.png`, `AhatDepth_synced.txt`, `Pose_sync.txt`, `Eyes/Eyes_sync.txt`, `Hands/Left_sync.txt`, `Hands/Right_sync.txt`, `Head/Head_sync.txt`, `IMU/*_sync.txt`, `Video/Pose_sync.txt`. |
| SDK required to read? | **No** for exported dataset text/PNG/MP4/JSON. PSI/HoloLens tooling only needed to understand raw export conventions, not to parse text files. |
| License | CDLAv2 permissive license per website. |
| Why interesting for "space as index" | HoloAssist is high-friction in size but low-friction in parsing: depth, gaze, hand pose, head pose, IMU, and calibration are already exported as text/PNG files. It is one of the best non-Aria sources for HoloLens-like spatial memory data without SDKs. |

Evidence: website download section lists sizes for depth/gaze/hand/head/IMU/calibration; README tree shows `_sync.txt` files for depth pose, eyes, hands, head, IMU, and video pose.

### 11. EPIC-KITCHENS-100

| Field | Value |
|---|---|
| Download URL | https://epic-kitchens.github.io/2025 ; annotations: https://github.com/epic-kitchens/epic-kitchens-100-annotations ; raw README: https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/master/README.md |
| Total size | Public page: 100 hours, 20M frames; exact GB depends videos/frames/features chosen. |
| Egocentric? | Yes, head-mounted kitchen videos. |
| Modalities pre-extracted as portable files | Narrations/action segments/object/action labels and automatic annotation assets; pre-extracted RGB/flow frames exist. No native 6DoF pose, depth, eye gaze, hand pose, IMU, or 3D object pose found in EK-100 docs/repo. |
| Pre-extracted file formats | CSV/PKL/JSON-style annotations in GitHub release assets; MP4 videos; JPG frames/flow if downloaded. Exact annotation file set from repo/release. |
| SDK required to read? | **No** for annotations; videos/frames standard. |
| License | Publicly available for research purposes; dataset terms on EPIC site. |
| Why interesting for "space as index" | Useful as semantic/narration benchmark, not metric spatial memory. It can test whether WorldMM’s text triples help in egocentric kitchens, but cannot validate pose/depth/gaze-based spatial indexing without external inference. |

Evidence: EPIC page says 100 hours, 20M frames, 90K action segments, 20K narrations; annotation repo README focuses on annotations/pre-extracted RGB/flow erratum, not spatial 3D signals.

### 12. ARCTIC

| Field | Value |
|---|---|
| Download URL | https://arctic.is.tue.mpg.de/ ; repo: https://github.com/zc-alexfan/arctic ; data docs: https://raw.githubusercontent.com/zc-alexfan/arctic/master/docs/data/README.md and https://raw.githubusercontent.com/zc-alexfan/arctic/master/docs/data/data_doc.md |
| Total size | Data docs list: images 649 GB; cropped images 116 GB; raw GT sequences 215 MB; splits 18 GB; features 14 GB; meta 91 MB; models 6 GB. |
| Egocentric? | Mixed: 8 third-person views + 1 egocentric view in mixed-reality setting. |
| Modalities pre-extracted as portable files | **Egocentric camera trajectory** `*.egocam.dist.npy`; MANO parameters `*.mano.npy`; object parameters/poses `*.object.npy`; SMPL-X `*.smplx.npy`; camera/object/subject metadata JSON; object template meshes OBJ; images optional. |
| Pre-extracted file formats | NPY, JSON, OBJ, JPG/PNG images, model weights. |
| SDK required to read? | **No for raw GT NPY/JSON/OBJ with numpy/json**. Visualization/reconstruction environment needs many packages, but raw numeric pose files are directly loadable. Account registration required. |
| License | ARCTIC account/license; MANO/SMPL-X separate licenses for models. Research use. |
| Why interesting for "space as index" | ARCTIC is compact if you ingest only `raw_seqs` + `meta`: hand/object/body/camera trajectories in NPY/JSON without decoding videos. It is less daily-life and more staged manipulation, but ideal for testing hand-object spatial memory representations. |

Evidence: ARCTIC docs say `raw_seqs` contains MANO, SMPLX, egocentric camera trajectory, object poses; folder tree lists `*.egocam.dist.npy`, `*.mano.npy`, `*.object.npy`, `*.smplx.npy`, and `meta/misc.json`/object templates.

### 13. ScanNet++ / Replica / Matterport HM3D

| Field | Value |
|---|---|
| Download URL | ScanNet++: https://kaldir.vc.in.tum.de/scannetpp/ and https://github.com/scannetpp/scannetpp ; Replica: https://github.com/facebookresearch/Replica-Dataset ; HM3D: https://github.com/matterport/habitat-matterport-3dresearch and https://aihabitat.org/datasets/hm3d/ |
| Total size | ScanNet++: not confirmed here. Replica: 18 scenes, size not confirmed here. HM3D v0.2 README lists train GLB 32 GB, train Habitat 27 GB, train semantic annotations 8.1 GB, val GLB 4 GB, val Habitat 3.3 GB, val semantic 2.0 GB, minival/example sizes as linked. |
| Egocentric? | No for raw dataset purpose; static room scans/posed images. Useful as spatial scaffolds. |
| Modalities pre-extracted as portable files | ScanNet++: high-fidelity 3D indoor scene assets, iPhone/DSLR frames/poses/depth/semantics through toolkit/docs. Replica: mesh PLY, semantic JSON/BIN, preseg JSON/BIN, Habitat assets/navmesh, HDR textures. HM3D: OBJ/GLB meshes, JPG textures, MTL, semantic annotations/configs. |
| Pre-extracted file formats | PLY, JSON, BIN, OBJ, GLB, JPG, MTL, navmesh, PNG/JPG/depth depending dataset. |
| SDK required to read? | **No for static mesh/JSON files**; Habitat recommended/needed for simulation rendering/navigation. ScanNet++ preprocessing may require scripts, but downloaded mesh/pose files are standard. |
| License | ScanNet++ terms via TUM site; Replica research license; HM3D academic non-commercial Matterport terms. |
| Why interesting for "space as index" | These are not egocentric memory streams, but they provide stable semantic 3D maps for retrieval experiments: map WorldMM locations/objects into static meshes and test place indexing. HM3D/Replica are especially useful if paired with Habitat trajectories. |

Evidence: Replica README lists `mesh.ply`, `semantic.json/bin`, Habitat `mesh_semantic.ply`, `info_semantic.json`, navmesh, textures; HM3D README lists OBJ/GLB, JPG textures, MTL and direct tar URLs/sizes.

### 14. Habitat-Sim / Habitat-Web

| Field | Value |
|---|---|
| Download URL | Habitat-Sim docs: https://aihabitat.org/docs/habitat-sim/ ; Habitat Lab: https://aihabitat.org/docs/habitat-lab/ ; HM3D dataset: https://aihabitat.org/datasets/hm3d/ |
| Total size | Simulator code small; dataset size depends chosen asset pack (HM3D/Replica/Matterport/Gibson). HM3D sizes above. |
| Egocentric? | Simulated egocentric agent trajectories, not recorded human egocentric data. |
| Modalities pre-extracted as portable files | Habitat can generate RGB/depth/semantic/pose/agent trajectory with full ground truth, but those are not generally shipped as one fixed dataset unless using a benchmark episode dataset. Static scene files are GLB/OBJ/PLY/JSON; episode datasets are JSON/JSON.GZ in Habitat-Lab tasks. |
| Pre-extracted file formats | GLB/OBJ/PLY/JSON/JSON.GZ/PNG/NPY depending generated task/logging. |
| SDK required to read? | **Yes for simulation/rendering**. **No for pre-generated episode JSON or mesh files**. For WorldMM's "no SDK" constraint, Habitat is only zero-SDK if someone else pre-renders/export trajectories. |
| License | Habitat code under open-source license; scene dataset license depends HM3D/Replica/MP3D/Gibson. |
| Why interesting for "space as index" | Habitat is a ground-truth generator, not a ready human-egocentric corpus. Use it to synthesize controlled spatial-memory ablations, but it violates the "no SDK / no sim run" ingestion goal unless we consume only already-generated JSON trajectories. |

Evidence: Habitat-Sim docs describe configurable agents, multiple sensors, and built-in support for Matterport3D, Gibson, Replica and other datasets; HM3D docs provide static GLB/OBJ assets.

## Additional candidate found during survey

### HOT3D-Clips as separate ingestion target

Although HOT3D is in the requested list, **HOT3D-Clips** deserves separate treatment: it is hosted on HF as WebDataset/tar clips with JSON per frame. It maximizes zero-SDK ingestion more than full HOT3D. Direct URL: https://huggingface.co/datasets/bop-benchmark/hot3d . Use `tarfile`, `json`, `numpy`, `pandas` only; no VRS path needed.

### EgoLife auxiliary IMU/gaze releases

Separate HF repos expose portable signals not obvious in the main `lmms-lab/EgoLife` dataset:

- https://huggingface.co/datasets/Wangtwohappy/EgoLife_IMU — per-chunk `.npz` IMU files.
- https://huggingface.co/datasets/Wangtwohappy/EgoLife_EyeTracking_EyeGaze — gaze `.csv` and eye-tracking `.mp4` files.

License/provenance should be checked before building on them, but ingestion friction is low.

## Shortlist: best 3 for WorldMM now

1. **Aria Everyday Activities (AEA)** — Best real daily-life Aria dataset for no-SLAM spatial indexing. It clearly ships closed-loop/open-loop trajectory CSV, eye-gaze CSV, semidense point-cloud CSV.GZ, speech CSV, and metadata JSON; WorldMM can ingest the MPS folder while ignoring VRS.

2. **Ego-Exo4D** — Best large-scale procedural/activity corpus with downloadable trajectory/gaze/point-cloud parts. The CLI/access step is gated, but once downloaded the core spatial signals are CSV/JSON/GZ and include take-level trimming aligned to videos and EgoPose annotations.

3. **HOT3D-Clips** — Best hand-object spatial benchmark with minimal parsing friction. Per-frame camera, hand, and object poses are JSON inside tar files, and object models are GLB; this is ideal for testing object-centric spatial memory keys without installing Project Aria SDK or decoding VRS.

Runner-up: **HoloAssist** if HoloLens-style depth + head/hand/gaze/IMU text streams matter more than Aria compatibility. **ARCTIC** if compact NPY hand/object/camera GT is preferred over daily-life realism.
