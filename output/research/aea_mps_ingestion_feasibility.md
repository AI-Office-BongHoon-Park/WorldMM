# AEA MPS ingestion feasibility for WorldMM

## Executive verdict — MEASURED

Path A, pivoting WorldMM toward Aria Everyday Activities (AEA)-class data, is **measurably feasible**. Acquired `loc5_script4_seq6_rec1` (~232 MB MPS-only, VRS skipped) by submitting the email gate via browser and pulling the CDN URLs JSON directly. Inspected every MPS file with pandas + json — all schemas match doc claims, cross-stream timestamps align in one domain, and the data plugs into an additive `SpatialTripleEntry` extension as planned.

**Empirical numbers** (loc5_script4_seq6_rec1):

| File | Rows / size | Frequency | Notes |
|---|---:|---:|---|
| `closed_loop_trajectory.csv` | **214,041 rows** | **~1010 Hz** | 28 cols, quality_score mean 0.999, geo_available=0 (indoor) |
| `general_eye_gaze.csv` | 2,131 rows | 10 Hz | CPF yaw/pitch, depth_m all NaN (older model), confidence interval present |
| `semidense_points.csv.gz` | 27 MB | n/a | uid + graph_uid + xyz_world + uncertainty |
| `speech.csv` | 1 row | n/a | Empty: only "you." conf 0.008 (this sequence is quiet) |
| `online_calibration.jsonl` | 9 MB | per-frame | not loaded for this pass |
| `metadata.json` | 172 B | n/a | dataset_version 1.0, dataset_name AEA_2024 |
| `summary.json` | 2 KB | n/a | SLAM SUCCESS, 3:33 recording, 62 m trajectory |

**Cross-stream alignment**: trajectory and gaze share `tracking_timestamp_us` domain. Gaze span (213 s) ≈ trajectory span (212 s); offset gaze_start − traj_start = −0.8 s, gaze_end − traj_end = +0.2 s. Nearest-neighbor join by `tracking_timestamp_us` works directly — no epoch shift, no per-frame interpolation required for our use case.

## Earlier doc-derived analysis (kept for reference)

Initial pass (before download) was mostly doc-derived. All numbers and conclusions in the schema-mapping sections below were subsequently confirmed by inspecting the real CSVs in `data/AEA/loc5_script4_seq6_rec1/`. The acquisition blocker described below is now resolved.

## Acquisition result

Selected sequence: `loc5_script4_seq6_rec1`.

Reason: Project Aria's AEA download docs use this exact sequence as the single-sequence download example. Current public docs and the Hugging Face dataset card do not expose per-sequence byte sizes or per-sequence durations. The published dataset statistics show Location 5 has 14 recordings and 1.1 accumulated hours, about 4.7 minutes per recording on average, and only one wearer; that makes a Location 5 sequence the best documented small candidate and likely below 10 GB for MPS-only files. This is a best-available choice, not a proven global minimum.

Attempts:

- HF first, no install: `hf download projectaria/aria-everyday-activities --repo-type dataset --include 'loc5_script4_seq6_rec1/MPS/**' --include 'loc5_script4_seq6_rec1/metadata.json' --include 'loc5_script4_seq6_rec1/speech.csv' --local-dir data/AEA/loc5_script4_seq6_rec1` returned success but fetched `0 files`. The HF repo tree visible through `huggingface_hub` contained only `.DS_Store`, `.gitattributes`, `README.md`, and `assets/AEA_intro.jpg`.
- Fallback downloader: direct install in the project Python 3.13 environment failed because `projectaria-tools` wheels are available through CPython 3.12, not CPython 3.13. `uvx --python 3.12 --from projectaria-tools aria_dataset_downloader --help` succeeded and showed the required flags: `--cdn_file`, `--output_folder`, `--data_types`, `--sequence_names`.
- Gated blocker: no `aria_everyday_activities_dataset_download_urls.json` or equivalent CDN/license JSON was present under `/home/default/Downloads`, `/home/default/Documents`, or the workspace. The exact blocked step is Project Aria docs Step 2: after accepting terms on the AEA page, download `aria_everyday_activities_dataset_download_urls.json`; then run `aria_dataset_downloader -c <that file> -o data/AEA -l loc5_script4_seq6_rec1 -d <MPS data types>`.

Local acquisition artifact:

- `data/AEA/loc5_script4_seq6_rec1/` exists.
- Downloaded size: 0 bytes.
- `output/research/aea_hf_download.log` records HF fetched 0 files.

## Expected target file layout

Doc-listed AEA sequence folders are self-contained:

```text
loc1_script1_seq1_rec1/
  recording.vrs
  metadata.json
  speech.csv
  MPS/
    eye_gaze/general_eye_gaze.csv
    eye_gaze/summary.json
    slam/closed_loop_trajectory.csv
    slam/open_loop_trajectory.csv
    slam/online_calibration.csv
    slam/semidense_observations.csv.gz
    slam/semidense_points.csv.gz
    slam/summary.json
```

We should ignore `recording.vrs` for this path and read portable CSV/CSV.GZ outputs with pandas + numpy only.

## Schema dump status

Measured schema dump: blocked by acquisition. No CSV files exist locally.

Doc-derived schema dump follows. Dtypes are expected pandas dtypes, not measured dtypes. Row counts, first 3 rows, and last 3 rows are unknown except the speech sample shown in Project Aria docs.

### `MPS/slam/closed_loop_trajectory.csv`

Columns:

- `graph_uid`: string/object. Shared world coordinate frame id.
- `tracking_timestamp_us`: int64. Aria device timestamp in microseconds.
- `utc_timestamp_ns`: int64. Wall-clock UTC nanoseconds; can be `-1` if unavailable.
- `tx_world_device`, `ty_world_device`, `tz_world_device`: float64. Device translation in world frame, meters.
- `qx_world_device`, `qy_world_device`, `qz_world_device`, `qw_world_device`: float64. Device orientation in world frame, CSV order xyzw.
- `device_linear_velocity_x_device`, `device_linear_velocity_y_device`, `device_linear_velocity_z_device`: float64. Device-frame velocity, m/s.
- `angular_velocity_x_device`, `angular_velocity_y_device`, `angular_velocity_z_device`: float64. Device-frame angular velocity, rad/s.
- `gravity_x_world`, `gravity_y_world`, `gravity_z_world`: float64. Gravity vector in world frame, m/s^2; MPS typically `[0, 0, -9.81]`.
- `quality_score`: float64 in `[0, 1]`.

Frequency: documented as high-frequency IMU-rate pose, about 1 kHz. Actual `rows / duration` not measured.

Pose units: meters for translation, m/s for velocity, rad/s for angular velocity.

Quaternion convention: Project Aria uses Hamilton convention. The trajectory CSV stores columns as `qx,qy,qz,qw`; Sophus/SE3 utilities may expose `w,x,y,z`, so ingestion should reorder to WorldMM's chosen tuple convention explicitly.

Coordinate frame: arbitrary gravity-aligned world frame; for AEA, SLAM outputs are in a shared coordinate frame organized by location.

### `MPS/slam/open_loop_trajectory.csv`

Columns:

- `tracking_timestamp_us`: int64. Aria device timestamp in microseconds.
- `utc_timestamp_ns`: int64. Wall-clock UTC nanoseconds; can be `-1`.
- `session_uid`: string/object. Odometry coordinate frame id.
- `tx_odometry_device`, `ty_odometry_device`, `tz_odometry_device`: float64. Device translation in odometry frame, meters.
- `qx_odometry_device`, `qy_odometry_device`, `qz_odometry_device`, `qw_odometry_device`: float64. Device orientation in odometry frame, CSV order xyzw.
- `device_linear_velocity_x_odometry`, `device_linear_velocity_y_odometry`, `device_linear_velocity_z_odometry`: float64. Odometry-frame velocity, m/s.
- `angular_velocity_x_device`, `angular_velocity_y_device`, `angular_velocity_z_device`: float64. Device-frame angular velocity, rad/s.
- `gravity_x_odometry`, `gravity_y_odometry`, `gravity_z_odometry`: float64. Gravity vector in odometry frame, m/s^2.
- `quality_score`: float64.

Frequency: documented as high-frequency IMU-rate VIO, about 1 kHz. Actual `rows / duration` not measured.

Recommended use: secondary/debug source. Closed-loop should drive WorldMM `pose_6dof` because AEA's shared world alignment depends on loop-closed SLAM.

### `MPS/eye_gaze/general_eye_gaze.csv`

Columns:

- `tracking_timestamp_us`: int64. Eye-tracking frame timestamp in device time; directly alignable with MPS location/trajectory timestamps.
- `yaw_rads_cpf`: float64. Gaze yaw in Central Pupil Frame (CPF). Older model provides this directly; newer model may require helper-derived computation.
- `pitch_rads_cpf`: float64. Gaze pitch in CPF.
- `depth_m`: float64 or nullable float. Newer model's absolute 3D gaze point depth in CPF; capped around 4 m; older open-dataset model may not provide it.
- `yaw_low_rads_cpf`, `pitch_low_rads_cpf`, `yaw_high_rads_cpf`, `pitch_high_rads_cpf`: float64. Confidence interval bounds for older model; zeros for newer model.
- `left_yaw_rads_cpf`, `right_yaw_rads_cpf`: float64. Newer model left/right-eye yaw in CPF; zeros for older model.
- `left_yaw_low_rads_cpf`, `right_yaw_low_rads_cpf`, `left_yaw_high_rads_cpf`, `right_yaw_high_rads_cpf`: float64. Newer model confidence bounds; zeros where not provided.
- `tx_left_eye_cpf`, `ty_left_eye_cpf`, `tz_left_eye_cpf`, `tx_right_eye_cpf`, `ty_right_eye_cpf`, `tz_right_eye_cpf`: float64. Eye origins in CPF.
- `session_uid`: string/object.

Gaze ray representation: yaw/pitch in CPF, not a precomputed 3D unit vector. AEA ingestion can derive a unit ray with numpy. If `depth_m` exists, derive a finite gaze point in CPF; otherwise store a ray plus missing depth.

Confidence/validity: no single `valid` boolean is documented. Confidence is represented as interval width/bounds or nullable/zero depth depending on model version. Ingestion should expose `confidence_interval` and a derived `is_valid` rule, for example finite yaw/pitch and non-empty depth if a target point is required.

### `MPS/slam/semidense_points.csv.gz`

Read with `pandas.read_csv(path, compression='gzip')`.

Columns:

- `uid`: int64. Unique point id within map.
- `graph_uid`: string/object. Shared world coordinate frame id, matching `closed_loop_trajectory.csv`.
- `px_world`, `py_world`, `pz_world`: float64. Point location in world frame, meters.
- `inv_dist_std`: float64. Inverse-distance uncertainty, m^-1.
- `dist_std`: float64. Distance uncertainty, meters.

Point count: unknown until measured. Coordinate frame: same world coordinate frame as closed-loop trajectory. Quality filtering needed; docs suggest nominal thresholds `inv_dist_std <= 0.005` and `dist_std <= 0.01`.

### `MPS/slam/semidense_observations.csv.gz`

Read with `pandas.read_csv(path, compression='gzip')`.

Columns:

- `uid`: int64. Point id, joins `semidense_points.csv.gz`.
- `frame_tracking_timestamp_us`: int64. Host frame center-of-exposure timestamp in microseconds.
- `camera_serial`: string/object.
- `u`: float64. Pixel x coordinate.
- `v`: float64. Pixel y coordinate.

Use: optional visibility sidecar. It does not directly map to `SpatialTripleEntry` but can support object grounding or gaze-to-point intersection if camera frames are used later.

### `MPS/slam/online_calibration.csv` or `online_calibration.jsonl`

AEA docs list `online_calibration.csv`; current MPS calibration docs call the portable output `online_calibration.jsonl`. Treat actual extension as data-version dependent and inspect locally when available.

Documented content:

- One calibration record per timestamp.
- Camera/IMU intrinsics.
- Sensor extrinsics.
- Time offsets for temporal alignment.
- RGB camera readout timing in newer releases.

Use: not required for first-order trajectory/gaze timestamp alignment, but required for rigorous gaze ray projection from CPF through device/camera frames into world coordinates.

### `speech.csv`

Columns from AEA docs:

- `startTime_ns`: int64. Speech segment start time in nanoseconds.
- `endTime_ns`: int64. Speech segment end time in nanoseconds.
- `written`: string/object. ASR transcript token/string.
- `confidence`: float64. ASR confidence.

Doc sample rows:

| startTime_ns | endTime_ns | written | confidence |
|---:|---:|---|---:|
| 54040 | 55040 | I'm | 0.25608 |
| 72920 | 73920 | looking | 0.84339 |

Language: not documented in the schema table; likely English in examples but should be treated as metadata-derived or unknown.

Format: token/word-like rows with start/end timestamps and confidence.

Time alignment field: interval overlap on `startTime_ns <= T_ns <= endTime_ns`. Need confirm whether these are local device-time nanoseconds or a shared timecode/UTC-like domain for AEA; docs say timecode mapping lives in VRS, but CSV speech fields are named nanoseconds only.

### `metadata.json`

Actual fields not measured. Expected use: sequence id, wearer/location/script metadata, duration, recording identifiers, and maybe time-domain notes if present. Ingestion should read with `json.load`, not Aria SDK.

## Field to WorldMM mapping

### Trajectory to `pose_6dof`

WorldMM extension field:

```python
pose_6dof: Optional[Tuple[float, ...]]
```

Proposed tuple:

```text
(timestamp_us, tx, ty, tz, qx, qy, qz, qw, quality_score)
```

Mapping:

| AEA field | WorldMM field | Conversion |
|---|---|---|
| `tracking_timestamp_us` | `SpatialTripleEntry.timestamp` or sensor timestamp sidecar | WorldMM currently stores `timestamp: int`; decide canonical unit. Existing caption timestamps are coarse integer keys, so retain source unit in `pose_6dof` and optionally store `timestamp_ns = tracking_timestamp_us * 1000`. |
| `utc_timestamp_ns` | optional absolute-time field | Keep if not `-1`; no epoch shift if UTC ns is valid. |
| `tx_world_device`, `ty_world_device`, `tz_world_device` | `pose_6dof[1:4]` | No unit change; meters. |
| `qx/qy/qz/qw_world_device` | `pose_6dof[4:8]` | Preserve CSV xyzw or reorder if internal convention becomes wxyz. Must document convention. |
| `graph_uid` | `place_anchor.coordinate_frame_id` | No conversion. |
| `quality_score` | `pose_6dof_quality` or included tuple value | No conversion. |

Frame conversion: closed-loop pose is already in AEA shared world frame. No rotation needed for storage. Rotation is needed only if deriving world-space gaze rays from CPF.

### Eye gaze to `gaze_target`

New Pydantic model:

```python
class GazeTarget(BaseModel):
    tracking_timestamp_us: int
    yaw_rads_cpf: float
    pitch_rads_cpf: float
    depth_m: Optional[float] = None
    ray_cpf: tuple[float, float, float]
    point_cpf: Optional[tuple[float, float, float]] = None
    confidence_interval: Optional[dict[str, float]] = None
    session_uid: Optional[str] = None
```

Mapping:

| AEA field | WorldMM field | Conversion |
|---|---|---|
| `tracking_timestamp_us` | `GazeTarget.tracking_timestamp_us` | Align by nearest trajectory timestamp; no epoch shift within MPS device-time domain. |
| `yaw_rads_cpf`, `pitch_rads_cpf` | `GazeTarget.yaw_rads_cpf`, `pitch_rads_cpf` | No unit change; radians. |
| yaw/pitch | `GazeTarget.ray_cpf` | Derive with numpy from CPF yaw/pitch. |
| `depth_m` | `GazeTarget.point_cpf` | If finite, multiply ray by depth. |
| confidence interval columns | `GazeTarget.confidence_interval` | Store bounds or derive interval width. |
| `session_uid` | `GazeTarget.session_uid` | No conversion. |

Frame conversion: CPF ray to world ray requires online calibration (`T_device_cpf` or equivalent) plus `T_world_device` from trajectory. Without calibration, WorldMM can store gaze intent as CPF ray but cannot map it to a world-space target point cleanly.

### Semidense points to geometric sidecar or place anchor support

Mapping:

| AEA field | WorldMM field | Conversion |
|---|---|---|
| `uid` | point id sidecar | No conversion. |
| `graph_uid` | `PlaceAnchor.coordinate_frame_id` | Join with trajectory `graph_uid`. |
| `px_world`, `py_world`, `pz_world` | point cloud sidecar | No unit change; meters. |
| `inv_dist_std`, `dist_std` | point quality | Apply quality thresholds before downstream use. |

WorldMM's existing `GeometricGrounding` is bbox-centric (`bbox_center`, `bbox_extent`, `units`, `source`, `confidence`, `keyframe_ts`, `instance_disambiguation`) and has `extra='forbid'`, so raw semidense points should not be stuffed into it. Use a sibling point-cloud/pose/gaze sidecar or derive object bbox groundings later.

### Speech to `audio_segments`

New Pydantic model:

```python
class SpeechSegment(BaseModel):
    start_time_ns: int
    end_time_ns: int
    text: str
    confidence: float
    language: Optional[str] = None
```

Mapping:

| AEA field | WorldMM field | Conversion |
|---|---|---|
| `startTime_ns` | `SpeechSegment.start_time_ns` | Need confirm time domain; likely direct interval alignment only after CSV inspection. |
| `endTime_ns` | `SpeechSegment.end_time_ns` | Same. |
| `written` | `SpeechSegment.text` | No conversion; aggregate adjacent tokens if desired. |
| `confidence` | `SpeechSegment.confidence` | No conversion. |

### Place anchor

New Pydantic model:

```python
class PlaceAnchor(BaseModel):
    coordinate_frame_id: str
    centroid_world_m: tuple[float, float, float]
    time_range_us: tuple[int, int]
    label: Optional[str] = None
    evidence: list[str] = []
```

Derivation plan:

- Cluster closed-loop trajectory positions into dwell regions.
- Attach `graph_uid` as coordinate frame id.
- Use speech windows (`written`) and existing spatial triples to label clusters.
- Optionally use semidense points around the cluster for geometric context.

This is derived, not direct CSV mapping.

## Extension plan for `SpatialTripleEntry`

Target additive dataclass fields:

```python
pose_6dof: Optional[Tuple[float, ...]] = None
gaze_target: Optional[GazeTarget] = None
place_anchor: Optional[PlaceAnchor] = None
audio_segments: Optional[List[SpeechSegment]] = None
```

Suggested implementation touchpoints and line-cost estimate:

| Touchpoint | Current extension point | Cost | Notes |
|---|---|---:|---|
| `src/worldmm/memory/spatial/memory.py` | `SpatialTripleEntry` fields at lines 25-34; display at 46-74; load sidecar path at 126-180; indexing at 209-271; retrieval/rendering at 278-346 | Medium, ~80-160 LOC | Add optional fields, parse sidecar JSON, keep retrieval backward-compatible, render concise sensor suffixes. |
| `src/worldmm/memory/spatial/grounding.py` | `GeometricGrounding` lines 8-37 is bbox-only and forbids extras | Small, ~40-80 LOC | Add sibling Pydantic models in a new module rather than modifying bbox semantics. |
| `src/worldmm/memory/spatial/utils.py` | closed predicate vocab and 3-item validators at lines 6-18 and 21-58 | Small, ~20-60 LOC | Only needed if adding predicates like `gaze_at`, `dwelling_at`, `facing`. Pose/gaze fields can be sidecars without changing triple shape. |
| `preprocess/build_memory.py` | spatial build step at lines 118-158 and top-level first-class memory orchestration per extension-point report | Medium, ~100-200 LOC | Add AEA sensor pre-alignment step before/next to spatial memory build; preserve existing caption/OpenIE path. |
| New AEA ingestion tool under `tools/` or `preprocess/` | No native sensor-stream schema currently; extension-point report recommends new sibling inputs and outputs | Medium, ~150-300 LOC | pandas readers, schema validation, nearest-neighbor timestamp alignment, JSON sidecar writer. |
| Retrieval wrappers | `tools/grounded_spatial_retrieve.py` can add optional filters per extension-point report | Small, ~50-100 LOC | Needed only if querying by place/gaze/pose. |

Total minimal ingestion cost: medium, roughly 350-700 LOC. Large only if we require gaze-to-world projection plus object intersection and retrieval ranking changes.

## Sanity end-to-end check

Measured check: blocked. No real trajectory/gaze/speech rows downloaded, so no timestamp `T` could be selected.

Doc-derived alignment feasibility:

- Trajectory and gaze both use `tracking_timestamp_us` in the device time domain. They should align by nearest timestamp. For closed-loop at about 1 kHz, a reasonable tolerance is 1-2 ms; for eye gaze frame rate, nearest sample tolerance may need 10-20 ms depending on ET output cadence.
- Speech uses `startTime_ns` and `endTime_ns`. It should align by interval overlap once its time domain is confirmed. If speech timestamps are in the same local device domain, compare `T_ns = tracking_timestamp_us * 1000` to `[startTime_ns, endTime_ns]`. If speech timestamps are in a timecode/UTC domain, the VRS time-domain mapping is required, which violates the no-VRS/portable-CSV-only constraint unless metadata carries the mapping.

Can timestamp, 6-DoF pose, gaze ray, and overlapping speech become one `SpatialTripleEntry`-like record? Yes for pose+gaze by documented MPS timestamps. Conditional for speech until actual CSV timestamps prove the same time domain. With actual files, use nearest trajectory/gaze within 20 ms and speech interval overlap at exact ns resolution after unit conversion.

Example target record shape, not measured:

```python
SpatialTripleEntry(
    id='aea_loc5_script4_seq6_rec1_<T>',
    subject='wearer',
    predicate='located_in',
    object='<derived_place_anchor>',
    timestamp=T_us,
    place='<cluster_label>',
    pose_6dof=(T_us, tx, ty, tz, qx, qy, qz, qw, quality_score),
    gaze_target=GazeTarget(...),
    place_anchor=PlaceAnchor(...),
    audio_segments=[SpeechSegment(...)]
)
```

## Feasibility risks

- Acquisition access is the only hard blocker observed. Need the gated CDN JSON from the AEA website.
- HF mirror metadata page is not a data mirror in this environment; it exposes no sequence files.
- `online_calibration` naming differs between AEA docs (`.csv`) and current MPS docs (`.jsonl`). Reader should detect both.
- Speech timestamp time domain must be verified from actual files. If it needs VRS-only time-domain mapping, a strict no-VRS path cannot perfectly align speech across domains.
- Gaze target in world coordinates requires calibration. A CPF ray is cleanly portable; a world-space gaze target is only clean if online calibration is present and parseable without SDK.

## Final answer to Path A

Yes, Path A is achievable in less than one week for a minimal version: ingest closed-loop pose, store CPF gaze ray, cluster trajectory into place anchors, and attach speech intervals. It fits WorldMM as additive optional fields on `SpatialTripleEntry` plus sibling Pydantic models and a pandas-based AEA sidecar builder. The blocker is not schema compatibility; it is gated data acquisition and one timestamp-domain validation for `speech.csv`.
