# Long-duration egocentric datasets for WorldMM-style memory

Scope: duration/longitudinal lens, not room-count lens. Current date basis: 2026-05-22 KST. “Portable spatial signals” means files one can pull and parse without a closed viewer: CSV/JSON/PNG/MP4/VRS with documented access. Unknown fields are marked `not reported`, not inferred.

## Executive takeaways

| Dataset | Session length distribution | Per-subject days | Portable spatial signals | Size / access | Verdict for WorldMM |
|---|---:|---:|---|---|---|
| **EgoLife** | ~30s clips grouped into day records; HF listing gives 42 subject-days, A1_JAKE DAY1 = 828 chunks ≈ 6.9h assuming 30s/chunk | **7 days × 6 people** | RGB/audio video; project claims Aria + 3rd-person + mmWave, but public HF view exposes mostly MP4/video rows | HF public, MIT; 32k rows; sampled A1_DAY1 estimate ≈12 GiB | Baseline, best longitudinal fit |
| **Ego4D** | Paper reports **1–10h** worn-camera sessions | Mostly broad population, not multi-day-per-subject memory | Audio subset, 3D meshes subset, gaze subset, stereo subset, IMU CSV subset, metadata with physical settings | Gated license; >3,700h | Best non-EgoLife long-session source, but spatial signals sparse/heterogeneous |
| **EPIC-KITCHENS-100** | 700 videos; computed from official CSV: min 0.003h, median 0.085h, max 1.03h; total 100.03h | multi-day kitchen captures, but released as many clips per participant | Audio/video, frames/flow/boxes/action segments; **no pose/depth/gaze/place** | Research download; 100h | Good longitudinal cooking memory; weak spatial axes |
| **Charades-Ego** | 68.8h total paired ego/exo; clips, not day sessions | no | video + action/text labels; no portable pose/depth/gaze/place/audio emphasis | AI2 data/GitHub baselines | Not useful for day-long memory |
| **SenseCam / personal-camera lifelogs** | Often day/multi-day image streams in literature | yes in some studies | mostly low-rate images + time/GPS/sensor tags; no modern pose/depth/gaze | fragmented, often not open | Historically relevant; weak pull target |
| **WALT / When-Where-What-Why** | no robust public egocentric dataset evidence found | unknown | unknown | unknown | Do not target until exact source identified |
| **NarrationBench / EgoSchema** | EgoSchema = 3-min clips, 250h total derived from Ego4D | no | inherits Ego4D video ids; no added spatial metadata | GitHub/Kaggle/Ego4D license | Benchmark only, not data source |
| **Project Aria day-in-life / Gen2 Pilot** | Gen2 Pilot: 12 sequences, not day-long | no day-long public corpus found | excellent portable MPS: VIO/SLAM trajectories, gaze/hand, depth, ASR, heart-rate, 2D/3D boxes | public docs/repo; dataset explorer | Great spatial schema; too short/not longitudinal |
| **Aria Replay** | no public full-day recordings found | unknown | Aria tools support replay/export; dataset evidence missing | tools open, data unclear | Not a pull target |
| **OpenEQA / Vista** | episode histories in HM3D/ScanNet, not egocentric day capture | no | RGB; optional depth + camera pose via Habitat/ScanNet extraction | OpenEQA repo; RGB HM3D 12GB, ScanNet 62GB; RGB-D+pose 16/70GB | QA/spatial benchmark, not wearable memory |
| **HoloAssist / HoloAssist-Long** | 169h total across 350 pairs; task sessions, not day-long | no | excellent HoloLens files: depth, head/video pose, eyes, hands, IMU, RGB/audio | CDLA permissive | Strong spatial-task corpus; poor duration/longitudinal fit |
| **TaskSDK / Affordances-in-the-Wild** | no exact public first-person long-duration corpus evidence found | unknown | unknown | unknown | Exclude unless exact repo/paper supplied |

## Evidence notes by dataset

### 1. EgoLife baseline

**Claim.** EgoLife is the only surveyed candidate matching WorldMM's current longitudinal target: six participants over a week, natural daily activities, Aria capture, and memory-assistant framing.

**Evidence.** The project README describes “six participants over a week,” Meta Aria glasses, synchronized third-person cameras, and mmWave sensors ([GitHub](https://github.com/EvolvingLMMs-Lab/EgoLife/blob/7a97157908757cc898c26835b718653055ecc5f5/README.md#L20-L23)); it links arXiv `2503.03803`, HF data, and release notes ([GitHub](https://github.com/EvolvingLMMs-Lab/EgoLife/blob/7a97157908757cc898c26835b718653055ecc5f5/README.md#L1-L13), [GitHub](https://github.com/EvolvingLMMs-Lab/EgoLife/blob/7a97157908757cc898c26835b718653055ecc5f5/README.md#L26-L35)). HF dataset card shows Video modality, MIT license, arXiv tag `2503.03803`, and 32k rows. Sampling HF HEAD responses for 30 A1_JAKE/DAY1 MP4 chunks gave mean 15.95 MB/chunk; 828 chunks imply ≈12.3 GiB for A1_JAKE DAY1, and 32,001 chunks imply ≈475 GiB for all public MP4 chunks. This is an empirical estimate from HF file listing + HEAD metadata, not an official size.

**Usefulness.** Best for all four WorldMM axes: day summaries, cross-day recurrence, personal object/place continuity, habit/event memory.

**Limit.** Public portable spatial files beyond video/audio are unclear. README claims Aria/mmWave, but the visible HF dataset is video rows; pose/depth/gaze may need separate unpublished/raw access.

### 2. Ego4D long videos

**Claim.** Ego4D is the strongest non-EgoLife long-session dataset: huge scale, sessions up to 10h, and some spatial/audio metadata, but not consistently multi-day per wearer.

**Evidence.** Repo states Ego4D has “over 3700 hours” of annotated first-person video ([GitHub](https://github.com/facebookresearch/Ego4d/blob/e33c2b9e42d3f7b17e6a76b5fdd16d7bcc12f901/README.md#L22-L25)); arXiv says 3,670h, 931 camera wearers, 74 locations, 9 countries, and each participant wore a camera for **1 to 10 hours at a time** ([arXiv:2110.07058](https://arxiv.org/abs/2110.07058)). Official metadata schema includes `duration_sec`, physical settings, video/audio durations, and concurrent-video sets ([Ego4D docs](https://ego4d-data.org/docs/data/metadata/)); IMU is released as flat CSV per video with canonical timestamps ([Ego4D docs](https://ego4d-data.org/docs/data/imu/)). The repo also exposes the downloader CLI ([GitHub](https://github.com/facebookresearch/Ego4d/blob/e33c2b9e42d3f7b17e6a76b5fdd16d7bcc12f901/README.md#L27-L34)).

**Spatial signals.** Pose: no general camera-pose release found. Depth: not general; 3D meshes/scans for subset. Place: physical settings and scenarios. Gaze: subset. Audio: subset. IMU: CSV subset. Features: precomputed SlowFast features from docs/paper.

**Usefulness.** Good stress test for single-session long memory (up to 10h) and broad daily-life diversity.

**Limit.** Spatial signals are uneven by capture device/site. Longest single session answer: **10h reported**; exact max/median from full metadata requires licensed metadata pull.

### 3. EPIC-KITCHENS-100

**Claim.** EPIC-KITCHENS-100 is longitudinal per participant/kitchen and public for research, but released recordings are mostly short clips; spatial metadata is weak.

**Evidence.** Official page says 45 kitchens, 4 cities, head-mounted camera, **100 hours**, 20M frames, 90K action segments, and multi-day kitchen activities ([EPIC official](https://epic-kitchens.github.io/2020-100)). GitHub README links arXiv `2006.13256` and calls it the largest first-person dataset extension ([GitHub](https://github.com/epic-kitchens/epic-kitchens-100-annotations/blob/ea8b40457a400c3fffa1c7f406ef3dc169cc2522/README.md#L1-L10)). The official `EPIC_100_video_info.csv` provides per-video durations ([GitHub](https://github.com/epic-kitchens/epic-kitchens-100-annotations/blob/ea8b40457a400c3fffa1c7f406ef3dc169cc2522/EPIC_100_video_info.csv#L1-L10)). Computing that CSV: 700 videos, total 100.03h, min 0.0029h, median 0.0855h (5.1 min), max 1.03h (`P01_109`), 37 participants, participant-hour median 2.32h, max 8.28h.

**Spatial signals.** Pose/depth/place/gaze: no. Audio: videos have audio/narration. Portable annotations: CSV actions, timestamps, object boxes, pre-extracted RGB/flow; README erratum confirms pre-extracted RGB/flow frames exist ([GitHub](https://github.com/epic-kitchens/epic-kitchens-100-annotations/blob/ea8b40457a400c3fffa1c7f406ef3dc169cc2522/README.md#L46-L58)).

**Usefulness.** Good for multi-day kitchen memory and action/object recurrence.

**Limit.** Not day-long sessions; poor for place/pose/depth/gaze axes.

### 4. Charades-Ego

**Claim.** Charades-Ego is not a long-duration longitudinal memory corpus; it is clip-based paired first/third-person activity video.

**Evidence.** arXiv reports 68,536 activity instances in **68.8 hours** of first/third-person video ([arXiv:1804.09626](https://arxiv.org/abs/1804.09626)). AI2 Charades page points to Charades-Ego and lists Charades clip data, annotations, features, and baseline GitHub ([AI2](https://prior.allenai.org/projects/charades)). Baseline repo is activity-recognition code, not spatial metadata ([GitHub](https://github.com/gsig/charades-algorithms/blob/927794cd04c588f1e749e96f5c0e69d81a1576e0/README.md#L1-L13)).

**Spatial signals.** No portable pose/depth/gaze/place. Audio not central.

**Usefulness.** Useful only as a short-clip activity classifier sanity set.

**Limit.** No day/session continuity, no personal longitudinal record.

### 5. University of Edinburgh / SenseCam lifelogs

**Claim.** Personal-camera lifelog datasets match the *longitudinal* idea historically, but not modern WorldMM spatial requirements.

**Evidence.** SenseCam-era research commonly used low-rate wearable photos over days/weeks, but robust public GitHub repos with portable pose/depth/gaze/audio were not found in this pass. GitHub search mainly surfaced unrelated SenseCam software/hardware, not a canonical open dataset. Treat as literature background, not pull target, unless a specific University of Edinburgh dataset URL is supplied.

**Spatial signals.** Usually image timestamps/GPS/accelerometer at best; no portable 6DoF pose/depth/gaze.

**Usefulness.** Longitudinal memory/event segmentation concepts.

**Limit.** Access fragmentation, privacy, low frame rate, weak spatial files.

### 6. WALT (When-Where-What-Why)

**Claim.** No exact, public, long-duration egocentric WALT dataset was verified. Do not use the name as evidence without exact paper/repo.

**Evidence.** GitHub repository/code search for “WALT egocentric” and “When Where What Why egocentric” returned no relevant dataset repo. This is an uncertainty, not a negative proof.

**Verdict.** Exclude from top candidates until source disambiguated.

### 7. NarrationBench / EgoSchema

**Claim.** EgoSchema is a long-video-understanding benchmark over Ego4D clips, not a day-long spatial dataset.

**Evidence.** README identifies EgoSchema as “Very Long-form Video Language Understanding,” links arXiv `2308.09126`, and provides Kaggle/Wasabi downloads ([GitHub](https://github.com/egoschema/EgoSchema/blob/505c787376b5e066d0ae406d0e0d41245cebba15/README.md#L1-L11), [GitHub](https://github.com/egoschema/EgoSchema/blob/505c787376b5e066d0ae406d0e0d41245cebba15/README.md#L28-L44)). Paper says >5,000 QA pairs, >250h total, each question over a **3-minute clip**, median temporal certificate ≈100s ([arXiv:2308.09126](https://arxiv.org/abs/2308.09126)).

**Spatial signals.** Inherits Ego4D video references; no added pose/depth/gaze/place files.

**NarrationBench.** No canonical public GitHub/arXiv match was verified in this pass; likely benchmark-only if based on existing videos.

**Verdict.** Good evaluation set for long-reasoning models, not a source for multi-day memory.

### 8–9. Project Aria day-in-life corpora and Aria Replay

**Claim.** Public Project Aria tooling is excellent for portable spatial signals, but public “day-in-life/full-day Aria Replay” corpora were not verified. The verified public Aria Gen2 Pilot dataset is short/multi-sequence, not day-long.

**Evidence.** Project Aria tools README says the toolkit supports Aria open datasets and Gen1/Gen2 data ([GitHub](https://github.com/facebookresearch/projectaria_tools/blob/1bfe7e50c52698583b2ee3bbb63c5c177d962311/README.md#L1-L8)). OSI docs list raw synchronized cameras/IMU/audio, on-device VIO/eye gaze/hand tracking, offline machine perception, calibration, and standard-format exports ([GitHub](https://github.com/facebookresearch/projectaria_tools/blob/1bfe7e50c52698583b2ee3bbb63c5c177d962311/website/docs-research-tools/osi.mdx#L17-L32), [GitHub](https://github.com/facebookresearch/projectaria_tools/blob/1bfe7e50c52698583b2ee3bbb63c5c177d962311/website/docs-research-tools/osi.mdx#L40-L54)). Gen2 Pilot docs say 12 sequences, raw VRS, MPS, ASR, heart-rate, depth, 2D/3D boxes ([GitHub](https://github.com/facebookresearch/projectaria_tools/blob/1bfe7e50c52698583b2ee3bbb63c5c177d962311/website/docs-research-tools/dataset/pilot/download.mdx#L40-L49)); dataset format lists `closed_loop_trajectory.csv`, point clouds, hand tracking, diarization, scene boxes, depth PNGs ([GitHub](https://github.com/facebookresearch/projectaria_tools/blob/1bfe7e50c52698583b2ee3bbb63c5c177d962311/website/docs-research-tools/dataset/pilot/format.mdx#L5-L53)). The Gen2 Pilot repo confirms VRS, MPS SLAM trajectories, hand tracking, diarization, depth, 3D scene reconstruction ([GitHub](https://github.com/facebookresearch/projectaria_gen2_pilot_dataset/blob/6c773d91b59f627953626381b983ec4c3f9bcd2c/README.md#L1-L11), [GitHub](https://github.com/facebookresearch/projectaria_gen2_pilot_dataset/blob/6c773d91b59f627953626381b983ec4c3f9bcd2c/README.md#L46-L52)).

**Spatial signals.** Best portable spatial schema in this survey: pose/VIO, depth, gaze, hand, audio/ASR, place/objects. MPS trajectory docs define 6DoF trajectory CSV fields ([GitHub](https://github.com/facebookresearch/projectaria_tools/blob/1bfe7e50c52698583b2ee3bbb63c5c177d962311/website/docs-technical-specs/mps/data_formats/slam/mps_trajectory.mdx#L6-L32)).

**Limit.** No verified public full-day/day-in-life recordings. Needs package/tools for convenient parsing, though files are portable.

### 10. OpenEQA / Vista

**Claim.** OpenEQA has spatial QA and portable RGB-D/pose extraction paths, but it is not egocentric day-long wearable capture.

**Evidence.** README defines OpenEQA as 1,600+ QA pairs from 180+ real-world environments ([GitHub](https://github.com/facebookresearch/open-eqa/blob/cfa3fce4595c1622bb2f8a38ae2ca9aae9eb685b/README.md#L10-L18)). Data README says episode histories come from HM3D and ScanNet, with HM3D RGB-only 12GB or RGB-D+intrinsics+pose 16GB, and ScanNet RGB-only 62GB or RGB-D+intrinsics+pose 70GB ([GitHub](https://github.com/facebookresearch/open-eqa/blob/cfa3fce4595c1622bb2f8a38ae2ca9aae9eb685b/data/README.md#L7-L18), [GitHub](https://github.com/facebookresearch/open-eqa/blob/cfa3fce4595c1622bb2f8a38ae2ca9aae9eb685b/data/README.md#L80-L99), [GitHub](https://github.com/facebookresearch/open-eqa/blob/cfa3fce4595c1622bb2f8a38ae2ca9aae9eb685b/data/README.md#L136-L154)).

**Spatial signals.** Pose/depth yes if extracted; place yes via 3D scans; gaze/audio no.

**Limit.** Simulated/scan episode histories, not wearable continuous life.

### 11. HoloAssist / HoloAssist-Long

**Claim.** HoloAssist is a strong spatial/task corpus, but not day-long/multi-day per wearer.

**Evidence.** Official page says 169h, 350 instructor-performer pairs, seven synchronized streams, physical manipulation tasks, and sample modalities RGB/depth/hand pose/eye gaze/IMUs ([HoloAssist](https://holoassist.github.io/)). The dataset README says CDLA v2 license and per-recording folders with synchronized modalities ([GitHub](https://github.com/holoAssist/holoassist.github.io/blob/8e2f38a8a7f5846fb31ceee17be2e9186ee62cfb/data_links/README.md#L1-L10), [GitHub](https://github.com/holoAssist/holoassist.github.io/blob/8e2f38a8a7f5846fb31ceee17be2e9186ee62cfb/data_links/README.md#L30-L33)). Folder layout includes `AhatDepth`, `Eyes_sync.txt`, `Hands`, `Head_sync.txt`, IMU, `Video/Pose_sync.txt`, and MP4 video ([GitHub](https://github.com/holoAssist/holoassist.github.io/blob/8e2f38a8a7f5846fb31ceee17be2e9186ee62cfb/data_links/README.md#L36-L73)).

**Spatial signals.** Pose yes, depth yes, gaze yes, audio/conversation yes, IMU yes, place/task context yes.

**Limit.** Task sessions; no evidence for ≥6h single sessions or multi-day subject continuity.

### 12. TaskSDK / Affordances-in-the-Wild

**Claim.** No exact long-duration first-person TaskSDK/Affordances-in-the-Wild corpus was verified. GitHub search for `TaskSDK` returned unrelated SDKs; `Affordances-in-the-Wild` returned no relevant public repo.

**Verdict.** Exclude until exact citation supplied.

## Ranking for WorldMM testing

1. **Ego4D long videos** — best next broad-scale long-session corpus. Pull metadata first, identify videos near 6–10h with audio/gaze/IMU/3D availability.
2. **EPIC-KITCHENS-100** — best public multi-day participant/kitchen continuity outside EgoLife, but individual videos short and spatial axes weak.
3. **HoloAssist** — best portable spatial signals outside Aria, but duration/longitudinal mismatch; useful for pose/depth/gaze/audio axis integration tests.

Honorable mention: **Project Aria Gen2 Pilot** for file format/spatial pipeline validation, not duration.

## Single best next pull

If criterion is “multi-day single-subject record, run all four axes, no further SDK install,” the single best next pull remains **EgoLife A1_JAKE all days**, not an external dataset. Among external datasets, **EPIC-KITCHENS-100 participant P01 or P04** is easiest: plain MP4/CSV, multi-day/person, no SDK, but cannot run true pose/depth/gaze axes. If true spatial axes matter more than multi-day, pull **HoloAssist sample/full subset**.

Recommendation: **pull EgoLife A1_JAKE DAY2–DAY7 next** for longitudinal WorldMM; use Ego4D metadata pull in parallel to find 6–10h sessions with gaze/IMU/audio.

## Storage estimate for best pull

Best pull: **EgoLife A1_JAKE**.

- One day, A1_JAKE DAY1: 828 MP4 chunks. HF HEAD sample mean 15.95 MB/chunk → **≈12.3 GiB**. Median estimate ≈11.9 GiB.
- All seven A1_JAKE days: chunks = 828 + 854 + 910 + 953 + 946 + 1068 + 707 = 6,266 chunks → mean estimate **≈93.1 GiB**.
- Full public EgoLife MP4 listing: 32,001 chunks → mean estimate **≈475 GiB**.

Caveat: estimates use sampled public MP4 HEAD `Content-Length`; official storage size not reported in README/HF card.
