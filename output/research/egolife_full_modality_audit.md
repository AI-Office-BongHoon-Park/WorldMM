# EgoLife full modality audit (HF `lmms-lab/EgoLife`)

Audit date: 2026-05-21 KST. Scope: EgoLife project page, arXiv `2503.03803` v3, GitHub project repo, and Hugging Face dataset `lmms-lab/EgoLife` main revision `143fb319be7aa5ae210c936bf4f0f3a86092afb0`. No dataset blobs downloaded; inventory uses HF metadata/tree listing.

## 1. Bottom line

HF `lmms-lab/EgoLife` currently exposes **video, subtitles/annotations, and instruction/QA JSON only**. The public main tree has:

- `*.mp4`: 32,001 files, 512,225,930,110 bytes (~477.05 GiB)
- `*.srt`: 808 files, 49,606,253 bytes (~47.3 MiB)
- `*.json`: 3 files, 40,223,681 bytes (~38.4 MiB)
- misc: `.gitattributes`, `README.md`, `.DS_Store`, one `.nfs...` file, one tiny `.mp3`

**No public HF file paths contain or use formats for spatial signals**: no `imu`, `gaze`, `slam`, `pose`, `depth`, `mmwave`, `radar`, `wifi`, `csv`, `hdf5`, `npy`, `pkl`, or body-keypoint files in the main release. The paper/project describe those signals as collected/processed, but the HF release does not ship them right now.

Most useful not-yet-pulled items for our local setup: `EgoLifeCap/DenseCaption/.../*.srt`, `EgoLifeCap/Transcript/.../*.srt`, `EgoIT/*.json`, and `EgoLifeQA/EgoLifeQA_A1_JAKE.json`. Spatial-rich downloadable modality: **none found**.

## 2. Evidence sources

- Project page says participants wore glasses recording **video, gaze, IMU**, synchronized with 15 GoPros, plus 3D scans; it also says annotations include **transcriptions** and **dense captions**: <https://egolife-ai.github.io/>.
- Paper §3 Figure 4 states pipeline synchronizes multi-source **video, audio, IMU** from Aria glasses and GoPros, then applies privacy protection, dense captioning, transcription, and EgoLifeQA construction: <https://ar5iv.labs.arxiv.org/html/2503.03803#S3.F4>.
- Paper §3.1 says six volunteers, seven days, Meta Aria glasses, 15 GoPro cameras, and millimeter-wave radars for spatial/motion data: <https://ar5iv.labs.arxiv.org/html/2503.03803#S3.SS1>.
- Paper Appendix D.7 lists planned/declared formats: MP4 for video+audio, CSV for IMU/gaze/radar, HDF5 for WiFi, JSON for annotations: <https://ar5iv.labs.arxiv.org/html/2503.03803#A4.SS7>.
- GitHub README says “EgoLife video is released at HuggingFace” and “EgoIT-99K dataset” released; it does not claim public release of gaze/IMU/mmWave. Evidence: [`README.md` lines 29-33](https://github.com/EvolvingLMMs-Lab/EgoLife/blob/7a97157908757cc898c26835b718653055ecc5f5/README.md#L29-L33). Same README describes project capture using Meta Aria, third-person cameras, and mmWave sensors at [`README.md` line 22](https://github.com/EvolvingLMMs-Lab/EgoLife/blob/7a97157908757cc898c26835b718653055ecc5f5/README.md#L22).
- HF dataset card metadata: `license: mit`, `private=False`, `gated=False`, tags include `modality:video`, `language:zh`.

## 3. Local state confirmed

Local path checked: `data/EgoLife/A1_JAKE/DAY1/*.mp4`.

| Local item | Value |
|---|---:|
| Directory exists | yes |
| MP4 files | 828 |
| Non-empty MP4 files | 91 |
| Empty MP4 files | 737 |
| Non-empty bytes | 1,312,588,676 |
| Non-empty size | ~1.22 GiB |
| First non-empty file | `data/EgoLife/A1_JAKE/DAY1/DAY1_A1_JAKE_11094208.mp4` |
| Last non-empty file | `data/EgoLife/A1_JAKE/DAY1/DAY1_A1_JAKE_22000000.mp4` |

HF main tree has `A1_JAKE/DAY1` with **828 MP4 paths**, so the local tree has all expected path names for that subject/day, but only 91 actual non-empty chunks.

Example HF video paths:

```text
A1_JAKE/DAY1/DAY1_A1_JAKE_11094208.mp4
A1_JAKE/DAY1/DAY1_A1_JAKE_11100000.mp4
A1_JAKE/DAY1/DAY1_A1_JAKE_11103000.mp4
```

## 4. Complete public HF main-tree inventory

### 4.1 Video/audio MP4 chunks

| Field | Value |
|---|---|
| Path pattern | `{subject}/DAY{n}/DAY{n}_{subject}_{timestamp}.mp4` |
| Example | `A1_JAKE/DAY1/DAY1_A1_JAKE_11094208.mp4` |
| Format | MP4, video+audio chunks |
| Coverage | 6 subjects × 7 days = 42 subject/day folders; 32,001 MP4 files total |
| Per subject/day range | min 358, max 1,107 chunks |
| Total size | 512,225,930,110 bytes (~477.05 GiB) |

Subject-level sizes:

| Subject | MP4 count | Size |
|---|---:|---:|
| `A1_JAKE` | 6,266 | ~95.72 GiB |
| `A2_ALICE` | 5,515 | ~79.73 GiB |
| `A3_TASHA` | 4,844 | ~73.08 GiB |
| `A4_LUCIA` | 5,266 | ~82.32 GiB |
| `A5_KATRINA` | 4,806 | ~69.84 GiB |
| `A6_SHURE` | 5,304 | ~76.36 GiB |

Low/high subject-day examples: `A5_KATRINA/DAY7` has 358 MP4s; `A2_ALICE/DAY6` has 1,107 MP4s. The release is complete enough to cover all subjects/days, but chunk counts vary by day.

### 4.2 DenseCaption subtitles

| Field | Value |
|---|---|
| Path pattern | `EgoLifeCap/DenseCaption/{subject}/DAY{n}/{subject}_DAY{n}_{hour}.srt` |
| Examples | `EgoLifeCap/DenseCaption/A1_JAKE/DAY1/A1_JAKE_DAY1_11000000.srt`; `EgoLifeCap/DenseCaption/A1_JAKE/DAY1/A1_JAKE_DAY1_12000000.srt` |
| Format | SRT subtitle text |
| Coverage | All 42 subject/day combos; 406 SRT files plus `EgoLifeCap/DenseCaption/A1_JAKE/.DS_Store` |
| Per subject/day range | 5-12 SRT files |
| Total size | 21,189,642 bytes (~20.21 MiB) |

These are downloadable now and not part of our local `data/EgoLife/A1_JAKE/DAY1/*.mp4` inventory.

### 4.3 Speech transcripts

| Field | Value |
|---|---|
| Path pattern | `EgoLifeCap/Transcript/{subject}/DAY{n}/{subject}_DAY{n}_{hour}.srt` |
| Examples | `EgoLifeCap/Transcript/A1_JAKE/DAY1/A1_JAKE_DAY1_11000000.srt`; `EgoLifeCap/Transcript/A1_JAKE/DAY1/A1_JAKE_DAY1_12000000.srt` |
| Format | SRT subtitle text |
| Coverage | All 42 subject/day combos; 402 SRT files |
| Per subject/day range | 5-12 SRT files |
| Total size | 28,422,759 bytes (~27.11 MiB) |

Paper §3.3 describes how transcripts were produced: synchronize egocentric videos, merge six participants' audio, ASR, diarization, human review, then split/refine per participant (<https://ar5iv.labs.arxiv.org/html/2503.03803#S3.SS3>).

### 4.4 EgoIT instruction-tuning JSON

| Field | Value |
|---|---|
| Paths | `EgoIT/EgoLife_Caption.json`; `EgoIT/EgoLife_QA.json` |
| Format | JSON |
| Coverage | Aggregate EgoIT data, not per subject/day folders |
| Total size | 39,632,426 bytes (~37.80 MiB) |

`EgoIT/EgoLife_Caption.json` is 12,787,895 bytes. `EgoIT/EgoLife_QA.json` is 26,844,531 bytes. GitHub README explicitly says EgoIT-99K was released at HuggingFace ([README line 31](https://github.com/EvolvingLMMs-Lab/EgoLife/blob/7a97157908757cc898c26835b718653055ecc5f5/README.md#L31)).

### 4.5 EgoLifeQA JSON

| Field | Value |
|---|---|
| Path | `EgoLifeQA/EgoLifeQA_A1_JAKE.json` |
| Format | JSON |
| Coverage | Only `A1_JAKE` visible in public HF main tree; no `EgoLifeQA_A2_ALICE.json` etc. found |
| Total size | 591,255 bytes (~0.56 MiB) |

Paper §3.5 says final benchmark target was 500 QA per participant and 3K total (<https://ar5iv.labs.arxiv.org/html/2503.03803#S3.SS5>), but the public HF main tree currently exposes only the A1 file path above. No `EgoLifeRetrievalQA` or `EgoLifeTemporalQA` file path found in the HF main tree.

### 4.6 Misc / generated files

| Path | Notes |
|---|---|
| `README.md` | 315-byte dataset card; says “Data cleaning, stay tuned!” |
| `.gitattributes` | repository metadata |
| `EgoLifeCap/DenseCaption/A1_JAKE/.DS_Store` | macOS metadata artifact |
| `A6_SHURE/DAY6/.nfs0000000102a0cad000000de0` | NFS artifact, no extension |
| one `.mp3` | tiny 253-byte artifact; not a meaningful audio modality release |

HF also has an auto-converted parquet ref (`refs/convert/parquet/default/train/0000.parquet`, 354,019 bytes). That is a Hugging Face viewer conversion, not the main release payload, and it does not add spatial modalities.

## 5. Modalities described by paper/project vs public release

| Modality / signal | Described as collected or processed? | Public HF main-tree files now? | Evidence / notes |
|---|---|---|---|
| Egocentric video+audio from Meta Aria | Yes | Yes: `{subject}/DAY{n}/*.mp4` | Project/paper + HF MP4 tree |
| Third-person GoPro/exo video | Yes, 15 cameras | Not as separate exo-camera paths | Paper §3.1 says GoPros recorded multi-angle; HF video folders are participant/day only, not camera IDs |
| Audio transcripts | Yes | Yes: `EgoLifeCap/Transcript/.../*.srt` | Paper §3.3 + HF SRT paths |
| Dense visual-audio captions | Yes | Yes: `EgoLifeCap/DenseCaption/.../*.srt`; `EgoIT/EgoLife_Caption.json` | Paper §3.4 + HF paths |
| EgoLifeQA | Yes | Partial public file: `EgoLifeQA/EgoLifeQA_A1_JAKE.json`; also `EgoIT/EgoLife_QA.json` | Paper says 3K total; HF shows only A1 benchmark JSON |
| EgoLifeRetrievalQA / EgoLifeTemporalQA | Not found as named release files | No | HF path search found 0 `retrieval`, 0 `temporal` |
| IMU | Paper pipeline says video/audio/IMU synchronized; Appendix D.7 says IMU CSV | No | HF path search: 0 `imu`, 0 `.csv` |
| Eye gaze | Project page says glasses record gaze; Appendix D.7 says gaze CSV | No | HF path search: 0 `gaze`, 0 `.csv` |
| SLAM / 6-DoF / Aria MPS trajectories | Figure 2 uses Aria Multi-MPS 3D reconstruction and participant traces | No | HF path search: 0 `slam`, 0 `pose`, 0 trajectory-like files |
| Depth | Not a released file type in HF | No | HF path search: 0 `depth` |
| mmWave / radar | Paper says two mmWave devices; Appendix says radar CSV | No | HF path search: 0 `mmwave`, 0 `radar`, 0 `.csv` |
| WiFi CSI | Appendix says Milan only, HDF5 | No | HF path search: 0 `wifi`, 0 `.hdf5`, 0 `.h5`; Beijing HF release has no WiFi paths |
| Body pose keypoints | Not found as public payload | No | HF path search: 0 `keypoint`, 0 `pose`; paper mentions GoPro/mmWave but no public body-keypoint files |
| Place / room labels | Activity timeline categories exist in paper; no room/place file | No | HF path search: 0 `place`, 0 `room`; no labels JSON besides QA/EgoIT |

Interpretation: the authors **collected/processed spatial-capable signals internally** (gaze, IMU, mmWave, 3D/MPS traces, GoPro multi-view; WiFi for Milan) but the current public HF dataset only releases videos and textual annotations/instruction data. Appendix D.7’s CSV/HDF5 formats are dataset-card declarations, not currently present in the public `lmms-lab/EgoLife` main tree.

## 6. License and access notes

HF metadata reports:

```text
private: False
gated: False
license: mit
language: zh
modality: video
```

So access is not gated, but the exposed artifact set is limited. License badge/card says MIT. Privacy cleaning described by paper includes face/license-plate blur and muting sensitive audio (<https://ar5iv.labs.arxiv.org/html/2503.03803#A4.SS2> and <https://ar5iv.labs.arxiv.org/html/2503.03803#A4.SS5>).

## 7. Answer to “what spatial-rich modality is downloadable right now?”

**None found in the HF main release.**

Downloadable now but not spatial-rich:

```text
EgoLifeCap/DenseCaption/{subject}/DAY{n}/*.srt
EgoLifeCap/Transcript/{subject}/DAY{n}/*.srt
EgoIT/EgoLife_Caption.json
EgoIT/EgoLife_QA.json
EgoLifeQA/EgoLifeQA_A1_JAKE.json
```

Specifically:

- Gaze: **not present** (`0` paths containing `gaze`; no CSV files).
- Pose / SLAM / 6-DoF: **not present** (`0` paths containing `pose` or `slam`).
- IMU: **not present** (`0` paths containing `imu`; no CSV files).
- mmWave / radar: **not present** (`0` paths containing `mmwave`/`radar`; no CSV files).
- Body pose keypoints: **not present** (`0` paths containing `keypoint`; no npy/pkl/json keypoint payloads).
- Place / room labels: **not present** as separate files (`0` paths containing `place`/`room`).

Practical next pull, if allowed: annotations only. Spatial trajectory features must be inferred from video/transcripts/captions, requested from authors, or reconstructed independently; HF does not provide raw spatial sensors today.
