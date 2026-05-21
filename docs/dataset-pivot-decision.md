# Dataset Pivot Decision — Where to get pre-extracted spatial signals

**Date:** 2026-05-21 KST
**Trigger:** Need 6-DoF pose / eye gaze / semantic place for spatial memory analysis WITHOUT building SLAM ourselves or installing Project Aria SDK for raw VRS decoding.

**Sister reports:**
- [`output/research/egolife_full_modality_audit.md`](../output/research/egolife_full_modality_audit.md) — what HF EgoLife actually ships
- [`output/research/preextracted_spatial_datasets.md`](../output/research/preextracted_spatial_datasets.md) — 14-dataset catalog
- [`output/research/codebase_spatial_extension_points.md`](../output/research/codebase_spatial_extension_points.md) — current adapter surface

---

## 1. Decisive finding: EgoLife에 추가 받을 spatial signal **없음**

HF `lmms-lab/EgoLife` 공개 release 전수조사 결과:

| 파일 유형 | 개수 | 크기 |
|---|---:|---:|
| `*.mp4` | 32,001 | 477 GiB |
| `*.srt` (captions) | 808 | 47 MiB |
| `*.json` (QA/IT) | 3 | 38 MiB |
| **pose / gaze / IMU / mmWave / depth** | **0** | **0** |

Paper 2503.03803 §3.1 + App D.7에는 Aria glasses에서 IMU·gaze 캡처, CSV·HDF5 포맷 명시되어 있으나 **HF에 미공개**. GitHub README도 "EgoLife video is released at HuggingFace"만 언급, IMU/gaze 공개 약속 없음.

→ **"EgoLife에 더 받는다"는 옵션은 죽음.** 데이터 자체가 없음.

---

## 2. 대안 평가

[`preextracted_spatial_datasets.md`](../output/research/preextracted_spatial_datasets.md)의 14 데이터셋 평가 결과:

| 데이터셋 | Egocentric | Spatial signals | 파일 형식 | SDK 필요 | License | 사이즈 | 비고 |
|---|:---:|---|---|:---:|---|---:|---|
| **AEA** | ✅ | ✅ 6DoF + gaze + pointcloud + speech | VRS + **CSV/JSON/GZ** | only for download | gated research | **353 GB** (143 seq) | **Aria, MPS portable** |
| ADT | ✅ | ✅ GT 6DoF + 3D mesh + objects | VRS + JSON + GLB | only for download | gated research | ~few hundred GB | Aria + Digital Twin |
| Aria Pilot | ✅ | ✅ VIO + gaze | VRS + JSON | partial | gated | ~tens GB | older Gen-1 |
| EgoExo4D | ✅+exo | ✅ trajectory + gaze CSV/JSON | CSV/JSON/GZ | partial | gated | hundreds GB | dual-perspective |
| Ego4D | ✅ | weak — narrations only | JSON | n/a | OK | TB scale | 3-D 거의 없음 |
| HOI4D | ✅ | ✅ hand 3D + object 6DoF | per-frame | partial | research | tens GB | static tasks |
| **HOT3D-Clips** | ✅ | ✅ camera+hand+object 6DoF | **JSON in tar** + GLB | no | research | tens GB | hand-object focus |
| HoloAssist | ✅ HoloLens | ✅ 6DoF + hand + gaze + IMU | text streams | no | research | tens GB | non-Aria, depth |
| ARCTIC | ✅ | ✅ NPY hand+object+camera | NPY | no | research | tens GB | controlled tasks |
| ScanNet++ / Replica | 3rd-person | mesh + 6DoF camera | PLY + JSON | no | research | tens GB | room scaffold |
| Habitat-Sim | ✅ sim | ✅ GT everything | JSON + npy | minor | permissive | small | synthetic |
| EPIC-KITCHENS | ✅ | weak — annotations | CSV | no | OK | hundreds GB | no pose |

---

## 3. 결론 — 3가지 path와 trade-off

### Path A: **AEA로 pivot** (recommended)
- Aria glasses 하드웨어로 capture된 single-person daily-life. EgoLife와 동일 센서 패밀리.
- **MPS 폴더 CSV는 pandas로 직접 ingest** — VRS 안 건드림 = SDK 사실상 불필요
- 1-2 sequence (~5-10 GB) 만 받아도 우리 spatial memory 전체 파이프라인 검증 가능
- Download는 `aria_dataset_downloader` Python script 1개 사용; SDK install 부담 아님
- `closed_loop_trajectory.csv` (6DoF) + `general_eye_gaze.csv` + `speech.csv` + `semidense_points.csv.gz` → 우리 `GeometricGrounding` 확장형으로 직매핑 가능

### Path B: **HOT3D-Clips로 pivot**
- 손-객체 정밀 6DoF (per-frame JSON in tar)
- "spatial memory beyond QA" 시나리오 중 cooking/DIY assistance와 align
- 단점: daily-life가 아니라 controlled hand-object tasks

### Path C: **EgoLife 유지 + 외부 dataset 보조 비교용**
- 현재 결과 (26/30 visual axis) 보존
- AEA 1 sequence를 portable CSV로 ingest해서 **"spatial-rich vs spatial-poor 데이터 비교"** report 작성
- 데이터셋 교체 없이 capability 차이만 측정
- 가장 가벼움, 가장 적은 risk

---

## 4. 권고: **Path C → Path A** 단계별 진입

### Step 1 (immediate): AEA single sequence MPS portable 검증
- 1 sequence만 다운로드 (~5 GB)
- MPS CSV 파일을 pandas로 직접 열어 schema/coverage 확인
- 우리 spatial memory의 `SpatialTripleEntry` 구조에 mapping 가능성 평가
- 산출: `output/research/aea_mps_ingestion_feasibility.md`

### Step 2 (if Step 1 viable): EgoLife vs AEA 비교 분석 report
- 같은 spatial memory 파이프라인을 EgoLife (no pose) 1 day + AEA (with pose) 1 sequence에 적용
- 비교 axis: closed-vocab triple coverage, place-graph density, gaze-anchored salience, retrieval quality
- 산출: `output/research/spatial_signal_richness_comparison.md`

### Step 3 (if comparison shows value): Path A full commit
- 추가 1-2 AEA sequences 다운로드
- `SpatialTripleEntry`에 `pose_6dof`, `gaze_target`, `place_anchor` optional fields 추가
- 새 `tools/build_aea_spatial_sidecar.py` builder

---

## 5. 즉시 의사결정 필요

**질문**: Step 1 (AEA single sequence 다운로드 + MPS schema 검증) 진행할까요?

- Yes → AEA license 동의 후 [aria_dataset_downloader](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_everyday_activities_dataset/aea_download_dataset) 가져와서 1 sequence (~5 GB) 다운로드 → MPS CSV 읽어서 schema report 작성
- No, EgoLife로 충분 → 현재 PR #1 마무리, 추후 작업

또는:

- **HOT3D-Clips를 더 보고 결정**하고 싶으면 비슷한 schema check task 한 번 더
- **EgoExo4D 다운로드 액세스 시도**하고 싶으면 access form 작성 안내
