# AI Glass / Smart Glass Sensor Inventory for Spatial-Indexed Memory

_Date: 2026-05-21 KST_

Scope: shipping consumer smart glasses, research glasses, and headset-class spatial computers relevant to WorldMM-style spatial memory. “Unknown” means not publicly disclosed in official docs/pages found during this pass. Product pages often disclose marketing features, not raw sensor APIs.

## Quick read

| Platform | Best spatial-memory value | Main limitation |
|---|---|---|
| Meta Project Aria / Aria Gen 2 | Research-grade raw egocentric streams: RGB, SLAM cams, IMU, eye gaze, GPS/GNSS, mic array, MPS/VIO/SLAM outputs | Not consumer shipping; access controlled |
| Ray-Ban Stories / Ray-Ban Meta | Shipping POV photo/video + mic array + assistant, good for episodic capture | No public raw IMU/SLAM/eye/depth streams; no real localization |
| Apple Vision Pro | Strongest room-scale sensing: LiDAR + TrueDepth + world tracking + eye/hand input | Privacy sandbox; apps do not get raw sensor feed broadly; headset not glasses |
| Snap Spectacles 4 AR | Lightweight AR developer platform with 6DoF + hand/surface/marker tracking | ~30 min battery; creator/developer access; sparse raw specs |
| Google Glass Enterprise 2 | Legacy Android monocular camera/display, useful as baseline | Discontinued; no depth/SLAM/eye; weak spatial signal |
| HoloLens 2 | Mature enterprise MR: depth, hand/eye tracking, spatial mapping | Discontinued path; headset form factor; limited raw camera access |
| EgoLife | Uses Meta Aria glasses + fixed GoPros + mmWave, valuable long-life dataset | Not custom glass hardware; raw details inherit Project Aria |
| 2025 AI glasses class | Growing camera/audio/display devices (Solos, Frame/Brilliant, Halliday, Even, Looktech) | Most lack disclosed depth/SLAM/eye/GPS; SDK/raw streams vary widely |

---

## 1. Meta Project Aria / Aria Gen 1 + Gen 2 (research)

Sources: [Aria Gen 1 hardware specs](https://facebookresearch.github.io/projectaria_tools/docs/tech_spec/hardware_spec), [Aria sensors and measurements](https://facebookresearch.github.io/Aria_data_tools/docs/sensors-measurements/), [Gen 1 recording profiles](https://facebookresearch.github.io/projectaria_tools/docs/tech_spec/recording_profiles), [Aria Gen 2 hardware](https://facebookresearch.github.io/projectaria_tools/gen2/technical-specs/device/hardware), [Aria Gen 2 technical specs](https://facebookresearch.github.io/projectaria_tools/gen2/technical-specs), [Aria Pilot Dataset](https://facebookresearch.github.io/Aria_data_tools/docs/pilotdata/pilotdata-index/), [Aria Digital Twin data format](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset/data_format), [Project Aria paper](https://arxiv.org/pdf/2308.13561), [Project Aria Tools docs/SDK](https://facebookresearch.github.io/projectaria_tools/).

| Item | Inventory |
|---|---|
| RGB camera | Gen 1: 1 rolling-shutter RGB/POV, HFOV 110°, max 2880×2880, nominal 10 fps; recording profiles include 1/10/15/20/30 fps and 1408×1408 or 2880×2880. Gen 2: 1 Sony IMX681 RGB, 12 MP full-res 4032×3024 @ 24 fps sensor, public profiles use 2016×1512 or 2560×1920 at 5–30 Hz. |
| SLAM / CV cameras | Gen 1: 2 mono scene/global shutter cameras, HFOV 150°/VFOV 120°, 640×480, max 30 fps, nominal 10 fps. Gen 2: 4 CV cameras, 512×512 @ 30 Hz. Used for VIO/SLAM and hand tracking. |
| Depth sensor | No hardware depth sensor disclosed. ADT includes depth images derived from ground-truth systems, not Aria onboard depth. |
| IMU | Gen 1: 2 IMUs; right 1 kHz, left 800 Hz, accel+gyro (6-axis). Gen 2: dual 6-axis gyro+accelerometer, max 1600 Hz, typical 800 Hz. |
| GPS/GNSS | Gen 1: GPS/Galileo receiver, 1 Hz in profiles with GPS. Gen 2: GNSS with dual-frequency GPS L1+L5 and Galileo E1+E5; profiles list GPS 1 Hz. |
| Magnetometer / barometer | Gen 1: magnetometer 10 Hz, barometer 50 Hz. Gen 2: magnetometer 100 Hz, barometer 50 Hz. |
| Eye tracking / gaze | Gen 1: 2 inward mono global-shutter ET cameras, 320×240 typical, up to 90 fps; MPS outputs per-frame eye gaze. Gen 2: 2 IR ET cameras, hardware 400×400 per eye, real-time gaze up to 90 Hz; profile table lists ET output 5–30 Hz depending profile. |
| Hand tracking | Gen 1: MPS hand tracking requires SLAM cameras. Gen 2: hardware-accelerated 3D articulated hand tracking; profile table includes HT 10–30 Hz. |
| Audio | Gen 1: 7-channel spatial mic array, 48 kHz. Gen 2: 7 acoustic microphones up to 48 kHz + 1 contact mic up to 48 kHz; profile table also lists 7+1 mics at 16 kHz for some outputs. |
| Haptic / display | Research capture glasses; no user display/haptic emphasized in public sensor specs. Gen 2 has speakers. |
| Computation | Gen 1: research capture device + Aria Research Kit / offline Machine Perception Services. Gen 2: Meta custom low-power coprocessor for on-device AI/compression/perception; RAM not disclosed publicly. |
| Data egress / SDK | Raw VRS recordings with timestamped streams; Project Aria Tools and Aria Research Kit expose players for RGB, SLAM cams, eye cams, IMUs, magnetometer, barometer, GPS, Bluetooth/Wi-Fi beacons, audio. Gen 2 docs mention H.265/HEVC image/video and OPUS audio compression, SDK/CLI profiles, MPS outputs for eye/hand/VIO/SLAM. |
| Battery under continuous capture | Gen 1: 2.5 Wh; ~1.5 h continuous recording on profile 0 + 30 h standby. Gen 2: 6–8 h continuous recording with custom lithium-ion battery. |
| Spatial localization | Strong. VIO/SLAM/MPS uses cameras+IMU and can use GNSS/Wi-Fi/barometer for robustness. ADT includes 6DoF Aria trajectory, closed-loop trajectory, semidense points, object poses, 2D/3D boxes, gaze. |
| Public datasets | Aria Pilot Dataset, Aria Everyday Activities, Aria Digital Twin, plus EgoLife (uses Meta Aria). |

**WorldMM implication:** Project Aria is closest to “spatial memory as raw substrate”: common nanosecond timestamps, calibrated coordinate frames, eye gaze, hand tracking, egocentric audio, GPS/GNSS, and MPS trajectories all map cleanly into spatial-indexed memory.

---

## 2. Meta Ray-Ban Stories / Ray-Ban Meta / Ray-Ban Meta Gen 2 (consumer)

Sources: [Ray-Ban Meta product page](https://www.ray-ban.com/usa/ray-ban-meta-ai-glasses), [Meta Ray-Ban Meta Gen 2 announcement](https://www.meta.com/blog/ray-ban-meta-gen-2-now-available-ai-glasses-extended-battery-life-3k-video/), [Ray-Ban Meta brochure/spec PDF surfaced by Luxottica](https://visionsourceshowcase.luxottica.com/wp-content/uploads/Ray-Ban-Meta-Brochure.pdf), [Google Play Meta AI / View app ecosystem](https://play.google.com/store/apps/details?id=com.facebook.stella), [Meta AI glasses developer policy context](https://www.meta.com/help/smart-glasses/).

| Item | Inventory |
|---|---|
| RGB camera | Ray-Ban Stories (1st gen): 2× 5 MP cameras, 1184×1184 video @ 30 fps, 2592×1944 photos. Ray-Ban Meta Gen 1/Gen 2: 1× 12 MP ultrawide camera (Sony IMX681 in brochure), 3024×4032 portrait photos; Gen 1 video 1440×1920 @ 30 fps up to ~60 s; Gen 2 adds 3K Ultra HD video, announced 1080p/3K modes and later HDR/60 fps features. |
| Depth sensor | None disclosed. |
| IMU | Not publicly exposed/disclosed in official consumer specs. Stabilization likely uses motion sensing but raw IMU access unknown. Mark as unknown for egress. |
| GPS | No built-in GPS disclosed; phone companion may provide location metadata depending app permissions. |
| Magnetometer | Unknown / not disclosed. |
| Eye tracking / gaze | None. |
| Hand tracking | None. |
| Audio | Stories: 3-mic array, open-ear speakers. Ray-Ban Meta: 5-mic array + 2 open-ear speakers in official/spec material; some 2026 reviews mention 6 mics for Gen 2 but official Ray-Ban/Meta materials found here say 5-mic. |
| Haptic | No haptic publicly emphasized; physical capture button, touchpad, voice. |
| Computation | On-device storage 32 GB for Ray-Ban Meta; Wi-Fi 6, Bluetooth 5.3. Chip/RAM not publicly disclosed on product pages. Phone app required for setup/import/cloud AI features. |
| Data egress / SDK | Consumer app workflow: photos/videos imported to phone, livestreaming to Meta platforms, Meta AI cloud features. No open raw sensor SDK found; no raw continuous camera/IMU stream access. |
| Battery under continuous capture | Gen 1 Ray-Ban Meta: up to ~4 h mixed use, case up to ~36 h in brochure. Gen 2: Meta announcement says up to 8 h typical use and case up to 48 h; continuous video much shorter and not disclosed as always-on capture. |
| Spatial localization | Weak. POV media + phone GPS metadata only. No public VIO/SLAM/depth/eye/hand streams. |
| Public datasets | No official public research datasets recorded with Ray-Ban Stories/Meta found; consumer videos/photos only. |

**WorldMM implication:** Useful as “life-log camera + audio events,” not as spatial-indexed memory without phone GPS, visual place recognition, or offline reconstruction. Treat location/pose as inferred, not sensor-native.

---

## 3. Apple Vision Pro (spatial computer / passthrough headset)

Sources: [Apple Vision Pro technical specifications, current M5 page](https://www.apple.com/apple-vision-pro/specs/), [Apple support tech specs for 2024 M2 model](https://support.apple.com/en-us/117810), [Apple Vision Pro overview/sensor privacy notes](https://www.apple.com/apple-vision-pro/), [visionOS enterprise/main camera access overview](https://developer.apple.com/visionos/).

| Item | Inventory |
|---|---|
| RGB camera | Stereoscopic 3D main camera system; spatial photo/video capture; 18 mm, f/2.00; 6.5 stereo megapixels. Apple lists 2 high-resolution main cameras + 6 world-facing tracking cameras. Public specs do not disclose all tracking camera resolutions/fps. |
| Depth sensor | LiDAR Scanner + TrueDepth camera. Range/frequency not publicly specified in product specs. |
| IMU | 4 inertial measurement units. Axis/rate not disclosed. |
| GPS | No built-in GPS disclosed; Wi-Fi location possible via OS, not headset GNSS. |
| Magnetometer | Not listed in Apple specs. |
| Eye tracking / gaze | 4 eye-tracking cameras; IR LED pattern eye tracking for input and Optic ID. Raw gaze access restricted; apps typically receive input abstractions, not raw eye images. |
| Hand tracking | Yes. Apple lists hands/eyes/voice input; overview claims precise head/hand tracking, real-time 3D mapping, low-light hand tracking with IR flood illuminators. Current page claims hand tracking up to 90 Hz for games. |
| Audio | Six-mic array with directional beamforming; Spatial Audio with dynamic head tracking. |
| Haptic | No built-in controller haptics; headset has no hand-held controller by default. |
| Computation | 2024 model: Apple M2 + R1, 16 GB unified memory, 256/512 GB/1 TB storage. Current 2025 M5 page: M5 10-core CPU/GPU, 16-core Neural Engine, 16 GB unified memory; R1 chip, 12 ms photon-to-photon, 256 GB/s R1 bandwidth. |
| Data egress / SDK | visionOS apps get high-level spatial APIs, ARKit/world tracking, hand/eye interaction abstractions; Apple states camera/sensor data is processed at system level so individual apps do not need to see surroundings. Enterprise APIs may allow main-camera access under managed entitlement; raw full sensor stream generally unavailable. |
| Battery under continuous capture | Current page: up to 2.5 h general use, 3 h video; 2024 support page: up to 2 h general, 2.5 h video; can be used while charging. |
| Spatial localization | Strong indoor spatial computing: visual-inertial world tracking, LiDAR/TrueDepth 3D mapping, head/hand/eye tracking. No outdoor GPS-native trajectory. |
| Public datasets | No major official public raw Vision Pro sensor dataset found. Captured spatial videos/photos exist, but not raw sensor datasets. |

**WorldMM implication:** Best for room-anchored, object/hand/gaze memory inside a controlled app. Weak for always-worn life memory due headset form, battery, and privacy sandbox.

---

## 4. Snap Spectacles 4th gen AR (2021 creator/developer AR glasses)

Sources: [Snap Newsroom 2021 next-generation Spectacles announcement](https://newsroom.snap.com/introducing-the-next-generation-of-spectacles), [Spectacles developer site / Snap OS](https://www.spectacles.com/), [Lens Studio / Snap AR developer docs](https://developers.snap.com/lens-studio/).

| Item | Inventory |
|---|---|
| RGB camera | 2 RGB cameras. Resolution/fps not disclosed in the announcement. |
| Depth sensor | None disclosed as dedicated depth; tracking via Snap Spatial Engine. |
| IMU | Not itemized publicly; 6DoF AR implies inertial/visual tracking, but axes/rate unknown. |
| GPS | Not disclosed. |
| Magnetometer | Not disclosed. |
| Eye tracking / gaze | Not disclosed. |
| Hand tracking | Yes. Snap Spatial Engine supports 6DoF, hand, marker, and surface tracking. |
| Audio | 4 built-in microphones, 2 stereo speakers. |
| Haptic / controls | Touchpad; voice/gesture/touch in broader Spectacles/Snap OS context. |
| Computation | Qualcomm Snapdragon XR1 platform; weight 134 g. RAM/storage not disclosed in announcement. |
| Data egress / SDK | Lens Studio integration; creators can wirelessly push Lenses to Spectacles for real-time testing. Public raw stream export not found. |
| Battery under continuous capture | Approximately 30 minutes per charge. |
| Spatial localization | Medium/strong for AR: on-device 6DoF, hand/marker/surface tracking. No public global map/GPS. |
| Public datasets | No official public dataset recorded with Spectacles 4 found. |

**WorldMM implication:** Good prototype class for “head pose + hand/surface interaction,” but battery and raw-data access make continuous spatial memory hard.

---

## 5. Google Glass Enterprise Edition 2 (legacy enterprise)

Sources: [Google Glass Enterprise discontinuation FAQ](https://www.google.com/glass/start/), [Glass Enterprise developer downloads/system images](https://developers.google.com/glass-enterprise/downloads/system-images), [Glass Enterprise Edition 2 device specs references from Google/X reseller materials where still mirrored](https://support.google.com/glass-enterprise/), [Qualcomm XR1 platform reference](https://www.qualcomm.com/products/mobile/snapdragon/xr-vr-ar/snapdragon-xr1-platform).

| Item | Inventory |
|---|---|
| RGB camera | 1 camera, commonly listed as 8 MP, 80° DFOV, 720p video. Official support pages now mostly redirect to discontinuation FAQ; treat detailed camera specs as legacy/mirrored, not currently prominent. |
| Depth sensor | None. |
| IMU | Public legacy specs list accelerometer/gyroscope/magnetometer; axis/rates not disclosed. |
| GPS | Not built-in in common EE2 specs; phone/Wi-Fi location possible through Android stack, but built-in GNSS not confirmed. |
| Magnetometer | Yes in legacy specs, but rate unknown. |
| Eye tracking / gaze | None. |
| Hand tracking | None native. |
| Audio | 3 beam-forming microphones often listed; mono speaker / USB audio support; exact array details vary by document. |
| Haptic / controls | Touchpad, voice, head gestures; no advanced haptics. |
| Computation | Qualcomm Snapdragon XR1, Android Oreo/AOSP base; 3 GB LPDDR4, 32 GB eMMC commonly listed. Phone not required for enterprise apps, but cloud/enterprise backends common. |
| Data egress / SDK | Android device with camera APIs; system images available until support ended. Open compared with consumer glasses, but product discontinued. |
| Battery under continuous capture | Official current FAQ does not state continuous capture. Legacy specs often list ~820 mAh battery; real continuous video likely short and app-dependent. Mark unknown. |
| Spatial localization | Weak. No depth/SLAM/eye. Can do visual recognition or phone/Wi-Fi location. |
| Public datasets | No major public dataset recorded with Glass EE2 found in this pass. Older egocentric datasets used Google Glass-class devices, but not EE2 official. |

**WorldMM implication:** Baseline “monocular POV + Android app” platform. Useful when openness matters more than spatial richness.

---

## 6. Microsoft HoloLens 2 (enterprise MR; discontinued/sunset path)

Sources: [Microsoft HoloLens 2 hardware](https://learn.microsoft.com/en-us/hololens/hololens2-hardware), [Microsoft HoloLens 2 hardware display/details docs](https://learn.microsoft.com/en-us/hololens/), [Microsoft Research Mode for HoloLens](https://learn.microsoft.com/en-us/windows/mixed-reality/develop/advanced-concepts/research-mode), [HoloLens 2 development / sensor APIs](https://learn.microsoft.com/en-us/windows/mixed-reality/develop/).

| Item | Inventory |
|---|---|
| RGB camera | 1 8 MP still camera; 1080p30 video commonly listed in Microsoft specs; world-understanding cameras are separate. |
| Depth sensor | 1-MP time-of-flight depth sensor in HoloLens 2 public specs; used for hand tracking/spatial mapping. Public range/fps not specified on main hardware page. |
| Tracking cameras | 4 visible-light head-tracking cameras commonly listed. |
| IMU | Accelerometer, gyroscope, magnetometer commonly listed; rates not disclosed in main docs. |
| GPS | No built-in GPS disclosed. |
| Magnetometer | Yes. |
| Eye tracking / gaze | Yes, 2 IR cameras for eye tracking; gaze APIs available. |
| Hand tracking | Yes, fully articulated hand tracking. |
| Audio | 5-channel microphone array; spatial sound speakers. |
| Haptic | No handheld controllers by default; interactions via hands/voice/gaze. |
| Computation | Qualcomm Snapdragon 850 compute platform + second-generation custom Holographic Processing Unit (HPU 2.0); 4 GB RAM, 64 GB UFS commonly listed. Untethered Windows Holographic OS. |
| Data egress / SDK | Mature MRTK/OpenXR/Unity/Unreal/Windows APIs. Research Mode exposes raw sensor streams for research but with caveats. Spatial mapping/world anchors available. |
| Battery under continuous capture | Up to ~2–3 h active use commonly listed; exact continuous capture unknown. |
| Spatial localization | Strong indoor MR: VIO/spatial mapping, depth, hand/eye/head tracking, world anchors. No GPS-native outdoor memory. |
| Public datasets | HoloLens research datasets exist in academia, but no single official public raw HoloLens 2 life-logging dataset analogous to Aria/ADT found here. |

**WorldMM implication:** Strong validation headset for indoor spatial memory and object/hand interactions. Less relevant for wearable daily life due bulk/battery/discontinuation.

---

## 7. EgoLife project hardware (arXiv:2503.03803)

Sources: [EgoLife arXiv abstract/page](https://arxiv.org/abs/2503.03803), [EgoLife project page](https://egolife-ai.github.io/), [EgoLife blog / paper HTML](https://huggingface.co/papers/2503.03803), [EgoLife GitHub](https://github.com/EvolvingLMMs-Lab/EgoLife), [EgoLife Hugging Face collection](https://huggingface.co/collections/lmms-lab/egolife-67c04574c2a9b64ab312c342).

| Item | Inventory |
|---|---|
| Glass hardware | Paper/project page state participants wear **Meta Aria glasses**. No separate custom EgoLife glasses hardware found. |
| RGB camera | Inherits Project Aria RGB/SLAM camera capabilities. EgoLife site says first-person view glasses record video, gaze, IMU data. |
| Depth sensor | No onboard Aria hardware depth; EgoLife augments with house 3D scans, synchronized third-person cameras, and mmWave devices. |
| IMU | Yes, from Meta Aria. |
| GPS | Likely from Meta Aria if enabled, but EgoLife indoor house setting emphasizes synchronized egocentric videos, gaze, IMU; GPS not highlighted. Mark unknown for released streams unless dataset docs confirm. |
| Magnetometer | Available on Aria, release inclusion unknown. |
| Eye tracking / gaze | Yes. Project page says glasses record gaze; paper figure says each participant wears Meta Aria recording ~8 h/day. |
| Hand tracking | Potential via Aria MPS, but EgoLife public summary emphasizes video/gaze/IMU/audio/annotations; hand tracking release unknown. |
| Audio | Egocentric video/audio implied; paper abstract emphasizes visual-audio models and conversations; Aria has 7-mic array. |
| Haptic/display | Not relevant; capture setup. |
| Computation | Data collection device is Meta Aria; model system EgoBulter/EgoGPT/EgoRAG offline/cloud training. |
| Data egress / SDK | Dataset released on Hugging Face, code on GitHub. Exact raw VRS vs processed video release needs dataset file inspection; public pages confirm 300 h egocentric/interpersonal/multiview/multimodal daily life data. |
| Battery under continuous capture | Project records approximately 8 h/day per participant; likely segmented/recharged operationally. Aria Gen 1 continuous capture is ~1.5 h per battery profile; paper does not disclose operational battery protocol in fetched pages. |
| Spatial localization | Strong potential because Aria + synchronized 15 GoPros + 2 mmWave devices + 3D house/participant scans. Public pages do not state released 6DoF trajectory format in fetched summary. |
| Public datasets | EgoLife Dataset (~300 h), EgoLifeQA (~3K long-context QA), EgoIT-99K, EgoGPT/EgoRAG code. |

**WorldMM implication:** EgoLife is the most relevant daily-life benchmark, but hardware answer is “Meta Aria plus environmental instrumentation,” not a new consumer glasses sensor stack.

---

## 8. 2025-announced / current AI-glasses class

Sources: [Solos AirGo / AirGo V product collection](https://solosglasses.com/collections/airgo%E2%84%A2-v-smartglasses), [Solos home](https://solosglasses.com/), [Even Realities G1/G2 product page](https://www.evenrealities.com/g1), [Even G1 product configurator](https://www.evenrealities.com/products/g1-a), [Brilliant Labs home](https://brilliant.xyz/), [Brilliant Frame hardware docs](https://docs.brilliant.xyz/frame/hardware/), [Brilliant developer docs](https://docs.brilliant.xyz/), [Halliday product page](https://hallidayglobal.com/products/halliday-glasses), [Halliday how-it-works page](https://hallidayglobal.com/pages/halliday-glasses), [Looktech product page](https://www.looktech.ai/pages/product).

### 8.1 Solos AirGo Vision / AirGo V

| Item | Inventory |
|---|---|
| RGB camera | AirGo V / Vision class has camera for visual AI; current collection pages found did not expose stable detailed camera resolution/fps in fetched text. Mark unknown; some marketing names imply vision capture. |
| Depth / SLAM / eye / hand | None disclosed. |
| IMU / GPS / magnetometer | Not disclosed. |
| Audio | Smart audio/assistant glasses with microphones and open-ear speakers; exact mic count from fetched official text unknown. |
| Computation | Phone/AI-assistant offload; on-device chip/RAM not disclosed in fetched official pages. |
| Data egress | Consumer app/assistant workflow; open raw SDK not found. |
| Battery | Unknown from fetched page. |
| Spatial localization | None beyond possible phone GPS/image recognition. |
| Dataset | None found. |

### 8.2 Even Realities G1 / G2

| Item | Inventory |
|---|---|
| RGB camera | None disclosed / product is display-first. |
| Depth / SLAM / eye / hand | None disclosed. |
| IMU / GPS / magnetometer | Not disclosed in fetched official pages. |
| Audio | G1 is primarily display/notification/teleprompter/translation class; mic/speaker details not clearly exposed in fetched text. Mark unknown. |
| Computation | Phone companion likely; chip/RAM not disclosed. |
| Data egress | Consumer app; no raw sensor SDK found. |
| Battery | Official fetched text did not expose battery details; unknown here. |
| Spatial localization | None disclosed. |
| Dataset | None found. |

### 8.3 Brilliant Labs Frame

| Item | Inventory |
|---|---|
| RGB camera | 720p low-power color camera. |
| Depth sensor | None. |
| IMU | 3-axis accelerometer with tap detection. No gyro disclosed in Frame hardware key features. |
| GPS | No. |
| Magnetometer | 3-axis e-compass. |
| Eye tracking / gaze | None. |
| Hand tracking | None native; camera can support CV apps. |
| Audio | Microphone. Speaker not emphasized for Frame in hardware key features. |
| Haptic / display | 640×400 color OLED, 20° FOV optic. |
| Computation | FPGA acceleration for graphics/imaging; Lua OS; Bluetooth 5.3; 210 mAh built-in Li-ion; charging dock with USB-C + 140 mAh battery. Full MCU/RAM details available in hardware docs but not summarized here. |
| Data egress / SDK | Strong openness: Frame SDK, Python/Flutter/Lua, Bluetooth spec, hardware docs. Likely best hobbyist raw access among consumer-style glasses. |
| Battery | 210 mAh; continuous capture/runtime not specified in key features. |
| Spatial localization | Weak. Camera + accel + compass can support visual place recognition and heading, not metric SLAM without custom CV and no gyro/depth. |
| Dataset | None official found. |

### 8.4 Halliday glasses

| Item | Inventory |
|---|---|
| RGB camera | None clearly disclosed in fetched official product/how-it-works pages; Halliday is “invisible display/proactive AI” class. |
| Depth / SLAM / eye / hand | None disclosed. |
| IMU / GPS / magnetometer | Not disclosed. |
| Audio | AI assistant implies microphone/audio path; exact mic count unknown. |
| Haptic / display | “DigiWindow” / invisible near-eye display; exact resolution/FOV not in fetched text. |
| Computation | Unknown; likely phone/cloud offload. |
| Data egress | Consumer app; no raw SDK found. |
| Battery | Unknown from fetched official text. |
| Spatial localization | None disclosed. |
| Dataset | None found. |

### 8.5 Looktech AI Glasses

| Item | Inventory |
|---|---|
| RGB camera | Official product page claims 13 MP camera and 4K POV capture. Fps unknown. |
| Depth / SLAM / eye / hand | None disclosed. |
| IMU / GPS / magnetometer | Not disclosed. |
| Audio | Open-ear audio; voice assistant. Mic count unknown in fetched text. |
| Haptic / display | No display emphasized in fetched product page. |
| Computation | GPT-5-branded cloud/assistant workflow; chip/RAM unknown. |
| Data egress | Consumer app/cloud assistant; raw stream SDK not found. |
| Battery | Unknown from fetched product page. |
| Spatial localization | None beyond phone/location metadata or image-based inference. |
| Dataset | None found. |

**WorldMM implication for 2025 class:** These are mostly “camera/audio/display assistant” products, not spatial sensors. Brilliant Frame is interesting because SDK/hardware openness exists; Looktech/Ray-Ban class is interesting for POV capture; Even/Halliday are display-first and weak for spatial indexing unless paired with phone/location/context.

---

## Cross-platform sensor fit for WorldMM spatial axis

| Signal WorldMM wants | Best platforms | Usable fallback | Notes |
|---|---|---|---|
| Egocentric RGB evidence | Aria, Ray-Ban Meta, Vision Pro, Snap, HoloLens, Frame, Looktech | Glass EE2, Solos | Need timestamped frames and privacy-safe sampling. Consumer glasses often only recordings. |
| 6DoF head trajectory | Aria, Vision Pro, HoloLens, Snap | Custom SfM/VPR from RGB | Critical for room/object spatial memory. Aria best raw export; Apple/Microsoft best runtime APIs. |
| Metric room geometry / depth | Vision Pro, HoloLens | Aria ADT derived depth / offline reconstruction | No depth on most consumer glasses. |
| Eye gaze / attention | Aria, Vision Pro, HoloLens | none | Most consumer AI glasses lack eye tracking. |
| Hand-object interaction | Aria Gen 2/MPS, Vision Pro, HoloLens, Snap | RGB action recognition | Requires hands in FOV + tracking. Consumer POV camera can infer but not robust. |
| Outdoor/global location | Aria GNSS | phone GPS metadata | Most glasses lack GPS. |
| Spatial audio / conversation | Aria, Vision Pro, HoloLens, Ray-Ban Meta | single mic transcription | Mic arrays enable direction-of-arrival / speaker localization if raw multichannel accessible. |
| Open raw data / SDK | Aria, HoloLens Research Mode, Brilliant Frame, Glass EE2 | Snap Lens Studio high-level | Ray-Ban/Even/Halliday/Looktech mostly closed consumer flows. |
| Continuous all-day capture | Aria Gen 2 claimed 6–8 h, EgoLife operational dataset | Ray-Ban typical use, phone-tethered sampling | Battery/thermal/privacy remain core blocker. |

---

## Spatial-axis signals you can ONLY get from AI Glasses (not regular phone or third-person camera)

1. **Gaze-anchored object salience** — “user looked at _mug_ for 1.2 s while asking where it was.” Phones see scene from hand position; third-person sees body, not foveal attention. Needs Aria/Vision Pro/HoloLens eye gaze.
2. **Head-pose-anchored room memory** — continuous 6DoF pose from wearer’s actual viewpoint. Enables “left of where I was standing at sink,” not just GPS or third-person coordinates.
3. **Egocentric hand-object contact** — hands enter first-person frame at manipulation moment. Better for “picked up keys,” “opened fridge,” “held medicine bottle” than static room cameras.
4. **Attention-weighted object permanence** — object becomes memory-worthy when seen + fixated + manipulated + later absent. Requires gaze/head/hand fusion.
5. **Conversation partner direction and turn-taking** — mic arrays on head capture wearer-centered audio direction; combine with gaze/head turns to identify who was addressed.
6. **First-person deictic grounding** — “this one,” “over there,” “behind me” can map to gaze ray/head ray at utterance time. Phones may be pocketed; third-person lacks speaker POV.
7. **Continuous dwelling path** — room-to-room path, stops, revisits, and dwell time from the human perspective. Phone GPS fails indoors; third-person cameras fragment across rooms.
8. **Personal affordance map** — shelves, handles, labels, displays seen at wearer height and reachability. Third-person maps geometry, not user-action affordances.
9. **Privacy-preserving selective memory triggers** — glasses can sample only when gaze/voice/hand events cross thresholds, avoiding full always-on recording. Needs on-device sensors.
10. **Embodied uncertainty signals** — head scanning, repeated gaze, hesitation, backtracking indicate “user searching/confused.” Hard from phone logs.
11. **Wearer-specific object salience over time** — same object gets different weights by user: looked at daily, ignored, searched for, discussed. AI glasses make salience personalized.
12. **Joint egocentric multi-user spatial memory** — multiple glasses in shared space can align trajectories and gaze/utterances (“Alice pointed to cup while Bob looked at shelf”), as Aria Pilot/ADT/EgoLife-style captures show.

## Recommended wiring priority for WorldMM

1. **Tier 0 now:** POV RGB keyframes + audio transcript + phone/OS coarse location + timestamp. Works on Ray-Ban/Looktech/Solos/Glass-class devices.
2. **Tier 1 research-grade:** Aria VRS/MPS: RGB, SLAM images, 6DoF trajectory, gaze, hand tracking, GPS/GNSS, mic array. Best fit for proving spatial axis.
3. **Tier 2 indoor MR validation:** Vision Pro/HoloLens: runtime world anchors, hand/gaze events, room mesh/depth. Use for deterministic room-memory demos.
4. **Tier 3 open tinkering:** Brilliant Frame: camera + compass + accelerometer + SDK; useful for low-cost prototypes, not metric spatial memory.
5. **Avoid over-investing until disclosed:** display-only or closed AI glasses with no camera/raw streams/GPS/SLAM. They help UI recall, not memory grounding.

