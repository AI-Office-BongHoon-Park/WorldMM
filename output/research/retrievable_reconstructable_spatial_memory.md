# Retrievable + Reconstructable Spatial Memory Systems

Date: 2026-05-22 KST  
Scope: systems where memory retrieval returns either a 3D state directly, or enough renderable/reconstructable latent state to synthesize visual/3D evidence at recall time. Pure encoders excluded unless they are useful near-misses.

## Executive takeaways

WorldMM's proposed extension asks for a **fourth spatial axis**: `(place, time, semantic query) -> text triple + decodable 3D scene latent`. The closest existing pattern is not one single paper; it is a composite:

1. **RenderMem / GSMem**: retrieval is a rendering operation over a persistent 3D scene state. Best fit for “what did X look like from viewpoint/time Y?”
2. **LangSplat / LERF / OpenScene**: language features are bound to 3D coordinates or Gaussians, so text retrieval selects 3D regions and can render feature/relevancy maps.
3. **3D-Mem / 3DLLM-Mem**: embodied-agent memory management: snapshots/tokens grow over time, are retrieved for reasoning, and are consolidated/filtered.

Major gap: almost none provides **time-indexed, closed-vocab symbolic triples + compressed decodable 3D latent in one retrieval unit**. Most systems store either semantic features in 3D, or renderable 3D, or episodic agent tokens—not all three with versioned temporal recall.

---

## Survey matrix

| System | Memory representation | Retrieval interface | Reconstruction output | Retrieval unit vectorized vs raw | Update / consolidation | Main limitations |
|---|---|---|---|---|---|---|
| 3DLLM-Mem | working 3D tokens + episodic key/value 3D tokens | current 3D tokens attend to past memory | unclear: feature fusion, not released decoder | vectorized dense 3D features; raw observations pre-stored for loading | working -> episodic transfer; environment-level replacement | no public code beyond placeholder; no explicit point-cloud/splat decode |
| RenderMem | persistent renderable 3D scene state | natural language -> viewpoint-conditioned render decision/spec | rendered images from query-implied pose | raw/renderable geometry stays backend; query rendered as image evidence | underlying map evolves; rendering reads latest state | paper-level framework; backend agnostic; no public code found |
| GSMem | 3D Gaussian Splatting + scene graph + language field | object-level + semantic-level ROI retrieval | novel-view RGB render from 3DGS | Gaussians raw/renderable; CLIP/language field vectorized | lifelong exploration updates GS + graph/language fields | 2026 preprint; code not found; fidelity depends on 3DGS mapping |
| 3D-Mem | TSDF/semantic map + memory snapshots + frontier snapshots | VLM chooses snapshot/frontier after prefilter | stored snapshot images + frontier images; 3D map supports exploration | snapshot image/raw observation + object lists; image features for frontiers | incremental TSDF/frontier/snapshot clustering; prefilter by relevant categories | not a decodable latent; mostly image snapshots over 3D map |
| LERF | NeRF/radiance field with multi-scale CLIP/DINO fields | text phrase -> CLIP relevancy over rendered rays | RGB/depth + relevancy render | vectorized CLIP/DINO fields in neural volume; raw RGB via NeRF | trained per scene; static after optimization | expensive per-scene training; no temporal memory or symbolic triples |
| LangSplat | 3D Gaussians + spherical harmonics + language features | open-vocabulary query over rendered language feature image | RGB render + language-feature image | per-Gaussian raw params + vectorized language feature | trained per scene; feature checkpoint captured/restored | static scene; not an episodic memory manager |
| OpenScene | 3D point cloud + multi-view fused 2D features / distilled 3D features | arbitrary text phrase highlights 3D regions | highlighted 3D point cloud/segmentation, not photoreal render | per-point fused feature vectors; point cloud raw | offline preprocessing/fusion/distillation | no temporal update; no reconstruction beyond existing point cloud |
| Neural Scene Graphs | graph of object/background radiance fields | object/node selection + render config | novel views and object compositions | neural radiance-field weights per object/background | optimized per scene/sequence | dynamic driving scenes; not language retrieval |
| SceneGraphFusion | incremental 3D scene graph from RGB-D + mesh/rendered views | graph/object/relation query; graph prediction | online rendered views from mesh; graph + semantic labels | graph nodes/relations vectorized by network; mesh/raw RGB-D separate | incremental RGB-D fusion + graph prediction | closed classes/relations; not neural latent; render is mesh-view support |
| Project Aria MPS / Replay | VRS time stream + trajectory + semidense point cloud + observations | timestamp/camera query; visible point lookup | semidense point cloud / 3D visualization, replayed sensor frames | raw VRS frames + trajectory/points; IDs link observations | MPS batch processing; timestamp-indexed retrieval | not semantic memory; no language retrieval by default |
| Habitat 3.0 | simulator scene + agent/humanoid state | programmatic simulator queries | rendered RGB-D/semantic frames | raw simulation state | simulator time evolves by physics | no documented persistent spatial memory API |
| Spatially-Aware Transformer | place-centric transformer episodic memory | spatial/temporal transformer attention | generated images in toy MiniWorld tasks | vectorized frame/place embeddings | FIFO/place memory + adaptive allocator | not real 3D reconstruction; useful allocator idea only |

---

## 1. 3DLLM-Mem — dense 3D token memory, but decoder unclear

**Evidence.** Paper: [arXiv:2505.22657](https://arxiv.org/abs/2505.22657). Public GitHub repo currently contains only placeholder README, not implementation ([GitHub permalink](https://github.com/3DLLM-Mem/3DLLM-Mem/blob/703c8e7a27ecb33ae49728cfbb4b7759bd5cd0a8/README.md#L1-L16)).

**Memory representation.** Paper describes a cognitive split: current observation `X[t=i]` as working memory and past observations `X[t=1:T]` as episodic memory. The memory bank stores key/value features `f^K`, `f^V` over timesteps and memory tokens. This is **dense 3D feature memory**, not closed-vocab triples.

**Retrieval interface.** Current working-memory 3D features are encoded into query features `f_t^Q`; retrieval is memory-query attention over episodic keys/values. User-level queries are task prompts for embodied QA/planning, but the memory read itself is feature attention.

**Reconstruction output.** Unclear. Paper emphasizes dense 3D representations and fusion into `f^M`; it does not expose a decoder that reconstructs point clouds or Gaussian splats at recall. Treat as **retrievable 3D memory**, not proven reconstructable memory.

**Embedded per retrieval unit.** Retrieval unit appears to be per timestep × memory token: vectorized `N x M` features. Raw/precomputed observations are stored locally for training/data loading, but not packaged as a recall output in the released repo.

**Update / consolidation.** Current environment state remains in working memory; when agent moves to a new environment, previous working memory transfers to episodic memory. If an environment already exists, it is updated to reflect latest state.

**Failure modes.** No public implementation; memory scale grows with timesteps/tokens; retrieval is learned attention, hard to inspect; no decodable latent guaranteed. For WorldMM, 3DLLM-Mem is useful for **fusion architecture**, not enough for “return reconstructed kitchen point cloud.”

---

## 2. RenderMem — rendering is memory retrieval

**Evidence.** Paper: [arXiv:2603.14669](https://arxiv.org/abs/2603.14669). It states the core design explicitly: “rendering is the read operation of 3D memory,” and query-conditioned visual evidence is rendered from a persistent 3D scene state. I found no official GitHub code; mark implementation details as paper-only.

**Memory representation.** RenderMem stores a **persistent renderable 3D scene representation**. It is backend-agnostic: mesh, NeRF, Gaussian splat, TSDF, or simulator state could serve as the memory substrate. It does not store fixed observations as primary memory.

**Retrieval interface.** Natural-language question is first classified for whether rendering is needed. If yes, the model produces a rendering specification: viewpoint/camera pose/direction implied by query. Example pattern: “What is behind the fridge?” -> render from a viewpoint looking behind/around fridge.

**Reconstruction output.** The output is not a point cloud by default; it is **query-conditioned rendered image evidence**. The rendered image is passed to an off-the-shelf VLM for final answer.

**Embedded per retrieval unit.** Query is text; memory unit is the global renderable scene. Vectorized representation lives in the VLM and possibly the reconstruction backend; raw geometry/appearance stay in the scene state.

**Update / consolidation.** Paper says when underlying scene representation evolves, RenderMem naturally adapts because readout renders the current map. No explicit episodic consolidation.

**Failure modes.** No official code found; temporal recall (“DAY1 11:14:30”) not central; render quality and correctness depend entirely on mapping backend. Still, this is the cleanest conceptual match for WorldMM’s “retrieve by rendering” step.

---

## 3. GSMem — 3DGS as persistent spatial memory + retrieval-rendering

**Evidence.** Paper: [arXiv:2603.19137](https://arxiv.org/abs/2603.19137). The abstract/method describe 3D Gaussian Splatting as persistent memory, with object-level scene graphs and semantic-level language fields for retrieval, followed by optimal viewpoint rendering.

**Memory representation.** Three-layer memory: (1) **3DGS** for geometry/appearance, (2) object-level scene graph for named entities/relations, (3) semantic language field for text-aligned regional retrieval.

**Retrieval interface.** Query is natural language. Retrieval localizes regions twice: object-level region retrieval from scene graph, and semantic-level retrieval from language-field similarity. The retrieved ROI then determines a best render viewpoint.

**Reconstruction output.** High-fidelity novel-view RGB render from 3DGS. It is not “decode latent to point cloud”; instead, the latent **is** renderable Gaussian memory.

**Embedded per retrieval unit.** Gaussians store raw/renderable spatial and appearance parameters. Language-field embeddings are vectorized; object graph stores symbolic object-level cues.

**Update / consolidation.** During exploration, the agent updates the 3DGS map and memory layers. The paper also combines task-aware VLM semantic scoring with 3DGS coverage objective.

**Failure modes.** No official code found by GitHub search; dynamic scenes and long temporal versioning remain unclear; reconstruction artifacts affect downstream VLM reasoning.

**WorldMM relevance.** Very high. GSMem already couples **semantic retrieval -> 3D ROI -> render -> VLM reasoning**. Missing piece: persistent time slices and triple alignment.

---

## 4. 3D-Mem — snapshot memory over 3D maps

**Evidence.** Official repo and paper links are in README ([GitHub](https://github.com/UMass-Embodied-AGI/3D-Mem/blob/f445e0828a2c5d5845ccdbd0992fc5eed871d19a/README.md#L17-L28), [arXiv:2411.17735](https://arxiv.org/abs/2411.17735)). Code defines frontiers with positions, image paths, and features; snapshots store image, observation point, object list, and clusters ([Frontier/SnapShot code](https://github.com/UMass-Embodied-AGI/3D-Mem/blob/f445e0828a2c5d5845ccdbd0992fc5eed871d19a/src/tsdf_planner.py#L24-L77)). Query pipeline sends snapshot objects/images, frontier images, egocentric views, and question to VLM ([query code](https://github.com/UMass-Embodied-AGI/3D-Mem/blob/f445e0828a2c5d5845ccdbd0992fc5eed871d19a/src/query_vlm_aeqa.py#L9-L48)).

**Memory representation.** TSDF/semantic map plus **Memory Snapshots** and **Frontier Snapshots**. The snapshot is an image-level view tied to a 3D observation point/object cluster, not a compressed full-scene latent.

**Retrieval interface.** Natural-language question + object map. VLM selects target `snapshot` or `frontier`; paper adds prefiltering by relevant object categories to reduce memory prompt size.

**Reconstruction output.** Retrieval returns stored images/frontier observations and navigation targets. The 3D map supports exploration; it does not decode a reconstructed point cloud for recall. However, it is a practical embodied-memory baseline for “retrieve scene evidence.”

**Embedded per retrieval unit.** Raw: snapshot image, all observations, object IDs. Vectorized: frontier image features; VLM sees images and object names.

**Update / consolidation.** Incremental construction during exploration; hierarchical clustering updates snapshots; frontiers are updated from occupancy/TSDF. Memory is consolidated by clustering and prefiltering.

**Failure modes.** Snapshot memory can miss unseen viewpoints; reconstructability is indirect; VLM prompt cost grows with snapshots; category prefilter can remove useful context.

---

## 5. LERF — language embedded radiance fields

**Evidence.** Repo: [kerrj/lerf](https://github.com/kerrj/lerf). Code renders CLIP/DINO features along rays and computes phrase relevancy ([relevancy loop](https://github.com/kerrj/lerf/blob/db08d578038d884542688511bd9ad7b489a65673/lerf/lerf/lerf.py#L70-L84)); inference outputs rendered `clip`, `dino`, `raw_relevancy`, and `best_scales` ([output code](https://github.com/kerrj/lerf/blob/db08d578038d884542688511bd9ad7b489a65673/lerf/lerf/lerf.py#L120-L140)). Paper: [Language Embedded Radiance Fields](https://arxiv.org/abs/2303.09553).

**Memory representation.** A trained NeRF/radiance field plus multi-scale language embeddings. This is a dense continuous 3D memory over one scene.

**Retrieval interface.** Natural-language phrase. Text embedding is compared to rendered CLIP field outputs; system returns relevancy maps over rendered views.

**Reconstruction output.** RGB/depth via NeRF, plus 2D relevancy maps for queried phrase. Mesh extraction possible in NeRF family but not LERF’s central recall output.

**Embedded per retrieval unit.** Neural field weights and hashgrid features are raw latent; CLIP/DINO outputs are vectorized. Retrieval unit is ray/image pixel/3D sample, not object or timestamp.

**Update / consolidation.** Offline per-scene optimization. No online consolidation.

**Failure modes.** Static scene; expensive training; ambiguous language can light up wrong regions; no symbolic text triples. Still important: it proves language queries can retrieve from a reconstructable 3D latent.

---

## 6. LangSplat — 3D language Gaussian splats

**Evidence.** Official repo: [minghanqin/LangSplat](https://github.com/minghanqin/LangSplat); paper: [arXiv:2312.16084](https://arxiv.org/abs/2312.16084). Gaussian model stores xyz, SH color features, scale, rotation, opacity, and optional language feature ([model fields](https://github.com/minghanqin/LangSplat/blob/d70edb86df0fcbda19dc0d9739a3e5140a5e65fc/scene/gaussian_model.py#L45-L54)); checkpoints can capture language features ([capture](https://github.com/minghanqin/LangSplat/blob/d70edb86df0fcbda19dc0d9739a3e5140a5e65fc/scene/gaussian_model.py#L63-L80)). Renderer normalizes per-Gaussian language features and rasterizes both RGB and `language_feature_image` ([renderer](https://github.com/minghanqin/LangSplat/blob/d70edb86df0fcbda19dc0d9739a3e5140a5e65fc/gaussian_renderer/__init__.py#L86-L112)).

**Memory representation.** 3D Gaussian Splatting scene with per-Gaussian language feature vectors. This is close to “compressed 3D latent + retrieval feature in one store.”

**Retrieval interface.** Open-vocabulary language query; query embedding compares against rendered feature image or per-Gaussian feature field.

**Reconstruction output.** RGB Gaussian-splat render and rendered language-feature map. Point/mesh extraction not primary.

**Embedded per retrieval unit.** Each Gaussian stores raw render parameters (`xyz`, opacity, scaling, rotation, SH color) plus vectorized language feature. This is a strong template for WorldMM retrieval units.

**Update / consolidation.** Offline scene optimization. Later works (4D LangSplat) add dynamic/temporal fields, but base LangSplat has no episodic update policy.

**Failure modes.** Static scenes; feature distillation errors; language field may not preserve fine instance identity; temporal versioning absent.

---

## 7. OpenScene — open-vocabulary 3D point-feature memory

**Evidence.** Official repo: [pengsongyou/openscene](https://github.com/pengsongyou/openscene). README states OpenScene performs 3D scene understanding with open-vocabulary queries ([README](https://github.com/pengsongyou/openscene/blob/0f369bc73d0724ae24b5e46bbada193f8ee9d193/README.md#L29-L29)); demo lets users type arbitrary phrases and highlights regions ([interactive demo](https://github.com/pengsongyou/openscene/blob/0f369bc73d0724ae24b5e46bbada193f8ee9d193/README.md#L69-L78)). It uses multi-view fused image features for each 3D point ([feature representation](https://github.com/pengsongyou/openscene/blob/0f369bc73d0724ae24b5e46bbada193f8ee9d193/README.md#L120-L167)). Paper: [arXiv:2211.15654](https://arxiv.org/abs/2211.15654).

**Memory representation.** 3D point cloud with per-point fused 2D features or distilled 3D features. This is a retrievable 3D semantic index.

**Retrieval interface.** Text phrase -> open-vocabulary segmentation/highlight over point cloud.

**Reconstruction output.** Existing point cloud with highlighted regions. It does not reconstruct photoreal RGB views, but it returns 3D geometry directly.

**Embedded per retrieval unit.** Raw point coordinates + vectorized OpenSeg/LSeg/fused features. Retrieval unit is a point or point cluster.

**Update / consolidation.** Offline preprocessing/fusion; no streaming consolidation.

**Failure modes.** Large feature storage; point cloud not time-aware; no raw image/appearance reconstruction; relies on 2D foundation model quality.

---

## 8. Neural Scene Graphs — graph-structured renderable neural fields

**Evidence.** Official repo says it “optimizes multiple radiance fields” for objects and static background and can render learned representations with novel object compositions/views ([README](https://github.com/princeton-computational-imaging/neural-scene-graphs/blob/8d3d9ce9064ded8231a1374c3866f004a4a281f8/README.md#L9-L11)). It provides pretrained-scene rendering commands ([render command](https://github.com/princeton-computational-imaging/neural-scene-graphs/blob/8d3d9ce9064ded8231a1374c3866f004a4a281f8/README.md#L37-L46)) and notes limitations on motion/camera regimes ([limitations](https://github.com/princeton-computational-imaging/neural-scene-graphs/blob/8d3d9ce9064ded8231a1374c3866f004a4a281f8/README.md#L49-L60)). Paper: [CVPR 2021 open access](https://openaccess.thecvf.com/content/CVPR2021/html/Ost_Neural_Scene_Graphs_for_Dynamic_Scenes_CVPR_2021_paper.html).

**Memory representation.** Scene graph whose nodes are neural radiance fields for objects/background, plus transforms over time.

**Retrieval interface.** Not natural-language by default. Retrieval is object/node/scene selection through graph/config, useful for object-centric recall.

**Reconstruction output.** Novel-view renders and object-composition renders.

**Embedded per retrieval unit.** Per-object NeRF weights and graph node transforms; raw images used for training. Unit is an object/background node.

**Update / consolidation.** Offline optimization from driving sequences; dynamic graph encodes moving objects. No lifelong append policy.

**Failure modes.** Not semantic-language memory; outdoor/driving assumptions; heavy optimization. For WorldMM, the useful idea is **object node -> renderable latent**, not retrieval UX.

---

## 9. SceneGraphFusion — incremental 3D scene graph with rendered views

**Evidence.** Official repo states framework and network split ([README](https://github.com/ShunChengWu/SceneGraphFusion/blob/5bf9017c00949aedca1430854240667b3fa06565/README.md#L1-L12)). It depends on Assimp for “loading meshes for online rendered view generation” ([mesh/render dependency](https://github.com/ShunChengWu/SceneGraphFusion/blob/5bf9017c00949aedca1430854240667b3fa06565/README.md#L31-L32)). Runtime can execute graph SLAM and GUI over 3RScan/ScanNet, generating rendered views online ([run section](https://github.com/ShunChengWu/SceneGraphFusion/blob/5bf9017c00949aedca1430854240667b3fa06565/README.md#L73-L85)). Paper: [SceneGraphFusion](https://arxiv.org/abs/2012.03651).

**Memory representation.** Incremental 3D scene graph from RGB-D sequences: objects, relationships, support predicates, semantic classes, and geometry/mesh context.

**Retrieval interface.** Graph/object/relation queries are natural; language not open-vocabulary. It can answer symbolic spatial queries if mapped to graph predicates.

**Reconstruction output.** Mesh/rendered views for online view generation, plus scene graph. Not neural photoreal reconstruction.

**Embedded per retrieval unit.** Object nodes and relation features vectorized for graph prediction; raw mesh/RGB-D remains separate.

**Update / consolidation.** Incremental RGB-D fusion and graph prediction over scan sequence.

**Failure modes.** Closed vocabulary (20 NYUv2 classes, 8 predicates per README area); hard to handle appearance recall like “what did kitchen look like?” unless mesh/texture available.

---

## 10. Project Aria MPS / Replay — industrial timestamped 3D recall substrate

**Evidence.** Project Aria tools docs say semi-dense point cloud is generated by MPS SLAM and stored as `semidense_points.csv.gz` plus `semidense_observations.csv.gz` ([docs](https://facebookresearch.github.io/projectaria_tools/docs/data_formats/mps/slam/mps_pointcloud)). Code sample shows querying point-cloud observations visible at a given time/camera ([sample README](https://github.com/facebookresearch/projectaria_tools/blob/1bfe7e50c52698583b2ee3bbb63c5c177d962311/examples/Gen1/python_samples/mps_semidense_point_visibility/Readme.md#L1-L6)); MPS point cloud consists of global 3D points and per-camera/timestamp visibility observations, with point IDs linking global points to observations ([sample details](https://github.com/facebookresearch/projectaria_tools/blob/1bfe7e50c52698583b2ee3bbb63c5c177d962311/examples/Gen1/python_samples/mps_semidense_point_visibility/Readme.md#L17-L36)).

**Memory representation.** Timestamped VRS sensor streams + closed-loop trajectory + semidense 3D points + per-frame observations.

**Retrieval interface.** Time/camera query, not language by default. Can retrieve frames and visible point tracks at timestamp.

**Reconstruction output.** Semidense point cloud / 3D visualization / replayed frames.

**Embedded per retrieval unit.** Raw sensor frames and point cloud; IDs and timestamps are structured indexes. No semantic vector embedding unless added externally.

**Update / consolidation.** Batch MPS processing after recording; streaming update not default.

**Failure modes.** Sparse/semidense, not complete kitchen splat; no language retrieval; privacy and calibration complexity. For WorldMM, Aria is best industrial substrate for **time-addressable 3D recall**.

---

## Near-misses requested by name

### Habitat 3.0 spatial memory
Habitat 3.0 is a simulator for humanoid/robot collaboration and can render RGB-D/semantic observations from simulator state ([site](https://aihabitat.org/habitat3/), [GitHub](https://github.com/facebookresearch/habitat-lab)). I did not find a documented “Habitat 3.0 spatial memory” module that persists retrieval + reconstruction as a memory system. Treat Habitat as environment/simulator substrate, not memory architecture.

### Scene Memory Transformer / EmbodiedMemoryNet
Scene Memory Transformer stores embedded observations and uses attention for long-horizon embodied policies; GitHub implementation exists ([repo](https://github.com/XZT008/Scene-memory-transformer)), paper [arXiv:1903.03878](https://arxiv.org/abs/1903.03878). It retrieves via attention, but does not reconstruct 3D. EmbodiedMemoryNet as named was not clearly found; likely related literature needs disambiguation.

### Spatially-Aware Transformer Memory
SAT is relevant to the “4th axis” idea because it adds place-centric memory and adaptive allocation. Official repo states it implements Spatially-Aware Transformers ([README](https://github.com/junmokane/spatially-aware-transformer/blob/5ed6f9f191d7d9d7484bfaf678e18daef68b01f6/README.md#L1-L4)) and uses Room Ballet datasets with agent trajectories plus video/dancer state ([dataset lines](https://github.com/junmokane/spatially-aware-transformer/blob/5ed6f9f191d7d9d7484bfaf678e18daef68b01f6/README.md#L15-L28)). It includes generation experiments, but not real reconstructable 3D scene memory. Useful only for **memory allocation policy**.

### MyLifeBits / SenseCam followups with 3D
Classic lifelogging systems are retrieval-heavy but not reconstructable 3D memory systems. I found no strong evidence for a MyLifeBits/SenseCam followup that adds persistent 3D scene reconstruction comparable to Aria MPS or 3DGS. Mark as unclear/negative.

### Memorizing Transformers + 3D hybrids
General memorizing transformers retrieve external key/value vectors; SAT and 3DLLM-Mem are closer embodied variants. I found no mature system that couples memorizing-transformer retrieval directly to decodable 3D latents and temporal scene reconstruction.

---

## Closest 3 systems to WorldMM's “4th spatial axis” pattern

### 1. GSMem
Minimal lift if WorldMM can add a 3DGS backend. It already has:
- persistent renderable spatial memory,
- semantic language-field retrieval,
- object-level scene graph fallback,
- query-conditioned ROI rendering.

Needed lift: add timestamped memory versions and align each Gaussian/ROI with WorldMM triples.

### 2. LangSplat
Best low-level data structure. Per-Gaussian `(xyz, opacity, scale, rotation, color SH, language_feature)` maps almost directly to “compressed scene latent + retrieval embedding.” Renderer already returns both RGB and language feature image. Needed lift: incremental streaming + temporal snapshots + symbolic triple index.

### 3. 3D-Mem / 3DLLM-Mem hybrid
3D-Mem gives practical embodied update, snapshots, frontier exploration, and VLM retrieval. 3DLLM-Mem gives learned working/episodic fusion over dense 3D tokens. Combine them with LangSplat/GSMem as decoder and WorldMM gets a viable architecture: triples retrieve candidate time/place, dense token memory reranks/fuses, 3DGS renders recall.

---

## Conceptual gap for WorldMM

None of the surveyed systems gives all requirements together:

1. **Temporal versioned 3D latent store.** Aria has timestamps but no language memory; LangSplat/GSMem have renderable latents but weak temporal versioning.
2. **Dual symbolic + reconstructable recall.** Scene graphs have symbols but weak appearance reconstruction; 3DGS/NeRF have appearance but weak closed-vocab triples.
3. **Unified retrieval unit.** WorldMM needs one unit like:
   ```text
   MemoryCell = {
     agent_id, place_id, time_span,
     text_triples[],
     visual_key_embedding,
     spatial_key_embedding,
     compressed_3d_latent_ref,   # Gaussian/point/mesh chunk
     decoder_backend,
     provenance_frames,
     confidence/change_state
   }
   ```
4. **Change-aware consolidation.** Existing systems update maps or append tokens, but do not track “kitchen at DAY1 11:14:30” vs “same kitchen after object moved” as first-class versions.
5. **Recall contract.** WorldMM uniquely needs API-level guarantee: a query returns both symbolic evidence and reconstructable visual/3D evidence, with provenance and uncertainty.

Recommended architecture: use WorldMM triples as the **index spine**, attach LangSplat/GSMem-style Gaussian chunks as **decodable payloads**, and use 3D-Mem/3DLLM-Mem-style retrieval/fusion to select relevant `(place, time, latent)` cells. RenderMem then becomes the read operator: `retrieve -> decode/render -> answer`.
