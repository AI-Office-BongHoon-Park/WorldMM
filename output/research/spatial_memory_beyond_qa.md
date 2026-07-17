# Spatial Memory Beyond QA: High-Impact AI-Glasses Scenarios

_Date: 2026-05-21 KST_

## Thesis

WorldMM currently demonstrates spatial memory through MCQ-style question answering. The larger opportunity is not "answer a question about a scene" but **make space the primary index into lived memory**. If an AI-glasses system continuously binds objects, people, actions, time, gaze, and user intent to a persistent 3D/semantic map, then recall becomes context-perfect: returning to a counter, doorway, workstation, or cafe automatically retrieves the right slice of the user's past.

The strongest precedents come from egocentric lifelogging and embodied AI: Microsoft SenseCam showed that passive first-person images can cue autobiographical memory; Ego4D formalized episodic-memory questions such as what happened, where, and when in daily-life video; HoloAssist showed that headset-captured RGB/depth/gaze/hand streams support real-time physical-task guidance; recent 2026 embodied-memory papers such as CAST, RenderMem, HIMM, and Pro^2Assist explicitly model memory as time/place/action scenes or spatial-temporal representations. The gap: most systems still surface this as QA or benchmark retrieval. AI glasses can turn the same substrate into **ambient, proactive, place-triggered assistance**.

## Evidence base used

- Microsoft Research describes SenseCam as a wearable camera whose reviewed images "tend to elicit quite vivid remembering" and notes extensive clinical/lifelogging follow-up work ([Microsoft Research SenseCam](https://www.microsoft.com/en-us/research/project/sensecam/)).
- The SenseCam memory-aid paper reports that reviewing automatically captured images improved recall in memory-impaired users ([Hodges et al., 2011](https://www.tandfonline.com/doi/abs/10.1080/09658211.2011.605591); earlier UbiComp paper PDF: [SenseCam: A Retrospective Memory Aid](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/09/sensecam-ubicomp-2006-camera-ready.pdf)).
- Ego4D is a 3,670-hour egocentric daily-life video dataset with benchmarks including episodic memory over household, workplace, outdoor, and leisure activities; privacy/ethics are central to collection ([Ego4D site](https://ego4d-data.org/), [CVPR 2022 paper](https://openaccess.thecvf.com/content/CVPR2022/html/Grauman_Ego4D_Around_the_World_in_3000_Hours_of_Egocentric_Video_CVPR_2022_paper.html)).
- HoloAssist captures mixed-reality headset data streams while people complete physical manipulation tasks; it includes RGB, depth, hands, gaze, head pose, IMU, action correctness, and interventions ([HoloAssist site](https://holoassist.github.io/), [ICCV 2023 paper](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_HoloAssist_an_Egocentric_Human_Interaction_Dataset_for_Interactive_AI_Assistants_ICCV_2023_paper.html)).
- Pro^2Assist and PROASSIST show streaming egocentric video assistants that maintain long-horizon procedural progress and decide when to proactively intervene ([Pro^2Assist, 2026](https://arxiv.org/html/2605.04227v1), [PROASSIST, 2025](https://arxiv.org/html/2506.05904)).
- CAST explicitly organizes episodic agent memory as scenes indexed by time, place, topic, action, and characters ([CAST, 2026](https://arxiv.org/html/2602.06051v2)).
- RenderMem treats rendering as the read operation over persistent 3D scene memory, useful for viewpoint/occlusion queries ([RenderMem, 2026](https://arxiv.org/html/2603.14669v1)).
- 3DLLM-Mem stores long-term 3D spatial-temporal memory for embodied planning/action ([3DLLM-Mem, 2025](https://arxiv.org/html/2505.22657)).
- HIMM separates semantic memory and physical space while preserving camera pose, object regions, and occupancy maps for embodied exploration/navigation ([HIMM, 2026](https://arxiv.org/html/2602.15513)).
- Clew is a deployed accessibility app that records a 3D route and camera-based environment map for later route following, showing route memory as a practical spatial-assistive primitive ([Clew privacy/route description](https://clewapp.org/)).

---

## 1. Memory aid for elderly / MCI / dementia

**Problem framing.** People with mild cognitive impairment, dementia, traumatic brain injury, or normal aging often fail not because the event was never perceived, but because the cue for retrieval is missing. The canonical consumer version is: **"Where did I leave my keys?"** A text log saying "keys seen at 08:12" is weak; a spatially anchored replay showing _the keys on the hallway console, next to the mail, after entering from the garage_ is much stronger. SenseCam's core finding supports this: reviewing passively captured first-person images can trigger vivid autobiographical remembering, including in memory-impaired users. Spatial indexing upgrades SenseCam from diary playback to active recall: when the user stands near the hallway console, glasses can surface the last relevant key episode without a verbal search.

**Key spatial memory requirement.** Store object identity, object pose/region, user pose, gaze/contact event, room/zone label, timestamp, and confidence. Also store object-state transitions: "keys entered pocket," "keys placed on tray," "tray later occluded by newspaper." For dementia/MCI, store reviewable visual snippets with minimal cognitive load, plus caregiver controls and redaction.

**Supporting paper/product.** SenseCam/Microsoft Research is direct precedent for wearable-camera memory aids ([Microsoft SenseCam](https://www.microsoft.com/en-us/research/project/sensecam/); [Hodges et al., 2011](https://www.tandfonline.com/doi/abs/10.1080/09658211.2011.605591)). Ego4D gives broader egocentric episodic-memory framing over daily-life video ([Ego4D](https://ego4d-data.org/)).

**Estimated impact / beneficiaries.** Highest for MCI/dementia, post-stroke/TBI, ADHD, caregivers, and older adults living independently. The benefit is not only answering "where" but reducing anxiety, caregiver calls, repeated searches, medication mistakes, and lost independence.

**Why AI Glass vs phone.** The phone is usually in the pocket, points the wrong direction, and misses the placement moment. Glasses see hands, gaze, containers, and surfaces at the exact moment memory is formed; they can also overlay the cue in the current field of view rather than require a separate search UI.

---

## 2. Cooking / DIY assistance with object-location memory

**Problem framing.** Long procedural tasks fail because the user loses both **step state** and **workspace state**: "Did I add salt?", "Where did I put the 10mm socket?", "Was this screw already removed?" Cooking and DIY are especially spatial: tools migrate, ingredients disappear into bowls, and errors happen when old state is confused with current state. Spatial memory turns the kitchen/workbench into a live task board: it knows the salt was placed behind the flour 30 minutes ago, the measuring spoon was used for vanilla, and the drill bit is on the left edge of the bench.

**Key spatial memory requirement.** Persistent 3D object tracks for ingredients/tools, containers, surfaces, hands, gaze, and action correctness. Need procedural state graph linked to spatial observations: step ID, completed/not completed, object used, object location after use, contamination/safety state, and temporal decay. For DIY, include part provenance: screw A came from bracket B and should return there.

**Supporting paper/product.** HoloAssist directly captures headset-based task execution with RGB, depth, gaze, hands, head pose, IMU, action correctness, and human interventions ([HoloAssist](https://holoassist.github.io/); [ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_HoloAssist_an_Egocentric_Human_Interaction_Dataset_for_Interactive_AI_Assistants_ICCV_2023_paper.html)). Pro^2Assist targets AR-glasses-based long-horizon procedural assistance, using egocentric perception, task progress, and historical predictions to intervene only when useful ([Pro^2Assist](https://arxiv.org/html/2605.04227v1)). PROASSIST adds streaming video dialogue and iterative progress summarization for long tasks ([PROASSIST](https://arxiv.org/html/2506.05904)).

**Estimated impact / beneficiaries.** Home cooks, older adults cooking safely, novice repairers, makers, field technicians, lab technicians, and vocational learners. Impact is high because the system prevents errors before they propagate: wrong ingredient, missed screw, unsafe tool, forgotten heat source.

**Why AI Glass vs phone.** A phone cannot continuously observe both hands during messy tasks and is unsafe to handle with wet/gloved hands. Glasses keep the view aligned to the active workspace and can overlay arrows or reminders exactly where the missing object or next action is.

---

## 3. AR overlay / re-encounter memory

**Problem framing.** Re-entering a place often means reconstructing context: "What was I doing at this desk yesterday?", "Which boxes did I already inspect in this storage room?", "What did I decide last time I stood here?" Spatial memory can make places self-annotating. When the user returns to a location, glasses retrieve the prior local episode and overlay relevant residue: last action, open decision, unfinished object, or warning.

**Key spatial memory requirement.** Place recognition robust to viewpoint/time changes; local coordinate frames; scene-level episodes; object changes; user-authored or inferred annotations; recency/frequency; permission boundaries. Needs retrieval keyed by current 6DoF pose + scene semantics, not only text.

**Supporting paper/product.** CAST models episodic memory as scenes grounded in time, place, topic, action, and characters, matching re-encounter retrieval ([CAST](https://arxiv.org/html/2602.06051v2)). RenderMem argues that a persistent 3D scene representation can be rendered from query-relevant viewpoints, a useful mechanism for returning to a location and asking what matters from the current view ([RenderMem](https://arxiv.org/html/2603.14669v1)). 3DLLM-Mem stores long-term spatial-temporal representations for embodied action across multi-room 3D environments ([3DLLM-Mem](https://arxiv.org/html/2505.22657)).

**Estimated impact / beneficiaries.** Knowledge workers, makers, inspectors, travelers, caregivers, and anyone managing many physical microtasks. Especially valuable in cluttered homes, labs, studios, hospitals, warehouses, and construction sites.

**Why AI Glass vs phone.** Re-encounter is triggered by being physically present. Glasses can detect that presence passively and render overlays registered to the scene. A phone requires the user to notice the need, unlock, search, and point the camera.

---

## 4. Lost-and-found / object provenance and proactive departure checks

**Problem framing.** "Where did I last see X?" is only the simplest form. More valuable: "You are leaving home; your wallet is still on the kitchen island," or "The badge you usually take to the office was last seen near the laundry basket." Spatial memory can infer object provenance: last seen, moved by whom, put into which container, and whether it is still likely there. This is beyond QA because the system acts at the threshold moment before loss occurs.

**Key spatial memory requirement.** Object identity across occlusion/reappearance; last-seen map cell; carry/put-down events; exit zones; routine model; item criticality; uncertainty and freshness. Need transition model: if keys were seen on table then later seen in hand near door, last known state should update.

**Supporting paper/product.** Ego4D's episodic-memory benchmark is built around daily-life first-person retrieval over long video, the same substrate needed for "last seen" object queries ([Ego4D site](https://ego4d-data.org/), [CVPR 2022 paper](https://openaccess.thecvf.com/content/CVPR2022/html/Grauman_Ego4D_Around_the_World_in_3000_Hours_of_Egocentric_Video_CVPR_2022_paper.html)). SenseCam establishes passive capture as a memory aid for personal events ([SenseCam](https://www.microsoft.com/en-us/research/project/sensecam/)). RenderMem is relevant when the object may be occluded or only visible from a prior viewpoint ([RenderMem](https://arxiv.org/html/2603.14669v1)).

**Estimated impact / beneficiaries.** Broad consumer utility; especially ADHD, older adults, travelers, parents, students, and professionals who must carry critical items. Impact high because it prevents wasted time and missed commitments rather than merely recovering after loss.

**Why AI Glass vs phone.** The placement event often happens while both hands are occupied and the phone is absent. Glasses observe the put-down moment and the departure moment; they can warn at the door with no explicit query.

---

## 5. Social / face-place binding

**Problem framing.** Humans often remember people through context: "I met her at the cafe after the robotics meetup," "He was the contractor near the north loading dock," "We discussed the grant in the hallway." AI glasses could bind a face, name, conversation topic, place, and time into a privacy-governed social memory. The killer feature is not face recognition alone; it is **episodic grounding**: the system reminds the wearer _why_ the person matters in the current place.

**Key spatial memory requirement.** Person embeddings/identity only with consent or policy; face/place/time association; conversation snippets or user-authored notes; relationship graph; location sensitivity; retention rules; bystander redaction. Retrieval should prefer place-conditioned social context: people seen in this room/building/cafe, not a global surveillance database.

**Supporting paper/product.** CAST's character-and-scene memory explicitly organizes events by characters and scenes: who, when, where, and what happened ([CAST](https://arxiv.org/html/2602.06051v2)). Ego4D highlights privacy/ethics as central for egocentric daily-life capture ([Ego4D privacy/ethics section](https://ego4d-data.org/)). SenseCam/lifelogging literature shows the memory value and the privacy tension of passive wearable capture ([Microsoft SenseCam](https://www.microsoft.com/en-us/research/project/sensecam/)).

**Estimated impact / beneficiaries.** Sales, recruiting, healthcare, education, conferences, community care, and users with social-memory impairment. But this is also one of the highest-risk scenarios: privacy, consent, face recognition policy, and social acceptability are product-defining constraints.

**Why AI Glass vs phone.** Social recall happens during live eye contact. A phone lookup is socially disruptive and usually too late. Glasses can offer a subtle, wearer-only cue anchored to the person/place, but only if privacy safeguards are strong.

---

## 6. Workplace ergonomics / OSHA / industrial spatial logbook

**Problem framing.** Many workplaces still rely on clipboards, badge logs, and after-the-fact incident reports: which station used which tool, who entered which zone, whether PPE was worn, which pallet was inspected, which valve was touched. Spatial memory can replace manual logs with an egocentric, location-grounded activity ledger. The value is not just compliance: it helps training, root-cause analysis, ergonomic redesign, and shift handover.

**Key spatial memory requirement.** Workstation zones, tool/object IDs, worker pose/path, PPE state, hands/tool-use events, hazard zones, timestamps, and audit trails. Must support selective retention, worker consent, union/legal boundaries, and on-device redaction for faces/bystanders. For ergonomics, store posture proxies, repeated reaches, dwell time, and awkward-zone interactions.

**Supporting paper/product.** HoloAssist already models headset-captured physical manipulation with action correctness and intervention categories ([HoloAssist](https://holoassist.github.io/)). Ego4D includes workplace scenarios in a large egocentric daily-life corpus ([Ego4D CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Grauman_Ego4D_Around_the_World_in_3000_Hours_of_Egocentric_Video_CVPR_2022_paper.html)). Microsoft Dynamics 365 field-service/remote-assist products show commercial appetite for guided industrial workflows, though they are not full persistent spatial memory ([Dynamics 365 Field Service](https://www.microsoft.com/en-us/dynamics-365/products/field-service)).

**Estimated impact / beneficiaries.** Manufacturing, utilities, hospitals, labs, warehouses, aviation maintenance, construction, safety officers, and frontline workers. High ROI where auditability, training, and safety incidents are expensive.

**Why AI Glass vs phone.** Industrial tasks are hands-busy, PPE-constrained, and zone-specific. Glasses observe tool use and movement continuously while leaving hands free; overlays can warn before a wrong-zone entry or missed step.

---

## 7. Navigation / route memory: "show my usual path"

**Problem framing.** GPS tells you how to reach a coordinate. It does not know the route you actually prefer inside a hospital, garage, campus, or office maze. Spatial memory can learn personal route episodes: "from this entrance, I usually turn right after the vending machine, use the quiet stairwell, and exit near row C." This is different from map navigation; it is **egocentric route replay**.

**Key spatial memory requirement.** Pose trajectory, local landmarks, turns, floor transitions, entrances/exits, accessibility constraints, route frequency, time-of-day variations, and failure points. Need coordinate-free landmark descriptions for humans plus 6DoF anchors for overlays.

**Supporting paper/product.** Clew records a 3D path using camera/AR mapping and later guides blind/low-vision users along it, proving the route-memory primitive in a deployed assistive setting ([Clew](https://clewapp.org/)). HIMM links semantic memory to physical maps/camera poses for embodied exploration and navigation ([HIMM](https://arxiv.org/html/2602.15513)). Ego4D includes daily-life egocentric movement across diverse locations ([Ego4D](https://ego4d-data.org/)).

**Estimated impact / beneficiaries.** Blind/low-vision users, older adults, people with cognitive impairment, hospital visitors, large-campus workers, travelers, and anyone navigating repeated indoor routes. Impact highest indoors where GPS is weak and maps are stale.

**Why AI Glass vs phone.** Navigation cues must be hands-free, continuous, and aligned with the user's head direction. Glasses can show arrows or landmark callouts in view; a phone forces attention down and can be inaccessible while carrying items or using a cane.

---

## 8. Health / habit tracking by location

**Problem framing.** Many health behaviors are spatial routines: fridge visits, medication-cabinet openings, desk time, bed exits, bathroom trips, kitchen snacking, screen exposure in the bedroom. Phones measure steps and app use, not the physical contexts where habits happen. Spatial memory creates behavioral analytics that are interpretable: "most late-night eating starts at the pantry after 22:00," "medication drawer was opened but pill bottle was not handled," "screen time is concentrated at the dining table."

**Key spatial memory requirement.** Room/zone visits, object interactions, routines, temporal patterns, duration, intensity, and privacy filters. Avoid storing raw video by default; retain derived events and allow user review. Need health-sensitive local processing and explicit sharing controls.

**Supporting paper/product.** Ego4D was designed around daily-life activity video across household/work/leisure settings, giving empirical precedent for egocentric behavioral activity understanding ([Ego4D](https://ego4d-data.org/)). SenseCam research also notes lifelogging's value for studying human behavior, beyond memory aid ([Microsoft SenseCam](https://www.microsoft.com/en-us/research/project/sensecam/)). PROASSIST/Pro^2Assist show long-horizon egocentric streaming memory and progress summarization, mechanisms reusable for habit timelines ([PROASSIST](https://arxiv.org/html/2506.05904), [Pro^2Assist](https://arxiv.org/html/2605.04227v1)).

**Estimated impact / beneficiaries.** Chronic disease management, aging-in-place, ADHD, sleep hygiene, nutrition coaching, occupational health, and caregiver-supported living. The impact depends on trust: the product must feel like self-knowledge, not surveillance.

**Why AI Glass vs phone.** Habits involve objects and rooms, not only body motion. Glasses see the fridge handle, pill bottle, keyboard, plate, and room context while the phone only sees coarse motion/location.

---

## 9. Spatial summarization: daily place timeline

**Problem framing.** A normal daily summary is chronological: "9:00 kitchen, 9:30 desk..." A spatial memory summary is place-centric: **"What did I do in the kitchen today?"** It clusters episodes by room, surface, object, and task: prepared breakfast, spilled coffee near sink, moved lunch box to fridge, returned at 19:00, left medication on counter. This becomes a high-value personal memory UI because users often remember by place, not timestamp.

**Key spatial memory requirement.** Scene segmentation, place clusters, event summarization, object/action tracks, time ranges, salience scoring, and contradiction/change detection. Need retrieval modes: by room, by object, by task, by person, by unfinished state.

**Supporting paper/product.** PROASSIST introduces iterative progress summarization for streaming egocentric videos, preserving long-horizon task context ([PROASSIST](https://arxiv.org/html/2506.05904)). CAST aggregates views into scenes by time/place/topic/action and summarizes them as episodic memory ([CAST](https://arxiv.org/html/2602.06051v2)). SenseCam viewer workflows already relied on reviewing captured life images as memory cues ([Microsoft SenseCam](https://www.microsoft.com/en-us/research/project/sensecam/)).

**Estimated impact / beneficiaries.** Everyone who keeps diaries, caregivers, clinicians, knowledge workers, students, and people with memory challenges. Strong value for retrospective reflection: "where did the day go?" becomes answerable by physical context.

**Why AI Glass vs phone.** Summaries need complete, passive coverage of spaces and objects. A phone captures only intentional photos or coarse GPS; glasses capture the micro-events that make a room timeline meaningful.

---

## 10. Embodied agent handover: "monitor this place for me"

**Problem framing.** The user often wants to delegate attention, not ask a question: "Watch the stove while I answer the door," "Tell me if the soldering iron remains on," "Monitor whether the printer job finishes," "Track whether the baby bottle is still on the counter." Spatial memory grounds the agent's attention in a place/object and lets the user hand over a live task.

**Key spatial memory requirement.** User-designated object/zone, current object state, expected state change, hazard thresholds, persistence across occlusion, notification policy, and escalation. Need a working memory for the active handover plus episodic memory for what usually happens at that spot.

**Supporting paper/product.** Pro^2Assist specifically frames proactive assistance on AR glasses, using continuous egocentric perception, task progress, and history to decide when to speak ([Pro^2Assist](https://arxiv.org/html/2605.04227v1)). HoloAssist shows human instructors monitoring a headset wearer's task in real time and intervening when necessary ([HoloAssist ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_HoloAssist_an_Egocentric_Human_Interaction_Dataset_for_Interactive_AI_Assistants_ICCV_2023_paper.html)). 3DLLM-Mem supports action planning over spatial-temporal memory, relevant to delegated embodied agents ([3DLLM-Mem](https://arxiv.org/html/2505.22657)).

**Estimated impact / beneficiaries.** Cooking safety, childcare, eldercare, labs, workshops, small businesses, and accessibility. High impact where missed visual monitoring causes hazard or waste.

**Why AI Glass vs phone.** Handover begins while the user is looking at the target. Glasses can bind the instruction "monitor this" to the current object/zone by gaze and pose. A phone requires framing and manual setup.

---

## 11. Lost map / "where am I?" disambiguation for first-time visitors and partially blind users

**Problem framing.** Indoor disorientation is not only a route problem; it is a place-recognition problem. A first-time visitor in a hospital corridor or a partially blind user in a transit station needs: "which hallway is this, which direction am I facing, and what nearby landmarks matter?" Spatial memory helps by matching the current egocentric view to a previously built local map or public/shared building memory.

**Key spatial memory requirement.** Robust place recognition, floor/zone labels, landmark inventory, current orientation, confidence, accessible path constraints, and low-vision-friendly output. Must operate with partial views, lighting changes, and similar-looking corridors.

**Supporting paper/product.** Clew's AR-based route capture and replay shows practical 3D path memory for blind/low-vision mobility ([Clew](https://clewapp.org/)). HIMM stores semantic features indexed by camera pose and links them to a physical occupancy map, a direct research analogue for place disambiguation ([HIMM](https://arxiv.org/html/2602.15513)). RenderMem's viewpoint-conditioned rendering is relevant when the system must reason about what is visible from where the wearer stands ([RenderMem](https://arxiv.org/html/2603.14669v1)).

**Estimated impact / beneficiaries.** Blind/low-vision users, neurodivergent users, older adults, hospital patients, travelers, and visitors to complex buildings. Impact is high because indoor wayfinding failures create stress, time loss, and safety risk.

**Why AI Glass vs phone.** The user needs both hands and continuous awareness; looking down at a phone worsens safety and accessibility. Glasses can provide audio/haptic/visual orientation tied to the user's head pose.

---

## 12. Real-world search engine: physical-world Google

**Problem framing.** The ultimate non-QA scenario is a personal physical-world search engine: "find the receipt I saw last week," "show all places I saw this logo," "when did I last see the red umbrella?", "what objects have lived on this shelf?" This extends lifelogging from chronological archive to **spatial search over lived reality**. The killer feature is that every visual memory is indexed by where it was seen and how it related to the user's body/actions.

**Key spatial memory requirement.** Scalable multimodal index over object, OCR, place, face/person policy, action, audio, time, gaze, and 3D pose; deduplication; personal ontology; private/local retrieval; provenance for every result; deletion and redaction. Needs multi-resolution storage: raw clips for opted-in moments, thumbnails for review, embeddings/events for default.

**Supporting paper/product.** MyLifeBits is Microsoft's classic lifelogging project for storing and searching personal digital memory ([MyLifeBits](https://www.microsoft.com/en-us/research/project/mylifebits/)). SenseCam provides the wearable visual capture precedent ([SenseCam](https://www.microsoft.com/en-us/research/project/sensecam/)). Ego4D scales egocentric capture to thousands of hours and benchmarks episodic retrieval ([Ego4D](https://ego4d-data.org/)). 3DLLM-Mem and RenderMem show how 3D/spatial-temporal memory can be queried for embodied reasoning rather than flat keyword search ([3DLLM-Mem](https://arxiv.org/html/2505.22657), [RenderMem](https://arxiv.org/html/2603.14669v1)).

**Estimated impact / beneficiaries.** Broad consumer market, professionals managing physical inventory, investigators, repair technicians, researchers, designers, parents, and memory-impaired users. Potentially the highest long-term platform value, but also the largest privacy/computation challenge.

**Why AI Glass vs phone.** A physical-world search engine needs continuous first-person indexing. Phones only index intentional captures and coarse location. Glasses see the same world the user sees and can register search results back into the world.

---

## Cross-scenario design implications for WorldMM

1. **Spatial axis must be first-class, not metadata.** Store 6DoF pose, room/zone, surface/container, object pose, and viewpoint with every memory. The retrieval key should often start with "where I am now," not a text query.
2. **Events beat frames.** Raw egocentric video is too dense. Convert streams into spatial events: object placed, object picked up, person encountered, zone entered, step completed, hazard state changed.
3. **Support proactive triggers.** Departure threshold, return-to-place, long dwell, object missing from routine, unsafe state, and route deviation are more impactful than user-initiated QA.
4. **Represent uncertainty.** For lost objects and health/safety, show confidence and last evidence: "wallet last seen on island 42 min ago; not observed since."
5. **Privacy is product architecture.** Face-place binding, health habits, workplace logs, and real-world search require local processing, consent, redaction, retention limits, and explainable provenance.
6. **AI glasses are justified when the moment matters.** Killer scenarios require first-person capture at placement/action time, hands-free retrieval, gaze/hand context, and overlay/audio guidance in the same physical space.

## Priority shortlist

If WorldMM needs 3 flagship demos beyond MCQ QA:

1. **Lost-and-found + proactive departure check** — clearest consumer pull; spatial memory is necessary.
2. **Cooking/DIY workspace memory** — shows long-horizon task + object location + step memory; strong HoloAssist/Pro^2Assist precedent.
3. **Spatial daily summary / re-encounter overlay** — demonstrates the premise "when space becomes the index, memory has perfect context" better than question answering.

Secondary verticals with high strategic value: MCI/dementia memory aid, indoor route memory/accessibility, industrial spatial logbook, and embodied handover.

## Note on ROAM / Aria Indexing

The requested "ROAM" and "Aria Indexing" references were not used as primary evidence here because I could not verify a stable source URL during this pass. Closest verified adjacent evidence: HoloAssist and Pro^2Assist for AR procedural assistance; MyLifeBits/SenseCam/Ego4D for lifelog search; RenderMem/3DLLM-Mem/HIMM for spatial-temporal embodied memory.
