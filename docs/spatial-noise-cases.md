# Spatial Memory — Noise Cases Catalog (DOWN flips, for PPT honesty slide)

**Date:** 2026-05-18 KST
**Purpose:** The honest counter-document to [`docs/spatial-signal-cases.md`](spatial-signal-cases.md). Every case below is one where the 4th-axis spatial memory measurably **hurt** the agent — the 3-axis baseline got it right, the 4-axis run got it wrong. Same per-case structure as the signal catalog: question text, all options, gold + both predictions, captions, spatial triples that misled, video URL, and a single sentence on *why* the spatial signal pulled the agent in the wrong direction.

**Why publish this:** the headline ablation number (−1.7 %p at n = 60) is essentially neutral; presenting only the UP cases would inflate the spatial layer's value. Pairing the 3 UP cases with the 3 DN cases makes the talk honest and gives the audience a concrete intuition for the failure mode that the planned reasoning-prompt routing fix has to attack.

---

## 0. Headline summary

| Ablation | n | DN cases (covered here) |
|---|---:|---|
| EgoLifeQA unfiltered first n=60 (3-axis vs 4-axis) | 60 | Q6 grow-flowers · Q45 coffee-timing · Q50 phone-habit |

**3 DN cases total.** All come from the unfiltered n=60 ablation. The synthetic WHERE-only set had **0 DN flips** (regression rate = 0/20), so DN exposure is purely on real EgoLifeQA traffic.

**EgoLife dataset URL format** for all video links: `https://huggingface.co/datasets/lmms-lab/EgoLife/resolve/main/A1_JAKE/DAY1/<filename>.mp4`

---

## 1. Q6 — "Who plans to grow flowers"

**Why this case for the PPT:** A TaskMaster (future-plan) question whose answer is buried in a direct quote from Katrina. The 3-axis baseline correctly resolved through the episodic captioning of her speech ("I was thinking of planting something … flowers from Yunnan"). The 4-axis run with spatial picked Alice instead — because the spatial layer surfaced `(flowers, in, box)` plus `(Shure, next_to, Alice)` right around the flower-arranging moment, and the agent confused *physical proximity to flowers* with *being the one who planned the project*.

**Question metadata**

| Field | Value |
|---|---|
| ID | 6 |
| Type | TaskMaster (future plan) |
| `query_time` | DAY1 11:57:36.06 |
| `target_time` | DAY1 11:34:40.19 |
| Trigger | "Alice took the flowers out of the vase" |
| Question | *Who plans to grow flowers* |
| Choice A | Alice |
| Choice B | Tasha |
| Choice C | Me (the camera wearer) |
| **Choice D (gold)** | **Katrina** |
| Annotator reason | (Katrina explicitly said she planned the planting / Earth-Day theme) |

**Predictions**

| Configuration | Prediction | Correct? |
|---|---|---|
| 3-axis (E + S + V no-op) | **D — Katrina** | ✅ |
| 4-axis (+ spatial) | **A — Alice** | ❌ |

**Video evidence**

- Video URL: <https://huggingface.co/datasets/lmms-lab/EgoLife/resolve/main/A1_JAKE/DAY1/DAY1_A1_JAKE_11343000.mp4>
- Fine caption around the target (DAY1 11:34:30 → 11:34:59):
  > "I turn around, walk into the restaurant, and carry a cardboard box to the dining table, straightening it, setting it down, and nudging it forward. **Katrina says, 'Because I saw … I was thinking of planting something. That fits the Earth Day theme, but it wouldn't sprout in 7 days, so I thought of buying these flowers. Directly shipped by SF Express, flowers from Yunnan.'** 'Show…' "
- The plan-author is *explicitly named in dialogue* by Katrina herself in this very window.

**Spatial triples that misled the agent**

- Chunk around DAY1 11:34:30:
  - `(Shure, next_to, Alice)` ← Alice is physically there
  - `(I, on, stool)`
  - `(Shure, on, whiteboard)`
  - `(I, in, Shure's original seat)`
  - `(Shure, near, whiteboard)`
  - `(Tasha, on, table)`
- Chunk around DAY1 11:39:30 (just after target):
  - `(I, located_in, restaurant)`
  - `(Katrina, located_in, restaurant)`
  - `(cardboard box, located_in, restaurant)`
  - **`(flowers, in, box)` ← the bait: flowers physically with the box, Alice was unpacking the box, so spatial-similarity to "flowers" pulls toward Alice**
  - `(Shure, near, I)`

**Why spatial misled.** The spatial layer correctly encoded the physical scene (Alice handling the box that contained the flowers), but the question was about *plans*, not *who is touching the flowers right now*. The spatial layer has no notion of "intent" — the planner can be physically distant from the object. Episodic caught the actual planning utterance; spatial overrode it with a physical-proximity heuristic.

**The fix this case argues for.** The reasoning prompt should not route TaskMaster / future-plan questions to the spatial axis at all. A type-aware routing rule ("if question contains *plans*, *will*, *intends*, route to episodic + semantic only") closes this DN at zero cost to the WHERE-grounded UP cases.

---

## 2. Q45 — "When was the last time we talked about coffee?"

**Why this case for the PPT:** A temporal question (when?) whose answer is "about four hours ago". The 3-axis baseline got it right. The 4-axis run picked "one day ago" — because the spatial layer kept emphasizing `(I, located_in, kitchen)` across many chunks, which the agent over-interpreted as "kitchen activity → cooking / drinks happen daily → the relevant coffee mention must be older than that".

**Question metadata**

| Field | Value |
|---|---|
| ID | 45 |
| Type | EntityLog (temporal flavor) |
| `query_time` | DAY1 17:49:01.08 |
| `target_time` | DAY1 13:34:57.08 |
| Question | *When was the last time we talked about coffee?* |
| **Choice A (gold)** | **About four hours ago** |
| Choice B | About eight hours ago |
| Choice C | About an hour ago |
| Choice D | One day ago |
| Annotator reason | "Jake suggested having some coffee (at DAY1 13:34)" |
| Keywords | talk about coffee |

**Predictions**

| Configuration | Prediction | Correct? |
|---|---|---|
| 3-axis (E + S + V no-op) | **A — about four hours ago** | ✅ |
| 4-axis (+ spatial) | **D — one day ago** | ❌ |

**Video evidence**

- Video URL: <https://huggingface.co/datasets/lmms-lab/EgoLife/resolve/main/A1_JAKE/DAY1/DAY1_A1_JAKE_13343000.mp4>
- Fine caption around the target (DAY1 13:34:30 → 13:35:00):
  > "I wipe the whiteboard while we talk. Lucia asks, 'Hey, doesn't this whiteboard have a back side? Only one side?' 'It has a back side. The back side is a blackboard. Yes, you can write with chalk,' I say. Lucia says, 'Ah, OK. So that's the design.' 'He also gave red pens and chalk, right,' I say. Tasha asks, 'This is a spare set, right?' Lucia replies, 'Right. That's the one. Th…' "
- The visible window is about whiteboards rather than coffee — the coffee mention is slightly earlier in the chunk (Lucia describing ordering coffee). Both the gold annotator and the 3-axis baseline correctly anchored on that earlier coffee utterance.

**Spatial triples that misled the agent**

- Chunk around DAY1 13:29:30 (just before target):
  - `(paper_bag, located_in, kitchen)`
  - **`(I, located_in, kitchen)`** ← persistent kitchen anchor
  - `(I, near, Lucia)`
  - `(water, on, table)`
- Chunk around DAY1 13:34:29 (the target chunk):
  - `(I, near, whiteboard)`
  - `(I, located_in, first_floor)`
  - `(kitchen, located_in, first_floor)`
  - `(activity_room, located_in, upstairs)`
  - `(meeting_room, located_in, first_floor)`
  - **`(I, located_in, kitchen)`** ← repeated
- Chunk around DAY1 13:39:30 (just after target):
  - `(I, located_in, bedroom)` / `(I, located_in, living_room)` / `(I, in_front_of, whiteboard)` — wearer moves around quickly

**Why spatial misled.** None of the spatial triples are about *coffee*. But the kitchen-anchor triples dominate the spatial context, and the agent's reasoning chain seems to have gone *"wearer is in the kitchen ↔ kitchen → cooking and drinks ↔ a coffee discussion in the kitchen is something that happens every day"* — therefore "one day ago" rather than "four hours ago". The spatial layer here did not surface a *wrong fact*; it surfaced *true facts that were irrelevant to a temporal question*, and the prompt's added context length pushed the answer.

**The fix this case argues for.** Temporal "when was the last time…" questions should route to episodic with explicit time-window retrieval (use `query_time` minus N hours), not to the spatial layer at all. The spatial axis adds true-but-irrelevant context that nudges the agent away from precise temporal answers.

---

## 3. Q50 — "What do I usually show everyone on my phone?"

**Why this case for the PPT:** A HabitInsight question whose answer is the highly specific "timer interface". The 3-axis baseline got it right by finding the explicit episodic moment of Jake showing the timer. The 4-axis run picked "share interesting videos" — because the spatial layer surfaced many `(I, on, phone)` and `(Shure, on, his phone)` triples scattered across the day, which the agent generalized into "phone-mediated social sharing".

**Question metadata**

| Field | Value |
|---|---|
| ID | 50 |
| Type | HabitInsight |
| `query_time` | DAY1 19:02:53.19 |
| `target_time` (range) | DAY1 12:23:43 + DAY1 17:11:44 |
| Question | *What do I usually show everyone on my phone?* |
| Choice A | Share interesting videos with everyone |
| Choice B | Show everyone what they want to eat |
| Choice C | I'm recording a video and want everyone to be in it |
| **Choice D (gold)** | **Take a look at the timer interface on my phone** |
| Annotator reason | "I opened my phone to the timer interface and showed it to everyone" |
| Keywords | show everyone, on phone |

**Predictions**

| Configuration | Prediction | Correct? |
|---|---|---|
| 3-axis (E + S + V no-op) | **D — timer interface** | ✅ |
| 4-axis (+ spatial) | **A — share interesting videos** | ❌ |

**Video evidence**

Two relevant target windows (the question's `target_time` is a range with two anchors):

- Anchor 1 — Video URL: <https://huggingface.co/datasets/lmms-lab/EgoLife/resolve/main/A1_JAKE/DAY1/DAY1_A1_JAKE_12233000.mp4>
  - Fine caption (DAY1 12:23:30 → 12:24:00):
    > " 'It's okay,' I say. **I keep operating my phone and show everyone the timer.** 'Actually, one person left is fine,' I say. 'But it's safer to take another round,' I say. **I keep showing the phones,** take my phone back, and use it with both hands. Katrina says, 'I found this business card so pretty.' Shure asks, 'Oh, did you watch *What Kind of Life Do You Want to Live*? The one by Ha…' "
- Anchor 2 — Video URL: <https://huggingface.co/datasets/lmms-lab/EgoLife/resolve/main/A1_JAKE/DAY1/DAY1_A1_JAKE_17113000.mp4>
  - Fine caption (DAY1 17:11:31 → 17:11:58):
    > "I adjust the framing and hand my phone to Alice to check. 'Come, take a look,' I say. Lucia replies, 'OK.' I take my phone back, check it, and pass it to Shure on the left to confirm before it goes to Tasha and Katrina. 'Looks good?' I ask. Shure replies, 'Looks good. Awesome.' Alice says, 'Looks good,' and I say, 'Looks good.' I take my phone back again, turn off the screen, p…"

The first anchor explicitly says "show everyone the timer". The 3-axis baseline found this and answered correctly.

**Spatial triples that misled the agent**

- Chunk around DAY1 14:14:29:
  - `(I, located_in, restaurant)`
  - `(I, located_in, courtyard)`
  - **`(I, on, Katrina's_phone)`** ← multi-phone, multi-person activity
  - `(Second_Ring_Road, located_in, beijing)`
  - `(Katrina, near, Beijing)`
- Chunk around DAY1 17:11:30 (target):
  - `(I, located_in, kitchen)`
  - `(I, on, stool)`
  - `(I, located_in, restaurant)`
  - **`(Shure, on, his phone)`** ← yet another "person on phone" pattern
- Chunk around DAY1 17:16:29:
  - `(Shure, left_of, I)`
  - `(I, behind, seat)`
  - **All present at supermarket** — group activity context

**Why spatial misled.** The spatial layer reliably encodes *which people are physically together using phones*, scattered across many chunks. The agent collapsed this into a habit ("group phone use → group video sharing"), losing the specificity that the *content* of the phone screen was a timer. The episodic layer has the explicit "show everyone the timer" sentence; spatial diluted it by surrounding the prompt with broader phone-with-people context.

**The fix this case argues for.** HabitInsight questions phrased as "what do I *show* / *say* / *do* …" should preferentially weight episodic content (specific actions described in captions) over spatial co-presence facts. A simple prompt nudge — "for content-of-action questions, retrieve episodic first and only consult spatial when episodic returns ambiguous results" — should close this case.

---

## 4. Pattern summary (one slide)

| Case | Question type | Why spatial hurt | Suggested routing fix |
|---|---|---|---|
| Q6 | TaskMaster (future plan) | Physical proximity to flowers conflated with planning intent | Skip spatial for *plan / will / intends* triggers |
| Q45 | EntityLog (temporal) | True-but-irrelevant kitchen anchors diluted the temporal signal | Skip spatial for *when / how long ago* triggers |
| Q50 | HabitInsight (content of action) | Broad "people on phones" pattern overrode the specific timer mention | For *what do I show / say / do*, retrieve episodic first |

**Across the 3 DN cases, the failure mode is identical: spatial surfaces true facts that are irrelevant to the question, and the extra prompt context pulls the reasoning agent toward an answer that is consistent with the spatial scene but not with the question's intent.**

This is the strongest argument for the routing fix proposed in [`docs/spatial-encoding-sensor-based.md`](spatial-encoding-sensor-based.md) §7: keep the 4-axis architecture, but constrain when the reasoning agent is allowed to pick the spatial branch.

---

## 5. PPT slide outline (suggested addition to the signal-cases slide deck)

After the 6 UP-case slides (slides 3–8 in `docs/spatial-signal-cases.md` §3), insert one slide per DN case (numbered 9 / 10 / 11), each with the same layout as the UP slides but a red-highlighted prediction box. Then close with a single "pattern" slide using the §4 table here.

Total deck: **3 (intro) + 6 (UP) + 3 (DN) + 1 (pattern) + 1 (next steps) = 14 slides**.

---

## 6. Reproduction (same as signal-cases)

```bash
# Build all four memories
bash script/3_build_memory.sh --step all --person A1_JAKE --model chatgpt-gpt-5.4

# Run the 3-axis vs 4-axis ablation; DN cases are in the per-question JSON
WORLDMM_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
WORLDMM_EMBED_DEVICE=cpu \
uv run python eval/three_vs_four_axis_ablation.py \
    --max-n 60 --max-rounds 3 \
    --output output/three_vs_four_n60.json

# Find the DN cases programmatically:
python -c "
import json
with open('output/three_vs_four_n60.json') as f:
    r = json.load(f)
for q in r['per_question']:
    if q['three_axis']['correct'] and not q['four_axis']['correct']:
        print(f\"Q{q['id']} [{q['type']}] gold={q['gold']} 3a={q['three_axis']['letter']} 4a={q['four_axis']['letter']}\")
"
```
