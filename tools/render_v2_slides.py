#!/usr/bin/env python3
"""Batch-render before/after V2 spatial slides from per-case data.

Reads a small `cases` array (defined inline below), substitutes into the V2
template extracted from docs/slides/q1-screwdriver-v2.html, and writes one
HTML per case. Each case ships with: question, choices, gold, baseline
prediction + reasoning, +spatial prediction + the *one* decisive triple,
optional scene thumbnail + Alice-style annotation, and an optional new
scenario flag (drawn from a hand-authored set, not from EgoLifeQA).
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Annotation:
    box: tuple[int, int, int, int]
    label: str
    arrow: Optional[str] = None


@dataclass
class Case:
    slide_id: str
    direction: str
    deck_index: int
    deck_total: int
    case_id: str
    qtype: str
    moment_label: str
    question: str
    choices: dict[str, str]
    gold: str
    baseline_letter: str
    baseline_short: str
    baseline_caption: str
    baseline_actor_note: str
    plus_label: str
    spatial_letter: str
    spatial_short: str
    triple_subj: str
    triple_pred: str
    triple_obj: str
    spatial_reading: str
    headline_left: str = "wrong"
    headline_right: str = "right"
    thumb: Optional[str] = None
    scene_label: Optional[str] = None
    scene_caption_text: str = ""
    scene_caption_small: str = ""
    annotation: Optional[Annotation] = None
    you_arrow: Optional[str] = None
    headline_phrase: str = "One extra spatial fact"
    why_panel_label: str = "key spatial triple"
    is_synthetic: bool = False
    is_new_scenario: bool = False


def render(case: Case) -> str:
    baseline_correct = case.baseline_letter == case.gold
    spatial_correct = case.spatial_letter == case.gold
    headline_left_color = "grn" if baseline_correct else "red"
    headline_right_color = "grn" if spatial_correct else "red"
    panel_left = "yes-sp" if baseline_correct else "no-sp"
    panel_right = "yes-sp" if spatial_correct else "no-sp"
    baseline_verdict = "✓ CORRECT" if baseline_correct else "✗ WRONG"
    spatial_verdict = "✓ CORRECT" if spatial_correct else "✗ WRONG"
    panel_left_h2 = "WITHOUT SPATIAL · 3-axis baseline"
    panel_right_h2 = "WITH SPATIAL · 4-axis"
    section_chip = "UP flip" if case.direction == "UP" else "DN flip"
    chip_tag_color = "var(--accent)" if case.direction == "UP" else "var(--bad)"

    scene_html = ""
    if case.thumb:
        ann_html = ""
        if case.annotation:
            x, y, w, h = case.annotation.box
            ann_html += (
                f'<div class="ann" style="left:{x}%;top:{y}%;width:{w}%;height:{h}%"></div>'
                f'<div class="ann-label" style="left:{x}%;top:{max(0, y - 9)}%">{html.escape(case.annotation.label)}</div>'
            )
        you_arrow = (
            f'<div class="you-arrow">{html.escape(case.you_arrow)}</div>' if case.you_arrow else ""
        )
        scene_label = case.scene_label or "scene"
        scene_html = f"""
  <div class="scene-row">
    <div class="scene">
      <img src="assets/{html.escape(case.thumb)}">
      <div class="scene-label">{html.escape(scene_label)}</div>
      {you_arrow}
      {ann_html}
    </div>
    <div class="scene-caption">
      {case.scene_caption_text}
      <div class="small">{case.scene_caption_small}</div>
    </div>
  </div>
"""
    new_badge = (
        ' <span style="margin-left:6px;color:#fbbf24;font-family:var(--mono);font-size:10px;letter-spacing:1px">NEW SCENARIO</span>'
        if case.is_new_scenario
        else ""
    )
    synth_badge = (
        ' <span style="margin-left:6px;color:#a5b4fc;font-family:var(--mono);font-size:10px;letter-spacing:1px">SYNTHETIC</span>'
        if case.is_synthetic
        else ""
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WorldMM · {html.escape(case.case_id)} — before / after spatial</title>
<style>
  :root{{
    --bg:#0b1020; --panel:#11192f; --panel2:#0e1730; --line:#1c2746; --ink:#e9ecf5; --muted:#8aa0c8;
    --good:#22c55e; --bad:#ef4444; --accent:#60a5fa; --hl:#fbbf24;
    --good-edge:rgba(34,197,94,.55); --bad-edge:rgba(239,68,68,.55);
    --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
    --sans:Inter,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
  }}
  *{{box-sizing:border-box}} html,body{{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans)}}
  .slide{{width:1280px;height:720px;margin:24px auto;padding:22px 26px;background:var(--panel);border-radius:16px;box-shadow:0 8px 40px rgba(0,0,0,.45);display:grid;grid-template-rows:auto auto 1fr auto;gap:10px}}
  header{{display:flex;justify-content:space-between;align-items:baseline;border-bottom:1px solid var(--line);padding-bottom:6px}}
  header h1{{margin:0;font-size:23px;font-weight:700;letter-spacing:-.01em}}
  header h1 .red{{color:var(--bad)}} header h1 .grn{{color:var(--good)}} header h1 .arrow{{color:var(--hl);margin:0 4px}}
  header .tag{{color:{chip_tag_color};font-family:var(--mono);font-size:11px;letter-spacing:1px;text-transform:uppercase}}
  header .meta{{color:var(--muted);font-family:var(--mono);font-size:11px;text-align:right}}
  .question-bar{{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:9px 16px;display:flex;align-items:center;gap:14px}}
  .question-bar .qlabel{{color:var(--accent);font-family:var(--mono);font-size:10.5px;text-transform:uppercase;letter-spacing:1px}}
  .question-bar .qtext{{font-size:17px;font-weight:600}}
  .question-bar .gold{{margin-left:auto;color:var(--good);font-family:var(--mono);font-size:12px}}

  .twocol{{display:grid;grid-template-columns:1fr 80px 1fr;gap:0;min-height:0}}
  .panel{{border-radius:14px;padding:14px 18px;border:2px solid;display:flex;flex-direction:column;gap:12px;min-height:0;background:#0a1325}}
  .panel.no-sp{{border-color:var(--bad-edge);box-shadow:inset 0 0 0 1px rgba(239,68,68,.18)}}
  .panel.yes-sp{{border-color:var(--good-edge);box-shadow:inset 0 0 0 1px rgba(34,197,94,.22)}}
  .panel h2{{margin:0;font-size:13px;font-weight:800;text-transform:uppercase;letter-spacing:1.2px;display:flex;align-items:center;gap:8px}}
  .panel.no-sp h2{{color:var(--bad)}} .panel.no-sp h2::before{{content:"✗";font-size:18px}}
  .panel.yes-sp h2{{color:var(--good)}} .panel.yes-sp h2::before{{content:"✓";font-size:18px}}
  .panel .axes{{display:flex;gap:6px;flex-wrap:wrap;font-family:var(--mono);font-size:10.5px}}
  .panel .axes span{{padding:2px 8px;border-radius:999px;background:rgba(255,255,255,.04);border:1px solid var(--line);color:var(--muted)}}
  .panel .axes span.added{{color:#062012;background:var(--good);border-color:var(--good);font-weight:700}}
  .panel .axes span.removed{{color:#fff;background:var(--bad);border-color:var(--bad);font-weight:700;text-decoration:line-through}}
  .ctx-box{{background:rgba(0,0,0,.32);border:1px solid var(--line);border-radius:8px;padding:10px 12px;font-size:12.5px;line-height:1.5;color:#cfd8ee}}
  .ctx-box .label{{color:var(--muted);font-family:var(--mono);font-size:10.5px;text-transform:uppercase;letter-spacing:0.6px;margin-bottom:6px}}
  .ctx-box em{{color:var(--ink);font-style:normal;font-weight:600}}

  .smoking-gun{{background:linear-gradient(180deg,rgba(34,197,94,.22),rgba(34,197,94,.10));border:2px solid var(--good);border-radius:10px;padding:14px 16px;box-shadow:0 0 28px rgba(34,197,94,.25)}}
  .panel.no-sp .smoking-gun{{background:linear-gradient(180deg,rgba(239,68,68,.22),rgba(239,68,68,.10));border-color:var(--bad);box-shadow:0 0 28px rgba(239,68,68,.25)}}
  .smoking-gun .lab{{font-family:var(--mono);color:var(--good);font-size:10.5px;letter-spacing:1px;text-transform:uppercase;margin-bottom:6px}}
  .panel.no-sp .smoking-gun .lab{{color:var(--bad)}}
  .smoking-gun .triple{{font-family:var(--mono);font-size:24px;color:#dcfce7;font-weight:800;letter-spacing:.5px;text-align:center}}
  .panel.no-sp .smoking-gun .triple{{color:#fecaca}}
  .smoking-gun .triple .you{{color:var(--hl)}}
  .smoking-gun .triple .pred{{color:#86efac;margin:0 6px}}
  .panel.no-sp .smoking-gun .triple .pred{{color:#fca5a5}}
  .smoking-gun .triple .alice{{color:#fff;text-decoration:underline;text-decoration-color:var(--good);text-underline-offset:4px}}
  .panel.no-sp .smoking-gun .triple .alice{{text-decoration-color:var(--bad)}}
  .smoking-gun .reading{{margin-top:8px;color:#bbf7d0;font-size:12.5px;line-height:1.45}}
  .panel.no-sp .smoking-gun .reading{{color:#fecaca}}

  .pred-final{{display:flex;align-items:center;gap:14px;padding:14px 18px;border-radius:10px;font-family:var(--mono)}}
  .panel.no-sp .pred-final{{background:rgba(239,68,68,.16);border:1px solid var(--bad-edge);color:#fecaca}}
  .panel.yes-sp .pred-final{{background:rgba(34,197,94,.18);border:1px solid var(--good-edge);color:#bbf7d0}}
  .pred-final .letter{{font-size:46px;font-weight:900;line-height:1;font-family:var(--sans)}}
  .pred-final .name{{font-size:18px;font-weight:700}}
  .pred-final .verdict{{margin-left:auto;font-size:14px;font-weight:700;letter-spacing:.4px}}

  .divider{{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px}}
  .divider .plus{{width:54px;height:54px;border-radius:50%;background:var(--hl);color:#1a1300;display:flex;align-items:center;justify-content:center;font-size:32px;font-weight:900;font-family:var(--sans);box-shadow:0 0 24px rgba(251,191,36,.45)}}
  .divider .what{{font-family:var(--mono);font-size:11px;color:var(--hl);letter-spacing:1px;text-transform:uppercase;text-align:center}}
  .divider .one{{font-size:18px;color:#fff8d6;font-weight:800;text-align:center;line-height:1.1}}
  .divider .arrow-down{{font-size:22px;color:var(--hl);font-weight:900;margin-top:2px}}

  .scene-row{{display:grid;grid-template-columns:auto 1fr;gap:14px;align-items:stretch}}
  .scene{{position:relative;border:1px solid var(--line);border-radius:10px;overflow:hidden;background:#06090f;width:280px;height:158px}}
  .scene img{{width:100%;height:100%;object-fit:cover;display:block;filter:saturate(1.06) contrast(1.05)}}
  .scene .scene-label{{position:absolute;top:6px;left:8px;font-family:var(--mono);font-size:10px;color:var(--muted);background:rgba(11,16,32,.78);padding:2px 7px;border-radius:5px;letter-spacing:.4px}}
  .scene .ann{{position:absolute;border:2px solid var(--good);border-radius:6px;box-shadow:0 0 0 2px rgba(34,197,94,.18),0 0 18px rgba(34,197,94,.45)}}
  .scene .ann-label{{position:absolute;background:var(--good);color:#062012;font-family:var(--mono);font-size:10px;padding:2px 7px;border-radius:5px;font-weight:800;letter-spacing:.4px;white-space:nowrap}}
  .scene .you-arrow{{position:absolute;top:6px;right:8px;background:rgba(11,16,32,.85);border:1px solid var(--hl);color:var(--hl);font-family:var(--mono);font-size:10px;padding:3px 8px;border-radius:5px;font-weight:700}}
  .scene-caption{{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:12px 16px;font-size:13px;line-height:1.5;color:#cfd8ee;display:flex;flex-direction:column;justify-content:center}}
  .scene-caption b{{color:var(--hl)}}
  .scene-caption .small{{color:var(--muted);font-size:11px;margin-top:6px;font-family:var(--mono)}}

  footer{{display:flex;justify-content:space-between;color:var(--muted);font-family:var(--mono);font-size:10.5px;border-top:1px solid var(--line);padding-top:6px}}
</style>
</head>
<body>

<div class="slide">

  <header>
    <div>
      <div class="tag">spatial memory · before / after · {section_chip}{new_badge}{synth_badge}</div>
      <h1>{html.escape(case.headline_phrase)}: <span class="{headline_left_color}">{html.escape(case.headline_left)}</span><span class="arrow">→</span><span class="{headline_right_color}">{html.escape(case.headline_right)}</span></h1>
    </div>
    <div class="meta">{html.escape(case.moment_label)}<br>3-axis 37/60 · 4-axis 36/60</div>
  </header>

  <div class="question-bar">
    <span class="qlabel">{html.escape(case.case_id)}</span>
    <span class="qtext">{html.escape(case.question)}</span>
    <span class="gold">correct answer: {html.escape(case.gold)} · {html.escape(case.choices[case.gold])}</span>
  </div>

  <div class="twocol">

    <div class="panel {panel_left}">
      <h2>{panel_left_h2}</h2>
      <div class="axes">
        <span>episodic ✓</span><span>semantic ✓</span><span>visual (no-op)</span>
      </div>
      <div class="ctx-box">
        <div class="label">{'retrieved context names everyone except the actor' if case.direction == 'UP' else 'baseline reads the episodic capture directly'}</div>
        {case.baseline_caption}
        <br><br>
        <span style="color:var(--muted)">{case.baseline_actor_note}</span>
      </div>
      <div class="pred-final">
        <span class="letter">{html.escape(case.baseline_letter)}</span><span class="name">{html.escape(case.baseline_short)}</span><span class="verdict">{baseline_verdict}</span>
      </div>
    </div>

    <div class="divider">
      <div class="plus">+</div>
      <div class="what">{html.escape(case.plus_label)}</div>
      <div class="one">spatial fact</div>
      <div class="arrow-down">↓</div>
    </div>

    <div class="panel {panel_right}">
      <h2>{panel_right_h2}</h2>
      <div class="axes">
        <span>episodic ✓</span><span>semantic ✓</span><span>visual (no-op)</span>
        <span class="added">+ spatial ★</span>
      </div>
      <div class="smoking-gun">
        <div class="lab">{html.escape(case.why_panel_label)}</div>
        <div class="triple">(<span class="you">{html.escape(case.triple_subj)}</span><span class="pred">{html.escape(case.triple_pred)}</span><span class="alice">{html.escape(case.triple_obj)}</span>)</div>
        <div class="reading">{case.spatial_reading}</div>
      </div>
      <div class="pred-final">
        <span class="letter">{html.escape(case.spatial_letter)}</span><span class="name">{html.escape(case.spatial_short)}</span><span class="verdict">{spatial_verdict}</span>
      </div>
    </div>

  </div>
  {scene_html}

  <footer>
    <div>worldmm · feat/spatial-memory · {section_chip} · {case.deck_index} / {case.deck_total}</div>
    <div>{'same retrieval, same QA, same model — only the spatial axis differs between panels' if case.direction == 'UP' else 'same retrieval, same QA, same model — spatial added a true-but-irrelevant fact'}</div>
  </footer>
</div>

</body>
</html>
"""


CASES: list[Case] = [
    Case(
        slide_id="q33-takeout-app-v2", direction="UP", deck_index=2, deck_total=9,
        case_id="Q33", qtype="EntityLog", moment_label="A1_JAKE · DAY1 13:35 · EntityLog",
        question="What app did I order the takeout in my hand?",
        choices={"A": "Meituan", "B": "Ele.me", "C": "JD.com", "D": "Taobao"}, gold="A",
        baseline_letter="D", baseline_short="Taobao",
        baseline_caption='"Lucia says, ‘This is ordering coffee...,’ I keep wiping the whiteboard, put down the eraser, walk forward smiling. ‘Yes,’ I say. Shure says, ‘Let\'s start lunch now.’ I carry my..."',
        baseline_actor_note="The captions discuss ordering coffee and starting lunch. The on-screen app is never named.",
        plus_label="add ONE",
        spatial_letter="A", spatial_short="Meituan",
        triple_subj="I (wearer)", triple_pred="located_in", triple_obj="kitchen",
        spatial_reading="Anchoring the wearer in the kitchen during lunch ordering lets the agent focus on the most explicit lunch-app evidence in episodic (Meituan).",
        thumb="q33_takeout_app.jpg", scene_label="DAY1 13:35:54 · phone in hand, kitchen",
        scene_caption_text="Spatial triple <code>(I, located_in, kitchen)</code> anchors the moment so the LLM stops drifting toward shopping-app priors.",
        scene_caption_small="grounding: spatial_extraction chunk DAY1 13:34:29",
    ),
    Case(
        slide_id="q53-hot-pot-v2", direction="UP", deck_index=3, deck_total=9,
        case_id="Q53", qtype="EntityLog", moment_label="A1_JAKE · DAY1 17:36 · EntityLog (Hema Fresh)",
        question="Who bought the hot pot base on the table?",
        choices={"A": "Shure", "B": "Me", "C": "Tasha", "D": "Lucia"}, gold="A",
        baseline_letter="B", baseline_short="Me",
        baseline_caption='"I put down the item in my hand... Alice says, ‘This is 5 yuan.’ ‘That one is 11 yuan, right?’ I ask. Alice replies, ‘Let\'s buy this one.’ ‘Did we buy eggs?’ I ask. Shure says, ‘We...’"',
        baseline_actor_note='The wearer talks about every item. The agent picks "Me" because the wearer narrates the most.',
        plus_label="add ONE",
        spatial_letter="A", spatial_short="Shure",
        triple_subj="Shure", triple_pred="located_in", triple_obj="hot-pot shelf",
        spatial_reading="Shure stands at the hot-pot shelf while the wearer is one shelf over with milk and drinks. The buyer at the shelf, not the narrator, is the answer.",
        thumb="q53_supermarket.jpg", scene_label="DAY1 17:36:49 · Hema Fresh aisle",
        scene_caption_text="Spatial co-locates <b>Shure</b> with the hot-pot base shelf. The narrator who handles everything verbally is <b>not</b> the buyer.",
        scene_caption_small="grounding: spatial_extraction chunk DAY1 17:31:29",
    ),
    Case(
        slide_id="p6-meeting-room-floor-v2", direction="UP", deck_index=4, deck_total=9,
        case_id="P6", qtype="place_lookup (synthetic)", moment_label="A1_JAKE · DAY1 · synthetic WHERE-only",
        question="Which floor of the house does the meeting room sit on?",
        choices={"A": "first_floor", "B": "second_floor", "C": "upstairs", "D": "courtyard"}, gold="A",
        baseline_letter="B", baseline_short="second_floor",
        baseline_caption='Episodic captions mention "second floor" frequently for other rooms ("the upstairs living room", "Katrina goes upstairs"). No episodic line says the meeting room itself is on the first floor.',
        baseline_actor_note='Without a direct place-to-place fact, the agent uses the most common floor mention as a prior.',
        plus_label="add ONE",
        spatial_letter="A", spatial_short="first_floor",
        triple_subj="meeting_room", triple_pred="located_in", triple_obj="first_floor",
        spatial_reading="A direct place-to-place fact, never expressed by any episodic action verb. One triple = full answer.",
        is_synthetic=True,
    ),
    Case(
        slide_id="n1-kitchen-proximity-v2", direction="UP", deck_index=5, deck_total=9,
        case_id="N1", qtype="proximity (synthetic)", moment_label="A1_JAKE · DAY1 · synthetic proximity",
        question="In the kitchen, who is the wearer most often next to (single companion)?",
        choices={"A": "Tasha", "B": "Shure", "C": "Lucia", "D": "Katrina"}, gold="B",
        baseline_letter="D", baseline_short="Katrina",
        baseline_caption='Episodic captures who talks the most. Katrina is loud across multiple kitchen scenes ("Katrina says...", "Katrina asks..."). Frequency heuristic → Katrina.',
        baseline_actor_note='Talking-most is not the same as standing-next-to-most.',
        plus_label="add ONE",
        spatial_letter="B", spatial_short="Shure",
        triple_subj="I (wearer)", triple_pred="next_to", triple_obj="Shure",
        spatial_reading="Direct proximity facts in kitchen chunks pile up on Shure, not Katrina. Episodic has no action verb that says 'standing beside'.",
        is_synthetic=True,
    ),
    Case(
        slide_id="c1-kitchen-contains-v2", direction="UP", deck_index=6, deck_total=9,
        case_id="C1", qtype="place_composition (synthetic)", moment_label="A1_JAKE · DAY1 · synthetic place-composition",
        question="What does the kitchen contain according to the recorded spatial layout?",
        choices={"A": "light", "B": "fridge", "C": "blender", "D": "oven"}, gold="A",
        baseline_letter="B", baseline_short="fridge",
        baseline_caption='No episodic caption literally lists kitchen contents; the model leans on a strong world prior ("kitchens have fridges").',
        baseline_actor_note='World priors win when the dataset is silent — but the actual recording captured a kitchen-light remark, not a fridge mention.',
        plus_label="add ONE",
        spatial_letter="A", spatial_short="light",
        triple_subj="kitchen", triple_pred="contains", triple_obj="light",
        spatial_reading="An observed-fact triple. World-knowledge priors say 'fridge', but the spatial layer records what was actually in the captioned scene.",
        is_synthetic=True,
    ),
    Case(
        slide_id="q6-grow-flowers-v2", direction="DN", deck_index=1, deck_total=3,
        case_id="Q6", qtype="TaskMaster (future plan)", moment_label="A1_JAKE · DAY1 11:34 · TaskMaster",
        question="Who plans to grow flowers",
        choices={"A": "Alice", "B": "Tasha", "C": "Me", "D": "Katrina"}, gold="D",
        baseline_letter="D", baseline_short="Katrina",
        baseline_caption='"I carry a cardboard box to the dining table... <em>Katrina says, ‘I was thinking of planting something. That fits the Earth Day theme. Flowers from Yunnan.’</em>"',
        baseline_actor_note='The plan-author is literally named in dialogue. Episodic gets it right.',
        plus_label="adding extra",
        spatial_letter="A", spatial_short="Alice",
        triple_subj="Shure", triple_pred="next_to", triple_obj="Alice",
        spatial_reading="Spatial co-locates Alice with the flowers and the box. Agent confuses physical proximity to flowers with planning intent.",
        thumb="q6_restaurant_flowers.jpg", scene_label="DAY1 11:34:30 · restaurant scene",
        scene_caption_text="Spatial surfaces true facts (Alice unpacks the box of flowers) that are <b>irrelevant</b> to the planning question. The planner was Katrina, who voiced the idea earlier.",
        scene_caption_small="failure mode: physical-proximity ≠ planning-intent",
        headline_left="right", headline_right="wrong",
        headline_phrase="One extra spatial fact",
        why_panel_label="the misleading triple",
    ),
    Case(
        slide_id="q45-coffee-timing-v2", direction="DN", deck_index=2, deck_total=3,
        case_id="Q45", qtype="EntityLog (temporal)", moment_label="A1_JAKE · DAY1 17:49 · EntityLog (temporal)",
        question="When was the last time we talked about coffee?",
        choices={"A": "About four hours ago", "B": "About eight hours ago", "C": "About an hour ago", "D": "One day ago"}, gold="A",
        baseline_letter="A", baseline_short="≈ 4 hr ago",
        baseline_caption='Episodic captions anchor the earlier coffee-ordering moment ~4 hrs back. The 3-axis baseline picks it correctly.',
        baseline_actor_note='Pure temporal reasoning: when was the topic discussed before now?',
        plus_label="adding extra",
        spatial_letter="D", spatial_short="≈ 1 day ago",
        triple_subj="I (wearer)", triple_pred="located_in", triple_obj="kitchen",
        spatial_reading="Spatial floods the prompt with kitchen anchors. Agent over-generalises: 'kitchen activities happen daily, so the relevant coffee mention must be old'.",
        thumb="q33_takeout_app.jpg", scene_label="DAY1 13:35 · kitchen anchors dominate prompt",
        scene_caption_text="The spatial triples are <b>true</b> but <b>irrelevant</b> to a temporal question. The extra context drags the answer to 'a day ago' instead of '4 hours'.",
        scene_caption_small="failure mode: irrelevant-but-loud kitchen anchors derail temporal reasoning",
        headline_left="right", headline_right="wrong",
        headline_phrase="One extra spatial fact",
        why_panel_label="the misleading triple",
    ),
    Case(
        slide_id="q50-phone-habit-v2", direction="DN", deck_index=3, deck_total=3,
        case_id="Q50", qtype="HabitInsight", moment_label="A1_JAKE · DAY1 · HabitInsight",
        question="What do I usually show everyone on my phone?",
        choices={"A": "Share interesting videos", "B": "Show what they want to eat", "C": "Recording a video", "D": "Take a look at the timer interface"}, gold="D",
        baseline_letter="D", baseline_short="timer interface",
        baseline_caption='"I keep operating my phone and show everyone the timer."  The episodic capture states the specific action.',
        baseline_actor_note='Direct evidence wins on the 3-axis side.',
        plus_label="adding extra",
        spatial_letter="A", spatial_short="share videos",
        triple_subj="I (wearer)", triple_pred="on", triple_obj="Katrina's_phone",
        spatial_reading="Spatial surfaces many people-on-phone triples across the day. Agent generalises 'group phone use' → 'sharing interesting videos'.",
        thumb="q50_phone_timer.jpg", scene_label="DAY1 12:23:43 · show the timer",
        scene_caption_text="Spatial dilutes the specific timer mention with broad people-with-phones context. The content of the phone screen is lost.",
        scene_caption_small="failure mode: pattern-induction overrides specific evidence",
        headline_left="right", headline_right="wrong",
        headline_phrase="One extra spatial fact",
        why_panel_label="the misleading triple",
    ),
    Case(
        slide_id="new-where-am-i-v2", direction="UP", deck_index=7, deck_total=9,
        case_id="NEW-1", qtype="WHERE-am-I (authored scenario)",
        moment_label="A1_JAKE · DAY1 · new spatial scenario authored from extracted triples",
        question="It is mid-afternoon and the wearer has just been wiping a whiteboard. Where is the wearer standing right now?",
        choices={"A": "courtyard", "B": "kitchen", "C": "bedroom", "D": "Hema Fresh"}, gold="B",
        baseline_letter="A", baseline_short="courtyard",
        baseline_caption='Episodic captions describe wiping the whiteboard, conversations about coffee and lunch, and group activity. Lots of "we" and dialogue. No caption literally says <em>where</em> the wearer is right now.',
        baseline_actor_note='Action-rich captions, location-poor. The agent guesses an outdoor place because of an earlier courtyard mention.',
        plus_label="add ONE",
        spatial_letter="B", spatial_short="kitchen",
        triple_subj="I (wearer)", triple_pred="located_in", triple_obj="kitchen",
        spatial_reading="A single direct WHERE-fact closes the question. No episodic action verb encodes the wearer's current room as cleanly as one spatial triple.",
        thumb="q33_takeout_app.jpg", scene_label="DAY1 13:35 · same moment as Q33 / Q45",
        scene_caption_text="Authored scenario showing the cleanest possible spatial gain: a pure <b>WHERE-am-I</b> question. Episodic has the action; only spatial has the location as a fact.",
        scene_caption_small="grounding: spatial_extraction chunk DAY1 13:34:29  ·  (I, located_in, kitchen) appears 4 different chunks",
        is_new_scenario=True,
    ),
]


def main() -> None:
    out_dir = Path("docs/slides")
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for c in CASES:
        out = out_dir / f"{c.slide_id}.html"
        out.write_text(render(c))
        written.append(out)
        print(f"wrote {out}")
    print(f"\n{len(written)} V2 slides written.")


if __name__ == "__main__":
    main()
