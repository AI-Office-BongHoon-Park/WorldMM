#!/usr/bin/env python3
"""Generate a one-day narrative summary that fuses all four WorldMM memories.

Inputs: the per-day 1h episodic captions, consolidated semantic triples,
consolidated spatial triples. Outputs an HTML report (and a JSON snapshot
of what the LLM was given) tying the day's narrative to a small set of
illustrative thumbnails sampled at hour boundaries from real MP4s.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from worldmm.llm import LLMModel

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from thumbnail_extractor import extract_thumbnail


PROMPT = """You are summarising one full day of egocentric video memory for a research-paper figure.

You are given:
  - HOURLY CAPTIONS: a concise first-person summary per hour of the day.
  - SEMANTIC TRIPLES: consolidated (subject, predicate, object) facts about people, habits, and roles.
  - SPATIAL TRIPLES: closed-vocab WHERE-facts (located_in, on, next_to, contains, ...).

Produce a JSON object with EXACTLY these keys:
  "title": <string, one short factual title for the day>,
  "synopsis": <string, 4-6 sentences narrating the arc of the day in first person ("I ...") using actual scene names from the captions>,
  "key_places": <list of 3-6 place names that actually appeared, ordered chronologically as visited>,
  "key_people": <list of 3-6 people other than I who showed up>,
  "highlights": <list of 4-6 objects each {"hour": "HH:00", "moment": <short sentence>, "memory": "episodic"|"semantic"|"spatial"|"mixed"}>,
  "spatial_signal_facts": <list of 3-5 WHERE-facts strictly grounded in the SPATIAL TRIPLES input>

Output ONLY a single JSON object. No prose, no markdown fences.
"""


class DaySummary:
    title: str
    synopsis: str
    key_places: List[str]
    key_people: List[str]
    highlights: List[Dict[str, str]]
    spatial_signal_facts: List[str]


def load_hourly_captions(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def load_consolidated_triples(path: Path, key: str) -> List[List[str]]:
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    last_ts = sorted(data.keys())[-1] if data else None
    if not last_ts:
        return []
    return data[last_ts].get(key, [])


def fmt_hourly(captions: List[Dict[str, Any]]) -> str:
    lines = []
    for c in captions:
        start = str(c.get("start_time", "?")).zfill(8)
        text = c.get("text", "").strip()
        if not text:
            continue
        hh = start[:2]
        lines.append(f"  {hh}:00 -> {text}")
    return "\n".join(lines)


def fmt_triples(triples: List[List[str]], cap: int = 80) -> str:
    out = []
    for t in triples[:cap]:
        if len(t) == 3:
            out.append(f"  ({t[0]}, {t[1]}, {t[2]})")
    return "\n".join(out)


def render_html(
    summary: Dict[str, Any],
    thumbnails: List[tuple],
    subject: str,
    day: str,
    model_name: str,
) -> str:
    def esc(x) -> str:
        return html.escape(str(x))

    thumb_html_parts = []
    for hour_label, ts_int, thumb_path in thumbnails:
        if thumb_path and Path(thumb_path).exists():
            b64 = base64.b64encode(Path(thumb_path).read_bytes()).decode("ascii")
            thumb_html_parts.append(
                f'<div class="hour"><div class="hour-tag">{esc(hour_label)}</div>'
                f'<img src="data:image/jpeg;base64,{b64}"></div>'
            )
        else:
            thumb_html_parts.append(
                f'<div class="hour"><div class="hour-tag">{esc(hour_label)}</div>'
                f'<div class="missing">no clip</div></div>'
            )
    thumbs_block = "\n      ".join(thumb_html_parts) or "<div class='missing'>no thumbnails available</div>"

    places = ", ".join(esc(p) for p in summary.get("key_places", []))
    people = ", ".join(esc(p) for p in summary.get("key_people", []))

    highlights_block = "\n        ".join(
        f'<li><span class="mem-{esc(h.get("memory","mixed"))}">[{esc(h.get("memory","mixed"))}]</span> '
        f'<b>{esc(h.get("hour",""))}</b> — {esc(h.get("moment",""))}</li>'
        for h in summary.get("highlights", [])
    )
    spatial_block = "\n        ".join(
        f"<li><code>{esc(f)}</code></li>" for f in summary.get("spatial_signal_facts", [])
    )

    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Day Summary · {esc(subject)} · {esc(day)}</title>
<style>
  :root{{
    --bg:#0b1020; --panel:#11192f; --line:#1c2746; --ink:#e9ecf5; --muted:#8aa0c8;
    --accent:#60a5fa; --good:#22c55e; --hl:#fbbf24; --orange:#f97316;
    --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
    --sans:Inter,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
  }}
  *{{box-sizing:border-box}}
  html,body{{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:15px}}
  .wrap{{max-width:1280px;margin:0 auto;padding:32px}}
  header{{display:flex;justify-content:space-between;align-items:baseline;border-bottom:1px solid var(--line);padding-bottom:12px;margin-bottom:22px}}
  header h1{{margin:0;font-size:28px}}
  header .meta{{color:var(--muted);font-size:13px;font-family:var(--mono);text-align:right}}
  .synopsis{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px 22px;line-height:1.65;margin-bottom:20px;font-size:16px}}
  .grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:22px}}
  .card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 18px}}
  .card h2{{margin:0 0 10px;font-size:13px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:1px}}
  .pill{{display:inline-block;background:#0a1429;border:1px solid var(--line);border-radius:999px;padding:3px 12px;margin:3px;font-size:13px}}
  .pill.place{{color:var(--accent)}}
  .pill.person{{color:var(--hl)}}
  ul{{margin:0;padding-left:18px;line-height:1.7}}
  code{{font-family:var(--mono);font-size:13px;color:#cfd8ee}}
  .mem-episodic{{color:var(--accent);font-family:var(--mono);font-size:11px;margin-right:6px}}
  .mem-semantic{{color:var(--good);font-family:var(--mono);font-size:11px;margin-right:6px}}
  .mem-spatial{{color:var(--orange);font-family:var(--mono);font-size:11px;margin-right:6px}}
  .mem-mixed{{color:var(--muted);font-family:var(--mono);font-size:11px;margin-right:6px}}
  .timeline{{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}}
  .hour{{display:flex;flex-direction:column;gap:4px}}
  .hour-tag{{font-family:var(--mono);font-size:11.5px;color:var(--muted)}}
  .hour img{{width:100%;aspect-ratio:16/9;object-fit:cover;border-radius:6px;display:block}}
  .hour .missing{{aspect-ratio:16/9;background:#06090f;border:1px dashed var(--line);border-radius:6px;display:flex;align-items:center;justify-content:center;color:var(--muted);font-family:var(--mono);font-size:11px}}
  footer{{color:var(--muted);font-size:12px;font-family:var(--mono);margin-top:24px;border-top:1px solid var(--line);padding-top:10px}}
</style></head><body>
<div class="wrap">
  <header>
    <div>
      <div style="color:var(--accent);font-family:var(--mono);font-size:12px;text-transform:uppercase;letter-spacing:1px">worldmm · day summary</div>
      <h1>{esc(summary.get("title", "Day summary"))}</h1>
    </div>
    <div class="meta">{esc(subject)} · {esc(day)}<br>{esc(model_name)}</div>
  </header>

  <div class="synopsis">{esc(summary.get("synopsis", ""))}</div>

  <h2 style="font-size:13px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin:0 0 10px">visual timeline (hourly thumbnails)</h2>
  <div class="timeline">
    {thumbs_block}
  </div>

  <div class="grid" style="margin-top:22px">
    <div class="card">
      <h2>key places (chronological)</h2>
      <div>{places or "<i>none surfaced</i>"}</div>
    </div>
    <div class="card">
      <h2>key people</h2>
      <div>{people or "<i>none surfaced</i>"}</div>
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <h2>highlights by memory axis</h2>
      <ul>
        {highlights_block or "<li><i>no highlights</i></li>"}
      </ul>
    </div>
    <div class="card">
      <h2>spatial-grounded facts</h2>
      <ul>
        {spatial_block or "<li><i>no spatial facts</i></li>"}
      </ul>
    </div>
  </div>

  <footer>{esc(subject)} · {esc(day)} · fused episodic (hourly), semantic, spatial · respond model={esc(model_name)}</footer>
</div>
</body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodic-1h", default="data/EgoLife/EgoLifeCap/A1_JAKE/A1_JAKE_1h.json")
    parser.add_argument("--semantic-file", default="output/metadata/semantic_memory/A1_JAKE/semantic_consolidation_results_chatgpt-gpt-5.4.json")
    parser.add_argument("--spatial-file", default="output/metadata/spatial_memory/A1_JAKE/spatial_consolidation_results_chatgpt-gpt-5.4.json")
    parser.add_argument("--video-dir", default="data/EgoLife/A1_JAKE/DAY1")
    parser.add_argument("--thumb-dir", default="output/thumbnails/A1_JAKE/DAY1")
    parser.add_argument("--out-html", default="output/day_summary_A1_JAKE_DAY1.html")
    parser.add_argument("--out-json", default="output/day_summary_A1_JAKE_DAY1.json")
    parser.add_argument("--model", default="chatgpt-gpt-5.4")
    parser.add_argument("--subject", default="A1_JAKE")
    parser.add_argument("--day", default="DAY1")
    parser.add_argument("--max-semantic", type=int, default=120)
    parser.add_argument("--max-spatial", type=int, default=120)
    args = parser.parse_args()

    captions = load_hourly_captions(Path(args.episodic_1h))
    semantic = load_consolidated_triples(Path(args.semantic_file), "consolidated_semantic_triples")
    spatial  = load_consolidated_triples(Path(args.spatial_file),  "consolidated_spatial_triples")

    user_msg = f"""HOURLY CAPTIONS:
{fmt_hourly(captions)}

SEMANTIC TRIPLES (top {args.max_semantic}):
{fmt_triples(semantic, args.max_semantic)}

SPATIAL TRIPLES (top {args.max_spatial}):
{fmt_triples(spatial, args.max_spatial)}
"""

    llm = LLMModel(model_name=args.model)
    response_text = llm.generate([
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": user_msg},
    ])
    text = response_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    summary = json.loads(text)

    video_dir = Path(args.video_dir)
    thumb_dir = Path(args.thumb_dir)
    thumbnails = []
    for c in captions:
        start = str(c.get("start_time", "0")).zfill(8)
        if not c.get("text"):
            continue
        day_prefix = str(c.get("date", "DAY1")).replace("DAY", "").replace("Day", "")
        ts_int = int(day_prefix + start)
        thumb = extract_thumbnail(ts_int, video_dir, thumb_dir, max_side=360)
        hour_label = f"{start[:2]}:00"
        thumbnails.append((hour_label, ts_int, thumb))

    html_doc = render_html(summary, thumbnails, args.subject, args.day, args.model)
    Path(args.out_html).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_html).write_text(html_doc)

    Path(args.out_json).write_text(json.dumps({
        "subject": args.subject,
        "day": args.day,
        "model": args.model,
        "summary": summary,
        "thumbnails": [{"hour": h, "ts": t, "path": str(p) if p else None} for h, t, p in thumbnails],
    }, indent=2, ensure_ascii=False))

    print(f"summary -> {args.out_html}")
    print(f"json    -> {args.out_json}")
    print(f"title   : {summary.get('title', '?')}")
    print(f"places  : {summary.get('key_places', [])}")
    print(f"people  : {summary.get('key_people', [])}")


if __name__ == "__main__":
    main()
