#!/usr/bin/env python3
"""Spatial-axis-only daily trajectory summary.

The point: this report is something the *spatial* axis alone can produce that
the other three cannot. Episodic captures actions ("I walked to the kitchen"),
semantic captures roles, visual captures pixels — none of them yield a clean
chronological list of "where the wearer was, with whom, and what was near them"
without a downstream synthesis step. The spatial extraction JSON already has
exactly that, chunk by chunk; this script renders it.

Outputs:
  - output/spatial_trajectory_<subject>_<day>.html  (timeline + per-stop card)
  - output/spatial_trajectory_<subject>_<day>.json  (machine-readable trajectory)
"""

from __future__ import annotations

import argparse
import base64
import html
import json
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from thumbnail_extractor import extract_thumbnail


WEARER_TOKENS = {"I", "i", "me", "my", "wearer"}
NON_PLACE_OBJECTS = {
    "table", "stool", "sofa", "chair", "shelf", "desk", "counter", "sink",
    "bench", "windowsill", "doorway", "foyer", "front", "seat", "bed",
    "board", "whiteboard", "the_board", "second-floor_living_room",
    "second_floor_living_room", "computer_room",
}
PLACE_PREDICATES = {"located_in", "in"}


@dataclass
class TrajectoryStop:
    chunk_ts: int
    place: str
    companions: List[str] = field(default_factory=list)
    nearby_objects: List[str] = field(default_factory=list)
    sample_triples: List[Tuple[str, str, str]] = field(default_factory=list)
    thumbnail_path: Optional[str] = None


def fmt(ts_int: int) -> str:
    s = str(ts_int).zfill(9)
    return f"DAY{s[0]} {s[1:3]}:{s[3:5]}:{s[5:7]}"


def is_place(token: str) -> bool:
    return bool(token) and token not in NON_PLACE_OBJECTS


def pick_chunk_place(triples: List[List[str]]) -> Optional[str]:
    candidates: List[str] = []
    for t in triples:
        if len(t) != 3:
            continue
        s, p, o = t
        if s in WEARER_TOKENS and p in PLACE_PREDICATES and is_place(o):
            candidates.append(o)
    if not candidates:
        return None
    counts = Counter(candidates)
    return counts.most_common(1)[0][0]


def gather_chunk_context(triples: List[List[str]], place: str) -> Tuple[List[str], List[str], List[Tuple[str, str, str]]]:
    companions: Set[str] = set()
    objects: Set[str] = set()
    sample: List[Tuple[str, str, str]] = []
    for t in triples:
        if len(t) != 3:
            continue
        s, p, o = t
        if s in WEARER_TOKENS and p in {"next_to", "near", "behind", "in_front_of", "left_of", "right_of"}:
            if o not in WEARER_TOKENS and o not in NON_PLACE_OBJECTS:
                companions.add(o)
        if p == "on" and s in WEARER_TOKENS and o not in WEARER_TOKENS:
            objects.add(o)
        if p == "on" and o in WEARER_TOKENS:
            objects.add(s)
        if len(sample) < 5:
            sample.append((s, p, o))
    return sorted(companions), sorted(objects), sample


def collapse_consecutive(stops: List[TrajectoryStop]) -> List[TrajectoryStop]:
    out: List[TrajectoryStop] = []
    for s in stops:
        if out and out[-1].place == s.place:
            companions = set(out[-1].companions) | set(s.companions)
            objects = set(out[-1].nearby_objects) | set(s.nearby_objects)
            out[-1].companions = sorted(companions)
            out[-1].nearby_objects = sorted(objects)
            continue
        out.append(s)
    return out


def build_trajectory(extraction_json: Path) -> List[TrajectoryStop]:
    with open(extraction_json) as f:
        data = json.load(f)
    stops: List[TrajectoryStop] = []
    for chunk_ts_str in sorted(data.get("spatial_triples", {}).keys()):
        triples = data["spatial_triples"][chunk_ts_str]
        place = pick_chunk_place(triples)
        if not place:
            continue
        companions, objects, sample = gather_chunk_context(triples, place)
        stops.append(TrajectoryStop(
            chunk_ts=int(chunk_ts_str),
            place=place,
            companions=companions,
            nearby_objects=objects,
            sample_triples=sample,
        ))
    return collapse_consecutive(stops)


def attach_thumbnails(
    stops: List[TrajectoryStop],
    video_dir: Path,
    thumb_dir: Path,
    max_side: int = 360,
) -> None:
    for s in stops:
        p = extract_thumbnail(s.chunk_ts, video_dir, thumb_dir, max_side=max_side)
        if p is not None:
            s.thumbnail_path = str(p)


PLACE_PALETTE = {
    "kitchen": "#22c55e",
    "bedroom": "#a855f7",
    "restaurant": "#fbbf24",
    "supermarket": "#60a5fa",
    "Hema Fresh": "#3b82f6",
    "courtyard": "#84cc16",
    "first_floor": "#06b6d4",
    "second_floor": "#0ea5e9",
    "second-floor_living_room": "#0ea5e9",
    "second_floor_living_room": "#0ea5e9",
    "upstairs": "#0ea5e9",
    "downstairs": "#06b6d4",
    "outside": "#f97316",
    "meeting_room": "#8b5cf6",
    "computer_room": "#6366f1",
    "store": "#3b82f6",
    "dorm": "#ef4444",
}


def colour_for(place: str) -> str:
    if place in PLACE_PALETTE:
        return PLACE_PALETTE[place]
    h = abs(hash(place)) % 360
    return f"hsl({h},65%,55%)"


def render_html(stops: List[TrajectoryStop], subject: str, day: str) -> str:
    def esc(x: object) -> str:
        return html.escape(str(x))

    ribbon_segments = []
    total = max(1, len(stops))
    for s in stops:
        c = colour_for(s.place)
        ribbon_segments.append(
            f'<div class="seg" style="background:{c};flex:1" '
            f'title="{esc(fmt(s.chunk_ts))} · {esc(s.place)}"></div>'
        )

    cards = []
    for s in stops:
        thumb_html = ""
        if s.thumbnail_path and Path(s.thumbnail_path).exists():
            b64 = base64.b64encode(Path(s.thumbnail_path).read_bytes()).decode()
            thumb_html = f'<img src="data:image/jpeg;base64,{b64}">'
        else:
            thumb_html = '<div class="no-thumb">no covering video clip</div>'

        companions_html = ", ".join(esc(c) for c in s.companions) if s.companions else "<i>alone</i>"
        objects_html = ", ".join(esc(o) for o in s.nearby_objects) if s.nearby_objects else "<i>—</i>"
        sample_html = "".join(
            f'<div class="triple">({esc(t[0])}, <span class="p">{esc(t[1])}</span>, {esc(t[2])})</div>'
            for t in s.sample_triples
        )
        cards.append(
            f'<div class="stop">'
            f'  <div class="stop-thumb">{thumb_html}</div>'
            f'  <div class="stop-body">'
            f'    <div class="stop-head">'
            f'      <span class="pill" style="background:{colour_for(s.place)}">{esc(s.place)}</span>'
            f'      <span class="ts">{esc(fmt(s.chunk_ts))}</span>'
            f'    </div>'
            f'    <div class="row"><span class="k">with</span><span>{companions_html}</span></div>'
            f'    <div class="row"><span class="k">objects</span><span>{objects_html}</span></div>'
            f'    <div class="triples">{sample_html}</div>'
            f'  </div>'
            f'</div>'
        )

    unique_places = list(OrderedDict.fromkeys(s.place for s in stops))
    legend = " · ".join(
        f'<span class="legend-pill" style="background:{colour_for(p)}">{esc(p)}</span>'
        for p in unique_places
    )

    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Spatial Trajectory · {esc(subject)} · {esc(day)}</title>
<style>
  :root{{
    --bg:#0b1020; --panel:#11192f; --panel2:#0e1730; --line:#1c2746; --ink:#e9ecf5; --muted:#8aa0c8;
    --accent:#60a5fa; --hl:#fbbf24; --good:#22c55e;
    --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
    --sans:Inter,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
  }}
  *{{box-sizing:border-box}} html,body{{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:14px}}
  .wrap{{max-width:1280px;margin:0 auto;padding:24px}}
  header{{display:flex;justify-content:space-between;align-items:baseline;border-bottom:1px solid var(--line);padding-bottom:10px;margin-bottom:14px}}
  header h1{{margin:0;font-size:24px;font-weight:700}}
  header .tag{{color:var(--accent);font-family:var(--mono);font-size:11px;text-transform:uppercase;letter-spacing:1px}}
  header .meta{{color:var(--muted);font-family:var(--mono);font-size:12px;text-align:right}}

  .why{{background:var(--panel2);border:1px dashed var(--good);border-radius:10px;padding:10px 14px;font-size:13px;color:#bbf7d0;margin-bottom:14px}}
  .why b{{color:#fff}}

  .ribbon{{display:flex;height:36px;border-radius:8px;overflow:hidden;border:1px solid var(--line);margin-bottom:8px}}
  .ribbon .seg{{height:100%;border-right:1px solid rgba(11,16,32,.7)}}
  .ribbon .seg:last-child{{border-right:none}}
  .legend{{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:18px;color:var(--muted);font-family:var(--mono);font-size:11px}}
  .legend-pill{{padding:2px 10px;border-radius:999px;color:#0a1325;font-weight:700}}

  .stops{{display:grid;gap:12px}}
  .stop{{display:grid;grid-template-columns:280px 1fr;gap:14px;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px;align-items:stretch}}
  .stop-thumb{{position:relative;background:#06090f;border:1px solid var(--line);border-radius:8px;overflow:hidden;aspect-ratio:16/9}}
  .stop-thumb img{{width:100%;height:100%;object-fit:cover;display:block}}
  .stop-thumb .no-thumb{{display:flex;align-items:center;justify-content:center;height:100%;color:var(--muted);font-family:var(--mono);font-size:11px}}
  .stop-body{{display:flex;flex-direction:column;gap:6px;min-width:0}}
  .stop-head{{display:flex;align-items:center;gap:10px}}
  .stop-head .pill{{padding:3px 12px;border-radius:999px;color:#062012;font-weight:800;font-size:13px}}
  .stop-head .ts{{color:var(--muted);font-family:var(--mono);font-size:12px}}
  .row{{display:flex;gap:8px;align-items:baseline;font-size:13px}}
  .row .k{{color:var(--muted);font-family:var(--mono);font-size:11px;text-transform:uppercase;width:64px;letter-spacing:.5px}}
  .row i{{color:var(--muted)}}
  .triples{{display:flex;flex-wrap:wrap;gap:6px;margin-top:4px}}
  .triple{{font-family:var(--mono);font-size:11.5px;color:#cfd8ee;background:rgba(0,0,0,.32);border:1px solid var(--line);border-radius:5px;padding:3px 8px}}
  .triple .p{{color:var(--good)}}

  footer{{color:var(--muted);font-family:var(--mono);font-size:11.5px;margin-top:18px;border-top:1px solid var(--line);padding-top:8px;display:flex;justify-content:space-between}}
</style></head><body>
<div class="wrap">
  <header>
    <div>
      <div class="tag">spatial axis · daily trajectory</div>
      <h1>Where the wearer was, in order, on {esc(day)}</h1>
    </div>
    <div class="meta">{esc(subject)} · 4th-axis-only narrative<br>{len(stops)} stops · {len(unique_places)} distinct places</div>
  </header>

  <div class="why">
    <b style="color:var(--hl)">ONLY spatial-axis facts were used to build this page.</b>
    Episodic captures actions ("I walked to the kitchen"), semantic captures roles, visual captures pixels — none of
    them yield a <i>chronological place trajectory</i> with co-presence and on/near object context as a structured
    fact list. The spatial-axis extraction does, chunk by chunk; this page just renders it. Thumbnails are pulled
    only where a covering MP4 was downloaded; they are illustrative, not part of the trajectory data.
  </div>

  <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;font-family:var(--mono);font-size:11px;color:var(--muted)">
    <span style="color:var(--good)">▶ morning start</span>
    <div class="ribbon" style="flex:1;margin:0">{"".join(ribbon_segments)}</div>
    <span style="color:var(--hl)">end of day ◀</span>
  </div>
  <div class="legend">{legend}</div>

  <div class="stops">
    {chr(10).join(cards)}
  </div>

  <footer>
    <div>worldmm · spatial axis · A1_JAKE DAY1 trajectory from spatial_extraction_results_chatgpt-gpt-5.4.json</div>
    <div>{len(stops)} stops · {sum(1 for s in stops if s.thumbnail_path)} with covering MP4 thumbnail</div>
  </footer>
</div>
</body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extraction-file",
        default="output/metadata/spatial_memory/A1_JAKE/spatial_extraction_results_chatgpt-gpt-5.4.json",
    )
    parser.add_argument("--video-dir", default="data/EgoLife/A1_JAKE/DAY1")
    parser.add_argument("--thumb-dir", default="output/thumbnails/A1_JAKE/DAY1")
    parser.add_argument("--subject", default="A1_JAKE")
    parser.add_argument("--day", default="DAY1")
    parser.add_argument(
        "--out-html",
        default="output/spatial_trajectory_A1_JAKE_DAY1.html",
    )
    parser.add_argument(
        "--out-json",
        default="output/spatial_trajectory_A1_JAKE_DAY1.json",
    )
    args = parser.parse_args()

    stops = build_trajectory(Path(args.extraction_file))
    attach_thumbnails(stops, Path(args.video_dir), Path(args.thumb_dir))

    Path(args.out_html).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_html).write_text(render_html(stops, args.subject, args.day))
    Path(args.out_json).write_text(
        json.dumps(
            {
                "subject": args.subject,
                "day": args.day,
                "stops": [asdict(s) for s in stops],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"wrote {args.out_html}  ({len(stops)} stops)")
    print(f"wrote {args.out_json}")
    for s in stops:
        print(f"  {fmt(s.chunk_ts)}  →  {s.place:<25} with={s.companions or '—'}")


if __name__ == "__main__":
    main()
