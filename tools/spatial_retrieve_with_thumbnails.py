#!/usr/bin/env python3
"""Run a spatial query and emit a self-contained HTML viewer with frame thumbnails.

Bridges the existing SpatialMemory.retrieve(query) text output with the
thumbnail_extractor so each retrieved triple is rendered alongside the actual
video frame captured at its timestamp.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from worldmm.embedding import EmbeddingModel
from worldmm.memory.spatial import SpatialMemory

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from thumbnail_extractor import extract_thumbnail


PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Spatial retrieval · {query}</title>
<style>
  :root{{
    --bg:#0b1020; --panel:#11192f; --line:#1c2746; --ink:#e9ecf5; --muted:#8aa0c8;
    --accent:#60a5fa; --good:#22c55e; --hl:#fbbf24;
    --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
    --sans:Inter,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
  }}
  *{{box-sizing:border-box}}
  html,body{{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans)}}
  .wrap{{max-width:1400px;margin:0 auto;padding:30px}}
  header{{display:flex;justify-content:space-between;align-items:baseline;border-bottom:1px solid var(--line);padding-bottom:10px;margin-bottom:18px}}
  header h1{{margin:0;font-size:22px;font-weight:600}}
  header .meta{{color:var(--muted);font-size:13px;font-family:var(--mono)}}
  .query{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 18px;margin-bottom:18px;display:flex;gap:14px;align-items:center}}
  .query .tag{{color:var(--accent);font-family:var(--mono);font-size:12px;text-transform:uppercase;letter-spacing:1px}}
  .query .q{{font-size:18px;font-weight:600}}
  .stats{{color:var(--muted);font-size:12.5px;font-family:var(--mono);margin-bottom:10px}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px}}
  .card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden;display:flex;flex-direction:column}}
  .card .thumb-wrap{{aspect-ratio:16/9;background:#06090f;display:flex;align-items:center;justify-content:center;border-bottom:1px solid var(--line)}}
  .card img{{width:100%;height:100%;object-fit:cover;display:block}}
  .card .none{{color:var(--muted);font-size:12px;font-family:var(--mono);text-align:center}}
  .card .body{{padding:10px 12px;display:flex;flex-direction:column;gap:6px}}
  .card .triple{{font-family:var(--mono);font-size:14px;color:var(--ink)}}
  .card .triple .s{{color:var(--accent)}}
  .card .triple .p{{color:var(--good);margin:0 4px}}
  .card .triple .o{{color:var(--hl)}}
  .card .ts{{color:var(--muted);font-size:11.5px;font-family:var(--mono);display:flex;justify-content:space-between}}
  .card .place{{color:var(--muted);font-size:11.5px}}
  .card .place b{{color:#cfd8ee}}
  footer{{color:var(--muted);font-size:11.5px;font-family:var(--mono);margin-top:24px;border-top:1px solid var(--line);padding-top:10px}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <div style="color:var(--accent);font-family:var(--mono);font-size:12px;text-transform:uppercase;letter-spacing:1px">spatial retrieval · with thumbnails</div>
      <h1>{query_h1}</h1>
    </div>
    <div class="meta">{subject} · {day} · {date}<br>{model}</div>
  </header>

  <div class="query">
    <span class="tag">query</span>
    <span class="q">{query_h1}</span>
  </div>

  <div class="stats">{stats}</div>

  <div class="grid">
    {cards}
  </div>

  <footer>{footer}</footer>
</div>
</body>
</html>
"""

CARD_WITH_THUMB = """<div class="card">
  <div class="thumb-wrap"><img src="data:image/jpeg;base64,{thumb_b64}" alt="frame thumbnail"></div>
  <div class="body">
    <div class="triple"><span class="s">{s}</span><span class="p">{p}</span><span class="o">{o}</span></div>
    <div class="ts"><span>ts {ts_int}</span><span>{ts_fmt}</span></div>
    {place_line}
  </div>
</div>"""

CARD_NO_THUMB = """<div class="card">
  <div class="thumb-wrap"><div class="none">no clip for this timestamp</div></div>
  <div class="body">
    <div class="triple"><span class="s">{s}</span><span class="p">{p}</span><span class="o">{o}</span></div>
    <div class="ts"><span>ts {ts_int}</span><span>{ts_fmt}</span></div>
    {place_line}
  </div>
</div>"""


def fmt_ts(ts_int: int) -> str:
    s = str(ts_int).zfill(9)
    return f"DAY{s[0]} {s[1:3]}:{s[3:5]}:{s[5:7]}.{s[7:]}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--spatial-file", default="output/metadata/spatial_memory/A1_JAKE/spatial_consolidation_results_chatgpt-gpt-5.4.json")
    parser.add_argument("--extraction-file", default="output/metadata/spatial_memory/A1_JAKE/spatial_extraction_results_chatgpt-gpt-5.4.json",
                        help="Per-chunk extraction JSON used to recover the FIRST-OBSERVED timestamp for each retrieved triple (consolidation accumulates and reports only the latest chunk).")
    parser.add_argument("--video-dir", default="data/EgoLife/A1_JAKE/DAY1")
    parser.add_argument("--thumb-dir", default="output/thumbnails/A1_JAKE/DAY1")
    parser.add_argument("--out-html", default="output/spatial_query_thumbnails.html")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--subject", default="A1_JAKE")
    parser.add_argument("--day", default="DAY1")
    args = parser.parse_args()

    embedding_model = EmbeddingModel(text_model_name=args.embedding_model, device="cpu")
    embedding_model.load_model(model_type="text")

    sm = SpatialMemory(embedding_model=embedding_model)
    sm.load_triples_from_file(args.spatial_file)
    sm.index(until_time=10**12)
    entries = sm.retrieve(args.query, top_k=args.top_k, as_context=False) or []

    triple_first_seen: Dict[tuple, int] = {}
    if Path(args.extraction_file).exists():
        with open(args.extraction_file) as f:
            ext = json.load(f)
        for chunk_ts_str in sorted(ext.get("spatial_triples", {}).keys()):
            chunk_ts = int(chunk_ts_str)
            for triple in ext["spatial_triples"][chunk_ts_str]:
                if len(triple) != 3:
                    continue
                key = (triple[0], triple[1], triple[2])
                triple_first_seen.setdefault(key, chunk_ts)

    video_dir = Path(args.video_dir)
    thumb_dir = Path(args.thumb_dir)

    cards: List[str] = []
    n_with = n_without = 0
    for e in entries:
        key = (e.subject, e.predicate, e.object)
        ts_int = int(triple_first_seen.get(key, e.timestamp))
        thumb_path = extract_thumbnail(ts_int, video_dir, thumb_dir)
        place_line = (
            f'<div class="place">place=<b>{html.escape(e.place)}</b></div>' if e.place else ""
        )
        common = dict(
            s=html.escape(e.subject),
            p=html.escape(e.predicate),
            o=html.escape(e.object),
            ts_int=ts_int,
            ts_fmt=fmt_ts(ts_int),
            place_line=place_line,
        )
        if thumb_path and thumb_path.exists():
            with open(thumb_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            cards.append(CARD_WITH_THUMB.format(thumb_b64=b64, **common))
            n_with += 1
        else:
            cards.append(CARD_NO_THUMB.format(**common))
            n_without += 1

    page = PAGE_TEMPLATE.format(
        query=html.escape(args.query),
        query_h1=html.escape(args.query),
        subject=args.subject,
        day=args.day,
        date="DAY1 11:00 – 22:00 (≈10 h egocentric footage)",
        model=args.embedding_model,
        stats=f"retrieved {len(entries)} triples · {n_with} with thumbnail · {n_without} clip missing",
        cards="\n    ".join(cards) or "<div class='stats'>no triples retrieved</div>",
        footer="worldmm · spatial axis (closed-vocab triples + igraph PPR) · frame thumbnails sampled at the triple's chunk timestamp",
    )
    Path(args.out_html).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_html).write_text(page)
    print(f"wrote {args.out_html}  ({len(entries)} triples, {n_with} thumbnails)")


if __name__ == "__main__":
    main()
