#!/usr/bin/env python3
"""Grounded GPT visual-triple retrieval with exact supporting-frame thumbnails."""

from __future__ import annotations

import argparse
import html
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from decord import VideoReader, cpu  # type: ignore[reportMissingImports]
from PIL import Image
from sentence_transformers import SentenceTransformer  # type: ignore[reportMissingImports]


@dataclass
class GroundedVisualHit:
    clip_id: str
    triple: str
    frame_idx: int
    frame_timestamp_s: float
    thumbnail_image: Path
    score: float
    f_provenance: str


def load_index(index_file: Path) -> Dict[str, Dict[str, Any]]:
    with index_file.open("rb") as handle:
        return pickle.load(handle)


def resize_long_side(image: Image.Image, max_side: int) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    long_side = max(width, height)
    if long_side <= max_side:
        return image
    scale = max_side / float(long_side)
    return image.resize((max(1, int(round(width * scale))), max(1, int(round(height * scale)))), Image.Resampling.LANCZOS)


def extract_sampled_frame(video_path: Path, frame_idx: int, max_side: int) -> Image.Image:
    reader = VideoReader(str(video_path), ctx=cpu(0))
    total = len(reader)
    if total <= 0:
        raise ValueError(f"empty video: {video_path}")
    sample_indices = np.linspace(0, total - 1, 16, dtype=int).tolist()
    exact_index = sample_indices[max(0, min(15, frame_idx))]
    return resize_long_side(Image.fromarray(reader[exact_index].asnumpy()), max_side)


def retrieve_visual_grounded(
    query: str,
    index: Dict[str, Dict[str, Any]],
    embedder: SentenceTransformer,
    top_k: int,
    out_dir: Path,
    query_index: int = 1,
    max_side: int = 360,
) -> List[GroundedVisualHit]:
    rows = list(index.items())
    if not rows:
        return []
    matrix = np.asarray([np.asarray(item["embedding"], dtype=np.float32) for _, item in rows], dtype=np.float32)
    query_emb = embedder.encode([query], convert_to_numpy=True, show_progress_bar=False).astype(np.float32)[0]
    denom = np.linalg.norm(matrix, axis=1) * max(float(np.linalg.norm(query_emb)), 1e-12)
    scores = (matrix @ query_emb) / np.maximum(denom, 1e-12)
    order = np.argsort(-scores)[:top_k]
    out_dir.mkdir(parents=True, exist_ok=True)
    hits: List[GroundedVisualHit] = []
    for rank, row_index in enumerate(order, start=1):
        triple_id, item = rows[int(row_index)]
        provenance = str(item.get("f_provenance", "fallback_middle"))
        frame_idx = int(item.get("frame_idx", 0)) if provenance == "gpt_anchored" else 7
        thumb_path = out_dir / f"q{query_index}_r{rank}.jpg"
        image = extract_sampled_frame(Path(str(item.get("video_path", ""))), frame_idx, max_side=max_side)
        image.save(thumb_path, "JPEG", quality=82, optimize=True)
        hits.append(GroundedVisualHit(
            clip_id=str(item.get("clip_id") or triple_id.split("#", 1)[0]),
            triple=str(item.get("triple_text", "")),
            frame_idx=frame_idx,
            frame_timestamp_s=float(item.get("frame_timestamp_s", 0.0)),
            thumbnail_image=thumb_path,
            score=float(scores[int(row_index)]),
            f_provenance=provenance,
        ))
    return hits


def hit_to_json(hit: GroundedVisualHit) -> dict[str, Any]:
    return {
        "clip_id": hit.clip_id,
        "triple": hit.triple,
        "frame_idx": hit.frame_idx,
        "frame_timestamp_s": hit.frame_timestamp_s,
        "thumbnail_image": str(hit.thumbnail_image),
        "score": hit.score,
        "f_provenance": hit.f_provenance,
    }


def write_html(demo: dict[str, list[dict[str, Any]]], html_file: Path) -> None:
    sections = []
    for query, hits in demo.items():
        cards = []
        for hit in hits:
            src = html.escape(str(hit["thumbnail_image"]))
            triple = html.escape(hit["triple"])
            clip = html.escape(hit["clip_id"])
            cards.append(f"""<div class="card"><img src="{src}" alt="thumbnail"><div><b>{triple}</b></div><div>{clip}</div><div>frame {hit['frame_idx']} | t={hit['frame_timestamp_s']:.3f}s | {html.escape(str(hit.get('f_provenance', '')))} | score={hit['score']:.3f}</div></div>""")
        sections.append(f"<section><h2>{html.escape(query)}</h2><div class=grid>{''.join(cards)}</div></section>")
    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Grounded Visual Triples Demo</title>
<style>
body{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;margin:24px;background:#0b1020;color:#e5e7eb}}
h1{{margin-bottom:8px}} section{{margin:22px 0}} .grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}
.card{{background:#111827;border:1px solid #374151;border-radius:10px;padding:10px;font-size:14px}}
img{{width:100%;height:150px;object-fit:cover;border-radius:8px;background:#1f2937}} b{{display:block;margin:8px 0 4px}}
</style></head><body><h1>Grounded Visual Triples</h1><p>Top-3 MiniLM text matches. Thumbnail = exact GPT supporting frame index.</p>{''.join(sections)}</body></html>"""
    html_file.write_text(html_doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="")
    parser.add_argument("--examples", nargs="*", default=["hard drive on dining table", "jake at kitchen", "box near laptop"])
    parser.add_argument("--index-file", type=Path, default=Path("output/metadata/visual_memory/A1_JAKE/visual_triples_gpt_index.pkl"))
    parser.add_argument("--thumb-dir", type=Path, default=Path("output/grounded_visual_thumbnails"))
    parser.add_argument("--out-json", type=Path, default=Path("output/grounded_visual_retrieve.json"))
    parser.add_argument("--html-file", type=Path, default=Path("output/grounded_visual_demo.html"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-side", type=int, default=360)
    parser.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    index = load_index(args.index_file)
    embedder = SentenceTransformer(args.embed_model, device=args.device)
    queries = [args.query] if args.query else args.examples[:3]
    demo: dict[str, list[dict[str, Any]]] = {}
    for q_index, query in enumerate(queries, start=1):
        hits = retrieve_visual_grounded(query, index, embedder, args.top_k, args.thumb_dir, query_index=q_index, max_side=args.max_side)
        demo[query] = [hit_to_json(hit) for hit in hits]
        print(json.dumps({"query": query, "hits": demo[query]}, ensure_ascii=False), flush=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(demo, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_html(demo, args.html_file)
    print(f"wrote {args.out_json} and {args.html_file}", flush=True)


if __name__ == "__main__":
    main()
