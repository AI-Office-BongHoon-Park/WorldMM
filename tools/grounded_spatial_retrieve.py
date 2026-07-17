#!/usr/bin/env python3
"""Grounded spatial retrieval — return text triples PLUS frame thumbnails in one call.

The point: every consolidated spatial triple already has a first-observed chunk
timestamp in the per-chunk extraction JSON. That timestamp also picks out an
EgoLife video clip. So a single retrieve() invocation can produce a bundle of
`(triple_text, thumbnail_image)` pairs that a vision-capable LLM consumes
directly — no separate visual axis call needed at QA time.

This file:
  - exposes `retrieve_spatial_grounded(memory, query, top_k, video_dir)` as a
    library function returning a list of `GroundedHit` dataclasses;
  - exposes `as_multimodal_content_block(hits, header)` to convert hits into the
    `[{type:text}, {type:image,image:PIL}, ...]` content array that
    `LiteLLMProxyModel.generate` (and the WorldMemory QA path) already accept;
  - ships a small CLI that prints retrieved triples and writes thumbnails for
    a hand-typed query, so you can smoke-test the bundle without a full eval.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thumbnail_extractor import (
    extract_thumbnail,
    find_clip,
    extract_frame,
    parse_filename_start,
    hhmmss_to_seconds,
    absolute_to_day_and_hhmmssff,
)

from worldmm.memory.spatial import SpatialMemory


@dataclass
class GroundedHit:
    triple: tuple[str, str, str]
    first_seen_ts: int
    consolidated_ts: int
    display: str
    image: Optional[Image.Image] = None
    thumbnail_path: Optional[Path] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def _first_seen_map(extraction_file: Path) -> Dict[tuple[str, str, str], int]:
    if not extraction_file.exists():
        return {}
    with open(extraction_file) as f:
        data = json.load(f)
    seen: Dict[tuple[str, str, str], int] = {}
    for chunk_ts in sorted(data.get("spatial_triples", {}).keys()):
        for tr in data["spatial_triples"][chunk_ts]:
            if len(tr) != 3:
                continue
            seen.setdefault((tr[0], tr[1], tr[2]), int(chunk_ts))
    return seen


def retrieve_spatial_grounded(
    memory: SpatialMemory,
    query: str,
    top_k: int,
    video_dir: Path,
    extraction_file: Path,
    thumb_dir: Path,
    max_side: int = 480,
) -> List[GroundedHit]:
    """Run a spatial PPR retrieval and attach the frame thumbnail at each
    triple's first-observed chunk timestamp."""
    triple_first_seen = _first_seen_map(extraction_file)
    entries = memory.retrieve(query, top_k=top_k, as_context=False) or []
    hits: List[GroundedHit] = []
    for e in entries:
        key = (e.subject, e.predicate, e.object)
        anchor_ts = triple_first_seen.get(key, int(e.timestamp))
        clip = find_clip(video_dir, anchor_ts)
        image = None
        thumb_path = None
        if clip is not None and clip.exists() and clip.stat().st_size > 0:
            try:
                clip_start = parse_filename_start(clip.name)
                if clip_start is not None:
                    _, target_hhmmss = absolute_to_day_and_hhmmssff(anchor_ts)
                    seconds_into = max(
                        0.0,
                        hhmmss_to_seconds(target_hhmmss) - hhmmss_to_seconds(clip_start),
                    )
                    image = extract_frame(clip, seconds_into, max_side=max_side)
                    thumb_path = extract_thumbnail(anchor_ts, video_dir, thumb_dir, max_side=max_side)
            except Exception:
                image = None
        hits.append(GroundedHit(
            triple=key,
            first_seen_ts=anchor_ts,
            consolidated_ts=int(e.timestamp),
            display=e.to_display_str(),
            image=image,
            thumbnail_path=thumb_path,
        ))
    return hits


def as_multimodal_content_block(hits: List[GroundedHit], header: str = "Retrieved spatial evidence") -> List[Dict[str, Any]]:
    """Convert hits into the multimodal content array accepted by
    LiteLLMProxyModel.generate (and the WorldMemory QA path)."""
    if not hits:
        return [{"type": "text", "text": f"{header}: (no hits)"}]
    block: List[Dict[str, Any]] = [{"type": "text", "text": f"{header}:"}]
    for h in hits:
        if h.image is not None:
            block.append({"type": "text", "text": h.display})
            block.append({"type": "image", "image": h.image})
        else:
            block.append({"type": "text", "text": f"{h.display}  [no covering video clip]"})
    return block


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--spatial-file",
        default="output/metadata/spatial_memory/A1_JAKE/spatial_consolidation_results_chatgpt-gpt-5.4.json",
    )
    parser.add_argument(
        "--extraction-file",
        default="output/metadata/spatial_memory/A1_JAKE/spatial_extraction_results_chatgpt-gpt-5.4.json",
    )
    parser.add_argument("--video-dir", default="data/EgoLife/A1_JAKE/DAY1")
    parser.add_argument("--thumb-dir", default="output/thumbnails/A1_JAKE/DAY1")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-side", type=int, default=480)
    parser.add_argument("--out-json", default="output/grounded_retrieve_demo.json")
    args = parser.parse_args()

    from worldmm.embedding import EmbeddingModel

    embedding_model = EmbeddingModel(
        text_model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )
    embedding_model.load_model(model_type="text")
    mem = SpatialMemory(embedding_model=embedding_model)
    mem.load_triples_from_file(args.spatial_file)
    mem.index(until_time=10 ** 12)

    hits = retrieve_spatial_grounded(
        mem,
        query=args.query,
        top_k=args.top_k,
        video_dir=Path(args.video_dir),
        extraction_file=Path(args.extraction_file),
        thumb_dir=Path(args.thumb_dir),
        max_side=args.max_side,
    )

    n_with = sum(1 for h in hits if h.image is not None)
    print(f"query='{args.query}'  hits={len(hits)}  with_thumbnail={n_with}")
    for h in hits:
        marker = "[img]" if h.image is not None else "[---]"
        print(f"  {marker}  ts={h.first_seen_ts}  {h.display}")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "query": args.query,
        "n_hits": len(hits),
        "n_with_thumbnail": n_with,
        "hits": [
            {
                "triple": list(h.triple),
                "first_seen_ts": h.first_seen_ts,
                "consolidated_ts": h.consolidated_ts,
                "display": h.display,
                "thumbnail_path": str(h.thumbnail_path) if h.thumbnail_path else None,
            }
            for h in hits
        ],
    }
    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
