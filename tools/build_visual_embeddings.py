#!/usr/bin/env python3
"""Compute per-clip CLIP embeddings + clip metadata for VisualMemory.

Reads every non-empty MP4 under a video directory, extracts the middle frame
via decord, encodes it with sentence-transformers/clip-ViT-B-32, and writes
two artefacts the existing VisualMemory class already knows how to load:
  - <out>/visual_embeddings_<model>.pkl   (Dict[clip_id, np.ndarray])
  - <out>/visual_clips_<model>.json       (List[Dict] with start_time/end_time/date/video_path/clip_id)
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from decord import VideoReader, cpu
from PIL import Image
from sentence_transformers import SentenceTransformer


CLIP_LEN_SECONDS = 30.0


def parse_filename_start(fname: str):
    m = re.search(r"DAY(\d)_[^_]+_[^_]+_(\d{8})\.mp4$", fname)
    if not m:
        return None
    day = int(m.group(1))
    hhmmssff = m.group(2)
    return day, hhmmssff


def hhmmssff_to_seconds(hhmmssff: str) -> float:
    s = hhmmssff.zfill(8)
    return int(s[0:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:6]) + int(s[6:8]) / 100.0


def add_seconds_to_hhmmssff(hhmmssff: str, secs: float) -> str:
    base = hhmmssff_to_seconds(hhmmssff)
    total_centis = int(round((base + secs) * 100))
    hh, rem = divmod(total_centis, 3600 * 100)
    mm, rem = divmod(rem, 60 * 100)
    ss, ff = divmod(rem, 100)
    return f"{hh:02d}{mm:02d}{ss:02d}{ff:02d}"


def extract_middle_frame(video_path: Path) -> Image.Image:
    vr = VideoReader(str(video_path), ctx=cpu(0))
    mid = max(0, len(vr) // 2)
    frame = vr[mid].asnumpy()
    return Image.fromarray(frame)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=Path, default=Path("data/EgoLife/A1_JAKE/DAY1"))
    parser.add_argument("--out-dir", type=Path, default=Path("output/metadata/visual_memory/A1_JAKE"))
    parser.add_argument("--model-name", default="clip-ViT-B-32")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    mp4s = sorted(p for p in args.video_dir.glob("*.mp4") if p.stat().st_size > 0)
    print(f"Found {len(mp4s)} non-empty mp4 files in {args.video_dir}")

    model = SentenceTransformer(args.model_name, device=args.device)

    embeddings: Dict[str, np.ndarray] = {}
    clips: List[Dict[str, Any]] = []
    for i, p in enumerate(mp4s):
        info = parse_filename_start(p.name)
        if not info:
            print(f"  skip (cannot parse name): {p.name}")
            continue
        day_i, start_hhmmssff = info
        end_hhmmssff = add_seconds_to_hhmmssff(start_hhmmssff, CLIP_LEN_SECONDS)

        try:
            frame = extract_middle_frame(p)
        except Exception as exc:
            print(f"  skip (frame extract failed): {p.name} — {exc}")
            continue

        emb = model.encode([frame], convert_to_numpy=True, show_progress_bar=False)[0]
        clip_id = f"visual_{day_i}_{start_hhmmssff}"
        embeddings[clip_id] = emb

        clips.append({
            "clip_id": clip_id,
            "video_path": str(p),
            "start_time": start_hhmmssff,
            "end_time": end_hhmmssff,
            "date": f"DAY{day_i}",
        })
        print(f"  [{i+1:>3}/{len(mp4s)}]  {p.name}  →  emb shape={emb.shape}  ({clip_id})")

    safe = args.model_name.replace("/", "_")
    emb_path = args.out_dir / f"visual_embeddings_{safe}.pkl"
    clips_path = args.out_dir / f"visual_clips_{safe}.json"

    with open(emb_path, "wb") as f:
        pickle.dump(embeddings, f)
    with open(clips_path, "w") as f:
        json.dump(clips, f, indent=2)

    print(f"\nwrote {emb_path}  ({len(embeddings)} embeddings)")
    print(f"wrote {clips_path}  ({len(clips)} clip entries)")


if __name__ == "__main__":
    main()
