#!/usr/bin/env python3
"""Build GPT16 visual-description MiniLM embeddings for VisualMemory."""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from decord import VideoReader, cpu  # type: ignore[reportMissingImports]
from PIL import Image
from sentence_transformers import SentenceTransformer  # type: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from worldmm.llm.litellm_proxy import LiteLLMProxyModel  # type: ignore[reportMissingImports]  # noqa: E402

CLIP_LEN_SECONDS = 30.0
PROMPT = (
    "Describe this 30-second egocentric video clip from the 16 frames. Write one structured single paragraph, strict prose only, no JSON, no headings. Cover: "
    "(a) scene type/location, (b) the 3-5 most prominent objects and where they are spatially, "
    "(c) people present and what they are doing, (d) any object handovers or movements across the frames, "
    "(e) one short verbatim quote of any speech if interpretable from lip movement/context; write Quote: empty if uncertain. Cap output at 300 tokens."
)


def parse_filename_start(fname: str):
    match = re.search(r"DAY(\d)_[^_]+_[^_]+_(\d{8})\.mp4$", fname)
    if not match:
        return None
    return int(match.group(1)), match.group(2)


def hhmmssff_to_seconds(hhmmssff: str) -> float:
    s = hhmmssff.zfill(8)
    return int(s[0:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:6]) + int(s[6:8]) / 100.0


def add_seconds_to_hhmmssff(hhmmssff: str, secs: float) -> str:
    total_centis = int(round((hhmmssff_to_seconds(hhmmssff) + secs) * 100))
    hh, rem = divmod(total_centis, 3600 * 100)
    mm, rem = divmod(rem, 60 * 100)
    ss, ff = divmod(rem, 100)
    return f"{hh:02d}{mm:02d}{ss:02d}{ff:02d}"


def resize_long_side(image: Image.Image, max_long_side: int = 384) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    long_side = max(width, height)
    if long_side <= max_long_side:
        return image
    scale = max_long_side / float(long_side)
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(size, Image.Resampling.LANCZOS)


def extract_frames(video_path: Path, frames_per_clip: int) -> tuple[List[Image.Image], float]:
    reader = VideoReader(str(video_path), ctx=cpu(0))
    total = len(reader)
    if total <= 0:
        raise ValueError("video has zero frames")
    try:
        fps = float(reader.get_avg_fps())
    except Exception:
        fps = 0.0
    duration_s = float(total / fps) if fps > 0 else CLIP_LEN_SECONDS
    count = min(frames_per_clip, total)
    indices = np.linspace(0, total - 1, count, dtype=int).tolist()
    batch = reader.get_batch(indices).asnumpy()
    frames = [resize_long_side(Image.fromarray(frame), 384) for frame in batch]
    return frames, duration_s


def describe_clip(model: LiteLLMProxyModel, frames: List[Image.Image]) -> str:
    content: List[Dict[str, Any]] = [{"type": "text", "text": PROMPT}]
    content.extend({"type": "image", "image": frame} for frame in frames)
    response = model.generate([{"role": "user", "content": content}])
    return " ".join(str(response).split())


def save_outputs(out_dir: Path, embeddings: Dict[str, np.ndarray], descriptions: Dict[str, Any], clips: List[Dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "visual_embeddings_gpt16_minilm.pkl").open("wb") as handle:
        pickle.dump(embeddings, handle)
    (out_dir / "visual_descriptions_gpt16.json").write_text(json.dumps(descriptions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "visual_clips_gpt16_minilm.json").write_text(json.dumps(clips, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_existing(out_dir: Path) -> tuple[Dict[str, np.ndarray], Dict[str, Any], List[Dict[str, Any]]]:
    emb_path = out_dir / "visual_embeddings_gpt16_minilm.pkl"
    desc_path = out_dir / "visual_descriptions_gpt16.json"
    clips_path = out_dir / "visual_clips_gpt16_minilm.json"
    embeddings = pickle.loads(emb_path.read_bytes()) if emb_path.exists() else {}
    descriptions = json.loads(desc_path.read_text(encoding="utf-8")) if desc_path.exists() else {}
    clips = json.loads(clips_path.read_text(encoding="utf-8")) if clips_path.exists() else []
    return embeddings, descriptions, clips


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=Path, default=Path("data/EgoLife/A1_JAKE/DAY1"))
    parser.add_argument("--out-dir", type=Path, default=Path("output/metadata/visual_memory/A1_JAKE"))
    parser.add_argument("--frames-per-clip", type=int, default=16)
    parser.add_argument("--model", default="chatgpt/gpt-5.4")
    parser.add_argument("--describe-model", default="chatgpt/gpt-5.4")
    parser.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mp4s = sorted(path for path in args.video_dir.glob("*.mp4") if path.stat().st_size > 0)
    embeddings, descriptions, clips = load_existing(args.out_dir) if args.resume else ({}, {}, [])
    clips_by_id = {clip["clip_id"]: clip for clip in clips}

    llm = LiteLLMProxyModel(model_name=args.describe_model, cache_dir=".cache/gpt16_visual_descriptions", request_timeout=180.0)
    embedder = SentenceTransformer(args.embed_model, device=args.device)

    skipped = 0
    processed = 0
    for index, video_path in enumerate(mp4s, start=1):
        info = parse_filename_start(video_path.name)
        if not info:
            print(f"[{index}/{len(mp4s)}] skip unparsable {video_path.name}", flush=True)
            continue
        day_i, start_time = info
        clip_id = f"visual_{day_i}_{start_time}"
        end_time = add_seconds_to_hhmmssff(start_time, CLIP_LEN_SECONDS)
        if clip_id in embeddings and descriptions.get(clip_id, {}).get("description"):
            print(f"[{index}/{len(mp4s)}] resume-skip {clip_id}", flush=True)
            continue

        meta = {
            "frames_sampled": 0,
            "duration_s": 0.0,
            "video_path": str(video_path),
            "start_time": start_time,
            "end_time": end_time,
            "date": f"DAY{day_i}",
        }
        try:
            frames, duration_s = extract_frames(video_path, args.frames_per_clip)
            description = describe_clip(llm, frames)
            if not description:
                raise ValueError("empty description")
            emb = embedder.encode([description], convert_to_numpy=True, show_progress_bar=False)[0].astype(np.float32, copy=False)
            embeddings[clip_id] = emb
            meta.update({"description": description, "frames_sampled": len(frames), "duration_s": duration_s})
            clips_by_id[clip_id] = {
                "clip_id": clip_id,
                "video_path": str(video_path),
                "start_time": start_time,
                "end_time": end_time,
                "date": f"DAY{day_i}",
                "description": description,
            }
            processed += 1
            print(f"[{index}/{len(mp4s)}] {clip_id} ok emb={emb.shape} desc_chars={len(description)}", flush=True)
        except Exception as exc:
            skipped += 1
            meta.update({"description": None, "error": str(exc)})
            print(f"[{index}/{len(mp4s)}] {clip_id} failed: {exc}", flush=True)
        descriptions[clip_id] = meta

        clips = [clips_by_id[key] for key in sorted(clips_by_id)]
        if index % 10 == 0:
            save_outputs(args.out_dir, embeddings, descriptions, clips)
            print(f"checkpoint index={index} embeddings={len(embeddings)} descriptions={len(descriptions)} skipped={skipped}", flush=True)

    clips = [clips_by_id[key] for key in sorted(clips_by_id)]
    save_outputs(args.out_dir, embeddings, descriptions, clips)
    non_empty = sum(1 for value in descriptions.values() if value.get("description"))
    failures = sum(1 for value in descriptions.values() if value.get("error"))
    print(
        f"GPT16 visual build summary: videos={len(mp4s)}, processed_now={processed}, embeddings={len(embeddings)}, "
        f"non_empty_descriptions={non_empty}, failures_recorded={failures}, skipped_this_run={skipped}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
