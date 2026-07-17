#!/usr/bin/env python3
"""Build GPT-vision structured visual triples with supporting frame anchors."""

from __future__ import annotations

import argparse
import base64
import io
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
from worldmm.memory.spatial.utils import SPATIAL_PREDICATE_VOCAB  # type: ignore[reportMissingImports]  # noqa: E402

CLIP_LEN_SECONDS = 30.0
PREFERRED_PREDICATE_ORDER = [
    "on", "in", "at", "next_to", "near", "under", "above", "behind",
    "in_front_of", "holding", "located_in", "contains", "occupies",
    "sitting_at", "standing_at", "left_of", "right_of",
]


def predicate_vocab() -> List[str]:
    ordered = [pred for pred in PREFERRED_PREDICATE_ORDER if pred in SPATIAL_PREDICATE_VOCAB]
    extras = sorted(pred for pred in SPATIAL_PREDICATE_VOCAB if pred not in ordered)
    return ordered + extras


def prompt_text(strict_retry: bool = False) -> str:
    predicates = ", ".join(predicate_vocab())
    retry = "\nOutput STRICT JSON only." if strict_retry else ""
    return f"""You are watching 16 frames sampled uniformly from a 30-second egocentric video clip from the EgoLife dataset. The frames are presented in chronological order, indexed 0 through 15.

Extract spatial / co-location facts you can SEE in the frames. Emit ONLY a JSON object with key "triples", whose value is a list of objects with EXACTLY these keys:
  - "s": subject (lowercase short noun phrase referencing an object, person, or place visible in the frames; underscore for multi-word, e.g. "puzzle_board", "dining_table", "jake")
  - "p": predicate (MUST be one of: {predicates})
  - "o": object (same format as "s")
  - "f": the 0-indexed frame number (0 through 15) that BEST shows this relation. Pick a SINGLE frame.

Rules:
- Use ONLY the listed predicates. Drop any fact whose relation does not fit.
- Subjects and objects MUST refer to entities visibly present in at least the chosen frame.
- Do not invent. Do not describe actions, attributes, or temporal events.
- Maximum 15 triples per clip. Aim for 6-12.
- Output JSON only. No prose, no markdown, no headings.{retry}"""


def parse_filename_start(fname: str) -> tuple[int, str] | None:
    match = re.search(r"DAY(\d)_[^_]+_[^_]+_(\d{8})\.mp4$", fname)
    if not match:
        return None
    return int(match.group(1)), match.group(2)


def hhmmssff_to_seconds(hhmmssff: str) -> float:
    value = hhmmssff.zfill(8)
    return int(value[0:2]) * 3600 + int(value[2:4]) * 60 + int(value[4:6]) + int(value[6:8]) / 100.0


def add_seconds_to_hhmmssff(hhmmssff: str, secs: float) -> str:
    total_centis = int(round((hhmmssff_to_seconds(hhmmssff) + secs) * 100))
    hh, rem = divmod(total_centis, 3600 * 100)
    mm, rem = divmod(rem, 60 * 100)
    ss, ff = divmod(rem, 100)
    return f"{hh:02d}{mm:02d}{ss:02d}{ff:02d}"


def resize_long_side(image: Image.Image, max_long_side: int) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    long_side = max(width, height)
    if long_side <= max_long_side:
        return image
    scale = max_long_side / float(long_side)
    return image.resize((max(1, int(round(width * scale))), max(1, int(round(height * scale)))), Image.Resampling.LANCZOS)


def image_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def frame_timestamps(start_time: str, frames_per_clip: int) -> List[float]:
    start_s = hhmmssff_to_seconds(start_time) if start_time else 0.0
    return [round(start_s + i * (CLIP_LEN_SECONDS / frames_per_clip), 3) for i in range(frames_per_clip)]


def extract_frames(video_path: Path, frames_per_clip: int, max_long_side: int) -> tuple[List[Image.Image], float]:
    reader = VideoReader(str(video_path), ctx=cpu(0))
    total = len(reader)
    if total <= 0:
        raise ValueError("video has zero frames")
    fps = float(reader.get_avg_fps() or 0.0)
    duration_s = float(total / fps) if fps > 0 else CLIP_LEN_SECONDS
    indices = np.linspace(0, total - 1, frames_per_clip, dtype=int).tolist()
    batch = reader.get_batch(indices).asnumpy()
    return [resize_long_side(Image.fromarray(frame), max_long_side) for frame in batch], duration_s


def extract_json_object(text: str) -> Dict[str, Any]:
    value = str(text).strip()
    if value.startswith("```"):
        value = re.sub(r"^```[a-zA-Z]*\s*\n?", "", value)
        value = re.sub(r"\n?```\s*$", "", value).strip()
    return json.loads(value)


def normalize_term(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"^[\W_]+|[\W_]+$", "", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def validate_triples(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_triples = data.get("triples", [])
    if not isinstance(raw_triples, list):
        raise ValueError("triples is not a list")
    clean: List[Dict[str, Any]] = []
    seen = set()
    for raw in raw_triples:
        if not isinstance(raw, dict):
            continue
        subject = normalize_term(raw.get("s", ""))
        predicate = normalize_term(raw.get("p", ""))
        obj = normalize_term(raw.get("o", ""))
        try:
            frame_idx = int(raw.get("f", -1))
        except (TypeError, ValueError):
            continue
        if predicate not in SPATIAL_PREDICATE_VOCAB or not subject or not obj or not 0 <= frame_idx <= 15:
            continue
        key = (subject, predicate, obj, frame_idx)
        if key in seen:
            continue
        seen.add(key)
        clean.append({"s": subject, "p": predicate, "o": obj, "f": frame_idx})
        if len(clean) >= 15:
            break
    return clean


def triples_for_clip(model: LiteLLMProxyModel, frames: List[Image.Image]) -> tuple[List[Dict[str, Any]], str | None]:
    last_error = "parse_fail"
    for attempt in range(2):
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text(strict_retry=attempt > 0)}]
        content.extend({"type": "image_url", "image_url": {"url": image_data_url(frame)}} for frame in frames)
        response = model.generate([{"role": "user", "content": content}])
        try:
            return validate_triples(extract_json_object(str(response))), None
        except Exception as exc:
            last_error = f"parse_fail: {exc}"
    return [], last_error


def load_clip_records(clips_file: Path | None, video_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if clips_file and clips_file.exists():
        data = json.loads(clips_file.read_text(encoding="utf-8"))
        for item in data:
            video_path = Path(str(item.get("video_path") or item.get("source_video_path") or ""))
            if video_path.exists() and video_path.stat().st_size > 0:
                records.append(dict(item))
        return sorted(records, key=lambda item: str(item.get("clip_id") or item.get("video_path")))
    for path in sorted(video_dir.glob("*.mp4")):
        if path.stat().st_size <= 0:
            continue
        info = parse_filename_start(path.name)
        if not info:
            continue
        day_i, start_time = info
        records.append({
            "clip_id": f"DAY{day_i}_A1_JAKE_{start_time}",
            "video_path": str(path),
            "start_time": start_time,
            "end_time": add_seconds_to_hhmmssff(start_time, CLIP_LEN_SECONDS),
            "date": f"DAY{day_i}",
        })
    return records


def save_triples(path: Path, triples_data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(triples_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def triple_text(triple: Dict[str, Any]) -> str:
    return f"({triple['s']}, {triple['p']}, {triple['o']})"


def build_index(triples_path: Path, index_path: Path, embed_model: str, device: str) -> Dict[str, Dict[str, Any]]:
    triples_data = json.loads(triples_path.read_text(encoding="utf-8"))
    rows: List[tuple[str, Dict[str, Any], Dict[str, Any]]] = []
    for clip_id, clip in sorted(triples_data.items()):
        timestamps = clip.get("frame_timestamps") or []
        for sequence_no, triple in enumerate(clip.get("triples") or []):
            if not isinstance(triple, dict) or not {"s", "p", "o", "f"}.issubset(triple):
                continue
            frame_idx = int(triple["f"])
            if frame_idx >= len(timestamps):
                timestamps = frame_timestamps(str(clip.get("start_time", "")), 16)
            rows.append((f"{clip_id}#{frame_idx}#{sequence_no}", clip, triple))
    texts = [triple_text(triple) for _, _, triple in rows]
    embedder = SentenceTransformer(embed_model, device=device)
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True) if texts else np.zeros((0, 384), dtype=np.float32)
    index: Dict[str, Dict[str, Any]] = {}
    for (triple_id, clip, triple), emb in zip(rows, embeddings):
        frame_idx = int(triple["f"])
        timestamps = clip.get("frame_timestamps") or frame_timestamps(str(clip.get("start_time", "")), 16)
        index[triple_id] = {
            "clip_id": str(clip.get("clip_id") or triple_id.split("#", 1)[0]),
            "frame_idx": frame_idx,
            "frame_timestamp_s": float(timestamps[frame_idx]),
            "triple_text": triple_text(triple),
            "embedding": emb.astype(np.float32, copy=False),
            "video_path": clip.get("video_path", ""),
            "start_time": clip.get("start_time", ""),
            "end_time": clip.get("end_time", ""),
            "date": clip.get("date", ""),
        }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("wb") as handle:
        pickle.dump(index, handle)
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=Path, default=Path("data/EgoLife/A1_JAKE/DAY1"))
    parser.add_argument("--clips-file", type=Path, default=Path("output/metadata/visual_memory/A1_JAKE/visual_clips_clip-ViT-B-32.json"))
    parser.add_argument("--out-file", type=Path, default=Path("output/metadata/visual_memory/A1_JAKE/visual_triples_gpt_structured.json"))
    parser.add_argument("--index-file", type=Path, default=Path("output/metadata/visual_memory/A1_JAKE/visual_triples_gpt_index.pkl"))
    parser.add_argument("--frames-per-clip", type=int, default=16)
    parser.add_argument("--max-long-side", type=int, default=384)
    parser.add_argument("--model", default="chatgpt/gpt-5.4")
    parser.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--index-only", action="store_true")
    args = parser.parse_args()

    print(f"SPATIAL_PREDICATE_VOCAB={predicate_vocab()}", flush=True)
    if args.index_only:
        index = build_index(args.out_file, args.index_file, args.embed_model, args.device)
        print(f"visual triples index summary: triples={len(index)} index_file={args.index_file}", flush=True)
        return

    existing: Dict[str, Any] = {}
    if args.resume and args.out_file.exists():
        existing = json.loads(args.out_file.read_text(encoding="utf-8"))

    records = load_clip_records(args.clips_file, args.video_dir)
    model = LiteLLMProxyModel(model_name=args.model, cache_dir=".cache/gpt_visual_triples_structured_frame_anchor_v2", request_timeout=180.0)
    processed = 0
    failed = 0

    for index, record in enumerate(records, start=1):
        clip_id = str(record.get("clip_id") or record.get("id") or Path(str(record.get("video_path"))).stem)
        if args.resume and clip_id in existing and "triples" in existing[clip_id] and "frame_timestamps" in existing[clip_id]:
            print(f"[{index}/{len(records)}] resume-skip {clip_id}", flush=True)
            continue
        video_path = Path(str(record.get("video_path") or record.get("source_video_path") or ""))
        start_time = str(record.get("start_time", ""))
        entry = {
            "video_path": str(video_path),
            "start_time": start_time,
            "end_time": str(record.get("end_time", add_seconds_to_hhmmssff(start_time, CLIP_LEN_SECONDS) if start_time else "")),
            "date": str(record.get("date", "DAY1")),
            "triples": [],
            "error": None,
            "frame_timestamps": frame_timestamps(start_time, args.frames_per_clip),
        }
        try:
            frames, duration_s = extract_frames(video_path, args.frames_per_clip, args.max_long_side)
            triples, error = triples_for_clip(model, frames)
            entry.update({"triples": triples, "error": "parse_fail" if error else None, "frames_sampled": len(frames), "duration_s": duration_s})
            if error:
                failed += 1
                print(f"[{index}/{len(records)}] {clip_id} parse failed triples=0", flush=True)
            else:
                processed += 1
                print(f"[{index}/{len(records)}] {clip_id} ok triples={len(triples)}", flush=True)
        except Exception as exc:
            entry["error"] = str(exc)
            failed += 1
            print(f"[{index}/{len(records)}] {clip_id} failed: {exc}", flush=True)
        existing[clip_id] = entry
        if index % 10 == 0:
            save_triples(args.out_file, existing)
            print(f"checkpoint index={index} clips={len(existing)} processed={processed} failed={failed}", flush=True)

    save_triples(args.out_file, existing)
    index_data = build_index(args.out_file, args.index_file, args.embed_model, args.device)
    clips_with_triples = sum(1 for item in existing.values() if item.get("triples"))
    print(f"visual triples build summary: clips={len(existing)}, clips_with_triples={clips_with_triples}, triples={len(index_data)}, processed_now={processed}, failed={failed}.", flush=True)


if __name__ == "__main__":
    main()
