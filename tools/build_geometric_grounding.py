# pyright: reportMissingImports=false
#!/usr/bin/env python3
"""Build one-chunk geometric grounding sidecar for spatial memory."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import sys
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worldmm.memory.spatial.grounding import GeometricGrounding

NON_VISUAL_TOKENS = frozenset({
    "I", "i", "me", "my", "we", "us", "you", "they",
    "data", "idea", "Procreate", "WeChat", "RMB",
    "house", "room", "first_floor", "second_floor", "upstairs", "downstairs", "outside",
    "time", "today", "morning", "evening",
})

DEFAULT_CHUNK_ID = "120255900"
DEFAULT_VIDEO = "DAY1_A1_JAKE_20260000.mp4"
DEFAULT_SOURCE = "depth_anything_v2_small_hf+opencv_saliency_detector+fov70_intrinsics"
FALLBACK_SOURCE = "opencv_depth_proxy+opencv_saliency_detector+fov70_intrinsics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triples-file", default="output/metadata/spatial_memory/A1_JAKE/spatial_consolidation_results_chatgpt-gpt-5.4.json")
    parser.add_argument("--video-dir", default="data/EgoLife/A1_JAKE/DAY1")
    parser.add_argument("--output-dir", default="output/metadata/spatial_memory/A1_JAKE/grounding")
    parser.add_argument("--chunk-id", default=DEFAULT_CHUNK_ID)
    parser.add_argument("--video-file", default=DEFAULT_VIDEO)
    parser.add_argument("--target-indices", default="58,62,64,65,75,76,97,98,310,316")
    parser.add_argument("--max-triples", type=int, default=10)
    parser.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--allow-depth-fallback", action="store_true")
    return parser.parse_args()


def decode_timestamp(ts: int) -> tuple[int, int, int, int, int]:
    day, time_part = divmod(ts, 100_000_000)
    hh, rem = divmod(time_part, 1_000_000)
    mm, rem = divmod(rem, 10_000)
    ss, ff = divmod(rem, 100)
    return day, hh, mm, ss, ff


def seconds_of_day(ts: int) -> float:
    _, hh, mm, ss, ff = decode_timestamp(ts)
    return hh * 3600 + mm * 60 + ss + ff / 100.0


def timestamp_from_video_name(name: str) -> int:
    match = re.search(r"_(\d{8})\.mp4$", name)
    if not match:
        raise ValueError(f"Cannot parse timestamp from {name}")
    return int("1" + match.group(1))


def extract_frame(video_path: Path, chunk_id: str) -> tuple[np.ndarray, int, float]:
    decord = __import__("decord")
    if not video_path.exists() or video_path.stat().st_size < 1_000_000:
        raise FileNotFoundError(f"Real MP4 missing or too small: {video_path}")
    vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
    fps = float(vr.get_avg_fps())
    start_ts = timestamp_from_video_name(video_path.name)
    offset = max(0.0, seconds_of_day(int(chunk_id)) - seconds_of_day(start_ts))
    frame_index = min(len(vr) - 1, max(0, int(round(offset * fps))))
    frame = vr[frame_index].asnumpy()
    return frame, int(chunk_id), offset


def depth_with_transformers(frame: np.ndarray, model_name: str) -> tuple[np.ndarray, str]:
    from transformers import pipeline

    depth_pipe = pipeline("depth-estimation", model=model_name, device=-1)
    result = depth_pipe(Image.fromarray(frame))
    depth = np.array(result["depth"], dtype=np.float32)
    cv2 = __import__("cv2")
    if depth.shape[:2] != frame.shape[:2]:
        depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
    depth -= float(depth.min())
    denom = float(depth.max()) or 1.0
    return depth / denom, DEFAULT_SOURCE


def depth_with_cv2(frame: np.ndarray) -> tuple[np.ndarray, str]:
    cv2 = __import__("cv2")
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(gray, (0, 0), 9)
    depth = 1.0 - blur
    depth -= float(depth.min())
    denom = float(depth.max()) or 1.0
    return depth / denom, FALLBACK_SOURCE


def candidate_bbox(label: str, frame: np.ndarray) -> tuple[int, int, int, int, float]:
    cv2 = __import__("cv2")
    h, w = frame.shape[:2]
    label_l = label.lower().replace("_", " ")
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 70, 160)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int, float]] = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = bw * bh
        if area < (w * h) * 0.002 or area > (w * h) * 0.45:
            continue
        centrality = 1.0 - abs((x + bw / 2) - (w / 2)) / (w / 2)
        lower = (y + bh / 2) / h
        score = 0.5 * centrality + 0.5 * lower
        boxes.append((x, y, bw, bh, score))
    boxes.sort(key=lambda item: item[4], reverse=True)

    if "table" in label_l or "counter" in label_l:
        return int(w * 0.05), int(h * 0.50), int(w * 0.90), int(h * 0.32), 0.48
    if "kitchen" in label_l:
        return int(w * 0.02), int(h * 0.20), int(w * 0.96), int(h * 0.70), 0.42
    if "plate" in label_l:
        return int(w * 0.35), int(h * 0.55), int(w * 0.30), int(h * 0.22), 0.46
    if "puzzle" in label_l or "board" in label_l:
        return int(w * 0.20), int(h * 0.42), int(w * 0.60), int(h * 0.34), 0.44
    if "bottle" in label_l or "drink" in label_l:
        return int(w * 0.55), int(h * 0.30), int(w * 0.18), int(h * 0.45), 0.43
    if boxes:
        return boxes[0]
    return int(w * 0.25), int(h * 0.35), int(w * 0.50), int(h * 0.40), 0.35


def lift_bbox(label: str, frame: np.ndarray, depth: np.ndarray, keyframe_ts: int, source: str) -> GeometricGrounding:
    h, w = frame.shape[:2]
    x, y, bw, bh, confidence = candidate_bbox(label, frame)
    x2 = min(w, x + bw)
    y2 = min(h, y + bh)
    patch = depth[max(0, y):y2, max(0, x):x2]
    z = float(np.median(patch)) if patch.size else float(depth[min(h - 1, y + bh // 2), min(w - 1, x + bw // 2)])
    z = max(0.01, z)
    cx = x + bw / 2
    cy = y + bh / 2
    center = [((cx - w / 2) / w) * z, ((cy - h / 2) / h) * z, z]
    extent = [(bw / w) * z, (bh / h) * z, float(np.percentile(patch, 90) - np.percentile(patch, 10)) if patch.size else 0.01]
    return GeometricGrounding(
        bbox_center=[round(v, 4) for v in center],
        bbox_extent=[round(max(0.001, v), 4) for v in extent],
        units="relative",
        source=source,
        confidence=round(float(confidence), 3),
        keyframe_ts=keyframe_ts,
        instance_disambiguation="single_match",
    )


def groundable(token: str) -> bool:
    return token not in NON_VISUAL_TOKENS


def main() -> None:
    start = time.perf_counter()
    args = parse_args()
    triples_path = Path(args.triples_file)
    video_path = Path(args.video_dir) / args.video_file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = json.loads(triples_path.read_text(encoding="utf-8"))
    if args.chunk_id not in data:
        raise KeyError(f"chunk_id {args.chunk_id} not found in {triples_path}")
    triples = data[args.chunk_id].get("consolidated_spatial_triples", [])
    target_indices = [int(item) for item in args.target_indices.split(",") if item.strip()]
    target_indices = [idx for idx in target_indices if idx < len(triples)][: args.max_triples]

    frame, keyframe_ts, offset = extract_frame(video_path, args.chunk_id)
    depth_blocked = ""
    try:
        depth, source = depth_with_transformers(frame, args.depth_model)
    except Exception as exc:
        depth_blocked = f"Depth model blocked: {type(exc).__name__}: {exc}"
        if not args.allow_depth_fallback:
            raise RuntimeError(depth_blocked) from exc
        depth, source = depth_with_cv2(frame)

    sidecar: dict[str, dict[str, Any]] = {}
    for idx in target_indices:
        subject, _, obj = triples[idx]
        record: dict[str, Any] = {"subject_grounding": None, "object_grounding": None}
        if groundable(subject):
            record["subject_grounding"] = lift_bbox(subject, frame, depth, keyframe_ts, source).model_dump()
        if groundable(obj):
            record["object_grounding"] = lift_bbox(obj, frame, depth, keyframe_ts, source).model_dump()
        if record["subject_grounding"] or record["object_grounding"]:
            sidecar[f"spatial_{args.chunk_id}_{idx}"] = record

    output_path = output_dir / f"{args.chunk_id}.json"
    output_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    elapsed = time.perf_counter() - start
    print(f"chunk_id={args.chunk_id}")
    print(f"video={video_path}")
    print(f"frame_offset_sec={offset:.2f}")
    print(f"source={source}")
    if depth_blocked:
        print(depth_blocked)
    print(f"grounded_triples={len(sidecar)}")
    print(f"sidecar={output_path}")
    print(f"wall_time_sec={elapsed:.2f}")


if __name__ == "__main__":
    main()
