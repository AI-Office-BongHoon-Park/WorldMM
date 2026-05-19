#!/usr/bin/env python3
"""Extract a JPEG frame from an EgoLife video given an absolute timestamp."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional

from decord import VideoReader, cpu
from PIL import Image


def parse_filename_start(filename: str) -> Optional[int]:
    """A1_JAKE/DAY1/DAY1_A1_JAKE_11150000.mp4 -> 11150000 (HHMMSSFF)."""
    base = os.path.basename(filename)
    m = re.search(r"_(\d{8})\.mp4$", base)
    return int(m.group(1)) if m else None


def parse_ts(ts: str | int) -> int:
    """`121143000` (DAY1 12:11:43.00) -> 121143000 int."""
    return int(str(ts).strip())


def hhmmss_to_seconds(hhmmssff: int) -> float:
    """Decode the trailing HHMMSSFF integer (no day prefix) into wall-clock seconds."""
    s = str(hhmmssff).zfill(8)
    hh = int(s[0:2])
    mm = int(s[2:4])
    ss = int(s[4:6])
    ff = int(s[6:8])
    return hh * 3600 + mm * 60 + ss + ff / 100.0


def absolute_to_day_and_hhmmssff(ts: int) -> tuple[int, int]:
    """`121143000` -> (1, 12114300)."""
    s = str(ts).zfill(9)
    return int(s[0]), int(s[1:9])


def find_clip(
    video_dir: Path,
    target_ts: int,
    clip_window_s: float = 30.0,
    fallback_window_s: float = 180.0,
) -> Optional[Path]:
    """Locate the MP4 covering the target timestamp.

    First tries the strict 30-second clip window. Falls back to the nearest
    MP4 whose start is within `fallback_window_s` (default 180 s = roughly the
    spatial-chunk granularity in EgoLife). Returns the nearest non-zero file.
    """
    target_day, target_hhmmss = absolute_to_day_and_hhmmssff(target_ts)
    target_s = hhmmss_to_seconds(target_hhmmss)
    candidates: List[tuple[float, Path]] = []
    for p in video_dir.glob(f"DAY{target_day}_*.mp4"):
        if p.stat().st_size == 0:
            continue
        start_hhmmss = parse_filename_start(p.name)
        if start_hhmmss is None:
            continue
        start_s = hhmmss_to_seconds(start_hhmmss)
        if start_s <= target_s < start_s + clip_window_s:
            return p
        candidates.append((abs(start_s - target_s), p))
    if candidates:
        candidates.sort()
        nearest_delta, nearest = candidates[0]
        if nearest_delta <= fallback_window_s:
            return nearest
    return None


def extract_frame(
    video_path: Path,
    seconds_into_clip: float,
    max_side: int = 480,
) -> Image.Image:
    """Read the closest frame at `seconds_into_clip` into a downscaled PIL.Image."""
    vr = VideoReader(str(video_path), ctx=cpu(0))
    fps = float(vr.get_avg_fps()) or 30.0
    idx = max(0, min(len(vr) - 1, int(round(seconds_into_clip * fps))))
    frame = vr[idx].asnumpy()
    img = Image.fromarray(frame)
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def extract_thumbnail(
    target_ts: int,
    video_dir: Path,
    out_dir: Path,
    max_side: int = 480,
    quality: int = 78,
) -> Optional[Path]:
    """Write a downscaled JPEG and return its path, or None if no clip covers the timestamp."""
    clip = find_clip(video_dir, target_ts)
    if clip is None or not clip.exists() or clip.stat().st_size == 0:
        return None
    clip_start_hhmmss = parse_filename_start(clip.name)
    if clip_start_hhmmss is None:
        return None
    _, target_hhmmss = absolute_to_day_and_hhmmssff(target_ts)
    seconds_into = max(0.0, hhmmss_to_seconds(target_hhmmss) - hhmmss_to_seconds(clip_start_hhmmss))
    img = extract_frame(clip, seconds_into, max_side=max_side)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"thumb_{target_ts}.jpg"
    img.save(out_path, "JPEG", quality=quality, optimize=True)
    return out_path


def extract_for_timestamps(
    timestamps: Iterable[int],
    video_dir: Path,
    out_dir: Path,
    max_side: int = 480,
) -> dict[int, Optional[str]]:
    result: dict[int, Optional[str]] = {}
    for ts in timestamps:
        path = extract_thumbnail(ts, video_dir, out_dir, max_side=max_side)
        result[ts] = str(path) if path else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract thumbnails for a list of EgoLife timestamps.")
    parser.add_argument("--timestamps", type=str, required=True,
                        help="Comma-separated list of absolute timestamps (e.g. 111524,113440).")
    parser.add_argument("--video-dir", type=Path, default=Path("data/EgoLife/A1_JAKE/DAY1"))
    parser.add_argument("--out-dir", type=Path, default=Path("output/thumbnails/A1_JAKE/DAY1"))
    parser.add_argument("--max-side", type=int, default=480)
    args = parser.parse_args()

    ts_list = [parse_ts(t) for t in args.timestamps.split(",") if t.strip()]
    out = extract_for_timestamps(ts_list, args.video_dir, args.out_dir, max_side=args.max_side)
    for ts, path in out.items():
        print(f"{ts}\t{path or 'NOT_FOUND'}")


if __name__ == "__main__":
    main()
