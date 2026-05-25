#!/usr/bin/env python3
"""Batch runner for Phase 4 geometric grounding sidecars."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = ROOT / "output/metadata/spatial_memory/A1_JAKE/phase4_chunk_selection.json"
DEFAULT_TRIPLES = ROOT / "output/metadata/spatial_memory/A1_JAKE/phase4_selected_spatial_triples.json"
DEFAULT_OUTPUT_DIR = ROOT / "output/metadata/spatial_memory/A1_JAKE/grounding"
DEFAULT_VIDEO_DIR = ROOT / "data/EgoLife/A1_JAKE/DAY1"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def video_for_chunk(video_dir: Path, chunk_id: str) -> str:
    time8 = str(int(chunk_id) % 100_000_000).zfill(8)
    name = f"DAY1_A1_JAKE_{time8}.mp4"
    path = video_dir / name
    if not path.exists():
        raise FileNotFoundError(path)
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--triples-file", type=Path, default=DEFAULT_TRIPLES)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-triples", type=int, default=10)
    parser.add_argument("--allow-depth-fallback", action="store_true")
    parser.add_argument("--log", type=Path, default=ROOT / "output/metadata/spatial_memory/A1_JAKE/phase4_grounding_batch_log.json")
    args = parser.parse_args()

    selection = read_json(args.selection)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log: list[dict[str, Any]] = []
    for row in selection["chunks"]:
        chunk_id = str(row["chunk_id"])
        out_path = args.output_dir / f"{chunk_id}.json"
        if out_path.exists():
            log.append({"chunk_id": chunk_id, "status": "skipped_existing", "sidecar": str(out_path)})
            print(f"skip existing {chunk_id}", flush=True)
            continue
        triples = row.get("triples", [])
        indices = ",".join(str(i) for i in range(min(args.max_triples, len(triples))))
        command = [
            sys.executable,
            str(ROOT / "tools/build_geometric_grounding.py"),
            "--triples-file",
            str(args.triples_file),
            "--video-dir",
            str(args.video_dir),
            "--output-dir",
            str(args.output_dir),
            "--chunk-id",
            chunk_id,
            "--video-file",
            video_for_chunk(args.video_dir, chunk_id),
            "--target-indices",
            indices,
            "--max-triples",
            str(args.max_triples),
        ]
        if args.allow_depth_fallback:
            command.append("--allow-depth-fallback")
        start = time.perf_counter()
        print(f"run {chunk_id} indices={indices}", flush=True)
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        elapsed = round(time.perf_counter() - start, 2)
        status = "ok" if completed.returncode == 0 else "failed"
        log.append({
            "chunk_id": chunk_id,
            "status": status,
            "returncode": completed.returncode,
            "wall_time_sec": elapsed,
            "sidecar": str(out_path),
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
        })
        write_json(args.log, {"selection": str(args.selection), "records": log})
        if completed.returncode != 0:
            print(f"failed {chunk_id}; continuing", flush=True)
    write_json(args.log, {"selection": str(args.selection), "records": log})
    ok = sum(1 for item in log if item["status"] in {"ok", "skipped_existing"})
    print(f"grounding batch complete ok_or_existing={ok}/{len(log)} log={args.log}")


if __name__ == "__main__":
    main()
