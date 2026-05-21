#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Smoke-test AEA sidecar chunks through SpatialTripleEntry display."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worldmm.memory.spatial import GazeTarget, Pose6DoF, SpatialTripleEntry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sidecar-dir",
        default="output/metadata/spatial_memory/AEA_loc5_script4_seq6_rec1",
    )
    parser.add_argument("--limit", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sidecar_dir = Path(args.sidecar_dir)
    chunk_files = sorted(sidecar_dir.glob("aea_*_chunk_*.json"))[: args.limit]
    if not chunk_files:
        raise FileNotFoundError(f"No AEA chunk files found in {sidecar_dir}")

    for chunk_file in chunk_files:
        chunk = json.loads(chunk_file.read_text())
        gaze = chunk["gaze_samples"][0] if chunk["gaze_samples"] else None
        if gaze:
            gaze = {
                key: gaze[key]
                for key in GazeTarget.model_fields
                if key in gaze
            }
        entry = SpatialTripleEntry(
            id=f"aea_smoke_{chunk['chunk_index']}",
            subject="wearer",
            predicate="located_in",
            object="aea_world_frame",
            timestamp=int(chunk["time_range_us"][0]),
            place=chunk["sequence_id"],
            pose_6dof=Pose6DoF(**{
                key: chunk["pose_6dof_median"][key]
                for key in Pose6DoF.model_fields
                if key in chunk["pose_6dof_median"]
            }),
            gaze_target=GazeTarget(**gaze) if gaze else None,
        )
        print(f"{chunk_file.name}: {entry.to_display_str()}")


if __name__ == "__main__":
    main()
