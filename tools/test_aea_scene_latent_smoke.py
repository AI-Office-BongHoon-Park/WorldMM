#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Smoke-test AEA scene latent chunks through SpatialTripleEntry display."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worldmm.memory.spatial import PointCloudSidecar, SceneLatentRef, SpatialTripleEntry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sidecar-dir",
        type=Path,
        default=Path("output/metadata/spatial_memory/AEA_loc5_script4_seq6_rec1/with_scene"),
    )
    parser.add_argument("--chunk-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunk_files = sorted(args.sidecar_dir.glob("aea_*_chunk_*.json"))
    if not chunk_files:
        raise FileNotFoundError(f"No AEA scene chunk files found in {args.sidecar_dir}")
    chunk_file = chunk_files[min(args.chunk_index, len(chunk_files) - 1)]
    chunk = json.loads(chunk_file.read_text())
    entry = SpatialTripleEntry(
        id=f"aea_scene_smoke_{chunk['chunk_index']}",
        subject="wearer",
        predicate="located_in",
        object="aea_world_frame",
        timestamp=int(chunk["time_range_us"][0]),
        place=chunk["sequence_id"],
        scene_latent_ref=SceneLatentRef(**chunk["scene_latent_ref"]),
        point_cloud=PointCloudSidecar(**chunk["point_cloud"]),
    )
    display = f"{chunk_file.name}: {entry.to_display_str()}"
    print(display)
    if "[scene_latent=" not in display or "[points=" not in display:
        raise AssertionError("SpatialTripleEntry display is missing scene latent or point placeholders")


if __name__ == "__main__":
    main()
