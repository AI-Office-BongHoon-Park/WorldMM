#!/usr/bin/env python3
"""Decode a spatial scene sidecar into a renderable file."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", type=Path, required=True, help="Augmented chunk JSON containing scene_latent_ref.")
    parser.add_argument("--format", choices=["ply"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def write_ascii_ply(points_world_m: list[list[float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points_world_m)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("end_header\n")
        for point in points_world_m:
            x, y, z = [float(value) for value in point]
            handle.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def resolve_storage_uri(uri: str, ref_path: Path) -> Path:
    path = Path(uri)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return ref_path.parent / path


def count_ply_vertices(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("element vertex "):
                return int(line.split()[-1])
    raise ValueError(f"PLY header lacks vertex count: {path}")


def main() -> None:
    args = parse_args()
    chunk = json.loads(args.ref.read_text())
    scene_latent_ref = chunk.get("scene_latent_ref")
    if not scene_latent_ref:
        raise ValueError(f"Missing scene_latent_ref in {args.ref}")
    if scene_latent_ref.get("backend") != "semidense_points":
        raise ValueError(f"Unsupported scene latent backend: {scene_latent_ref.get('backend')}")

    point_cloud = chunk.get("point_cloud")
    if point_cloud and point_cloud.get("points_world_m") is not None:
        points_world_m = point_cloud["points_world_m"]
        write_ascii_ply(points_world_m, args.output)
        point_count = len(points_world_m)
    else:
        source_path = resolve_storage_uri(scene_latent_ref["storage_uri"], args.ref)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, args.output)
        point_count = count_ply_vertices(args.output)

    print(f"input={args.ref} point_count={point_count} output={args.output} output_file_size={args.output.stat().st_size}")


if __name__ == "__main__":
    main()
