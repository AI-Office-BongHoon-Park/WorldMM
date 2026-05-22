#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Attach AEA semidense point-cloud scene latents to spatial sidecar chunks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worldmm.memory.spatial import PointCloudSidecar, SceneLatentRef


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aea-dir", type=Path, required=True)
    parser.add_argument("--sidecar-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-points-per-chunk", type=int, default=1000)
    parser.add_argument(
        "--quality-filter",
        default="inv_dist_std<=0.005,dist_std<=0.01",
        help="Comma-separated column<=threshold predicates.",
    )
    return parser.parse_args()


def parse_quality_filter(expression: str) -> list[tuple[str, float]]:
    filters: list[tuple[str, float]] = []
    for part in expression.split(","):
        part = part.strip()
        if not part:
            continue
        if "<=" not in part:
            raise ValueError(f"Unsupported quality filter predicate: {part!r}")
        column, threshold = part.split("<=", 1)
        filters.append((column.strip(), float(threshold.strip())))
    if not filters:
        raise ValueError("quality filter must include at least one predicate")
    return filters


def load_filtered_points(path: Path, filters: list[tuple[str, float]]) -> tuple[pd.DataFrame, int]:
    points = pd.read_csv(path, compression="gzip")
    mask = pd.Series(True, index=points.index)
    for column, threshold in filters:
        if column not in points.columns:
            raise KeyError(f"Missing semidense point column: {column}")
        mask &= points[column] <= threshold
    filtered = points.loc[mask, ["graph_uid", "px_world", "py_world", "pz_world", "inv_dist_std", "dist_std"]]
    if filtered.empty:
        raise ValueError(f"No semidense points remain after filter: {path}")
    return filtered.reset_index(drop=True), int(len(points))


def downsample_uniform(points: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if max_points < 1:
        raise ValueError("max-points-per-chunk must be positive")
    if len(points) <= max_points:
        return points.copy()
    indices = np.linspace(0, len(points) - 1, num=max_points, dtype=np.int64)
    return points.iloc[indices].reset_index(drop=True)


def write_ascii_ply(points_world_m: list[list[float]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points_world_m)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("end_header\n")
        for x, y, z in points_world_m:
            handle.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
    return path.stat().st_size


def chunk_sort_key(path: Path) -> tuple[str, int]:
    stem = path.stem
    try:
        return (stem, int(stem.rsplit("_", 1)[-1]))
    except ValueError:
        return (stem, -1)


def build_chunk(
    chunk_path: Path,
    out_dir: Path,
    sampled: pd.DataFrame,
    filtered_count: int,
    raw_count: int,
) -> dict[str, Any]:
    chunk = json.loads(chunk_path.read_text())
    points_world_m = sampled[["px_world", "py_world", "pz_world"]].astype(float).values.tolist()
    uncertainty = sampled["dist_std"].astype(float).tolist()
    graph_uid = str(chunk.get("pose_6dof_median", {}).get("graph_uid") or sampled["graph_uid"].mode().iloc[0])
    time_range_us = tuple(int(value) for value in chunk["time_range_us"])

    point_cloud = PointCloudSidecar(
        points_world_m=points_world_m,
        uncertainty=uncertainty,
        graph_uid=graph_uid,
        time_range_us=time_range_us,
        source="aea_semidense",
    )
    sequence_id = str(chunk.get("sequence_id") or chunk_path.parent.parent.name.removeprefix("AEA_"))
    chunk_index = int(chunk.get("chunk_index", chunk_sort_key(chunk_path)[1]))
    ply_path = out_dir / "scene_latents" / f"aea_{sequence_id}_chunk_{chunk_index:03d}_semidense.ply"
    storage_bytes = write_ascii_ply(point_cloud.points_world_m, ply_path)
    scene_latent_ref = SceneLatentRef(
        backend="semidense_points",
        storage_uri=str(ply_path),
        coord_frame_id=graph_uid,
        time_us=int(time_range_us[0]),
        state_kind="static_scene",
        decoder_version="aea_semidense_ascii_ply.v1",
        confidence=1.0,
        provenance_frames=[f"aea:{sequence_id}:chunk_{chunk_index:03d}"],
        storage_bytes=storage_bytes,
    )

    chunk["point_cloud"] = point_cloud.model_dump(mode="json")
    chunk["scene_latent_ref"] = scene_latent_ref.model_dump(mode="json")
    chunk.setdefault("scene_latent_stats", {})
    chunk["scene_latent_stats"].update(
        {
            "raw_points": raw_count,
            "points_after_quality_filter": filtered_count,
            "points_after_downsample": len(point_cloud.points_world_m),
        }
    )
    return chunk


def main() -> None:
    args = parse_args()
    filters = parse_quality_filter(args.quality_filter)
    filtered, raw_count = load_filtered_points(args.aea_dir / "semidense_points.csv.gz", filters)
    sampled = downsample_uniform(filtered, args.max_points_per_chunk)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths = sorted(args.sidecar_dir.glob("aea_*_chunk_*.json"), key=chunk_sort_key)
    if not chunk_paths:
        raise FileNotFoundError(f"No chunk JSON files found in {args.sidecar_dir}")

    for chunk_path in chunk_paths:
        chunk = build_chunk(chunk_path, args.out_dir, sampled, len(filtered), raw_count)
        output_path = args.out_dir / chunk_path.name
        output_path.write_text(json.dumps(chunk, indent=2) + "\n", encoding="utf-8")
        print(
            f"{chunk_path.name}: points_in_filter={len(filtered)} "
            f"points_after_downsample={len(sampled)} output_path={output_path}"
        )
    print(
        f"AEA semidense scene latent build: raw_points={raw_count} "
        f"filtered_points={len(filtered)} chunks={len(chunk_paths)} out_dir={args.out_dir}"
    )


if __name__ == "__main__":
    main()
