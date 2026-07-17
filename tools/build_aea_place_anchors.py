#!/usr/bin/env python3
"""Cluster AEA metric pose trajectories into place anchors for sidecar chunks."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aea-dir", type=Path, default=Path("data/AEA/loc5_script4_seq6_rec1"))
    parser.add_argument("--sidecar-dir", type=Path, default=Path("output/metadata/spatial_memory/AEA_loc5_script4_seq6_rec1"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--eps", type=float, default=0.5, help="DBSCAN epsilon in meters.")
    parser.add_argument("--min-samples", type=int, default=200, help="Raw-pose sample count; scaled after 10 Hz stride.")
    parser.add_argument("--dwell-ms", type=int, default=1500, help="Minimum contiguous visit duration.")
    return parser.parse_args()


def load_trajectory(path: Path) -> dict[str, np.ndarray]:
    timestamps: list[int] = []
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    graph_uids: list[str] = []
    quality_scores: list[float] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            timestamps.append(int(row["tracking_timestamp_us"]))
            xs.append(float(row["tx_world_device"]))
            ys.append(float(row["ty_world_device"]))
            zs.append(float(row["tz_world_device"]))
            graph_uids.append(row["graph_uid"])
            quality_scores.append(float(row["quality_score"]))
    if not timestamps:
        raise ValueError(f"No trajectory rows found in {path}")
    return {
        "timestamp_us": np.asarray(timestamps, dtype=np.int64),
        "x": np.asarray(xs, dtype=np.float64),
        "y": np.asarray(ys, dtype=np.float64),
        "z": np.asarray(zs, dtype=np.float64),
        "graph_uid": np.asarray(graph_uids, dtype=object),
        "quality_score": np.asarray(quality_scores, dtype=np.float64),
    }


def downsample_stride(trajectory: dict[str, np.ndarray], target_hz: float = 10.0) -> tuple[dict[str, np.ndarray], int, float]:
    timestamps = trajectory["timestamp_us"]
    duration_s = max((int(timestamps[-1]) - int(timestamps[0])) / 1_000_000.0, 1e-9)
    raw_hz = max((len(timestamps) - 1) / duration_s, 1.0)
    stride = max(1, int(round(raw_hz / target_hz)))
    downsampled = {key: values[::stride] for key, values in trajectory.items()}
    return downsampled, stride, raw_hz


def row_end_times_us(timestamps: np.ndarray) -> np.ndarray:
    if len(timestamps) == 1:
        return timestamps + 100_000
    deltas = np.diff(timestamps)
    median_delta = int(np.median(deltas))
    ends = np.empty_like(timestamps)
    ends[:-1] = timestamps[1:]
    ends[-1] = timestamps[-1] + median_delta
    return ends


def build_visits(
    trajectory: dict[str, np.ndarray], labels: np.ndarray, dwell_us: int
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    timestamps = trajectory["timestamp_us"]
    row_ends = row_end_times_us(timestamps)
    visits: list[dict[str, Any]] = []
    clusters: dict[int, dict[str, Any]] = {}
    start = 0
    while start < len(labels):
        label = int(labels[start])
        end = start + 1
        while end < len(labels) and int(labels[end]) == label:
            end += 1
        if label != -1:
            start_us = int(timestamps[start])
            end_us = int(row_ends[end - 1])
            duration_us = end_us - start_us
            if duration_us >= dwell_us:
                indices = np.arange(start, end)
                visits.append(
                    {
                        "cluster_id": label,
                        "start_us": start_us,
                        "end_us": end_us,
                        "duration_us": duration_us,
                        "sample_count": int(len(indices)),
                        "indices": indices,
                    }
                )
        start = end

    used_cluster_ids = sorted({visit["cluster_id"] for visit in visits}, key=lambda c: min(v["start_us"] for v in visits if v["cluster_id"] == c))
    label_map = {cluster_id: f"place_{idx}" for idx, cluster_id in enumerate(used_cluster_ids)}
    for visit in visits:
        visit["label"] = label_map[visit["cluster_id"]]

    for cluster_id, label in label_map.items():
        mask = labels == cluster_id
        clusters[cluster_id] = {
            "cluster_id": cluster_id,
            "label": label,
            "centroid_world_m": [
                float(np.mean(trajectory["x"][mask])),
                float(np.mean(trajectory["y"][mask])),
                float(np.mean(trajectory["z"][mask])),
            ],
            "sample_count": int(np.sum(mask)),
        }

    visits.sort(key=lambda visit: visit["start_us"])
    for idx, visit in enumerate(visits):
        visit["visit_id"] = f"visit_{idx:03d}"
    return visits, clusters


def overlap_us(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def evidence_for_visit(visit: dict[str, Any], visits: list[dict[str, Any]]) -> list[str]:
    same_place = [candidate for candidate in visits if candidate["label"] == visit["label"] and candidate["visit_id"] != visit["visit_id"]]
    nearby = sorted(same_place, key=lambda candidate: abs(candidate["start_us"] - visit["start_us"]))[:3]
    return [
        f"{candidate['visit_id']} {candidate['label']} {candidate['duration_us'] / 1_000_000.0:.1f}s"
        for candidate in nearby
    ]


def augment_chunks(sidecar_dir: Path, out_dir: Path, visits: list[dict[str, Any]], clusters: dict[int, dict[str, Any]]) -> tuple[list[Path], int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths = sorted(sidecar_dir.glob("*_chunk_*.json"))
    if not chunk_paths:
        raise FileNotFoundError(f"No chunk JSONs found under {sidecar_dir}")
    written: list[Path] = []
    anchored = 0
    for chunk_path in chunk_paths:
        chunk = json.loads(chunk_path.read_text())
        chunk_start, chunk_end_inclusive = [int(value) for value in chunk["time_range_us"]]
        chunk_end = chunk_end_inclusive + 1
        candidates = []
        for visit in visits:
            overlap = overlap_us(chunk_start, chunk_end, int(visit["start_us"]), int(visit["end_us"]))
            if overlap > 0:
                candidates.append((overlap, visit))
        if candidates:
            overlap, dominant = max(candidates, key=lambda item: item[0])
            clipped_start = max(chunk_start, int(dominant["start_us"]))
            clipped_end = min(chunk_end, int(dominant["end_us"]))
            cluster = clusters[int(dominant["cluster_id"])]
            frame_id = str(chunk.get("pose_6dof_median", {}).get("graph_uid") or chunk.get("pose_6dof_mean", {}).get("graph_uid", ""))
            chunk["place_anchor"] = {
                "coordinate_frame_id": frame_id,
                "centroid_world_m": cluster["centroid_world_m"],
                "time_range_us": [int(clipped_start), int(clipped_end - 1)],
                "label": dominant["label"],
                "evidence": evidence_for_visit(dominant, visits),
            }
            if isinstance(chunk.get("coverage_gaps"), dict):
                chunk["coverage_gaps"]["place_clustering"] = False
            anchored += 1
        else:
            chunk["place_anchor"] = None
            if isinstance(chunk.get("coverage_gaps"), dict):
                chunk["coverage_gaps"]["place_clustering"] = True
        output_path = out_dir / chunk_path.name
        output_path.write_text(json.dumps(chunk, indent=2) + "\n")
        written.append(output_path)
    return written, anchored


def summarize_places(
    args: argparse.Namespace,
    trajectory: dict[str, np.ndarray],
    downsampled: dict[str, np.ndarray],
    stride: int,
    raw_hz: float,
    effective_min_samples: int,
    visits: list[dict[str, Any]],
    clusters: dict[int, dict[str, Any]],
    transitions: list[dict[str, Any]],
    out_dir: Path,
    anchored_chunks: int,
    chunk_count: int,
) -> dict[str, Any]:
    sequence_id = args.aea_dir.name
    duration_us = int(trajectory["timestamp_us"][-1]) - int(trajectory["timestamp_us"][0])
    places = []
    for cluster_id in sorted(clusters, key=lambda cluster_id: clusters[cluster_id]["label"]):
        cluster = clusters[cluster_id]
        cluster_visits = [visit for visit in visits if visit["cluster_id"] == cluster_id]
        total_dwell_us = sum(int(visit["duration_us"]) for visit in cluster_visits)
        places.append(
            {
                "label": cluster["label"],
                "cluster_id": int(cluster_id),
                "centroid_world_m": cluster["centroid_world_m"],
                "total_dwell_s": round(total_dwell_us / 1_000_000.0, 3),
                "visit_count": len(cluster_visits),
                "sample_count": cluster["sample_count"],
                "time_ranges_us": [[int(visit["start_us"]), int(visit["end_us"] - 1)] for visit in cluster_visits],
            }
        )
    visit_rows = [
        {
            "visit_id": visit["visit_id"],
            "label": visit["label"],
            "cluster_id": int(visit["cluster_id"]),
            "time_range_us": [int(visit["start_us"]), int(visit["end_us"] - 1)],
            "duration_s": round(int(visit["duration_us"]) / 1_000_000.0, 3),
            "sample_count": int(visit["sample_count"]),
        }
        for visit in visits
    ]
    dwell_total_s = sum(place["total_dwell_s"] for place in places)
    longest_visit_s = max((visit["duration_s"] for visit in visit_rows), default=0.0)
    mean_dwell_s = float(np.mean([visit["duration_s"] for visit in visit_rows])) if visit_rows else 0.0
    summary = {
        "schema_version": "aea_place_clustering.v1",
        "sequence_id": sequence_id,
        "parameters": {
            "eps_m": args.eps,
            "min_samples_raw": args.min_samples,
            "effective_min_samples_downsampled": effective_min_samples,
            "dwell_ms": args.dwell_ms,
            "target_hz": 10,
            "stride": stride,
            "raw_hz": round(raw_hz, 3),
        },
        "trajectory": {
            "raw_rows": int(len(trajectory["timestamp_us"])),
            "downsampled_rows": int(len(downsampled["timestamp_us"])),
            "duration_s": round(duration_us / 1_000_000.0, 3),
            "time_range_us": [int(trajectory["timestamp_us"][0]), int(trajectory["timestamp_us"][-1])],
        },
        "stats": {
            "place_count": len(places),
            "visit_count": len(visits),
            "stop_count": len(visits),
            "mean_dwell_s": round(mean_dwell_s, 3),
            "longest_visit_s": round(longest_visit_s, 3),
            "place_coverage_pct": round((dwell_total_s / max(duration_us / 1_000_000.0, 1e-9)) * 100.0, 2),
            "anchored_chunks": anchored_chunks,
            "chunk_count": chunk_count,
        },
        "places": places,
        "visits": visit_rows,
        "transitions": transitions,
    }
    summary_path = out_dir.parent / "places_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def build_transitions(visits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter()
    for prev, nxt in zip(visits, visits[1:]):
        if prev["label"] != nxt["label"]:
            counts[(prev["label"], nxt["label"])] += 1
    return [
        {"source": source, "target": target, "count": count}
        for (source, target), count in sorted(counts.items())
    ]


def print_summary(summary: dict[str, Any]) -> None:
    print("Per-place dwell summary:")
    for place in summary["places"]:
        centroid = ", ".join(f"{value:.2f}" for value in place["centroid_world_m"])
        print(
            f"  {place['label']}: dwell={place['total_dwell_s']:.1f}s visits={place['visit_count']} centroid=[{centroid}]"
        )
    stats = summary["stats"]
    params = summary["parameters"]
    print(
        "AEA place clustering summary: "
        f"{stats['place_count']} places, {stats['visit_count']} stops, "
        f"mean dwell {stats['mean_dwell_s']:.1f}s, longest {stats['longest_visit_s']:.1f}s, "
        f"coverage {stats['place_coverage_pct']:.1f}%, "
        f"anchored {stats['anchored_chunks']}/{stats['chunk_count']} chunks, "
        f"stride {params['stride']} to {summary['trajectory']['downsampled_rows']} rows, "
        f"DBSCAN eps={params['eps_m']}m min_samples={params['effective_min_samples_downsampled']} effective."
    )


def main() -> None:
    args = parse_args()
    sidecar_dir = args.sidecar_dir
    out_dir = args.out_dir or sidecar_dir / "with_place"
    trajectory = load_trajectory(args.aea_dir / "closed_loop_trajectory.csv")
    downsampled, stride, raw_hz = downsample_stride(trajectory)
    effective_min_samples = max(1, int(math.ceil(args.min_samples / stride)))
    labels = DBSCAN(eps=args.eps, min_samples=effective_min_samples).fit_predict(
        np.column_stack([downsampled["x"], downsampled["y"]])
    )
    visits, clusters = build_visits(downsampled, labels, args.dwell_ms * 1_000)
    transitions = build_transitions(visits)
    written, anchored_chunks = augment_chunks(sidecar_dir, out_dir, visits, clusters)
    summary = summarize_places(
        args,
        trajectory,
        downsampled,
        stride,
        raw_hz,
        effective_min_samples,
        visits,
        clusters,
        transitions,
        out_dir,
        anchored_chunks,
        len(written),
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
