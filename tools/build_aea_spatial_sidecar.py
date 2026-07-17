#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportArgumentType=false, reportIndexIssue=false
"""Build AEA spatial sidecar chunks for WorldMM spatial memory."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worldmm.memory.spatial import GazeTarget, Pose6DoF, SpeechSegment


CHUNK_US = 30_000_000
MAX_TOTAL_BYTES = 50 * 1024 * 1024
SPEECH_CONFIDENCE_FLOOR = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aea-dir", default="data/AEA/loc5_script4_seq6_rec1")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--time-window-ms", type=int, default=100)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def finite_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    result = float(value)
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def normalize_quaternion(qw: float, qx: float, qy: float, qz: float) -> tuple[float, float, float, float]:
    magnitude = math.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    if magnitude == 0.0:
        return 1.0, 0.0, 0.0, 0.0
    return qw / magnitude, qx / magnitude, qy / magnitude, qz / magnitude


def pose_from_series(row: pd.Series) -> dict[str, Any]:
    qw, qx, qy, qz = normalize_quaternion(
        float(row["qw_world_device"]),
        float(row["qx_world_device"]),
        float(row["qy_world_device"]),
        float(row["qz_world_device"]),
    )
    pose = Pose6DoF(
        tracking_timestamp_us=int(row["tracking_timestamp_us"]),
        tx=float(row["tx_world_device"]),
        ty=float(row["ty_world_device"]),
        tz=float(row["tz_world_device"]),
        qw=qw,
        qx=qx,
        qy=qy,
        qz=qz,
        quality_score=float(row["quality_score"]),
        graph_uid=str(row["graph_uid"]),
    )
    return pose.model_dump()


def aggregate_pose(pose_rows: pd.DataFrame, reducer: str) -> dict[str, Any]:
    if reducer == "mean":
        row = pose_rows[[
            "tracking_timestamp_us",
            "tx_world_device",
            "ty_world_device",
            "tz_world_device",
            "qw_world_device",
            "qx_world_device",
            "qy_world_device",
            "qz_world_device",
            "quality_score",
        ]].mean(numeric_only=True)
    elif reducer == "median":
        row = pose_rows[[
            "tracking_timestamp_us",
            "tx_world_device",
            "ty_world_device",
            "tz_world_device",
            "qw_world_device",
            "qx_world_device",
            "qy_world_device",
            "qz_world_device",
            "quality_score",
        ]].median(numeric_only=True)
    else:
        raise ValueError(f"Unsupported pose reducer: {reducer}")

    row["graph_uid"] = str(pose_rows["graph_uid"].mode().iat[0])
    return pose_from_series(row)


def trajectory_length_m(pose_rows: pd.DataFrame) -> float:
    coords = pose_rows[["tx_world_device", "ty_world_device", "tz_world_device"]].to_numpy()
    if len(coords) < 2:
        return 0.0
    diffs = coords[1:] - coords[:-1]
    return float(((diffs**2).sum(axis=1) ** 0.5).sum())


def downsample_gaze(gaze_rows: pd.DataFrame, pose_times: pd.Series, time_window_us: int) -> list[dict[str, Any]]:
    if gaze_rows.empty:
        return []
    stride = max(1, math.ceil(len(gaze_rows) / 5))
    samples = gaze_rows.iloc[::stride].head(5)
    pose_values = pose_times.to_numpy()
    output: list[dict[str, Any]] = []
    for _, row in samples.iterrows():
        confidence_interval = {
            "yaw_low_rads_cpf": float(row["yaw_low_rads_cpf"]),
            "yaw_high_rads_cpf": float(row["yaw_high_rads_cpf"]),
            "pitch_low_rads_cpf": float(row["pitch_low_rads_cpf"]),
            "pitch_high_rads_cpf": float(row["pitch_high_rads_cpf"]),
        }
        point_cpf = None
        depth = finite_float(row.get("depth_m"))
        if depth is not None:
            yaw = float(row["yaw_rads_cpf"])
            pitch = float(row["pitch_rads_cpf"])
            point_cpf = (
                depth * math.tan(yaw),
                depth * math.tan(pitch),
                depth,
            )
        gaze = GazeTarget(
            tracking_timestamp_us=int(row["tracking_timestamp_us"]),
            yaw_rads_cpf=float(row["yaw_rads_cpf"]),
            pitch_rads_cpf=float(row["pitch_rads_cpf"]),
            point_cpf=point_cpf,
            confidence_interval=confidence_interval,
            session_uid=str(row["session_uid"]),
        )
        sample = gaze.model_dump()
        nearest_index = abs(pose_values - gaze.tracking_timestamp_us).argmin()
        nearest_pose_time = int(pose_values[nearest_index])
        delta_us = int(gaze.tracking_timestamp_us - nearest_pose_time)
        sample["nearest_pose_tracking_timestamp_us"] = nearest_pose_time
        sample["nearest_pose_delta_us"] = delta_us
        sample["aligned_within_time_window"] = abs(delta_us) <= time_window_us
        output.append(sample)
    return output


def speech_segments_for_window(
    speech: pd.DataFrame,
    start_us: int,
    end_us: int,
    time_window_us: int,
) -> list[dict[str, Any]]:
    if speech.empty:
        return []
    start_ns = (start_us - time_window_us) * 1_000
    end_ns = (end_us + time_window_us) * 1_000
    rows = speech[
        (speech["endTime_ns"] >= start_ns)
        & (speech["startTime_ns"] < end_ns)
        & (speech["confidence"] >= SPEECH_CONFIDENCE_FLOOR)
    ]
    segments: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        segment = SpeechSegment(
            start_time_ns=int(row["startTime_ns"]),
            end_time_ns=int(row["endTime_ns"]),
            text=str(row["written"]),
            confidence=float(row["confidence"]),
        )
        segments.append(segment.model_dump())
    return segments


def build_chunks(aea_dir: Path, out_dir: Path, time_window_ms: int) -> dict[str, Any]:
    trajectory = pd.read_csv(aea_dir / "closed_loop_trajectory.csv")
    gaze = pd.read_csv(aea_dir / "general_eye_gaze.csv")
    speech = pd.read_csv(aea_dir / "speech.csv")
    metadata = read_json(aea_dir / "metadata.json")
    source_summary = read_json(aea_dir / "summary.json")
    sequence_id = aea_dir.name
    time_window_us = time_window_ms * 1_000

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_us = int(trajectory["tracking_timestamp_us"].min())
    end_us = int(trajectory["tracking_timestamp_us"].max())
    chunk_count = math.ceil((end_us - start_us + 1) / CHUNK_US)
    chunk_files: list[str] = []
    totals = {
        "trajectory_length_m": 0.0,
        "gaze_samples_input": int(len(gaze)),
        "speech_segments_input": int(len(speech)),
        "speech_segments_emitted": 0,
    }

    for idx in range(chunk_count):
        chunk_start = start_us + idx * CHUNK_US
        chunk_end = min(chunk_start + CHUNK_US, end_us + 1)
        pose_rows = trajectory[
            (trajectory["tracking_timestamp_us"] >= chunk_start)
            & (trajectory["tracking_timestamp_us"] < chunk_end)
        ]
        if pose_rows.empty:
            continue
        gaze_rows = gaze[
            (gaze["tracking_timestamp_us"] >= chunk_start - time_window_us)
            & (gaze["tracking_timestamp_us"] < chunk_end + time_window_us)
        ]
        speech_segments = speech_segments_for_window(speech, chunk_start, chunk_end, time_window_us)
        length_m = trajectory_length_m(pose_rows)
        duration_s = max((chunk_end - chunk_start) / 1_000_000.0, 1e-6)
        chunk = {
            "schema_version": "aea_spatial_sidecar.v1",
            "sequence_id": sequence_id,
            "chunk_index": idx,
            "time_range_us": [chunk_start, chunk_end - 1],
            "duration_s": duration_s,
            "time_window_ms": time_window_ms,
            "metadata": metadata,
            "pose_6dof_mean": aggregate_pose(pose_rows, "mean"),
            "pose_6dof_median": aggregate_pose(pose_rows, "median"),
            "gaze_samples": downsample_gaze(gaze_rows, pose_rows["tracking_timestamp_us"], time_window_us),
            "speech_segments": speech_segments,
            "trajectory_length_m": length_m,
            "dwell_speed_mps": length_m / duration_s,
            "pose_sample_count": int(len(pose_rows)),
            "gaze_sample_count_input": int(len(gaze_rows)),
            "source_files": {
                "closed_loop_trajectory": "closed_loop_trajectory.csv",
                "general_eye_gaze": "general_eye_gaze.csv",
                "speech": "speech.csv",
                "metadata": "metadata.json",
            },
            "coverage_gaps": {
                "geo_available": int(trajectory["geo_available"].max()) if "geo_available" in trajectory else None,
                "gaze_depth_all_nan": bool(gaze["depth_m"].isna().all()) if "depth_m" in gaze else None,
                "online_calibration_parsed": False,
                "place_clustering": False,
            },
        }
        totals["trajectory_length_m"] += length_m
        totals["speech_segments_emitted"] += len(speech_segments)
        output_path = out_dir / f"aea_{sequence_id}_chunk_{idx:03d}.json"
        output_path.write_text(json.dumps(chunk, indent=2) + "\n")
        chunk_files.append(output_path.name)

    summary = {
        "schema_version": "aea_spatial_sidecar_summary.v1",
        "sequence_id": sequence_id,
        "chunk_size_s": 30,
        "chunk_count": len(chunk_files),
        "time_range_us": [start_us, end_us],
        "output_dir": str(out_dir),
        "chunk_files": chunk_files,
        "metadata": metadata,
        "source_summary": source_summary,
        "totals": totals,
        "coverage_gaps": {
            "geo_available": int(trajectory["geo_available"].max()) if "geo_available" in trajectory else None,
            "gaze_depth_all_nan": bool(gaze["depth_m"].isna().all()) if "depth_m" in gaze else None,
            "online_calibration_parsed": False,
            "place_clustering": False,
            "speech_confidence_floor": SPEECH_CONFIDENCE_FLOOR,
        },
    }
    summary_path = out_dir / f"aea_{sequence_id}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    total_bytes = sum(path.stat().st_size for path in out_dir.glob("*.json"))
    summary["total_output_bytes"] = total_bytes
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    total_bytes = sum(path.stat().st_size for path in out_dir.glob("*.json"))
    if total_bytes > MAX_TOTAL_BYTES:
        raise RuntimeError(f"AEA sidecar output {total_bytes} bytes exceeds 50 MB cap")
    summary["total_output_bytes"] = total_bytes
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    args = parse_args()
    aea_dir = Path(args.aea_dir)
    out_dir = Path(args.out_dir) if args.out_dir else Path(
        "output/metadata/spatial_memory"
    ) / f"AEA_{aea_dir.name}"
    summary = build_chunks(aea_dir, out_dir, args.time_window_ms)
    print(
        "AEA sidecar build: "
        f"sequence={summary['sequence_id']} chunks={summary['chunk_count']} "
        f"out_dir={summary['output_dir']} bytes={summary['total_output_bytes']}"
    )


if __name__ == "__main__":
    main()
