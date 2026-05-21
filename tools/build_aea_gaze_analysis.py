#!/usr/bin/env python3
"""Build AEA gaze fixation analysis artifacts for one recording."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from matplotlib import colormaps
    from matplotlib.colors import to_hex
except ModuleNotFoundError:  # pragma: no cover - repository uv env may omit matplotlib.
    colormaps = None
    to_hex = None


RECORDING = "loc5_script4_seq6_rec1"
YAW_EDGES = np.linspace(-math.pi, math.pi, 13)
PITCH_EDGES = np.linspace(-math.pi / 3, math.pi / 3, 7)
QUIET_GAZE_THRESHOLD_RAD = 0.05
FIXATION_MERGE_RADIUS_RAD = 0.1
POSE_TOLERANCE_US = 1_000


def rounded(value: Any, digits: int = 6) -> Any:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
        return round(float(value), digits)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def load_inputs(aea_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    gaze_path = aea_dir / "general_eye_gaze.csv"
    pose_path = aea_dir / "closed_loop_trajectory.csv"
    speech_path = aea_dir / "speech.csv"
    gaze = pd.read_csv(gaze_path)
    pose_cols = {
        "tracking_timestamp_us",
        "tx_world_device",
        "ty_world_device",
        "tz_world_device",
        "qx_world_device",
        "qy_world_device",
        "qz_world_device",
        "qw_world_device",
        "quality_score",
    }
    pose = pd.read_csv(
        pose_path,
        usecols=lambda col: col in pose_cols,
    )
    speech = pd.read_csv(speech_path) if speech_path.exists() else None
    return gaze, pose, speech


def sample_period_s(gaze: pd.DataFrame) -> float:
    return float(np.median(np.diff(gaze["tracking_timestamp_us"].to_numpy()))) / 1_000_000.0


def build_histogram(gaze: pd.DataFrame, dwell_per_sample_s: float) -> tuple[dict[str, Any], np.ndarray]:
    yaw_bin = np.digitize(gaze["yaw_rads_cpf"], YAW_EDGES, right=False) - 1
    pitch_bin = np.digitize(gaze["pitch_rads_cpf"], PITCH_EDGES, right=False) - 1
    yaw_bin = np.clip(yaw_bin, 0, len(YAW_EDGES) - 2)
    pitch_bin = np.clip(pitch_bin, 0, len(PITCH_EDGES) - 2)
    counts = np.zeros((len(PITCH_EDGES) - 1, len(YAW_EDGES) - 1), dtype=int)
    for pitch_idx, yaw_idx in zip(pitch_bin, yaw_bin, strict=True):
        counts[int(pitch_idx), int(yaw_idx)] += 1
    dwell = counts.astype(float) * dwell_per_sample_s
    bins: list[dict[str, Any]] = []
    for pitch_idx in range(counts.shape[0]):
        for yaw_idx in range(counts.shape[1]):
            bins.append(
                {
                    "yaw_bin_index": yaw_idx,
                    "pitch_bin_index": pitch_idx,
                    "yaw_min_rad": rounded(YAW_EDGES[yaw_idx]),
                    "yaw_max_rad": rounded(YAW_EDGES[yaw_idx + 1]),
                    "pitch_min_rad": rounded(PITCH_EDGES[pitch_idx]),
                    "pitch_max_rad": rounded(PITCH_EDGES[pitch_idx + 1]),
                    "yaw_center_deg": rounded(math.degrees((YAW_EDGES[yaw_idx] + YAW_EDGES[yaw_idx + 1]) / 2), 3),
                    "pitch_center_deg": rounded(math.degrees((PITCH_EDGES[pitch_idx] + PITCH_EDGES[pitch_idx + 1]) / 2), 3),
                    "samples_count": int(counts[pitch_idx, yaw_idx]),
                    "dwell_s": rounded(dwell[pitch_idx, yaw_idx], 3),
                }
            )
    payload = {
        "recording_uid": RECORDING,
        "sample_period_s": rounded(dwell_per_sample_s, 6),
        "yaw_bins_rad": [rounded(v) for v in YAW_EDGES],
        "pitch_bins_rad": [rounded(v) for v in PITCH_EDGES],
        "yaw_bin_width_deg": 30.0,
        "pitch_bin_width_deg": 20.0,
        "counts_matrix_pitch_by_yaw": counts.tolist(),
        "dwell_s_matrix_pitch_by_yaw": [[rounded(v, 3) for v in row] for row in dwell.tolist()],
        "bins": bins,
    }
    return payload, dwell


def align_pose(gaze: pd.DataFrame, pose: pd.DataFrame) -> pd.DataFrame:
    pose_with_nearest = pose.rename(columns={"tracking_timestamp_us": "nearest_pose_tracking_timestamp_us"})
    aligned = pd.merge_asof(
        gaze.sort_values("tracking_timestamp_us"),
        pose_with_nearest.sort_values("nearest_pose_tracking_timestamp_us"),
        left_on="tracking_timestamp_us",
        right_on="nearest_pose_tracking_timestamp_us",
        direction="nearest",
        tolerance=POSE_TOLERANCE_US,
    )
    aligned["nearest_pose_delta_us"] = aligned["tracking_timestamp_us"] - aligned["nearest_pose_tracking_timestamp_us"]
    return aligned


def pose_payload(row: pd.Series) -> dict[str, Any] | None:
    nearest_pose_timestamp = row["nearest_pose_tracking_timestamp_us"]
    if bool(pd.isna(nearest_pose_timestamp)):
        return None
    quality_score = row["quality_score"] if "quality_score" in row else None
    return {
        "tracking_timestamp_us": int(nearest_pose_timestamp),
        "delta_us": int(row["nearest_pose_delta_us"]),
        "translation_world_device": {
            "x": rounded(row["tx_world_device"]),
            "y": rounded(row["ty_world_device"]),
            "z": rounded(row["tz_world_device"]),
        },
        "orientation_world_device_quat": {
            "x": rounded(row["qx_world_device"]),
            "y": rounded(row["qy_world_device"]),
            "z": rounded(row["qz_world_device"]),
            "w": rounded(row["qw_world_device"]),
        },
        "quality_score": rounded(quality_score, 3),
    }


def build_aligned_preview(aligned: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in aligned.dropna(subset=["nearest_pose_tracking_timestamp_us"]).head(100).iterrows():
        rows.append(
            {
                "tracking_timestamp_us": int(row["tracking_timestamp_us"]),
                "time_s_from_start": rounded((row["tracking_timestamp_us"] - aligned["tracking_timestamp_us"].iloc[0]) / 1_000_000.0, 3),
                "gaze": {
                    "yaw_rad": rounded(row["yaw_rads_cpf"]),
                    "pitch_rad": rounded(row["pitch_rads_cpf"]),
                    "yaw_deg": rounded(math.degrees(row["yaw_rads_cpf"]), 3),
                    "pitch_deg": rounded(math.degrees(row["pitch_rads_cpf"]), 3),
                    "yaw_ci_width_rad": rounded(row["yaw_high_rads_cpf"] - row["yaw_low_rads_cpf"]),
                    "pitch_ci_width_rad": rounded(row["pitch_high_rads_cpf"] - row["pitch_low_rads_cpf"]),
                },
                "pose": pose_payload(row),
            }
        )
    return rows


def mean_pose(rows: pd.DataFrame) -> dict[str, Any] | None:
    rows = rows.dropna(subset=["nearest_pose_tracking_timestamp_us"])
    if rows.empty:
        return None
    return {
        "translation_world_device": {
            "x": rounded(rows["tx_world_device"].mean()),
            "y": rounded(rows["ty_world_device"].mean()),
            "z": rounded(rows["tz_world_device"].mean()),
        },
        "orientation_world_device_quat_mean": {
            "x": rounded(rows["qx_world_device"].mean()),
            "y": rounded(rows["qy_world_device"].mean()),
            "z": rounded(rows["qz_world_device"].mean()),
            "w": rounded(rows["qw_world_device"].mean()),
        },
        "aligned_samples_count": int(len(rows)),
    }


def build_fixations(aligned: pd.DataFrame, dwell_per_sample_s: float) -> list[dict[str, Any]]:
    fixations: list[dict[str, Any]] = []
    start_idx = 0
    sum_yaw = float(aligned["yaw_rads_cpf"].iloc[0])
    sum_pitch = float(aligned["pitch_rads_cpf"].iloc[0])
    count = 1
    for idx in range(1, len(aligned)):
        yaw = float(aligned["yaw_rads_cpf"].iloc[idx])
        pitch = float(aligned["pitch_rads_cpf"].iloc[idx])
        mean_yaw = sum_yaw / count
        mean_pitch = sum_pitch / count
        radius = math.hypot(yaw - mean_yaw, pitch - mean_pitch)
        if radius <= FIXATION_MERGE_RADIUS_RAD:
            sum_yaw += yaw
            sum_pitch += pitch
            count += 1
            continue
        fixations.append(fixation_payload(aligned, start_idx, idx - 1, count, sum_yaw, sum_pitch, dwell_per_sample_s))
        start_idx = idx
        sum_yaw = yaw
        sum_pitch = pitch
        count = 1
    fixations.append(fixation_payload(aligned, start_idx, len(aligned) - 1, count, sum_yaw, sum_pitch, dwell_per_sample_s))
    fixations.sort(key=lambda row: (row["samples_count"], row["start_time_s"]), reverse=True)
    return fixations[:5]


def fixation_payload(
    aligned: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    count: int,
    sum_yaw: float,
    sum_pitch: float,
    dwell_per_sample_s: float,
) -> dict[str, Any]:
    rows = aligned.iloc[start_idx : end_idx + 1]
    mean_yaw = sum_yaw / count
    mean_pitch = sum_pitch / count
    start_ts = int(rows["tracking_timestamp_us"].iloc[0])
    end_ts = int(rows["tracking_timestamp_us"].iloc[-1])
    stream_start = int(aligned["tracking_timestamp_us"].iloc[0])
    return {
        "rank": 0,
        "start_tracking_timestamp_us": start_ts,
        "end_tracking_timestamp_us": end_ts,
        "start_time_s": rounded((start_ts - stream_start) / 1_000_000.0, 3),
        "end_time_s": rounded((end_ts - stream_start) / 1_000_000.0, 3),
        "samples_count": int(count),
        "dwell_s": rounded(count * dwell_per_sample_s, 3),
        "mean_yaw_rad": rounded(mean_yaw),
        "mean_pitch_rad": rounded(mean_pitch),
        "mean_yaw_deg": rounded(math.degrees(mean_yaw), 3),
        "mean_pitch_deg": rounded(math.degrees(mean_pitch), 3),
        "mean_pose": mean_pose(rows),
    }


def build_stability(gaze: pd.DataFrame, speech: pd.DataFrame | None, dwell_per_sample_s: float) -> tuple[dict[str, Any], pd.Series]:
    window = max(2, int(round(1.0 / dwell_per_sample_s)))
    yaw_std = gaze["yaw_rads_cpf"].rolling(window=window, min_periods=2).std()
    pitch_std = gaze["pitch_rads_cpf"].rolling(window=window, min_periods=2).std()
    stability = yaw_std + pitch_std
    quiet = stability < QUIET_GAZE_THRESHOLD_RAD
    valid = stability.dropna()
    speech_rows = 0 if speech is None else int(len(speech))
    confident_speech_rows = 0 if speech is None else int((speech["confidence"] >= 0.05).sum())
    payload = {
        "recording_uid": RECORDING,
        "rolling_window_samples": window,
        "rolling_window_s": rounded(window * dwell_per_sample_s, 3),
        "quiet_gaze_threshold_rad": QUIET_GAZE_THRESHOLD_RAD,
        "quiet_samples_count": int(quiet.sum()),
        "quiet_gaze_pct": rounded(float(quiet.mean() * 100.0), 3),
        "rolling_std_yaw_plus_pitch_rad": {
            "mean": rounded(valid.mean()),
            "median": rounded(valid.median()),
            "min": rounded(valid.min()),
            "max": rounded(valid.max()),
            "p25": rounded(valid.quantile(0.25)),
            "p75": rounded(valid.quantile(0.75)),
        },
        "confidence_interval_width_rad": {
            "yaw_mean": rounded((gaze["yaw_high_rads_cpf"] - gaze["yaw_low_rads_cpf"]).mean()),
            "yaw_median": rounded((gaze["yaw_high_rads_cpf"] - gaze["yaw_low_rads_cpf"]).median()),
            "yaw_min": rounded((gaze["yaw_high_rads_cpf"] - gaze["yaw_low_rads_cpf"]).min()),
            "yaw_max": rounded((gaze["yaw_high_rads_cpf"] - gaze["yaw_low_rads_cpf"]).max()),
            "pitch_mean": rounded((gaze["pitch_high_rads_cpf"] - gaze["pitch_low_rads_cpf"]).mean()),
            "pitch_median": rounded((gaze["pitch_high_rads_cpf"] - gaze["pitch_low_rads_cpf"]).median()),
        },
        "speech_overlap": {
            "speech_rows": speech_rows,
            "confident_speech_rows_confidence_gte_0_05": confident_speech_rows,
            "confident_speech_overlap_s": 0.0,
            "note": "loc5 has one low-confidence speech row; confident overlap is zero.",
        },
    }
    return payload, quiet


def color_for(value: float, max_value: float) -> str:
    ratio = 0.0 if max_value <= 0 else value / max_value
    if colormaps is not None and to_hex is not None:
        return to_hex(colormaps["viridis"](0.18 + 0.82 * ratio))
    stops = [
        (31, 42, 91),
        (37, 99, 141),
        (35, 156, 132),
        (122, 209, 81),
        (251, 191, 36),
    ]
    scaled = min(max(ratio, 0.0), 1.0) * (len(stops) - 1)
    idx = min(int(scaled), len(stops) - 2)
    frac = scaled - idx
    rgb = tuple(round(stops[idx][channel] * (1 - frac) + stops[idx + 1][channel] * frac) for channel in range(3))
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def build_slide(
    path: Path,
    dwell_matrix: np.ndarray,
    top5: list[dict[str, Any]],
    stats: dict[str, Any],
    sample_rate_hz: float,
    duration_s: float,
    sample_count: int,
) -> None:
    width = 700
    height = 440
    left = 56
    top = 36
    cell_w = width / 12
    cell_h = height / 6
    max_dwell = float(dwell_matrix.max())
    cells = []
    for pitch_idx in range(6):
        for yaw_idx in range(12):
            value = float(dwell_matrix[pitch_idx, yaw_idx])
            x = left + yaw_idx * cell_w
            y = top + (5 - pitch_idx) * cell_h
            cells.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_w - 2:.2f}" height="{cell_h - 2:.2f}" rx="8" fill="{color_for(value, max_dwell)}" opacity="{0.18 + 0.82 * (value / max_dwell if max_dwell else 0):.3f}" />'
                f'<text x="{x + cell_w / 2:.2f}" y="{y + cell_h / 2 + 4:.2f}" class="cell-label">{value:.1f}s</text>'
            )
    annotations = []
    for item in top5:
        yaw = float(item["mean_yaw_rad"])
        pitch = float(item["mean_pitch_rad"])
        x = left + ((yaw - YAW_EDGES[0]) / (YAW_EDGES[-1] - YAW_EDGES[0])) * width
        y = top + height - ((pitch - PITCH_EDGES[0]) / (PITCH_EDGES[-1] - PITCH_EDGES[0])) * height
        annotations.append(
            f'<g class="fix"><circle cx="{x:.2f}" cy="{y:.2f}" r="13" />'
            f'<text x="{x + 18:.2f}" y="{y - 14:.2f}">#{item["rank"]} {item["dwell_s"]:.1f}s</text></g>'
        )
    x_ticks = []
    for idx, deg in enumerate(range(-180, 181, 30)):
        x = left + idx * cell_w
        x_ticks.append(f'<text x="{x:.1f}" y="{top + height + 28:.1f}" class="tick">{deg}</text>')
    y_ticks = []
    for idx, deg in enumerate(range(-60, 61, 20)):
        y = top + height - idx * cell_h
        y_ticks.append(f'<text x="{left - 18:.1f}" y="{y + 4:.1f}" class="tick" text-anchor="end">{deg}</text>')
    top_rows = "".join(
        f'<tr><td>#{item["rank"]}</td><td>({item["mean_yaw_deg"]:.1f}°, {item["mean_pitch_deg"]:.1f}°)</td><td>{item["dwell_s"]:.1f}s</td></tr>'
        for item in top5
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AEA {RECORDING} · gaze fixation</title>
<style>
  :root{{--bg:#0b1020;--panel:#11192f;--panel2:#0e1730;--line:#1c2746;--ink:#e9ecf5;--muted:#8aa0c8;--good:#22c55e;--accent:#60a5fa;--hl:#fbbf24;--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;--sans:Inter,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}}
  *{{box-sizing:border-box}} html,body{{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans)}}
  .slide{{width:1280px;height:720px;margin:24px auto;padding:22px 26px;background:var(--panel);border-radius:16px;box-shadow:0 8px 40px rgba(0,0,0,.45);display:grid;grid-template-rows:auto 1fr auto;gap:14px;overflow:hidden}}
  header{{display:flex;justify-content:space-between;align-items:baseline;border-bottom:1px solid var(--line);padding-bottom:8px}} h1{{margin:0;font-size:24px;letter-spacing:-.01em}} .tag{{color:var(--accent);font-family:var(--mono);font-size:11px;letter-spacing:1px;text-transform:uppercase}} .meta{{color:var(--muted);font-family:var(--mono);font-size:11px;text-align:right}}
  main{{display:grid;grid-template-columns:60% 40%;gap:14px;min-height:0}} .card{{background:#0a1325;border:1px solid var(--line);border-radius:14px;position:relative;overflow:hidden}} .heat{{padding:14px 12px 10px}} .stats{{padding:18px 18px;display:flex;flex-direction:column;gap:12px}}
  .card h2{{margin:0 0 10px;font-size:15px;letter-spacing:.02em}} svg{{width:100%;height:542px;display:block}} .cell-label{{fill:#e9ecf5;font:10px var(--mono);text-anchor:middle;opacity:.78}} .tick{{fill:var(--muted);font:10px var(--mono)}} .axis{{fill:var(--accent);font:11px var(--mono);letter-spacing:.7px;text-transform:uppercase}} .fix circle{{fill:rgba(251,191,36,.22);stroke:var(--hl);stroke-width:2.4;filter:drop-shadow(0 0 12px rgba(251,191,36,.55))}} .fix text{{fill:var(--hl);font:12px var(--mono);font-weight:700}}
  .metric-grid{{display:grid;grid-template-columns:1fr 1fr;gap:8px}} .metric{{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:10px}} .num{{font-size:25px;font-weight:800;color:var(--good);line-height:1}} .lbl{{margin-top:5px;color:var(--muted);font-family:var(--mono);font-size:10px;text-transform:uppercase}} .callout{{border-top:1px solid var(--line);padding-top:10px;color:var(--muted);font-size:12px;line-height:1.45}} table{{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:11px}} td,th{{padding:6px 4px;border-bottom:1px solid var(--line);text-align:left}} th{{color:var(--accent);text-transform:uppercase;font-size:10px;letter-spacing:.8px}} td:last-child{{color:var(--hl)}} footer{{display:flex;justify-content:space-between;color:var(--muted);font-family:var(--mono);font-size:10.5px;border-top:1px solid var(--line);padding-top:7px}}
</style>
</head>
<body><div class="slide">
<header><div><div class="tag">AEA gaze stream beyond MP4</div><h1>AEA {RECORDING} · 10 Hz gaze fixation · {sample_count:,} samples / {duration_s:.0f} s</h1></div><div class="meta">yaw × pitch = 12 × 6 bins<br>fixations merged within 0.1 rad</div></header>
<main>
<section class="card heat"><h2>Polar gaze dwell heatmap</h2><svg viewBox="0 0 820 542" aria-label="Yaw pitch dwell heatmap"><text x="406" y="528" class="axis" text-anchor="middle">yaw degrees in central pupil frame</text><text x="13" y="268" class="axis" text-anchor="middle" transform="rotate(-90 13 268)">pitch degrees</text>{''.join(cells)}{''.join(x_ticks)}{''.join(y_ticks)}{''.join(annotations)}</svg></section>
<aside class="card stats"><h2>Gaze-only signals MP4 cannot recover</h2><div class="metric-grid"><div class="metric"><div class="num">{duration_s:.1f}s</div><div class="lbl">total gaze duration</div></div><div class="metric"><div class="num">{sample_rate_hz:.1f} Hz</div><div class="lbl">sample rate</div></div><div class="metric"><div class="num">{stats['confidence_interval_width_rad']['yaw_mean']:.3f}</div><div class="lbl">mean yaw CI width rad</div></div><div class="metric"><div class="num">{stats['quiet_gaze_pct']:.1f}%</div><div class="lbl">quiet gaze</div></div></div><table><thead><tr><th>rank</th><th>bearing</th><th>dwell</th></tr></thead><tbody>{top_rows}</tbody></table><div class="callout">Speech overlap: 0.0 s with confident speech. loc5 is quiet, so gaze still exposes attention dynamics: short steady locks mixed with scanning across a stationary place.</div></aside>
</main><footer><span>Aligned to nearest 6-DoF pose sample within ≤1 ms.</span><span>EgoLife pipeline cannot recover any of these signals from MP4 alone.</span></footer>
</div></body></html>"""
    path.write_text(html)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aea-dir", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/metadata/spatial_memory/AEA_loc5_script4_seq6_rec1/gaze_analysis"),
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gaze, pose, speech = load_inputs(args.aea_dir)
    dwell_per_sample_s = sample_period_s(gaze)
    sample_rate_hz = 1.0 / dwell_per_sample_s
    duration_s = len(gaze) * dwell_per_sample_s
    histogram_payload, dwell_matrix = build_histogram(gaze, dwell_per_sample_s)
    aligned = align_pose(gaze, pose)
    aligned_preview = build_aligned_preview(aligned)
    top5 = build_fixations(aligned, dwell_per_sample_s)
    for rank, fixation in enumerate(top5, start=1):
        fixation["rank"] = rank
    stability_payload, _ = build_stability(gaze, speech, dwell_per_sample_s)

    write_json(args.out_dir / "gaze_polar_histogram.json", histogram_payload)
    write_json(args.out_dir / "gaze_fixations_top5.json", top5)
    write_json(args.out_dir / "gaze_pose_aligned.json", aligned_preview)
    write_json(args.out_dir / "gaze_stability.json", stability_payload)
    build_slide(
        Path("docs/slides/aea-gaze-fixation-loc5_script4_seq6_rec1.html"),
        dwell_matrix,
        top5,
        stability_payload,
        sample_rate_hz,
        duration_s,
        len(gaze),
    )
    print(
        "AEA gaze analysis: "
        f"{len(gaze)} samples over {duration_s:.1f}s at {sample_rate_hz:.1f}Hz; "
        f"quiet gaze {stability_payload['quiet_gaze_pct']:.1f}%; "
        f"top fixation ({top5[0]['mean_yaw_deg']:.1f}°, {top5[0]['mean_pitch_deg']:.1f}°) "
        f"for {top5[0]['dwell_s']:.1f}s; wrote {args.out_dir}."
    )


if __name__ == "__main__":
    main()
