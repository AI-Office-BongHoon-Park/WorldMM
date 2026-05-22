#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Build a MASt3R pointmap scene latent from one EgoLife MP4 chunk.

This PoC clones/uses MASt3R under ``tools/_mast3r_src`` and loads the
``naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric`` checkpoint through
Hugging Face. MASt3R code and checkpoint are CC BY-NC-SA 4.0 non-commercial;
this builder is for research/non-commercial evaluation only and is not for
production or commercial deployment.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import decord
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worldmm.memory.spatial import PointCloudSidecar, SceneLatentRef


CHECKPOINT = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
DECODER_VERSION = "mast3r_512_catmlpdpt_metric"
MAST3R_REPO_URL = "https://github.com/naver/mast3r"
MAST3R_SRC = Path(__file__).resolve().parent / "_mast3r_src"
MAX_INLINE_POINTS = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/metadata/spatial_memory/A1_JAKE/scene_latents_mast3r"),
    )
    parser.add_argument("--num-keyframes", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def ensure_mast3r_source() -> Path:
    if not MAST3R_SRC.exists():
        print(f"MASt3R source missing; cloning {MAST3R_REPO_URL} to {MAST3R_SRC}", flush=True)
        subprocess.run(["git", "clone", MAST3R_REPO_URL, str(MAST3R_SRC)], check=True)
    if not (MAST3R_SRC / "dust3r" / "dust3r").exists():
        print("MASt3R dust3r submodule missing; initializing submodules", flush=True)
        subprocess.run(["git", "-C", str(MAST3R_SRC), "submodule", "update", "--init", "--recursive"], check=True)
    return MAST3R_SRC


def load_mast3r_modules(src: Path) -> tuple[Any, Any, Any]:
    sys.path.insert(0, str(src))
    try:
        import mast3r.utils.path_to_dust3r  # noqa: F401
        from dust3r.inference import inference
        from dust3r.utils.image import load_images
        from mast3r.model import AsymmetricMASt3R
    except Exception as exc:
        raise RuntimeError(
            "MASt3R source import failed. Blocker: install the Python dependencies from "
            "tools/_mast3r_src/requirements.txt and tools/_mast3r_src/dust3r/requirements.txt "
            f"without system package changes. Original error: {type(exc).__name__}: {exc}"
        ) from exc
    return AsymmetricMASt3R, inference, load_images


def keyframe_indices(frame_count: int, num_keyframes: int) -> list[int]:
    if num_keyframes != 2:
        raise ValueError("single-chunk MASt3R PoC requires --num-keyframes 2")
    if frame_count < 2:
        raise ValueError(f"Need at least 2 frames, got {frame_count}")
    return np.linspace(0, frame_count - 1, num=num_keyframes, dtype=np.int64).tolist()


def extract_keyframes(video_path: Path, num_keyframes: int, tmp_dir: Path) -> tuple[list[Path], list[int], float | None]:
    reader = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
    indices = keyframe_indices(len(reader), num_keyframes)
    fps = float(reader.get_avg_fps()) if reader.get_avg_fps() else None
    frames = reader.get_batch(indices).asnumpy()
    image_paths: list[Path] = []
    for index, frame in zip(indices, frames, strict=True):
        image_path = tmp_dir / f"frame_{index:06d}.jpg"
        Image.fromarray(frame).save(image_path, quality=95)
        image_paths.append(image_path)
    return image_paths, indices, fps


def pick_canonical_pointmap(output: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, str, dict[str, float]]:
    pred1 = output["pred1"]
    pred2 = output["pred2"]
    pts1 = pred1["pts3d"][0].detach().cpu().float()
    pts2_key = "pts3d" if "pts3d" in pred2 else "pts3d_in_other_view"
    pts2 = pred2[pts2_key][0].detach().cpu().float()
    conf1 = pred1["conf"][0].detach().cpu().float()
    conf2 = pred2["conf"][0].detach().cpu().float()
    mean1 = float(conf1[torch.isfinite(conf1)].mean().item())
    mean2 = float(conf2[torch.isfinite(conf2)].mean().item())
    if mean1 <= mean2:
        return pts1, conf1, "pred1_lower_confidence", {"pred1_mean_conf": mean1, "pred2_mean_conf": mean2}
    return pts2, conf2, "pred2_lower_confidence", {"pred1_mean_conf": mean1, "pred2_mean_conf": mean2}


def finite_points(pointmap: torch.Tensor, confidence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    points = pointmap.reshape(-1, 3)
    conf = confidence.reshape(-1)
    mask = torch.isfinite(points).all(dim=1) & torch.isfinite(conf) & (conf > 0)
    return points[mask], conf[mask]


def downsample(points: torch.Tensor, confidence: torch.Tensor, max_points: int) -> tuple[torch.Tensor, torch.Tensor]:
    if len(points) <= max_points:
        return points, confidence
    indices = torch.linspace(0, len(points) - 1, steps=max_points).round().long()
    return points[indices], confidence[indices]


def output_paths(video_path: Path, out_dir: Path) -> tuple[Path, Path]:
    stem = video_path.stem
    return out_dir / f"{stem}_mast3r_pointmap.pt", out_dir / f"{stem}_mast3r_pointmap_sample.json"


def build_sidecar(
    video_path: Path,
    pt_path: Path,
    points: torch.Tensor,
    confidence: torch.Tensor,
    sampled_points: torch.Tensor,
    sampled_confidence: torch.Tensor,
    keyframes: list[int],
    fps: float | None,
    selected_side: str,
    confidence_stats: dict[str, float],
    inference_time_s: float,
    vram_peak_bytes: int,
) -> dict[str, Any]:
    start_us = int((keyframes[0] / fps) * 1_000_000) if fps else 0
    end_us = int((keyframes[-1] / fps) * 1_000_000) if fps else 0
    point_cloud = PointCloudSidecar(
        points_world_m=sampled_points.tolist(),
        uncertainty=(1.0 / sampled_confidence.clamp_min(1e-6)).tolist(),
        graph_uid=f"mast3r:{video_path.stem}",
        time_range_us=(start_us, end_us),
        source="mast3r_pointmap",
    )
    scene_latent_ref = SceneLatentRef(
        backend="mast3r_pointmap",
        storage_uri=str(pt_path),
        coord_frame_id=point_cloud.graph_uid,
        time_us=start_us,
        state_kind="static_scene",
        decoder_version=DECODER_VERSION,
        confidence=min(float(sampled_confidence.mean().item()), 1.0) if len(sampled_confidence) else 0.0,
        provenance_frames=[f"video:{video_path}:{index}" for index in keyframes],
        storage_bytes=pt_path.stat().st_size,
    )
    return {
        "video_path": str(video_path),
        "chunk_id": video_path.stem,
        "sequence_id": "A1_JAKE_DAY1",
        "keyframe_indices": keyframes,
        "time_range_us": [start_us, end_us],
        "point_cloud": point_cloud.model_dump(mode="json"),
        "scene_latent_ref": scene_latent_ref.model_dump(mode="json"),
        "scene_latent_stats": {
            "raw_points": int(len(points)),
            "points_after_downsample": int(len(sampled_points)),
            "selected_side": selected_side,
            "confidence": confidence_stats,
            "full_confidence_mean": float(confidence.mean().item()) if len(confidence) else 0.0,
            "sample_confidence_mean": float(sampled_confidence.mean().item()) if len(sampled_confidence) else 0.0,
            "inference_time_s": inference_time_s,
            "vram_peak_bytes": vram_peak_bytes,
            "license": "MASt3R code/checkpoint: CC BY-NC-SA 4.0 non-commercial; not for production/commercial deployment.",
        },
    }


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    src = ensure_mast3r_source()
    AsymmetricMASt3R, inference, load_images = load_mast3r_modules(src)

    torch.backends.cuda.matmul.allow_tf32 = True
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    with tempfile.TemporaryDirectory(prefix="mast3r_keyframes_") as tmp_name:
        image_paths, indices, fps = extract_keyframes(args.video, args.num_keyframes, Path(tmp_name))
        images = load_images([str(path) for path in image_paths], size=512, verbose=False)
        model = AsymmetricMASt3R.from_pretrained(CHECKPOINT).to(args.device).eval()
        start = time.perf_counter()
        output = inference([tuple(images)], model, args.device, batch_size=1, verbose=False)
        inference_time_s = time.perf_counter() - start

    pointmap, confidence, selected_side, confidence_stats = pick_canonical_pointmap(output)
    points, conf = finite_points(pointmap, confidence)
    sampled_points, sampled_conf = downsample(points, conf, MAX_INLINE_POINTS)
    pt_path, json_path = output_paths(args.video, args.out_dir)
    torch.save(
        {
            "backend": "mast3r_pointmap",
            "decoder_version": DECODER_VERSION,
            "checkpoint": CHECKPOINT,
            "video_path": str(args.video),
            "keyframe_indices": indices,
            "selected_side": selected_side,
            "pointmap": pointmap,
            "confidence": confidence,
            "points_world_m": points,
            "points_confidence": conf,
        },
        pt_path,
    )
    vram_peak_bytes = torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else 0
    sidecar = build_sidecar(
        args.video,
        pt_path,
        points,
        conf,
        sampled_points,
        sampled_conf,
        indices,
        fps,
        selected_side,
        confidence_stats,
        inference_time_s,
        int(vram_peak_bytes),
    )
    json_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    print(
        f"MASt3R pointmap scene latent: chunk={args.video.name} keyframes={indices} "
        f"selected_side={selected_side} inference_time_s={inference_time_s:.2f} "
        f"vram_peak_bytes={int(vram_peak_bytes)} points_before_sample={len(points)} "
        f"points_after_sample={len(sampled_points)} pt_file={pt_path} pt_bytes={pt_path.stat().st_size} "
        f"json_file={json_path} json_bytes={json_path.stat().st_size}. "
        "License: MASt3R is CC BY-NC-SA 4.0 non-commercial; this artifact is PoC only, not production/commercial."
    )


if __name__ == "__main__":
    main()
