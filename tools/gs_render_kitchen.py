#!/usr/bin/env python3
"""Render the A1_JAKE DAY1 kitchen PoC view.

This mirrors the docs/spatial-rendering-mocks.md gs_render(place, pose)
contract, but the current artifact is NOT a real Gaussian Splatting model.
The checked-in CLI uses an honest Depth Anything V2 depth-warp fake: it loads
one kitchen frame, estimates relative depth, applies a small synthetic lateral
camera shift, and writes a labelled PNG. The sidecar JSON records why real GS
training was blocked on this 6 GB GPU workstation.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, PngImagePlugin

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "output" / "gs_models" / "A1_JAKE" / "DAY1"
RENDER_DIR = ROOT / "output" / "gs_renders" / "A1_JAKE" / "DAY1"
SOURCE_IMAGE = ROOT / "output" / "thumbnails" / "A1_JAKE" / "DAY1" / "thumb_113355400.jpg"
SOURCE_VIDEO = ROOT / "data" / "EgoLife" / "A1_JAKE" / "DAY1" / "DAY1_A1_JAKE_11353000.mp4"
MODEL_JSON = MODEL_DIR / "kitchen.ply.status.json"
MODEL_PLY = MODEL_DIR / "kitchen.ply"
DEFAULT_OUTPUT = RENDER_DIR / "kitchen_view_01.png"
KST = timezone(timedelta(hours=9))


def _resize_width(image: Image.Image, width: int) -> Image.Image:
    if image.width == width:
        return image.convert("RGB")
    height = max(1, round(image.height * width / image.width))
    return image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)


def _load_source_image(image_size_px: int) -> Image.Image:
    if SOURCE_IMAGE.exists():
        return _resize_width(Image.open(SOURCE_IMAGE), image_size_px)

    from decord import VideoReader, cpu  # type: ignore[import-not-found]

    vr = VideoReader(str(SOURCE_VIDEO), ctx=cpu(0))
    frame = vr[min(len(vr) - 1, 100)].asnumpy()
    return _resize_width(Image.fromarray(frame), image_size_px)


def _depth_anything_v2(image: Image.Image) -> tuple[np.ndarray, str, str | None]:
    try:
        import torch
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        pipe = pipeline(
            "depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=device,
        )
        result = pipe(image)
        tensor = result.get("predicted_depth")
        if tensor is not None:
            depth = tensor.detach().float().cpu().numpy()
        else:
            depth = np.asarray(result["depth"].resize(image.size), dtype=np.float32)
        backend = "Depth Anything V2 Small (transformers depth-anything/Depth-Anything-V2-Small-hf)"
        return _normalize_depth(depth), backend, None
    except Exception as exc:  # pragma: no cover - only used if model cache/runtime fails.
        gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        y_ramp = np.linspace(0.15, 1.0, gray.shape[0], dtype=np.float32)[:, None]
        depth = (0.65 * (1.0 - gray)) + (0.35 * y_ramp)
        return _normalize_depth(depth), "proxy grayscale+ramp depth", f"{type(exc).__name__}: {exc}"


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    depth = depth - float(np.nanmin(depth))
    scale = float(np.nanmax(depth)) or 1.0
    return np.clip(depth / scale, 0.0, 1.0)


def _depth_warp(image: Image.Image, depth: np.ndarray, shift_px: int) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    if depth.shape != (height, width):
        depth_img = Image.fromarray(np.uint8(depth * 255)).resize((width, height), Image.Resampling.BILINEAR)
        depth = np.asarray(depth_img, dtype=np.float32) / 255.0

    yy, xx = np.indices((height, width))
    displacement = np.round((depth - 0.5) * float(shift_px)).astype(np.int32)
    src_x = np.clip(xx - displacement, 0, width - 1)
    warped = rgb[yy, src_x]

    # Fill small tears by blending with the source frame; this is still labelled fake.
    blended = (0.92 * warped.astype(np.float32) + 0.08 * rgb.astype(np.float32)).astype(np.uint8)
    return Image.fromarray(blended, "RGB")


def _write_proxy_ply(image: Image.Image, depth: np.ndarray, path: Path) -> None:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    step = max(1, width // 80)
    points: list[tuple[float, float, float, int, int, int]] = []
    focal = float(width)
    for y in range(0, height, step):
        for x in range(0, width, step):
            z = 0.5 + float(depth[y, x]) * 2.0
            px = (x - width / 2.0) * z / focal
            py = (y - height / 2.0) * z / focal
            r, g, b = [int(v) for v in rgb[y, x]]
            points.append((px, -py, z, r, g, b))

    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write("comment WorldMM kitchen depth-warp fake point cloud; not a real Gaussian splat\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for px, py, z, r, g, b in points:
            handle.write(f"{px:.6f} {py:.6f} {z:.6f} {r} {g} {b}\n")


def _write_status(depth_backend: str, depth_error: str | None, output_path: Path) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")
    payload: dict[str, Any] = {
        "place": "kitchen",
        "artifact_kind": "depth-warp fake fallback",
        "is_real_gaussian_splat": False,
        "ply_path": str(MODEL_PLY),
        "render_path": str(output_path),
        "source_image": str(SOURCE_IMAGE),
        "source_video": str(SOURCE_VIDEO),
        "created_at": now,
        "depth_backend": depth_backend,
        "depth_backend_error": depth_error,
        "training_attempts": [
            {
                "stage": "full gsplat training",
                "result": "blocked",
                "details": "gsplat==1.5.3 installed and imported, but COLMAP is not installed, so no calibrated camera poses/intrinsics could be generated for 3DGS training.",
            },
            {
                "stage": "instant-ngp / smaller GS variant",
                "result": "blocked",
                "details": "instant-ngp executable was absent; uv pip install instant-ngp==0.1.0 failed because no such registry package exists.",
            },
            {
                "stage": "Depth Anything V2 + simple depth warp",
                "result": "succeeded",
                "details": "Depth Anything V2 Small estimated relative depth for one kitchen thumbnail, then a small lateral warp generated the labelled PNG.",
            },
        ],
        "vram_measurements_mib": {
            "nvidia_smi_initial_total_used_free": [6144, 965, 4807],
            "torch_mem_get_info_free_total_bytes": [4949475328, 6050938880],
            "after_probes_total_used_free": [6144, 965, 4807],
        },
    }
    MODEL_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def gs_render(
    place: str,
    pose: dict | None = None,
    model_dir: Path = MODEL_DIR,
    image_size_px: int = 480,
) -> Image.Image | None:
    """Return a 480px-wide novel-ish kitchen view as a depth-warp fake.

    This preserves the WorldMM `gs_render(place, pose)` interface, but the
    backend is not a real Gaussian splat. `pose` may include `shift_px`; other
    fields are accepted for contract compatibility and ignored.
    """
    if place != "kitchen":
        return None
    shift_px = int((pose or {}).get("shift_px", 24))
    source = _load_source_image(image_size_px)
    depth, _backend, _error = _depth_anything_v2(source)
    return _depth_warp(source, depth, shift_px=shift_px)


def render_to_file(output_path: Path = DEFAULT_OUTPUT, image_size_px: int = 480) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    source = _load_source_image(image_size_px)
    depth, backend, depth_error = _depth_anything_v2(source)
    rendered = _depth_warp(source, depth, shift_px=24)
    _write_proxy_ply(source, depth, MODEL_PLY)
    _write_status(backend, depth_error, output_path)

    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("WorldMM-render-kind", "depth-warp fake; NOT real Gaussian Splatting")
    metadata.add_text("WorldMM-place", "kitchen")
    metadata.add_text("WorldMM-depth-backend", backend)
    metadata.add_text("WorldMM-model-status", str(MODEL_JSON))
    rendered.save(output_path, pnginfo=metadata)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render WorldMM kitchen GS PoC fallback view.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="PNG output path")
    parser.add_argument("--width", type=int, default=480, help="output width in pixels")
    args = parser.parse_args()
    path = render_to_file(args.out, args.width)
    print(path)


if __name__ == "__main__":
    main()
