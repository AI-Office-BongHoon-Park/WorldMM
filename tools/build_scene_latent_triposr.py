#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Build a TripoSR single-image SceneLatentRef sidecar from one EgoLife MP4 chunk.

TripoSR source/model license: MIT (VAST-AI-Research/TripoSR; Stability AI/Tripo AI).
This PoC stores an object-level single-frame proxy mesh, not faithful room geometry.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worldmm.memory.spatial import PointCloudSidecar, SceneLatentRef

TRIPOSR_SOURCE = Path(__file__).resolve().parent / "_triposr_src"
MODEL_NAME = "stabilityai/TripoSR"
MAX_LONG_SIDE = 512
MAX_INLINE_POINTS = 1000
CLIP_LEN_SECONDS = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Path to one EgoLife MP4 chunk.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/metadata/spatial_memory/A1_JAKE/scene_latents_triposr"),
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--mesh-format", choices=["ply"], default="ply")
    parser.add_argument("--mc-resolution", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=4096)
    return parser.parse_args()


def ensure_triposr_importable() -> Any:
    try:
        from tsr.system import TSR

        return TSR
    except ModuleNotFoundError as exc:
        if TRIPOSR_SOURCE.exists():
            sys.path.insert(0, str(TRIPOSR_SOURCE))
            try:
                from tsr.system import TSR

                return TSR
            except ModuleNotFoundError as source_exc:
                raise RuntimeError(
                    "TripoSR source exists but dependencies are missing. "
                    "Install per MIT TripoSR README: uv run python -m pip install "
                    "omegaconf==2.3.0 einops==0.7.0 trimesh==4.0.5 rembg "
                    "huggingface-hub git+https://github.com/tatsy/torchmcubes.git"
                ) from source_exc
        raise RuntimeError(
            "TripoSR is not importable. PyPI package 'triposr' was unavailable in this environment; "
            "clone MIT repo with: git clone https://github.com/VAST-AI-Research/TripoSR tools/_triposr_src"
        ) from exc


def extract_middle_frame(video_path: Path) -> tuple[Image.Image, int, int]:
    video = VideoReader(str(video_path), ctx=cpu(0))
    frame_index = max(0, len(video) // 2)
    frame = video[frame_index].asnumpy()
    return Image.fromarray(frame).convert("RGB"), frame_index, len(video)


def resize_long_side(image: Image.Image, max_long_side: int = MAX_LONG_SIDE) -> Image.Image:
    width, height = image.size
    long_side = max(width, height)
    if long_side <= max_long_side:
        return image
    scale = max_long_side / float(long_side)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def parse_egolife_time_us(video_path: Path) -> int:
    token = video_path.stem.rsplit("_", 1)[-1].zfill(8)
    if len(token) != 8 or not token.isdigit():
        return 0
    seconds = int(token[0:2]) * 3600 + int(token[2:4]) * 60 + int(token[4:6]) + int(token[6:8]) / 100.0
    return int(seconds * 1_000_000)


def load_model(TSR: Any, device: str, chunk_size: int) -> Any:
    model = TSR.from_pretrained(MODEL_NAME, config_name="config.yaml", weight_name="model.ckpt")
    model.renderer.set_chunk_size(chunk_size)
    model.to(device)
    model.eval()
    return model


def is_cuda_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "cuda" in message and "out of memory" in message


def synchronize_if_cuda(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def infer_mesh(image: Image.Image, requested_device: str, mc_resolution: int, chunk_size: int) -> tuple[Any, str, float, int]:
    TSR = ensure_triposr_importable()
    candidates = ["cuda"] if requested_device == "cuda" and torch.cuda.is_available() else []
    candidates.append("cpu")
    last_error: BaseException | None = None

    for index, device in enumerate(candidates):
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        try:
            model = load_model(TSR, device, chunk_size)
            synchronize_if_cuda(device)
            with torch.no_grad():
                scene_codes = model([image], device=device)
                mesh = model.extract_mesh(scene_codes, True, resolution=mc_resolution)[0]
            synchronize_if_cuda(device)
            elapsed = time.perf_counter() - start
            vram_peak = int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
            return mesh, device, elapsed, vram_peak
        except BaseException as exc:
            last_error = exc
            if device == "cuda" and is_cuda_oom(exc) and index + 1 < len(candidates):
                torch.cuda.empty_cache()
                continue
            raise
    raise RuntimeError("TripoSR inference failed") from last_error


def as_vertices_faces(mesh: Any) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Unexpected vertex array shape: {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Unexpected face array shape: {faces.shape}")
    return vertices, faces


def write_ascii_mesh_ply(vertices: np.ndarray, faces: np.ndarray, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write(f"element face {len(faces)}\n")
        handle.write("property list uchar int vertex_indices\n")
        handle.write("end_header\n")
        for x, y, z in vertices:
            handle.write(f"{float(x):.6f} {float(y):.6f} {float(z):.6f}\n")
        for a, b, c in faces:
            handle.write(f"3 {int(a)} {int(b)} {int(c)}\n")
    return path.stat().st_size


def sample_vertices(vertices: np.ndarray, max_points: int = MAX_INLINE_POINTS) -> list[list[float]]:
    if len(vertices) <= max_points:
        sampled = vertices
    else:
        indices = np.linspace(0, len(vertices) - 1, num=max_points, dtype=np.int64)
        sampled = vertices[indices]
    return sampled.astype(float).tolist()


def build_sidecar(
    video_path: Path,
    mesh_path: Path,
    mesh_bytes: int,
    vertices: np.ndarray,
    faces: np.ndarray,
    frame_index: int,
    frame_count: int,
    input_size: tuple[int, int],
    device: str,
    inference_s: float,
    vram_peak_bytes: int,
) -> dict[str, Any]:
    coord_frame_id = f"egolife:{video_path.stem}:triposr_single_frame_proxy"
    time_us = parse_egolife_time_us(video_path)
    point_cloud = PointCloudSidecar(
        points_world_m=sample_vertices(vertices),
        graph_uid=coord_frame_id,
        time_range_us=(time_us, time_us + int(CLIP_LEN_SECONDS * 1_000_000)),
        source="triposr_single_image_mesh_vertices",
    )
    scene_latent_ref = SceneLatentRef(
        backend="triplane_triposr",
        storage_uri=str(mesh_path),
        coord_frame_id=coord_frame_id,
        time_us=time_us,
        state_kind="object_instance",
        decoder_version="triposr_v1",
        confidence=0.35,
        provenance_frames=[f"egolife:{video_path.name}:frame_{frame_index}_of_{frame_count}"],
        storage_bytes=mesh_bytes,
    )
    return {
        "video_path": str(video_path),
        "chunk_id": video_path.stem,
        "scene_latent_ref": scene_latent_ref.model_dump(mode="json"),
        "point_cloud": point_cloud.model_dump(mode="json"),
        "scene_latent_stats": {
            "license": "TripoSR source and pretrained model are MIT licensed.",
            "source_frame_index": frame_index,
            "source_frame_count": frame_count,
            "input_size_px": list(input_size),
            "device": device,
            "inference_s": round(inference_s, 3),
            "vram_peak_bytes": vram_peak_bytes,
            "vram_peak_mb": round(vram_peak_bytes / (1024 * 1024), 1),
            "vertices": int(len(vertices)),
            "faces": int(len(faces)),
            "inline_points": len(point_cloud.points_world_m),
        },
    }


def main() -> None:
    args = parse_args()
    if not args.video.exists():
        raise FileNotFoundError(args.video)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    image, frame_index, frame_count = extract_middle_frame(args.video)
    image = resize_long_side(image)
    mesh, used_device, inference_s, vram_peak = infer_mesh(image, args.device, args.mc_resolution, args.chunk_size)
    vertices, faces = as_vertices_faces(mesh)

    mesh_path = args.out_dir / f"{args.video.stem}_triposr_mesh.{args.mesh_format}"
    json_path = args.out_dir / f"{args.video.stem}_triposr.json"
    mesh_bytes = write_ascii_mesh_ply(vertices, faces, mesh_path)
    sidecar = build_sidecar(
        args.video,
        mesh_path,
        mesh_bytes,
        vertices,
        faces,
        frame_index,
        frame_count,
        image.size,
        used_device,
        inference_s,
        vram_peak,
    )
    json_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    json_bytes = json_path.stat().st_size
    print(
        "TripoSR scene latent PoC: "
        f"video={args.video} frame={frame_index}/{frame_count} device={used_device} "
        f"inference_s={inference_s:.3f} vram_peak_mb={vram_peak / (1024 * 1024):.1f} "
        f"vertices={len(vertices)} faces={len(faces)} "
        f"mesh={mesh_path} mesh_bytes={mesh_bytes} json={json_path} json_bytes={json_bytes} "
        "license=MIT verdict=single-frame object-level proxy, not faithful room reconstruction."
    )


if __name__ == "__main__":
    main()
