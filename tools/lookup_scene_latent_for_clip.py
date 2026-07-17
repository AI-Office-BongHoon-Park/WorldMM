#!/usr/bin/env python3
"""Look up a spatial SceneLatentRef for a visual clip id.

Current EgoLife visual clip ids and AEA spatial chunks are not aligned, so most
queries return null. The helper still scans the spatial sidecar tree and returns
any scene latent whose sidecar payload explicitly references the requested clip.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_SPATIAL_ROOT = Path("output/metadata/spatial_memory")


def resolve_storage_uri(uri: str, ref_path: Path) -> Path:
    path = Path(uri)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    candidate = ref_path.parent / path
    if candidate.exists():
        return candidate
    return path


def payload_mentions_clip(payload: Any, clip_id: str) -> bool:
    if isinstance(payload, str):
        return payload == clip_id
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in {"scene_latent_ref", "point_cloud"}:
                continue
            if payload_mentions_clip(value, clip_id):
                return True
    if isinstance(payload, list):
        return any(payload_mentions_clip(item, clip_id) for item in payload)
    return False


def lookup_scene_latent_for_clip(clip_id: str, spatial_root: Path = DEFAULT_SPATIAL_ROOT) -> dict[str, Any] | None:
    for chunk_path in sorted(spatial_root.rglob("*.json")):
        try:
            chunk = json.loads(chunk_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(chunk, dict):
            continue
        scene_latent_ref = chunk.get("scene_latent_ref")
        if not scene_latent_ref or not payload_mentions_clip(chunk, clip_id):
            continue
        ply_path = resolve_storage_uri(str(scene_latent_ref.get("storage_uri", "")), chunk_path)
        return {
            "clip_id": clip_id,
            "chunk_path": str(chunk_path),
            "ply_path": str(ply_path),
            "scene_latent_ref": scene_latent_ref,
            "point_cloud": chunk.get("point_cloud"),
            "scene_latent_stats": chunk.get("scene_latent_stats"),
        }
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip_id")
    parser.add_argument("--spatial-root", type=Path, default=DEFAULT_SPATIAL_ROOT)
    args = parser.parse_args()
    result = lookup_scene_latent_for_clip(args.clip_id, args.spatial_root)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
