#!/usr/bin/env python3
"""Build one SpatialMemory grounding file from A1_JAKE geometry sidecars."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("output/metadata/spatial_memory/A1_JAKE")
DEFAULT_SPATIAL_FILE = DEFAULT_ROOT / "spatial_consolidation_results_chatgpt-gpt-5.4.json"
DEFAULT_OUTPUT = DEFAULT_ROOT / "unified_grounding_a1_jake.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def chunk_id_to_timestamp(chunk_id: str) -> str | None:
    match = re.search(r"DAY(\d+)_A1_JAKE_(\d+)", chunk_id)
    if match:
        return f"{int(match.group(1))}{match.group(2).zfill(8)}"
    if re.fullmatch(r"\d+", chunk_id):
        return chunk_id
    return None


def triple_ids_for_timestamp(spatial_data: dict[str, Any], timestamp: str) -> list[str]:
    triples = spatial_data.get(timestamp, {}).get("consolidated_spatial_triples", [])
    return [f"spatial_{timestamp}_{idx}" for idx in range(len(triples))]


def merge_primary(record: dict[str, Any], sidecar: dict[str, Any]) -> None:
    if sidecar.get("scene_latent_ref") and "scene_latent_ref" not in record:
        record["scene_latent_ref"] = sidecar["scene_latent_ref"]
    if sidecar.get("point_cloud") and "point_cloud" not in record:
        record["point_cloud"] = sidecar["point_cloud"]


def add_latent_sidecars(unified: dict[str, Any], spatial_data: dict[str, Any], sidecar_path: Path) -> int:
    sidecar = read_json(sidecar_path)
    timestamp = chunk_id_to_timestamp(str(sidecar.get("chunk_id", "")))
    if not timestamp:
        return 0
    triple_ids = triple_ids_for_timestamp(spatial_data, timestamp)
    for triple_id in triple_ids:
        record = unified.setdefault(triple_id, {})
        merge_primary(record, sidecar)
        record.setdefault("sidecar_sources", []).append(str(sidecar_path))
    return len(triple_ids)


def build_unified(root: Path, spatial_file: Path) -> dict[str, Any]:
    spatial_data = read_json(spatial_file) if spatial_file.exists() else {}
    unified: dict[str, Any] = {}

    for path in sorted((root / "grounding").glob("*.json")):
        data = read_json(path)
        for triple_id, record in data.items():
            unified.setdefault(triple_id, {}).update(record)
            unified[triple_id].setdefault("sidecar_sources", []).append(str(path))

    for pattern in ["scene_latents_mast3r/*.json", "scene_latents_triposr/*.json"]:
        for path in sorted(root.glob(pattern)):
            add_latent_sidecars(unified, spatial_data, path)

    return dict(sorted(unified.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--spatial-file", type=Path, default=DEFAULT_SPATIAL_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    unified = build_unified(args.root, args.spatial_file)
    write_json(args.output, unified)
    source_counts = {
        "depth_grounding_records": sum(1 for item in unified.values() if item.get("subject_grounding") or item.get("object_grounding")),
        "scene_latent_records": sum(1 for item in unified.values() if item.get("scene_latent_ref")),
        "point_cloud_records": sum(1 for item in unified.values() if item.get("point_cloud")),
    }
    print(f"Wrote {len(unified)} unified grounding records to {args.output}")
    print(json.dumps(source_counts, sort_keys=True))


if __name__ == "__main__":
    main()
