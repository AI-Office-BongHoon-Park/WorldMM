#!/usr/bin/env python3
"""Generate Phase 4 grounded Type1 golden QA from selected chunks."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "output/metadata/spatial_memory/A1_JAKE/phase4_chunk_selection.json"
GROUNDING_DIR = ROOT / "output/metadata/spatial_memory/A1_JAKE/grounding"
SPATIAL_OUT = ROOT / "output/golden_qa_phase4_spatial.json"
GOLD_OUT = ROOT / "output/golden_qa_phase4.json"

PLACE_WORDS = {
    "table", "dining_table", "floor", "shelf", "bin", "kitchen", "living_room", "sofa", "desk", "counter",
    "bedroom", "room", "whiteboard", "rack", "box", "chair", "stool", "refrigerator", "doorway", "ground",
}
RELATIONS = {"on", "in", "at", "near", "next_to", "behind", "in_front_of", "left_of", "right_of", "located_in", "with"}
BAD_ANSWERS = {"I", "i", "me", "my", "we", "you", "they", "unknown", "original_position"}
LETTERS = ["A", "B", "C", "D"]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def label(ts: str) -> str:
    raw = str(int(ts) % 100_000_000).zfill(8)
    return f"DAY1 {raw[:2]}:{raw[2:4]}:{raw[4:6]}"


def query_time(ts: str) -> dict[str, str]:
    raw = str(int(ts) % 100_000_000).zfill(8)
    return {"date": "DAY1", "time": f"{raw[:2]}:{raw[2:4]}:{raw[4:6]}"}


def clean(text: str) -> str:
    return str(text).replace("_", " ").strip()


def make_choices(gold: str, pool: list[str]) -> tuple[dict[str, str], str]:
    values = [clean(gold)]
    for candidate in pool:
        c = clean(candidate)
        if c and c.lower() != values[0].lower() and c not in values:
            values.append(c)
        if len(values) == 4:
            break
    fallback = ["table", "floor", "shelf", "kitchen", "living room", "sofa", "desk", "box", "counter", "chair"]
    for candidate in fallback:
        if candidate.lower() != values[0].lower() and candidate not in values:
            values.append(candidate)
        if len(values) == 4:
            break
    choices = {letter: values[i] for i, letter in enumerate(LETTERS)}
    return choices, "A"


def center(record: dict[str, Any], side: str) -> list[float] | None:
    g = record.get(f"{side}_grounding")
    if not g:
        return None
    return g.get("bbox_center")


def dist(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def grounded_records(sidecar: dict[str, Any], triples: list[list[str]]) -> list[tuple[int, list[str], dict[str, Any]]]:
    rows = []
    for key, record in sorted(sidecar.items(), key=lambda item: int(item[0].split("_")[-1])):
        idx = int(key.split("_")[-1])
        if idx < len(triples):
            rows.append((idx, triples[idx], record))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--grounding-dir", type=Path, default=GROUNDING_DIR)
    parser.add_argument("--spatial-output", type=Path, default=SPATIAL_OUT)
    parser.add_argument("--output", type=Path, default=GOLD_OUT)
    parser.add_argument("--target", type=int, default=60)
    args = parser.parse_args()

    selection = read_json(args.selection)
    selected = []
    spatial_doc: dict[str, Any] = {}
    all_objects: list[str] = []
    for row in selection["chunks"]:
        cid = str(row["chunk_id"])
        triples = row["triples"]
        sidecar_path = args.grounding_dir / f"{cid}.json"
        if not sidecar_path.exists():
            continue
        sidecar = read_json(sidecar_path)
        if not sidecar:
            continue
        spatial_doc[cid] = {"consolidated_spatial_triples": triples}
        selected.append((cid, triples, sidecar))
        for triple in triples:
            all_objects.extend([triple[0], triple[2]])

    questions: list[dict[str, Any]] = []
    for cid, triples, sidecar in selected:
        rows = grounded_records(sidecar, triples)
        if not rows:
            continue
        chunk_label = label(cid)
        pool = [t[2] for t in triples] + all_objects
        made = 0
        for idx, triple, record in rows:
            subj, pred, obj = triple
            if pred not in RELATIONS or str(obj) in BAD_ANSWERS or str(subj) in BAD_ANSWERS:
                continue
            choices, gold = make_choices(obj, pool)
            questions.append({
                "id": f"GQA-PH4-WHERE-{cid}-{idx:02d}",
                "source_pool": "phase4_grounded_generated",
                "source_id": f"{cid}:{idx}",
                "type_label": "Type1",
                "type_name": "Spatial-required",
                "question": f"Where was {clean(subj)} at approximately {chunk_label}?",
                "choices": choices,
                "gold": gold,
                "gold_answer": clean(obj),
                "query_time": query_time(cid),
                "reasoning_trace_evidence": f"phase4 grounded chunk {cid}: ({subj}, {pred}, {obj})",
                "evidence_axis_contains_answer": ["spatial+grounded"],
                "evidence_chunk": cid,
                "evidence_chunk_label": chunk_label,
                "evidence_triples": [{"id": f"spatial_{cid}_{idx}", "triple": triple}],
            })
            made += 1
            if made >= 1:
                break
        candidates = []
        for idx, triple, record in rows:
            subj_c = center(record, "subject")
            obj_c = center(record, "object")
            if subj_c and obj_c:
                candidates.append((idx, triple, subj_c, obj_c, dist(subj_c, obj_c)))
        if candidates:
            nearest = min(candidates, key=lambda item: item[4])
            choices_map = {}
            options = [nearest]
            for item in candidates:
                if item[0] != nearest[0]:
                    options.append(item)
                if len(options) == 4:
                    break
            while len(options) < 4:
                options.append(options[-1])
            for letter, item in zip(LETTERS, options):
                t = item[1]
                choices_map[letter] = f"{clean(t[0])} and {clean(t[2])}"
            questions.append({
                "id": f"GQA-PH4-DIST-{cid}",
                "source_pool": "phase4_grounded_generated",
                "source_id": f"{cid}:distance",
                "type_label": "Type1",
                "type_name": "Spatial-required",
                "question": f"Using grounded relative 3D centers at {chunk_label}, which subject-object pair was closest?",
                "choices": choices_map,
                "gold": "A",
                "gold_answer": choices_map["A"],
                "query_time": query_time(cid),
                "reasoning_trace_evidence": f"phase4 grounded bbox centers in chunk {cid}",
                "evidence_axis_contains_answer": ["spatial+grounded"],
                "evidence_chunk": cid,
                "evidence_chunk_label": chunk_label,
                "evidence_triples": [{"id": f"spatial_{cid}_{nearest[0]}", "triple": nearest[1]}],
            })

    questions = questions[: args.target]
    doc = {
        "schema_version": 1,
        "target_balance": "phase4 grounded Type1 generated 1-2 per grounded chunk",
        "counts": dict(Counter(q["type_label"] for q in questions)),
        "coverage": {
            "selected_chunks": len(selection["chunks"]),
            "grounded_chunks_with_questions": len({q["evidence_chunk"] for q in questions}),
            "questions": len(questions),
            "grounded_chunks": sorted({q["evidence_chunk"] for q in questions}),
        },
        "questions": questions,
    }
    write_json(args.spatial_output, spatial_doc)
    write_json(args.output, doc)
    print(f"wrote {len(questions)} questions from {doc['coverage']['grounded_chunks_with_questions']} grounded chunks to {args.output}")


if __name__ == "__main__":
    main()
