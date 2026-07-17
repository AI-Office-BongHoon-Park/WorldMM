#!/usr/bin/env python3
"""Curate Phase 4 golden QA by grounded lift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def top_display_snippet(question: dict[str, Any]) -> str:
    texts: list[str] = []
    for trial in question.get("trials", {}).get("spatial_ON_grounded", []):
        texts.extend(trial.get("retrieved_spatial_text", []))
    grounded = [t for t in texts if any(marker in t for marker in ["_center=", "@ (", "scene_latent=", "[points="])]
    source = grounded[0] if grounded else (texts[0] if texts else "")
    return source[:1200]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("output/golden_qa_phase4_results.json"))
    parser.add_argument("--output", type=Path, default=Path("output/golden_qa_phase4_curated.json"))
    parser.add_argument("--min-items", type=int, default=12)
    parser.add_argument("--max-items", type=int, default=20)
    args = parser.parse_args()

    results = read_json(args.results)
    trials = int(results["protocol"]["trials_per_config"])
    ranked = []
    for question in results["per_question"]:
        off = question.get("spatial_OFF_correct", 0)
        text = question.get("spatial_ON_text_correct", 0)
        grounded = question.get("spatial_ON_grounded_correct", 0)
        item = dict(question)
        item["lift_grounded_vs_off"] = grounded - off
        item["delta_grounding_vs_text"] = grounded - text
        item["accuracy_lift_grounded_vs_off_pp"] = round((grounded - off) / trials * 100, 1)
        item["accuracy_delta_grounding_vs_text_pp"] = round((grounded - text) / trials * 100, 1)
        item["top_display_str_snippet"] = top_display_snippet(question)
        ranked.append(item)
    ranked.sort(key=lambda q: (q["lift_grounded_vs_off"], q["delta_grounding_vs_text"], q["spatial_ON_grounded_correct"]), reverse=True)
    positive = [q for q in ranked if q["lift_grounded_vs_off"] > 0]
    curated = positive[: args.max_items]
    if len(curated) < args.min_items:
        used = {q["id"] for q in curated}
        curated.extend([q for q in ranked if q["id"] not in used][: args.min_items - len(curated)])
    doc = {
        "schema_version": 1,
        "source_results": str(args.results),
        "selection_rule": "rank by spatial_ON_grounded_correct - spatial_OFF_correct, tie-break by grounded-text delta",
        "trials_per_config": trials,
        "count": len(curated),
        "aggregates": results.get("aggregates", {}),
        "delta_analysis": results.get("delta_analysis", {}),
        "questions": curated,
        "top5_case_studies": curated[:5],
    }
    write_json(args.output, doc)
    print(f"wrote {len(curated)} curated questions to {args.output}")


if __name__ == "__main__":
    main()
