#!/usr/bin/env python3
"""Complete Phase 4 ablation by reusing worker files and filling missing singleton trials."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ["spatial_OFF", "spatial_ON_text", "spatial_ON_grounded"]
CONFIG_DESC = {
    "spatial_OFF": "E+S+V, no spatial axis",
    "spatial_ON_text": "E+S+V+Spatial, text triples only",
    "spatial_ON_grounded": "E+S+V+Spatial, unified geometry grounding",
}


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def add_existing_worker_results(by_id: dict[str, dict[str, Any]], work_dir: Path, trials: int) -> None:
    for trial in range(1, trials + 1):
        for config in CONFIGS:
            for path in sorted(work_dir.glob(f"result_{config}_{trial}_*.json")):
                try:
                    rows = read_json(path)
                except Exception:
                    continue
                for row in rows:
                    qid = row.get("question_id")
                    if qid in by_id and len(by_id[qid]["trials"][config]) < trial:
                        # Worker files are one result per question for this exact trial.
                        by_id[qid]["trials"][config].append(row["result"])


def run_single(question: dict[str, Any], config: str, trial: int, args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
    work_dir = Path(args.work_dir)
    safe = question["id"].replace("/", "_")
    payload_path = work_dir / f"singleton_payload_{config}_{trial}_{safe}.json"
    output_path = work_dir / f"singleton_result_{config}_{trial}_{safe}.json"
    if output_path.exists():
        rows = read_json(output_path)
        return question["id"], config, rows[0]["result"]
    payload = {
        "config": config,
        "trial": trial,
        "questions": [question],
        "episodic_caption_dir": args.episodic_caption_dir,
        "semantic_file": args.semantic_file,
        "spatial_file": args.spatial_file,
        "grounding_file": args.grounding_file,
        "visual_embeddings_file": args.visual_embeddings_file,
        "visual_clips_file": args.visual_clips_file,
        "load_episodic": args.load_episodic,
        "load_semantic": args.load_semantic,
        "load_visual": args.load_visual,
        "model": args.model,
        "embedding_model": args.embedding_model,
        "max_rounds": args.max_rounds,
        "cache_dir": f".cache/golden_phase4_singleton_{config}_{trial}_{safe}",
        "cache_tag": f"golden_phase4_singleton_{config}_{trial}_{safe}",
    }
    write_json(payload_path, payload)
    command = [sys.executable, str(ROOT / "tools/run_golden_qa_phase4_ablation.py"), "--worker-payload", str(payload_path), "--worker-output", str(output_path)]
    start = time.perf_counter()
    try:
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=args.timeout_sec, check=False)
    except subprocess.TimeoutExpired as exc:
        result = {
            "trial": trial,
            "prediction": "",
            "letter": "",
            "correct": False,
            "axis_selections": [],
            "retrieved_spatial_text": [],
            "retrieved_grounding_string": False,
            "reasoning_summary": f"singleton timeout after {args.timeout_sec}s",
            "num_rounds": 0,
            "error": "timeout",
        }
        write_json(output_path, [{"question_id": question["id"], "result": result}])
        return question["id"], config, result
    if completed.returncode != 0:
        result = {
            "trial": trial,
            "prediction": "",
            "letter": "",
            "correct": False,
            "axis_selections": [],
            "retrieved_spatial_text": [],
            "retrieved_grounding_string": False,
            "reasoning_summary": (completed.stderr or completed.stdout)[-900:],
            "num_rounds": 0,
            "error": f"returncode {completed.returncode}",
        }
        write_json(output_path, [{"question_id": question["id"], "result": result}])
        return question["id"], config, result
    rows = read_json(output_path)
    result = rows[0]["result"]
    result.setdefault("wall_time_sec", round(time.perf_counter() - start, 2))
    return question["id"], config, result


def finalize_questions(by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in by_id.values():
        row = dict(item)
        for config in CONFIGS:
            row[f"{config}_correct"] = sum(1 for trial in row["trials"].get(config, []) if trial.get("correct"))
        rows.append(row)
    return sorted(rows, key=lambda item: item["id"])


def aggregate(per_question: list[dict[str, Any]], trials: int) -> dict[str, Any]:
    total = len(per_question) * trials
    return {config: {"correct": sum(item[f"{config}_correct"] for item in per_question), "total": total, "accuracy": round(sum(item[f"{config}_correct"] for item in per_question) / total, 4) if total else 0.0} for config in CONFIGS}


def sample_grounding_text(per_question: list[dict[str, Any]]) -> str:
    for item in per_question:
        for trial in item["trials"].get("spatial_ON_grounded", []):
            for text in trial.get("retrieved_spatial_text", []):
                if any(marker in text for marker in ["_center=", "@ (", "scene_latent=", "[points="]):
                    return text[:1200]
    return ""


def make_results(args: argparse.Namespace, sample_doc: dict[str, Any], per_question: list[dict[str, Any]]) -> dict[str, Any]:
    aggs = aggregate(per_question, args.trials)
    off, text, grounded = aggs["spatial_OFF"], aggs["spatial_ON_text"], aggs["spatial_ON_grounded"]
    total = off["total"]
    text_delta = ((text["correct"] - off["correct"]) / total * 100) if total else 0.0
    grounded_delta = ((grounded["correct"] - off["correct"]) / total * 100) if total else 0.0
    return {
        "protocol": {
            "subject": "A1_JAKE",
            "day": "DAY1",
            "trials_per_config": args.trials,
            "workers": args.workers,
            "configs": CONFIG_DESC,
            "model": args.model,
            "embedding_model": args.embedding_model,
            "sample_file": args.sample,
            "spatial_file": args.spatial_file,
            "grounding_file": args.grounding_file,
            "axis_data_loaded": {"episodic": args.load_episodic, "semantic": args.load_semantic, "visual": args.load_visual, "spatial_text": True, "spatial_grounding": True},
            "completion_mode": "singleton_fill_missing_with_timeout",
            "timeout_sec": args.timeout_sec,
        },
        "coverage": sample_doc.get("coverage", {}),
        "aggregates": aggs,
        "delta_analysis": {
            "spatial_ON_text_minus_spatial_OFF_pp": round(text_delta, 1),
            "spatial_ON_grounded_minus_spatial_OFF_pp": round(grounded_delta, 1),
            "grounded_minus_text_pp": round(grounded_delta - text_delta, 1),
        },
        "geometry_display_sample": sample_grounding_text(per_question),
        "per_question": per_question,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", default="output/golden_qa_phase4.json")
    parser.add_argument("--output", default="output/golden_qa_phase4_results.json")
    parser.add_argument("--work-dir", default="output/golden_qa_phase4_workers")
    parser.add_argument("--episodic-caption-dir", default="data/EgoLife/EgoLifeCap/A1_JAKE")
    parser.add_argument("--semantic-file", default="output/metadata/semantic_memory/A1_JAKE/semantic_consolidation_results_chatgpt-gpt-5.4.json")
    parser.add_argument("--spatial-file", default="output/golden_qa_phase4_spatial.json")
    parser.add_argument("--grounding-file", default="output/metadata/spatial_memory/A1_JAKE/unified_grounding_a1_jake.json")
    parser.add_argument("--visual-embeddings-file", default="output/metadata/visual_memory/A1_JAKE/visual_embeddings_clip-ViT-B-32.pkl")
    parser.add_argument("--visual-clips-file", default="output/metadata/visual_memory/A1_JAKE/visual_clips_clip-ViT-B-32.json")
    parser.add_argument("--load-episodic", action="store_true")
    parser.add_argument("--load-semantic", action="store_true")
    parser.add_argument("--load-visual", action="store_true")
    parser.add_argument("--model", default="chatgpt-gpt-5.4")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-sec", type=int, default=240)
    args = parser.parse_args()

    sample_doc = read_json(args.sample)
    questions = sorted(sample_doc["questions"], key=lambda q: q["id"])
    by_id = {q["id"]: {**q, "trials": {config: [] for config in CONFIGS}} for q in questions}
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    add_existing_worker_results(by_id, work_dir, args.trials)
    print("loaded existing", {config: sum(len(q["trials"][config]) for q in by_id.values()) for config in CONFIGS}, flush=True)

    for trial in range(1, args.trials + 1):
        for config in CONFIGS:
            missing = [q for q in questions if len(by_id[q["id"]]["trials"][config]) < trial]
            print(f"fill trial={trial} config={config} missing={len(missing)}", flush=True)
            if not missing:
                continue
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(run_single, question, config, trial, args) for question in missing]
                done = 0
                for future in as_completed(futures):
                    qid, cfg, result = future.result()
                    by_id[qid]["trials"][cfg].append(result)
                    done += 1
                    if done % 5 == 0 or done == len(missing):
                        per_question = finalize_questions(by_id)
                        write_json(args.output, make_results(args, sample_doc, per_question))
                        print(f"  {done}/{len(missing)}", flush=True)
            per_question = finalize_questions(by_id)
            write_json(args.output, make_results(args, sample_doc, per_question))
    per_question = finalize_questions(by_id)
    write_json(args.output, make_results(args, sample_doc, per_question))
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
