#!/usr/bin/env python3
"""Run combined truly-spatial plus grounded-geometry golden QA smoke ablation."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from eval.three_vs_four_axis_ablation import build_memory, normalize_letter  # noqa: E402
from worldmm.embedding import EmbeddingModel  # type: ignore  # noqa: E402
from worldmm.llm import LLMModel  # type: ignore  # noqa: E402
from worldmm.memory.utils import QAResult  # type: ignore  # noqa: E402

DEFAULT_SPATIAL_FILE = "output/metadata/spatial_memory/A1_JAKE/spatial_consolidation_results_chatgpt-gpt-5.4.json"
DEFAULT_GROUNDING_FILE = "output/metadata/spatial_memory/A1_JAKE/unified_grounding_a1_jake.json"
DEFAULT_VISUAL_EMBEDDINGS = "output/metadata/visual_memory/A1_JAKE/visual_embeddings_clip-ViT-B-32.pkl"
DEFAULT_VISUAL_CLIPS = "output/metadata/visual_memory/A1_JAKE/visual_clips_clip-ViT-B-32.json"

CONFIGS = {
    "spatial_OFF": {
        "template": "memory_reasoning_3axis",
        "spatial": False,
        "grounding": False,
        "description": "E+S+V, no spatial axis",
    },
    "spatial_ON_text": {
        "template": "memory_reasoning",
        "spatial": True,
        "grounding": False,
        "description": "E+S+V+Spatial, text triples only",
    },
    "spatial_ON_grounded": {
        "template": "memory_reasoning",
        "spatial": True,
        "grounding": True,
        "description": "E+S+V+Spatial, unified geometry grounding",
    },
}


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_caption_files(caption_dir: Path) -> dict[str, str]:
    files = {g: str(caption_dir / f"A1_JAKE_{g}.json") for g in ["30sec", "3min", "10min", "1h"]}
    available = {g: p for g, p in files.items() if Path(p).exists()}
    if "30sec" not in available:
        raise SystemExit(f"Missing required 30sec captions at {files['30sec']}")
    return available


def to_until_time(query_time: dict[str, Any]) -> int:
    day = str(query_time.get("date", "DAY1")).replace("DAY", "").replace("Day", "") or "1"
    raw = str(query_time.get("time", "000000"))
    digits = re.sub(r"\D", "", raw)
    if len(digits) <= 6:
        digits = digits.ljust(6, "0") + "00"
    else:
        digits = digits[:8].ljust(8, "0")
    return int(day + digits)


def timestamp_label(timestamp: int) -> str:
    value = str(timestamp)
    return f"DAY{value[0]} {value[1:3]}:{value[3:5]}:{value[5:7]}"


def query_time_from_timestamp(timestamp: int) -> dict[str, str]:
    value = str(timestamp)
    return {"date": f"DAY{value[0]}", "time": f"{value[1:3]}:{value[3:5]}:{value[5:7]}"}


def dist(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(a, b)))


def center_of(grounding: dict[str, Any], triple_id: str, side: str) -> list[float] | None:
    record = grounding.get(triple_id, {})
    side_record = record.get(f"{side}_grounding")
    return side_record.get("bbox_center") if side_record else None


def make_choices(gold: str, distractors: list[str]) -> tuple[dict[str, str], str]:
    labels = ["A", "B", "C", "D"]
    options = [gold] + [item for item in distractors if item != gold][:3]
    if len(options) < 4:
        options += [f"none_{idx}" for idx in range(4 - len(options))]
    return {label: option for label, option in zip(labels, options)}, "A"


def generated_120255900(grounding: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    timestamp = 120255900
    chunk_label = timestamp_label(timestamp)
    triples_by_index = {
        58: ["puzzle_piece", "on", "plate"],
        76: ["jigsaw puzzle", "on", "table"],
    }
    puzzle_piece = center_of(grounding, "spatial_120255900_58", "subject")
    plate = center_of(grounding, "spatial_120255900_58", "object")
    jigsaw = center_of(grounding, "spatial_120255900_76", "subject")
    table = center_of(grounding, "spatial_120255900_76", "object")
    if puzzle_piece and plate and jigsaw and table:
        distances = {
            "puzzle_piece and plate": dist(puzzle_piece, plate),
            "puzzle_piece and jigsaw puzzle": dist(puzzle_piece, jigsaw),
            "jigsaw puzzle and table": dist(jigsaw, table),
            "plate and table": dist(plate, table),
        }
        closest_pair = min(distances, key=lambda item: distances[item])
    else:
        closest_pair = "puzzle_piece and plate"
    pair_choices = {
        "A": "puzzle_piece and plate",
        "B": "puzzle_piece and jigsaw puzzle",
        "C": "jigsaw puzzle and table",
        "D": "plate and table",
    }
    pair_gold = next(label for label, choice in pair_choices.items() if choice == closest_pair)
    questions = [
        {
            "id": "GQA-GROUND-120255900-58",
            "source_pool": "grounded_smoke_generated",
            "source_id": "120255900:58",
            "type_label": "Type1",
            "type_name": "Spatial-required",
            "question": f"Where was puzzle_piece at approximately {chunk_label}?",
            "choices": {"A": "plate", "B": "table", "C": "kitchen", "D": "paper_bag"},
            "gold": "A",
            "gold_answer": "plate",
            "query_time": query_time_from_timestamp(timestamp),
            "reasoning_trace_evidence": "generated from grounded chunk 120255900: (puzzle_piece, on, plate)",
            "evidence_axis_contains_answer": ["spatial"],
            "evidence_chunk": str(timestamp),
            "evidence_chunk_label": chunk_label,
            "evidence_triples": [{"id": "spatial_120255900_58", "triple": triples_by_index[58]}],
        },
        {
            "id": "GQA-GROUND-120255900-76",
            "source_pool": "grounded_smoke_generated",
            "source_id": "120255900:76",
            "type_label": "Type1",
            "type_name": "Spatial-required",
            "question": f"Where was jigsaw puzzle at approximately {chunk_label}?",
            "choices": {"A": "plate", "B": "table", "C": "kitchen", "D": "paper_bag"},
            "gold": "B",
            "gold_answer": "table",
            "query_time": query_time_from_timestamp(timestamp),
            "reasoning_trace_evidence": "generated from grounded chunk 120255900: (jigsaw puzzle, on, table)",
            "evidence_axis_contains_answer": ["spatial"],
            "evidence_chunk": str(timestamp),
            "evidence_chunk_label": chunk_label,
            "evidence_triples": [{"id": "spatial_120255900_76", "triple": triples_by_index[76]}],
        },
        {
            "id": "GQA-GROUND-120255900-DIST",
            "source_pool": "grounded_smoke_generated",
            "source_id": "120255900:58+76",
            "type_label": "Type1",
            "type_name": "Spatial-required",
            "question": f"Using the relative 3D centers at {chunk_label}, which grounded object pair was closest?",
            "choices": pair_choices,
            "gold": pair_gold,
            "gold_answer": closest_pair,
            "query_time": query_time_from_timestamp(timestamp),
            "reasoning_trace_evidence": "generated from grounded bbox centers in chunk 120255900",
            "evidence_axis_contains_answer": ["spatial"],
            "evidence_chunk": str(timestamp),
            "evidence_chunk_label": chunk_label,
            "evidence_triples": [
                {"id": "spatial_120255900_58", "triple": triples_by_index[58]},
                {"id": "spatial_120255900_76", "triple": triples_by_index[76]},
            ],
        },
    ]
    return questions, {"timestamp": str(timestamp), "triples_by_index": triples_by_index}


def generated_from_spatial_chunk(spatial_data: dict[str, Any], timestamp: int, limit: int) -> list[dict[str, Any]]:
    chunk = spatial_data[str(timestamp)]
    triples = chunk["consolidated_spatial_triples"]
    chunk_label = timestamp_label(timestamp)
    picked = [
        (33, triples[33], ["table", "stool", "left-hand side"]),
        (34, triples[34], ["floor", "table", "stool"]),
        (23, triples[23], ["floor", "box", "left-hand side"]),
    ][:limit]
    questions = []
    for idx, triple, distractors in picked:
        choices, gold = make_choices(triple[2], distractors)
        questions.append({
            "id": f"GQA-GROUND-{timestamp}-{idx:02d}",
            "source_pool": "grounded_smoke_generated",
            "source_id": f"{timestamp}:{idx}",
            "type_label": "Type1",
            "type_name": "Spatial-required",
            "question": f"Where was {triple[0]} at approximately {chunk_label}?",
            "choices": choices,
            "gold": gold,
            "gold_answer": triple[2],
            "query_time": query_time_from_timestamp(timestamp),
            "reasoning_trace_evidence": f"generated from grounded chunk {timestamp}: ({triple[0]}, {triple[1]}, {triple[2]})",
            "evidence_axis_contains_answer": ["spatial"],
            "evidence_chunk": str(timestamp),
            "evidence_chunk_label": chunk_label,
            "evidence_triples": [{"id": f"spatial_{timestamp}_{idx}", "triple": triple}],
        })
    return questions


def assemble_smoke_set(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    true_doc = read_json(args.truly_spatial_sample)
    full_doc = read_json(args.full_sample)
    spatial_data = read_json(args.spatial_file)
    grounding = read_json(args.grounding_file)
    true_questions = true_doc["questions"]
    targets = [120255900, 117513000]
    near = []
    for question in full_doc["questions"]:
        if question.get("type_label") != "Type1":
            continue
        chunk = int(question["evidence_chunk"])
        if any(abs(chunk - target) <= 500 for target in targets):
            near.append(question)
    generated_120, synthetic_chunk = generated_120255900(grounding)
    generated_117 = generated_from_spatial_chunk(spatial_data, 117513000, 3)
    generated = (near[:5] or (generated_120 + generated_117))[:5]
    questions = true_questions + [q for q in generated if q["id"] not in {item["id"] for item in true_questions}]
    questions = questions[:10]
    smoke_spatial = dict(spatial_data)
    if "120255900" not in smoke_spatial:
        triples: list[list[str]] = [["placeholder", "near", "placeholder"] for _ in range(317)]
        for idx, triple in synthetic_chunk["triples_by_index"].items():
            triples[idx] = triple
        smoke_spatial["120255900"] = {"consolidated_spatial_triples": triples}
    write_json(args.smoke_spatial_file, smoke_spatial)
    doc = {
        "schema_version": 1,
        "target_balance": "combined smoke",
        "counts": dict(Counter(q["type_label"] for q in questions)),
        "coverage": {
            "truly_spatial_questions": len(true_questions),
            "grounded_chunk_questions": len(questions) - len(true_questions),
            "grounded_chunks": ["120255900", "17513000 (loaded as timestamp 117513000)"],
            "near_pool_questions_within_5min": len(near),
            "generated_fallback_used": not bool(near),
        },
        "questions": questions,
    }
    write_json(args.sample_output, doc)
    return doc, args.smoke_spatial_file


def summarize_rounds(round_history: list[dict[str, Any]]) -> tuple[list[str], list[str], str, bool]:
    axes: list[str] = []
    spatial_texts: list[str] = []
    summaries: list[str] = []
    grounding_seen = False
    for round_info in round_history:
        memory_type = str(round_info.get("memory_type", ""))
        axes.append(memory_type)
        content = str(round_info.get("retrieved_content", ""))
        if memory_type == "spatial":
            spatial_texts.append(content)
            grounding_seen = grounding_seen or any(marker in content for marker in ["_center=", "@ (", "scene_latent=", "[points="])
        compact = re.sub(r"\s+", " ", content).strip()
        if len(compact) > 300:
            compact = compact[:297] + "..."
        summaries.append(f"R{round_info.get('round_num')} {memory_type}: {compact}")
    return axes, spatial_texts, " | ".join(summaries)[:900], grounding_seen


def run_question(world_memory: Any, question: dict[str, Any], trial: int, config: str) -> dict[str, Any]:
    query = question["question"] + f"\nTrial note: answer independently for combined smoke {config} trial {trial}."
    result: QAResult = world_memory.answer(query=query, choices=question["choices"], until_time=to_until_time(question["query_time"]))
    letter = normalize_letter(result.answer or "")
    axes, spatial_texts, summary, grounding_seen = summarize_rounds(result.round_history)
    return {
        "trial": trial,
        "prediction": result.answer or "",
        "letter": letter,
        "correct": letter == question["gold"],
        "axis_selections": axes,
        "retrieved_spatial_text": spatial_texts,
        "retrieved_grounding_string": grounding_seen,
        "reasoning_summary": summary,
        "num_rounds": result.num_rounds,
    }


def worker_main(args: argparse.Namespace) -> None:
    payload = read_json(args.worker_payload)
    config = CONFIGS[payload["config"]]
    caption_files = build_caption_files(Path(payload["episodic_caption_dir"])) if payload.get("load_episodic") else {}
    embedding_model = EmbeddingModel(text_model_name=payload["embedding_model"], device="cpu")
    embedding_model.load_model(model_type="text")
    llm = LLMModel(model_name=payload["model"], cache_dir=payload["cache_dir"])
    world_memory = build_memory(
        reasoning_template_name=config["template"],
        embedding_model=embedding_model,
        llm_model=llm,
        episodic_caption_files=caption_files,
        semantic_file=payload["semantic_file"] if payload.get("load_semantic") else "",
        spatial_file=payload["spatial_file"] if config["spatial"] else "",
        spatial_grounding_file=payload["grounding_file"] if config["grounding"] else "",
        max_rounds=payload["max_rounds"],
        episodic_cache_tag=payload["cache_tag"],
        visual_embeddings_file=payload["visual_embeddings_file"] if payload.get("load_visual") else "",
        visual_clips_file=payload["visual_clips_file"] if payload.get("load_visual") else "",
    )
    results = []
    for question in payload["questions"]:
        results.append({"question_id": question["id"], "result": run_question(world_memory, question, payload["trial"], payload["config"])})
    write_json(args.worker_output, results)


def chunked(items: list[dict[str, Any]], parts: int) -> list[list[dict[str, Any]]]:
    return [items[index::parts] for index in range(parts) if items[index::parts]]


def run_batch(payload_path: Path, output_path: Path) -> list[dict[str, Any]]:
    command = [sys.executable, str(Path(__file__).resolve()), "--worker-payload", str(payload_path), "--worker-output", str(output_path)]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"worker failed {payload_path}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}")
    return read_json(output_path)


def aggregate(per_question: list[dict[str, Any]], configs: list[str], trials: int) -> dict[str, Any]:
    total = len(per_question) * trials
    return {
        config: {
            "correct": sum(item[f"{config}_correct"] for item in per_question),
            "total": total,
            "accuracy": round(sum(item[f"{config}_correct"] for item in per_question) / total, 4) if total else 0.0,
        }
        for config in configs
    }


def sample_grounding_text(per_question: list[dict[str, Any]]) -> str:
    for item in per_question:
        for trial in item["trials"].get("spatial_ON_grounded", []):
            for text in trial.get("retrieved_spatial_text", []):
                if any(marker in text for marker in ["_center=", "@ (", "scene_latent=", "[points="]):
                    return text[:1200]
    return ""


def make_results(args: argparse.Namespace, sample_doc: dict[str, Any], per_question: list[dict[str, Any]], started_sample: str) -> dict[str, Any]:
    configs = list(CONFIGS)
    aggregates = aggregate(per_question, configs, args.trials)
    off = aggregates["spatial_OFF"]
    text = aggregates["spatial_ON_text"]
    grounded = aggregates["spatial_ON_grounded"]
    total = off["total"]
    text_delta = ((text["correct"] - off["correct"]) / total * 100) if total else 0.0
    grounded_delta = ((grounded["correct"] - off["correct"]) / total * 100) if total else 0.0
    return {
        "protocol": {
            "subject": "A1_JAKE",
            "day": "DAY1",
            "trials_per_config": args.trials,
            "workers": args.workers,
            "configs": {name: cfg["description"] for name, cfg in CONFIGS.items()},
            "model": args.model,
            "embedding_model": args.embedding_model,
            "sample_file": started_sample,
            "spatial_file": args.smoke_spatial_file,
            "grounding_file": args.grounding_file,
            "visual_embeddings_file": args.visual_embeddings_file,
            "visual_clips_file": args.visual_clips_file,
            "axis_data_loaded": {
                "episodic": args.load_episodic,
                "semantic": args.load_semantic,
                "visual": args.load_visual,
                "spatial_text": True,
                "spatial_grounding": True,
            },
        },
        "coverage": sample_doc["coverage"],
        "aggregates": aggregates,
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
    parser.add_argument("--truly-spatial-sample", default="output/golden_qa_sample_truly_spatial.json")
    parser.add_argument("--full-sample", default="output/golden_qa_sample.json")
    parser.add_argument("--sample-output", default="output/golden_qa_grounded_subset_sample.json")
    parser.add_argument("--output", default="output/golden_qa_grounded_subset_results.json")
    parser.add_argument("--episodic-caption-dir", default="data/EgoLife/EgoLifeCap/A1_JAKE")
    parser.add_argument("--semantic-file", default="output/metadata/semantic_memory/A1_JAKE/semantic_consolidation_results_chatgpt-gpt-5.4.json")
    parser.add_argument("--spatial-file", default=DEFAULT_SPATIAL_FILE)
    parser.add_argument("--smoke-spatial-file", default="output/golden_qa_grounded_subset_spatial.json")
    parser.add_argument("--grounding-file", default=DEFAULT_GROUNDING_FILE)
    parser.add_argument("--visual-embeddings-file", default=DEFAULT_VISUAL_EMBEDDINGS)
    parser.add_argument("--visual-clips-file", default=DEFAULT_VISUAL_CLIPS)
    parser.add_argument("--load-episodic", action="store_true", help="Load episodic captions. Off by default to keep smoke under budget.")
    parser.add_argument("--load-semantic", action="store_true", help="Load semantic triples. Off by default to keep smoke under budget.")
    parser.add_argument("--load-visual", action="store_true", help="Load visual clips/embeddings. Off by default to keep smoke under budget.")
    parser.add_argument("--model", default="chatgpt-gpt-5.4")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--worker-payload", default="")
    parser.add_argument("--worker-output", default="")
    args = parser.parse_args()
    if args.worker_payload:
        worker_main(args)
        return

    sample_doc, smoke_spatial_file = assemble_smoke_set(args)
    questions = sorted(sample_doc["questions"], key=lambda q: q["id"])
    by_id = {q["id"]: {**q, "trials": {config: [] for config in CONFIGS}} for q in questions}
    work_dir = Path("output/golden_qa_grounded_subset_workers")
    work_dir.mkdir(parents=True, exist_ok=True)
    for trial in range(1, args.trials + 1):
        for config in CONFIGS:
            chunks = chunked(questions, args.workers)
            print(f"=== trial {trial}/{args.trials} {config} chunks={len(chunks)} ===", flush=True)
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = []
                for worker_index, chunk in enumerate(chunks):
                    payload = {
                        "config": config,
                        "trial": trial,
                        "questions": chunk,
                        "episodic_caption_dir": args.episodic_caption_dir,
                        "semantic_file": args.semantic_file,
                        "spatial_file": smoke_spatial_file,
                        "grounding_file": args.grounding_file,
                        "visual_embeddings_file": args.visual_embeddings_file,
                        "visual_clips_file": args.visual_clips_file,
                        "load_episodic": args.load_episodic,
                        "load_semantic": args.load_semantic,
                        "load_visual": args.load_visual,
                        "model": args.model,
                        "embedding_model": args.embedding_model,
                        "max_rounds": args.max_rounds,
                        "cache_dir": f".cache/golden_grounded_{config}_{trial}_{worker_index}",
                        "cache_tag": f"golden_grounded_{config}_{trial}_{worker_index}",
                    }
                    payload_path = work_dir / f"payload_{config}_{trial}_{worker_index}.json"
                    output_path = work_dir / f"result_{config}_{trial}_{worker_index}.json"
                    write_json(payload_path, payload)
                    futures.append(executor.submit(run_batch, payload_path, output_path))
                completed = 0
                for future in as_completed(futures):
                    for row in future.result():
                        by_id[row["question_id"]]["trials"][config].append(row["result"])
                        completed += 1
                    print(f"[{completed}/{len(questions)}] merged {config} trial {trial}", flush=True)
            per_question = finalize_questions(by_id, list(CONFIGS))
            write_json(args.output, make_results(args, sample_doc, per_question, args.sample_output))
    per_question = finalize_questions(by_id, list(CONFIGS))
    results = make_results(args, sample_doc, per_question, args.sample_output)
    write_json(args.output, results)
    total = results["aggregates"]["spatial_OFF"]["total"]
    sample = results["geometry_display_sample"].replace("\n", " ")[:500]
    print(
        "Combined smoke: "
        f"spatial_OFF {results['aggregates']['spatial_OFF']['correct']}/{total}; "
        f"spatial_ON_text {results['aggregates']['spatial_ON_text']['correct']}/{total} "
        f"(Δ {results['delta_analysis']['spatial_ON_text_minus_spatial_OFF_pp']:+.1f}pp); "
        f"spatial_ON_grounded {results['aggregates']['spatial_ON_grounded']['correct']}/{total} "
        f"(Δ {results['delta_analysis']['spatial_ON_grounded_minus_spatial_OFF_pp']:+.1f}pp; "
        f"grounded-vs-text {results['delta_analysis']['grounded_minus_text_pp']:+.1f}pp). "
        f"Geometry sample: {sample}",
        flush=True,
    )


def finalize_questions(by_id: dict[str, dict[str, Any]], configs: list[str]) -> list[dict[str, Any]]:
    per_question = []
    for item in by_id.values():
        copy_item = dict(item)
        copy_item["trials"] = item["trials"]
        for config in configs:
            copy_item[f"{config}_correct"] = sum(1 for trial in item["trials"][config] if trial["correct"])
            copy_item[f"{config}_axis_selections"] = [trial["axis_selections"] for trial in item["trials"][config]]
        per_question.append(copy_item)
    per_question.sort(key=lambda q: q["id"])
    return per_question


if __name__ == "__main__":
    main()
