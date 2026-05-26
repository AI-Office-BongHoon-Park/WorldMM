#!/usr/bin/env python3
"""Run Phase 4 natural grounded golden QA 3-config ablation."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from eval.three_vs_four_axis_ablation import build_memory, normalize_letter  # noqa: E402
from worldmm.embedding import EmbeddingModel  # type: ignore  # noqa: E402
from worldmm.llm import LLMModel  # type: ignore  # noqa: E402
from worldmm.memory.utils import QAResult  # type: ignore  # noqa: E402

CONFIGS = {
    "spatial_OFF": {"template": "memory_reasoning_3axis", "spatial": False, "grounding": False, "description": "E+S+V, no spatial axis"},
    "spatial_ON_text": {"template": "memory_reasoning", "spatial": True, "grounding": False, "description": "E+S+V+Spatial, text triples only"},
    "spatial_ON_grounded": {"template": "memory_reasoning", "spatial": True, "grounding": True, "description": "E+S+V+Spatial, unified geometry grounding"},
}


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_caption_files(caption_dir: Path) -> dict[str, str]:
    files = {g: str(caption_dir / f"A1_JAKE_{g}.json") for g in ["30sec", "3min", "10min", "1h"]}
    return {g: p for g, p in files.items() if Path(p).exists()}


def to_until_time(query_time: dict[str, Any]) -> int:
    day = str(query_time.get("date", "DAY1")).replace("DAY", "").replace("Day", "") or "1"
    hh, mm, ss = [int(part) for part in str(query_time.get("time", "00:00:00")).split(":")[:3]]
    return int(day) * 100_000_000 + hh * 1_000_000 + mm * 10_000 + ss * 100


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
    query = question["question_en"]
    query = query + f"\nTrial note: answer independently for Phase 4 natural {config} trial {trial}."
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
    rows = [{"question_id": question["id"], "result": run_question(world_memory, question, payload["trial"], payload["config"])} for question in payload["questions"]]
    write_json(args.worker_output, rows)


def chunked(items: list[dict[str, Any]], parts: int) -> list[list[dict[str, Any]]]:
    return [items[index::parts] for index in range(parts) if items[index::parts]]


def read_existing_batch(output_path: Path) -> list[dict[str, Any]]:
    return read_json(output_path)


def run_batch(payload_path: Path, output_path: Path) -> list[dict[str, Any]]:
    command = [sys.executable, str(Path(__file__).resolve()), "--worker-payload", str(payload_path), "--worker-output", str(output_path)]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"worker failed {payload_path}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}")
    return read_json(output_path)


def finalize_questions(by_id: dict[str, dict[str, Any]], configs: list[str]) -> list[dict[str, Any]]:
    rows = []
    for item in by_id.values():
        row = dict(item)
        for config in configs:
            row[f"{config}_correct"] = sum(1 for trial in row["trials"].get(config, []) if trial.get("correct"))
        rows.append(row)
    return sorted(rows, key=lambda item: item["id"])


def aggregate(per_question: list[dict[str, Any]], configs: list[str], trials: int) -> dict[str, Any]:
    total = len(per_question) * trials
    return {config: {"correct": sum(item[f"{config}_correct"] for item in per_question), "total": total, "accuracy": round(sum(item[f"{config}_correct"] for item in per_question) / total, 4) if total else 0.0} for config in configs}


def sample_grounding_text(per_question: list[dict[str, Any]]) -> str:
    for item in per_question:
        for trial in item["trials"].get("spatial_ON_grounded", []):
            for text in trial.get("retrieved_spatial_text", []):
                if any(marker in text for marker in ["_center=", "@ (", "scene_latent=", "[points="]):
                    return text[:1200]
    return ""


def make_results(args: argparse.Namespace, sample_doc: dict[str, Any], per_question: list[dict[str, Any]]) -> dict[str, Any]:
    configs = list(CONFIGS)
    aggregates = aggregate(per_question, configs, args.trials)
    off, text, grounded = aggregates["spatial_OFF"], aggregates["spatial_ON_text"], aggregates["spatial_ON_grounded"]
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
            "sample_file": args.sample,
            "spatial_file": args.spatial_file,
            "grounding_file": args.grounding_file,
            "axis_data_loaded": {"episodic": args.load_episodic, "semantic": args.load_semantic, "visual": args.load_visual, "spatial_text": True, "spatial_grounding": True},
        },
        "coverage": sample_doc.get("coverage", {}),
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
    parser.add_argument("--sample", default="output/golden_qa_phase4_natural.json")
    parser.add_argument("--output", default="output/golden_qa_phase4_natural_results.json")
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
    parser.add_argument("--worker-payload", default="")
    parser.add_argument("--worker-output", default="")
    args = parser.parse_args()
    if args.worker_payload:
        worker_main(args)
        return

    sample_doc = read_json(args.sample)
    questions = sorted(sample_doc["questions"], key=lambda q: q["id"])
    by_id = {q["id"]: {**q, "trials": {config: [] for config in CONFIGS}} for q in questions}
    work_dir = Path("output/golden_qa_phase4_natural_workers")
    work_dir.mkdir(parents=True, exist_ok=True)
    for trial in range(1, args.trials + 1):
        for config in CONFIGS:
            chunks = chunked(questions, args.workers)
            print(f"=== trial {trial}/{args.trials} {config} chunks={len(chunks)} ===", flush=True)
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = []
                for worker_index, chunk in enumerate(chunks):
                    payload = {
                        "config": config, "trial": trial, "questions": chunk,
                        "episodic_caption_dir": args.episodic_caption_dir, "semantic_file": args.semantic_file,
                        "spatial_file": args.spatial_file, "grounding_file": args.grounding_file,
                        "visual_embeddings_file": args.visual_embeddings_file, "visual_clips_file": args.visual_clips_file,
                        "load_episodic": args.load_episodic, "load_semantic": args.load_semantic, "load_visual": args.load_visual,
                        "model": args.model, "embedding_model": args.embedding_model, "max_rounds": args.max_rounds,
                        "cache_dir": f".cache/golden_phase4_natural_{config}_{trial}_{worker_index}",
                        "cache_tag": f"golden_phase4_natural_{config}_{trial}_{worker_index}",
                    }
                    payload_path = work_dir / f"payload_{config}_{trial}_{worker_index}.json"
                    output_path = work_dir / f"result_{config}_{trial}_{worker_index}.json"
                    write_json(payload_path, payload)
                    if output_path.exists():
                        futures.append(executor.submit(read_existing_batch, output_path))
                    else:
                        futures.append(executor.submit(run_batch, payload_path, output_path))
                completed = 0
                for future in as_completed(futures):
                    for row in future.result():
                        by_id[row["question_id"]]["trials"][config].append(row["result"])
                        completed += 1
                    print(f"[{completed}/{len(questions)}] merged {config} trial {trial}", flush=True)
            per_question = finalize_questions(by_id, list(CONFIGS))
            write_json(args.output, make_results(args, sample_doc, per_question))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
