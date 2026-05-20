#!/usr/bin/env python3
"""Measure whether geometric grounding strings change spatial-hero answers."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tools.spatial_hero_harness import normalize_letter, query_time_from_ts, timestamp_label, to_until_time  # noqa: E402
from worldmm.embedding import EmbeddingModel  # type: ignore[reportMissingImports]  # noqa: E402
from worldmm.llm import LLMModel, PromptTemplateManager  # type: ignore[reportMissingImports]  # noqa: E402
from worldmm.memory import WorldMemory  # type: ignore[reportMissingImports]  # noqa: E402
from worldmm.memory.utils import QAResult  # type: ignore[reportMissingImports]  # noqa: E402

CHUNK_ID = 120255900
CHUNK_LABEL = "DAY1 12:02:55"
QUESTION_TIME = {"date": "DAY1", "time": "12:02:55"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def center_of(sidecar: dict[str, Any], triple_id: str, side: str) -> list[float] | None:
    record = sidecar.get(triple_id, {})
    grounding = record.get(f"{side}_grounding")
    if not grounding:
        return None
    return grounding.get("bbox_center")


def dist(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(a, b)))


def make_triple_evidence(triple_id: str, triple: list[str], sidecar: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": triple_id,
        "timestamp": CHUNK_ID,
        "timestamp_label": CHUNK_LABEL,
        "triple": triple,
        "subject_center": center_of(sidecar, triple_id, "subject"),
        "object_center": center_of(sidecar, triple_id, "object"),
    }


def generated_questions(spatial_file: Path, grounding_file: Path) -> list[dict[str, Any]]:
    spatial = read_json(spatial_file)
    sidecar = read_json(grounding_file)
    chunk = spatial[str(CHUNK_ID)]
    triples = chunk["consolidated_spatial_triples"]
    by_id = {triple_id: triples[int(triple_id.rsplit("_", 1)[1])] for triple_id in sidecar}

    def evidence(*ids: str) -> list[dict[str, Any]]:
        return [make_triple_evidence(triple_id, by_id[triple_id], sidecar) for triple_id in ids]

    puzzle_piece = center_of(sidecar, "spatial_120255900_58", "subject")
    plate = center_of(sidecar, "spatial_120255900_58", "object")
    jigsaw = center_of(sidecar, "spatial_120255900_76", "subject")
    table = center_of(sidecar, "spatial_120255900_76", "object")
    kitchen = center_of(sidecar, "spatial_120255900_97", "object")
    water = center_of(sidecar, "spatial_120255900_98", "subject")
    assert puzzle_piece and plate and jigsaw and table and kitchen and water

    closest_pairs = {
        "puzzle_piece and jigsaw puzzle": dist(puzzle_piece, jigsaw),
        "puzzle_piece and plate": dist(puzzle_piece, plate),
        "jigsaw puzzle and table": dist(jigsaw, table),
        "water and kitchen": dist(water, kitchen),
    }
    closest_pair = min(closest_pairs, key=lambda item: closest_pairs[item])
    z_values = {
        "kitchen": kitchen[2],
        "puzzle_piece": puzzle_piece[2],
        "plate": plate[2],
        "table": table[2],
    }
    closest_object = min(z_values, key=lambda item: z_values[item])
    closest_object_gold = {"table": "A", "plate": "B", "kitchen": "C", "puzzle_piece": "D"}[closest_object]
    closest_pair_gold = {
        "puzzle_piece and jigsaw puzzle": "A",
        "puzzle_piece and plate": "B",
        "jigsaw puzzle and table": "C",
        "water and kitchen": "D",
    }[closest_pair]

    return [
        {
            "id": "GH-001",
            "template": "location",
            "question": f"Where was jigsaw puzzle at approximately {CHUNK_LABEL}?",
            "choices": {"A": "plate", "B": "table", "C": "kitchen", "D": "paper_bag"},
            "gold": "B",
            "gold_answer": "table",
            "evidence_triples": evidence("spatial_120255900_76"),
        },
        {
            "id": "GH-002",
            "template": "containment",
            "question": f"Which object was on the plate at approximately {CHUNK_LABEL}?",
            "choices": {"A": "puzzle_piece", "B": "water", "C": "trash bag", "D": "table"},
            "gold": "A",
            "gold_answer": "puzzle_piece",
            "evidence_triples": evidence("spatial_120255900_58"),
        },
        {
            "id": "GH-003",
            "template": "depth_ordering",
            "question": f"Using the relative 3D centers at {CHUNK_LABEL}, was puzzle_piece in front of or behind plate? Treat smaller z as closer to the camera/in front.",
            "choices": {"A": "puzzle_piece was in front of plate", "B": "puzzle_piece was behind plate", "C": "they had the same depth", "D": "the grounded triples do not mention plate"},
            "gold": "A",
            "gold_answer": "puzzle_piece was in front of plate",
            "evidence_triples": evidence("spatial_120255900_58"),
        },
        {
            "id": "GH-004",
            "template": "depth_min",
            "question": f"Using the relative 3D centers at {CHUNK_LABEL}, which grounded object had the smallest z value and was closest to the camera?",
            "choices": {"A": "table", "B": "plate", "C": "kitchen", "D": "puzzle_piece"},
            "gold": closest_object_gold,
            "gold_answer": closest_object,
            "evidence_triples": evidence("spatial_120255900_58", "spatial_120255900_76", "spatial_120255900_97"),
        },
        {
            "id": "GH-005",
            "template": "distance_pair",
            "question": f"Using the relative 3D centers at {CHUNK_LABEL}, which pair of grounded objects was closest to each other?",
            "choices": {"A": "puzzle_piece and jigsaw puzzle", "B": "puzzle_piece and plate", "C": "jigsaw puzzle and table", "D": "water and kitchen"},
            "gold": closest_pair_gold,
            "gold_answer": closest_pair,
            "evidence_triples": evidence("spatial_120255900_58", "spatial_120255900_76", "spatial_120255900_97", "spatial_120255900_98"),
        },
        {
            "id": "GH-006",
            "template": "depth_max",
            "question": f"Using the relative 3D centers at {CHUNK_LABEL}, which table item had a larger z value than jigsaw puzzle?",
            "choices": {"A": "water", "B": "plate", "C": "kitchen", "D": "none of these"},
            "gold": "A",
            "gold_answer": "water",
            "evidence_triples": evidence("spatial_120255900_76", "spatial_120255900_98"),
        },
    ]


def build_memory(args: argparse.Namespace, embedding_model: EmbeddingModel, label: str) -> WorldMemory:
    llm = LLMModel(model_name=args.model, cache_dir=f".cache/spatial_hero_grounded_{label}")
    wm = WorldMemory(
        embedding_model=embedding_model,
        retriever_llm_model=llm,
        respond_llm_model=llm,
        prompt_template_manager=PromptTemplateManager(),
        episodic_granularities=[],
        episodic_cache_root=f".cache/episodic_memory_grounded_{label}",
        qa_template_name="qa_egolife",
        reasoning_template_name=args.reasoning_template,
        max_rounds=args.max_rounds,
        max_errors=3,
    )
    if args.semantic_file and Path(args.semantic_file).exists():
        wm.load_semantic_triples(file_path=args.semantic_file)
    if label == "grounded":
        wm.spatial_memory.load_triples_from_file(args.spatial_file, grounding_file=args.grounding_file)
    else:
        wm.spatial_memory.load_triples_from_file(args.spatial_file)
    wm.set_retrieval_top_k(spatial=args.spatial_top_k, semantic=args.semantic_top_k)
    return wm


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
            grounding_seen = grounding_seen or "_center=(" in content or "@ (" in content
        compact = re.sub(r"\s+", " ", content).strip()
        if len(compact) > 240:
            compact = compact[:237] + "..."
        summaries.append(f"R{round_info.get('round_num')} {memory_type}: {compact}")
    return axes, spatial_texts, " | ".join(summaries)[:700], grounding_seen


def run_question(wm: WorldMemory, question: dict[str, Any], trial: int, mode: str) -> dict[str, Any]:
    query = (
        question["question"]
        + f"\nAblation mode: {mode}. Use retrieved memory only; if relative 3D centers are present, compare their numeric (x,y,z) values exactly."
    )
    result: QAResult = wm.answer(query=query, choices=question["choices"], until_time=CHUNK_ID)
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


def finalize(per_question: list[dict[str, Any]], trials: int) -> dict[str, Any]:
    mode_a_total = 0
    mode_b_total = 0
    flips_to_grounded = 0
    flips_to_plain = 0
    grounding_retrieved = 0
    for entry in per_question:
        plain_correct = sum(1 for trial in entry["trials"]["plain"] if trial["correct"])
        grounded_correct = sum(1 for trial in entry["trials"]["grounded"] if trial["correct"])
        entry["plain_correct"] = plain_correct
        entry["grounded_correct"] = grounded_correct
        entry["verdict"] = "grounding_better" if grounded_correct > plain_correct else "plain_better" if plain_correct > grounded_correct else "tie"
        mode_a_total += plain_correct
        mode_b_total += grounded_correct
        if grounded_correct > plain_correct:
            flips_to_grounded += 1
        if plain_correct > grounded_correct:
            flips_to_plain += 1
        if any(trial["retrieved_grounding_string"] for trial in entry["trials"]["grounded"]):
            grounding_retrieved += 1
    return {
        "questions_completed": len(per_question),
        "trials_per_mode": trials,
        "plain_correct_total": mode_a_total,
        "grounded_correct_total": mode_b_total,
        "total_trials_per_mode": len(per_question) * trials,
        "questions_grounding_better": flips_to_grounded,
        "questions_plain_better": flips_to_plain,
        "questions_with_grounding_retrieved": grounding_retrieved,
    }


def make_results(args: argparse.Namespace, questions: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol": {
            "subject": "A1_JAKE",
            "day": "DAY1",
            "chunk_id": CHUNK_ID,
            "chunk_label": CHUNK_LABEL,
            "trials_per_mode": args.trials,
            "modes": {
                "plain": "SpatialMemory triples only; no grounding_file argument",
                "grounded": "SpatialMemory triples plus grounding_file; to_display_str renders relative 3D centers inline",
            },
            "model": args.model,
            "embedding_model": args.embedding_model,
            "embedding_device": "cpu",
            "reasoning_template": args.reasoning_template,
            "spatial_top_k": args.spatial_top_k,
            "semantic_top_k": args.semantic_top_k,
            "semantic_file_loaded": bool(args.semantic_file and Path(args.semantic_file).exists()),
            "spatial_file": args.spatial_file,
            "grounding_file": args.grounding_file,
            "question_source": "generated from real triples in grounding sidecar because output/spatial_hero_curated.json had no exact 120255900 / DAY1 12:02:55 case",
        },
        "summary": summary,
        "per_question": questions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spatial-file", default="output/metadata/spatial_memory/A1_JAKE/spatial_consolidation_results_chatgpt-gpt-5.4.json")
    parser.add_argument("--grounding-file", default="output/metadata/spatial_memory/A1_JAKE/grounding/120255900.json")
    parser.add_argument("--semantic-file", default="output/metadata/semantic_memory/A1_JAKE/semantic_consolidation_results_chatgpt-gpt-5.4.json")
    parser.add_argument("--output", default="output/spatial_hero_grounded_results.json")
    parser.add_argument("--model", default="chatgpt-gpt-5.4")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--reasoning-template", default="memory_reasoning_essp")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--spatial-top-k", type=int, default=25)
    parser.add_argument("--semantic-top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = generated_questions(Path(args.spatial_file), Path(args.grounding_file))
    for question in questions:
        question.update({"evidence_chunk": CHUNK_ID, "evidence_chunk_label": CHUNK_LABEL, "query_time": QUESTION_TIME, "trials": {"plain": [], "grounded": []}})

    embedding_model = EmbeddingModel(text_model_name=args.embedding_model, device="cpu")
    embedding_model.load_model(model_type="text")

    for trial in range(1, args.trials + 1):
        for mode in ["plain", "grounded"]:
            print(f"=== trial {trial}/{args.trials} mode={mode} ===", flush=True)
            wm = build_memory(args, embedding_model, mode)
            for idx, question in enumerate(questions, start=1):
                print(f"[{idx}/{len(questions)}] {question['id']} {question['question']}", flush=True)
                question["trials"][mode].append(run_question(wm, question, trial, mode))
            summary = finalize(questions, args.trials)
            write_json(Path(args.output), make_results(args, questions, summary))

    summary = finalize(questions, args.trials)
    results = make_results(args, questions, summary)
    write_json(Path(args.output), results)
    total = summary["total_trials_per_mode"]
    print(
        f"Grounded ablation: plain {summary['plain_correct_total']}/{total}, "
        f"grounded {summary['grounded_correct_total']}/{total}; "
        f"question flips to grounded {summary['questions_grounding_better']}, flips to plain {summary['questions_plain_better']}; "
        f"grounding string retrieved in {summary['questions_with_grounding_retrieved']}/{summary['questions_completed']} questions. "
        "Inspect output/spatial_hero_grounded_results.json for per-trial axes and retrieved spatial text."
    )


if __name__ == "__main__":
    main()
