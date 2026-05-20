#!/usr/bin/env python3
"""Run a targeted real-visual 3-axis vs 4-axis EgoLifeQA review."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from eval.three_vs_four_axis_ablation import (  # noqa: E402
    build_memory,
    load_questions,
    normalize_letter,
    to_until_time,
)
from worldmm.embedding import EmbeddingModel  # type: ignore[reportMissingImports]  # noqa: E402
from worldmm.llm import LLMModel  # type: ignore[reportMissingImports]  # noqa: E402
from worldmm.memory import WorldMemory  # type: ignore[reportMissingImports]  # noqa: E402
from worldmm.memory.utils import QAResult  # type: ignore[reportMissingImports]  # noqa: E402

IMAGE_RE = re.compile(r"\[(\d+) images from (\d+) clips\]")

CATALOG_CASES: Dict[str, Dict[str, str]] = {
    "1": {"expected_verdict": "UP", "old_3": "C", "old_4": "B"},
    "33": {"expected_verdict": "UP", "old_3": "D", "old_4": "A"},
    "53": {"expected_verdict": "UP", "old_3": "B", "old_4": "A"},
    "6": {"expected_verdict": "DN", "old_3": "D", "old_4": "A"},
    "45": {"expected_verdict": "DN", "old_3": "A", "old_4": "D"},
    "50": {"expected_verdict": "DN", "old_3": "D", "old_4": "A"},
}


def parse_ids(raw_ids: str) -> List[str]:
    ids = [part.strip() for part in raw_ids.split(",") if part.strip()]
    if not ids:
        raise SystemExit("--ids must include at least one EgoLifeQA ID.")
    if len(set(ids)) != len(ids):
        raise SystemExit("--ids contains duplicate IDs; each requested ID must run once.")
    return ids


def question_by_requested_id(questions: Iterable[Dict[str, Any]], requested_ids: List[str]) -> List[Dict[str, Any]]:
    lookup = {str(q.get("ID")): q for q in questions}
    missing = [qid for qid in requested_ids if qid not in lookup]
    if missing:
        raise SystemExit(f"Requested IDs not found for selected day: {', '.join(missing)}")
    return [lookup[qid] for qid in requested_ids]


def build_available_caption_files(caption_dir: str) -> Dict[str, str]:
    granularities = ["30sec", "3min", "10min", "1h"]
    files = {g: os.path.join(caption_dir, f"A1_JAKE_{g}.json") for g in granularities}
    available = {g: p for g, p in files.items() if os.path.exists(p)}
    if "30sec" not in available:
        raise SystemExit(f"Missing required 30sec captions at {files['30sec']}.")
    return available


def run_question(wm: WorldMemory, question: Dict[str, Any]) -> QAResult:
    choices = {
        "A": question["choice_a"],
        "B": question["choice_b"],
        "C": question["choice_c"],
        "D": question["choice_d"],
    }
    return wm.answer(
        query=question["question"],
        choices=choices,
        until_time=to_until_time(question.get("query_time", {})),
    )


def summarize_rounds(round_history: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]], int, int, int]:
    axis_selections: List[str] = []
    rounds: List[Dict[str, Any]] = []
    visual_selections = 0
    visual_hits = 0
    image_payloads = 0

    for round_info in round_history:
        memory_type = str(round_info.get("memory_type", ""))
        retrieved_content = str(round_info.get("retrieved_content", ""))
        axis_selections.append(memory_type)

        images = 0
        clips = 0
        match = IMAGE_RE.fullmatch(retrieved_content.strip())
        if match:
            images = int(match.group(1))
            clips = int(match.group(2))

        if memory_type == "visual":
            visual_selections += 1
            if images > 0:
                visual_hits += 1
                image_payloads += images

        rounds.append({
            "round_num": round_info.get("round_num"),
            "memory_type": memory_type,
            "search_query": round_info.get("search_query"),
            "retrieved_content_summary": retrieved_content if len(retrieved_content) <= 240 else retrieved_content[:237] + "...",
            "image_payloads": images,
            "clip_payloads": clips,
        })

    return axis_selections, rounds, visual_selections, visual_hits, image_payloads


def verdict(gold: str, three_letter: str, four_letter: str) -> str:
    ok3 = three_letter == gold
    ok4 = four_letter == gold
    if not ok3 and ok4:
        return "UP"
    if ok3 and not ok4:
        return "DN"
    if ok3 and ok4:
        return "same_correct"
    if three_letter == four_letter:
        return "same_wrong"
    return "both_wrong_different"


def catalog_status(qid: str, actual_verdict: str, three_letter: str, four_letter: str) -> str | None:
    old = CATALOG_CASES.get(qid)
    if not old:
        return None
    if actual_verdict != old["expected_verdict"]:
        return "flipped"
    if three_letter == old["old_3"] and four_letter == old["old_4"]:
        return "confirmed"
    return "drifted"


def run_config(label: str, wm: WorldMemory, question: Dict[str, Any]) -> Dict[str, Any]:
    qa_result = run_question(wm, question)
    letter = normalize_letter(qa_result.answer or "")
    axis_selections, rounds, visual_selections, visual_hits, image_payloads = summarize_rounds(qa_result.round_history)
    return {
        "prediction": qa_result.answer or "",
        "letter": letter,
        "correct": letter == question["answer"],
        "axis_selections": axis_selections,
        "visual_selections": visual_selections,
        "visual_hits": visual_hits,
        "image_payloads": image_payloads,
        "rounds": rounds,
        "num_rounds": qa_result.num_rounds,
        "config_label": label,
    }


def load_baseline(path: str) -> Dict[str, Dict[str, Any]]:
    baseline_path = Path(path)
    if not baseline_path.exists():
        return {}
    data = json.loads(baseline_path.read_text(encoding="utf-8"))
    return {str(entry.get("id")): entry for entry in data.get("per_question", [])}


def baseline_letters(entry: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(entry.get("gold", "")),
        str(entry.get("three_axis", {}).get("letter", "")),
        str(entry.get("four_axis", {}).get("letter", "")),
    )


def visual_payload_total(entry: Dict[str, Any]) -> int:
    return int(entry["three_axis"].get("image_payloads", 0)) + int(entry["four_axis"].get("image_payloads", 0))


def attach_baseline(entry: Dict[str, Any], baseline_entry: Optional[Dict[str, Any]]) -> None:
    if not baseline_entry:
        entry["baseline"] = None
        entry["baseline_changed"] = None
        entry["visual_driven_candidate"] = False
        return

    gold, baseline_three, baseline_four = baseline_letters(baseline_entry)
    current_tuple = (entry["gold"], entry["three_axis"]["letter"], entry["four_axis"]["letter"])
    baseline_tuple = (gold, baseline_three, baseline_four)
    changed = current_tuple != baseline_tuple
    entry["baseline"] = {
        "gold": gold,
        "three_axis_letter": baseline_three,
        "four_axis_letter": baseline_four,
    }
    entry["baseline_changed"] = changed
    entry["visual_driven_candidate"] = (
        not entry.get("catalog_status")
        and visual_payload_total(entry) > 0
        and changed
    )


def format_status_summary(entries: List[Dict[str, Any]]) -> str:
    catalog_entries = [e for e in entries if e.get("catalog_status")]
    held = [f"Q{e['id']}" for e in catalog_entries if e["catalog_status"] in {"confirmed", "drifted"}]
    drifted = [f"Q{e['id']}" for e in catalog_entries if e["catalog_status"] == "drifted"]
    flipped = [f"Q{e['id']}" for e in catalog_entries if e["catalog_status"] == "flipped"]
    candidates = [f"Q{e['id']}" for e in entries if e.get("visual_driven_candidate")]
    held_text = ", ".join(held) if held else "none"
    drifted_text = ", ".join(drifted) if drifted else "none"
    flipped_text = ", ".join(flipped) if flipped else "none"
    candidate_text = ", ".join(candidates) if candidates else "none"
    return (
        f"Targeted real-visual review complete: held={held_text}; "
        f"drifted={drifted_text}; flipped={flipped_text}; "
        f"new visual-driven candidates={candidate_text}."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", required=True, help="Comma-separated EgoLifeQA IDs, e.g. 1,6,33")
    parser.add_argument("--episodic-caption-dir", default="data/EgoLife/EgoLifeCap/A1_JAKE")
    parser.add_argument("--semantic-file", default="output/metadata/semantic_memory/A1_JAKE/semantic_consolidation_results_chatgpt-gpt-5.4.json")
    parser.add_argument("--spatial-file", default="output/metadata/spatial_memory/A1_JAKE/spatial_consolidation_results_chatgpt-gpt-5.4.json")
    parser.add_argument("--visual-embeddings-file", default="output/metadata/visual_memory/A1_JAKE/visual_embeddings_clip-ViT-B-32.pkl")
    parser.add_argument("--visual-clips-file", default="output/metadata/visual_memory/A1_JAKE/visual_clips_clip-ViT-B-32.json")
    parser.add_argument("--qa-file", default="data/EgoLife/EgoLifeQA/EgoLifeQA_A1_JAKE.json")
    parser.add_argument("--day", default="DAY1")
    parser.add_argument("--model", default="chatgpt-gpt-5.4")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--output", default="output/targeted_case_review.json")
    parser.add_argument("--baseline-file", default="output/three_vs_four_n30_real_visual_expanded.json")
    args = parser.parse_args()

    requested_ids = parse_ids(args.ids)
    all_questions = load_questions(args.qa_file, args.day, 0)
    questions = sorted(
        question_by_requested_id(all_questions, requested_ids),
        key=lambda question: to_until_time(question.get("query_time", {})),
    )
    available_captions = build_available_caption_files(args.episodic_caption_dir)
    baseline = load_baseline(args.baseline_file)

    print(f"Loaded {len(questions)} requested {args.day} questions: {', '.join(str(q["ID"]) for q in questions)}")
    print(f"Episodic granularities available: {list(available_captions.keys())}")

    embedding_model = EmbeddingModel(text_model_name=args.embedding_model, device="cpu")
    embedding_model.load_model(model_type="text")
    llm_model = LLMModel(model_name=args.model)

    print("\n=== building 3-axis WorldMemory ===")
    wm3 = build_memory(
        reasoning_template_name="memory_reasoning_3axis",
        embedding_model=embedding_model,
        llm_model=llm_model,
        episodic_caption_files=available_captions,
        semantic_file=args.semantic_file,
        spatial_file=args.spatial_file,
        max_rounds=args.max_rounds,
        episodic_cache_tag="3axis_targeted",
        visual_embeddings_file=args.visual_embeddings_file,
        visual_clips_file=args.visual_clips_file,
    )
    print("=== building 4-axis WorldMemory ===")
    wm4 = build_memory(
        reasoning_template_name="memory_reasoning",
        embedding_model=embedding_model,
        llm_model=llm_model,
        episodic_caption_files=available_captions,
        semantic_file=args.semantic_file,
        spatial_file=args.spatial_file,
        max_rounds=args.max_rounds,
        episodic_cache_tag="4axis_targeted",
        visual_embeddings_file=args.visual_embeddings_file,
        visual_clips_file=args.visual_clips_file,
    )

    entries: List[Dict[str, Any]] = []
    for index, question in enumerate(questions, 1):
        qid = str(question["ID"])
        gold = question["answer"]
        print(f"\n[{index:>2}/{len(questions)}] Q{qid} {question.get('type', '?')}: {question['question'][:90]}")
        three_axis = run_config("3-axis", wm3, question)
        four_axis = run_config("4-axis", wm4, question)
        actual_verdict = verdict(gold, three_axis["letter"], four_axis["letter"])
        status = catalog_status(qid, actual_verdict, three_axis["letter"], four_axis["letter"])
        marker = actual_verdict.upper() if actual_verdict in {"UP", "DN"} else "--"
        print(
            f"Q{qid}: gold={gold} 3a={three_axis['letter'] or '?'} "
            f"4a={four_axis['letter'] or '?'} {marker} "
            f"images=3a:{three_axis['image_payloads']} 4a:{four_axis['image_payloads']}"
        )

        entry = {
            "id": qid,
            "type": question.get("type"),
            "question": question["question"],
            "gold": gold,
            "choices": {
                "A": question["choice_a"],
                "B": question["choice_b"],
                "C": question["choice_c"],
                "D": question["choice_d"],
            },
            "three_axis": three_axis,
            "four_axis": four_axis,
            "verdict": actual_verdict,
            "catalog_expected_verdict": CATALOG_CASES.get(qid, {}).get("expected_verdict"),
            "catalog_status": status,
        }
        attach_baseline(entry, baseline.get(qid))
        entries.append(entry)

    payload = {
        "config": {
            "ids": requested_ids,
            "day": args.day,
            "model": args.model,
            "embedding_model": args.embedding_model,
            "max_rounds": args.max_rounds,
            "visual_embeddings_file": args.visual_embeddings_file,
            "visual_clips_file": args.visual_clips_file,
            "baseline_file": args.baseline_file,
            "episodic_granularities": list(available_captions.keys()),
        },
        "per_question": entries,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {output_path}")
    print(format_status_summary(entries))


if __name__ == "__main__":
    main()
