#!/usr/bin/env python3
"""Re-run curated spatial-hero V2 cases with visual axis enabled."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from eval.three_vs_four_axis_ablation import build_memory, normalize_letter  # noqa: E402
from tools import spatial_hero_harness as harness  # noqa: E402

IMAGE_RE = re.compile(r"\[(\d+) images from (\d+) clips\]")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def to_until_time(evidence_chunk: int | str) -> int:
    return int(str(evidence_chunk))


def summarize_rounds(world_memory: Any, round_history: list[dict[str, Any]]) -> dict[str, Any]:
    axes: list[str] = []
    rounds: list[dict[str, Any]] = []
    spatial_chain_quotes: list[str] = []
    visual_selections = 0
    visual_hits = 0
    image_payload_count = 0
    clip_payload_count = 0

    for round_info in round_history:
        memory_type = str(round_info.get("memory_type", ""))
        content = str(round_info.get("retrieved_content", ""))
        axes.append(memory_type)

        images = 0
        clips = 0
        match = IMAGE_RE.fullmatch(content.strip())
        if match:
            images = int(match.group(1))
            clips = int(match.group(2))

        if memory_type == "visual":
            visual_selections += 1
            if clips == 0:
                try:
                    clip_results = world_memory.visual_memory.retrieve(str(round_info.get("search_query", "")), top_k=world_memory.visual_top_k, as_context=False)
                    clips = len(clip_results) if clip_results else 0
                except Exception:
                    clips = 0
            if clips > 0:
                visual_hits += 1
                clip_payload_count += clips
            if images > 0:
                image_payload_count += images
        elif memory_type == "spatial" and content and content != "[No results]":
            spatial_chain_quotes.append(content)

        rounds.append({
            "round_num": round_info.get("round_num"),
            "memory_type": memory_type,
            "search_query": round_info.get("search_query"),
            "retrieved_content_summary": content if len(content) <= 500 else content[:497] + "...",
            "image_payload_count": images,
            "clip_payload_count": clips,
        })

    return {
        "axis_selections": axes,
        "rounds": rounds,
        "visual_selections": visual_selections,
        "visual_hit_count": visual_hits,
        "image_payload_count": image_payload_count,
        "clip_payload_count": clip_payload_count,
        "spatial_chain_quoted": spatial_chain_quotes,
    }


def run_question(world_memory: Any, case: dict[str, Any], trial: int, config: str) -> dict[str, Any]:
    query = case["question"] + f"\nTrial note: answer independently for spatial hero visual-ON {config} trial {trial}."
    result = world_memory.answer(
        query=query,
        choices=case["choices"],
        until_time=to_until_time(case["evidence_chunk"]),
    )
    letter = normalize_letter(result.answer or "")
    trace = summarize_rounds(world_memory, result.round_history)
    return {
        "trial": trial,
        "prediction": result.answer or "",
        "letter": letter,
        "correct": letter == case["gold"],
        "num_rounds": result.num_rounds,
        **trace,
    }


def classify_entry(entry: dict[str, Any], trials: int) -> str:
    three = entry.get("three_axis_correct", 0)
    four = entry.get("four_axis_correct", 0)
    if four >= 2 and three <= 1:
        return "candidate_winner"
    if three >= 2 and four <= 1:
        return "candidate_loser"
    if four > three:
        return "spatial_better"
    if three > four:
        return "baseline_better"
    return "tie"


def finalise_entries(entries: list[dict[str, Any]], trials: int) -> list[dict[str, Any]]:
    finalised: list[dict[str, Any]] = []
    for entry in entries:
        copy_entry = dict(entry)
        copy_entry["trials"] = entry["trials"]
        copy_entry["three_axis_correct"] = sum(1 for trial in entry["trials"]["three_axis"] if trial["correct"])
        copy_entry["four_axis_correct"] = sum(1 for trial in entry["trials"]["four_axis"] if trial["correct"])
        copy_entry["verdict"] = classify_entry(copy_entry, trials)
        copy_entry["verdict_across_trials"] = {
            "three_axis": f"{copy_entry['three_axis_correct']}/{trials}",
            "four_axis": f"{copy_entry['four_axis_correct']}/{trials}",
        }
        copy_entry["visual_totals"] = {
            label: {
                "visual_selections": sum(t["visual_selections"] for t in entry["trials"][label]),
                "visual_hit_count": sum(t["visual_hit_count"] for t in entry["trials"][label]),
                "image_payload_count": sum(t["image_payload_count"] for t in entry["trials"][label]),
                "clip_payload_count": sum(t["clip_payload_count"] for t in entry["trials"][label]),
            }
            for label in ("three_axis", "four_axis")
        }
        finalised.append(copy_entry)
    return finalised


def make_results(per_question: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    winners = sum(1 for entry in per_question if entry.get("verdict") == "candidate_winner")
    losers = sum(1 for entry in per_question if entry.get("verdict") == "candidate_loser")
    return {
        "protocol": {
            "subject": "A1_JAKE",
            "day": "DAY1",
            "trials_per_config": args.trials,
            "cases": len(per_question),
            "configs": {
                "three_axis": "episodic+semantic+visual, memory_reasoning_3axis, spatial unavailable to reasoner",
                "four_axis": "episodic+semantic+visual+spatial, memory_reasoning, all four axes available",
            },
            "model": args.model,
            "embedding_model": args.embedding_model,
            "visual_query_embedder": "ClipQueryEmbedder(sentence-transformers/clip-ViT-B-32)",
            "visual_embeddings_file": args.visual_embeddings_file,
            "visual_clips_file": args.visual_clips_file,
            "visual_axis": "enabled in both configs",
        },
        "summary": {
            "questions_completed": len(per_question),
            "candidate_winners": winners,
            "candidate_losers": losers,
            "three_axis_correct_total": sum(entry["three_axis_correct"] for entry in per_question),
            "four_axis_correct_total": sum(entry["four_axis_correct"] for entry in per_question),
            "visual_hit_questions": sum(
                1
                for entry in per_question
                if entry["visual_totals"]["three_axis"]["visual_hit_count"]
                or entry["visual_totals"]["four_axis"]["visual_hit_count"]
            ),
            "image_payload_total": sum(
                entry["visual_totals"][label]["image_payload_count"]
                for entry in per_question
                for label in ("three_axis", "four_axis")
            ),
        },
        "per_question": per_question,
    }


def evaluate(args: argparse.Namespace, cases: list[dict[str, Any]]) -> dict[str, Any]:
    caption_files = harness.build_caption_files(Path(args.episodic_caption_dir))
    embedding_model = harness.EmbeddingModel(text_model_name=args.embedding_model, device="cpu")
    embedding_model.load_model(model_type="text")

    by_id: dict[str, dict[str, Any]] = {}
    for case in cases:
        by_id[case["id"]] = {
            "case_num": case.get("case_num"),
            "id": case["id"],
            "template": case["template"],
            "question": case["question"],
            "choices": case["choices"],
            "gold": case["gold"],
            "gold_answer": case["gold_answer"],
            "evidence_chunk": case["evidence_chunk"],
            "evidence_chunk_label": case["evidence_chunk_label"],
            "spatial_chain": case.get("spatial_chain", []),
            "evidence_axis": "spatial",
            "trials": {"three_axis": [], "four_axis": []},
        }

    ordered_cases = sorted(cases, key=lambda case: (int(case.get("evidence_chunk", 0)), case["id"]))
    configs = [
        ("three_axis", "memory_reasoning_3axis"),
        ("four_axis", "memory_reasoning"),
    ]
    for trial in range(1, args.trials + 1):
        for label, template in configs:
            print(f"=== visual-ON trial {trial}/{args.trials} {label} ===", flush=True)
            llm = harness.LLMModel(model_name=args.model, cache_dir=f".cache/spatial_hero_visual_on_{label}_trial_{trial}")
            world_memory = build_memory(
                reasoning_template_name=template,
                embedding_model=embedding_model,
                llm_model=llm,
                episodic_caption_files=caption_files,
                semantic_file=args.semantic_file,
                spatial_file=args.spatial_file,
                max_rounds=args.max_rounds,
                episodic_cache_tag=f"spatial_hero_visual_on_{label}_{trial}",
                visual_embeddings_file=args.visual_embeddings_file,
                visual_clips_file=args.visual_clips_file,
            )
            if not world_memory.visual_memory.clips:
                raise SystemExit("Visual axis wiring failed: no visual clips loaded.")
            for idx, case in enumerate(ordered_cases, start=1):
                print(f"[{idx}/{len(ordered_cases)}] {case['id']} {case['template']} {case['question']}", flush=True)
                by_id[case["id"]]["trials"][label].append(run_question(world_memory, case, trial, label))
            partial = make_results(finalise_entries([by_id[case["id"]] for case in cases], args.trials), args)
            write_json(Path(args.results_file), partial)

    return make_results(finalise_entries([by_id[case["id"]] for case in cases], args.trials), args)


def load_visual_off(topk_path: Path, fallback_path: Path, case_ids: set[str]) -> dict[str, dict[str, Any]]:
    topk_data = read_json(topk_path)
    fallback_data = read_json(fallback_path)
    topk = {entry["id"]: dict(entry, visual_off_source=str(topk_path)) for entry in topk_data.get("per_question", [])}
    fallback = {entry["id"]: dict(entry, visual_off_source=str(fallback_path)) for entry in fallback_data.get("per_question", [])}
    merged: dict[str, dict[str, Any]] = {}
    for case_id in case_ids:
        if case_id in topk:
            merged[case_id] = topk[case_id]
        elif case_id in fallback:
            merged[case_id] = fallback[case_id]
    return merged


def format_summary(results: dict[str, Any], visual_off: dict[str, dict[str, Any]], trials: int) -> str:
    case_ids = [entry["id"] for entry in results["per_question"]]
    off_three = sum(visual_off[qid]["three_axis_correct"] for qid in case_ids)
    off_four = sum(visual_off[qid]["four_axis_correct"] for qid in case_ids)
    on_three = results["summary"]["three_axis_correct_total"]
    on_four = results["summary"]["four_axis_correct_total"]
    total = len(case_ids) * trials
    off_gap = off_four - off_three
    on_gap = on_four - on_three
    visual_cases = [
        entry["id"]
        for entry in results["per_question"]
        if entry["visual_totals"]["three_axis"]["visual_hit_count"]
        or entry["visual_totals"]["four_axis"]["visual_hit_count"]
    ]
    flips = []
    for entry in results["per_question"]:
        off_entry = visual_off[entry["id"]]
        if (off_entry["three_axis_correct"], off_entry["four_axis_correct"]) != (entry["three_axis_correct"], entry["four_axis_correct"]):
            has_visual = entry["id"] in visual_cases
            flips.append(f"{entry['id']}{'*' if has_visual else ''}")
    return (
        f"Visual-ON recheck summary: visual-OFF curated aggregate was 3-axis {off_three}/{total}, "
        f"4-axis {off_four}/{total} (spatial gap +{off_gap}); visual-ON is 3-axis {on_three}/{total}, "
        f"4-axis {on_four}/{total} (spatial gap +{on_gap}, delta {on_gap - off_gap:+d}). "
        f"Visual hits appeared in {len(visual_cases)}/{len(case_ids)} cases ({', '.join(visual_cases) or 'none'}); "
        f"changed aggregate case scores: {', '.join(flips) or 'none'} (* means changed case also had non-zero visual clip hits)."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curated-file", default="output/spatial_hero_curated.json")
    parser.add_argument("--visual-off-results-file", default="output/spatial_hero_topk_verified.json")
    parser.add_argument("--visual-off-fallback-results-file", default="output/spatial_hero_results.json")
    parser.add_argument("--results-file", default="output/spatial_hero_visual_on_results.json")
    parser.add_argument("--spatial-file", default="output/metadata/spatial_memory/A1_JAKE/spatial_consolidation_results_chatgpt-gpt-5.4.json")
    parser.add_argument("--semantic-file", default="output/metadata/semantic_memory/A1_JAKE/semantic_consolidation_results_chatgpt-gpt-5.4.json")
    parser.add_argument("--episodic-caption-dir", default="data/EgoLife/EgoLifeCap/A1_JAKE")
    parser.add_argument("--visual-embeddings-file", default="output/metadata/visual_memory/A1_JAKE/visual_embeddings_clip-ViT-B-32.pkl")
    parser.add_argument("--visual-clips-file", default="output/metadata/visual_memory/A1_JAKE/visual_clips_clip-ViT-B-32.json")
    parser.add_argument("--model", default="chatgpt-gpt-5.4")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument("--skip-run", action="store_true", help="Only print summary from an existing visual-ON results file.")
    args = parser.parse_args()

    cases = read_json(Path(args.curated_file))
    if not isinstance(cases, list):
        raise SystemExit("Curated file must be the existing list-shaped output/spatial_hero_curated.json.")
    if len(cases) != 10:
        raise SystemExit(f"Expected 10 curated cases, found {len(cases)}.")

    case_ids = {case["id"] for case in cases}
    visual_off = load_visual_off(Path(args.visual_off_results_file), Path(args.visual_off_fallback_results_file), case_ids)
    missing = sorted(case_ids - set(visual_off))
    if missing:
        raise SystemExit(f"Curated IDs missing from shipped visual-OFF results: {missing}")
    fallback_used = sorted(qid for qid, entry in visual_off.items() if entry.get("visual_off_source") == args.visual_off_fallback_results_file)
    if fallback_used:
        print(f"Visual-OFF note: {len(fallback_used)} curated IDs absent from topk_verified; using shipped spatial_hero_results for {fallback_used} without re-running visual-OFF.", flush=True)

    print(f"Visual-ON curated run: {len(cases)} cases, templates {dict(Counter(case['template'] for case in cases))}", flush=True)
    if args.skip_run:
        results = read_json(Path(args.results_file))
    else:
        results = evaluate(args, cases)
        write_json(Path(args.results_file), results)

    total_visual_hits = sum(
        entry["visual_totals"][label]["visual_hit_count"]
        for entry in results["per_question"]
        for label in ("three_axis", "four_axis")
    )
    if total_visual_hits <= 0:
        raise SystemExit("Visual axis wiring check failed: all visual hit counts are zero.")

    print(format_summary(results, visual_off, args.trials), flush=True)


if __name__ == "__main__":
    main()
