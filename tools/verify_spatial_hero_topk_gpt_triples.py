#!/usr/bin/env python3
"""Re-run curated spatial-hero cases with GPT structured visual triples."""

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

from eval.three_vs_four_axis_ablation import normalize_letter  # noqa: E402
from tools import spatial_hero_harness as harness  # noqa: E402
from tools.visual_triple_query_embedder import VisualTripleQueryEmbedder  # noqa: E402
from tools.visual_triples_memory import VisualTriplesMemory  # noqa: E402
from worldmm.llm import PromptTemplateManager  # type: ignore  # noqa: E402
from worldmm.memory import WorldMemory  # type: ignore  # noqa: E402

TRIPLE_RE = re.compile(r"\([^\n()]+,\s*[^\n()]+,\s*[^\n()]+\)")
FRAME_RE = re.compile(r"t=([0-9]+(?:\.[0-9]+)?)s")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def to_until_time(evidence_chunk: int | str) -> int:
    return int(str(evidence_chunk))


def build_memory(
    *,
    reasoning_template_name: str,
    embedding_model: Any,
    llm_model: Any,
    episodic_caption_files: dict[str, str],
    semantic_file: str,
    spatial_file: str,
    max_rounds: int,
    episodic_cache_tag: str,
    visual_index_file: str,
    visual_embed_model: str,
    device: str,
) -> WorldMemory:
    world_memory = WorldMemory(
        embedding_model=embedding_model,
        retriever_llm_model=llm_model,
        respond_llm_model=llm_model,
        prompt_template_manager=PromptTemplateManager(),
        episodic_granularities=list(episodic_caption_files.keys()),
        episodic_cache_root=f".cache/episodic_memory_{episodic_cache_tag}",
        qa_template_name="qa_egolife",
        reasoning_template_name=reasoning_template_name,
        max_rounds=max_rounds,
        max_errors=3,
    )
    world_memory.load_episodic_captions(caption_files=episodic_caption_files)
    if Path(semantic_file).exists():
        world_memory.load_semantic_triples(file_path=semantic_file)
    if Path(spatial_file).exists():
        world_memory.load_spatial_triples(file_path=spatial_file)
    embedder = VisualTripleQueryEmbedder(visual_embed_model, device=device)
    world_memory.visual_memory = VisualTriplesMemory(visual_index_file, embedder=embedder)  # type: ignore[assignment]
    return world_memory


def clean_summary(text: str, limit: int = 360) -> str:
    compact = re.sub(r"\s+", " ", text or "").strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def summarize_rounds(world_memory: Any, round_history: list[dict[str, Any]]) -> dict[str, Any]:
    axes: list[str] = []
    rounds: list[dict[str, Any]] = []
    spatial_chain_quotes: list[str] = []
    visual_selections = 0
    visual_hit_count = 0
    visual_triple_payload_count = 0
    f_anchored_hits = 0
    first_visual_triple_text: str | None = None
    frame_timestamps: set[float] = set()

    for round_info in round_history:
        memory_type = str(round_info.get("memory_type", ""))
        search_query = str(round_info.get("search_query", ""))
        content = str(round_info.get("retrieved_content", ""))
        axes.append(memory_type)
        triples = TRIPLE_RE.findall(content)
        frame_values = [float(value) for value in FRAME_RE.findall(content)]
        retrieved_hits = []

        if memory_type == "visual":
            visual_selections += 1
            try:
                retrieved_hits = world_memory.visual_memory.retrieve(search_query, top_k=world_memory.visual_top_k, as_context=False)
            except Exception:
                retrieved_hits = []
            f_anchored_hits += sum(1 for hit in retrieved_hits if getattr(hit, "f_provenance", "") == "gpt_anchored")
            if not triples:
                try:
                    hits = retrieved_hits
                    triples = [hit.triple_text for hit in hits]
                    frame_values = [float(hit.frame_timestamp_s) for hit in hits]
                except Exception:
                    triples = []
                    frame_values = []
            if triples:
                visual_hit_count += 1
                visual_triple_payload_count += len(triples)
                if first_visual_triple_text is None:
                    first_visual_triple_text = triples[0]
            frame_timestamps.update(frame_values)
        elif memory_type == "spatial" and content and content != "[No results]":
            spatial_chain_quotes.append(content)

        rounds.append({
            "round_num": round_info.get("round_num"),
            "memory_type": memory_type,
            "search_query": round_info.get("search_query"),
            "retrieved_content_summary": clean_summary(content),
            "visual_triples_retrieved": len(triples) if memory_type == "visual" else 0,
            "first_visual_triple_text": triples[0] if memory_type == "visual" and triples else None,
            "frame_timestamp_s_set": sorted(set(frame_values)) if memory_type == "visual" else [],
            "f_anchored_hits": sum(1 for hit in retrieved_hits if getattr(hit, "f_provenance", "") == "gpt_anchored") if memory_type == "visual" else 0,
        })

    return {
        "axis_selections": axes,
        "rounds": rounds,
        "visual_selections": visual_selections,
        "visual_hit_count": visual_hit_count,
        "visual_triple_payload_count": visual_triple_payload_count,
        "f_anchored_hits": f_anchored_hits,
        "first_visual_triple_text": first_visual_triple_text,
        "frame_timestamp_s_set": sorted(frame_timestamps),
        "spatial_chain_quoted": spatial_chain_quotes,
    }


def run_question(world_memory: Any, case: dict[str, Any], trial: int, config: str) -> dict[str, Any]:
    query = case["question"] + f"\nTrial note: answer independently for spatial hero GPT-triples visual {config} trial {trial}."
    result = world_memory.answer(query=query, choices=case["choices"], until_time=to_until_time(case["evidence_chunk"]))
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
    finalised = []
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
                "visual_triple_payload_count": sum(t["visual_triple_payload_count"] for t in entry["trials"][label]),
                "f_anchored_hits": sum(t.get("f_anchored_hits", 0) for t in entry["trials"][label]),
                "first_visual_triple_text": next((t["first_visual_triple_text"] for t in entry["trials"][label] if t["first_visual_triple_text"]), None),
                "frame_timestamp_s_set": sorted({ts for t in entry["trials"][label] for ts in t["frame_timestamp_s_set"]}),
            }
            for label in ("three_axis", "four_axis")
        }
        finalised.append(copy_entry)
    return finalised


def make_results(per_question: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "protocol": {
            "subject": "A1_JAKE",
            "day": "DAY1",
            "trials_per_config": args.trials,
            "cases": len(per_question),
            "configs": {
                "three_axis": "episodic+semantic+GPT structured triples visual, memory_reasoning_3axis, spatial unavailable",
                "four_axis": "episodic+semantic+GPT structured triples visual+spatial, memory_reasoning, all four axes available",
            },
            "model": args.model,
            "embedding_model": args.embedding_model,
            "visual_query_embedder": f"VisualTripleQueryEmbedder({args.visual_embed_model})",
            "visual_index_file": args.visual_index_file,
            "visual_axis": "GPT structured triples retrieved as text; thumbnails are not passed to reasoner",
            "frame_anchor_provenance": "full: all indexed triples use gpt_anchored",
        },
        "summary": {
            "questions_completed": len(per_question),
            "three_axis_correct_total": sum(entry["three_axis_correct"] for entry in per_question),
            "four_axis_correct_total": sum(entry["four_axis_correct"] for entry in per_question),
            "candidate_winners": sum(1 for entry in per_question if entry["verdict"] == "candidate_winner"),
            "candidate_losers": sum(1 for entry in per_question if entry["verdict"] == "candidate_loser"),
            "visual_hit_cases": sum(1 for entry in per_question if entry["visual_totals"]["three_axis"]["visual_hit_count"] or entry["visual_totals"]["four_axis"]["visual_hit_count"]),
            "visual_triple_payload_total": sum(entry["visual_totals"][label]["visual_triple_payload_count"] for entry in per_question for label in ("three_axis", "four_axis")),
            "f_anchored_hits_total": sum(entry["visual_totals"][label].get("f_anchored_hits", 0) for entry in per_question for label in ("three_axis", "four_axis")),
        },
        "per_question": per_question,
    }


def evaluate(args: argparse.Namespace, cases: list[dict[str, Any]]) -> dict[str, Any]:
    caption_files = harness.build_caption_files(Path(args.episodic_caption_dir))
    embedding_model = harness.EmbeddingModel(text_model_name=args.embedding_model, device=args.device)
    embedding_model.load_model(model_type="text")

    by_id = {}
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
    configs = [("three_axis", "memory_reasoning_3axis"), ("four_axis", "memory_reasoning")]
    for trial in range(1, args.trials + 1):
        for label, template in configs:
            print(f"=== GPT-triples visual trial {trial}/{args.trials} {label} ===", flush=True)
            llm = harness.LLMModel(model_name=args.model, cache_dir=f".cache/spatial_hero_gpt_triples_{label}_trial_{trial}")
            world_memory = build_memory(
                reasoning_template_name=template,
                embedding_model=embedding_model,
                llm_model=llm,
                episodic_caption_files=caption_files,
                semantic_file=args.semantic_file,
                spatial_file=args.spatial_file,
                max_rounds=args.max_rounds,
                episodic_cache_tag=f"spatial_hero_gpt_triples_{label}_{trial}",
                visual_index_file=args.visual_index_file,
                visual_embed_model=args.visual_embed_model,
                device=args.device,
            )
            if not world_memory.visual_memory.clips:
                raise RuntimeError("GPT structured visual triples did not load")
            for index, case in enumerate(ordered_cases, start=1):
                print(f"[{index}/{len(ordered_cases)}] {case['id']} {case['template']} {case['question']}", flush=True)
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
        entry = topk.get(case_id) or fallback.get(case_id)
        if entry is not None:
            merged[case_id] = entry
    return merged


def format_summary(results: dict[str, Any], visual_off: dict[str, dict[str, Any]], trials: int) -> str:
    case_ids = [entry["id"] for entry in results["per_question"]]
    off_three = sum(visual_off[qid]["three_axis_correct"] for qid in case_ids)
    off_four = sum(visual_off[qid]["four_axis_correct"] for qid in case_ids)
    on_three = results["summary"]["three_axis_correct_total"]
    on_four = results["summary"]["four_axis_correct_total"]
    total = len(case_ids) * trials
    return (
        f"GPT-triples visual summary: visual-OFF was 3-axis {off_three}/{total}, 4-axis {off_four}/{total}; "
        f"GPT-triples visual is 3-axis {on_three}/{total}, 4-axis {on_four}/{total}. "
        f"Visual triple text payloads={results['summary']['visual_triple_payload_total']} across "
        f"{results['summary']['visual_hit_cases']}/{len(case_ids)} cases; "
        f"f_anchored_hits={results['summary'].get('f_anchored_hits_total', 0)}."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curated-file", default="output/spatial_hero_curated.json")
    parser.add_argument("--visual-off-results-file", default="output/spatial_hero_topk_verified.json")
    parser.add_argument("--visual-off-fallback-results-file", default="output/spatial_hero_results.json")
    parser.add_argument("--results-file", default="output/spatial_hero_gpt_triples_results.json")
    parser.add_argument("--episodic-caption-dir", default="data/EgoLife/EgoLifeCap/A1_JAKE")
    parser.add_argument("--semantic-file", default="output/metadata/semantic_memory/A1_JAKE/semantic_consolidation_results_chatgpt-gpt-5.4.json")
    parser.add_argument("--spatial-file", default="output/metadata/spatial_memory/A1_JAKE/spatial_consolidation_results_chatgpt-gpt-5.4.json")
    parser.add_argument("--visual-index-file", default="output/metadata/visual_memory/A1_JAKE/visual_triples_gpt_index.pkl")
    parser.add_argument("--model", default="chatgpt-gpt-5.4")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--visual-embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    cases = read_json(Path(args.curated_file))
    if not isinstance(cases, list) or len(cases) != 10:
        raise SystemExit(f"Expected 10 curated cases, found {len(cases) if isinstance(cases, list) else 'non-list'}.")
    case_ids = {case["id"] for case in cases}
    visual_off = load_visual_off(Path(args.visual_off_results_file), Path(args.visual_off_fallback_results_file), case_ids)
    missing = sorted(case_ids - set(visual_off))
    if missing:
        raise SystemExit(f"Curated IDs missing from visual-OFF results: {missing}")

    print(f"GPT-triples visual curated run: {len(cases)} cases, templates {dict(Counter(case['template'] for case in cases))}", flush=True)
    if args.skip_run:
        results = read_json(Path(args.results_file))
    else:
        results = evaluate(args, cases)
        write_json(Path(args.results_file), results)

    expected_trials = len(cases) * args.trials * 2
    actual_trials = sum(len(entry["trials"][label]) for entry in results["per_question"] for label in ("three_axis", "four_axis"))
    if actual_trials != expected_trials:
        raise SystemExit(f"Expected {expected_trials} trial records, found {actual_trials}")
    print(format_summary(results, visual_off, args.trials), flush=True)


if __name__ == "__main__":
    main()
