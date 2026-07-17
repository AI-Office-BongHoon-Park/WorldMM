#!/usr/bin/env python3
"""
3-axis (episodic + semantic + visual) vs 4-axis (+ spatial) ablation using the
real WorldMemory.iterative_reasoning path on EgoLifeQA DAY1 questions.

Both runs share identical memory data, identical retrieval, identical QA call.
The only variable is which `memory_reasoning` template the reasoning agent sees:
  - 3-axis: memory_reasoning_3axis (Spatial entry removed from prompt)
  - 4-axis: memory_reasoning           (existing 4-type prompt, ships with WorldMM)

Visual memory is initialized empty for this run because the LiteLLM proxy
exposes only text-LLMs, so even with visual clips loaded the QA respond_llm
could not consume them. The "visual" branch is therefore a no-op for both
configurations and is reported as such in the summary.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from worldmm.embedding import EmbeddingModel  # type: ignore[reportMissingImports]
from worldmm.llm import LLMModel, PromptTemplateManager  # type: ignore[reportMissingImports]
from worldmm.memory import WorldMemory  # type: ignore[reportMissingImports]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from clip_query_embedder import ClipQueryEmbedder  # type: ignore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)


def build_memory(
    *,
    reasoning_template_name: str,
    embedding_model: EmbeddingModel,
    llm_model: LLMModel,
    episodic_caption_files: Dict[str, str],
    semantic_file: str,
    spatial_file: str,
    max_rounds: int,
    episodic_cache_tag: str,
    spatial_grounding_file: str = "",
    visual_embeddings_file: str = "",
    visual_clips_file: str = "",
) -> WorldMemory:
    pm = PromptTemplateManager()
    wm = WorldMemory(
        embedding_model=embedding_model,
        retriever_llm_model=llm_model,
        respond_llm_model=llm_model,
        prompt_template_manager=pm,
        episodic_granularities=list(episodic_caption_files.keys()),
        episodic_cache_root=f".cache/episodic_memory_{episodic_cache_tag}",
        qa_template_name="qa_egolife",
        reasoning_template_name=reasoning_template_name,
        max_rounds=max_rounds,
        max_errors=3,
    )
    wm.load_episodic_captions(caption_files=episodic_caption_files)
    if os.path.exists(semantic_file):
        wm.load_semantic_triples(file_path=semantic_file)
    if os.path.exists(spatial_file):
        if spatial_grounding_file and os.path.exists(spatial_grounding_file):
            wm.spatial_memory.load_triples_from_file(spatial_file, grounding_file=spatial_grounding_file)
        else:
            wm.load_spatial_triples(file_path=spatial_file)
    if visual_embeddings_file and visual_clips_file \
            and os.path.exists(visual_embeddings_file) and os.path.exists(visual_clips_file):
        wm.visual_memory.embedding_model = ClipQueryEmbedder()
        wm.load_visual_clips(
            embeddings_path=visual_embeddings_file,
            clips_path=visual_clips_file,
        )
    return wm


def normalize_letter(text: str) -> str:
    if not text:
        return ""
    stripped = text.strip()
    for ch in stripped:
        if ch.upper() in "ABCD":
            return ch.upper()
    return ""


def load_questions(qa_file: str, day: str, max_n: int = 0) -> List[Dict[str, Any]]:
    with open(qa_file) as f:
        all_qa = json.load(f)
    filtered = [q for q in all_qa if q.get("query_time", {}).get("date") == day]
    if max_n > 0:
        filtered = filtered[:max_n]
    return filtered


def to_until_time(query_time: Dict[str, str]) -> int:
    day = str(query_time.get("date", "")).replace("DAY", "").replace("Day", "")
    time_str = str(query_time.get("time", "")).zfill(8)
    return int(day + time_str)


def run_one(wm: WorldMemory, question: Dict[str, Any]) -> str:
    choices = {
        "A": question["choice_a"],
        "B": question["choice_b"],
        "C": question["choice_c"],
        "D": question["choice_d"],
    }
    until_time = to_until_time(question.get("query_time", {}))
    qa_result = wm.answer(
        query=question["question"],
        choices=choices,
        until_time=until_time,
    )
    return qa_result.answer or ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodic-caption-dir",
        default="data/EgoLife/EgoLifeCap/A1_JAKE",
    )
    parser.add_argument(
        "--semantic-file",
        default="output/metadata/semantic_memory/A1_JAKE/semantic_consolidation_results_chatgpt-gpt-5.4.json",
    )
    parser.add_argument(
        "--spatial-file",
        default="output/metadata/spatial_memory/A1_JAKE/spatial_consolidation_results_chatgpt-gpt-5.4.json",
    )
    parser.add_argument(
        "--spatial-grounding-file",
        default="",
        help="Optional unified SpatialMemory grounding sidecar.",
    )
    parser.add_argument("--visual-embeddings-file", default="output/metadata/visual_memory/A1_JAKE/visual_embeddings_clip-ViT-B-32.pkl")
    parser.add_argument("--visual-clips-file", default="output/metadata/visual_memory/A1_JAKE/visual_clips_clip-ViT-B-32.json")
    parser.add_argument("--qa-file", default="data/EgoLife/EgoLifeQA/EgoLifeQA_A1_JAKE.json")
    parser.add_argument("--day", default="DAY1")
    parser.add_argument("--max-n", type=int, default=10, help="0 = all DAY1 questions.")
    parser.add_argument("--model", default="chatgpt-gpt-5.4")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--output", default="output/three_vs_four_axis.json")
    args = parser.parse_args()

    questions = load_questions(args.qa_file, args.day, args.max_n)
    print(f"Loaded {len(questions)} DAY1 questions.")

    granularities = ["30sec", "3min", "10min", "1h"]
    episodic_caption_files = {
        g: os.path.join(args.episodic_caption_dir, f"A1_JAKE_{g}.json")
        for g in granularities
    }
    available = {g: p for g, p in episodic_caption_files.items() if os.path.exists(p)}
    if "30sec" not in available:
        sys.exit(f"Missing required 30sec captions at {episodic_caption_files['30sec']}.")
    print(f"Episodic granularities available: {list(available.keys())}")

    embedding_model = EmbeddingModel(text_model_name=args.embedding_model, device="cpu")
    embedding_model.load_model(model_type="text")
    llm_model = LLMModel(model_name=args.model)

    print("\n=== building 3-axis WorldMemory ===")
    wm3 = build_memory(
        reasoning_template_name="memory_reasoning_3axis",
        embedding_model=embedding_model,
        llm_model=llm_model,
        episodic_caption_files=available,
        semantic_file=args.semantic_file,
        spatial_file=args.spatial_file,
        spatial_grounding_file=args.spatial_grounding_file,
        max_rounds=args.max_rounds,
        episodic_cache_tag="3axis",
        visual_embeddings_file=args.visual_embeddings_file,
        visual_clips_file=args.visual_clips_file,
    )
    print("=== building 4-axis WorldMemory ===")
    wm4 = build_memory(
        reasoning_template_name="memory_reasoning",
        embedding_model=embedding_model,
        llm_model=llm_model,
        episodic_caption_files=available,
        semantic_file=args.semantic_file,
        spatial_file=args.spatial_file,
        spatial_grounding_file=args.spatial_grounding_file,
        max_rounds=args.max_rounds,
        episodic_cache_tag="4axis",
        visual_embeddings_file=args.visual_embeddings_file,
        visual_clips_file=args.visual_clips_file,
    )

    results: List[Dict[str, Any]] = []
    hits3 = 0
    hits4 = 0
    differ = 0
    flips_up = 0
    flips_down = 0

    for i, q in enumerate(questions, 1):
        gold = q["answer"]
        try:
            pred3_raw = run_one(wm3, q)
            pred4_raw = run_one(wm4, q)
        except Exception as exc:
            logger.warning("Q%s failed: %s", q.get("ID"), exc)
            continue
        p3 = normalize_letter(pred3_raw)
        p4 = normalize_letter(pred4_raw)
        ok3 = p3 == gold
        ok4 = p4 == gold
        if ok3:
            hits3 += 1
        if ok4:
            hits4 += 1
        if p3 != p4:
            differ += 1
            if not ok3 and ok4:
                flips_up += 1
            elif ok3 and not ok4:
                flips_down += 1

        marker = "  "
        if not ok3 and ok4:
            marker = "UP"
        elif ok3 and not ok4:
            marker = "DN"
        print(
            f"[{i:>3}/{len(questions)}] Q{q['ID']} [{q.get('type','?'):<14}] "
            f"gold={gold} 3a={p3 or '?'} 4a={p4 or '?'} {marker} | {q['question'][:70]}"
        )

        results.append({
            "id": q["ID"],
            "type": q.get("type"),
            "question": q["question"],
            "gold": gold,
            "three_axis": {"prediction": pred3_raw, "letter": p3, "correct": ok3},
            "four_axis": {"prediction": pred4_raw, "letter": p4, "correct": ok4},
        })

    n = len(results)
    pct = lambda h: 100 * h / n if n else 0
    print("\n=== ABLATION SUMMARY ===")
    print(f"  3-axis (E+S+V no-op):       {hits3}/{n} ({pct(hits3):.1f}%)")
    print(f"  4-axis (E+S+V no-op + Sp):  {hits4}/{n} ({pct(hits4):.1f}%)")
    print(f"  delta:                       {hits4 - hits3:+d}  ({pct(hits4) - pct(hits3):+.1f}%p)")
    print(f"  answers differ:              {differ}/{n}  (UP={flips_up} DN={flips_down})")

    summary = {
        "totals": {
            "n": n,
            "three_axis_correct": hits3,
            "four_axis_correct": hits4,
            "three_axis_pct": pct(hits3),
            "four_axis_pct": pct(hits4),
            "delta_correct": hits4 - hits3,
            "delta_pct_points": pct(hits4) - pct(hits3),
            "answers_differ": differ,
            "flips_up": flips_up,
            "flips_down": flips_down,
        },
        "config": {
            "model": args.model,
            "embedding_model": args.embedding_model,
            "max_rounds": args.max_rounds,
            "episodic_granularities": list(available.keys()),
            "visual_axis_note": "no-op (text-only LLM proxy + no visual clips loaded)",
        },
        "per_question": results,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
