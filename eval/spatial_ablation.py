#!/usr/bin/env python3
"""
Tiny spatial-ablation harness.

Compares answers to WHERE-style questions under two contexts:
- baseline: episodic triples only (the 3 pre-existing memories minus semantic+visual
  which weren't built for this PoC; episodic carries the same surface text as the
  spatial layer's input, so this is the strongest available baseline for our data).
- with_spatial: episodic triples + consolidated spatial triples.

Why this script (not eval_egolife.py): the full WorldMemory eval needs multi-scale
episodic captions plus semantic+visual memories that weren't built. This harness
isolates the spatial signal on the 6 spatial triples + 183 episodic triples we
actually have from A1_JAKE DAY1 22:00.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

from worldmm.llm import LLMModel


SYSTEM_PROMPT = (
    "You answer a multiple-choice question about a short stretch of an "
    "egocentric video. Use ONLY the provided context. If the context is "
    "insufficient, pick the most consistent option but DO NOT invent facts. "
    "Output ONLY the letter (A, B, C, or D) on a single line. No explanation."
)


def load_episodic_triples(path: str) -> List[List[str]]:
    with open(path) as f:
        data = json.load(f)
    triples = []
    for _ts, ts_triples in data.get("episodic_triples", {}).items():
        triples.extend(ts_triples)
    return triples


def load_spatial_triples(path: str) -> List[List[str]]:
    with open(path) as f:
        data = json.load(f)
    timestamps = sorted(data.keys())
    if not timestamps:
        return []
    final = data[timestamps[-1]].get("consolidated_spatial_triples", [])
    return final


def format_triples(triples: List[List[str]], header: str) -> str:
    lines = [header]
    for t in triples:
        lines.append(f"  ({', '.join(t)})")
    return "\n".join(lines)


def ask(model: LLMModel, question: Dict[str, Any], context: str) -> str:
    choices = "\n".join([
        f"A. {question['A']}",
        f"B. {question['B']}",
        f"C. {question['C']}",
        f"D. {question['D']}",
    ])
    user = (
        f"Context:\n{context}\n\n"
        f"Question: {question['question']}\n\n"
        f"Choices:\n{choices}\n\n"
        "Answer (single letter only):"
    )
    return model.generate([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodic-file",
        default="output/metadata/episodic_memory/A1_JAKE/episodic_triple_results_chatgpt-gpt-5.4.json",
    )
    parser.add_argument(
        "--spatial-file",
        default="output/metadata/spatial_memory/A1_JAKE/spatial_consolidation_results_chatgpt-gpt-5.4.json",
    )
    parser.add_argument("--qa-file", default="eval/spatial_ablation_questions.json")
    parser.add_argument("--model", default="chatgpt-gpt-5.4")
    parser.add_argument("--output", default="output/spatial_ablation_results.json")
    args = parser.parse_args()

    episodic = load_episodic_triples(args.episodic_file)
    spatial = load_spatial_triples(args.spatial_file)
    print(f"Loaded {len(episodic)} episodic triples, {len(spatial)} spatial triples.")

    with open(args.qa_file) as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} ablation questions.")

    model = LLMModel(model_name=args.model)

    ep_ctx = format_triples(episodic, "Episodic triples:")
    sp_ctx = format_triples(spatial, "Spatial triples:")
    combined_ctx = f"{ep_ctx}\n\n{sp_ctx}"

    results = []
    baseline_hits = 0
    with_spatial_hits = 0

    for q in questions:
        gold = q["answer"]
        baseline_pred = ask(model, q, ep_ctx)
        with_spatial_pred = ask(model, q, combined_ctx)
        baseline_letter = baseline_pred.strip()[:1].upper()
        with_spatial_letter = with_spatial_pred.strip()[:1].upper()
        baseline_ok = baseline_letter == gold
        with_spatial_ok = with_spatial_letter == gold
        if baseline_ok:
            baseline_hits += 1
        if with_spatial_ok:
            with_spatial_hits += 1
        print(f"\nQ{q['id']}: {q['question']}")
        print(f"  gold={gold}  baseline={baseline_letter} ({'OK' if baseline_ok else 'X'})  "
              f"+spatial={with_spatial_letter} ({'OK' if with_spatial_ok else 'X'})")
        results.append({
            "id": q["id"],
            "question": q["question"],
            "gold": gold,
            "baseline": {"prediction": baseline_pred, "letter": baseline_letter, "correct": baseline_ok},
            "with_spatial": {"prediction": with_spatial_pred, "letter": with_spatial_letter, "correct": with_spatial_ok},
        })

    total = len(questions)
    print("\n=== ABLATION SUMMARY ===")
    print(f"baseline (episodic-only):    {baseline_hits}/{total}")
    print(f"with_spatial (ep+spatial):   {with_spatial_hits}/{total}")

    summary = {
        "totals": {"n": total, "baseline_correct": baseline_hits, "with_spatial_correct": with_spatial_hits},
        "per_question": results,
        "context": {"episodic_triples": len(episodic), "spatial_triples": len(spatial)},
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
