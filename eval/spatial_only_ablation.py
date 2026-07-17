#!/usr/bin/env python3
"""
Spatial-only ablation on hand-curated WHERE-style questions grounded in our
extracted spatial triples. Same retrieval mechanics as
spatial_ablation_egolifeqa.py (keyword overlap top-K), but loads a custom
question file instead of EgoLifeQA.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from spatial_ablation_egolifeqa import (  # type: ignore
    load_episodic_triples,
    load_spatial_triples,
    retrieve_top_k,
    format_triples,
    build_query_tokens,
    ask,
)
from worldmm.llm import LLMModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodic-file",
        default="output/metadata/episodic_memory/A1_JAKE/episodic_triple_results_chatgpt-gpt-5.4.json",
    )
    parser.add_argument(
        "--spatial-file",
        default="output/metadata/spatial_memory/A1_JAKE/spatial_consolidation_results_chatgpt-gpt-5.4.json",
    )
    parser.add_argument("--qa-file", default="eval/spatial_only_questions.json")
    parser.add_argument("--episodic-top-k", type=int, default=50)
    parser.add_argument("--spatial-top-k", type=int, default=25)
    parser.add_argument("--model", default="chatgpt-gpt-5.4")
    parser.add_argument("--output", default="output/spatial_only_ablation.json")
    args = parser.parse_args()

    episodic = load_episodic_triples(args.episodic_file)
    spatial = load_spatial_triples(args.spatial_file)
    print(f"Loaded {len(episodic)} episodic, {len(spatial)} spatial triples.")

    with open(args.qa_file) as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} synthetic spatial-only questions.")

    model = LLMModel(model_name=args.model)
    results = []
    baseline_hits = 0
    with_spatial_hits = 0
    differ_count = 0
    flips_up = 0
    flips_down = 0
    by_category: dict = {}

    for q in questions:
        qt = build_query_tokens(q)
        ep_top = retrieve_top_k(episodic, qt, args.episodic_top_k)
        sp_top = retrieve_top_k(spatial, qt, args.spatial_top_k)
        ep_ctx = format_triples(ep_top, "Episodic triples (top relevant):")
        sp_ctx = format_triples(sp_top, "Spatial triples (top relevant):")
        combined_ctx = f"{ep_ctx}\n\n{sp_ctx}"

        bl_pred = ask(model, q, ep_ctx)
        wl_pred = ask(model, q, combined_ctx)
        bl = bl_pred.strip()[:1].upper()
        wl = wl_pred.strip()[:1].upper()
        bok = bl == q["answer"]
        wok = wl == q["answer"]
        if bok:
            baseline_hits += 1
        if wok:
            with_spatial_hits += 1
        if bl != wl:
            differ_count += 1
            if not bok and wok:
                flips_up += 1
            elif bok and not wok:
                flips_down += 1

        cat = q.get("category", "?")
        by_category.setdefault(cat, {"n": 0, "bl": 0, "sp": 0})
        by_category[cat]["n"] += 1
        if bok:
            by_category[cat]["bl"] += 1
        if wok:
            by_category[cat]["sp"] += 1

        marker = "  "
        if not bok and wok:
            marker = "UP"
        elif bok and not wok:
            marker = "DN"
        print(f"{q['id']:>4} [{cat:<18}] gold={q['answer']} bl={bl} sp={wl} {marker}  | {q['question'][:70]}")

        results.append({
            "id": q["id"],
            "category": cat,
            "question": q["question"],
            "gold": q["answer"],
            "baseline": {"prediction": bl_pred, "letter": bl, "correct": bok},
            "with_spatial": {"prediction": wl_pred, "letter": wl, "correct": wok},
            "grounding": q.get("grounding"),
            "spatial_retrieved": sp_top,
        })

    total = len(questions)
    print("\n=== ABLATION SUMMARY ===")
    print(f"  baseline (episodic-only):  {baseline_hits}/{total} ({100*baseline_hits/total:.1f}%)")
    print(f"  with_spatial:              {with_spatial_hits}/{total} ({100*with_spatial_hits/total:.1f}%)")
    print(f"  delta:                     {with_spatial_hits - baseline_hits:+d}  ({100*(with_spatial_hits-baseline_hits)/total:+.1f}%p)")
    print(f"  answers differ:            {differ_count}/{total}  (UP={flips_up} DN={flips_down})")

    print("\nBy category:")
    for cat, stats in sorted(by_category.items()):
        print(f"  {cat:<22} n={stats['n']:>2}  baseline={stats['bl']}/{stats['n']}  with_spatial={stats['sp']}/{stats['n']}")

    summary = {
        "totals": {
            "n": total,
            "baseline_correct": baseline_hits,
            "with_spatial_correct": with_spatial_hits,
            "baseline_pct": 100 * baseline_hits / total,
            "with_spatial_pct": 100 * with_spatial_hits / total,
            "delta": with_spatial_hits - baseline_hits,
            "answers_differ": differ_count,
            "flips_up": flips_up,
            "flips_down": flips_down,
        },
        "by_category": by_category,
        "per_question": results,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
