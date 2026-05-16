#!/usr/bin/env python3
"""
Spatial ablation against the EgoLifeQA subset (DAY1, spatial-flavored).

Compares two contexts on identical questions:
  - baseline:     top-K episodic triples (keyword overlap retrieval) only
  - with_spatial: top-K episodic + top-M spatial triples

Why simple keyword retrieval (not HippoRAG/PPR): we did not build the
multi-scale episodic captions or HippoRAG indices required by the full
eval_egolife.py pipeline. Keyword overlap is a fair proxy that does NOT
favor either side - it filters episodic and spatial in the same way.
"""

import argparse
import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Set, Tuple

from worldmm.llm import LLMModel


SYSTEM_PROMPT = (
    "You answer a multiple-choice question about a 7-day egocentric video. "
    "Use ONLY the provided context. If the context is insufficient, pick the "
    "most consistent option but DO NOT invent facts. "
    "Output ONLY the letter (A, B, C, or D) on a single line. No explanation."
)


SPATIAL_PATTERNS = re.compile(
    r"\b(where|located|room|kitchen|bedroom|table|floor|living|near|next to|"
    r"in front|in the [a-z]+ room|on the|behind)\b",
    re.IGNORECASE,
)


STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "at", "for", "with", "and", "or",
    "is", "are", "was", "were", "be", "been", "being", "i", "me", "my", "you",
    "your", "we", "he", "she", "it", "his", "her", "their", "this", "that",
    "these", "those", "do", "does", "did", "have", "has", "had", "from",
    "what", "when", "where", "who", "whom", "whose", "why", "how",
    "while", "before", "after", "first", "ever", "any", "by", "as",
}


def tokenize(text: str) -> Set[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z_]+", text.lower())
    return {t for t in tokens if t not in STOPWORDS and len(t) > 1}


def load_episodic_triples(path: str) -> List[List[str]]:
    with open(path) as f:
        data = json.load(f)
    triples: List[List[str]] = []
    for _ts, ts_triples in data.get("episodic_triples", {}).items():
        triples.extend(ts_triples)
    return triples


def load_spatial_triples(path: str) -> List[List[str]]:
    with open(path) as f:
        data = json.load(f)
    timestamps = sorted(data.keys())
    if not timestamps:
        return []
    return data[timestamps[-1]].get("consolidated_spatial_triples", [])


def score_triple(triple_tokens: Set[str], query_tokens: Set[str]) -> int:
    return len(triple_tokens & query_tokens)


def retrieve_top_k(triples: List[List[str]], query_tokens: Set[str], top_k: int) -> List[List[str]]:
    scored: List[Tuple[int, int, List[str]]] = []
    for idx, t in enumerate(triples):
        toks = tokenize(" ".join(map(str, t)))
        s = score_triple(toks, query_tokens)
        if s > 0:
            scored.append((s, idx, t))
    scored.sort(key=lambda x: (-x[0], x[1]))
    if len(scored) >= top_k:
        return [t for _, _, t in scored[:top_k]]
    fallback = [t for _, _, t in scored]
    extras_needed = top_k - len(fallback)
    fallback.extend(triples[:extras_needed])
    return fallback


def format_triples(triples: List[List[str]], header: str) -> str:
    if not triples:
        return f"{header}\n  (none)"
    lines = [header]
    for t in triples:
        lines.append(f"  ({', '.join(map(str, t))})")
    return "\n".join(lines)


def build_query_tokens(question: Dict[str, Any]) -> Set[str]:
    parts = [question["question"]]
    for letter in ("A", "B", "C", "D"):
        parts.append(question[letter])
    return tokenize(" ".join(parts))


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


def filter_questions(path: str, day: str = "DAY1") -> List[Dict[str, Any]]:
    with open(path) as f:
        qa = json.load(f)
    out = []
    for q in qa:
        if q.get("query_time", {}).get("date") != day:
            continue
        if not SPATIAL_PATTERNS.search(q.get("question", "")):
            continue
        out.append({
            "id": q["ID"],
            "question": q["question"],
            "A": q["choice_a"],
            "B": q["choice_b"],
            "C": q["choice_c"],
            "D": q["choice_d"],
            "answer": q["answer"],
            "type": q.get("type"),
        })
    return out


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
    parser.add_argument(
        "--qa-file",
        default="data/EgoLife/EgoLifeQA/EgoLifeQA_A1_JAKE.json",
    )
    parser.add_argument("--day", default="DAY1")
    parser.add_argument("--episodic-top-k", type=int, default=50)
    parser.add_argument("--spatial-top-k", type=int, default=25)
    parser.add_argument("--model", default="chatgpt-gpt-5.4")
    parser.add_argument("--output", default="output/spatial_ablation_egolifeqa.json")
    args = parser.parse_args()

    episodic = load_episodic_triples(args.episodic_file)
    spatial = load_spatial_triples(args.spatial_file)
    print(f"Loaded {len(episodic)} episodic triples, {len(spatial)} spatial triples.")

    questions = filter_questions(args.qa_file, day=args.day)
    print(f"Filtered to {len(questions)} {args.day} spatial-flavored questions.")

    model = LLMModel(model_name=args.model)
    results = []
    baseline_hits = 0
    with_spatial_hits = 0
    differ_count = 0

    for q in questions:
        qt = build_query_tokens(q)
        ep_top = retrieve_top_k(episodic, qt, args.episodic_top_k)
        sp_top = retrieve_top_k(spatial, qt, args.spatial_top_k)
        ep_ctx = format_triples(ep_top, "Episodic triples (top relevant):")
        sp_ctx = format_triples(sp_top, "Spatial triples (top relevant):")
        combined_ctx = f"{ep_ctx}\n\n{sp_ctx}"

        baseline_pred = ask(model, q, ep_ctx)
        with_spatial_pred = ask(model, q, combined_ctx)
        bl = baseline_pred.strip()[:1].upper()
        wl = with_spatial_pred.strip()[:1].upper()
        bok = bl == q["answer"]
        wok = wl == q["answer"]
        if bok:
            baseline_hits += 1
        if wok:
            with_spatial_hits += 1
        if bl != wl:
            differ_count += 1
        marker = "  " if bok == wok else ("UP" if (not bok and wok) else "DN")
        print(f"Q{q['id']} [{q['type']:<13}] gold={q['answer']} bl={bl} sp={wl} {marker}  | {q['question'][:80]}")
        results.append({
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "gold": q["answer"],
            "baseline": {"prediction": baseline_pred, "letter": bl, "correct": bok},
            "with_spatial": {"prediction": with_spatial_pred, "letter": wl, "correct": wok},
            "spatial_retrieved": sp_top,
        })

    total = len(questions)
    bl_pct = baseline_hits / total * 100 if total else 0
    sp_pct = with_spatial_hits / total * 100 if total else 0
    print("\n=== ABLATION SUMMARY ===")
    print(f"  baseline (episodic-only):  {baseline_hits}/{total} ({bl_pct:.1f}%)")
    print(f"  with_spatial:              {with_spatial_hits}/{total} ({sp_pct:.1f}%)")
    print(f"  delta:                     {with_spatial_hits - baseline_hits:+d}  ({sp_pct - bl_pct:+.1f}%p)")
    print(f"  answers differ:            {differ_count}/{total}")

    summary = {
        "totals": {
            "n": total,
            "baseline_correct": baseline_hits,
            "with_spatial_correct": with_spatial_hits,
            "baseline_pct": bl_pct,
            "with_spatial_pct": sp_pct,
            "delta_correct": with_spatial_hits - baseline_hits,
            "delta_pct_points": sp_pct - bl_pct,
            "answers_differ": differ_count,
        },
        "per_question": results,
        "context": {
            "total_episodic_triples": len(episodic),
            "total_spatial_triples": len(spatial),
            "episodic_top_k": args.episodic_top_k,
            "spatial_top_k": args.spatial_top_k,
        },
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
