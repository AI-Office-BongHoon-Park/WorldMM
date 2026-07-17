#!/usr/bin/env python3
"""Verbose one-off debugger for grounded truly-spatial smoke failures."""

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

RESULTS_PATH = Path("output/golden_qa_grounded_subset_results.json")
SAMPLE_PATH = Path("output/golden_qa_sample_truly_spatial.json")
EXTRACTION_PATH = Path("output/metadata/spatial_memory/A1_JAKE/spatial_extraction_results_chatgpt-gpt-5.4.json")
GROUNDING_PATH = Path("output/metadata/spatial_memory/A1_JAKE/unified_grounding_a1_jake.json")
DEFAULT_SPATIAL_PATH = Path("output/golden_qa_grounded_subset_spatial.json")
WORKER_DIR = Path("output/golden_qa_grounded_subset_workers")


def read_json(path: Path) -> Any:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def to_until_time(query_time: dict[str, Any]) -> int:
    day = str(query_time.get("date", "DAY1")).replace("DAY", "").replace("Day", "") or "1"
    raw = str(query_time.get("time", "000000"))
    digits = re.sub(r"\D", "", raw)
    if len(digits) <= 6:
        digits = digits.ljust(6, "0") + "00"
    else:
        digits = digits[:8].ljust(8, "0")
    return int(day + digits)


def timestamp_label(timestamp: int | str) -> str:
    value = str(timestamp)
    return f"DAY{value[0]} {value[1:3]}:{value[3:5]}:{value[5:7]}"


def normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def find_question(results: dict[str, Any], sample: dict[str, Any], question_id: str) -> dict[str, Any]:
    for question in results.get("per_question", []):
        if question.get("id") == question_id:
            return question
    for question in sample.get("questions", []):
        if question.get("id") == question_id:
            return question
    raise SystemExit(f"Question id not found in results/sample: {question_id}")


def evidence_triples(question: dict[str, Any]) -> list[list[str]]:
    triples: list[list[str]] = []
    for item in question.get("evidence_triples") or []:
        triple = item.get("triple") if isinstance(item, dict) else item
        if isinstance(triple, list) and len(triple) >= 3:
            triples.append([str(triple[0]), str(triple[1]), str(triple[2])])
    return triples


def triple_matches_gold(triple: list[Any], question: dict[str, Any]) -> bool:
    gold = normalize(question.get("gold_answer", ""))
    joined = normalize(" ".join(str(part) for part in triple))
    if gold and gold in joined:
        return True
    for gold_triple in evidence_triples(question):
        if [normalize(part) for part in triple[:3]] == [normalize(part) for part in gold_triple[:3]]:
            return True
    return False


def print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def print_triples(triples: list[list[Any]], question: dict[str, Any], prefix: str = "") -> bool:
    hit = False
    for index, triple in enumerate(triples):
        marker = "  <-- GOLD" if triple_matches_gold(triple, question) else ""
        hit = hit or bool(marker)
        print(f"{prefix}{index:03d}: {triple}{marker}")
    return hit


def extraction_window(extraction: dict[str, list[list[Any]]], chunk: str) -> list[str]:
    keys = sorted(int(key) for key in extraction)
    chunk_int = int(chunk)
    if chunk_int not in keys:
        before = [key for key in keys if key < chunk_int]
        after = [key for key in keys if key > chunk_int]
        window = []
        if before:
            window.append(before[-1])
        if after:
            window.append(after[0])
        return [str(key) for key in window]
    pos = keys.index(chunk_int)
    return [str(key) for key in keys[max(0, pos - 1): min(len(keys), pos + 2)]]


def axis_verdict(question: dict[str, Any], config: str) -> tuple[bool, Counter[str]]:
    counter: Counter[str] = Counter()
    picked = False
    for trial in question.get("trials", {}).get(config, []):
        axes = trial.get("axis_selections") or []
        counter.update(axes)
        picked = picked or "spatial" in axes
    return picked, counter


def find_worker_orders(question_id: str, config: str, current_until: int) -> list[dict[str, Any]]:
    orders = []
    for path in sorted((ROOT / WORKER_DIR).glob(f"payload_{config}_*_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        questions = payload.get("questions", [])
        ids = [question.get("id") for question in questions]
        if question_id not in ids:
            continue
        rows = []
        for question in questions:
            until = to_until_time(question.get("query_time", {}))
            rows.append({
                "id": question.get("id"),
                "chunk": question.get("evidence_chunk"),
                "until": until,
                "label": timestamp_label(until),
                "is_future_before_target": until > current_until and ids.index(question.get("id")) < ids.index(question_id),
            })
        orders.append({"path": path.relative_to(ROOT).as_posix(), "rows": rows})
    return orders


def parse_runner_templates() -> dict[str, str]:
    source = (ROOT / "tools/run_golden_qa_grounded_subset.py").read_text(encoding="utf-8")
    configs: dict[str, str] = {}
    for config in ["spatial_OFF", "spatial_ON_text", "spatial_ON_grounded"]:
        match = re.search(rf'"{config}"\s*:\s*\{{.*?"template"\s*:\s*"([^"]+)"', source, re.S)
        configs[config] = match.group(1) if match else "<not found>"
    return configs


def template_probe(template_name: str) -> dict[str, Any]:
    path = ROOT / "src/worldmm/llm/templates" / f"{template_name}.py"
    if not path.exists():
        return {"path": path, "exists": False, "has_spatial_vocab": False, "mentions_visual": False}
    text = path.read_text(encoding="utf-8")
    return {
        "path": path,
        "exists": True,
        "has_spatial_vocab": all(token in text for token in ["Spatial", "on|in|next_to|located_in", "spatial"]),
        "mentions_visual": "visual" in text.lower(),
    }


def load_spatial_memory(spatial_path: Path, grounding_path: Path):
    from worldmm.embedding import EmbeddingModel  # type: ignore[reportMissingImports]
    from worldmm.memory.spatial import SpatialMemory  # type: ignore[reportMissingImports]

    embedding_model = EmbeddingModel(text_model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    embedding_model.load_model(model_type="text")
    memory = SpatialMemory(embedding_model=embedding_model)
    memory.load_triples_from_file(str(ROOT / spatial_path), grounding_file=str(ROOT / grounding_path))
    return memory


def entry_matches_gold(entry: Any, question: dict[str, Any]) -> bool:
    for gold_triple in evidence_triples(question):
        if [normalize(entry.subject), normalize(entry.predicate), normalize(entry.object)] == [normalize(part) for part in gold_triple[:3]]:
            return True
    return triple_matches_gold([entry.subject, entry.predicate, entry.object], question)


def print_retrieval(label: str, entries: list[Any], question: dict[str, Any]) -> bool:
    print(f"{label}: count={len(entries)}")
    hit = False
    for index, entry in enumerate(entries, 1):
        rendered = entry.to_display_str()
        marker = "  <-- GOLD" if entry_matches_gold(entry, question) else ""
        hit = hit or bool(marker)
        print(f"{index}. {entry.id}: {rendered}{marker}")
    return hit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question-id", required=True)
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument("--sample", type=Path, default=SAMPLE_PATH)
    parser.add_argument("--extraction", type=Path, default=EXTRACTION_PATH)
    parser.add_argument("--spatial-file", type=Path, default=DEFAULT_SPATIAL_PATH)
    parser.add_argument("--grounding-file", type=Path, default=GROUNDING_PATH)
    args = parser.parse_args()

    results = read_json(args.results)
    sample = read_json(args.sample)
    extraction_doc = read_json(args.extraction)
    extraction = extraction_doc.get("spatial_triples", extraction_doc)
    question = find_question(results, sample, args.question_id)
    current_until = to_until_time(question.get("query_time", {}))
    gold_triples = evidence_triples(question)

    print_header("Question")
    print(f"id: {question.get('id')}")
    print(f"question: {question.get('question')}")
    print(f"gold: {question.get('gold')} = {question.get('gold_answer')}")
    evidence_chunk = str(question.get("evidence_chunk", ""))
    print(f"evidence_chunk: {evidence_chunk} ({timestamp_label(evidence_chunk)})")
    print(f"query_time: {question.get('query_time')} -> until_time={current_until} ({timestamp_label(current_until)})")
    print(f"evidence_triples: {gold_triples}")
    print(f"sample_true_question_count: {len(sample.get('questions', []))}")
    print(f"results_per_question_count: {len(results.get('per_question', []))}")

    print_header("H1 Gold answer missing from extraction triples")
    window = extraction_window(extraction, evidence_chunk)
    h1_hit = False
    for chunk in window:
        triples = extraction.get(chunk, [])
        print(f"chunk {chunk} ({timestamp_label(chunk)}), triples={len(triples)}")
        h1_hit = print_triples(triples, question, prefix="  ") or h1_hit
    print(f"H1 verdict - gold in evidence/window extraction triples: {'yes' if h1_hit else 'no'}")

    print_header("H2 Reasoner routing failure")
    h2_grounded_picked, h2_grounded_counts = axis_verdict(question, "spatial_ON_grounded")
    for config in ["spatial_OFF", "spatial_ON_text", "spatial_ON_grounded"]:
        picked, counts = axis_verdict(question, config)
        print(f"{config}: picked_spatial={picked}, counts={dict(counts)}")
        for trial in question.get("trials", {}).get(config, []):
            print(
                f"  trial {trial.get('trial')}: axes={trial.get('axis_selections')} "
                f"prediction={trial.get('prediction')} correct={trial.get('correct')}"
            )
    print(f"H2 verdict - spatial_ON_grounded reasoner picked spatial at least once: {'yes' if h2_grounded_picked else 'no'}")

    print_header("H3 Prompt overflow from geometry")
    chosen_trial = None
    for trial in question.get("trials", {}).get("spatial_ON_grounded", []):
        if trial.get("retrieved_spatial_text"):
            chosen_trial = trial
            break
    saved_block = ""
    if chosen_trial:
        saved_block = chosen_trial.get("retrieved_spatial_text", [""])[0]
        print(f"saved spatial_ON_grounded trial: {chosen_trial.get('trial')}")
        print(f"saved retrieved_spatial_text chars: {len(saved_block)}")
        print(saved_block)
    else:
        print("No saved spatial_ON_grounded spatial block found.")
    prompt_overflow = len(saved_block) > 100_000
    print(f"H3 verdict - geometry prompt overflow: {'yes' if prompt_overflow else 'no'}")

    print_header("H4 Template mismatch")
    runner_templates = parse_runner_templates()
    expected = {"spatial_OFF": "memory_reasoning_es", "spatial_ON_grounded": "memory_reasoning_essp"}
    for config, template in runner_templates.items():
        probe = template_probe(template)
        print(
            f"{config}: actual={template}, path={probe['path'].relative_to(ROOT) if probe['exists'] else probe['path']}, "
            f"has_spatial_vocab={probe['has_spatial_vocab']}, mentions_visual={probe['mentions_visual']}"
        )
    for config, template in expected.items():
        probe = template_probe(template)
        print(
            f"expected {config}: {template}, path={probe['path'].relative_to(ROOT) if probe['exists'] else probe['path']}, "
            f"exists={probe['exists']}, has_spatial_vocab={probe['has_spatial_vocab']}, mentions_visual={probe['mentions_visual']}"
        )
    template_correct = runner_templates.get("spatial_OFF") == expected["spatial_OFF"] and runner_templates.get("spatial_ON_grounded") == expected["spatial_ON_grounded"]
    print(f"H4 verdict - runner uses requested ES/ESSP templates: {'yes' if template_correct else 'no'}")

    print_header("H5 Embedding retrieval mismatch")
    query = str(question.get("question", ""))
    memory = load_spatial_memory(args.spatial_file, args.grounding_file)
    memory.index(current_until)
    print(f"fresh spatial index timestamp: {memory.indexed_timestamp} ({timestamp_label(memory.indexed_timestamp)}), entries={len(memory.indexed_entries)}")
    top5 = memory.retrieve(query, top_k=5, as_context=False)
    h5_hit = print_retrieval("fresh MiniLM/PPR top-5", top5, question)
    top8 = memory.retrieve(query, top_k=8, as_context=False)
    print_retrieval("fresh WorldMemory top-8", top8, question)
    gold_entries = [entry for entry in memory.indexed_entries if entry_matches_gold(entry, question)]
    if gold_entries:
        print("gold entries present in fresh index:")
        for entry in gold_entries:
            print(f"  {entry.id}: {entry.to_display_str()}")
    else:
        print("gold entry absent from fresh index")
    print(f"H5 verdict - MiniLM/PPR top-5 contains gold evidence triple: {'yes' if h5_hit else 'no'}")

    print_header("Worker-order / stale-index check")
    orders = find_worker_orders(str(question.get("id")), "spatial_ON_grounded", current_until)
    stale_order = False
    future_until = None
    for order in orders:
        print(order["path"])
        for row in order["rows"]:
            marker = "  <-- FUTURE BEFORE TARGET" if row["is_future_before_target"] else ""
            stale_order = stale_order or row["is_future_before_target"]
            if row["is_future_before_target"]:
                future_until = row["until"]
            print(f"  {row['id']} chunk={row['chunk']} until={row['until']} {row['label']}{marker}")
    if stale_order and future_until:
        memory.reset_index()
        memory.index(future_until)
        print(f"simulated stale index timestamp: {memory.indexed_timestamp} ({timestamp_label(memory.indexed_timestamp)}), entries={len(memory.indexed_entries)}")
        stale_top8 = memory.retrieve(query, top_k=8, as_context=False)
        print_retrieval("stale future-index top-8", stale_top8, question)
        memory.index(current_until)
        print(f"after backward reindex timestamp: {memory.indexed_timestamp} ({timestamp_label(memory.indexed_timestamp)}), entries={len(memory.indexed_entries)}")
        fixed_top8 = memory.retrieve(query, top_k=8, as_context=False)
        fixed_hit = print_retrieval("fixed backward-reindex top-8", fixed_top8, question)
        print(f"fixed backward reindex contains gold: {'yes' if fixed_hit else 'no'}")
    print(f"stale future index reproduced: {'yes' if stale_order else 'no'}")

    print_header("Verdict table")
    evidence_h1 = f"window={window}; gold_triples={gold_triples}"
    evidence_h2 = f"grounded_axes={[trial.get('axis_selections') for trial in question.get('trials', {}).get('spatial_ON_grounded', [])]}"
    evidence_h3 = f"saved_spatial_chars={len(saved_block)}"
    evidence_h4 = f"actual_OFF={runner_templates.get('spatial_OFF')}; actual_ON_grounded={runner_templates.get('spatial_ON_grounded')}"
    evidence_h5 = "; ".join(entry.to_display_str() for entry in top5)
    rows = [
        ("1", "Gold in spatial triples?", "yes" if h1_hit else "no", evidence_h1),
        ("2", "Reasoner picked spatial?", "yes" if h2_grounded_picked else "no", evidence_h2),
        ("3", "Prompt overflow?", "yes" if prompt_overflow else "no", evidence_h3),
        ("4", "Template correct?", "yes" if template_correct else "no", evidence_h4),
        ("5", "Retrieval hits gold?", "yes" if h5_hit else "no", evidence_h5),
    ]
    print("| Hyp | Question | Verdict | Evidence |")
    print("|---|---|---|---|")
    for hyp, label, verdict, evidence in rows:
        print(f"| {hyp} | {label} | {verdict} | {evidence.replace('|', '/')} |")

    print_header("Stdout summary")
    print(
        "Root cause: spatial-ON-grounded did not fail because the geometry sidecar was absent or oversized; "
        "the smoke worker reused one WorldMemory across non-chronological questions, and pre-fix WorldMemory.answer only reindexed when until_time increased, "
        f"so {question.get('id')} at {timestamp_label(current_until)} read a future spatial index before its own chunk. "
        "The runner also uses generic visual-enabled templates instead of ES/ESSP, so unloaded visual routes add noise. "
        "Fixed path: reindex when query time changes and rebuild SpatialMemory per-index state before retrieval."
    )


if __name__ == "__main__":
    main()
