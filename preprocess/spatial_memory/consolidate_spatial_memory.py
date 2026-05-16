#!/usr/bin/env python3
"""
Consolidate spatial triples across timestamps.
Mirrors preprocess/semantic_memory/consolidate_semantic_memory.py.
"""

import argparse
import json
import os
import logging
from typing import Dict, Any
from tqdm import tqdm

from worldmm.memory.spatial import SpatialConsolidation
from worldmm.embedding import EmbeddingModel
from worldmm.llm import LLMModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_spatial_extraction_results(json_file: str) -> Dict[str, Any]:
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_spatial_consolidation(
    spatial_file: str,
    output_dir: str,
    model_name: str = "chatgpt/gpt-5.4",
    llm_model: LLMModel = None,
    embedding_model: EmbeddingModel = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(spatial_file):
        logger.error(f"Spatial extraction results file not found: {spatial_file}")
        return

    spatial_results = load_spatial_extraction_results(spatial_file)
    timestamps_count = len(spatial_results.get('spatial_triples', {}))
    logger.info(f"Loaded spatial extraction results with {timestamps_count} timestamps")

    total_before = sum(len(t) for t in spatial_results.get('spatial_triples', {}).values())
    logger.info(f"Total spatial triples before consolidation: {total_before}")

    if embedding_model is None:
        embedding_model = EmbeddingModel(text_model_name="Qwen/Qwen3-Embedding-4B")
        embedding_model.load_model(model_type="text")
    if llm_model is None:
        llm_model = LLMModel(model_name=model_name)

    spatial_consolidation = SpatialConsolidation(llm_model, embedding_model)

    timestamps = sorted(spatial_results.get('spatial_triples', {}).keys())
    accumulated_triples: list = []
    accumulated_evidence: list = []
    timestamped_results: Dict[str, Any] = {}

    for i, timestamp in tqdm(enumerate(timestamps), total=len(timestamps)):
        logger.info(f"Processing timestamp {timestamp} ({i + 1}/{len(timestamps)})")
        current_triples = spatial_results['spatial_triples'].get(timestamp, [])
        transformed_current_evidence = [[] for _ in current_triples]

        existing_results = (accumulated_triples.copy(), accumulated_evidence.copy())
        new_results = (current_triples, transformed_current_evidence)

        consolidated_triples, consolidated_evidence, triples_to_remove = (
            spatial_consolidation.batch_spatial_consolidation(existing_results, new_results)
        )

        triples_to_remove_set = set()
        for triple, evidence in triples_to_remove:
            triples_to_remove_set.add((tuple(triple), tuple(evidence)))

        new_acc_t: list = []
        new_acc_e: list = []
        for acc_triple, acc_evidence in zip(accumulated_triples, accumulated_evidence):
            key = (tuple(acc_triple), tuple(acc_evidence))
            if key not in triples_to_remove_set:
                new_acc_t.append(acc_triple)
                new_acc_e.append(acc_evidence)

        accumulated_triples = new_acc_t
        accumulated_evidence = new_acc_e
        accumulated_triples.extend(consolidated_triples)
        accumulated_evidence.extend(consolidated_evidence)

        timestamped_results[timestamp] = {
            "consolidated_spatial_triples": accumulated_triples.copy(),
        }

    total_after = len(accumulated_triples)
    logger.info(f"Total spatial triples after consolidation: {total_after}")
    if total_before > 0:
        reduction = total_before - total_after
        pct = (reduction / total_before * 100) if total_before else 0.0
        logger.info(f"Reduction in triples: {reduction} ({pct:.1f}%)")

    safe_model = llm_model.model_name.replace('/', '_')
    output_file = os.path.join(output_dir, f"spatial_consolidation_results_{safe_model}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(timestamped_results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results have been saved to: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidate spatial triples across timestamps.")
    parser.add_argument("--spatial-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="output/metadata/spatial_memory")
    parser.add_argument("--model", type=str, default="chatgpt/gpt-5.4")
    args = parser.parse_args()

    run_spatial_consolidation(
        spatial_file=args.spatial_file,
        output_dir=args.output_dir,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
