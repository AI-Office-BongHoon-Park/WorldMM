#!/usr/bin/env python3
"""
Extract spatial triples from episodic triples.
Mirrors preprocess/semantic_memory/extract_semantic_triples.py: groups caption
data into chunks, retrieves corresponding OpenIE results, and extracts
spatial knowledge from them via SpatialExtraction.
"""

import argparse
import json
import os
import logging
from typing import List, Dict, Any

from worldmm.memory.spatial import SpatialExtraction
from worldmm.memory.episodic.utils import compute_mdhash_id
from worldmm.llm import LLMModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_PERIOD = 10


def load_caption_data(json_file: str) -> List[Dict]:
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_openie_results(json_file: str) -> Dict[str, Any]:
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def group_captions_and_get_openie_triples(
    caption_data: List[Dict],
    openie_data: Dict[str, Any],
    period: int = DEFAULT_PERIOD,
) -> Dict[str, List[List[str]]]:
    if 'triple_results' not in openie_data:
        raise ValueError("OpenIE data must contain 'triple_results' key")

    episodic_triples_batch: Dict[str, List[List[str]]] = {}

    for i in range(0, len(caption_data), period):
        chunk = caption_data[i:i + period]
        last_item = chunk[-1]
        group_key = last_item['date'][-1] + last_item['end_time'].zfill(8)

        group_triples: List[List[str]] = []
        for caption_item in chunk:
            text_hash = compute_mdhash_id(caption_item['text'], prefix="chunk-")
            if text_hash not in openie_data['triple_results']:
                raise ValueError(
                    f"Text hash {text_hash} not found in OpenIE results for text: "
                    f"{caption_item['text'][:100]}..."
                )
            group_triples.extend(openie_data['triple_results'][text_hash])
        episodic_triples_batch[group_key] = group_triples

    return episodic_triples_batch


def run_spatial_extraction(
    caption_file: str,
    openie_file: str,
    output_dir: str,
    model_name: str = "chatgpt/gpt-5.4",
    llm_model: LLMModel = None,
    period: int = DEFAULT_PERIOD,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(caption_file):
        logger.error(f"Caption file not found: {caption_file}")
        return
    caption_data = load_caption_data(caption_file)
    logger.info(f"Loaded {len(caption_data)} caption entries")

    if not os.path.exists(openie_file):
        logger.error(f"OpenIE results file not found: {openie_file}")
        return
    openie_data = load_openie_results(openie_file)
    logger.info(f"Loaded OpenIE results with {len(openie_data.get('triple_results', {}))} chunks")

    logger.info(f"Grouping captions into chunks of {period}...")
    try:
        episodic_triples_batch = group_captions_and_get_openie_triples(
            caption_data, openie_data, period=period
        )
        logger.info(f"Created {len(episodic_triples_batch)} caption groups")
        total_triples = sum(len(t) for t in episodic_triples_batch.values())
        logger.info(f"Total episodic triples: {total_triples}")
    except Exception as e:
        logger.error(f"Error grouping captions: {e}")
        return

    if llm_model is None:
        llm_model = LLMModel(model_name=model_name)

    spatial_extraction = SpatialExtraction(llm_model)
    logger.info("Processing with Spatial Extraction...")
    spatial_triples_results, _ = spatial_extraction.batch_spatial_extraction(
        episodic_triples_batch,
        output_dir=output_dir,
    )

    total = sum(len(r) for r in spatial_triples_results.values())
    logger.info(f"Total spatial triples extracted: {total}")
    safe_model = llm_model.model_name.replace('/', '_')
    logger.info(f"Results saved to: {output_dir}/spatial_extraction_results_{safe_model}.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract spatial triples from episodic triples.")
    parser.add_argument("--caption-file", type=str, required=True)
    parser.add_argument("--openie-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="output/metadata/spatial_memory")
    parser.add_argument("--model", type=str, default="chatgpt/gpt-5.4")
    parser.add_argument("--period", type=int, default=DEFAULT_PERIOD)
    args = parser.parse_args()

    run_spatial_extraction(
        caption_file=args.caption_file,
        openie_file=args.openie_file,
        output_dir=args.output_dir,
        model_name=args.model,
        period=args.period,
    )


if __name__ == "__main__":
    main()
