import json
import os
import logging
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from .utils import SpatialConsolidationRawOutput, SPATIAL_PREDICATE_VOCAB
from ...llm import LLMModel, PromptTemplateManager
from ...embedding import EmbeddingModel

logger = logging.getLogger(__name__)


class SpatialConsolidation:
    def __init__(self, llm_model: LLMModel, embedding_model: EmbeddingModel):
        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={"system": "system", "user": "user", "assistant": "assistant"}
        )
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.similarity_threshold = 0.6

    def find_relevant_triples(
        self,
        new_triples: List[List[str]],
        existing_triples: List[Tuple[List[str], List[str]]],
        top_k: int = 20,
    ) -> List[List[Tuple[List[str], List[str]]]]:
        if not existing_triples:
            return [[] for _ in new_triples]

        new_triple_texts = [" ".join(triple) for triple in new_triples]
        existing_triple_texts = [" ".join(triple) for triple, _ in existing_triples]

        new_embeddings = self.embedding_model.encode(new_triple_texts, modality="text")
        existing_embeddings = self.embedding_model.encode(existing_triple_texts, modality="text")

        similarities = cosine_similarity(new_embeddings, existing_embeddings)
        top_k_indices = np.argsort(-similarities, axis=1)[:, :top_k]

        return [
            [existing_triples[idx] for idx in indices if similarities[i, idx] >= self.similarity_threshold]
            for i, indices in enumerate(top_k_indices)
        ]

    def consolidate_triple(
        self,
        new_triple: List[str],
        relevant_existing_triples: List[List[str]],
    ) -> Tuple[List[str], List[int]]:
        if not relevant_existing_triples:
            return new_triple, []

        formatted_existing_triples = "\n".join(
            f"{i}. {triple}" for i, triple in enumerate(relevant_existing_triples)
        )
        messages = self.prompt_template_manager.render(
            name='spatial_consolidation',
            new_triple=new_triple,
            existing_triples=formatted_existing_triples,
        )

        try:
            if isinstance(messages, str):
                raise ValueError("Expected chat template to return List[Dict], got string")
            response = self.llm_model.generate(messages, text_format=SpatialConsolidationRawOutput)
        except Exception as e:
            logger.warning(e)
            return new_triple, []

        if response.updated_triple[1] not in SPATIAL_PREDICATE_VOCAB:
            logger.debug(
                f"Consolidation produced out-of-vocab predicate {response.updated_triple[1]!r}; "
                "keeping original new triple."
            )
            return new_triple, []

        return response.updated_triple, response.triples_to_remove

    def batch_spatial_consolidation(
        self,
        existing_spatial_results: Tuple[List[List[str]], List[List[str]]],
        new_spatial_results: Tuple[List[List[str]], List[List[str]]],
    ) -> Tuple[List[List[str]], List[List[str]], List[Tuple[List[str], List[str]]]]:
        existing_spatial_triples, existing_episodic_evidence = existing_spatial_results
        new_spatial_triples, new_episodic_evidence = new_spatial_results

        if not new_spatial_triples:
            return [], [], []

        accumulated_triples: List[Tuple[List[str], List[str]]] = []
        for triple, evidence in zip(existing_spatial_triples, existing_episodic_evidence):
            accumulated_triples.append((triple, evidence))

        consolidated_results = self._process_timestamp_triples_concurrent(
            new_spatial_triples, new_episodic_evidence, accumulated_triples
        )

        consolidated_triples = [result["updated_triple"] for result in consolidated_results]
        consolidated_evidence = [result["merged_evidence"] for result in consolidated_results]

        all_triples_to_remove: List[Tuple[List[str], List[str]]] = []
        for result in consolidated_results:
            all_triples_to_remove.extend(result["triples_to_remove"])

        return consolidated_triples, consolidated_evidence, all_triples_to_remove

    def _process_timestamp_triples_concurrent(
        self,
        current_triples: List[List[str]],
        current_evidence: List[List[str]],
        accumulated_triples: List[Tuple[List[str], List[str]]],
    ) -> List[Dict[str, Any]]:
        all_relevant_existing_data = self.find_relevant_triples(current_triples, accumulated_triples)

        def process_single_triple(triple_idx: int) -> Dict[str, Any]:
            new_triple = current_triples[triple_idx]
            new_evidence = current_evidence[triple_idx]
            relevant_existing = all_relevant_existing_data[triple_idx]

            if not relevant_existing:
                return {
                    "updated_triple": new_triple,
                    "triples_to_remove": [],
                    "merged_evidence": new_evidence,
                    "triple_idx": triple_idx,
                }

            relevant_triples_only = [triple for triple, _ in relevant_existing]
            updated_triple, indices_to_remove = self.consolidate_triple(new_triple, relevant_triples_only)

            merged_evidence = list(new_evidence)
            triples_to_remove_data = []
            for remove_idx in indices_to_remove:
                if remove_idx < len(relevant_existing):
                    removed_triple, removed_evidence = relevant_existing[remove_idx]
                    merged_evidence.extend(removed_evidence)
                    triples_to_remove_data.append((removed_triple, removed_evidence))

            return {
                "updated_triple": updated_triple,
                "triples_to_remove": triples_to_remove_data,
                "merged_evidence": merged_evidence,
                "triple_idx": triple_idx,
            }

        results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_single_triple, i): i for i in range(len(current_triples))}
            pbar = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Consolidating spatial triples",
                leave=False,
            )
            for future in pbar:
                results.append(future.result())

        results.sort(key=lambda x: x["triple_idx"])
        return results

    def save_results(self, results: Dict[str, Any], output_dir: str = ".") -> None:
        json_results: Dict[str, Any] = {}
        for key, value in results.items():
            if hasattr(value, '__dict__'):
                json_results[key] = value.__dict__
            else:
                json_results[key] = value

        os.makedirs(output_dir, exist_ok=True)
        fname = f"spatial_consolidation_results_{self.llm_model.model_name.replace('/', '_')}.json"
        with open(os.path.join(output_dir, fname), 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
