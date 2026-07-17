import json
import os
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging

from .utils import SpatialRawOutput, SpatialOutput
from ...llm import LLMModel, PromptTemplateManager

logger = logging.getLogger(__name__)


class SpatialExtraction:
    def __init__(self, llm_model: LLMModel):
        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={"system": "system", "user": "user", "assistant": "assistant"}
        )
        self.llm_model = llm_model

    def spatial_extraction(
        self, chunk_key: str, episodic_triples: List[List[str]]
    ) -> SpatialOutput:
        formatted_triples = "\n".join(f"{i}. {triple}" for i, triple in enumerate(episodic_triples))
        messages = self.prompt_template_manager.render(
            name='spatial_extraction', episodic_triples=formatted_triples
        )

        try:
            response = self.llm_model.generate(messages, text_format=SpatialRawOutput)
        except Exception as e:
            logger.warning(e)
            return SpatialOutput(
                chunk_id=chunk_key,
                spatial_triples=[],
                episodic_evidence=[],
            )

        return SpatialOutput(
            chunk_id=chunk_key,
            spatial_triples=response.spatial_triples,
            episodic_evidence=response.episodic_evidence,
        )

    def save_results(self, results: Dict[str, Any], output_dir: str = ".") -> None:
        json_results: Dict[str, Any] = {}
        for key, value in results.items():
            if hasattr(value, '__dict__'):
                json_results[key] = value.__dict__
            else:
                json_results[key] = value

        os.makedirs(output_dir, exist_ok=True)
        fname = f"spatial_extraction_results_{self.llm_model.model_name.replace('/', '_')}.json"
        with open(os.path.join(output_dir, fname), 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)

    def batch_spatial_extraction(
        self,
        episodic_triples_batch: Dict[str, List[List[str]]],
        output_dir: str = ".",
    ) -> Tuple[Dict[str, List[List[str]]], Dict[str, List[List[int]]]]:
        results: List[SpatialOutput] = []
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.spatial_extraction, chunk_key, episodic_triples): episodic_triples
                for chunk_key, episodic_triples in episodic_triples_batch.items()
            }
            pbar = tqdm(as_completed(futures), total=len(futures), desc="Extracting spatial triples")
            for future in pbar:
                results.append(future.result())

        spatial_triples_map = {res.chunk_id: res.spatial_triples for res in results}
        episodic_evidence_map = {res.chunk_id: res.episodic_evidence for res in results}

        chunk_keys = list(episodic_triples_batch.keys())
        ordered_spatial_triples = {key: spatial_triples_map.get(key, []) for key in chunk_keys}
        ordered_episodic_evidence = {key: episodic_evidence_map.get(key, []) for key in chunk_keys}

        combined_results = {
            "spatial_triples": ordered_spatial_triples,
        }
        self.save_results(combined_results, output_dir)
        return ordered_spatial_triples, ordered_episodic_evidence
