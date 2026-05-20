# pyright: reportMissingImports=false
#!/usr/bin/env python3
"""Demo grounded spatial retrieval through SpatialMemory.retrieve()."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worldmm.memory.spatial import SpatialMemory


class TokenEmbeddingModel:
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def encode_text(self, texts, **kwargs):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        vectors = np.vstack([self._encode_one(item) for item in items]).astype(np.float32)
        return vectors[0] if single else vectors

    def _encode_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in re.findall(r"[A-Za-z0-9_]+", text.lower()):
            aliases = [token]
            if token in {"puzzle", "board"}:
                aliases.extend(["jigsaw", "puzzle_piece", "plate", "table"])
            for alias in aliases:
                vec[hash(alias) % self.dim] += 1.0
        norm = float(np.linalg.norm(vec)) or 1.0
        return vec / norm


def main() -> None:
    triples_file = Path("output/metadata/spatial_memory/A1_JAKE/spatial_consolidation_results_chatgpt-gpt-5.4.json")
    grounding_file = Path("output/metadata/spatial_memory/A1_JAKE/grounding/120255900.json")
    memory = SpatialMemory(TokenEmbeddingModel())
    memory.load_triples_from_file(str(triples_file), grounding_file=str(grounding_file))
    memory.index(120255900)
    query = "where was the puzzle board?"
    result = memory.retrieve(query, top_k=25, as_context=True)
    print(f"query: {query}")
    print("grounded spatial retrieval:")
    print(result)


if __name__ == "__main__":
    main()
