#!/usr/bin/env python3
"""MiniLM query embedder for GPT structured visual triples.

Visual triples are embedded as plain triple text with
`sentence-transformers/all-MiniLM-L6-v2`; queries must use the same encoder.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np


class VisualTripleQueryEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        from sentence_transformers import SentenceTransformer  # type: ignore[reportMissingImports]

        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    def encode_vis_query(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        emb = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return emb.astype(np.float32, copy=False)

    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        return self.encode_vis_query(texts)
