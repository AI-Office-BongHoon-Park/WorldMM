"""Lightweight CLIP-text-side embedder for VisualMemory cross-modal retrieval.

VisualMemory expects `embedding_model.encode_vis_query(text)` to produce an
embedding in the same space as the precomputed clip image embeddings. The full
EmbeddingModel routes that call to VLM2Vec V2 which is too heavy for our
~5 GB GPU budget. This wrapper instead reuses the same
`sentence-transformers/clip-ViT-B-32` model that built the clip embeddings,
encoding only the *text* side of CLIP at query time.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np


class ClipQueryEmbedder:
    def __init__(self, model_name: str = "clip-ViT-B-32", device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    def encode_vis_query(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        emb = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return emb

    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        return self.encode_vis_query(texts)
