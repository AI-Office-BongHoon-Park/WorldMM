#!/usr/bin/env python3
"""Text-only visual memory shim for GPT structured triples."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from tools.visual_triple_query_embedder import VisualTripleQueryEmbedder


def _timestamp_int(date: str, time_str: str) -> int:
    day = str(date).replace("DAY", "").replace("Day", "") or "1"
    return int(day + str(time_str).zfill(8))


def _transform_timestamp(ts_str: str) -> str:
    if len(ts_str) < 7:
        return ts_str
    return f"DAY{ts_str[0]} {ts_str[1:3]}:{ts_str[3:5]}:{ts_str[5:7]}"


@dataclass
class VisualTripleEntry:
    id: str
    clip_id: str
    triple_text: str
    embedding: np.ndarray
    frame_idx: int
    frame_timestamp_s: float
    video_path: str = ""
    start_time: str = ""
    end_time: str = ""
    date: str = ""

    @property
    def timestamp_int(self) -> Tuple[int, int]:
        return _timestamp_int(self.date, self.start_time), _timestamp_int(self.date, self.end_time)

    def to_display_str(self) -> str:
        start, end = self.timestamp_int
        return f"{_transform_timestamp(str(start))} - {_transform_timestamp(str(end))}"

    def to_context_line(self) -> str:
        return f"[{self.clip_id} frame={self.frame_idx} t={self.frame_timestamp_s:.3f}s] {self.triple_text}"


class VisualTriplesMemory:
    def __init__(self, index_path: str, embedder: Optional[VisualTripleQueryEmbedder] = None):
        self.index_path = index_path
        self.embedding_model = embedder or VisualTripleQueryEmbedder()
        self.entries: List[VisualTripleEntry] = []
        self.clips: List[VisualTripleEntry] = self.entries
        self.indexed_entries: List[VisualTripleEntry] = []
        self.indexed_time = 0
        self.embeddings: Optional[torch.Tensor] = None
        self.load_index(index_path)

    def load_index(self, index_path: str) -> None:
        with Path(index_path).open("rb") as handle:
            raw: Dict[str, Dict[str, Any]] = pickle.load(handle)
        self.entries.clear()
        for triple_id, item in sorted(raw.items()):
            self.entries.append(
                VisualTripleEntry(
                    id=triple_id,
                    clip_id=str(item.get("clip_id", "")),
                    triple_text=str(item.get("triple_text", "")),
                    embedding=np.asarray(item["embedding"], dtype=np.float32),
                    frame_idx=int(item.get("frame_idx", 0)),
                    frame_timestamp_s=float(item.get("frame_timestamp_s", 0.0)),
                    video_path=str(item.get("video_path", "")),
                    start_time=str(item.get("start_time", "")),
                    end_time=str(item.get("end_time", "")),
                    date=str(item.get("date", "")),
                )
            )
        self.entries.sort(key=lambda entry: (entry.timestamp_int[0], entry.id))
        self.clips = self.entries

    def index(self, until_time: int) -> None:
        if self.indexed_time >= until_time:
            return
        self.indexed_entries = [entry for entry in self.entries if entry.timestamp_int[1] <= until_time]
        if self.indexed_entries:
            self.embeddings = torch.tensor(
                np.array([entry.embedding for entry in self.indexed_entries]),
                dtype=torch.float32,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        else:
            self.embeddings = None
        self.indexed_time = until_time

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        fps: float = 1.0,
        max_frames: int = 64,
        as_context: bool = True,
    ) -> Union[List[VisualTripleEntry], Dict[str, List[Any]]]:
        del fps, max_frames
        if not self.indexed_entries or self.embeddings is None:
            return {} if as_context else []
        q_emb = self.embedding_model.encode_vis_query(query)
        if len(q_emb.shape) == 1:
            q_emb = q_emb.reshape(1, -1)
        query_tensor = torch.tensor(q_emb, dtype=torch.float32, device=self.embeddings.device)
        similarities = F.cosine_similarity(query_tensor, self.embeddings, dim=1)
        k = min(top_k, len(self.indexed_entries))
        _, top_indices = torch.topk(similarities, k)
        results = [self.indexed_entries[idx] for idx in top_indices.cpu().tolist()]
        if not as_context:
            return results
        context: Dict[str, List[Any]] = {}
        for entry in results:
            context.setdefault(entry.to_display_str(), []).append(entry.to_context_line())
        return context

    def reset_index(self) -> None:
        self.indexed_entries = []
        self.indexed_time = 0
        self.embeddings = None

    def cleanup(self) -> None:
        self.embeddings = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
