"""
Spatial Memory module for WorldMM.

Mirrors SemanticMemory shape (graph + Personalized PageRank retrieval) but
encodes WHERE-axis knowledge: object/place co-occurrence + closed-vocab
spatial relations. Place vertices and object vertices share one igraph so
PPR can walk between them.
"""

import json
import logging
import torch
import torch.nn.functional as F
ig = __import__("igraph")
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass

from ...embedding import EmbeddingModel
from .utils import SPATIAL_PREDICATE_VOCAB
from .grounding import GazeTarget, GeometricGrounding, PlaceAnchor, Pose6DoF

logger = logging.getLogger(__name__)


@dataclass
class SpatialTripleEntry:
    id: str
    subject: str
    predicate: str
    object: str
    timestamp: int
    place: Optional[str] = None
    subject_grounding: Optional[GeometricGrounding] = None
    object_grounding: Optional[GeometricGrounding] = None
    pose_6dof: Optional[Pose6DoF] = None
    gaze_target: Optional[GazeTarget] = None
    place_anchor: Optional[PlaceAnchor] = None

    @property
    def triple(self) -> List[str]:
        return [self.subject, self.predicate, self.object]

    @property
    def text(self) -> str:
        if self.place:
            return f"{self.subject} {self.predicate} {self.object} @ {self.place}"
        return f"{self.subject} {self.predicate} {self.object}"

    def to_display_str(self) -> str:
        def render_side(name: str, grounding: Optional[GeometricGrounding]) -> str:
            if grounding is None:
                return ""
            center = grounding.bbox_center
            extent = grounding.bbox_extent
            return (
                f" [{name}_center=({center[0]:.2f},{center[1]:.2f},{center[2]:.2f})"
                f" {name}_extent=({extent[0]:.2f},{extent[1]:.2f},{extent[2]:.2f})"
                f" units={grounding.units}]"
            )

        if self.subject_grounding or self.object_grounding:
            subject = self.subject
            obj = self.object
            if self.subject_grounding:
                c = self.subject_grounding.bbox_center
                subject = f"{subject} @ ({c[0]:.2f},{c[1]:.2f},{c[2]:.2f}) rel"
            if self.object_grounding:
                c = self.object_grounding.bbox_center
                obj = f"{obj} @ ({c[0]:.2f},{c[1]:.2f},{c[2]:.2f}) rel"
            base = f"({subject}) [{self.predicate}] ({obj})"
        else:
            base = f"({self.subject}, {self.predicate}, {self.object})"
        if self.place:
            base += f" [place={self.place}]"
        base += render_side("subj", self.subject_grounding)
        base += render_side("obj", self.object_grounding)
        if self.pose_6dof:
            pose = self.pose_6dof
            base += (
                f" [pose_6dof=({pose.tx:.2f},{pose.ty:.2f},{pose.tz:.2f})"
                f" q=({pose.qw:.3f},{pose.qx:.3f},{pose.qy:.3f},{pose.qz:.3f})"
                f" quality={pose.quality_score:.3f} frame={pose.frame}]"
            )
        if self.gaze_target:
            gaze = self.gaze_target
            base += f" [gaze_target=yaw:{gaze.yaw_rads_cpf:.3f} pitch:{gaze.pitch_rads_cpf:.3f}]"
        if self.place_anchor:
            anchor = self.place_anchor
            centroid = anchor.centroid_world_m
            label = f" label={anchor.label}" if anchor.label else ""
            base += (
                f" [place_anchor=({centroid[0]:.2f},{centroid[1]:.2f},{centroid[2]:.2f})"
                f" frame={anchor.coordinate_frame_id}{label}]"
            )
        return base


def _transform_timestamp(ts_str: str) -> str:
    if len(ts_str) < 7:
        return ts_str
    day = ts_str[0]
    time_str = ts_str[1:]
    hh = time_str[0:2]
    mm = time_str[2:4]
    ss = time_str[4:6]
    return f"DAY{day} {hh}:{mm}:{ss}"


class SpatialMemory:
    """
    Spatial memory for WHERE-axis reasoning using a triple graph + PPR.

    Vertices are entities (objects, agents) AND place tokens. Triples form
    edges between them. Retrieval mirrors SemanticMemory exactly:
    1. Top-k similar triples by embedding cosine
    2. Personalize PPR on entities from those triples
    3. Score every indexed triple by sum of subject/object PPR scores
    4. Return top-k by PPR score
    """

    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model

        self.triple_id_to_entry: Dict[str, SpatialTripleEntry] = {}
        self.timestamp_to_triples: Dict[int, List[SpatialTripleEntry]] = {}
        self.available_timestamps: List[int] = []

        self.indexed_entries: List[SpatialTripleEntry] = []
        self.indexed_time: int = 0
        self.indexed_timestamp: int = 0

        self.graph: Optional[Any] = None
        self.embeddings: Optional[torch.Tensor] = None
        self.triple_to_entities: Dict[str, Tuple[str, str]] = {}

        self.max_object_vertices: int = 500

    def load_triples_from_file(self, file_path: str, grounding_file: Optional[str] = None) -> None:
        with open(file_path, 'r') as f:
            data = json.load(f)
        grounding_data = None
        if grounding_file:
            with open(grounding_file, 'r') as f:
                grounding_data = json.load(f)
        self.load_triples_from_data(data, grounding_data=grounding_data)

    def load_triples_from_data(
        self,
        data: Dict[str, Dict[str, Any]],
        grounding_data: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """
        Expected format (mirrors semantic):
        {
          "<timestamp>": {
            "consolidated_spatial_triples": [[subj, pred, obj], ...],
            "places": {<idx>: <place>, ...}  # optional, idx aligned with triples
          },
          ...
        }
        """
        for timestamp_str, content in data.items():
            try:
                timestamp = int(timestamp_str)
            except (TypeError, ValueError):
                logger.warning(f"Skipping non-integer timestamp key: {timestamp_str!r}")
                continue

            triples = content.get("consolidated_spatial_triples", [])
            places = content.get("places") or {}

            timestamp_entries: List[SpatialTripleEntry] = []
            for idx, triple in enumerate(triples):
                if len(triple) < 3:
                    logger.warning(f"Skipping invalid triple at {timestamp_str}[{idx}]: {triple}")
                    continue
                predicate = triple[1]
                if predicate not in SPATIAL_PREDICATE_VOCAB:
                    logger.debug(f"Drop out-of-vocab predicate {predicate!r} at {timestamp_str}[{idx}]")
                    continue

                triple_id = f"spatial_{timestamp}_{idx}"
                place = places.get(str(idx)) or places.get(idx)
                grounding_record = (grounding_data or {}).get(triple_id, {})
                subject_grounding = grounding_record.get("subject_grounding")
                object_grounding = grounding_record.get("object_grounding")
                pose_6dof = grounding_record.get("pose_6dof")
                gaze_target = grounding_record.get("gaze_target")
                place_anchor = grounding_record.get("place_anchor")
                entry = SpatialTripleEntry(
                    id=triple_id,
                    subject=triple[0],
                    predicate=predicate,
                    object=triple[2],
                    timestamp=timestamp,
                    place=place,
                    subject_grounding=GeometricGrounding(**subject_grounding) if subject_grounding else None,
                    object_grounding=GeometricGrounding(**object_grounding) if object_grounding else None,
                    pose_6dof=Pose6DoF(**pose_6dof) if pose_6dof else None,
                    gaze_target=GazeTarget(**gaze_target) if gaze_target else None,
                    place_anchor=PlaceAnchor(**place_anchor) if place_anchor else None,
                )
                self.triple_id_to_entry[triple_id] = entry
                timestamp_entries.append(entry)

            if timestamp_entries:
                self.timestamp_to_triples[timestamp] = timestamp_entries

        self.available_timestamps = sorted(self.timestamp_to_triples.keys())
        logger.info(f"Loaded spatial triples across {len(self.available_timestamps)} timestamps")

    def _cap_object_vertices(self, entries: List[SpatialTripleEntry]) -> List[SpatialTripleEntry]:
        """Drop triples whose subject or object is below frequency cap.

        Keeps the top `max_object_vertices` most frequent vertices across all
        indexed entries. Triples referencing dropped vertices are removed
        entirely so PPR cannot reach orphan endpoints.
        """
        freq: Dict[str, int] = {}
        for entry in entries:
            for token in (entry.subject, entry.object):
                if token:
                    freq[token] = freq.get(token, 0) + 1
        if len(freq) <= self.max_object_vertices:
            return entries

        kept = {tok for tok, _ in sorted(freq.items(), key=lambda kv: -kv[1])[: self.max_object_vertices]}
        survivors = [e for e in entries if e.subject in kept and e.object in kept]
        dropped = len(entries) - len(survivors)
        logger.info(
            f"Vertex cap applied: keep {len(kept)} entities, drop {dropped} triples "
            f"({100 * dropped / max(1, len(entries)):.1f}%)"
        )
        return survivors

    def index(self, until_time: int) -> None:
        closest_timestamp = None
        for ts in reversed(self.available_timestamps):
            if ts <= until_time:
                closest_timestamp = ts
                break

        if closest_timestamp is None:
            logger.debug(f"No timestamp found up to {until_time}")
            return

        if self.indexed_timestamp == closest_timestamp:
            logger.debug(f"Already indexed timestamp {closest_timestamp}, skipping")
            return

        entries_to_index = list(self.timestamp_to_triples.get(closest_timestamp, []))
        if not entries_to_index:
            logger.debug(f"No entries at timestamp {closest_timestamp}")
            return

        entries_to_index = self._cap_object_vertices(entries_to_index)
        if not entries_to_index:
            logger.warning("All entries dropped by vertex cap; skipping index.")
            return

        all_entities: Set[str] = set()
        self.triple_to_entities = {}
        for entry in entries_to_index:
            subj, obj = entry.subject, entry.object
            if subj:
                all_entities.add(subj)
            if obj:
                all_entities.add(obj)
            self.triple_to_entities[entry.id] = (subj, obj)

        graph = ig.Graph()
        self.graph = graph
        entity_list = list(all_entities)
        graph.add_vertices(entity_list)
        entity_to_vertex = {entity: i for i, entity in enumerate(entity_list)}

        edges_to_add: List[Tuple[int, int]] = []
        for entry in entries_to_index:
            subj, obj = self.triple_to_entities.get(entry.id, ("", ""))
            if subj in entity_to_vertex and obj in entity_to_vertex:
                sv = entity_to_vertex[subj]
                ov = entity_to_vertex[obj]
                if sv != ov:
                    edges_to_add.append((sv, ov))
        if edges_to_add:
            graph.add_edges(edges_to_add)

        all_texts = [entry.text for entry in entries_to_index]
        all_embeddings = self.embedding_model.encode_text(all_texts)

        self.embeddings = torch.tensor(
            all_embeddings,
            dtype=torch.float32,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.indexed_entries = entries_to_index
        self.indexed_time = until_time
        self.indexed_timestamp = closest_timestamp

        logger.info(
            f"Indexed {len(entries_to_index)} spatial triples from timestamp "
            f"{closest_timestamp} (query time: {until_time})"
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        as_context: bool = True,
    ) -> Union[List[SpatialTripleEntry], str]:
        if not self.indexed_entries or self.embeddings is None or self.graph is None:
            logger.warning("No spatial triples indexed. Call index(until_time) before retrieve().")
            return "" if as_context else []

        device = self.embeddings.device
        query_embedding = self.embedding_model.encode_text(query)
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_tensor = torch.tensor(query_embedding, dtype=torch.float32, device=device)

        similarities = F.cosine_similarity(query_tensor, self.embeddings, dim=1)
        num_available = len(self.indexed_entries)
        top_k_sim = min(top_k, num_available)
        top_values, top_pos_indices = torch.topk(similarities, top_k_sim)
        top_sim_entries = [self.indexed_entries[pos] for pos in top_pos_indices.cpu().tolist()]

        personalization_entities: Set[str] = set()
        for entry in top_sim_entries:
            subj, obj = self.triple_to_entities.get(entry.id, ("", ""))
            if subj:
                personalization_entities.add(subj)
            if obj:
                personalization_entities.add(obj)

        if not personalization_entities:
            if as_context:
                return self.retrieve_triples_as_str(top_sim_entries)
            return top_sim_entries

        num_entities = self.graph.vcount()
        entity_list = [self.graph.vs[i]['name'] for i in range(num_entities)]
        reset = [
            1.0 / len(personalization_entities) if entity in personalization_entities else 0.0
            for entity in entity_list
        ]

        ppr_scores = self.graph.personalized_pagerank(
            directed=False,
            damping=0.85,
            reset=reset,
            implementation='prpack',
        )
        entity_to_ppr = {entity_list[i]: ppr_scores[i] for i in range(num_entities)}

        triple_scores: Dict[str, float] = {}
        for entry in self.indexed_entries:
            subj, obj = self.triple_to_entities.get(entry.id, ("", ""))
            subj_score = entity_to_ppr.get(subj, 0.0) if subj else 0.0
            obj_score = entity_to_ppr.get(obj, 0.0) if obj else 0.0
            triple_scores[entry.id] = subj_score + obj_score

        sorted_entries = sorted(
            self.indexed_entries,
            key=lambda e: triple_scores.get(e.id, 0.0),
            reverse=True,
        )[:top_k]

        if as_context:
            return self.retrieve_triples_as_str(sorted_entries)
        return sorted_entries

    def retrieve_triples_as_str(self, entries: List[SpatialTripleEntry]) -> str:
        return "\n".join(entry.to_display_str() for entry in entries)

    def cleanup(self) -> None:
        if self.embeddings is not None:
            del self.embeddings
            self.embeddings = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reset_index(self) -> None:
        self.graph = None
        self.embeddings = None
        self.indexed_entries = []
        self.indexed_time = 0
        self.indexed_timestamp = 0
        self.triple_to_entities = {}
        logger.info("Spatial index reset - graph and embeddings cleared")

    def get_indexed_time(self) -> str:
        return _transform_timestamp(str(self.indexed_time))

    def get_indexed_timestamp(self) -> str:
        return _transform_timestamp(str(self.indexed_timestamp)) if self.indexed_timestamp > 0 else "Not indexed"

    def get_triple_by_id(self, triple_id: str) -> Optional[SpatialTripleEntry]:
        return self.triple_id_to_entry.get(triple_id)

    def get_indexed_count(self) -> int:
        return len(self.indexed_entries)
