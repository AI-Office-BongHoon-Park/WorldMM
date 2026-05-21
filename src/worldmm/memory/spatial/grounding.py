"""Geometric grounding schema for spatial triples."""

import math
from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class GeometricGrounding(BaseModel):
    """Relative or metric 3D grounding for one side of a spatial triple."""

    model_config = ConfigDict(extra="forbid")

    bbox_center: list[float] = Field(..., min_length=3, max_length=3)
    bbox_extent: list[float] = Field(..., min_length=3, max_length=3)
    units: Literal["meters", "relative"]
    source: str
    confidence: float = Field(..., ge=-1.0, le=1.0)
    keyframe_ts: int
    instance_disambiguation: Literal[
        "single_match",
        "closest_in_time",
        "max_confidence",
        "tied_dropped",
    ]

    @field_validator("bbox_center", "bbox_extent")
    @classmethod
    def _validate_vec3(cls, value: list[float]) -> list[float]:
        return [float(item) for item in value]

    @model_validator(mode="after")
    def _validate_units_source(self) -> "GeometricGrounding":
        metric_source = "vggt_intrinsics" in self.source or "colmap_intrinsics" in self.source
        metric_depth = "metric" in self.source.lower()
        if self.units == "meters" and not (metric_source and metric_depth):
            raise ValueError("units='meters' requires metric depth and VGGT/COLMAP intrinsics in source")
        return self


class Pose6DoF(BaseModel):
    """6-DoF wearer pose in a metric world frame."""

    model_config = ConfigDict(extra="forbid")

    tracking_timestamp_us: int
    tx: float
    ty: float
    tz: float
    qw: float
    qx: float
    qy: float
    qz: float
    quality_score: float = Field(..., ge=0.0, le=1.0)
    graph_uid: str
    units: Literal["meters"] = "meters"
    frame: str = "world"

    @model_validator(mode="after")
    def _validate_quaternion(self) -> "Pose6DoF":
        magnitude = math.sqrt(self.qw**2 + self.qx**2 + self.qy**2 + self.qz**2)
        if not math.isclose(magnitude, 1.0, abs_tol=1e-3):
            raise ValueError("quaternion magnitude must be approximately 1")
        return self


class GazeTarget(BaseModel):
    """CPF-frame eye-gaze target sample."""

    model_config = ConfigDict(extra="forbid")

    tracking_timestamp_us: int
    yaw_rads_cpf: float
    pitch_rads_cpf: float
    point_cpf: Optional[Tuple[float, float, float]] = None
    confidence_interval: Optional[dict[str, float]] = None
    session_uid: Optional[str] = None


class PlaceAnchor(BaseModel):
    """Metric place anchor reserved for later clustering output."""

    model_config = ConfigDict(extra="forbid")

    coordinate_frame_id: str
    centroid_world_m: Tuple[float, float, float]
    time_range_us: Tuple[int, int]
    label: Optional[str] = None
    evidence: List[str] = []


class SpeechSegment(BaseModel):
    """Transcript segment aligned to the AEA time domain."""

    model_config = ConfigDict(extra="forbid")

    start_time_ns: int
    end_time_ns: int
    text: str
    confidence: float
    language: Optional[str] = None
