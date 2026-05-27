"""Pydantic request/response models.

These are the contract the frontend types are generated from (FastAPI exposes
them at /openapi.json). Keep names and shapes stable.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class ModelInfo(BaseModel):
    name: str
    weights_path: str
    size_mb: float
    modified: float  # epoch seconds
    active: bool = False


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
class Detection(BaseModel):
    cls_id: int
    cls_name: str
    confidence: float
    # xyxy in pixel coords
    x1: float
    y1: float
    x2: float
    y2: float


class FrameDetections(BaseModel):
    image: str
    width: int
    height: int
    detections: List[Detection]
    error: Optional[str] = None


class InferRequest(BaseModel):
    # Provide exactly one source: explicit paths, a directory, or a session name.
    paths: Optional[List[str]] = None
    directory: Optional[str] = None
    session: Optional[str] = None
    model: Optional[str] = Field(
        default=None, description="Model name or weights path; default = configured active model"
    )
    conf: float = 0.25
    limit: int = Field(default=24, description="Max frames to run (0 = all)")


class InferResponse(BaseModel):
    model: str
    classes: dict[int, str]
    conf: float
    frames: List[FrameDetections]
    total_detections: int
    frames_with_detection: int


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class ClassCount(BaseModel):
    cls_id: int
    cls_name: str
    instances: int


class DatasetStats(BaseModel):
    yaml: str
    classes: List[ClassCount]
    train_images: int
    val_images: int
    train_labels: int
    val_labels: int
    total_instances: int


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
class ReportInfo(BaseModel):
    name: str
    size_bytes: int
    modified: float


class ReportContent(BaseModel):
    name: str
    content: str


# --------------------------------------------------------------------------- #
# Jobs (long-running: train, annotate)
# --------------------------------------------------------------------------- #
class JobInfo(BaseModel):
    id: str
    kind: str
    status: str  # pending | running | done | error
    cmd: List[str]
    return_code: Optional[int] = None
    started: float
    ended: Optional[float] = None
    line_count: int


class TrainRequest(BaseModel):
    # training_model.py reads conf/training_conf.ini; no args needed by default.
    pass


class AnnotateRequest(BaseModel):
    session: str
    dataset: Optional[str] = None
    rpm: int = 10
    max_frames: int = 0
    no_examples: bool = False
    dry_run: bool = False
