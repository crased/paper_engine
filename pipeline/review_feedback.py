"""
Review Feedback — persistence for batch image review sessions.

Tracks per-image review data: what the user described, what bounding boxes the LLM
generated, what corrections the user made, and timing. Stored as JSON in
reports/review_feedback/ (gitignored via *reports/).

Used by the Describe & Review workflow in AnnotationWindow (review_results.py).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path shim
# ---------------------------------------------------------------------------

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ---------------------------------------------------------------------------
# Default output directory
# ---------------------------------------------------------------------------

_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "reports" / "review_feedback"


# ---------------------------------------------------------------------------
# Per-image review record
# ---------------------------------------------------------------------------


@dataclass
class ImageReview:
    """Review record for a single image."""

    image_path: str
    image_name: str
    description: str  # what the user typed
    generated_detections: List[Dict]  # LLM output (before user corrections)
    final_detections: List[Dict]  # after user corrections (what was saved)
    corrections_made: bool  # True if user modified the LLM output
    timestamp: str  # ISO format
    review_time_sec: float  # time spent on this image

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "image_name": self.image_name,
            "description": self.description,
            "generated_detections": _serialize_detections(self.generated_detections),
            "final_detections": _serialize_detections(self.final_detections),
            "corrections_made": self.corrections_made,
            "timestamp": self.timestamp,
            "review_time_sec": self.review_time_sec,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ImageReview:
        return cls(
            image_path=d["image_path"],
            image_name=d["image_name"],
            description=d["description"],
            generated_detections=d.get("generated_detections", []),
            final_detections=d.get("final_detections", []),
            corrections_made=d.get("corrections_made", False),
            timestamp=d.get("timestamp", ""),
            review_time_sec=d.get("review_time_sec", 0.0),
        )


# ---------------------------------------------------------------------------
# Review session (batch of image reviews)
# ---------------------------------------------------------------------------


@dataclass
class ReviewSession:
    """A batch review session (e.g., 'Review 100 images')."""

    session_id: str  # e.g., "review_20260325_143000"
    started_at: str  # ISO format
    completed_at: Optional[str] = None
    total_images: int = 0
    reviewed_count: int = 0
    reviews: List[ImageReview] = field(default_factory=list)
    source_path: str = ""  # where images came from

    @classmethod
    def create(cls, total_images: int, source_path: str = "") -> ReviewSession:
        """Create a new review session with a generated ID."""
        now = datetime.now()
        session_id = f"review_{now.strftime('%Y%m%d_%H%M%S')}"
        return cls(
            session_id=session_id,
            started_at=now.isoformat(),
            total_images=total_images,
            source_path=source_path,
        )

    def add_review(self, review: ImageReview) -> None:
        """Append a completed image review."""
        self.reviews.append(review)
        self.reviewed_count = len(self.reviews)

    def finish(self) -> None:
        """Mark session as complete."""
        self.completed_at = datetime.now().isoformat()
        self.reviewed_count = len(self.reviews)

    def save(self, output_dir: Optional[Path] = None) -> Path:
        """Save session to JSON file. Returns the path written."""
        out_dir = Path(output_dir) if output_dir else _DEFAULT_OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        path = out_dir / f"{self.session_id}.json"
        data = {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_images": self.total_images,
            "reviewed_count": self.reviewed_count,
            "source_path": self.source_path,
            "reviews": [r.to_dict() for r in self.reviews],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("Saved review session to %s", path)
        return path

    @classmethod
    def load(cls, path: Path) -> ReviewSession:
        """Load a review session from JSON file."""
        with open(path) as f:
            data = json.load(f)

        session = cls(
            session_id=data["session_id"],
            started_at=data["started_at"],
            completed_at=data.get("completed_at"),
            total_images=data.get("total_images", 0),
            reviewed_count=data.get("reviewed_count", 0),
            source_path=data.get("source_path", ""),
            reviews=[ImageReview.from_dict(r) for r in data.get("reviews", [])],
        )
        return session

    def summary(self) -> str:
        """Generate a text summary of the review session."""
        corrections = sum(1 for r in self.reviews if r.corrections_made)
        total_dets = sum(len(r.final_detections) for r in self.reviews)
        avg_time = (
            sum(r.review_time_sec for r in self.reviews) / len(self.reviews)
            if self.reviews
            else 0.0
        )

        lines = [
            f"Review Session: {self.session_id}",
            f"  Images reviewed: {self.reviewed_count} / {self.total_images}",
            f"  Corrections made: {corrections} ({corrections}/{self.reviewed_count} images)" if self.reviewed_count else "  Corrections made: 0",
            f"  Total detections saved: {total_dets}",
            f"  Avg review time: {avg_time:.1f}s per image",
        ]

        if self.started_at and self.completed_at:
            lines.append(f"  Started: {self.started_at}")
            lines.append(f"  Completed: {self.completed_at}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _serialize_detections(detections: List[Dict]) -> List[Dict]:
    """Ensure detection dicts are JSON-serializable (convert tuples to lists)."""
    out = []
    for det in detections:
        d = dict(det)
        if "bbox" in d and isinstance(d["bbox"], tuple):
            d["bbox"] = list(d["bbox"])
        if "bbox_norm" in d and isinstance(d["bbox_norm"], tuple):
            d["bbox_norm"] = list(d["bbox_norm"])
        out.append(d)
    return out


def list_sessions(output_dir: Optional[Path] = None) -> List[Path]:
    """List all saved review session files, sorted by modification time (newest first)."""
    out_dir = Path(output_dir) if output_dir else _DEFAULT_OUTPUT_DIR
    if not out_dir.exists():
        return []
    return sorted(out_dir.glob("review_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
