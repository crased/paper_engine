"""In-process engine services: fast reads + inference.

These wrap pipeline/tools logic directly (no subprocess) because they are quick.
Long-running work (train, annotate) lives in jobs.py as subprocesses instead.

Heavy deps (ultralytics/torch) are imported lazily so importing this module —
and therefore the whole API — stays cheap.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .schemas import (
    ClassCount,
    DatasetStats,
    Detection,
    FrameDetections,
    ModelInfo,
    ReportContent,
    ReportInfo,
)

# PROJECT_ROOT = the backend/ package root (code lives here: api, pipeline, ...).
# DATA_ROOT = the repo root (one level up), where data dirs live (recordings,
# reports, yolo_dataset, ...). Code-local dirs (conf, bot_logic) use PROJECT_ROOT.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT.parent
MODELS_DIR = PROJECT_ROOT / "bot_logic" / "models"
REPORTS_DIR = DATA_ROOT / "reports"
SESSIONS_DIR = DATA_ROOT / "recordings" / "sessions"
TRAINING_CONF = PROJECT_ROOT / "conf" / "training_conf.ini"

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


# --------------------------------------------------------------------------- #
# Path safety
# --------------------------------------------------------------------------- #
def _resolve_within(base: Path, candidate: str) -> Path:
    """Resolve `candidate` and ensure it stays under `base` (no traversal)."""
    p = (base / candidate).resolve() if not os.path.isabs(candidate) else Path(candidate).resolve()
    base_r = base.resolve()
    if base_r not in p.parents and p != base_r:
        raise ValueError(f"Path escapes allowed directory: {candidate}")
    return p


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def active_weights_path() -> Optional[Path]:
    """Read best_model_path from training_conf.ini, resolved to absolute."""
    if not TRAINING_CONF.exists():
        return None
    for raw in TRAINING_CONF.read_text().splitlines():
        line = raw.strip()
        if line.startswith("best_model_path"):
            _, _, val = line.partition("=")
            val = val.strip()
            if val:
                p = Path(val)
                return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()
    return None


def list_models() -> List[ModelInfo]:
    active = active_weights_path()
    out: List[ModelInfo] = []
    if not MODELS_DIR.exists():
        return out
    for weights in sorted(MODELS_DIR.glob("*/weights/best.pt")):
        stat = weights.stat()
        out.append(
            ModelInfo(
                name=weights.parent.parent.name,
                weights_path=str(weights),
                size_mb=round(stat.st_size / 1e6, 2),
                modified=stat.st_mtime,
                active=(active is not None and weights.resolve() == active),
            )
        )
    out.sort(key=lambda m: m.modified, reverse=True)
    return out


def _resolve_model(model: Optional[str]) -> Path:
    """Accept a model name, weights path, or None (active)."""
    if model:
        cand = Path(model)
        if cand.is_file():
            return cand.resolve()
        named = MODELS_DIR / model / "weights" / "best.pt"
        if named.is_file():
            return named.resolve()
        raise FileNotFoundError(f"Model not found: {model}")
    active = active_weights_path()
    if active and active.is_file():
        return active
    models = list_models()
    if not models:
        raise FileNotFoundError("No trained models found")
    return Path(models[0].weights_path)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def _gather_images(
    paths: Optional[List[str]],
    directory: Optional[str],
    session: Optional[str],
    limit: int,
) -> List[Path]:
    imgs: List[Path] = []
    if paths:
        imgs = [Path(p).resolve() for p in paths]
    elif directory:
        d = Path(directory).resolve()
        imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    elif session:
        d = _resolve_within(SESSIONS_DIR, session)
        imgs = sorted(d.glob("frame_*.png"))
    else:
        raise ValueError("Provide one of: paths, directory, session")

    imgs = [p for p in imgs if p.exists()]
    if limit and len(imgs) > limit:
        # even spread across the set, not just the first N
        step = (len(imgs) - 1) / (limit - 1) if limit > 1 else 0
        imgs = [imgs[int(round(i * step))] for i in range(limit)]
    return imgs


def run_inference(
    model: Optional[str],
    paths: Optional[List[str]],
    directory: Optional[str],
    session: Optional[str],
    conf: float,
    limit: int,
) -> dict:
    from ultralytics import YOLO  # lazy: heavy import

    weights = _resolve_model(model)
    yolo = YOLO(str(weights))
    names: Dict[int, str] = {int(k): v for k, v in yolo.names.items()}

    images = _gather_images(paths, directory, session, limit)
    frames: List[FrameDetections] = []
    total = 0
    with_det = 0
    for img in images:
        try:
            r = yolo.predict(str(img), conf=conf, verbose=False)[0]
        except Exception as exc:  # corrupt/truncated frame, etc.
            frames.append(
                FrameDetections(image=str(img), width=0, height=0, detections=[], error=str(exc))
            )
            continue
        h, w = (r.orig_shape if r.orig_shape else (0, 0))
        dets: List[Detection] = []
        for box in r.boxes:
            cid = int(box.cls.item())
            xyxy = box.xyxy[0].tolist()
            dets.append(
                Detection(
                    cls_id=cid,
                    cls_name=names.get(cid, str(cid)),
                    confidence=round(float(box.conf.item()), 4),
                    x1=round(xyxy[0], 1),
                    y1=round(xyxy[1], 1),
                    x2=round(xyxy[2], 1),
                    y2=round(xyxy[3], 1),
                )
            )
        total += len(dets)
        with_det += 1 if dets else 0
        frames.append(FrameDetections(image=str(img), width=int(w), height=int(h), detections=dets))

    return {
        "model": str(weights),
        "classes": names,
        "conf": conf,
        "frames": frames,
        "total_detections": total,
        "frames_with_detection": with_det,
    }


# --------------------------------------------------------------------------- #
# Dataset stats
# --------------------------------------------------------------------------- #
def dataset_stats(yaml_name: str = "dataset.yaml") -> DatasetStats:
    yaml_path = _resolve_within(DATA_ROOT / "yolo_dataset", yaml_name)
    data = yaml.safe_load(yaml_path.read_text())
    names_raw = data.get("names", {})
    if isinstance(names_raw, list):
        names = {i: n for i, n in enumerate(names_raw)}
    else:
        names = {int(k): v for k, v in names_raw.items()}

    base = yaml_path.parent
    counts: Dict[int, int] = {cid: 0 for cid in names}

    def count_split(split: str) -> tuple[int, int]:
        labels_dir = base / split / "labels"
        images_dir = base / split / "images"
        n_labels = 0
        if labels_dir.exists():
            for txt in labels_dir.glob("*.txt"):
                n_labels += 1
                for line in txt.read_text().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        cid = int(line.split()[0])
                    except (ValueError, IndexError):
                        continue
                    counts[cid] = counts.get(cid, 0) + 1
        n_images = (
            sum(1 for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            if images_dir.exists()
            else 0
        )
        return n_images, n_labels

    train_images, train_labels = count_split("train")
    val_images, val_labels = count_split("val")

    classes = [
        ClassCount(cls_id=cid, cls_name=names[cid], instances=counts.get(cid, 0))
        for cid in sorted(names)
    ]
    return DatasetStats(
        yaml=str(yaml_path),
        classes=classes,
        train_images=train_images,
        val_images=val_images,
        train_labels=train_labels,
        val_labels=val_labels,
        total_instances=sum(counts.values()),
    )


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def list_reports() -> List[ReportInfo]:
    out: List[ReportInfo] = []
    if not REPORTS_DIR.exists():
        return out
    for f in REPORTS_DIR.glob("*.txt"):
        stat = f.stat()
        out.append(ReportInfo(name=f.name, size_bytes=stat.st_size, modified=stat.st_mtime))
    out.sort(key=lambda r: r.modified, reverse=True)
    return out


def read_report(name: str) -> ReportContent:
    # name only — no subpaths
    if "/" in name or "\\" in name or name.startswith("."):
        raise ValueError("Invalid report name")
    path = (REPORTS_DIR / name).resolve()
    if REPORTS_DIR.resolve() not in path.parents:
        raise ValueError("Path escapes reports directory")
    if not path.is_file():
        raise FileNotFoundError(name)
    return ReportContent(name=name, content=path.read_text(errors="replace"))


def list_sessions() -> List[str]:
    if not SESSIONS_DIR.exists():
        return []
    return sorted(
        (d.name for d in SESSIONS_DIR.iterdir() if d.is_dir()),
        reverse=True,
    )
