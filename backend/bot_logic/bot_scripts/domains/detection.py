"""
Domain 1 — Detection (YOLO thread)

Runs YOLO inference in a background daemon thread. Captures screenshots,
runs the model, writes DetectionFrame to a thread-safe buffer. Other
domains read the latest frame without blocking.

The main loop never waits on this — it grabs whatever is latest.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import mss  # fast screenshot capture (~3ms)
import numpy
from ultralytics import YOLO  # YOLO11 inference

from bot_logic.bot_scripts.domains.contracts import Detection, DetectionFrame
from conf.config_parser import main_conf

logger = logging.getLogger(__name__)


class DetectionDomain:
    """Background YOLO detection thread. Write-once, read-many."""

    def __init__(self, model_path: str, target_fps: float = 10.0, conf: float = 0.25):
        self._model_path = model_path
        self._target_fps = target_fps
        self._conf = conf
        self._monitor_index: int = main_conf.get("DEFAULT_MONITOR", 0)

        self._model: Optional[YOLO] = None
        self._sct: Optional[mss.mss] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Shared buffer — protected by lock
        self._lock = threading.Lock()
        self._latest: Optional[DetectionFrame] = None

    # --- public API (called from main thread) ---

    def start(self):
        self._model = YOLO(self._model_path)
        self._sct = mss.mss()
        self._running = True
        self._thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._sct is not None:
            self._sct.close()
            self._sct = None

    def get_latest(self) -> Optional[DetectionFrame]:
        with self._lock:
            return self._latest

    # --- internal (runs in background thread) ---

    def _detection_loop(self):
        frametime = 1.0 / self._target_fps
        while self._running:
            try:
                t0 = time.monotonic()
                frame = self._capture_screenshot()
                if frame is None:
                    time.sleep(frametime)
                    continue
                results = self._model.predict(frame, conf=self._conf, verbose=False)
                detections = self._results_to_detections(results)
                df = DetectionFrame(detections=detections, timestamp=time.monotonic())
                with self._lock:
                    self._latest = df
                elapsed = time.monotonic() - t0
                time.sleep(max(0, frametime - elapsed))
            except Exception:
                logger.exception("detection loop error")
                time.sleep(frametime)

    def _capture_screenshot(self):
        if self._sct is None:
            return None
        monitor = self._sct.monitors[self._monitor_index + 1]
        img = self._sct.grab(monitor)
        frame = numpy.array(img)[:, :, :3]  # BGRA → BGR
        return frame

    def _results_to_detections(self, results) -> list[Detection]:
        boxes = results[0].boxes
        detections = []
        for i in range(len(boxes)):
            class_id = int(boxes.cls[i])
            class_name = results[0].names[class_id]
            bbox = tuple(boxes.xywhn[i].tolist())
            confidence = float(boxes.conf[i])
            detections.append(Detection(class_id, class_name, bbox, confidence))
        return detections
