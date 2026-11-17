"""Layout detectors: a fine-tuned YOLO / RT-DETR model, or the keyword fallback."""
from __future__ import annotations

from typing import List

import numpy as np

from certiscan.extract import rule_regions
from certiscan.types import Region, Word


class UltralyticsLayout:
    """Wraps any Ultralytics detector (YOLOv8/YOLO11/RT-DETR) trained on our region classes."""

    name = "yolo"

    def __init__(self, weights: str, conf: float = 0.25, imgsz: int = 640):
        from ultralytics import YOLO, RTDETR

        self.model = RTDETR(weights) if "rtdetr" in weights.lower() else YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, image: np.ndarray, words: List[Word] | None = None) -> List[Region]:
        res = self.model.predict(image, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        names = res.names
        regions = []
        for xyxy, cls, conf in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
            regions.append(Region(names[int(cls)], tuple(xyxy), float(conf)))
        return regions


class RuleLayout:
    """No model: find table titles by keyword. Used as a baseline and when weights are absent."""

    name = "rules"

    def detect(self, image: np.ndarray, words: List[Word] | None = None) -> List[Region]:
        return rule_regions(words or [])
