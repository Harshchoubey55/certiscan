"""End-to-end pipeline: image -> deskew -> layout + OCR -> fields -> compliance."""
from __future__ import annotations

import os
import time
from typing import List, Optional

import numpy as np

from certiscan.extract import extract_document
from certiscan.preprocess import deskew
from certiscan.types import Region, Word
from certiscan.validate import validate_document


class Pipeline:
    def __init__(self, layout=None, ocr=None, specs: Optional[dict] = None, do_deskew: bool = True):
        self.layout = layout
        self.ocr = ocr
        self.specs = specs
        self.do_deskew = do_deskew

    def run(
        self,
        image: Optional[np.ndarray] = None,
        words: Optional[List[Word]] = None,
        regions: Optional[List[Region]] = None,
    ) -> dict:
        """Pass `words`/`regions` to bypass OCR/detection (used for oracle evaluation)."""
        t = {}
        angle = 0.0
        start = time.perf_counter()
        if image is not None and self.do_deskew and words is None:
            image, angle = deskew(image)
        t["deskew_s"] = time.perf_counter() - start

        start = time.perf_counter()
        if words is None:
            if self.ocr is None:
                raise RuntimeError("No OCR engine configured")
            words = self.ocr.read(image)
        t["ocr_s"] = time.perf_counter() - start

        start = time.perf_counter()
        if regions is None:
            if self.layout is None:
                raise RuntimeError("No layout detector configured")
            regions = self.layout.detect(image, words)
        t["layout_s"] = time.perf_counter() - start

        start = time.perf_counter()
        doc = extract_document(words, regions)
        doc["compliance"] = validate_document(doc, self.specs)
        t["extract_s"] = time.perf_counter() - start

        return {
            "document": doc,
            "regions": [r.to_dict() for r in regions],
            "words": [w.to_dict() for w in words],
            "skew_angle": angle,
            "image": image,
            "timings": {k: round(v, 3) for k, v in t.items()},
        }


def build_pipeline(ocr: Optional[str] = None, layout: Optional[str] = None, weights: Optional[str] = None) -> Pipeline:
    """Build from arguments or env vars CERTISCAN_OCR / CERTISCAN_LAYOUT / CERTISCAN_WEIGHTS."""
    from certiscan.layout import RuleLayout, UltralyticsLayout
    from certiscan.ocr import get_engine

    ocr = ocr or os.getenv("CERTISCAN_OCR", "easyocr")
    weights = weights or os.getenv("CERTISCAN_WEIGHTS")
    layout = layout or os.getenv("CERTISCAN_LAYOUT", "yolo" if weights else "rules")
    det = UltralyticsLayout(weights) if layout == "yolo" else RuleLayout()
    return Pipeline(layout=det, ocr=get_engine(ocr))
