"""Pluggable OCR engines. Each returns a list of Word objects in image coordinates."""
from __future__ import annotations

from typing import List

import numpy as np

from certiscan.types import Word


class EasyOCREngine:
    name = "easyocr"

    def __init__(self, gpu: bool | None = None, langs=("en",)):
        import easyocr

        if gpu is None:
            try:
                import torch

                gpu = torch.cuda.is_available()
            except ImportError:
                gpu = False
        self.reader = easyocr.Reader(list(langs), gpu=gpu, verbose=False)

    def read(self, image: np.ndarray) -> List[Word]:
        # width_ths below the 0.5 default keeps adjacent table columns from being merged;
        # min_size below the default 20px keeps one-letter element headers ("C", "P", "S")
        results = self.reader.readtext(image, paragraph=False, width_ths=0.3, min_size=5, decoder="greedy")
        words = []
        for pts, text, conf in results:
            xs = [float(p[0]) for p in pts]  # EasyOCR returns numpy ints: not JSON-serialisable
            ys = [float(p[1]) for p in pts]
            if text.strip():
                words.append(Word(text.strip(), (min(xs), min(ys), max(xs), max(ys)), float(conf)))
        return words


class TesseractEngine:
    name = "tesseract"

    def __init__(self, psm: int = 11):
        import pytesseract

        self.pt = pytesseract
        # psm 11 = sparse text: works better than full-page layout analysis on forms/tables
        self.config = f"--oem 1 --psm {psm}"

    def read(self, image: np.ndarray) -> List[Word]:
        data = self.pt.image_to_data(image, config=self.config, output_type=self.pt.Output.DICT)
        words = []
        for i, text in enumerate(data["text"]):
            text = (text or "").strip()
            conf = float(data["conf"][i])
            if text and conf >= 0:
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                words.append(Word(text, (float(x), float(y), float(x + w), float(y + h)), conf / 100))
        return words


def get_engine(name: str):
    name = name.lower()
    if name == "easyocr":
        return EasyOCREngine()
    if name == "tesseract":
        return TesseractEngine()
    raise ValueError(f"Unknown OCR engine: {name}")
